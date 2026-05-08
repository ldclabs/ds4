// ds4.rs - DeepSeek V4 Flash inference engine (Rust port of ds4.c)
//
// This engine is intentionally narrow: fixed DeepSeek V4 Flash architecture,
// not a generic GGUF runner. The CPU path is the main target.

pub mod gguf;
pub mod quant;
pub mod model;
pub mod tokenizer;
pub mod forward;
pub mod session;
pub mod constants;

pub use constants::*;

/// Return the compression ratio for a given layer (matches ds4_layer_compress_ratio in ds4.c).
/// Layers 0-1: 0 (dense). Layer >=2 even: 4. Layer >=2 odd: 128.
pub fn layer_compress_ratio(il: u32) -> u32 {
    if il < 2 { return 0; }
    if (il & 1) == 0 { 4 } else { 128 }
}

/// GCG-style splitexp hash routing: returns the expert index.
pub fn hash_routed_expert(token: i32, router_idx: u32, n_expert: u32) -> u32 {
    let mut h: u64 = (token as u32 as u64) ^ 0x9E3779B9u64;
    h = h.wrapping_add(router_idx as u64);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EBCA77u64);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2AE35u64);
    h ^= h >> 16;
    (h % n_expert as u64) as u32
}

/// Sigmoid with clamping for numerical stability.
#[inline]
pub fn sigmoid_stable(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// SiLU (Swish) activation.
#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid_stable(x)
}

/// SwiGLU: gated SiLU.
#[inline]
pub fn swiglu(gate: f32, up: f32) -> f32 {
    silu(gate) * up
}

/// Softplus with clamping for numerical stability.
#[inline]
pub fn softplus_stable(x: f32) -> f32 {
    if x > 20.0 { return x; }
    if x < -20.0 { return x.exp(); }
    (1.0 + x.exp()).ln()
}

/// RMS normalization (f64 accumulator for numerical fidelity, matching ds4.c).
/// The plain loops auto-vectorize when compiled with -C target-cpu=native.
#[inline]
pub fn rms_norm(out: &mut [f32], x: &[f32], n: usize, eps: f32) {
    let mut ss: f64 = 0.0;
    for i in 0..n {
        let v = x[i] as f64;
        ss += v * v;
    }
    let scale = (1.0f64 / ((ss / n as f64) + eps as f64).sqrt()) as f32;
    for i in 0..n {
        out[i] = x[i] * scale;
    }
}

/// RMS normalization with weight (f64 accumulator, matching ds4.c).
#[inline]
pub fn rms_norm_weighted(out: &mut [f32], x: &[f32], weight: &[f32], n: usize, eps: f32) {
    let mut ss: f64 = 0.0;
    for i in 0..n {
        let v = x[i] as f64;
        ss += v * v;
    }
    let scale = (1.0f64 / ((ss / n as f64) + eps as f64).sqrt()) as f32;
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

/// RMS normalization without weight (output only, f64 accumulator).
#[inline]
pub fn rms_norm_no_weight(out: &mut [f32], x: &[f32], n: usize, eps: f32) {
    let mut ss: f64 = 0.0;
    for i in 0..n {
        let v = x[i] as f64;
        ss += v * v;
    }
    let scale = (1.0f64 / ((ss / n as f64) + eps as f64).sqrt()) as f32;
    for i in 0..n {
        out[i] = x[i] * scale;
    }
}

/// Softmax in-place.
pub fn softmax(x: &mut [f32], n: usize) {
    let mut max_val = NEG_INF;
    for v in x[..n].iter() {
        if *v > max_val { max_val = *v; }
    }
    let mut sum = 0.0f32;
    for v in x[..n].iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x[..n].iter_mut() { *v /= sum; }
    }
}

/// FP16 to F32 conversion.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let val = f32::from_bits((sign << 31) | ((mant as u32) << 12));
            val / (1 << 24) as f32
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits((sign << 31) | 0x7f800000)
        } else {
            f32::NAN
        }
    } else {
        f32::from_bits((sign << 31) | (((exp as u32) + 127 - 15) << 23) | (mant << 13))
    }
}

/// F32 to FP16 conversion.
#[inline]
pub fn f32_to_f16(f: f32) -> u16 {
    let u = f.to_bits();
    let sign = ((u >> 16) & 0x8000) as u16;
    let exp = ((u >> 23) & 0xff) as i32 - 127 + 15;
    let mant = u & 0x7fffff;

    if exp <= 0 {
        if exp < -10 { return sign; }
        let mant2 = mant | 0x800000;
        let shift = (14 - exp) as u32;
        let mut half_mant = mant2 >> shift;
        if (mant2 >> (shift - 1)) & 1 != 0 { half_mant += 1; }
        sign | (half_mant as u16)
    } else if exp >= 31 {
        sign | 0x7c00
    } else {
        let mut half = sign | ((exp as u16) << 10) | ((mant >> 13) as u16);
        if mant & 0x1000 != 0 { half += 1; }
        half
    }
}

/// Round f32 values to f16 precision in-place (for KV cache compression).
pub fn f16_round_inplace(x: &mut [f32], n: usize) {
    for v in x[..n].iter_mut() {
        *v = f16_to_f32(f32_to_f16(*v));
    }
}

/// FP8 E4M3-style quantize for KV cache: block-wise round-trip on non-rotary dims.
/// Rotary dimensions (first n_rot) are preserved unchanged.
/// Non-rotary dims are processed in blocks of 64, each block with its own scale.
/// Matches dsv4_fp8_kv_quantize_row_inplace_cpu from ds4.c exactly.
pub fn fp8_kv_quantize_row_inplace(x: &mut [f32], head_dim: usize, n_rot: usize) {
    // E4M3 value table: maps [0..126] to representable positive E4M3 values.
    fn e4m3_value(i: usize) -> f32 {
        if i > 126 { return 448.0; }
        let exp = (i >> 3) & 0x0f;
        let mant = (i & 0x07) as f32;
        if exp == 0 {
            mant * 0.001953125 // denorm: mant / 512
        } else {
            let scale = 0.015625f32 * (1u32 << (exp - 1)) as f32; // 2^(exp-7)
            (1.0 + mant * 0.125) * scale
        }
    }

    // Binary search nearest E4M3 value for |x|.
    fn e4m3_dequant(x: f32) -> f32 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let ax = x.abs().min(448.0);
        let mut lo: usize = 0;
        let mut hi: usize = 126;
        while lo < hi {
            let mid = (lo + hi + 1) >> 1;
            if e4m3_value(mid) <= ax { lo = mid; }
            else { hi = mid - 1; }
        }
        let mut best = lo;
        if best < 126 {
            let bd = (ax - e4m3_value(best)).abs();
            let nd = (ax - e4m3_value(best + 1)).abs();
            if nd < bd || (nd == bd && ((best + 1) & 1) == 0 && (best & 1) != 0) {
                best += 1;
            }
        }
        sign * e4m3_value(best)
    }

    let n_nope = head_dim.saturating_sub(n_rot);

    // Process in blocks of 64
    let mut off = 0usize;
    while off + 64 <= n_nope {
        let mut amax = 0.0f32;
        for i in off..off + 64 {
            let av = x[i].abs();
            if av > amax { amax = av; }
        }
        if amax < 1.0e-4 { amax = 1.0e-4; }
        let scale = (2.0f32).powf((amax / 448.0).log2().ceil());
        for i in off..off + 64 {
            let v = (x[i] / scale).clamp(-448.0, 448.0);
            x[i] = e4m3_dequant(v) * scale;
        }
        off += 64;
    }

    // Remainder (less than 64 elements)
    if off < n_nope {
        let mut amax = 0.0f32;
        for i in off..n_nope {
            let av = x[i].abs();
            if av > amax { amax = av; }
        }
        if amax < 1.0e-4 { amax = 1.0e-4; }
        let scale = (2.0f32).powf((amax / 448.0).log2().ceil());
        for i in off..n_nope {
            let v = (x[i] / scale).clamp(-448.0, 448.0);
            x[i] = e4m3_dequant(v) * scale;
        }
    }
}

/// HC split + Sinkhorn: from mix vector [n_hc (pre) + n_hc (post) + n_hc*n_hc (comb)],
/// produce: pre_weights [n_hc], post_weights [n_hc], comb_matrix [n_hc*n_hc].
///
/// - pre[i] = sigmoid(mix[i] * scale[0] + base[i]) + eps
/// - post[i] = 2 * sigmoid(mix[n_hc+i] * scale[1] + base[n_hc+i])
/// - comb: softmax each row then Sinkhorn iters iterations
pub fn hc_split_sinkhorn_one(
    out: &mut [f32],       // [2*n_hc + n_hc*n_hc] = pre + post + comb flattened
    mix: &[f32],           // [2*n_hc + n_hc*n_hc]
    scale: &[f32],         // [3]: pre_scale, post_scale, comb_scale
    base: &[f32],          // [2*n_hc + n_hc*n_hc]
    n_hc: usize,
    n_iter: u32,
    eps: f32,
) {
    let pre_scale = scale[0];
    let post_scale = scale[1];
    let comb_scale = scale[2];

    // Pre weights: sigmoid
    for i in 0..n_hc {
        let z = mix[i] * pre_scale + base[i];
        out[i] = sigmoid_stable(z) + eps;
    }

    // Post weights: 2 * sigmoid
    for i in 0..n_hc {
        let off = n_hc + i;
        let z = mix[off] * post_scale + base[off];
        out[off] = 2.0 * sigmoid_stable(z);
    }

    // Comb matrix: softmax per row then Sinkhorn
    let comb_off = 2 * n_hc;

    // Row-wise softmax
    for dst in 0..n_hc {
        let mut row_max = NEG_INF;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            let z = mix[comb_off + idx] * comb_scale + base[comb_off + idx];
            out[comb_off + idx] = z;
            if z > row_max { row_max = z; }
        }
        let mut row_sum = 0.0f32;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            let v = (out[comb_off + idx] - row_max).exp();
            out[comb_off + idx] = v;
            row_sum += v;
        }
        let inv = 1.0 / row_sum;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            out[comb_off + idx] = out[comb_off + idx] * inv + eps;
        }
    }

    // Column normalization (first Sinkhorn iteration)
    for src in 0..n_hc {
        let mut sum = 0.0f32;
        for dst in 0..n_hc { sum += out[comb_off + src + dst * n_hc]; }
        let inv = 1.0 / (sum + eps);
        for dst in 0..n_hc { out[comb_off + src + dst * n_hc] *= inv; }
    }

    // Remaining Sinkhorn iterations
    for _ in 1..n_iter {
        // Row normalize
        for dst in 0..n_hc {
            let mut sum = 0.0f32;
            for src in 0..n_hc { sum += out[comb_off + src + dst * n_hc]; }
            let inv = 1.0 / (sum + eps);
            for src in 0..n_hc { out[comb_off + src + dst * n_hc] *= inv; }
        }
        // Column normalize
        for src in 0..n_hc {
            let mut sum = 0.0f32;
            for dst in 0..n_hc { sum += out[comb_off + src + dst * n_hc]; }
            let inv = 1.0 / (sum + eps);
            for dst in 0..n_hc { out[comb_off + src + dst * n_hc] *= inv; }
        }
    }
}

/// Weighted sum of HC streams: out[d] = sum_h x[h*n_embd + d] * weights[h]
pub fn hc_weighted_sum_one(
    out: &mut [f32],        // [n_embd]
    x: &[f32],              // [n_hc * n_embd]
    weights: &[f32],        // [n_hc]
    n_embd: usize,
    n_hc: usize,
) {
    for d in 0..n_embd {
        let mut acc = 0.0f32;
        for h in 0..n_hc {
            acc += x[h * n_embd + d] * weights[h];
        }
        out[d] = acc;
    }
}

/// HC post step: mix attention/FFN output with residual HC streams.
/// out[dst*n_embd + d] = block_out[d] * post[dst] + sum_src comb[dst+src*n_hc] * residual_hc[src*n_embd+d]
pub fn hc_post_one(
    out_hc: &mut [f32],     // [n_hc * n_embd]
    block_out: &[f32],      // [n_embd]
    residual_hc: &[f32],    // [n_hc * n_embd]
    post: &[f32],           // [n_hc]
    comb: &[f32],           // [n_hc * n_hc]
    n_embd: usize,
    n_hc: usize,
) {
    for dst in 0..n_hc {
        for d in 0..n_embd {
            let mut acc = block_out[d] * post[dst];
            for src in 0..n_hc {
                acc += comb[dst + src * n_hc] * residual_hc[src * n_embd + d];
            }
            out_hc[dst * n_embd + d] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_stable() {
        assert!((sigmoid_stable(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid_stable(10.0) > 0.999);
        assert!(sigmoid_stable(-10.0) < 0.001);
    }

    #[test]
    fn test_f16_roundtrip() {
        for &x in &[0.0f32, 1.0, -1.0, 0.5, 2.0, 65504.0] {
            let round = f16_to_f32(f32_to_f16(x));
            assert!((round - x).abs() < 0.01 * x.abs().max(0.01));
        }
    }

    #[test]
    fn test_rms_norm() {
        let x = vec![2.0f32; 4];
        let mut out = vec![0.0f32; 4];
        rms_norm(&mut out, &x, 4, 1e-6);
        // ss = 4+4+4+4 = 16, ss/n = 4, sqrt(4) = 2, scale = 1/2 = 0.5
        // out[i] = x[i] * 0.5 = 1.0
        for v in &out {
            assert!((*v - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_layer_compress_ratio() {
        assert_eq!(layer_compress_ratio(0), 0);
        assert_eq!(layer_compress_ratio(1), 0);
        assert_eq!(layer_compress_ratio(2), 4);
        assert_eq!(layer_compress_ratio(3), 128);
        assert_eq!(layer_compress_ratio(4), 4);
        assert_eq!(layer_compress_ratio(5), 128);
        assert_eq!(layer_compress_ratio(6), 4);
        assert_eq!(layer_compress_ratio(7), 128);
    }

    // ========================================================================
    // Q2_K matvec unit tests
    // ========================================================================

    /// Build a synthetic Q2_K block with controlled 2-bit values.
    /// The C Q2_K layout (used by ds4.c scalar path):
    ///   qs[64] bytes, each byte packs one 2-bit value at each of 4 shifts (0,2,4,6).
    ///   - Bytes 0..15:  sub-blocks sharing scales[0,2,4,6]   (chunk 0 half 0)
    ///   - Bytes 16..31: sub-blocks sharing scales[1,3,5,7]   (chunk 0 half 1)
    ///   - Bytes 32..47: sub-blocks sharing scales[8,10,12,14] (chunk 1 half 0)
    ///   - Bytes 48..63: sub-blocks sharing scales[9,11,13,15] (chunk 1 half 1)
    ///
    /// Each byte packs elements at shift 0 (bits 0-1), shift 2 (bits 2-3),
    ///   shift 4 (bits 4-5), shift 6 (bits 6-7).
    ///
    /// q2_vals[256] = the 2-bit integer for each element (0,1,2,3).
    fn make_q2k_block(q2_vals: &[u8; 256], scales: &[u8; 16], d: f32, dmin: f32) -> crate::quant::BlockQ2K {
        let mut qs = [0u8; 64];
        for e in 0..256 {
            let k = e / 128;                      // chunk 0 or 1
            let local = e % 128;
            let half = local / 64;                 // 0 or 1
            let sub = local % 64;
            let shift_layer = sub / 16;            // 0..3 → shifts 0,2,4,6
            let pos = sub % 16;                    // 0..15
            let byte_idx = k * 32 + half * 16 + pos;
            let shift = (shift_layer * 2) as u32;
            qs[byte_idx] |= (q2_vals[e] & 3) << shift;
        }
        crate::quant::BlockQ2K {
            qs,
            scales: *scales,
            d: crate::f32_to_f16(d),
            dmin: crate::f32_to_f16(dmin),
        }
    }

    /// Manual Q2_K × Q8_K dot product matching the C scalar path algorithm.
    fn manual_q2k_q8k_dot(q2k: &crate::quant::BlockQ2K, q8k: &crate::quant::BlockQ8K) -> f64 {
        let d = crate::f16_to_f32(q2k.d) as f64;
        let dmin = crate::f16_to_f32(q2k.dmin) as f64;
        let q8_d = q8k.d as f64;
        let sc = &q2k.scales;
        let qs = &q2k.qs;
        let q8_qs = &q8k.qs;

        // summs = sum of bsums[j] * (sc[j] >> 4)
        let mut summs = 0i64;
        for j in 0..16 {
            summs += (q8k.bsums[j] as i64) * ((sc[j] >> 4) as i64);
        }

        let dall = q8_d * d;
        let dmin_scaled = q8_d * dmin;

        let mut isum = 0i64;
        // 2 chunks × 4 shifts × 2 halves = 16 sub-blocks, matching the C loop
        let mut q2_ptr = 0usize;
        let mut q8_ptr = 0usize;
        let mut is = 0usize;

        for _k in 0..2 {
            for shift in [0u32, 2, 4, 6] {
                // First half: bytes q2[0..15], q8[0..15]
                let ds = (sc[is] & 0x0f) as i64;
                is += 1;
                for i in 0..16 {
                    let q2_val = ((qs[q2_ptr + i] >> shift) & 3) as i64;
                    isum += ds * q2_val * (q8_qs[q8_ptr + i] as i64);
                }

                // Second half: bytes q2[16..31], q8[16..31]
                let ds = (sc[is] & 0x0f) as i64;
                is += 1;
                for i in 0..16 {
                    let q2_val = ((qs[q2_ptr + 16 + i] >> shift) & 3) as i64;
                    isum += ds * q2_val * (q8_qs[q8_ptr + 16 + i] as i64);
                }

                q8_ptr += 32;
            }
            q2_ptr += 32;
        }

        dall * (isum as f64) - dmin_scaled * (summs as f64)
    }

    #[test]
    fn test_q2k_dot_simple() {
        // All zeros: dot product should be 0
        let q2k = make_q2k_block(&[0u8; 256], &[0u8; 16], 1.0, 0.0);
        let mut q8k = crate::quant::BlockQ8K { d: 1.0, qs: [0i8; 256], bsums: [0i16; 16] };
        for i in 0..256 { q8k.qs[i] = 1; }
        for j in 0..16 { q8k.bsums[j] = 16; }

        let result = crate::quant::vec_dot_q2_k_q8_k(
            core::slice::from_ref(&q2k),
            core::slice::from_ref(&q8k),
            1,
        );
        // All Q2 values = 0, so dot = 0 - d*dmin*summs = 0 (dmin=0)
        assert!((result - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_q2k_dot_uniform() {
        // All Q2 values = 1 at all shifts, all Q8 values = 2
        let q2_vals = [1u8; 256];
        // But we need the actual qs bytes to encode 1 at all 4 shifts: 0b01010101 = 0x55
        let q2k = make_q2k_block(&q2_vals, &[0x55u8; 16], 1.0, 0.0);
        let mut q8k = crate::quant::BlockQ8K { d: 2.0, qs: [2i8; 256], bsums: [0i16; 16] };
        for j in 0..16 { q8k.bsums[j] = 32; } // 16 * 2

        let result = crate::quant::vec_dot_q2_k_q8_k(
            core::slice::from_ref(&q2k),
            core::slice::from_ref(&q8k),
            1,
        );
        let expected = manual_q2k_q8k_dot(&q2k, &q8k);
        assert!((result as f64 - expected).abs() < 0.01,
            "result={} expected={}", result, expected);

        // All scales are 5 (lower nibble) → each sub-block gets scale 5
        // Each sub-block: 16 elements × (q2=1) × (q8=2) × scale=5 = 160
        // 16 sub-blocks: 16 × 160 = 2560
        // dall = 2.0 * 1.0 = 2.0, dmin = 0
        // expected = 2.0 * 2560 = 5120.0
        assert!((result - 5120.0).abs() < 1.0,
            "Uniform case should be ~5120, got {}", result);
    }

    #[test]
    fn test_q2k_dot_with_dmin() {
        // Q2 values = 2, scales = 3, d = 1.5, dmin = 0.25, Q8 d = 1.0, q8 = all 1
        let q2_vals = [2u8; 256];
        let scales = [0x33u8; 16]; // lower nibble = 3, upper nibble = 3
        let q2k = make_q2k_block(&q2_vals, &scales, 1.5, 0.25);
        let mut q8k = crate::quant::BlockQ8K { d: 1.0, qs: [1i8; 256], bsums: [0i16; 16] };
        for j in 0..16 { q8k.bsums[j] = 16; }

        let result = crate::quant::vec_dot_q2_k_q8_k(
            core::slice::from_ref(&q2k),
            core::slice::from_ref(&q8k),
            1,
        );
        let expected = manual_q2k_q8k_dot(&q2k, &q8k);
        assert!((result as f64 - expected).abs() < 0.1,
            "result={} expected={}", result, expected);
    }

    #[test]
    fn test_q2k_dot_alternating() {
        // Alternating Q2 values: even elements = 1, odd elements = 3
        let mut q2_vals = [0u8; 256];
        for i in 0..256 {
            q2_vals[i] = if i % 2 == 0 { 1 } else { 3 };
        }
        // Alternating scales: even sub-blocks scale=1, odd scale=7
        let mut scales = [0u8; 16];
        for j in 0..16 {
            scales[j] = if j % 2 == 0 { 0x11 } else { 0x77 };
        }
        let q2k = make_q2k_block(&q2_vals, &scales, 2.0, 0.1);
        // Alternating Q8 values
        let mut q8k = crate::quant::BlockQ8K { d: 0.5, qs: [0i8; 256], bsums: [0i16; 16] };
        for i in 0..256 {
            q8k.qs[i] = if i % 2 == 0 { 3 } else { -1 };
        }
        for j in 0..16 {
            // bsums[j] = sum of q8.qs[j*16..j*16+16]
            let mut s = 0i16;
            for i in 0..16 {
                s += q8k.qs[j * 16 + i] as i16;
            }
            q8k.bsums[j] = s;
        }

        let result = crate::quant::vec_dot_q2_k_q8_k(
            core::slice::from_ref(&q2k),
            core::slice::from_ref(&q8k),
            1,
        );
        let expected = manual_q2k_q8k_dot(&q2k, &q8k);
        assert!((result as f64 - expected).abs() < 0.1,
            "result={} expected={}", result, expected);
    }

    #[test]
    fn test_q2k_dot_multi_block() {
        // Two blocks with different values
        let q2_vals_a = [2u8; 256];
        let q2_vals_b = [1u8; 256];
        let scales = [0x44u8; 16]; // scale=4
        let q2k_a = make_q2k_block(&q2_vals_a, &scales, 1.0, 0.0);
        let q2k_b = make_q2k_block(&q2_vals_b, &scales, 2.0, 0.0);
        let blocks = [q2k_a, q2k_b];

        let q8k_a = crate::quant::BlockQ8K { d: 1.0, qs: [1i8; 256], bsums: [16i16; 16] };
        let q8k_b = crate::quant::BlockQ8K { d: 3.0, qs: [2i8; 256], bsums: [32i16; 16] };
        let q8_blocks = [q8k_a, q8k_b];

        let result = crate::quant::vec_dot_q2_k_q8_k(&blocks, &q8_blocks, 2);
        let expected_a = manual_q2k_q8k_dot(&blocks[0], &q8_blocks[0]);
        let expected_b = manual_q2k_q8k_dot(&blocks[1], &q8_blocks[1]);
        let expected = expected_a + expected_b;
        assert!((result as f64 - expected).abs() < 0.2,
            "result={} expected={} (a={} + b={})", result, expected, expected_a, expected_b);
    }

    // ========================================================================
    // IQ2_XXS matvec unit tests
    // ========================================================================

    /// Build a synthetic IQ2_XXS block with explicit grid/sign/ls values.
    /// 8 groups of 4 u16 each, matching the C encoding.
    fn make_iq2xxs_block(
        grid_indices: &[u8; 32],  // 4 per group × 8 groups
        sign_indices: &[u8; 32],  // 4 per group × 8 groups (0..127)
        extra: &[u8; 8],          // group scale extra (0..15)
        d: f32,
    ) -> crate::quant::BlockIq2Xxs {
        let mut qs = [0u16; 32];
        for g in 0..8 {
            let base = g * 4;
            // lo: grid indices (low bytes of first two u16)
            let lo: u32 =
                (grid_indices[base] as u32) |
                ((grid_indices[base + 1] as u32) << 8) |
                ((grid_indices[base + 2] as u32) << 16) |
                ((grid_indices[base + 3] as u32) << 24);
            // hi: sign indices (7 bits each) + extra (4 bits at top)
            let hi: u32 =
                ((sign_indices[base] as u32) & 0x7f) |
                (((sign_indices[base + 1] as u32) & 0x7f) << 7) |
                (((sign_indices[base + 2] as u32) & 0x7f) << 14) |
                (((sign_indices[base + 3] as u32) & 0x7f) << 21) |
                (((extra[g] as u32) & 0xf) << 28);
            qs[base] = (lo & 0xffff) as u16;
            qs[base + 1] = (lo >> 16) as u16;
            qs[base + 2] = (hi & 0xffff) as u16;
            qs[base + 3] = (hi >> 16) as u16;
        }
        crate::quant::BlockIq2Xxs {
            d: crate::f32_to_f16(d),
            qs,
        }
    }

    /// Manual IQ2_XXS × Q8_K dot product matching C's scalar path.
    fn manual_iq2xxs_q8k_dot(iq2: &crate::quant::BlockIq2Xxs, q8k: &crate::quant::BlockQ8K) -> f64 {
        use crate::quant::{IQ2XXS_GRID, KSIGNS_IQ2XS, iq2xxs_grid_byte};

        let d = crate::f16_to_f32(iq2.d) as f64 * q8k.d as f64;
        let qs = &iq2.qs;
        let q8_qs = &q8k.qs;
        let mut bsum = 0i64;

        for g in 0..8 {
            let base = g * 4;
            let lo: u32 = (qs[base] as u32) | ((qs[base + 1] as u32) << 16);
            let hi: u32 = (qs[base + 2] as u32) | ((qs[base + 3] as u32) << 16);

            let gidx = [
                (lo & 0xff) as usize,
                ((lo >> 8) & 0xff) as usize,
                ((lo >> 16) & 0xff) as usize,
                ((lo >> 24) & 0xff) as usize,
            ];
            let sidx = [
                (hi & 0x7f) as usize,
                ((hi >> 7) & 0x7f) as usize,
                ((hi >> 14) & 0x7f) as usize,
                ((hi >> 21) & 0x7f) as usize,
            ];
            let extra = ((hi >> 28) & 0xf) as i64;
            let ls = 2 * extra + 1;

            let elem_base = g * 32;
            let mut group_sum = 0i64;

            for pair in 0..2 {
                let gi0 = gidx[pair * 2];
                let gi1 = gidx[pair * 2 + 1];
                let si0 = sidx[pair * 2];
                let si1 = sidx[pair * 2 + 1];

                let grid0 = IQ2XXS_GRID[gi0];
                let grid1 = IQ2XXS_GRID[gi1];
                let sbyte0 = KSIGNS_IQ2XS[si0];
                let sbyte1 = KSIGNS_IQ2XS[si1];

                let poff = elem_base + pair * 16;
                for j in 0..8 {
                    let b0 = iq2xxs_grid_byte(grid0, j) as i64;
                    let b1 = iq2xxs_grid_byte(grid1, j) as i64;
                    let sign0 = if (sbyte0 >> j) & 1 != 0 { -1i64 } else { 1i64 };
                    let sign1 = if (sbyte1 >> j) & 1 != 0 { -1i64 } else { 1i64 };
                    group_sum += b0 * sign0 * (q8_qs[poff + j] as i64);
                    group_sum += b1 * sign1 * (q8_qs[poff + 8 + j] as i64);
                }
            }
            bsum += group_sum * ls;
        }

        d * (bsum as f64) * 0.125
    }

    #[test]
    fn test_iq2xxs_dot_simple() {
        // All grid=0, all signs=0 (positive), extra=0 → ls=1
        let grid = [0u8; 32];
        let signs = [0u8; 32];
        let extras = [0u8; 8];
        let iq2 = make_iq2xxs_block(&grid, &signs, &extras, 1.0);

        // Q8 all ones
        let mut q8k = crate::quant::BlockQ8K { d: 1.0, qs: [1i8; 256], bsums: [0i16; 16] };
        for j in 0..16 { q8k.bsums[j] = 16; }

        let result = crate::quant::vec_dot_iq2_xxs_q8_k(
            core::slice::from_ref(&iq2),
            core::slice::from_ref(&q8k),
            1,
        );
        let expected = manual_iq2xxs_q8k_dot(&iq2, &q8k);
        assert!((result as f64 - expected).abs() < 0.01,
            "result={} expected={}", result, expected);
    }

    #[test]
    fn test_iq2xxs_dot_known_grid() {
        // Use grid index 0 (all bytes = 0x08) and grid index 1 (0x2b in some positions)
        // with different signs and extras
        let mut grid = [0u8; 32];
        let mut signs = [0u8; 32];
        let mut extras = [0u8; 8];
        for g in 0..8 {
            let b = g * 4;
            grid[b] = 0;     // grid 0: all 0x08
            grid[b + 1] = 1; // grid 1: 0x2b in positions
            grid[b + 2] = 0;
            grid[b + 3] = 1;
            signs[b] = 0;       // all positive
            signs[b + 1] = 0;
            signs[b + 2] = 127;  // sign index 127 = 0xff = all negative
            signs[b + 3] = 127;
            extras[g] = (g % 4) as u8; // ls = 1, 3, 5, 7, 1, 3, 5, 7
        }
        let iq2 = make_iq2xxs_block(&grid, &signs, &extras, 2.0);

        // Q8 alternating
        let mut q8k = crate::quant::BlockQ8K { d: 0.5, qs: [0i8; 256], bsums: [0i16; 16] };
        for i in 0..256 {
            q8k.qs[i] = (if i % 2 == 0 { 2 } else { -1 }) as i8;
        }
        for j in 0..16 {
            let mut s = 0i16;
            for i in 0..16 {
                s += q8k.qs[j * 16 + i] as i16;
            }
            q8k.bsums[j] = s;
        }

        let result = crate::quant::vec_dot_iq2_xxs_q8_k(
            core::slice::from_ref(&iq2),
            core::slice::from_ref(&q8k),
            1,
        );
        let expected = manual_iq2xxs_q8k_dot(&iq2, &q8k);
        assert!((result as f64 - expected).abs() < 0.1,
            "result={} expected={}", result, expected);
    }

    #[test]
    fn test_iq2xxs_dot_multi_block() {
        let grid = [0u8; 32];
        let signs = [0u8; 32];
        let extras = [0u8; 8];
        let iq2_a = make_iq2xxs_block(&grid, &signs, &extras, 1.0);
        let iq2_b = make_iq2xxs_block(&grid, &signs, &extras, 3.0);
        let blocks = [iq2_a, iq2_b];

        let q8k_a = crate::quant::BlockQ8K { d: 1.0, qs: [1i8; 256], bsums: [16i16; 16] };
        let q8k_b = crate::quant::BlockQ8K { d: 2.0, qs: [2i8; 256], bsums: [32i16; 16] };
        let q8_blocks = [q8k_a, q8k_b];

        let result = crate::quant::vec_dot_iq2_xxs_q8_k(&blocks, &q8_blocks, 2);
        let expected_a = manual_iq2xxs_q8k_dot(&blocks[0], &q8_blocks[0]);
        let expected_b = manual_iq2xxs_q8k_dot(&blocks[1], &q8_blocks[1]);
        let expected = expected_a + expected_b;
        assert!((result as f64 - expected).abs() < 0.1,
            "result={} expected={} (a={} + b={})", result, expected, expected_a, expected_b);
    }

    // ========================================================================
    // IQ2_XXS matvec: dequantize-vs-native golden tests
    // These verify vec_dot_iq2_xxs_q8_k against a full f32 dequantize baseline
    // ========================================================================

    /// Dequantize an IQ2_XXS block to f32 and compute dot product with f32 input.
    /// This is the ground truth: no quantization on the input side.
    fn f32_dot_iq2xxs(block: &crate::quant::BlockIq2Xxs, x: &[f32]) -> f64 {
        let mut f32_w = [0.0f32; 256];
        crate::quant::dequantize_iq2_xxs(block, &mut f32_w);
        let mut sum = 0.0f64;
        for i in 0..256 {
            sum += f32_w[i] as f64 * x[i] as f64;
        }
        sum
    }

    #[test]
    fn test_iq2xxs_vs_f32_dequantize_grid0() {
        // Grid 0: all bytes = 0x08, signs all positive, extra=0 → ls=1
        let grid = [0u8; 32];
        let signs = [0u8; 32];
        let extras = [0u8; 8];
        let iq2 = make_iq2xxs_block(&grid, &signs, &extras, 1.5);

        // Build a known f32 input
        let mut x = [0.0f32; 256];
        for i in 0..256 {
            x[i] = ((i as i32 - 128) as f32) / 64.0; // range roughly -2..+2
        }

        // Quantize x to Q8_K
        let mut xq = [crate::quant::BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
        crate::quant::quantize_q8_k(&x, 256, &mut xq);

        let result = crate::quant::vec_dot_iq2_xxs_q8_k(
            core::slice::from_ref(&iq2),
            core::slice::from_ref(&xq[0]),
            1,
        ) as f64;

        let expected = f32_dot_iq2xxs(&iq2, &x);
        // Q8_K quantization introduces ~0.5% relative error for smooth inputs
        let tolerance = (expected.abs() * 0.01).max(1e-3);
        assert!((result - expected).abs() < tolerance,
            "result={} expected={} diff={:.2e}", result, expected, (result - expected).abs());
    }

    #[test]
    fn test_iq2xxs_vs_f32_dequantize_varied_grids() {
        // Use diverse grid indices and sign patterns
        let mut rng_state: u32 = 0xDEADBEEF;
        let mut next_u32 = move || -> u32 {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            rng_state
        };

        for block_seed in 0..8 {
            let mut grid = [0u8; 32];
            let mut signs = [0u8; 32];
            let mut extras = [0u8; 8];
            for g in 0..8 {
                let b = g * 4;
                let r = next_u32();
                grid[b] = (r & 0xff) as u8;
                grid[b + 1] = ((r >> 8) & 0xff) as u8;
                grid[b + 2] = ((r >> 16) & 0xff) as u8;
                grid[b + 3] = ((r >> 24) & 0xff) as u8;
                let r2 = next_u32();
                signs[b] = (r2 & 0x7f) as u8;
                signs[b + 1] = ((r2 >> 7) & 0x7f) as u8;
                signs[b + 2] = ((r2 >> 14) & 0x7f) as u8;
                signs[b + 3] = ((r2 >> 21) & 0x7f) as u8;
                extras[g] = (next_u32() & 0xf) as u8;
            }
            let d = 0.5 + (block_seed as f32) * 0.5;

            let iq2 = make_iq2xxs_block(&grid, &signs, &extras, d);

            // Sinusoidal input
            let mut x = [0.0f32; 256];
            for i in 0..256 {
                x[i] = ((i as f32 * 0.123 + block_seed as f32 * 0.7).sin() * 3.0) as f32;
            }

            let mut xq = [crate::quant::BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
            crate::quant::quantize_q8_k(&x, 256, &mut xq);

            let result = crate::quant::vec_dot_iq2_xxs_q8_k(
                core::slice::from_ref(&iq2),
                core::slice::from_ref(&xq[0]),
                1,
            ) as f64;

            let expected = f32_dot_iq2xxs(&iq2, &x);
            // Q8_K quantization: up to 5% relative for sinusoidal inputs with varied grids;
            // purpose is to catch catastrophic deviations (e.g. 8x errors), not bit-exactness
            let tolerance = (expected.abs() * 0.05).max(1e-3);
            assert!((result - expected).abs() < tolerance,
                "seed={}: result={} expected={} diff={:.2e}",
                block_seed, result, expected, (result - expected).abs());
        }
    }

    #[test]
    fn test_iq2xxs_edge_max_ls() {
        // Maximum ls = 31 (extra=15), maximum grid values
        // Grid index 255: bytes = [0x2b, 0x2b, 0x2b, 0x19, 0x08, 0x08, 0x08, 0x19]
        let mut grid = [0u8; 32];
        let mut signs = [0u8; 32];
        let mut extras = [0u8; 8];
        for g in 0..8 {
            let b = g * 4;
            grid[b] = 255;
            grid[b + 1] = 255;
            grid[b + 2] = 255;
            grid[b + 3] = 255;
            signs[b] = 0;     // all positive
            signs[b + 1] = 0;
            signs[b + 2] = 0;
            signs[b + 3] = 0;
            extras[g] = 15;   // ls = 2*15+1 = 31
        }
        let iq2 = make_iq2xxs_block(&grid, &signs, &extras, 1.0);

        // Input: ramp
        let mut x = [0.0f32; 256];
        for i in 0..256 {
            x[i] = (i as f32) / 256.0;
        }

        let mut xq = [crate::quant::BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
        crate::quant::quantize_q8_k(&x, 256, &mut xq);

        let result = crate::quant::vec_dot_iq2_xxs_q8_k(
            core::slice::from_ref(&iq2),
            core::slice::from_ref(&xq[0]),
            1,
        ) as f64;

        let expected = f32_dot_iq2xxs(&iq2, &x);
        // Q8_K quantization error acceptable for edge case with large ls
        let tolerance = (expected.abs() * 0.01).max(1e-3);
        assert!((result - expected).abs() < tolerance,
            "max_ls: result={} expected={} diff={:.2e}",
            result, expected, (result - expected).abs());
    }

    #[test]
    fn test_iq2xxs_edge_mixed_signs_large_ls() {
        // Alternating sign patterns with large ls values
        let mut grid = [0u8; 32];
        let mut signs = [0u8; 32];
        let mut extras = [0u8; 8];
        for g in 0..8 {
            let b = g * 4;
            grid[b] = 0;
            grid[b + 1] = 128;  // 0x2b...
            grid[b + 2] = 0;
            grid[b + 3] = 128;
            signs[b] = 0;       // positive
            signs[b + 1] = 127; // all negative
            signs[b + 2] = 85;  // alternating signs
            signs[b + 3] = 42;  // sparse signs
            extras[g] = if g < 4 { 0 } else { 7 }; // ls = 1 or 15
        }
        let iq2 = make_iq2xxs_block(&grid, &signs, &extras, 4.0);

        // Input: large magnitudes
        let mut x = [0.0f32; 256];
        for i in 0..256 {
            x[i] = (i as f32 - 128.0) * 0.05; // range -6.4..+6.35
        }

        let mut xq = [crate::quant::BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
        crate::quant::quantize_q8_k(&x, 256, &mut xq);

        let result = crate::quant::vec_dot_iq2_xxs_q8_k(
            core::slice::from_ref(&iq2),
            core::slice::from_ref(&xq[0]),
            1,
        ) as f64;

        let expected = f32_dot_iq2xxs(&iq2, &x);
        let tolerance = (expected.abs() * 0.01).max(1e-3);
        assert!((result - expected).abs() < tolerance,
            "mixed_signs: result={} expected={} diff={:.2e}",
            result, expected, (result - expected).abs());
    }

    // ========================================================================
    // IQ2_XXS multi-block matvec: simulate expert gate/up row projection
    // In the real forward pass, each expert gate/up row spans n_embd/QK_K blocks
    // ========================================================================

    #[test]
    fn test_iq2xxs_multi_block_vs_f32_dequantize() {
        // Simulate one row of an IQ2_XXS expert gate/up matrix:
        // 16 IQ2_XXS blocks for 4096 input dim, dot with quantized input
        let n_blocks: usize = 16;

        // Build diverse IQ2_XXS blocks
        let mut iq2_blocks = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let mut grid = [0u8; 32];
            let mut signs = [0u8; 32];
            let mut extras = [0u8; 8];
            for g in 0..8 {
                let off = g * 4;
                grid[off] = ((b * 7 + g * 3) % 256) as u8;
                grid[off + 1] = ((b * 13 + g * 5 + 1) % 256) as u8;
                grid[off + 2] = ((b * 17 + g * 7 + 2) % 256) as u8;
                grid[off + 3] = ((b * 19 + g * 11 + 3) % 256) as u8;
                signs[off] = ((b * 23 + g * 13) % 128) as u8;
                signs[off + 1] = ((b * 29 + g * 17 + 1) % 128) as u8;
                signs[off + 2] = ((b * 31 + g * 19 + 2) % 128) as u8;
                signs[off + 3] = ((b * 37 + g * 23 + 3) % 128) as u8;
                extras[g] = ((b * 3 + g * 2) % 16) as u8;
            }
            let d = 0.7 + (b as f32) * 0.15;
            iq2_blocks.push(make_iq2xxs_block(&grid, &signs, &extras, d));
        }

        // Build f32 input: decaying sinusoid
        let mut x = vec![0.0f32; n_blocks * 256];
        for i in 0..x.len() {
            x[i] = ((i as f32 * 0.05).sin() * (i as f32 * 0.001).exp()) as f32;
        }

        // Quantize x to Q8_K blocks
        let mut xq_blocks = vec![crate::quant::BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }; n_blocks];
        crate::quant::quantize_q8_k(&x, x.len(), &mut xq_blocks);

        // Native dot
        let result = crate::quant::vec_dot_iq2_xxs_q8_k(&iq2_blocks, &xq_blocks, n_blocks) as f64;

        // F32 dequantize ground truth
        let mut expected = 0.0f64;
        for b in 0..n_blocks {
            expected += f32_dot_iq2xxs(&iq2_blocks[b], &x[b * 256..(b + 1) * 256]);
        }

        // Q8_K quantization: ~5% relative tolerance for multi-block with decaying signal
        let tolerance = (expected.abs() * 0.05).max(1e-2);
        assert!((result - expected).abs() < tolerance,
            "multi_block: result={} expected={} diff={:.2e}",
            result, expected, (result - expected).abs());
    }

    #[test]
    fn test_iq2xxs_matvec_zero_input() {
        // All-zero input: result should be zero
        let grid = [0u8; 32];
        let signs = [0u8; 32];
        let extras = [0u8; 8];
        let iq2 = make_iq2xxs_block(&grid, &signs, &extras, 1.0);

        let x = [0.0f32; 256];
        let mut xq = [crate::quant::BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
        crate::quant::quantize_q8_k(&x, 256, &mut xq);

        let result = crate::quant::vec_dot_iq2_xxs_q8_k(
            core::slice::from_ref(&iq2),
            core::slice::from_ref(&xq[0]),
            1,
        );
        assert!(result.abs() < 1e-6, "zero input should give zero, got {}", result);
    }
}
