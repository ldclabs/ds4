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
}
