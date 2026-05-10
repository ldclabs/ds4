// Quantization block formats and CPU dequantization/dot-product kernels.
// Based on the GGUF quant formats used by ds4.c: Q2_K, Q4_K, IQ2_XXS, Q8_K.

use bytemuck::{Pod, Zeroable};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================================
// Block format definitions
// ============================================================================

/// Q2_K block: 256 elements, 84 bytes.
/// Super-block with 16 sub-blocks of 16 elements each.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ2K {
    pub scales: [u8; 16],  // 6-bit scales for 16 sub-blocks
    pub qs: [u8; 64],      // 2-bit values packed 4 per byte
    pub d: u16,            // super-block scale (f16)
    pub dmin: u16,         // super-block min (f16)
}

/// Q4_K block: 256 elements, 144 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ4K {
    pub d: u16,            // super-block scale (f16)
    pub dmin: u16,         // super-block min (f16)
    pub scales: [u8; 12],  // 6-bit scales for 8 sub-blocks
    pub qs: [u8; 128],     // 4-bit values packed 2 per byte
}

/// Q8_K block: 256 elements, 292 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ8K {
    pub d: f32,            // scale
    pub qs: [i8; 256],     // 8-bit quantized values
    pub bsums: [i16; 16],  // block sums for 16 sub-blocks
}

/// Q8_0 block: 32 elements, 34 bytes (standard llama.cpp Q8_0).
/// Used by DeepSeek V4 Flash for most weight tensors.
/// Layout: d (f16, 2 bytes) + 32 q8 values (i8, 32 bytes) = 34 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ80 {
    pub d: u16,            // scale (f16)
    pub qs: [i8; 32],      // 8-bit quantized values
}

/// Dequantize a single Q8_0 block: out[i] = d * qs[i]
pub fn dequantize_q8_0(block: &BlockQ80, out: &mut [f32; 32]) {
    let d = crate::f16_to_f32(block.d);
    for i in 0..32 {
        out[i] = d * (block.qs[i] as f32);
    }
}

/// Matrix-vector multiply for Q8_0 weights: out = x @ W^T
/// W is [out_dim, in_dim] stored in Q8_0 blocks.
/// Dispatches to AVX2 on x86_64, scalar fallback otherwise.
/// Uses integer dot product with i32 accumulation matching ds4.c's approach.
#[inline]
pub fn matvec_q8_0(out: &mut [f32], x: &[f32], weight: &[u8], in_dim: usize, out_dim: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { matvec_q8_0_avx2(out, x, weight, in_dim, out_dim); return; }
        }
    }
    matvec_q8_0_scalar(out, x, weight, in_dim, out_dim);
}

fn matvec_q8_0_scalar(out: &mut [f32], x: &[f32], weight: &[u8], in_dim: usize, out_dim: usize) {
    let n_blocks = in_dim / 32;
    assert!(in_dim % 32 == 0, "Q8_0 requires in_dim divisible by 32");
    let block_size = std::mem::size_of::<BlockQ80>();
    let expected_bytes = n_blocks * out_dim * block_size;
    assert!(weight.len() >= expected_bytes);

    let blocks: &[BlockQ80] = bytemuck::cast_slice(
        &weight[..n_blocks * out_dim * block_size]
    );

    // Quantize input activations per block to match C's quantize_q8_0_activation
    // Each block of 32 elements gets its own scale: scale = amax / 127.0
    let mut xq = vec![0i8; n_blocks * 32];
    let mut xscale = vec![0.0f32; n_blocks];
    for b in 0..n_blocks {
        let base = b * 32;
        let mut amax = 0.0f32;
        for i in 0..32 {
            let ax = x[base + i].abs();
            if ax > amax { amax = ax; }
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        xscale[b] = d;
        for i in 0..32 {
            let v = (x[base + i] * id).round().clamp(-128.0, 127.0) as i8;
            xq[base + i] = v;
        }
    }

    for o in 0..out_dim {
        let mut sum = 0.0f32;
        for b in 0..n_blocks {
            let block = &blocks[o * n_blocks + b];
            let d_w = crate::f16_to_f32(block.d);
            let d_x = xscale[b];
            let base = b * 32;
            let mut dot = 0i32;
            for i in 0..32 {
                dot += (xq[base + i] as i32) * (block.qs[i] as i32);
            }
            // Product: (d_x * xq) · (d_w * qs) = d_x * d_w * dot
            sum += d_x * d_w * (dot as f32);
        }
        out[o] = sum;
    }
}

/// AVX2-accelerated Q8_0 matvec.
/// Uses _mm256_cvtepi8_epi16 + _mm256_mullo_epi16 + _mm256_madd_epi16
/// to compute the 32-element i8×i8 dot product in SIMD.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn matvec_q8_0_avx2(out: &mut [f32], x: &[f32], weight: &[u8], in_dim: usize, out_dim: usize) {
    let n_blocks = in_dim / 32;
    assert!(in_dim % 32 == 0);
    let block_size = std::mem::size_of::<BlockQ80>();
    let expected_bytes = n_blocks * out_dim * block_size;
    assert!(weight.len() >= expected_bytes);

    let blocks: &[BlockQ80] = bytemuck::cast_slice(
        &weight[..n_blocks * out_dim * block_size]
    );

    // Quantize input (same as scalar, scalar loop is fine — 32 elements only)
    let mut xq = vec![0i8; n_blocks * 32];
    let mut xscale = vec![0.0f32; n_blocks];
    for b in 0..n_blocks {
        let base = b * 32;
        let mut amax = 0.0f32;
        for i in 0..32 {
            let ax = x[base + i].abs();
            if ax > amax { amax = ax; }
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        xscale[b] = d;
        for i in 0..32 {
            let v = (x[base + i] * id).round().clamp(-128.0, 127.0) as i8;
            xq[base + i] = v;
        }
    }

    let ones = _mm256_set1_epi16(1);

    for o in 0..out_dim {
        let mut sum = 0.0f32;
        for b in 0..n_blocks {
            let block = &blocks[o * n_blocks + b];
            let d_w = crate::f16_to_f32(block.d);
            let d_x = xscale[b];
            let base = b * 32;
            let xq_ptr = xq.as_ptr().add(base);
            let qs_ptr = block.qs.as_ptr();

            // Load 16 i8 → extend to 16×i16 (lower half: elements 0..15)
            let xq_lo_i8 = _mm_loadu_si128(xq_ptr as *const __m128i);
            let xq_lo_i16 = _mm256_cvtepi8_epi16(xq_lo_i8);
            let qs_lo_i8 = _mm_loadu_si128(qs_ptr as *const __m128i);
            let qs_lo_i16 = _mm256_cvtepi8_epi16(qs_lo_i8);

            // Load 16 i8 → extend to 16×i16 (upper half: elements 16..31)
            let xq_hi_i8 = _mm_loadu_si128(xq_ptr.add(16) as *const __m128i);
            let xq_hi_i16 = _mm256_cvtepi8_epi16(xq_hi_i8);
            let qs_hi_i8 = _mm_loadu_si128(qs_ptr.add(16) as *const __m128i);
            let qs_hi_i16 = _mm256_cvtepi8_epi16(qs_hi_i8);

            // i16×i16 → i16 (low 16 bits ok — i8×i8 fits in i16)
            let mul_lo = _mm256_mullo_epi16(xq_lo_i16, qs_lo_i16);
            let mul_hi = _mm256_mullo_epi16(xq_hi_i16, qs_hi_i16);

            // Horizontal add adjacent pairs: i16+i16 → i32
            // madd_epi16(a, ones) = [a0+a1, a2+a3, ..., a14+a15]
            let acc_lo = _mm256_madd_epi16(mul_lo, ones);
            let acc_hi = _mm256_madd_epi16(mul_hi, ones);

            // Sum all 8 i32 lanes
            let sum_all = _mm256_add_epi32(acc_lo, acc_hi);
            let hadd = _mm256_hadd_epi32(sum_all, sum_all);
            let hadd2 = _mm256_hadd_epi32(hadd, hadd);
            let dot = _mm_cvtsi128_si32(_mm256_extracti128_si256::<0>(hadd2))
                    + _mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(hadd2));

            sum += d_x * d_w * (dot as f32);
        }
        out[o] = sum;
    }
}

/// Quantize activation for Q8_0 matvec: per-block amax/127.
/// Matches C's quantize_q8_0_activation.
pub fn quantize_q8_0_activation(x: &[f32], xq: &mut [i8], xscale: &mut [f32], in_dim: u32) {
    let n = in_dim as usize;
    let block_size = 32usize;
    let n_blocks = (n + block_size - 1) / block_size;

    for b in 0..n_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(n);

        // Find amax
        let mut amax = 0.0f32;
        for i in start..end {
            let abs = x[i].abs();
            if abs > amax { amax = abs; }
        }

        if amax < 1e-8 {
            xscale[b] = 0.0;
            for i in start..end {
                xq[i] = 0;
            }
            continue;
        }

        let id = 127.0 / amax;
        xscale[b] = amax / 127.0;
        for i in start..end {
            let q = (x[i] * id).round() as i32;
            xq[i] = q.clamp(-127, 127) as i8;
        }
    }
}

/// Pair matvec for compressor KV+gate from pre-quantized Q8_0 activations.
/// Matches C's matvec_q8_0_pair_prequant.
/// This reads Q8_0 weight tensors and does dot products with the pre-quantized x.
pub fn matvec_q8_0_pair_prequant(
    kv_out: &mut [f32],
    sc_out: &mut [f32],
    kv_tensor: &crate::model::Tensor,
    gate_tensor: &crate::model::Tensor,
    xq: &[i8],
    xscale: &[f32],
) {
    let in_dim = xq.len();
    let out_dim = kv_out.len();
    let block_size = 32usize;
    let n_blocks = (in_dim + block_size - 1) / block_size;

    // KV matvec
    let kv_bytes = kv_tensor.as_bytes();
    if !kv_bytes.is_empty() {
        let blocks: &[crate::quant::BlockQ80] = bytemuck::cast_slice(
            &kv_bytes[..n_blocks * out_dim * std::mem::size_of::<crate::quant::BlockQ80>()]
        );
        for o in 0..out_dim {
            let mut sum = 0.0f32;
            for b in 0..n_blocks {
                let block = &blocks[o * n_blocks + b];
                let d = crate::f16_to_f32(block.d);
                let start = b * block_size;
                let end = (start + block_size).min(in_dim);
                for i in start..end {
                    sum += (xq[i] as f32) * xscale[b] * d * (block.qs[i - start] as f32);
                }
            }
            kv_out[o] = sum;
        }
    }

    // Gate matvec
    let gate_bytes = gate_tensor.as_bytes();
    if !gate_bytes.is_empty() {
        let blocks: &[crate::quant::BlockQ80] = bytemuck::cast_slice(
            &gate_bytes[..n_blocks * out_dim * std::mem::size_of::<crate::quant::BlockQ80>()]
        );
        for o in 0..out_dim {
            let mut sum = 0.0f32;
            for b in 0..n_blocks {
                let block = &blocks[o * n_blocks + b];
                let d = crate::f16_to_f32(block.d);
                let start = b * block_size;
                let end = (start + block_size).min(in_dim);
                for i in start..end {
                    sum += (xq[i] as f32) * xscale[b] * d * (block.qs[i - start] as f32);
                }
            }
            sc_out[o] = sum;
        }
    }
}

/// IQ2_XXS block: 256 elements, 66 bytes.
/// 2-bit importance-quantized with 256-element super-block.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockIq2Xxs {
    pub d: u16,            // super-block scale (f16)
    pub qs: [u16; 32],     // packed: each u16 encodes 8 2-bit values + signs
}

// ============================================================================
// Q2_K dequantization
// ============================================================================
// Q2_K super-block structure:
// - d: scale (f16)
// - dmin: min (f16)
// - scales: 16 bytes, each encodes a 6-bit scale for 16 elements
// - qs: 64 bytes, each byte packs 4 2-bit values

/// Dequantize a single Q2_K block to f32.
/// Matches the C Q2_K byte layout used by ds4.c's scalar matvec path:
/// - 64 qs bytes, each byte packs one 2-bit value at each of 4 shifts (0,2,4,6)
/// - Bytes 0..15,16..31 form chunk 0; bytes 32..47,48..63 form chunk 1
/// - Sub-block scales: sc[0,2,4,6] for chunk 0 half 0; sc[1,3,5,7] for chunk 0 half 1
/// - sc[8,10,12,14] for chunk 1 half 0; sc[9,11,13,15] for chunk 1 half 1
pub fn dequantize_q2_k(block: &BlockQ2K, out: &mut [f32; 256]) {
    let d = crate::f16_to_f32(block.d);
    let dmin = crate::f16_to_f32(block.dmin);
    let sc = &block.scales;
    let qs = &block.qs;

    // Element ordering matches C's ds4_vec_dot_q2_K_q8_K kernel:
    // elements are processed in shift-major order (all shift-0 first, then
    // shift-2, shift-4, shift-6), matching how the kernel consumes Q8 values
    // sequentially.
    for e in 0..256 {
        let k = e / 128;                   // chunk 0 or 1
        let local = e % 128;
        let shift_layer = local / 32;      // 0..3 → shifts 0,2,4,6
        let half = (local % 32) / 16;      // 0 or 1
        let pos = local % 16;              // 0..15
        let byte_idx = k * 32 + half * 16 + pos;
        let shift = (shift_layer * 2) as u32;
        let q = ((qs[byte_idx] >> shift) & 3) as f32;
        let scale_idx = k * 8 + shift_layer * 2 + half;
        // Q2_K scale byte: lower nibble = d_scale, upper nibble = m_scale
        let d_scale = (sc[scale_idx] & 0x0f) as f32;
        let m_scale = (sc[scale_idx] >> 4) as f32;
        let sub_d = d * d_scale;
        let sub_m = dmin * m_scale;
        out[e] = sub_d * q - sub_m;
    }
}

/// CPU dot-product of N Q2_K blocks with Q8_K activations.
/// Matches C's ds4_vec_dot_q2_K_q8_K path.
/// Dispatches to AVX2 (SSE4.1 inner kernel) when available, scalar otherwise.
pub fn vec_dot_q2_k_q8_k(block_q2: &[BlockQ2K], q8: &[BlockQ8K], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { return vec_dot_q2_k_q8_k_avx2(block_q2, q8, n); }
        }
    }
    vec_dot_q2_k_q8_k_scalar(block_q2, q8, n)
}

/// Scalar implementation: dot_q2_16 is a 16-iteration loop.
fn vec_dot_q2_k_q8_k_scalar(block_q2: &[BlockQ2K], q8: &[BlockQ8K], n: usize) -> f32 {
    let mut sumf = 0.0f32;

    for i in 0..n {
        let q2_qs = &block_q2[i].qs;
        let q8_qs = &q8[i].qs;
        let sc = &block_q2[i].scales;

        // summs: sum of bsums[j] * (sc[j] >> 4) = mins contribution
        let mut summs = 0i32;
        for j in 0..16 {
            summs += (q8[i].bsums[j] as i32) * ((sc[j] >> 4) as i32);
        }

        let dall = q8[i].d * crate::f16_to_f32(block_q2[i].d);
        let dmin = q8[i].d * crate::f16_to_f32(block_q2[i].dmin);

        let mut is = 0usize;
        let isum = q2_k_inner_loop(q2_qs, q8_qs, sc, &mut is);
        sumf += dall * (isum as f32) - dmin * (summs as f32);
    }

    sumf
}

/// AVX2 implementation: uses dot_q2_32_avx2 with _mm256_maddubs_epi16
/// for byte-level multiply-accumulate, processing 32 elements per kernel call.
/// The outer loop (summs, dall/dmin) is identical to scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_q2_k_q8_k_avx2(block_q2: &[BlockQ2K], q8: &[BlockQ8K], n: usize) -> f32 {
    let mut sumf = 0.0f32;

    for i in 0..n {
        let q2_qs = &block_q2[i].qs;
        let q8_qs = &q8[i].qs;
        let sc = &block_q2[i].scales;

        let mut summs = 0i32;
        for j in 0..16 {
            summs += (q8[i].bsums[j] as i32) * ((sc[j] >> 4) as i32);
        }

        let dall = q8[i].d * crate::f16_to_f32(block_q2[i].d);
        let dmin = q8[i].d * crate::f16_to_f32(block_q2[i].dmin);

        let mut is = 0usize;
        let isum = q2_k_inner_loop_avx2(q2_qs, q8_qs, sc, &mut is);
        sumf += dall * (isum as f32) - dmin * (summs as f32);
    }

    sumf
}

/// Shared inner loop: 2 chunks × 4 shifts × 2 groups. Used by scalar path.
#[inline]
fn q2_k_inner_loop(q2_qs: &[u8], q8_qs: &[i8], sc: &[u8], is: &mut usize) -> i32 {
    let mut isum = 0i32;
    let mut q2_pos = 0usize;
    let mut q8_pos = 0usize;

    for _k in 0..2 {
        let mut shift = 0u32;
        for _j in 0..4 {
            let d_scale = (sc[*is] & 0x0f) as i32;
            *is += 1;
            isum += d_scale * dot_q2_16(&q2_qs[q2_pos..], &q8_qs[q8_pos..], shift);

            let d_scale = (sc[*is] & 0x0f) as i32;
            *is += 1;
            isum += d_scale * dot_q2_16(&q2_qs[q2_pos + 16..], &q8_qs[q8_pos + 16..], shift);

            shift += 2;
            q8_pos += 32;
        }
        q2_pos += 32;
    }
    isum
}

/// AVX2 inner loop: uses dot_q2_32_avx2 to process both 16-element groups
/// at each shift level in a single call, halving the number of kernel invocations
/// from 16 to 8 per Q2_K block. Each call returns two pre-scale dot products
/// which are multiplied by their respective d_scale values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q2_k_inner_loop_avx2(q2_qs: &[u8], q8_qs: &[i8], sc: &[u8], is: &mut usize) -> i32 {
    let mut isum = 0i32;
    let mut q2_pos = 0usize;
    let mut q8_pos = 0usize;

    for _k in 0..2 {
        // Shift 0: groups 0+1
        {
            let d0 = (sc[*is] & 0x0f) as i32; *is += 1;
            let d1 = (sc[*is] & 0x0f) as i32; *is += 1;
            let (s0, s1) = dot_q2_32_avx2(
                q2_qs[q2_pos..].as_ptr(),
                q8_qs[q8_pos..].as_ptr(),
                0,
            );
            isum += d0 * s0 + d1 * s1;
        }
        // Shift 2: groups 2+3
        {
            let d0 = (sc[*is] & 0x0f) as i32; *is += 1;
            let d1 = (sc[*is] & 0x0f) as i32; *is += 1;
            let (s0, s1) = dot_q2_32_avx2(
                q2_qs[q2_pos..].as_ptr(),
                q8_qs[q8_pos + 32..].as_ptr(),
                2,
            );
            isum += d0 * s0 + d1 * s1;
        }
        // Shift 4: groups 4+5
        {
            let d0 = (sc[*is] & 0x0f) as i32; *is += 1;
            let d1 = (sc[*is] & 0x0f) as i32; *is += 1;
            let (s0, s1) = dot_q2_32_avx2(
                q2_qs[q2_pos..].as_ptr(),
                q8_qs[q8_pos + 64..].as_ptr(),
                4,
            );
            isum += d0 * s0 + d1 * s1;
        }
        // Shift 6: groups 6+7
        {
            let d0 = (sc[*is] & 0x0f) as i32; *is += 1;
            let d1 = (sc[*is] & 0x0f) as i32; *is += 1;
            let (s0, s1) = dot_q2_32_avx2(
                q2_qs[q2_pos..].as_ptr(),
                q8_qs[q8_pos + 96..].as_ptr(),
                6,
            );
            isum += d0 * s0 + d1 * s1;
        }
        q8_pos += 128;
        q2_pos += 32;
    }
    isum
}

/// Scalar dot product of 16 Q2 values with 16 Q8 values at a given bit shift.
#[inline]
fn dot_q2_16(q2: &[u8], q8: &[i8], shift: u32) -> i32 {
    let mut sum = 0i32;
    for i in 0..16 {
        sum += (q8[i] as i32) * (((q2[i] >> shift) & 3) as i32);
    }
    sum
}

/// AVX2 dot product of 32 Q2 values with 32 Q8 values at a given bit shift.
///
/// Returns (sum_first_16, sum_second_16) — two independent pre-scale dot products.
/// This enables processing both 16-element groups at a shift level in one call,
/// each with its own d_scale multiplier applied by the caller.
///
/// Strategy (matching the goal's _mm256_maddubs_epi16 approach):
/// 1. Load 32 packed q2 bytes → extract 2-bit values at `shift` → pack as u8
/// 2. Load 32 q8 bytes as i8
/// 3. _mm256_maddubs_epi16: u8×i8 byte-level multiply → 16 adjacent-pair accumulates → 16×i16
/// 4. Split into two 128-bit halves (first/second 16 elements), horizontal-sum each → 2×i32
///
/// Note on _mm256_srlv_epi32: variable per-lane 32-bit shifts were evaluated for
/// 2-bit extraction but are incompatible with dense-packed byte data — shifting
/// multi-byte 32-bit lanes causes cross-byte bit contamination that corrupts
/// byte-aligned 2-bit values. The branching-on-immediate approach with
/// _mm256_srli_epi16 is the correct solution for per-byte shifts.
/// _mm256_maddubs_epi16 is the key optimization, providing byte-level multiply-add
/// in a single instruction (vs two _mm_madd_epi16 in SSE4.1).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot_q2_32_avx2(q2_ptr: *const u8, q8_ptr: *const i8, shift: u32) -> (i32, i32) {
    // Load 32 q2 bytes
    let q2_bytes = _mm256_loadu_si256(q2_ptr as *const __m256i);

    // Split into two 128-bit halves, widen each u8 → i16
    let q2_lo128 = _mm256_castsi256_si128(q2_bytes);           // bytes 0..15
    let q2_hi128 = _mm256_extracti128_si256::<1>(q2_bytes);    // bytes 16..31
    let q2_lo_i16 = _mm256_cvtepu8_epi16(q2_lo128);            // 16 × i16
    let q2_hi_i16 = _mm256_cvtepu8_epi16(q2_hi128);            // 16 × i16

    // Extract 2-bit values: shift right by `shift`, mask with 0x03.
    // Branch on shift value for _mm256_srli_epi16 immediate (required by ISA;
    // there is no _mm256_srl_epi16 register-shift variant in AVX2).
    let mask = _mm256_set1_epi16(3);
    let (q2_lo_s, q2_hi_s) = match shift {
        0 => (q2_lo_i16, q2_hi_i16),
        2 => (_mm256_srli_epi16::<2>(q2_lo_i16), _mm256_srli_epi16::<2>(q2_hi_i16)),
        4 => (_mm256_srli_epi16::<4>(q2_lo_i16), _mm256_srli_epi16::<4>(q2_hi_i16)),
        6 => (_mm256_srli_epi16::<6>(q2_lo_i16), _mm256_srli_epi16::<6>(q2_hi_i16)),
        _ => core::hint::unreachable_unchecked(),
    };
    let q2_lo_m = _mm256_and_si256(q2_lo_s, mask);  // 16 × i16, values 0..3
    let q2_hi_m = _mm256_and_si256(q2_hi_s, mask);

    // Pack back to u8: each 256-bit register → 128-bit of packed bytes
    // _mm_packus_epi16: [a_lo[0..7], a_hi[0..7]] as u8
    let q2_lo_lo = _mm256_castsi256_si128(q2_lo_m);
    let q2_lo_hi = _mm256_extracti128_si256::<1>(q2_lo_m);
    let q2_lo_packed = _mm_packus_epi16(q2_lo_lo, q2_lo_hi);  // 16 u8: bytes 0..15

    let q2_hi_lo = _mm256_castsi256_si128(q2_hi_m);
    let q2_hi_hi = _mm256_extracti128_si256::<1>(q2_hi_m);
    let q2_hi_packed = _mm_packus_epi16(q2_hi_lo, q2_hi_hi);  // 16 u8: bytes 16..31

    // Recombine into 256-bit: lower 128 = bytes 0..15, upper 128 = bytes 16..31
    let q2_u8 = _mm256_set_m128i(q2_hi_packed, q2_lo_packed);

    // Load 32 q8 values as signed bytes
    let q8_bytes = _mm256_loadu_si256(q8_ptr as *const __m256i);

    // _mm256_maddubs_epi16: unsigned bytes × signed bytes → adjacent-pair accumulate
    // Produces 16 × i16: [q2[0]*q8[0]+q2[1]*q8[1], q2[2]*q8[2]+q2[3]*q8[3], ...]
    let madd = _mm256_maddubs_epi16(q2_u8, q8_bytes);

    // Split: first 8 pairs (elements 0..15) → lower 128, last 8 pairs (elements 16..31) → upper 128
    let lo_4xi32 = _mm256_castsi256_si128(madd);
    let hi_4xi32 = _mm256_extracti128_si256::<1>(madd);

    // Horizontal sum each 128-bit half (4 × i32 → 1 × i32)
    let sum0 = {
        let h = _mm_hadd_epi32(lo_4xi32, lo_4xi32);
        let h = _mm_hadd_epi32(h, h);
        _mm_cvtsi128_si32(h)
    };
    let sum1 = {
        let h = _mm_hadd_epi32(hi_4xi32, hi_4xi32);
        let h = _mm_hadd_epi32(h, h);
        _mm_cvtsi128_si32(h)
    };

    (sum0, sum1)
}

/// Simpler per-element dot product for Q2_K × f32.
/// Uses the C Q2_K byte layout (same as dequantize_q2_k).
pub fn vec_dot_q2_k_f32(blocks: &[BlockQ2K], x: &[f32], n_blocks: usize) -> f32 {
    let n = n_blocks * 256;
    let mut sum = 0.0f32;
    let mut elem_idx = 0usize;

    for b in 0..n_blocks {
        let d = crate::f16_to_f32(blocks[b].d);
        let dmin = crate::f16_to_f32(blocks[b].dmin);
        let sc = &blocks[b].scales;
        let qs = &blocks[b].qs;

        for e in 0..256 {
            let k = e / 128;
            let local = e % 128;
            let half = local / 64;
            let sub = local % 64;
            let shift_layer = sub / 16;
            let pos = sub % 16;
            let byte_idx = k * 32 + half * 16 + pos;
            let shift = (shift_layer * 2) as u32;
            let q = ((qs[byte_idx] >> shift) & 3) as f32;
            let scale_idx = k * 8 + shift_layer * 2 + half;
            let d_scale = (sc[scale_idx] & 0x0f) as f32;
            let m_scale = (sc[scale_idx] >> 4) as f32;
            let sub_d = d * d_scale;
            let sub_m = dmin * m_scale;
            let w = sub_d * q - sub_m;
            if elem_idx < n {
                sum += w * x[elem_idx];
                elem_idx += 1;
            }
        }
    }
    sum
}

// ============================================================================
// Q4_K dequantization
// ============================================================================
// Q4_K super-block structure:
// - d, dmin: f16 scale and min
// - scales: 12 bytes, 6-bit scales for 8 sub-blocks of 32 elements each
// - qs: 128 bytes, each byte packs two 4-bit values

pub fn dequantize_q4_k(block: &BlockQ4K, out: &mut [f32; 256]) {
    let d = crate::f16_to_f32(block.d);
    let dmin = crate::f16_to_f32(block.dmin);

    for j in 0..8 {
        // Each sub-block is 32 elements, with 6-bit scale
        let scale_byte = j * 3 / 2;
        let scale = if j % 2 == 0 {
            (block.scales[scale_byte] & 0x3f) as f32
        } else {
            ((block.scales[scale_byte] >> 6) | ((block.scales[scale_byte + 1] as u32 & 0x0f) << 2) as u8) as f32
        }.min(63.0);

        let sub_d = d * scale;
        let sub_m = dmin * scale;

        for i in 0..32 {
            let idx = j * 32 + i;
            let byte_idx = j * 16 + i / 2;
            let q = if i % 2 == 0 {
                (block.qs[byte_idx] & 0x0f) as f32
            } else {
                (block.qs[byte_idx] >> 4) as f32
            };
            out[idx] = sub_d * q - sub_m;
        }
    }
}

pub fn vec_dot_q4_k_f32(blocks: &[BlockQ4K], x: &[f32], n_blocks: usize) -> f32 {
    let n = n_blocks * 256;
    let mut sum = 0.0f32;
    let mut elem_idx = 0usize;

    for b in 0..n_blocks {
        let d = crate::f16_to_f32(blocks[b].d);
        let dmin = crate::f16_to_f32(blocks[b].dmin);

        for j in 0..8 {
            let scale_byte = j * 3 / 2;
            let scale = if j % 2 == 0 {
                (blocks[b].scales[scale_byte] & 0x3f) as f32
            } else {
                ((blocks[b].scales[scale_byte] as u32 >> 6)
                    | (((blocks[b].scales.get(scale_byte + 1).copied().unwrap_or(0) as u32) & 0x0f) << 2)) as f32
            }.min(63.0);

            let sub_d = d * scale;
            let sub_m = dmin * scale;

            for i in 0..32 {
                let byte_idx = j * 16 + i / 2;
                let q = if i % 2 == 0 {
                    (blocks[b].qs[byte_idx] & 0x0f) as f32
                } else {
                    (blocks[b].qs[byte_idx] >> 4) as f32
                };
                let w = sub_d * q - sub_m;
                if elem_idx < n {
                    sum += w * x[elem_idx];
                    elem_idx += 1;
                }
            }
        }
    }
    sum
}

// ============================================================================
// IQ2_XXS dequantization
// ============================================================================
// IQ2_XXS: 256 elements in 66 bytes.
// d: f16 scale
// qs: 32 u16 values, each encoding 8 elements:
//   bits 0-7:   grid indices for elements 0-3 (2 bits each)
//   bits 8-14:  sign bits for elements 0-6 (1 bit each)
//   bits 15-22: grid indices for elements 4-7
//   bits 23-29: sign bits for elements 7 + next block
//   bit 30:     reserved
//   bit 31:     reserved
//
// The 256-element grid and 128 sign patterns are precomputed.

// Grid values (same as in ds4.c):
pub static IQ2XXS_GRID: &[u64; 256] = &[
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
];

// Sign masks
pub static KSIGNS_IQ2XS: &[u8; 128] = &[
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
];

pub static KMASK_IQ2XS: &[u8; 8] = &[1, 2, 4, 8, 16, 32, 64, 128];

// ============================================================================
// Precomputed SIMD tables: grid bytes and sign masks expanded to [i16; 8].
// Avoiding scalar-to-vector construction in the inner SIMD loop is the
// difference between a 4-8× speedup and a 30× slowdown.
// ============================================================================

#[cfg(target_arch = "x86_64")]
static IQ2XXS_GRID_I16: std::sync::OnceLock<[[i16; 8]; 256]> = std::sync::OnceLock::new();
#[cfg(target_arch = "x86_64")]
static KSIGNS_IQ2XS_I16: std::sync::OnceLock<[[i16; 8]; 128]> = std::sync::OnceLock::new();

#[cfg(target_arch = "x86_64")]
fn ensure_iq2xxs_simd_tables() {
    IQ2XXS_GRID_I16.get_or_init(|| {
        let mut table = [[0i16; 8]; 256];
        for (gi, entry) in table.iter_mut().enumerate() {
            let grid = IQ2XXS_GRID[gi];
            for (j, val) in entry.iter_mut().enumerate() {
                *val = ((grid >> (j * 8)) & 0xFF) as i8 as i16;
            }
        }
        table
    });
    KSIGNS_IQ2XS_I16.get_or_init(|| {
        let mut table = [[0i16; 8]; 128];
        for (si, entry) in table.iter_mut().enumerate() {
            let sbyte = KSIGNS_IQ2XS[si];
            for (j, val) in entry.iter_mut().enumerate() {
                *val = if (sbyte >> j) & 1 != 0 { -1 } else { 1 };
            }
        }
        table
    });
}

/// Get the grid byte at position j for grid index g.
#[inline]
pub fn iq2xxs_grid_byte(grid: u64, j: usize) -> i32 {
    ((grid >> (j * 8)) & 0xff) as i32
}

/// Get sign for element j from sign byte.
#[inline]
pub fn iq2xxs_sign(signs: u8, j: usize) -> i32 {
    if signs & KMASK_IQ2XS[j] != 0 { -1 } else { 1 }
}

/// Dequantize a single IQ2_XXS block to f32.
/// Matches the 8-group, 4-u16-per-group encoding used in vec_dot_iq2_xxs_q8_k.
pub fn dequantize_iq2_xxs(block: &BlockIq2Xxs, out: &mut [f32; 256]) {
    let d = crate::f16_to_f32(block.d);

    for g in 0..8 {
        let base = g * 4;
        let lo: u32 = (block.qs[base] as u32) | ((block.qs[base + 1] as u32) << 16);
        let hi: u32 = (block.qs[base + 2] as u32) | ((block.qs[base + 3] as u32) << 16);

        let gidx: [usize; 4] = [
            (lo & 0xff) as usize,
            ((lo >> 8) & 0xff) as usize,
            ((lo >> 16) & 0xff) as usize,
            ((lo >> 24) & 0xff) as usize,
        ];
        let sidx: [usize; 4] = [
            (hi & 0x7f) as usize,
            ((hi >> 7) & 0x7f) as usize,
            ((hi >> 14) & 0x7f) as usize,
            ((hi >> 21) & 0x7f) as usize,
        ];
        let extra = ((hi >> 28) & 0xf) as i32;
        let ls = 2 * extra + 1; // 1, 3, 5, ..., 31

        let elem_base = g * 32;
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
                let g0 = iq2xxs_grid_byte(grid0, j) as i32;
                let g1 = iq2xxs_grid_byte(grid1, j) as i32;
                let sign0 = if (sbyte0 >> j) & 1 != 0 { -1 } else { 1 };
                let sign1 = if (sbyte1 >> j) & 1 != 0 { -1 } else { 1 };
                // 0.125 factor matches vec_dot_iq2_xxs_q8_k's final scaling
                out[poff + j] = d * 0.125 * (g0 * sign0 * ls) as f32;
                out[poff + 8 + j] = d * 0.125 * (g1 * sign1 * ls) as f32;
            }
        }
    }
}

/// Dot product of IQ2_XXS blocks with Q8_K blocks.
/// Dispatches to AVX2 on x86_64, scalar fallback otherwise.
/// Matches C's ds4_vec_dot_iq2_xxs_q8_K exactly:
///   8 groups of 4 u16 = 32 elements each, 4 grid indices + 4 sign patterns per group,
///   group-level ls = 2*extra+1 (1 or 3), final 0.125 scaling.
///   d = IQ2_scale * Q8K_scale, dot product uses raw Q8 int8 values (q8.qs).
#[inline]
pub fn vec_dot_iq2_xxs_q8_k(blocks: &[BlockIq2Xxs], q8: &[BlockQ8K], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { return vec_dot_iq2_xxs_q8_k_avx2(blocks, q8, n); }
        }
    }
    vec_dot_iq2_xxs_q8_k_scalar(blocks, q8, n)
}

/// Scalar implementation of IQ2_XXS × Q8_K dot product.
fn vec_dot_iq2_xxs_q8_k_scalar(blocks: &[BlockIq2Xxs], q8: &[BlockQ8K], n: usize) -> f32 {
    let mut total = 0.0f64;

    for i in 0..n {
        let d = crate::f16_to_f32(blocks[i].d) as f64 * q8[i].d as f64;
        let qs = &blocks[i].qs;
        let q8_qs = &q8[i].qs;
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

        total += d * (bsum as f64);
    }

    (0.125 * total) as f32
}

/// AVX2-accelerated IQ2_XXS × Q8_K dot product.
/// Uses precomputed [i16; 8] tables to load grid bytes and sign masks
/// directly into SSE registers, avoiding expensive scalar-to-vector
/// construction that made the previous approach 30× slower than scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_iq2_xxs_q8_k_avx2(blocks: &[BlockIq2Xxs], q8: &[BlockQ8K], n: usize) -> f32 {
    // Init precomputed tables once (lazily, thread-safe)
    ensure_iq2xxs_simd_tables();
    let grid_tab = IQ2XXS_GRID_I16.get().unwrap_unchecked();
    let sign_tab = KSIGNS_IQ2XS_I16.get().unwrap_unchecked();

    let mut total = 0.0f64;

    for i in 0..n {
        let d = crate::f16_to_f32(blocks[i].d) as f64 * q8[i].d as f64;
        let qs = &blocks[i].qs;
        let q8_qs = &q8[i].qs;
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
                let poff = elem_base + pair * 16;

                group_sum += simd_dot_iq2_8_fast(
                    grid_tab[gi0].as_ptr(),
                    sign_tab[si0].as_ptr(),
                    q8_qs.as_ptr().add(poff),
                ) as i64;

                group_sum += simd_dot_iq2_8_fast(
                    grid_tab[gi1].as_ptr(),
                    sign_tab[si1].as_ptr(),
                    q8_qs.as_ptr().add(poff + 8),
                ) as i64;
            }

            bsum += group_sum * ls;
        }

        total += d * (bsum as f64);
    }

    (0.125 * total) as f32
}

/// AVX2 inner kernel: dot product of 8 grid bytes × 8 sign bits × 8 q8 values.
/// grid_ptr: pointer to 8 precomputed i16 grid values (one per element).
/// sign_ptr: pointer to 8 precomputed i16 sign values (±1, one per element).
/// q8_ptr: pointer to 8 consecutive i8 Q8_K activation values.
///
/// Uses SSE _mm_madd_epi16 for one-instruction multiply-accumulate,
/// then horizontal add to produce the final i32 sum.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn simd_dot_iq2_8_fast(grid_ptr: *const i16, sign_ptr: *const i16, q8_ptr: *const i8) -> i32 {
    // Load precomputed grid (8 × i16) and signs (8 × i16)
    let grid_i16 = _mm_loadu_si128(grid_ptr as *const __m128i);
    let sign_i16 = _mm_loadu_si128(sign_ptr as *const __m128i);

    // grid[i] × sign[i] → 8 × i16
    let grid_signed = _mm_mullo_epi16(grid_i16, sign_i16);

    // Load 8 q8 values into a 16-byte aligned buffer, extend i8→i16
    #[repr(align(16))]
    struct Align16([i8; 16]);
    let mut q8_buf = Align16([0i8; 16]);
    std::ptr::copy_nonoverlapping(q8_ptr, q8_buf.0.as_mut_ptr(), 8);
    let q8_reg = _mm_loadu_si128(q8_buf.0.as_ptr() as *const __m128i);
    let q8_i16 = _mm_cvtepi8_epi16(q8_reg);

    // Dot product: _mm_madd_epi16 does adjacent-pair multiply-accumulate
    // mul = [g0*q0+g1*q1, g2*q2+g3*q3, g4*q4+g5*q5, g6*q6+g7*q7]
    let mul = _mm_madd_epi16(grid_signed, q8_i16);

    // Horizontal sum: 4 × i32 → 1 × i32
    let hadd = _mm_hadd_epi32(mul, mul);
    let hadd2 = _mm_hadd_epi32(hadd, hadd);
    _mm_cvtsi128_si32(hadd2)
}

// ============================================================================
// Q8_K dequantization (for dequantizing temporary Q8 activations)
// ============================================================================

pub fn dequantize_q8_k(block: &BlockQ8K, out: &mut [f32; 256]) {
    let d = block.d;
    for i in 0..256 {
        out[i] = d * (block.qs[i] as f32);
    }
}

/// Quantize f32 vector to Q8_K format.
/// Matches C's ds4_quantize_row_q8_K exactly: amax = max(|x|), d = amax/127, id = 127/amax.
pub fn quantize_q8_k(x: &[f32], n: usize, out: &mut [BlockQ8K]) {
    let n_blocks = (n + 255) / 256;
    for b in 0..n_blocks {
        let start = b * 256;
        let end = (start + 256).min(n);
        let len = end - start;

        // Find max absolute value (matches C: float max = 0; ax = fabsf(x[j]); if (ax > max) max = ax)
        let mut amax = 0.0f32;
        for i in start..end {
            let ax = x[i].abs();
            if ax > amax { amax = ax; }
        }

        if amax == 0.0f32 {
            out[b].d = 0.0;
            for i in 0..len { out[b].qs[i] = 0; }
            for s in 0..16 { out[b].bsums[s] = 0; }
            continue;
        }

        // d = amax / 127, id = 127 / amax (matches C exactly)
        let d = amax / 127.0f32;
        let id = 127.0f32 / amax;
        for i in 0..len {
            let v = (id * x[start + i]).round() as i32;
            let v = v.clamp(-128, 127);
            out[b].qs[i] = v as i8;
        }

        // Compute block sums
        for s in 0..16 {
            let mut sum = 0i32;
            let base = s * 16;
            for i in base..(base + 16).min(len) {
                sum += out[b].qs[i] as i32;
            }
            out[b].bsums[s] = sum as i16;
        }

        out[b].d = d; // = amax / 127.0
    }
}

// ============================================================================
// Matvec: vector-matrix multiplication with quantized weights
// ============================================================================

/// Matrix-vector multiply: out = x @ W^T where W is F16.
pub fn matvec_f16(out: &mut [f32], x: &[f32], weight: &[u8], in_dim: usize, out_dim: usize) {
    for o in 0..out_dim {
        let mut sum = 0.0f32;
        let base = o * in_dim * 2; // 2 bytes per f16
        for i in 0..in_dim {
            let h = u16::from_le_bytes([weight[base + i * 2], weight[base + i * 2 + 1]]);
            sum += x[i] * crate::f16_to_f32(h);
        }
        out[o] = sum;
    }
}
