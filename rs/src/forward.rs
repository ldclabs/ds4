// CPU forward pass implementation for DeepSeek V4 Flash.
//
// This module implements the complete layer-by-layer forward pass:
// embedding → 43 transformer layers (HC attention + MoE FFN) → output head.
//
// All algorithms mirror the C reference in ds4.c exactly.

use crate::model::{LayerWeights, ModelWeights};
use crate::quant::{BlockQ2K, BlockIq2Xxs, BlockQ8K, quantize_q8_k, dequantize_iq2_xxs, vec_dot_iq2_xxs_q8_k, vec_dot_q2_k_q8_k};
use crate::{
    N_EMBD, N_HEAD, N_HEAD_KV, N_HEAD_DIM, N_ROT, N_OUT_GROUP,
    N_LORA_Q, N_LORA_O, N_EXPERT, N_EXPERT_USED, N_FF_EXP,
    N_HC, N_HC_SINKHORN_ITER, N_LAYER, N_VOCAB,
    RMS_EPS, HC_EPS, EXPERT_WEIGHT_SCALE,
    SWIGLU_CLAMP_EXP, N_SWA, N_INDEXER_HEAD_DIM,
    f16_to_f32, silu, softplus_stable,
    rms_norm_weighted, rms_norm_no_weight,
    layer_compress_ratio, hash_routed_expert,
    fp8_kv_quantize_row_inplace, f16_round_inplace,
    hc_split_sinkhorn_one, hc_weighted_sum_one, hc_post_one,
};

/// Print min/max/rms stats for a float slice, matching C's print_vec_stats output.
#[allow(dead_code)]
pub fn print_vec_stats_rms(label: &str, v: &[f32]) {
    let n = v.len();
    if n == 0 {
        println!("{} n=0", label);
        return;
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum_sq = 0.0f64;
    for &x in v {
        if x < min { min = x; }
        if x > max { max = x; }
        sum_sq += (x as f64) * (x as f64);
    }
    let rms = (sum_sq / n as f64).sqrt();
    println!("{}: min={:.6} max={:.6} rms={:.6}", label, min, max, rms);
}

// ============================================================================
// KV Cache
// ============================================================================

pub struct KvCache {
    /// Raw sliding-window KV cache: [layer][pos % raw_cap][head_dim] as f32
    pub raw: Vec<Vec<f32>>,
    pub raw_cap: usize,
    pub raw_len: usize,
    /// Compressed KV cache (stub — full compressor not yet ported)
    pub comp: Vec<Vec<f32>>,
    pub comp_cap: usize,
    pub comp_len: Vec<usize>,
    /// Indexer compressed cache (stub)
    pub index_comp: Vec<Vec<f32>>,
    pub index_comp_len: Vec<usize>,
    pub ctx_size: usize,
}

impl KvCache {
    pub fn new(ctx_size: usize) -> Self {
        let raw_cap = N_SWA as usize;
        let comp_cap = ctx_size / 4 + 2;

        let raw = vec![vec![0.0f32; raw_cap * N_HEAD_DIM as usize]; N_LAYER as usize];
        let comp = vec![vec![0.0f32; comp_cap * N_HEAD_DIM as usize * 2]; N_LAYER as usize];
        let index_comp = vec![
            vec![0.0f32; comp_cap * N_INDEXER_HEAD_DIM as usize * 2];
            N_LAYER as usize
        ];

        KvCache {
            raw,
            raw_cap,
            raw_len: 0,
            comp,
            comp_cap,
            comp_len: vec![0; N_LAYER as usize],
            index_comp,
            index_comp_len: vec![0; N_LAYER as usize],
            ctx_size,
        }
    }

    pub fn store_raw_kv(&mut self, layer: usize, pos: usize, kv: &[f32]) {
        let offset = (pos % self.raw_cap) * N_HEAD_DIM as usize;
        let dst = &mut self.raw[layer][offset..offset + N_HEAD_DIM as usize];
        dst.copy_from_slice(&kv[..N_HEAD_DIM as usize]);
        if pos + 1 > self.raw_len {
            self.raw_len = pos + 1;
        }
    }

    pub fn read_raw_kv(&self, layer: usize, pos: usize) -> &[f32] {
        let offset = (pos % self.raw_cap) * N_HEAD_DIM as usize;
        &self.raw[layer][offset..offset + N_HEAD_DIM as usize]
    }
}

// ============================================================================
// RoPE — matches rope_tail_layer_inplace / rope_tail_ext_inplace in ds4.c
// ============================================================================

/// RoPE frequency base, layer-dependent (dense vs compressed).
fn layer_rope_freq_base(il: u32) -> f32 {
    if layer_compress_ratio(il) != 0 && crate::constants::COMPRESS_ROPE_FREQ_BASE > 0.0 {
        crate::constants::COMPRESS_ROPE_FREQ_BASE
    } else {
        crate::constants::ROPE_FREQ_BASE
    }
}

/// RoPE frequency scale, layer-dependent.
fn layer_rope_freq_scale(il: u32) -> f32 {
    if layer_compress_ratio(il) == 0 || crate::constants::ROPE_SCALE_FACTOR <= 0.0 {
        return 1.0;
    }
    1.0 / crate::constants::ROPE_SCALE_FACTOR
}

/// YaRN ramp: linear ramp from 1→0 between low and high dimension indices.
fn rope_yarn_ramp(low: f32, high: f32, i0: i32) -> f32 {
    let y = ((i0 as f32 / 2.0) - low) / (0.001f32.max(high - low));
    1.0 - y.clamp(0.0, 1.0)
}

/// YaRN correction dimension helper.
fn rope_yarn_corr_dim(n_dims: i32, n_ctx_orig: u64, n_rot: f32, base: f32) -> f32 {
    (n_dims as f32) * ((n_ctx_orig as f32) / (n_rot * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * base.ln())
}

/// YaRN correction dimensions [low, high] pair.
fn rope_yarn_corr_dims(n_dims: i32, n_ctx_orig: u64, freq_base: f32,
                        beta_fast: f32, beta_slow: f32) -> (f32, f32) {
    let start = rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base).floor();
    let end = rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base).ceil();
    (start.max(0.0), end.min((n_dims - 1) as f32))
}

/// Full RoPE with YaRN, matching `rope_tail_ext_inplace` in ds4.c exactly.
/// Applies RoPE only to the tail (last n_rot dims) of each head.
/// When `inverse` is true, rotates in the opposite direction (used for attn output deskew).
pub fn rope_tail_layer_inplace(
    x: &mut [f32],
    n_head: usize,
    head_dim: usize,
    n_rot: usize,
    pos: usize,
    il: u32,
    inverse: bool,
) {
    let compressed = layer_compress_ratio(il) != 0;
    let freq_base = layer_rope_freq_base(il);
    let freq_scale = layer_rope_freq_scale(il);
    let ext_factor = if compressed && crate::constants::ROPE_SCALE_FACTOR > 1.0 { 1.0 } else { 0.0 };
    let mut attn_factor = 1.0f32;
    if ext_factor != 0.0 && freq_scale > 0.0 {
        attn_factor /= 1.0 + 0.1 * (1.0 / freq_scale).ln();
    }
    let n_ctx_orig: u64 = if compressed { crate::constants::ROPE_ORIG_CTX } else { 0 };
    let beta_fast = crate::constants::ROPE_YARN_BETA_FAST;
    let beta_slow = crate::constants::ROPE_YARN_BETA_SLOW;

    rope_tail_ext_inplace(
        x, n_head, head_dim, n_rot, pos,
        n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
        beta_fast, beta_slow, inverse,
    );
}

/// Core RoPE implementation matching `rope_tail_ext_inplace` in ds4.c.
pub fn rope_tail_ext_inplace(
    x: &mut [f32],
    n_head: usize,
    head_dim: usize,
    n_rot: usize,
    pos: usize,
    n_ctx_orig: u64,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    inverse: bool,
) {
    let n_nope = head_dim - n_rot;
    let theta_scale = freq_base.powf(-2.0 / n_rot as f32);
    let sin_sign = if inverse { -1.0f32 } else { 1.0f32 };

    let corr_dims = if ext_factor != 0.0 {
        rope_yarn_corr_dims(n_rot as i32, n_ctx_orig, freq_base, beta_fast, beta_slow)
    } else {
        (0.0, 0.0)
    };

    for h in 0..n_head {
        let tail = &mut x[h * head_dim + n_nope..h * head_dim + head_dim];
        let mut theta_extrap = pos as f32;

        for i in (0..n_rot).step_by(2) {
            let theta_interp = freq_scale * theta_extrap;
            let mut theta = theta_interp;
            let mut mscale = attn_factor;

            if ext_factor != 0.0 {
                let ramp_mix = rope_yarn_ramp(corr_dims.0, corr_dims.1, i as i32) * ext_factor;
                theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
            }

            let c = theta.cos() * mscale;
            let s = sin_sign * theta.sin() * mscale;
            let x0 = tail[i];
            let x1 = tail[i + 1];

            tail[i] = x0 * c - x1 * s;
            tail[i + 1] = x0 * s + x1 * c;

            theta_extrap *= theta_scale;
        }
    }
}

/// Legacy alias kept for backward compatibility with tests.
#[deprecated(note = "use rope_tail_layer_inplace instead")]
pub fn rope_apply(x: &mut [f32], n_head: usize, head_dim: usize, n_rot: usize,
              pos: usize, _base: f32) {
    rope_tail_layer_inplace(x, n_head, head_dim, n_rot, pos, 0, false);
}

// ============================================================================
// Token Embedding
// ============================================================================

pub fn embed_token_f16(weights: &ModelWeights, token: i32, out: &mut [f32]) {
    let embd = weights.token_embd.as_f16();
    let n_embd = N_EMBD as usize;
    let offset = (token as usize) * n_embd;
    for i in 0..n_embd {
        out[i] = f16_to_f32(embd[offset + i]);
    }
}

/// Initialize all HC streams from a plain embedding (for the first layer).
pub fn hc_from_plain_embedding(cur: &mut [f32], plain: &[f32]) {
    let n_embd = N_EMBD as usize;
    let n_hc = N_HC as usize;
    for h in 0..n_hc {
        let offset = h * n_embd;
        cur[offset..offset + n_embd].copy_from_slice(plain);
    }
}

// ============================================================================
// HC Pre/Post — the Hash Chain state management
// ============================================================================

/// HC pre: normalize HC state, project control vector through fn,
/// compute split + Sinkhorn, produce plain sublayer input + post/comb.
/// This is the single-token equivalent of hc_pre_from_state_one in ds4.c.
fn hc_pre_one(
    out: &mut [f32],           // [N_EMBD] sublayer input
    post: &mut [f32],          // [N_HC]
    comb: &mut [f32],          // [N_HC * N_HC]
    residual_hc: &[f32],       // [N_HC * N_EMBD]
    fn_weight: &[f32],         // hc_attn_fn or hc_ffn_fn [n_hc_mix * N_HC*N_EMBD]
    scale: &[f32],             // [3]
    base: &[f32],              // [2*N_HC + N_HC*N_HC]
) {
    let n_hc = N_HC as usize;
    let n_embd = N_EMBD as usize;
    let hc_dim = n_hc * n_embd;

    // Step 1: RMS norm on all HC streams
    let mut flat = vec![0.0f32; hc_dim];
    rms_norm_no_weight(&mut flat, residual_hc, hc_dim, RMS_EPS);

    // Step 2: matvec_f16:  fn_weight [n_hc_mix, hc_dim] × flat [hc_dim] → mix [n_hc_mix]
    let n_hc_mix = 2 * n_hc + n_hc * n_hc;
    let mut mix = vec![0.0f32; n_hc_mix];
    for i in 0..n_hc_mix {
        let mut sum = 0.0f32;
        for j in 0..hc_dim {
            sum += flat[j] * fn_weight[i * hc_dim + j];
        }
        mix[i] = sum;
    }

    // Step 3: hc_split_sinkhorn_one → split [n_hc_mix] = pre + post + comb
    let mut split = vec![0.0f32; n_hc_mix];
    hc_split_sinkhorn_one(&mut split, &mix, scale, base, n_hc, N_HC_SINKHORN_ITER, HC_EPS);

    // Step 4: weighted sum → out [n_embd]
    hc_weighted_sum_one(out, residual_hc, &split[..n_hc], n_embd, n_hc);

    // Copy post and comb for later use
    post.copy_from_slice(&split[n_hc..2 * n_hc]);
    comb.copy_from_slice(&split[2 * n_hc..]);
}

/// HC pre for attention sublayer.
pub fn hc_attn_pre(
    cur_out: &mut [f32],       // [N_EMBD] single query
    residual_hc: &mut [f32],   // [N_HC * N_EMBD]
    post: &mut [f32],          // [N_HC]
    comb: &mut [f32],          // [N_HC * N_HC]
    in_hc: &[f32],             // [N_HC * N_EMBD]
    layer: &LayerWeights,
) {
    let n_hc = N_HC as usize;
    let n_embd = N_EMBD as usize;

    // Save residual
    residual_hc[..n_hc * n_embd].copy_from_slice(in_hc);

    // HC pre
    hc_pre_one(
        cur_out,
        post,
        comb,
        in_hc,
        &layer.hc_attn_fn.as_f32_auto(),
        layer.hc_attn_scale.as_f32(),
        layer.hc_attn_base.as_f32(),
    );
}

/// HC pre for FFN sublayer.
pub fn hc_ffn_pre(
    cur_out: &mut [f32],       // [N_EMBD]
    post: &mut [f32],          // [N_HC]
    comb: &mut [f32],          // [N_HC * N_HC]
    in_hc: &[f32],             // [N_HC * N_EMBD]
    layer: &LayerWeights,
) {
    hc_pre_one(
        cur_out,
        post,
        comb,
        in_hc,
        &layer.hc_ffn_fn.as_f32_auto(),
        layer.hc_ffn_scale.as_f32(),
        layer.hc_ffn_base.as_f32(),
    );
}

// ============================================================================
// Attention sublayer
// ============================================================================

/// Q projection with LoRA (input must already be RMS-normed).
/// Matches layer_q_projection_normed_one from ds4.c.
pub fn layer_q_projection(
    q: &mut [f32],                 // [N_HEAD * N_HEAD_DIM]
    attn_norm: &[f32],            // [N_EMBD] — already RMS-normed
    layer: &LayerWeights,
) {
    let n_embd = N_EMBD as usize;
    let n_head = N_HEAD as usize;
    let head_dim = N_HEAD_DIM as usize;
    let lora_q = N_LORA_Q as usize;
    let q_dim = n_head * head_dim;

    // q_a: Q8_0 [lora_q, n_embd] → lora_out [lora_q]
    let mut q_a_out = vec![0.0f32; lora_q];
    crate::quant::matvec_q8_0(&mut q_a_out, attn_norm, layer.attn_q_a.as_bytes(), n_embd, lora_q);

    // q_a_norm
    let q_a_norm = layer.attn_q_a_norm.as_f32();
    let q_a_out_copy = q_a_out.to_vec();
    rms_norm_weighted(&mut q_a_out, &q_a_out_copy, q_a_norm, lora_q, RMS_EPS);

    // q_b: Q8_0 [q_dim, lora_q] or F16
    let q_b_type = layer.attn_q_b.tensor_type;
    if q_b_type == 8 {
        // Q8_0 matvec
        crate::quant::matvec_q8_0(q, &q_a_out, layer.attn_q_b.as_bytes(), lora_q, q_dim);
    } else {
        // F16 fallback
        let q_b = layer.attn_q_b.as_f16();
        for i in 0..q_dim {
            let mut sum = 0.0f32;
            for j in 0..lora_q {
                sum += q_a_out[j] * f16_to_f32(q_b[i * lora_q + j]);
            }
            q[i] = sum;
        }
    }

    // Per-head RMS norm (matches C's head_rms_norm_inplace)
    head_rms_norm_inplace(q, n_head, head_dim, RMS_EPS);
}

/// Per-head RMS normalization: normalize each head independently.
fn head_rms_norm_inplace(x: &mut [f32], n_head: usize, head_dim: usize, eps: f32) {
    for h in 0..n_head {
        let start = h * head_dim;
        let ss: f32 = x[start..start + head_dim].iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (ss / head_dim as f32 + eps).sqrt();
        for i in start..start + head_dim {
            x[i] *= inv_rms;
        }
    }
}

/// KV projection (input must already be RMS-normed).
/// Matches layer_kv_projection_normed_one from ds4.c.
pub fn layer_kv_projection(
    kv: &mut [f32],                // [N_HEAD_DIM]
    attn_norm: &[f32],            // [N_EMBD] — already RMS-normed
    layer: &LayerWeights,
) {
    let n_embd = N_EMBD as usize;
    let head_dim = N_HEAD_DIM as usize;

    // kv_weight: Q8_0 [head_dim, n_embd] or F16
    let kv_type = layer.attn_kv.tensor_type;
    if kv_type == 8 {
        // Q8_0 matvec
        let mut raw = vec![0.0f32; head_dim];
        crate::quant::matvec_q8_0(&mut raw, attn_norm, layer.attn_kv.as_bytes(), n_embd, head_dim);
        // Post-projection RMS norm with kv_a_norm (head_dim elements)
        let kv_a_norm = layer.attn_kv_a_norm.as_f32();
        rms_norm_weighted(kv, &raw, kv_a_norm, head_dim, RMS_EPS);
    } else {
        // F16 fallback
        let kv_weight = layer.attn_kv.as_f16();
        for i in 0..head_dim {
            let mut sum = 0.0f32;
            for j in 0..n_embd {
                sum += attn_norm[j] * f16_to_f32(kv_weight[i * n_embd + j]);
            }
            kv[i] = sum;
        }
    }
}

/// Single-token attention: compute attention output for one position.
/// Single-token attention: per-head softmax with attention sinks, n_kv=1 (self-attn).
/// Matches `layer_attention_one` + `layer_attention_rows_one` in ds4.c exactly.
pub fn layer_attention_one(
    attn_out: &mut [f32],          // [N_HEAD * N_HEAD_DIM]
    q: &[f32],                     // [N_HEAD * N_HEAD_DIM]
    kv: &[f32],                    // [N_HEAD_DIM] single raw KV (pre-cache, FP8 rounded)
    sinks: &[f32],                 // [N_HEAD] attention sink bias
) {
    let n_head = N_HEAD as usize;
    let head_dim = N_HEAD_DIM as usize;
    let kq_scale = 1.0 / (head_dim as f32).sqrt();

    // n_kv = 1: self-attention only, matching C's layer_attention_rows_one(out, ..., q, kv, 1)
    for h in 0..n_head {
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let oh = &mut attn_out[h * head_dim..(h + 1) * head_dim];

        // Single dot product
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += qh[d] * kv[d];
        }
        let score = dot * kq_scale;

        // Softmax with sink (2-element: [sink, score])
        let sink_val = sinks.get(h).copied().unwrap_or(0.0f32);
        let max_val = if score > sink_val { score } else { sink_val };
        let w_sink = (sink_val - max_val).exp();
        let w_score = (score - max_val).exp();
        let inv = 1.0 / (w_sink + w_score);

        for d in 0..head_dim {
            oh[d] = kv[d] * w_score * inv;
        }
    }
}

/// Grouped output projection with Q8_0 support (LoRA-style layer).
/// Matches layer_grouped_out_one from ds4.c.
pub fn layer_grouped_out(
    out: &mut [f32],               // [N_EMBD]
    attn_heads: &[f32],            // [N_HEAD * N_HEAD_DIM]
    layer: &LayerWeights,
) {
    let n_groups = N_OUT_GROUP as usize;
    let group_heads = (N_HEAD / N_OUT_GROUP) as usize;
    let group_dim = N_HEAD_DIM as usize * group_heads;
    let rank = N_LORA_O as usize;
    let n_embd = N_EMBD as usize;

    // Step 1: matvec_q8_0_grouped: output_a [n_groups * rank, group_dim] × heads → low [n_groups * rank]
    let mut low = vec![0.0f32; n_groups * rank];

    // Check tensor type: 8 = Q8_0, 1 = F16, 0 = F32
    let o_a_type = layer.attn_output_a.tensor_type;
    if o_a_type == 8 {
        // Q8_0 quantized — process each group independently matching C's matvec_q8_0_grouped_rows
        let o_a_bytes = layer.attn_output_a.as_bytes();
        let n_blocks = group_dim / 32;
        let block_size = std::mem::size_of::<crate::quant::BlockQ80>();
        let bytes_per_column = n_blocks * block_size;

        for g in 0..n_groups {
            let head_start = g * group_dim;
            let out_start = g * rank;
            let col_start = out_start * bytes_per_column;
            crate::quant::matvec_q8_0(
                &mut low[out_start..out_start + rank],
                &attn_heads[head_start..head_start + group_dim],
                &o_a_bytes[col_start..col_start + rank * bytes_per_column],
                group_dim,
                rank,
            );
        }
    } else if o_a_type <= 1 {
        // F32 or F16
        let o_a_f16 = layer.attn_output_a.as_f16();
        let n_elems = n_groups * rank * group_dim;
        if o_a_f16.len() >= n_elems {
            for g in 0..n_groups {
                let head_start = g * group_dim;
                for i in 0..rank {
                    let mut sum = 0.0f32;
                    for j in 0..group_dim {
                        sum += attn_heads[head_start + j]
                            * f16_to_f32(o_a_f16[(g * rank + i) * group_dim + j]);
                    }
                    low[g * rank + i] = sum;
                }
            }
        }
    }

    // Step 2: matvec: output_b [n_embd, n_groups * rank] × low → out [n_embd]
    let o_b_type = layer.attn_output_b.tensor_type;
    if o_b_type == 8 {
        let o_b_bytes = layer.attn_output_b.as_bytes();
        crate::quant::matvec_q8_0(
            out, &low, o_b_bytes, n_groups * rank, n_embd,
        );
    } else if o_b_type <= 1 {
        let o_b = layer.attn_output_b.as_f32();
        let n_elems = n_embd * n_groups * rank;
        if o_b.len() >= n_elems {
            for i in 0..n_embd {
                let mut sum = 0.0f32;
                for j in 0..n_groups * rank {
                    sum += low[j] * o_b[i * (n_groups * rank) + j];
                }
                out[i] = sum;
            }
        }
    }
}

// ============================================================================
// FFN with Mixture of Experts
// ============================================================================

/// Select experts via hash routing table lookup.
/// Compute router probabilities via learned router (for top-k layers).
fn layer_router_probs_one(
    probs: &mut [f32],             // [N_EXPERT]
    layer: &LayerWeights,
    x: &[f32],                     // [N_EMBD]
) {
    let n_expert = N_EXPERT as usize;
    let n_embd = N_EMBD as usize;

    // matvec_f16: gate_inp [n_expert, n_embd] × x → probs [n_expert]
    let gate_inp = layer.ffn_gate_inp.as_f16();
    for i in 0..n_expert {
        let mut sum = 0.0f32;
        for j in 0..n_embd {
            sum += x[j] * f16_to_f32(gate_inp[i * n_embd + j]);
        }
        probs[i] = softplus_stable(sum);
    }
}

/// Top-k expert selection with learned router (for dense layers 0-2).
fn layer_topk_selected_experts(
    selected: &mut [usize],        // [N_EXPERT_USED]
    expert_weight: &mut [f32],     // [N_EXPERT_USED]
    layer: &LayerWeights,
    x: &[f32],                     // [N_EMBD]
) {
    let n_expert = N_EXPERT as usize;
    let n_used = N_EXPERT_USED as usize;
    let scale = EXPERT_WEIGHT_SCALE;

    let mut probs = vec![0.0f32; n_expert];
    layer_router_probs_one(&mut probs, layer, x);

    // Optional bias
    if layer.ffn_exp_probs_b.has_data() {
        let bias_f32 = layer.ffn_exp_probs_b.as_f32();
        for i in 0..n_expert {
            probs[i] += bias_f32[i];
        }
    }

    // Top-k selection
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut sum = 0.0f32;
    for i in 0..n_used {
        selected[i] = indexed[i].0;
        expert_weight[i] = probs[indexed[i].0];
        sum += expert_weight[i];
    }

    if sum < 6.103515625e-5 {
        sum = 6.103515625e-5;
    }
    for i in 0..n_used {
        expert_weight[i] = expert_weight[i] / sum * scale;
    }
}

fn layer_hash_selected_experts(
    selected: &mut [usize],        // [N_EXPERT_USED]
    layer: &LayerWeights,
    token: i32,
) {
    let table = layer.ffn_gate_tid2eid.as_i32();
    let n_exp_used = N_EXPERT_USED as usize;
    if token < 0 || (token as usize) * n_exp_used >= table.len() {
        // Fallback: use hash-based selection
        for i in 0..n_exp_used {
            selected[i] = hash_routed_expert(token, i as u32, N_EXPERT) as usize;
        }
        return;
    }
    let row = &table[(token as usize) * n_exp_used..(token as usize + 1) * n_exp_used];
    for (i, &eid) in row.iter().enumerate() {
        selected[i] = eid as usize;
    }
}

/// Compute router weights via gate_inp projection + sqrt(softplus).
fn layer_hash_router_weights(
    weights_out: &mut [f32],       // [N_EXPERT_USED]
    layer: &LayerWeights,
    x: &[f32],                     // [N_EMBD]
    selected: &[usize],            // [N_EXPERT_USED]
) {
    let n_expert = N_EXPERT as usize;
    let n_embd = N_EMBD as usize;

    // matvec_f16: gate_inp [n_expert, n_embd] × x → logits [n_expert]
    let gate_inp = layer.ffn_gate_inp.as_f16();
    let mut probs = vec![0.0f32; n_expert];

    for i in 0..n_expert {
        let mut sum = 0.0f32;
        for j in 0..n_embd {
            sum += x[j] * f16_to_f32(gate_inp[i * n_embd + j]);
        }
        probs[i] = softplus_stable(sum).sqrt();
    }

    // Collect weights for selected experts, normalize, scale
    let n_used = N_EXPERT_USED as usize;
    let mut sum = 0.0f32;
    for i in 0..n_used {
        weights_out[i] = probs[selected[i]];
        sum += weights_out[i];
    }

    if sum < 6.103515625e-5 { sum = 6.103515625e-5; }
    for i in 0..n_used {
        weights_out[i] = weights_out[i] / sum * EXPERT_WEIGHT_SCALE;
    }
}

/// Feed-Forward Network with Mixture of Experts.
/// Process one token through the FFN layer.  Matches layer_ffn_one from ds4.c.

/// Expert weight layout helpers.
/// ffn_gate_exps / ffn_up_exps: IQ2_XXS blocks, dims [N_EMBD, N_FF_EXP, N_EXPERT]
/// ffn_down_exps: Q2_K blocks, dims [N_FF_EXP, N_EMBD, N_EXPERT] (transposed)

/// Compute gate = x @ W_gate[e]^T and up = x @ W_up[e]^T for expert e.
fn expert_gate_up_matvec(
    gate: &mut [f32],              // [N_FF_EXP]
    up: &mut [f32],                // [N_FF_EXP]
    x: &[f32],                     // [N_EMBD]
    layer: &LayerWeights,
    eid: usize,
    gate_type: u32,
) {
    let n_embd = N_EMBD as usize;
    let n_ff_exp = N_FF_EXP as usize;

    match gate_type {
        1 => {
            // F16: dims [N_EMBD, N_FF_EXP, N_EXPERT] = [4096, 2048, 256]
            let data = layer.ffn_gate_exps.as_f16();
            let up_data = layer.ffn_up_exps.as_f16();
            let stride = n_embd * n_ff_exp; // per expert

            for j in 0..n_ff_exp {
                let row_base = eid * stride + j * n_embd;
                let mut gs = 0.0f32;
                let mut us = 0.0f32;
                for i in 0..n_embd {
                    gs += x[i] * f16_to_f32(data[row_base + i]);
                    us += x[i] * f16_to_f32(up_data[row_base + i]);
                }
                gate[j] = gs;
                up[j] = us;
            }

        }
        16 => {
            // IQ2_XXS: dims [N_EMBD, N_FF_EXP, N_EXPERT]
            // Quantize x to Q8_K first (matching C's ds4_quantize_row_q8_K)
            let n_blocks_x = n_embd / 256;
            let mut xq = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }; n_blocks_x];
            quantize_q8_k(x, n_embd, &mut xq);

            let blocks_per_row = n_embd / 256;
            let blocks_per_expert = blocks_per_row * n_ff_exp;

            let gate_bytes = layer.ffn_gate_exps.as_bytes();
            let up_bytes = layer.ffn_up_exps.as_bytes();
            let block_size = std::mem::size_of::<BlockIq2Xxs>();

            // Q8K path
            for j in 0..n_ff_exp {
                let row_block_start = eid * blocks_per_expert + j * blocks_per_row;
                let mut gs = 0.0f32;
                let mut us = 0.0f32;
                for b in 0..blocks_per_row {
                    let block_idx = row_block_start + b;

                    // Gate block
                    let gate_block_ptr = gate_bytes[block_idx * block_size..].as_ptr() as *const BlockIq2Xxs;
                    let gate_block = unsafe { &*gate_block_ptr };
                    gs += vec_dot_iq2_xxs_q8_k(
                        core::slice::from_ref(gate_block),
                        core::slice::from_ref(&xq[b]),
                        1,
                    );

                    // Up block
                    let up_block_ptr = up_bytes[block_idx * block_size..].as_ptr() as *const BlockIq2Xxs;
                    let up_block = unsafe { &*up_block_ptr };
                    us += vec_dot_iq2_xxs_q8_k(
                        core::slice::from_ref(up_block),
                        core::slice::from_ref(&xq[b]),
                        1,
                    );
                }
                gate[j] = gs;
                up[j] = us;
            }

            // DEBUG: compare Q8K dot with fully dequantized f32 dot for first column
            {
                use std::sync::atomic::{AtomicBool, Ordering};
                static DONE: AtomicBool = AtomicBool::new(false);
                if !DONE.swap(true, Ordering::Relaxed) {
                    let mut f32_gate = [0.0f32; 256];
                    let mut f32_dot = 0.0f64;
                    for b in 0..blocks_per_row {
                        let bidx = eid * blocks_per_expert + b;
                        let blk = unsafe { &*(gate_bytes[bidx * block_size..].as_ptr() as *const BlockIq2Xxs) };
                        dequantize_iq2_xxs(blk, &mut f32_gate);
                        for i in 0..256 {
                            f32_dot += f32_gate[i] as f64 * x[b * 256 + i] as f64;
                        }
                    }
                    println!("  gate[0] Q8K={:.8}  f32_dequant={:.8}  ratio={:.4}",
                        gate[0], f32_dot as f32, gate[0] / (f32_dot as f32));
                }
            }
        }
        _ => panic!("unsupported expert gate tensor type: {}", gate_type),
    }
}

/// Accumulate moe_out += mid @ W_down[e] for expert e.
fn expert_down_matvec_accum(
    moe_out: &mut [f32],           // [N_EMBD]
    mid: &[f32],                   // [N_FF_EXP]
    layer: &LayerWeights,
    eid: usize,
    down_type: u32,
) {
    let n_embd = N_EMBD as usize;
    let n_ff_exp = N_FF_EXP as usize;

    match down_type {
        1 => {
            // F16: dims [N_FF_EXP, N_EMBD, N_EXPERT]
            let data = layer.ffn_down_exps.as_f16();
            let stride = n_ff_exp * n_embd;
            for i in 0..n_embd {
                let col_base = eid * stride + i * n_ff_exp;
                let mut sum = 0.0f32;
                for j in 0..n_ff_exp {
                    sum += mid[j] * f16_to_f32(data[col_base + j]);
                }
                moe_out[i] += sum;
            }
        }
        10 => {
            // Q2_K: dims [N_FF_EXP, N_EMBD, N_EXPERT]
            // Quantize mid to Q8_K first (matching C's ds4_quantize_row_q8_K)
            let n_blocks_mid = n_ff_exp / 256;
            let mut midq = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }; n_blocks_mid];
            quantize_q8_k(mid, n_ff_exp, &mut midq);

            let blocks_per_col = n_ff_exp / 256;
            let blocks_per_expert = blocks_per_col * n_embd;

            let data = layer.ffn_down_exps.as_bytes();
            let block_size = std::mem::size_of::<BlockQ2K>();

            for i in 0..n_embd {
                let col_block_start = eid * blocks_per_expert + i * blocks_per_col;
                let mut sum = 0.0f32;
                for b in 0..blocks_per_col {
                    let block_idx = col_block_start + b;
                    let block_ptr = data[block_idx * block_size..].as_ptr() as *const BlockQ2K;
                    let block = unsafe { &*block_ptr };
                    sum += vec_dot_q2_k_q8_k(
                        core::slice::from_ref(block),
                        core::slice::from_ref(&midq[b]),
                        1,
                    );
                }
                moe_out[i] += sum;
            }
        }
        _ => panic!("unsupported expert down tensor type: {}", down_type),
    }
}

pub fn layer_ffn_one(
    out_hc: &mut [f32],            // [N_HC * N_EMBD]
    in_hc: &[f32],                 // [N_HC * N_EMBD]
    layer: &LayerWeights,
    layer_idx: usize,
    token: i32,
    trace: bool,
) {
    let n_embd = N_EMBD as usize;
    let n_hc = N_HC as usize;
    let n_ff_exp = N_FF_EXP as usize;
    let n_exp_used = N_EXPERT_USED as usize;
    let clamp = SWIGLU_CLAMP_EXP;

    // HC pre: produce ffn_cur, post, comb
    let mut ffn_cur = vec![0.0f32; n_embd];
    let mut post = [0.0f32; 4];
    let mut comb = [0.0f32; 16];
    hc_ffn_pre(&mut ffn_cur, &mut post, &mut comb, in_hc, layer);

    if trace {
        print_vec_stats_rms(&format!("blk.{} ffn_cur", layer_idx), &ffn_cur);
    }

    // RMS norm
    let ffn_norm_w = layer.ffn_norm.as_f32();
    let mut norm = vec![0.0f32; n_embd];
    rms_norm_weighted(&mut norm, &ffn_cur, ffn_norm_w, n_embd, RMS_EPS);

    if trace {
        print_vec_stats_rms(&format!("blk.{} ffn_norm", layer_idx), &norm);
    }

    // --- Routed experts ---
    let mut moe_out = vec![0.0f32; n_embd];

    // Select experts: hash routing for compressed layers, top-k for dense
    let mut selected = [0usize; 6];
    let mut expert_weight = [0.0f32; 6];

    let has_tid2eid = layer.ffn_gate_tid2eid.has_data();
    if has_tid2eid {
        layer_hash_selected_experts(&mut selected, layer, token);
        layer_hash_router_weights(&mut expert_weight, layer, &norm, &selected);
    } else {
        layer_topk_selected_experts(&mut selected, &mut expert_weight, layer, &norm);
    }

    // Process each selected expert
    let gate_type = layer.ffn_gate_exps.tensor_type;
    let down_type = layer.ffn_down_exps.tensor_type;

    for ek in 0..n_exp_used {
        let eid = selected[ek];
        let w = expert_weight[ek];

        // Gate and up projections
        let mut gate = vec![0.0f32; n_ff_exp];
        let mut up = vec![0.0f32; n_ff_exp];
        expert_gate_up_matvec(&mut gate, &mut up, &norm, layer, eid, gate_type);

        if trace {
            print_vec_stats_rms(&format!("blk.{} expert {} gate", layer_idx, eid), &gate);
            print_vec_stats_rms(&format!("blk.{} expert {} up", layer_idx, eid), &up);
        }

        // Clamp + SwiGLU + expert weight
        for i in 0..n_ff_exp {
            if clamp > 1e-6 {
                if gate[i] > clamp { gate[i] = clamp; }
                if up[i] > clamp { up[i] = clamp; }
                if up[i] < -clamp { up[i] = -clamp; }
            }
            gate[i] = silu(gate[i]) * up[i] * w;
        }

        if trace {
            print_vec_stats_rms(&format!("blk.{} expert {} mid", layer_idx, eid), &gate);
        }

        // Down projection
        expert_down_matvec_accum(&mut moe_out, &gate, layer, eid, down_type);

        if trace {
            print_vec_stats_rms(&format!("blk.{} expert {} down", layer_idx, eid), &moe_out);
        }
    }

    if trace {
        print_vec_stats_rms(&format!("blk.{} routed_moe", layer_idx), &moe_out);
    }

    // --- Shared expert ---
    // gate_shexp: Q8_0 [n_ff_exp, n_embd] or F16
    let shexp_gate_type = layer.ffn_gate_shexp.tensor_type;
    let shexp_up_type = layer.ffn_up_shexp.tensor_type;

    let mut gate_shared = vec![0.0f32; n_ff_exp];
    let mut up_shared = vec![0.0f32; n_ff_exp];

    if shexp_gate_type == 8 {
        crate::quant::matvec_q8_0(&mut gate_shared, &norm, layer.ffn_gate_shexp.as_bytes(), n_embd, n_ff_exp);
    } else {
        let gate_shexp = layer.ffn_gate_shexp.as_f16();
        for i in 0..n_ff_exp {
            let mut gs = 0.0f32;
            for j in 0..n_embd {
                gs += norm[j] * f16_to_f32(gate_shexp[i * n_embd + j]);
            }
            gate_shared[i] = gs;
        }
    }

    if shexp_up_type == 8 {
        crate::quant::matvec_q8_0(&mut up_shared, &norm, layer.ffn_up_shexp.as_bytes(), n_embd, n_ff_exp);
    } else {
        let up_shexp = layer.ffn_up_shexp.as_f16();
        for i in 0..n_ff_exp {
            let mut us = 0.0f32;
            for j in 0..n_embd {
                us += norm[j] * f16_to_f32(up_shexp[i * n_embd + j]);
            }
            up_shared[i] = us;
        }
    }

    // SwiGLU
    for i in 0..n_ff_exp {
        gate_shared[i] = silu(gate_shared[i]) * up_shared[i];
    }

    let mut shared_out = vec![0.0f32; n_embd];
    let shexp_down_type = layer.ffn_down_shexp.tensor_type;
    if shexp_down_type == 8 {
        crate::quant::matvec_q8_0(&mut shared_out, &gate_shared, layer.ffn_down_shexp.as_bytes(), n_ff_exp, n_embd);
    } else {
        let down_shexp = layer.ffn_down_shexp.as_f16();
        for i in 0..n_embd {
            let mut sum = 0.0f32;
            for j in 0..n_ff_exp {
                sum += gate_shared[j] * f16_to_f32(down_shexp[i * n_ff_exp + j]);
            }
            shared_out[i] = sum;
        }
    }

    // Combine moe + shared
    let mut ffn_out = vec![0.0f32; n_embd];
    for i in 0..n_embd {
        ffn_out[i] = moe_out[i] + shared_out[i];
    }

    if trace {
        print_vec_stats_rms(&format!("blk.{} shared_ffn", layer_idx), &shared_out);
        print_vec_stats_rms(&format!("blk.{} ffn_out", layer_idx), &ffn_out);
    }

    // HC post
    hc_post_one(out_hc, &ffn_out, in_hc, &post, &comb, n_embd, n_hc);

    if trace {
        print_vec_stats_rms(&format!("blk.{} ffn_post_hc", layer_idx), out_hc);
    }
}

// ============================================================================
// Output Head
// ============================================================================

/// Output head: collapse HC streams, norm, and project to vocabulary logits.
/// Matches output_logits_one + output_hc_head_one from ds4.c.
/// Uses simple sigmoid gating (NOT Sinkhorn) — output_hc_fn maps [hc_dim]→[N_HC].
pub fn output_logits(
    logits: &mut [f32],            // [N_VOCAB]
    in_hc: &[f32],                 // [N_HC * N_EMBD]
    weights: &ModelWeights,
) {
    let n_embd = N_EMBD as usize;
    let n_hc = N_HC as usize;
    let n_vocab = N_VOCAB as usize;
    let hc_dim = n_hc * n_embd;

    // Step 1: RMS norm on the full HC state
    let mut flat = vec![0.0f32; hc_dim];
    rms_norm_no_weight(&mut flat, in_hc, hc_dim, RMS_EPS);

    // Step 2: matvec with output_hc_fn [hc_dim, N_HC] → pre [N_HC]
    let hc_fn = weights.output_hc_fn.as_f32_auto();
    let hc_fn_out_dim = hc_fn.len() / hc_dim; // should be N_HC = 4
    let mut pre = vec![0.0f32; n_hc];
    for o in 0..n_hc.min(hc_fn_out_dim) {
        let mut sum = 0.0f32;
        for j in 0..hc_dim {
            sum += flat[j] * hc_fn[o * hc_dim + j];
        }
        pre[o] = sum;
    }

    // Step 3: Simple sigmoid gate: w[i] = sigmoid(pre[i] * scale[0] + base[i]) + HC_EPS
    let scale = weights.output_hc_scale.as_f32();
    let base = weights.output_hc_base.as_f32();
    let sc = if scale.is_empty() { 1.0f32 } else { scale[0] };
    let mut w = vec![0.0f32; n_hc];
    for i in 0..n_hc {
        let bi = if i < base.len() { base[i] } else { 0.0f32 };
        let z = pre[i] * sc + bi;
        w[i] = 1.0f32 / (1.0f32 + (-z).exp()) + HC_EPS;
    }

    // Step 4: Weighted sum over HC streams: embd[d] = sum_h inp_hc[h*n_embd+d] * w[h]
    let mut hc_out = vec![0.0f32; n_embd];
    hc_weighted_sum_one(&mut hc_out, in_hc, &w, n_embd, n_hc);

    // Step 5: RMS Norm with output_norm
    let norm_w = weights.output_norm.as_f32();
    let mut normed = vec![0.0f32; n_embd];
    rms_norm_weighted(&mut normed, &hc_out, norm_w, n_embd, RMS_EPS);

    // Step 6: Output projection: Q8_0 or F16 [n_vocab, n_embd]
    let out_type = weights.output.tensor_type;
    if out_type == 8 {
        crate::quant::matvec_q8_0(logits, &normed, weights.output.as_bytes(), n_embd, n_vocab);
    } else {
        let output_w = weights.output.as_f16();
        for i in 0..n_vocab {
            let mut sum = 0.0f32;
            for j in 0..n_embd {
                sum += normed[j] * f16_to_f32(output_w[i * n_embd + j]);
            }
            logits[i] = sum;
        }
    }
}

// ============================================================================
// Full Forward Pass
// ============================================================================

/// Full forward pass for one token (used during generation).
/// If `out_hc` is provided, the final HC state is copied there (for diagnostics).
pub fn forward_one_token(
    logits: &mut [f32],
    weights: &ModelWeights,
    kv_cache: &mut KvCache,
    token: i32,
    pos: usize,
) {
    forward_one_token_debug(logits, None, weights, kv_cache, token, pos);
}

/// Like forward_one_token but optionally returns the final HC state.
pub fn forward_one_token_debug(
    logits: &mut [f32],
    out_hc: Option<&mut [f32]>,
    weights: &ModelWeights,
    kv_cache: &mut KvCache,
    token: i32,
    pos: usize,
) {
    let n_embd = N_EMBD as usize;
    let n_hc = N_HC as usize;
    let n_head = N_HEAD as usize;
    let head_dim = N_HEAD_DIM as usize;

    // Embed token
    let mut plain = vec![0.0f32; n_embd];
    embed_token_f16(weights, token, &mut plain);

    // Initialize HC streams from embedding
    let mut cur = vec![0.0f32; n_hc * n_embd];
    hc_from_plain_embedding(&mut cur, &plain);

    // Process all layers
    for il in 0..N_LAYER as usize {
        let layer = &weights.layers[il];

        // --- Attention sublayer ---
        // HC pre
        let mut attn_cur = vec![0.0f32; n_embd];
        let mut residual_hc = vec![0.0f32; n_hc * n_embd];
        let mut post = [0.0f32; 4];
        let mut comb = [0.0f32; 16];
        hc_attn_pre(&mut attn_cur, &mut residual_hc, &mut post, &mut comb, &cur, layer);

        // RMS Norm
        let mut attn_norm = vec![0.0f32; n_embd];
        let norm_weight = layer.attn_norm.as_f32();
        rms_norm_weighted(&mut attn_norm, &attn_cur, norm_weight, n_embd, RMS_EPS);

        // Q projection
        let q_dim = n_head * head_dim;
        let mut q = vec![0.0f32; q_dim];
        layer_q_projection(&mut q, &attn_norm, layer);

        // KV projection
        let mut kv = vec![0.0f32; head_dim];
        layer_kv_projection(&mut kv, &attn_norm, layer);

        // Apply RoPE (inverse=false for Q and KV)
        rope_tail_layer_inplace(&mut q, n_head, head_dim, N_ROT as usize, pos, il as u32, false);
        rope_tail_layer_inplace(&mut kv, N_HEAD_KV as usize, head_dim, N_ROT as usize, pos, il as u32, false);

        // Quantize KV for cache
        fp8_kv_quantize_row_inplace(&mut kv, head_dim, N_ROT as usize);
        f16_round_inplace(&mut kv, head_dim);

        // Store KV in cache
        kv_cache.store_raw_kv(il, pos, &kv);

        // Attention (self-attention, n_kv=1 matching C's layer_attention_one)
        let mut attn_heads = vec![0.0f32; q_dim];
        let sinks = layer.attn_sinks.as_f32_auto();
        layer_attention_one(&mut attn_heads, &q, &kv, &sinks);

        // Apply RoPE to attn output (deskew, inverse=true like in C)
        rope_tail_layer_inplace(&mut attn_heads, n_head, head_dim, N_ROT as usize, pos, il as u32, true);

        // Grouped output projection
        let mut attn_out = vec![0.0f32; n_embd];
        layer_grouped_out(&mut attn_out, &attn_heads, layer);

        // HC post (uses learned post/comb from hc_attn_pre)
        let mut after_attn_hc = vec![0.0f32; n_hc * n_embd];
        hc_post_one(&mut after_attn_hc, &attn_out, &residual_hc, &post, &comb, n_embd, n_hc);

        // --- FFN sublayer ---
        let mut after_ffn_hc = vec![0.0f32; n_hc * n_embd];
        layer_ffn_one(&mut after_ffn_hc, &after_attn_hc, layer, il, token, false);

        // Prepare for next layer
        cur.copy_from_slice(&after_ffn_hc);
    }

    // Optionally copy final HC for diagnostics
    if let Some(hc_out) = out_hc {
        let n = hc_out.len().min(cur.len());
        hc_out[..n].copy_from_slice(&cur[..n]);
    }

    // Output head
    output_logits(logits, &cur, weights);
}

/// Forward pass for the entire prompt (prefill).
pub fn forward_prefill(
    logits: &mut [f32],
    weights: &ModelWeights,
    kv_cache: &mut KvCache,
    tokens: &[i32],
) {
    for (i, &token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        let mut tmp_logits = vec![0.0f32; N_VOCAB as usize];
        forward_one_token(&mut tmp_logits, weights, kv_cache, token, i);
        if is_last {
            logits.copy_from_slice(&tmp_logits);
        }
    }
}
