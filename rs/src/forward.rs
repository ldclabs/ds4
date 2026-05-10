// CPU forward pass implementation for DeepSeek V4 Flash.
//
// This module implements the complete layer-by-layer forward pass:
// embedding → 43 transformer layers (HC attention + MoE FFN) → output head.
//
// All algorithms mirror the C reference in ds4.c exactly.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::model::{LayerWeights, ModelWeights};
use crate::quant::{BlockQ2K, BlockIq2Xxs, BlockQ8K, quantize_q8_k, vec_dot_iq2_xxs_q8_k_dual_multi, vec_dot_q2_k_q8_k};

/// When true, print HC state statistics after each layer (diagnostic only — very verbose).
pub static TRACE_HC: AtomicBool = AtomicBool::new(false);
/// Maximum number of layers to trace (counts down each layer). 0 = unlimited.
pub static TRACE_HC_REMAINING: AtomicUsize = AtomicUsize::new(0);

use std::sync::atomic::AtomicUsize;
use crate::{
    N_EMBD, N_HEAD, N_HEAD_KV, N_HEAD_DIM, N_ROT, N_OUT_GROUP,
    N_LORA_Q, N_LORA_O, N_EXPERT, N_EXPERT_USED, N_FF_EXP,
    N_HC, N_HC_SINKHORN_ITER, N_LAYER, N_VOCAB,
    RMS_EPS, HC_EPS, EXPERT_WEIGHT_SCALE,
    SWIGLU_CLAMP_EXP, N_SWA, N_INDEXER_HEAD_DIM, N_INDEXER_HEAD, N_INDEXER_TOP_K, NEG_INF,
    f16_to_f32, f32_to_f16, silu, softplus_stable,
    rms_norm_weighted, rms_norm_no_weight,
    layer_compress_ratio, hash_routed_expert,
    fp8_kv_quantize_row_inplace,
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
// KV Cache — per-layer cache matching C's ds4_layer_cache / ds4_kv_cache
// ============================================================================

/// Per-layer KV cache, matching C's ds4_layer_cache.
pub struct LayerCache {
    /// Raw SWA KV rows (circular buffer): [cap_raw * N_HEAD_DIM] as f32 (FP16-rounded)
    pub raw_kv: Vec<f32>,
    pub n_raw: u32,
    pub cap_raw: u32,

    /// Compression ratio for this layer (0 = none, 4 or 128)
    pub compress_ratio: u32,

    /// Compressed KV rows for attention: [comp_cap * N_HEAD_DIM]
    pub attn_comp_kv: Vec<f32>,
    pub n_comp: u32,
    pub comp_cap: u32,

    /// Compressor sliding-window state for attention KV
    /// Layout: ratio-128 → [ratio * N_HEAD_DIM]; ratio-4 → [8 * (2*N_HEAD_DIM)]
    pub attn_state_kv: Vec<f32>,
    pub attn_state_score: Vec<f32>,

    /// Compressed KV rows for indexer (ratio-4 only): [comp_cap * N_INDEXER_HEAD_DIM]
    pub index_comp_kv: Vec<f32>,
    pub n_index_comp: u32,

    /// Compressor sliding-window state for indexer KV (ratio-4 only)
    /// Layout: [8 * (2*N_INDEXER_HEAD_DIM)]
    pub index_state_kv: Vec<f32>,
    pub index_state_score: Vec<f32>,
}

impl LayerCache {
    pub fn new(ctx_size: usize, il: u32) -> Self {
        let raw_cap = (N_SWA as usize).min(ctx_size).max(1);
        let ratio = layer_compress_ratio(il);
        let comp_cap = ctx_size / 4 + 2;
        let head_dim = N_HEAD_DIM as usize;

        let (attn_comp_kv, attn_state_kv, attn_state_score) = if ratio != 0 {
            let coff: usize = if ratio == 4 { 2 } else { 1 };
            let width = coff * head_dim;
            let state_rows = if ratio == 4 { coff * ratio as usize } else { ratio as usize };
            (
                vec![0.0f32; comp_cap * head_dim],
                vec![0.0f32; state_rows * width],
                {
                    let mut s = vec![NEG_INF; state_rows * width];
                    // For ratio-4, rows 0..ratio in the first lane should start as zeros
                    // instead of NEG_INF (they are the "present" buffer)
                    if ratio == 4 {
                        for r in 0..ratio as usize {
                            for j in 0..head_dim {
                                s[r * width + j] = 0.0;
                            }
                        }
                    }
                    s
                },
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        let (index_comp_kv, index_state_kv, index_state_score) = if ratio == 4 {
            let indexer_head_dim = N_INDEXER_HEAD_DIM as usize;
            let coff = 2usize;
            let width = coff * indexer_head_dim;
            let state_rows = coff * ratio as usize;
            (
                vec![0.0f32; comp_cap * indexer_head_dim],
                vec![0.0f32; state_rows * width],
                {
                    let mut s = vec![NEG_INF; state_rows * width];
                    for r in 0..ratio as usize {
                        for j in 0..indexer_head_dim {
                            s[r * width + j] = 0.0;
                        }
                    }
                    s
                },
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        LayerCache {
            raw_kv: vec![0.0f32; raw_cap * head_dim],
            n_raw: 0,
            cap_raw: raw_cap as u32,
            compress_ratio: ratio,
            attn_comp_kv,
            n_comp: 0,
            comp_cap: comp_cap as u32,
            attn_state_kv,
            attn_state_score,
            index_comp_kv,
            n_index_comp: 0,
            index_state_kv,
            index_state_score,
        }
    }
}

/// Per-layer checkpoint data for speculative decoding KV cache save/restore.
pub struct LayerCheckpoint {
    pub n_raw: u32,
    pub n_comp: u32,
    pub n_index_comp: u32,
    pub raw_kv: Vec<f32>,
    pub attn_comp_kv: Vec<f32>,
    pub attn_state_kv: Vec<f32>,
    pub attn_state_score: Vec<f32>,
    pub index_comp_kv: Vec<f32>,
    pub index_state_kv: Vec<f32>,
    pub index_state_score: Vec<f32>,
}

/// Checkpoint of the KV cache state for a range of layers.
pub struct KvCacheCheckpoint {
    pub layer_states: Vec<LayerCheckpoint>,
}

pub struct KvCache {
    pub layers: Vec<LayerCache>,
    pub ctx_size: usize,
}

impl KvCache {
    pub fn new(ctx_size: usize) -> Self {
        let layers: Vec<LayerCache> = (0..N_LAYER)
            .map(|il| LayerCache::new(ctx_size, il))
            .collect();
        KvCache { layers, ctx_size }
    }

    /// Push a raw KV row into the sliding window (FP16-rounded, matches C's kv_cache_push_raw).
    pub fn push_raw(&mut self, il: usize, kv: &[f32]) {
        let layer = &mut self.layers[il];
        let head_dim = N_HEAD_DIM as usize;

        if layer.n_raw < layer.cap_raw {
            let dst_start = layer.n_raw as usize * head_dim;
            for i in 0..head_dim {
                layer.raw_kv[dst_start + i] = f16_to_f32(f32_to_f16(kv[i]));
            }
            layer.n_raw += 1;
        } else {
            // Slide: shift all rows back by one
            let slide_bytes = (layer.cap_raw as usize - 1) * head_dim;
            layer.raw_kv.copy_within(head_dim..head_dim + slide_bytes, 0);
            let dst_start = (layer.cap_raw as usize - 1) * head_dim;
            for i in 0..head_dim {
                layer.raw_kv[dst_start + i] = f16_to_f32(f32_to_f16(kv[i]));
            }
        }
    }

    /// Push a compressed KV row (FP16-rounded, matches C's kv_cache_push_comp).
    pub fn push_comp(rows: &mut Vec<f32>, n_rows: &mut u32, cap_rows: u32, row_dim: usize, kv: &[f32]) {
        assert!((*n_rows as usize) < cap_rows as usize, "compressed KV cache capacity exceeded");
        let dst_start = (*n_rows as usize) * row_dim;
        for i in 0..row_dim {
            rows[dst_start + i] = f16_to_f32(f32_to_f16(kv[i]));
        }
        *n_rows += 1;
    }

    /// Snapshot KV cache state for a range of layers, enabling speculative
    /// decoding checkpoint/restore. Saves only the mutable state counters
    /// and the data they reference; untracked tail data is not saved.
    pub fn checkpoint_layers(&self, end_layer: usize) -> KvCacheCheckpoint {
        let n_layers = end_layer.min(self.layers.len());
        let mut layer_states = Vec::with_capacity(n_layers);
        for il in 0..n_layers {
            let lc = &self.layers[il];
            let n_raw = lc.n_raw as usize;
            let n_comp = lc.n_comp as usize;
            let n_index_comp = lc.n_index_comp as usize;

            layer_states.push(LayerCheckpoint {
                n_raw: lc.n_raw,
                n_comp: lc.n_comp,
                n_index_comp: lc.n_index_comp,
                raw_kv: lc.raw_kv[..n_raw * N_HEAD_DIM as usize].to_vec(),
                attn_comp_kv: lc.attn_comp_kv[..n_comp * N_HEAD_DIM as usize].to_vec(),
                attn_state_kv: lc.attn_state_kv.clone(),
                attn_state_score: lc.attn_state_score.clone(),
                index_comp_kv: if lc.compress_ratio == 4 {
                    let idx_dim = N_INDEXER_HEAD_DIM as usize;
                    lc.index_comp_kv[..n_index_comp * idx_dim].to_vec()
                } else {
                    Vec::new()
                },
                index_state_kv: lc.index_state_kv.clone(),
                index_state_score: lc.index_state_score.clone(),
            });
        }
        KvCacheCheckpoint { layer_states }
    }

    /// Restore KV cache state from a checkpoint. Only layers present in the
    /// checkpoint are restored; the checkpoint must have been created from a
    /// compatible KvCache (same ctx_size). The saved data overwrites the
    /// corresponding slices; untracked tail data is zero-filled to match the
    /// original snapshot boundary.
    pub fn restore_layers(&mut self, ckpt: &KvCacheCheckpoint) {
        let n = ckpt.layer_states.len().min(self.layers.len());
        for il in 0..n {
            let cs = &ckpt.layer_states[il];
            let lc = &mut self.layers[il];
            lc.n_raw = cs.n_raw;
            lc.n_comp = cs.n_comp;
            lc.n_index_comp = cs.n_index_comp;

            let raw_len = cs.raw_kv.len();
            lc.raw_kv[..raw_len].copy_from_slice(&cs.raw_kv);
            // Zero out tail beyond restored region
            lc.raw_kv[raw_len..].fill(0.0);

            let comp_len = cs.attn_comp_kv.len();
            if comp_len > 0 && !lc.attn_comp_kv.is_empty() {
                lc.attn_comp_kv[..comp_len].copy_from_slice(&cs.attn_comp_kv);
                lc.attn_comp_kv[comp_len..].fill(0.0);
            }

            lc.attn_state_kv.copy_from_slice(&cs.attn_state_kv);
            lc.attn_state_score.copy_from_slice(&cs.attn_state_score);

            if lc.compress_ratio == 4 {
                let idx_len = cs.index_comp_kv.len();
                if idx_len > 0 && !lc.index_comp_kv.is_empty() {
                    lc.index_comp_kv[..idx_len].copy_from_slice(&cs.index_comp_kv);
                    lc.index_comp_kv[idx_len..].fill(0.0);
                }
                if !lc.index_state_kv.is_empty() {
                    lc.index_state_kv.copy_from_slice(&cs.index_state_kv);
                    lc.index_state_score.copy_from_slice(&cs.index_state_score);
                }
            }
        }
    }

    /// Finish prefill states: clear partial compressor windows so decode starts
    /// from the same state the streaming path would produce (matches C's kv_cache_finish_prefill_states).
    pub fn finish_prefill_states(&mut self, n_tokens: usize) {
        for il in 0..N_LAYER as usize {
            let layer = &mut self.layers[il];
            let ratio = layer.compress_ratio;
            if ratio == 0 {
                continue;
            }
            compressor_finish_prefill_state(
                &mut layer.attn_state_kv,
                &mut layer.attn_state_score,
                N_HEAD_DIM,
                ratio,
                n_tokens,
            );
            if ratio == 4 {
                compressor_finish_prefill_state(
                    &mut layer.index_state_kv,
                    &mut layer.index_state_score,
                    N_INDEXER_HEAD_DIM,
                    ratio,
                    n_tokens,
                );
            }
        }
    }
}

// ============================================================================
// Compressor functions — match compressor_* in ds4.c
// ============================================================================

/// Clear partial compressor windows so decode starts from the same state the
/// streaming path would have produced (matches C's compressor_finish_prefill_state_cpu).
fn compressor_finish_prefill_state(
    state_kv: &mut [f32],
    state_score: &mut [f32],
    head_dim: u32,
    compress_ratio: u32,
    n_tokens: usize,
) {
    if state_kv.is_empty() || state_score.is_empty() || head_dim == 0 || compress_ratio == 0 {
        return;
    }

    let coff: usize = if compress_ratio == 4 { 2 } else { 1 };
    let width = coff * head_dim as usize;
    let rem = n_tokens % compress_ratio as usize;
    let clear_start = if compress_ratio == 4 {
        compress_ratio as usize + rem
    } else {
        rem
    };
    let clear_end = if compress_ratio == 4 {
        2 * compress_ratio as usize
    } else {
        compress_ratio as usize
    };

    for row in clear_start..clear_end {
        let kv_start = row * width;
        let sc_start = row * width;
        state_kv[kv_start..kv_start + width].fill(0.0);
        state_score[sc_start..sc_start + width].fill(NEG_INF);
    }
}

/// Pool the current compression window with a softmax over per-dimension scores.
/// Matches C's compressor_pool_decode_state.
fn compressor_pool_decode_state(
    out: &mut [f32],
    state_kv: &[f32],
    state_score: &[f32],
    head_dim: u32,
    compress_ratio: u32,
) {
    let coff: usize = if compress_ratio == 4 { 2 } else { 1 };
    let width = coff * head_dim as usize;
    let hd = head_dim as usize;

    for j in 0..hd {
        let mut max_score = NEG_INF;

        if compress_ratio == 4 {
            for r in 0..compress_ratio as usize {
                let sp = state_score[r * width + j];
                let sc = state_score[(compress_ratio as usize + r) * width + hd + j];
                if sp > max_score { max_score = sp; }
                if sc > max_score { max_score = sc; }
            }
        } else {
            for r in 0..compress_ratio as usize {
                let s = state_score[r * width + j];
                if s > max_score { max_score = s; }
            }
        }

        if max_score <= NEG_INF * 0.5 {
            out[j] = 0.0;
            continue;
        }

        let mut denom = 0.0f32;
        let mut sum = 0.0f32;
        if compress_ratio == 4 {
            for r in 0..compress_ratio as usize {
                let wp = (state_score[r * width + j] - max_score).exp();
                let wc = (state_score[(compress_ratio as usize + r) * width + hd + j] - max_score).exp();
                denom += wp + wc;
                sum += wp * state_kv[r * width + j];
                sum += wc * state_kv[(compress_ratio as usize + r) * width + hd + j];
            }
        } else {
            for r in 0..compress_ratio as usize {
                let w = (state_score[r * width + j] - max_score).exp();
                denom += w;
                sum += w * state_kv[r * width + j];
            }
        }

        out[j] = if denom > 0.0 { sum / denom } else { 0.0 };
    }
}

/// Streaming compressor update for one token.
/// Returns true if a compressed row was emitted (on ratio boundary).
/// Matches C's compressor_decode_one.
fn compressor_decode_one(
    out_comp: &mut [f32],           // [head_dim] — compressed output if returned true
    _layer: &crate::model::LayerWeights,
    kv_tensor: &crate::model::Tensor,      // attn_compressor_kv or indexer_compressor_kv
    gate_tensor: &crate::model::Tensor,    // attn_compressor_gate or indexer_compressor_gate
    ape_tensor: &crate::model::Tensor,     // attn_compressor_ape or indexer_compressor_ape
    norm_tensor: &crate::model::Tensor,    // attn_compressor_norm or indexer_compressor_norm
    x: &[f32],                       // [N_EMBD] — normalized attention input
    state_kv: &mut [f32],
    state_score: &mut [f32],
    head_dim: u32,
    compress_ratio: u32,
    il: u32,
    pos: usize,
) -> bool {
    let coff: usize = if compress_ratio == 4 { 2 } else { 1 };
    let width = coff * head_dim as usize;
    let pos_mod = pos % compress_ratio as usize;
    let row = if compress_ratio == 4 {
        compress_ratio as usize + pos_mod
    } else {
        pos_mod
    };
    let should_compress = (pos + 1) % compress_ratio as usize == 0;

    // Project input through wkv and wgate to get kv_cur and sc_cur
    let n_embd = N_EMBD as usize;
    let mut kv_cur = vec![0.0f32; width];
    let mut sc_cur = vec![0.0f32; width];

    let kv_f16 = kv_tensor.as_f16();
    let gate_f16 = gate_tensor.as_f16();

    // Check if quantized Q8_0 path is available (both tensors are Q8_0)
    if kv_tensor.tensor_type == 8 && gate_tensor.tensor_type == 8 && !kv_f16.is_empty() && !gate_f16.is_empty() {
        let in_dim = n_embd;
        let blocks = (in_dim + 31) / 32;
        let mut xq = vec![0i8; blocks * 32];
        let mut xscale = vec![0.0f32; blocks];
        crate::quant::quantize_q8_0_activation(x, &mut xq, &mut xscale, in_dim as u32);
        crate::quant::matvec_q8_0_pair_prequant(&mut kv_cur, &mut sc_cur, kv_tensor, gate_tensor, &xq, &xscale);
    } else {
        // Generic matvec path for F16 weights
        let kv_bytes = kv_tensor.as_bytes();
        if !kv_bytes.is_empty() {
            crate::quant::matvec_f16(&mut kv_cur, x, kv_bytes, n_embd, width);
        }
        let gate_bytes = gate_tensor.as_bytes();
        if !gate_bytes.is_empty() {
            crate::quant::matvec_f16(&mut sc_cur, x, gate_bytes, n_embd, width);
        }
    }

    // Add APE (additive position embedding) to scores
    let ape_f32 = ape_tensor.as_f32();
    if !ape_f32.is_empty() {
        let ape_dim = width; // APE has shape [width] or [ratio, width]
        for j in 0..width.min(ape_dim) {
            let ape_val = if ape_f32.len() > width {
                // 2D APE: [ratio, width] or similar
                ape_f32.get(pos_mod * width + j).copied().unwrap_or(0.0)
            } else {
                ape_f32.get(j).copied().unwrap_or(0.0)
            };
            sc_cur[j] += ape_val;
        }
    }

    // Store into state
    let kv_dst_start = row * width;
    let sc_dst_start = row * width;
    state_kv[kv_dst_start..kv_dst_start + width].copy_from_slice(&kv_cur);
    state_score[sc_dst_start..sc_dst_start + width].copy_from_slice(&sc_cur);

    if !should_compress {
        return false;
    }

    // Pool the window
    let hd = head_dim as usize;
    let mut pooled = vec![0.0f32; hd];
    compressor_pool_decode_state(&mut pooled, state_kv, state_score, head_dim, compress_ratio);

    // RMS norm the pooled result
    let mut ss = 0.0f64;
    for i in 0..hd {
        ss += (pooled[i] as f64) * (pooled[i] as f64);
    }
    let rms = 1.0 / ((ss / hd as f64) as f32 + RMS_EPS).sqrt();
    let norm_f32 = norm_tensor.as_f32();
    for i in 0..hd {
        let n = if norm_f32.len() > i { norm_f32[i] } else { 1.0 };
        out_comp[i] = pooled[i] * rms * n;
    }

    // Apply RoPE at compressed position and quantize
    let comp_pos = pos + 1 - compress_ratio as usize;
    rope_tail_layer_inplace(out_comp, 1, hd, N_ROT as usize, comp_pos, il, false);
    if head_dim == N_HEAD_DIM {
        fp8_kv_quantize_row_inplace(out_comp, hd, N_ROT as usize);
    }

    // For ratio-4: shift second lane (rows 4-7) down to first lane (rows 0-3)
    if compress_ratio == 4 {
        let ratio = compress_ratio as usize;
        for r in 0..ratio {
            let src = (ratio + r) * width;
            let dst = r * width;
            state_kv.copy_within(src..src + width, dst);
            state_score.copy_within(src..src + width, dst);
        }
        // Clear the second lane
        for r in ratio..2 * ratio {
            let start = r * width;
            state_kv[start..start + width].fill(0.0);
            state_score[start..start + width].fill(NEG_INF);
        }
    }

    true
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
    // Clamp n_rot to head_dim (indexer head_dim can be smaller than N_ROT)
    let n_rot = if n_rot > head_dim { head_dim } else { n_rot };
    if n_rot == 0 { return; }
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

/// Q projection with LoRA. Also returns the intermediate `qr_norm` [N_LORA_Q]
/// used by the indexer for ratio-4 layers.
/// Matches layer_q_projection_with_lora_one from ds4.c.
pub fn layer_q_projection_with_lora(
    q: &mut [f32],                 // [N_HEAD * N_HEAD_DIM]
    qr_norm: &mut [f32],           // [N_LORA_Q] — intermediate LoRA representation (RMS-normed)
    attn_norm: &[f32],             // [N_EMBD] — already RMS-normed
    layer: &LayerWeights,
) {
    let n_embd = N_EMBD as usize;
    let n_head = N_HEAD as usize;
    let head_dim = N_HEAD_DIM as usize;
    let lora_q = N_LORA_Q as usize;
    let q_dim = n_head * head_dim;

    // q_a: Q8_0 [lora_q, n_embd] → q_a_out [lora_q]
    let mut q_a_out = vec![0.0f32; lora_q];
    crate::quant::matvec_q8_0(&mut q_a_out, attn_norm, layer.attn_q_a.as_bytes(), n_embd, lora_q);

    // RMS-norm on q_a_out → qr_norm (the intermediate LoRA representation)
    let q_a_norm = layer.attn_q_a_norm.as_f32();
    rms_norm_weighted(qr_norm, &q_a_out, q_a_norm, lora_q, RMS_EPS);

    // q_b: Q8_0 [q_dim, lora_q] or F16 — up-project from qr_norm
    let q_b_type = layer.attn_q_b.tensor_type;
    if q_b_type == 8 {
        // Q8_0 matvec
        crate::quant::matvec_q8_0(q, qr_norm, layer.attn_q_b.as_bytes(), lora_q, q_dim);
    } else {
        // F16 fallback
        let q_b = layer.attn_q_b.as_f16();
        for i in 0..q_dim {
            let mut sum = 0.0f32;
            for j in 0..lora_q {
                sum += qr_norm[j] * f16_to_f32(q_b[i * lora_q + j]);
            }
            q[i] = sum;
        }
    }

    // Per-head RMS norm (matches C's head_rms_norm_inplace)
    head_rms_norm_inplace(q, n_head, head_dim, RMS_EPS);
}

/// Q projection with LoRA (convenience wrapper that discards qr_norm).
/// Matches layer_q_projection_normed_one from ds4.c.
pub fn layer_q_projection(
    q: &mut [f32],                 // [N_HEAD * N_HEAD_DIM]
    attn_norm: &[f32],            // [N_EMBD] — already RMS-normed
    layer: &LayerWeights,
) {
    let lora_q = N_LORA_Q as usize;
    let mut qr_norm = vec![0.0f32; lora_q];
    layer_q_projection_with_lora(q, &mut qr_norm, attn_norm, layer);
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

/// Single-token attention over raw SWA rows only (for ratio=0 layers).
/// Matches C's `layer_attention_rows_one`.
pub fn layer_attention_rows_one(
    attn_out: &mut [f32],          // [N_HEAD * N_HEAD_DIM]
    q: &[f32],                     // [N_HEAD * N_HEAD_DIM]
    raw_kv: &[f32],                // [n_raw * N_HEAD_DIM] raw SWA KV rows
    n_raw: u32,
    sinks: &[f32],                 // [N_HEAD] attention sink bias
) {
    let head_dim = N_HEAD_DIM as usize;
    let kq_scale = 1.0 / (head_dim as f32).sqrt();

    use rayon::prelude::*;
    attn_out.par_chunks_mut(head_dim).zip(q.par_chunks(head_dim)).enumerate().for_each(|(h, (oh, qh))| {
        oh.fill(0.0);

        let sink_val = sinks.get(h).copied().unwrap_or(0.0f32);
        let mut max_score = sink_val;
        let mut scores = vec![0.0f32; n_raw as usize];

        for r in 0..n_raw as usize {
            let kv = &raw_kv[r * head_dim..(r + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * kv[d];
            }
            scores[r] = dot * kq_scale;
            if scores[r] > max_score { max_score = scores[r]; }
        }

        let mut denom = (sink_val - max_score).exp();
        for r in 0..n_raw as usize {
            let weight = (scores[r] - max_score).exp();
            let kv = &raw_kv[r * head_dim..(r + 1) * head_dim];
            denom += weight;
            for d in 0..head_dim {
                oh[d] += kv[d] * weight;
            }
        }

        let inv = 1.0 / denom;
        for d in 0..head_dim {
            oh[d] *= inv;
        }
    });
}

/// Single-token attention over raw SWA rows + compressed rows.
/// Matches C's `layer_attention_mixed_one`.
pub fn layer_attention_mixed_one(
    attn_out: &mut [f32],          // [N_HEAD * N_HEAD_DIM]
    q: &[f32],                     // [N_HEAD * N_HEAD_DIM]
    raw_kv: &[f32],                // [n_raw * N_HEAD_DIM]
    n_raw: u32,
    comp_kv: &[f32],               // [n_comp * N_HEAD_DIM]
    n_comp: u32,
    comp_allowed: Option<&[bool]>, // [n_comp] — indexer mask (None = all allowed)
    sinks: &[f32],                 // [N_HEAD] attention sink bias
) {
    let head_dim = N_HEAD_DIM as usize;
    let kq_scale = 1.0 / (head_dim as f32).sqrt();
    let n_total = n_raw as usize + n_comp as usize;
    let n_raw_usize = n_raw as usize;
    let n_comp_usize = n_comp as usize;

    use rayon::prelude::*;
    attn_out.par_chunks_mut(head_dim).zip(q.par_chunks(head_dim)).enumerate().for_each(|(h, (oh, qh))| {
        oh.fill(0.0);

        let sink_val = sinks.get(h).copied().unwrap_or(0.0f32);
        let mut max_score = sink_val;
        let mut scores = vec![0.0f32; n_total];

        // Raw entries
        for r in 0..n_raw_usize {
            let kv = &raw_kv[r * head_dim..(r + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * kv[d];
            }
            scores[r] = dot * kq_scale;
            if scores[r] > max_score { max_score = scores[r]; }
        }

        // Compressed entries
        for (c, score) in scores.iter_mut().skip(n_raw_usize).enumerate() {
            if let Some(allowed) = comp_allowed {
                if !allowed[c] {
                    *score = NEG_INF;
                    continue;
                }
            }
            let kv = &comp_kv[c * head_dim..(c + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * kv[d];
            }
            *score = dot * kq_scale;
            if *score > max_score { max_score = *score; }
        }

        let mut denom = (sink_val - max_score).exp();

        // Weighted sum over raw
        for r in 0..n_raw_usize {
            let weight = (scores[r] - max_score).exp();
            let kv = &raw_kv[r * head_dim..(r + 1) * head_dim];
            denom += weight;
            for d in 0..head_dim {
                oh[d] += kv[d] * weight;
            }
        }

        // Weighted sum over compressed
        for c in 0..n_comp_usize {
            let score = scores[n_raw_usize + c];
            if score <= NEG_INF * 0.5 { continue; }
            let weight = (score - max_score).exp();
            let kv = &comp_kv[c * head_dim..(c + 1) * head_dim];
            denom += weight;
            for d in 0..head_dim {
                oh[d] += kv[d] * weight;
            }
        }

        let inv = 1.0 / denom;
        for d in 0..head_dim {
            oh[d] *= inv;
        }
    });
}

/// Indexer: compute which compressed rows are allowed for the current token.
/// Matches C's `indexer_allowed_decode_one`.
///
/// Algorithm (matching ds4.c):
///   1. q = indexer_attn_q_b @ qr_norm     (F16 matvec, input=qr_norm)
///   2. RoPE on q
///   3. weights = indexer_proj @ attn_norm  (F16 matvec, input=attn_norm)
///   4. scale = 1/sqrt(head_dim * n_head), weights *= scale
///   5. For each compressed row c, score = sum_h(ReLU(dot(kv[c], q_h)) * weights[h])
///   6. Top-k selection: pick the k highest-scoring rows
///
/// Returns a Vec<bool> of length n_comp where true = allowed.
pub fn indexer_allowed_decode_one(
    layer: &crate::model::LayerWeights,
    attn_norm: &[f32],             // [N_EMBD] — attention input (for indexer_proj)
    qr_norm: &[f32],               // [N_LORA_Q] — intermediate LoRA Q representation
    index_comp_kv: &[f32],         // [n_comp * N_INDEXER_HEAD_DIM]
    n_comp: u32,
    il: u32,
    pos: usize,
) -> Vec<bool> {
    let n_indexer_head = N_INDEXER_HEAD as usize;
    let indexer_head_dim = N_INDEXER_HEAD_DIM as usize;
    let n_embd = N_EMBD as usize;
    let lora_q = N_LORA_Q as usize;
    let n_comp = n_comp as usize;
    let top_k_raw = N_INDEXER_TOP_K as usize;
    let top_k = if top_k_raw < n_comp { top_k_raw } else { n_comp };
    let mut allowed = vec![false; n_comp];

    if n_comp == 0 {
        return allowed;
    }

    // All rows allowed if n_comp <= top_k
    if top_k == n_comp {
        for a in allowed.iter_mut() { *a = true; }
        return allowed;
    }

    // Step 1: q = indexer_attn_q_b @ qr_norm
    // Weight [N_LORA_Q, n_indexer_head * indexer_head_dim] × qr_norm [N_LORA_Q] → q
    let q_dim = n_indexer_head * indexer_head_dim;
    let mut index_q = vec![0.0f32; q_dim];
    let q_b_bytes = layer.indexer_attn_q_b.as_bytes();
    if !q_b_bytes.is_empty() {
        crate::quant::matvec_f16(&mut index_q, qr_norm, q_b_bytes, lora_q, q_dim);
    }

    // Step 2: RoPE on indexer Q
    rope_tail_layer_inplace(&mut index_q, n_indexer_head, indexer_head_dim, N_ROT as usize, pos, il, false);

    // Step 3: weights = indexer_proj @ attn_norm
    // Weight [N_EMBD, N_INDEXER_HEAD] × attn_norm [N_EMBD] → weights [N_INDEXER_HEAD]
    let mut weights = vec![0.0f32; n_indexer_head];
    let proj_bytes = layer.indexer_proj.as_bytes();
    if !proj_bytes.is_empty() {
        crate::quant::matvec_f16(&mut weights, attn_norm, proj_bytes, n_embd, n_indexer_head);
    }

    // Step 4: scale weights
    let scale = 1.0 / ((indexer_head_dim * n_indexer_head) as f32).sqrt();
    for w in weights.iter_mut() { *w *= scale; }

    // Step 5-6: score each compressed row, then top-k
    let mut scores = vec![0.0f32; n_comp];

    for c in 0..n_comp {
        let kv = &index_comp_kv[c * indexer_head_dim..(c + 1) * indexer_head_dim];
        let mut s = 0.0f32;
        for h in 0..n_indexer_head {
            let qh = &index_q[h * indexer_head_dim..(h + 1) * indexer_head_dim];
            let mut dot = 0.0f32;
            for d in 0..indexer_head_dim {
                dot += qh[d] * kv[d];
            }
            // ReLU on per-head dot
            if dot < 0.0 { dot = 0.0; }
            s += dot * weights[h];
        }
        scores[c] = s;
    }

    // Top-k selection (no heap for deterministic tie-breaking, matching C's linear scan)
    for _k in 0..top_k {
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..n_comp {
            if !allowed[c] && scores[c] > best_score {
                best_idx = c;
                best_score = scores[c];
            }
        }
        allowed[best_idx] = true;
    }

    allowed
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

        use rayon::prelude::*;
        low.par_chunks_mut(rank).enumerate().for_each(|(g, low_chunk)| {
            let head_start = g * group_dim;
            crate::quant::matvec_q8_0(
                low_chunk,
                &attn_heads[head_start..head_start + group_dim],
                &o_a_bytes[g * rank * bytes_per_column..(g * rank + rank) * bytes_per_column],
                group_dim,
                rank,
            );
        });
    } else if o_a_type <= 1 {
        // F32 or F16
        let o_a_f16 = layer.attn_output_a.as_f16();
        let n_elems = n_groups * rank * group_dim;
        if o_a_f16.len() >= n_elems {
            use rayon::prelude::*;
            low.par_chunks_mut(rank).enumerate().for_each(|(g, low_chunk)| {
                let head_start = g * group_dim;
                for i in 0..rank {
                    let mut sum = 0.0f32;
                    for j in 0..group_dim {
                        sum += attn_heads[head_start + j]
                            * f16_to_f32(o_a_f16[(g * rank + i) * group_dim + j]);
                    }
                    low_chunk[i] = sum;
                }
            });
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

            // P5: parallelize across output rows
            use rayon::prelude::*;
            let results: Vec<(f32, f32)> = (0..n_ff_exp).into_par_iter().map(|j| {
                let row_base = eid * stride + j * n_embd;
                let mut gs = 0.0f32;
                let mut us = 0.0f32;
                for i in 0..n_embd {
                    gs += x[i] * f16_to_f32(data[row_base + i]);
                    us += x[i] * f16_to_f32(up_data[row_base + i]);
                }
                (gs, us)
            }).collect();
            for (j, (gs, us)) in results.into_iter().enumerate() {
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

            // P1 + P3 + P4: batched blocks + fused dual-channel + wide multi-row.
            // Process 2048 output rows in chunks of 4 via rayon. Within each chunk,
            // Q8_K activation data is loaded once and shared across all 4 rows,
            // reducing memory traffic and improving cache utilization.
            const CHUNK_SIZE: usize = 4;
            use rayon::prelude::*;
            let row_indices: Vec<usize> = (0..n_ff_exp).collect();
            let results: Vec<(f32, f32)> = row_indices.par_chunks(CHUNK_SIZE).flat_map(|chunk| {
                let start = chunk[0];
                let n = chunk.len();
                let row_block_start = eid * blocks_per_expert + start * blocks_per_row;
                let byte_offset = row_block_start * block_size;

                let gate_slice = unsafe {
                    std::slice::from_raw_parts(
                        gate_bytes[byte_offset..].as_ptr() as *const BlockIq2Xxs,
                        n * blocks_per_row,
                    )
                };
                let up_slice = unsafe {
                    std::slice::from_raw_parts(
                        up_bytes[byte_offset..].as_ptr() as *const BlockIq2Xxs,
                        n * blocks_per_row,
                    )
                };

                let mut gate_out = vec![0.0f32; n];
                let mut up_out = vec![0.0f32; n];
                vec_dot_iq2_xxs_q8_k_dual_multi(
                    gate_slice, up_slice, &xq, blocks_per_row, n,
                    &mut gate_out, &mut up_out,
                );
                gate_out.into_iter().zip(up_out.into_iter()).collect::<Vec<_>>()
            }).collect();
            for (j, (g, u)) in results.into_iter().enumerate() {
                gate[j] = g;
                up[j] = u;
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
            // P5: parallelize across output rows
            use rayon::prelude::*;
            let results: Vec<f32> = (0..n_embd).into_par_iter().map(|i| {
                let col_base = eid * stride + i * n_ff_exp;
                let mut sum = 0.0f32;
                for j in 0..n_ff_exp {
                    sum += mid[j] * f16_to_f32(data[col_base + j]);
                }
                sum
            }).collect();
            for (i, sum) in results.into_iter().enumerate() {
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

            // Q2_K path — batched + parallel rows (P1 + P3)
            // Process 4096 output rows in parallel via rayon
            use rayon::prelude::*;
            let results: Vec<f32> = (0..n_embd).into_par_iter().map(|i| {
                let col_block_start = eid * blocks_per_expert + i * blocks_per_col;
                let byte_offset = col_block_start * block_size;

                let down_blocks = unsafe {
                    std::slice::from_raw_parts(
                        data[byte_offset..].as_ptr() as *const BlockQ2K,
                        blocks_per_col,
                    )
                };

                vec_dot_q2_k_q8_k(down_blocks, &midq, blocks_per_col)
            }).collect();
            for (i, sum) in results.into_iter().enumerate() {
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

    // P4: Process routed + shared experts in parallel via rayon::join
    // Both paths are independent — share only read-only `norm` and layer weights.
    let gate_type = layer.ffn_gate_exps.tensor_type;
    let down_type = layer.ffn_down_exps.tensor_type;
    let shexp_gate_type = layer.ffn_gate_shexp.tensor_type;
    let shexp_up_type = layer.ffn_up_shexp.tensor_type;
    use rayon::prelude::*;

    let (expert_outputs, shared_out) = rayon::join(
        || {
            // --- Routed experts (parallel across experts + rows) ---
            (0..n_exp_used).into_par_iter().map(|ek| {
                let eid = selected[ek];
                let w = expert_weight[ek];

                // Gate and up projections
                let mut gate = vec![0.0f32; n_ff_exp];
                let mut up = vec![0.0f32; n_ff_exp];
                expert_gate_up_matvec(&mut gate, &mut up, &norm, layer, eid, gate_type);

                // Clamp + SwiGLU + expert weight
                for i in 0..n_ff_exp {
                    if clamp > 1e-6 {
                        if gate[i] > clamp { gate[i] = clamp; }
                        if up[i] > clamp { up[i] = clamp; }
                        if up[i] < -clamp { up[i] = -clamp; }
                    }
                    gate[i] = silu(gate[i]) * up[i] * w;
                }

                // Down projection into local buffer
                let mut down_out = vec![0.0f32; n_embd];
                expert_down_matvec_accum(&mut down_out, &gate, layer, eid, down_type);
                down_out
            }).collect::<Vec<Vec<f32>>>()
        },
        || {
            // --- Shared expert (runs in parallel with routed) ---
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
            shared_out
        }
    );

    // Sequential sum of routed expert outputs into moe_out
    for down_out in &expert_outputs {
        for i in 0..n_embd {
            moe_out[i] += down_out[i];
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
    forward_one_token_debug(logits, None, None, weights, kv_cache, token, pos);
}

/// Like forward_one_token but optionally accepts an input HC state and returns
/// the final HC state. When `in_hc` is None, the HC state is initialized from
/// the token embedding (appropriate for the first token in a session). When
/// `in_hc` is provided, it is used as the initial HC state (appropriate for
/// continuing from a previous token's output). This matches the C code's
/// `residual_hc` carry-over across tokens.
pub fn forward_one_token_debug(
    logits: &mut [f32],
    in_hc: Option<&[f32]>,
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

    // Initialize HC streams from embedding or from provided state
    let mut cur = vec![0.0f32; n_hc * n_embd];
    if let Some(hc) = in_hc {
        let n = hc.len().min(cur.len());
        cur[..n].copy_from_slice(&hc[..n]);
    } else {
        hc_from_plain_embedding(&mut cur, &plain);
    }

    // Process all layers
    for il in 0..N_LAYER as usize {
        let layer = &weights.layers[il];
        let ratio = kv_cache.layers[il].compress_ratio;

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

        // Q projection (with LoRA) — also compute qr_norm for indexer
        let q_dim = n_head * head_dim;
        let lora_q = N_LORA_Q as usize;
        let mut q = vec![0.0f32; q_dim];
        let mut qr_norm = vec![0.0f32; lora_q];
        layer_q_projection_with_lora(&mut q, &mut qr_norm, &attn_norm, layer);

        // KV projection
        let mut kv = vec![0.0f32; head_dim];
        layer_kv_projection(&mut kv, &attn_norm, layer);

        // Apply RoPE (inverse=false for Q and KV)
        rope_tail_layer_inplace(&mut q, n_head, head_dim, N_ROT as usize, pos, il as u32, false);
        rope_tail_layer_inplace(&mut kv, N_HEAD_KV as usize, head_dim, N_ROT as usize, pos, il as u32, false);

        // FP8 quantize KV
        fp8_kv_quantize_row_inplace(&mut kv, head_dim, N_ROT as usize);

        // Push to raw SWA cache
        kv_cache.push_raw(il, &kv);

        // --- Attention (raw SWA + optional compressed rows) ---
        let mut attn_heads = vec![0.0f32; q_dim];
        let sinks = layer.attn_sinks.as_f32_auto();

        let mut comp_allowed: Option<Vec<bool>> = None;

        if ratio != 0 {
            // Destructure layer cache to allow per-field mutable borrows
            let lc = &mut kv_cache.layers[il];
            let attn_state_kv = &mut lc.attn_state_kv;
            let attn_state_score = &mut lc.attn_state_score;

            // Compressor: try to emit a compressed attention KV row
            let mut comp = vec![0.0f32; head_dim];
            let have_comp = compressor_decode_one(
                &mut comp,
                layer,
                &layer.attn_compressor_kv,
                &layer.attn_compressor_gate,
                &layer.attn_compressor_ape,
                &layer.attn_compressor_norm,
                &attn_norm,
                attn_state_kv,
                attn_state_score,
                N_HEAD_DIM,
                ratio,
                il as u32,
                pos,
            );
            if have_comp {
                let attn_comp_kv = &mut lc.attn_comp_kv;
                let n_comp = &mut lc.n_comp;
                KvCache::push_comp(
                    attn_comp_kv,
                    n_comp,
                    lc.comp_cap,
                    head_dim,
                    &comp,
                );
            }

            // For ratio-4: also run indexer compressor
            if ratio == 4 {
                let indexer_head_dim = N_INDEXER_HEAD_DIM as usize;
                let index_state_kv = &mut lc.index_state_kv;
                let index_state_score = &mut lc.index_state_score;

                let mut index_comp = vec![0.0f32; indexer_head_dim];
                let have_index_comp = compressor_decode_one(
                    &mut index_comp,
                    layer,
                    &layer.indexer_compressor_kv,
                    &layer.indexer_compressor_gate,
                    &layer.indexer_compressor_ape,
                    &layer.indexer_compressor_norm,
                    &attn_norm,
                    index_state_kv,
                    index_state_score,
                    N_INDEXER_HEAD_DIM,
                    ratio,
                    il as u32,
                    pos,
                );
                if have_index_comp {
                    let index_comp_kv = &mut lc.index_comp_kv;
                    let n_index_comp = &mut lc.n_index_comp;
                    KvCache::push_comp(
                        index_comp_kv,
                        n_index_comp,
                        lc.comp_cap,
                        indexer_head_dim,
                        &index_comp,
                    );
                }

                // Indexer: determine which compressed rows are allowed
                comp_allowed = Some(indexer_allowed_decode_one(
                    layer,
                    &attn_norm,
                    &qr_norm,
                    &lc.index_comp_kv,
                    lc.n_index_comp,
                    il as u32,
                    pos,
                ));
            }

            // Reborrow for mixed attention (immutable)
            let n_raw = lc.n_raw;
            let n_comp = lc.n_comp;
            layer_attention_mixed_one(
                &mut attn_heads,
                &q,
                &lc.raw_kv,
                n_raw,
                &lc.attn_comp_kv,
                n_comp,
                comp_allowed.as_deref(),
                &sinks,
            );
        } else {
            // Raw-only attention (ratio == 0)
            let lc = &kv_cache.layers[il];
            layer_attention_rows_one(
                &mut attn_heads,
                &q,
                &lc.raw_kv,
                lc.n_raw,
                &sinks,
            );
        }

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
        if TRACE_HC.load(Ordering::Relaxed) && TRACE_HC_REMAINING.load(Ordering::Relaxed) != 0 {
            TRACE_HC_REMAINING.fetch_sub(1, Ordering::Relaxed);
            print_vec_stats_rms(&format!("blk.{}.hc", il), &cur);
        }
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
        forward_one_token_debug(&mut tmp_logits, None, None, weights, kv_cache, token, i);
        if is_last {
            logits.copy_from_slice(&tmp_logits);
        }
    }
    // Finish prefill states (align compressor windows for decode)
    kv_cache.finish_prefill_states(tokens.len());
}

/// Batched forward prefill: layer-major order with parallel token processing.
///
/// For each layer:
///   Phase 1 (parallel): HC attn pre → RMS norm → Q/KV projections → RoPE → FP8 quant
///   Phase 2 (sequential): push KV to cache + compressor updates (must be ordered)
///   Phase 3 (parallel): attention → grouped output → HC post → FFN MoE
///
/// This achieves both ③ token parallelism and ④ batched weight access.
/// The causal attention mask is preserved by tracking per-token n_raw/n_comp.
pub fn forward_prefill_batched(
    logits: &mut [f32],
    weights: &ModelWeights,
    kv_cache: &mut KvCache,
    tokens: &[i32],
) {
    use rayon::prelude::*;

    let n_tokens = tokens.len();
    if n_tokens == 0 {
        return;
    }
    let n_embd = N_EMBD as usize;
    let n_hc = N_HC as usize;
    let n_head = N_HEAD as usize;
    let head_dim = N_HEAD_DIM as usize;
    let q_dim = n_head * head_dim;
    let lora_q = N_LORA_Q as usize;
    let nlayers = N_LAYER as usize;

    // Embed all tokens and init HC states
    let mut hc_states: Vec<Vec<f32>> = tokens.iter()
        .map(|&token| {
            let mut plain = vec![0.0f32; n_embd];
            embed_token_f16(weights, token, &mut plain);
            let mut cur = vec![0.0f32; n_hc * n_embd];
            hc_from_plain_embedding(&mut cur, &plain);
            cur
        })
        .collect();

    // Process layer by layer
    for il in 0..nlayers {
        let layer = &weights.layers[il];
        let ratio = kv_cache.layers[il].compress_ratio;

        // ================================================================
        // Phase 1: Q/KV projections for all tokens (fully parallel)
        // ================================================================
        struct PerTokenPhase1 {
            q: Vec<f32>,
            qr_norm: Vec<f32>,
            kv: Vec<f32>,
            attn_norm: Vec<f32>,
            residual_hc: Vec<f32>,
            post: [f32; 4],
            comb: [f32; 16],
        }

        let pt1: Vec<PerTokenPhase1> = (0..n_tokens).into_par_iter().map(|i| {
            let cur = &hc_states[i];

            // HC attn pre
            let mut attn_cur = vec![0.0f32; n_embd];
            let mut residual_hc = vec![0.0f32; n_hc * n_embd];
            let mut post = [0.0f32; 4];
            let mut comb = [0.0f32; 16];
            hc_attn_pre(&mut attn_cur, &mut residual_hc, &mut post, &mut comb, cur, layer);

            // RMS Norm
            let mut attn_norm = vec![0.0f32; n_embd];
            let norm_weight = layer.attn_norm.as_f32();
            rms_norm_weighted(&mut attn_norm, &attn_cur, norm_weight, n_embd, RMS_EPS);

            // Q projection + LoRA
            let mut q = vec![0.0f32; q_dim];
            let mut qr_norm = vec![0.0f32; lora_q];
            layer_q_projection_with_lora(&mut q, &mut qr_norm, &attn_norm, layer);

            // KV projection
            let mut kv = vec![0.0f32; head_dim];
            layer_kv_projection(&mut kv, &attn_norm, layer);

            // RoPE
            rope_tail_layer_inplace(&mut q, n_head, head_dim, N_ROT as usize, i, il as u32, false);
            rope_tail_layer_inplace(&mut kv, N_HEAD_KV as usize, head_dim, N_ROT as usize, i, il as u32, false);

            // FP8 quantize KV
            fp8_kv_quantize_row_inplace(&mut kv, head_dim, N_ROT as usize);

            PerTokenPhase1 { q, qr_norm, kv, attn_norm, residual_hc, post, comb }
        }).collect();

        // ================================================================
        // Phase 2: Push KV + compressor (sequential, lightweight)
        // Track per-token n_raw/n_comp for causal attention masking
        // ================================================================
        struct PerTokenPhase2 {
            n_raw: u32,
            n_comp: u32,
            comp_allowed: Option<Vec<bool>>,
        }
        let mut pt2: Vec<PerTokenPhase2> = Vec::with_capacity(n_tokens);

        for i in 0..n_tokens {
            let p1 = &pt1[i];

            // Push raw KV
            kv_cache.push_raw(il, &p1.kv);

            let (n_raw_now, mut n_comp_now) = {
                let lc = &kv_cache.layers[il];
                (lc.n_raw, lc.n_comp)
            };
            let mut comp_allowed_now: Option<Vec<bool>> = None;

            if ratio != 0 {
                let lc = &mut kv_cache.layers[il];
                let attn_state_kv = &mut lc.attn_state_kv;
                let attn_state_score = &mut lc.attn_state_score;

                // Compressor decode
                let mut comp = vec![0.0f32; head_dim];
                let have_comp = compressor_decode_one(
                    &mut comp,
                    layer,
                    &layer.attn_compressor_kv,
                    &layer.attn_compressor_gate,
                    &layer.attn_compressor_ape,
                    &layer.attn_compressor_norm,
                    &p1.attn_norm,
                    attn_state_kv,
                    attn_state_score,
                    N_HEAD_DIM,
                    ratio,
                    il as u32,
                    i,
                );
                if have_comp {
                    let attn_comp_kv = &mut lc.attn_comp_kv;
                    let n_comp = &mut lc.n_comp;
                    KvCache::push_comp(
                        attn_comp_kv, n_comp, lc.comp_cap, head_dim, &comp,
                    );
                    n_comp_now = *n_comp;
                }

                // Indexer (ratio == 4)
                if ratio == 4 {
                    let indexer_head_dim = N_INDEXER_HEAD_DIM as usize;
                    let index_state_kv = &mut lc.index_state_kv;
                    let index_state_score = &mut lc.index_state_score;

                    let mut index_comp = vec![0.0f32; indexer_head_dim];
                    let have_index_comp = compressor_decode_one(
                        &mut index_comp,
                        layer,
                        &layer.indexer_compressor_kv,
                        &layer.indexer_compressor_gate,
                        &layer.indexer_compressor_ape,
                        &layer.indexer_compressor_norm,
                        &p1.attn_norm,
                        index_state_kv,
                        index_state_score,
                        N_INDEXER_HEAD_DIM,
                        ratio,
                        il as u32,
                        i,
                    );
                    if have_index_comp {
                        let index_comp_kv = &mut lc.index_comp_kv;
                        let n_index_comp = &mut lc.n_index_comp;
                        KvCache::push_comp(
                            index_comp_kv, n_index_comp, lc.comp_cap,
                            indexer_head_dim, &index_comp,
                        );
                    }

                    // Indexer allowed
                    let lc_ro = &kv_cache.layers[il];
                    comp_allowed_now = Some(indexer_allowed_decode_one(
                        layer,
                        &p1.attn_norm,
                        &p1.qr_norm,
                        &lc_ro.index_comp_kv,
                        lc_ro.n_index_comp,
                        il as u32,
                        i,
                    ));
                }
            }

            pt2.push(PerTokenPhase2 {
                n_raw: n_raw_now,
                n_comp: n_comp_now,
                comp_allowed: comp_allowed_now,
            });
        }

        // ================================================================
        // Phase 3: Attention + output for all tokens (fully parallel)
        // ================================================================
        let sinks = layer.attn_sinks.as_f32_auto();

        let new_hc_states: Vec<Vec<f32>> = (0..n_tokens).into_par_iter().map(|i| {
            let p1 = &pt1[i];
            let p2 = &pt2[i];

            // Attention (causal: only tokens 0..i via n_raw/n_comp)
            let mut attn_heads = vec![0.0f32; q_dim];
            if ratio != 0 {
                let lc = &kv_cache.layers[il];
                layer_attention_mixed_one(
                    &mut attn_heads,
                    &p1.q,
                    &lc.raw_kv,
                    p2.n_raw,
                    &lc.attn_comp_kv,
                    p2.n_comp,
                    p2.comp_allowed.as_deref(),
                    &sinks,
                );
            } else {
                let lc = &kv_cache.layers[il];
                layer_attention_rows_one(
                    &mut attn_heads,
                    &p1.q,
                    &lc.raw_kv,
                    p2.n_raw,
                    &sinks,
                );
            }

            // RoPE deskew (inverse)
            rope_tail_layer_inplace(&mut attn_heads, n_head, head_dim, N_ROT as usize, i, il as u32, true);

            // Grouped output
            let mut attn_out = vec![0.0f32; n_embd];
            layer_grouped_out(&mut attn_out, &attn_heads, layer);

            // HC post
            let mut after_attn_hc = vec![0.0f32; n_hc * n_embd];
            hc_post_one(&mut after_attn_hc, &attn_out, &p1.residual_hc, &p1.post, &p1.comb, n_embd, n_hc);

            // FFN MoE
            let mut after_ffn_hc = vec![0.0f32; n_hc * n_embd];
            layer_ffn_one(&mut after_ffn_hc, &after_attn_hc, layer, il, tokens[i], false);

            after_ffn_hc
        }).collect();

        // Update HC states for next layer
        for i in 0..n_tokens {
            hc_states[i].copy_from_slice(&new_hc_states[i]);
        }
    }

    // Output head for last token
    kv_cache.finish_prefill_states(n_tokens);
    output_logits(logits, &hc_states[n_tokens - 1], weights);
}

// ============================================================================
// Speculative Decoding Infrastructure (P5: Coarse-grained Parallelism)
// ============================================================================
//
// Single-token decode has strict layer-to-layer data dependencies (layer N+1
// needs layer N's FFN output), making intra-token layer pipelining impossible.
// Instead, we use speculative decoding: generate K draft tokens with a fast
// partial model (first N layers), then verify all K drafts in a single full
// forward pass. This achieves K× throughput when drafts are accurate.
//
// Architecture:
//   1. forward_partial: run first DRAFT_LAYERS, output logits (the "draft model")
//   2. Sample K greedy draft tokens from draft logits
//   3. forward_speculative_verify: run full 43 layers on [token, draft0, ...]
//      Compare full model's predicted next-token at each position with draft.
//      Accept prefix of matching tokens; reject from first mismatch.
//
// Expected speedup (with DRAFT_LAYERS=8, DRAFT_TOKENS=3, ~70% acceptance):
//   ~2× tok/s on typical text (3 drafts × 0.7 acceptance rate).

/// Default number of layers used for the draft model.
pub const SPECULATIVE_DRAFT_LAYERS: usize = 8;

/// Default number of draft tokens to generate per speculative step.
pub const SPECULATIVE_DRAFT_TOKENS: usize = 3;

/// Run forward pass through only the first `n_layers` layers, producing
/// intermediate logits. This serves as the "draft model" for speculative
/// decoding — much faster than full 43 layers but lower quality.
///
/// Returns logits [N_VOCAB] from the partial model.
pub fn forward_partial(
    logits: &mut [f32],
    weights: &ModelWeights,
    kv_cache: &mut KvCache,
    token: i32,
    pos: usize,
    n_layers: usize,
) {
    let n_embd = N_EMBD as usize;
    let n_hc = N_HC as usize;
    let n_head = N_HEAD as usize;
    let head_dim = N_HEAD_DIM as usize;
    let nlayers = n_layers.min(N_LAYER as usize);

    // Embed token
    let mut plain = vec![0.0f32; n_embd];
    embed_token_f16(weights, token, &mut plain);

    // Initialize HC streams
    let mut cur = vec![0.0f32; n_hc * n_embd];
    hc_from_plain_embedding(&mut cur, &plain);

    // Process layers 0..nlayers
    for il in 0..nlayers {
        let layer = &weights.layers[il];
        let ratio = kv_cache.layers[il].compress_ratio;

        // --- Attention sublayer ---
        let mut attn_cur = vec![0.0f32; n_embd];
        let mut residual_hc = vec![0.0f32; n_hc * n_embd];
        let mut post = [0.0f32; 4];
        let mut comb = [0.0f32; 16];
        hc_attn_pre(&mut attn_cur, &mut residual_hc, &mut post, &mut comb, &cur, layer);

        let mut attn_norm = vec![0.0f32; n_embd];
        let norm_weight = layer.attn_norm.as_f32();
        rms_norm_weighted(&mut attn_norm, &attn_cur, norm_weight, n_embd, RMS_EPS);

        let q_dim = n_head * head_dim;
        let lora_q = N_LORA_Q as usize;
        let mut q = vec![0.0f32; q_dim];
        let mut qr_norm = vec![0.0f32; lora_q];
        layer_q_projection_with_lora(&mut q, &mut qr_norm, &attn_norm, layer);

        let mut kv = vec![0.0f32; head_dim];
        layer_kv_projection(&mut kv, &attn_norm, layer);

        rope_tail_layer_inplace(&mut q, n_head, head_dim, N_ROT as usize, pos, il as u32, false);
        rope_tail_layer_inplace(&mut kv, N_HEAD_KV as usize, head_dim, N_ROT as usize, pos, il as u32, false);
        fp8_kv_quantize_row_inplace(&mut kv, head_dim, N_ROT as usize);
        kv_cache.push_raw(il, &kv);

        let mut attn_heads = vec![0.0f32; q_dim];
        let sinks = layer.attn_sinks.as_f32_auto();

        if ratio != 0 {
            let lc = &mut kv_cache.layers[il];
            let mut comp = vec![0.0f32; head_dim];
            let have_comp = compressor_decode_one(
                &mut comp, layer,
                &layer.attn_compressor_kv, &layer.attn_compressor_gate,
                &layer.attn_compressor_ape, &layer.attn_compressor_norm,
                &attn_norm,
                &mut lc.attn_state_kv, &mut lc.attn_state_score,
                N_HEAD_DIM, ratio, il as u32, pos,
            );
            if have_comp {
                KvCache::push_comp(&mut lc.attn_comp_kv, &mut lc.n_comp, lc.comp_cap, head_dim, &comp);
            }
            if ratio == 4 {
                let indexer_head_dim = N_INDEXER_HEAD_DIM as usize;
                let mut index_comp = vec![0.0f32; indexer_head_dim];
                let have_index_comp = compressor_decode_one(
                    &mut index_comp, layer,
                    &layer.indexer_compressor_kv, &layer.indexer_compressor_gate,
                    &layer.indexer_compressor_ape, &layer.indexer_compressor_norm,
                    &attn_norm,
                    &mut lc.index_state_kv, &mut lc.index_state_score,
                    N_INDEXER_HEAD_DIM, ratio, il as u32, pos,
                );
                if have_index_comp {
                    KvCache::push_comp(&mut lc.index_comp_kv, &mut lc.n_index_comp,
                        lc.comp_cap, indexer_head_dim, &index_comp);
                }
            }
            let n_raw = lc.n_raw;
            let n_comp = lc.n_comp;
            let comp_allowed: Option<Vec<bool>> = if ratio == 4 {
                Some(indexer_allowed_decode_one(layer, &attn_norm, &qr_norm,
                    &lc.index_comp_kv, lc.n_index_comp, il as u32, pos))
            } else { None };
            layer_attention_mixed_one(&mut attn_heads, &q, &lc.raw_kv, n_raw,
                &lc.attn_comp_kv, n_comp, comp_allowed.as_deref(), &sinks);
        } else {
            let lc = &kv_cache.layers[il];
            layer_attention_rows_one(&mut attn_heads, &q, &lc.raw_kv, lc.n_raw, &sinks);
        }

        rope_tail_layer_inplace(&mut attn_heads, n_head, head_dim, N_ROT as usize, pos, il as u32, true);
        let mut attn_out = vec![0.0f32; n_embd];
        layer_grouped_out(&mut attn_out, &attn_heads, layer);

        let mut after_attn_hc = vec![0.0f32; n_hc * n_embd];
        hc_post_one(&mut after_attn_hc, &attn_out, &residual_hc, &post, &comb, n_embd, n_hc);

        let mut after_ffn_hc = vec![0.0f32; n_hc * n_embd];
        layer_ffn_one(&mut after_ffn_hc, &after_attn_hc, layer, il, token, false);
        cur.copy_from_slice(&after_ffn_hc);
    }

    // Output head on intermediate HC state
    output_logits(logits, &cur, weights);
}

/// Verify draft tokens against full model in a single batch forward pass.
///
/// Runs the full 43-layer model sequentially on [current_token, draft_tokens...].
/// At each position, compares the full model's argmax prediction with the draft
/// token. Accepts consecutive matches, stops at first mismatch.
///
/// Returns (num_accepted, final_logits) where:
///   num_accepted: count of consecutive matching draft tokens (0..K)
///   final_logits: logits at the last accepted position for next-token sampling
///
/// The KV cache is updated for the current token + all accepted drafts.
/// Rejected draft tokens are NOT fed to the model — the KV cache reflects
/// the last accepted position, ready for the next decode step.
///
/// NOTE: For production use with batched verification (processing all drafts
/// in a single forward call for speed), the caller should snapshot/restore
/// the KV cache around the batch call. The sequential approach here is
/// correct but doesn't achieve the full parallelism benefit. See the
/// speculative decode section in the architecture docs for the batched path.
pub fn forward_speculative_verify(
    weights: &ModelWeights,
    kv_cache: &mut KvCache,
    current_token: i32,
    current_pos: usize,
    draft_tokens: &[i32],
) -> (usize, Vec<f32>) {
    let n_vocab = N_VOCAB as usize;

    // Process current token first
    let mut logits = vec![0.0f32; n_vocab];
    let hc_dim = (N_HC * N_EMBD) as usize;
    let mut hc = vec![0.0f32; hc_dim];
    forward_one_token_debug(&mut logits, None, Some(&mut hc), weights, kv_cache, current_token, current_pos);

    let mut accepted = 0usize;

    // For each draft token, check if full model agrees
    for (i, &draft) in draft_tokens.iter().enumerate() {
        // Find argmax of current logits
        let mut best_token = 0i32;
        let mut best_val = f32::NEG_INFINITY;
        for t in 0..n_vocab {
            if logits[t] > best_val {
                best_val = logits[t];
                best_token = t as i32;
            }
        }

        if best_token == draft {
            accepted += 1;
            // Feed the accepted token and get next logits
            let mut next_hc = vec![0.0f32; hc_dim];
            forward_one_token_debug(&mut logits, Some(&hc), Some(&mut next_hc), weights, kv_cache, draft, current_pos + 1 + i);
            hc.copy_from_slice(&next_hc);
        } else {
            // Mismatch: stop. KV cache is at position (current_pos + accepted).
            // logits are from the last accepted position — ready for next sampling.
            break;
        }
    }

    (accepted, logits)
}

// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{LayerWeights, Tensor};

    /// Create a synthetic F16 tensor with all elements = f16(1.0) = 0x3c00.
    unsafe fn make_f16_ones_tensor(len: usize) -> (Vec<u16>, Tensor) {
        let data: Vec<u16> = vec![0x3c00u16; len];
        let ptr = data.as_ptr() as *const u8;
        let tensor = Tensor {
            name: String::new(),
            tensor_type: 1, // F16
            data: ptr,
            elements: len as u64,
            bytes: (len * 2) as u64,
            dims: vec![],
        };
        (data, tensor)
    }

    /// Test indexer_allowed_decode_one against C reference output.
    /// Uses the same inputs as tools/test_indexer.c.
    #[test]
    #[cfg(feature = "test-dimensions")]
    fn test_indexer_allowed_decode_one_vs_c() {
        unsafe {
            // Constants matching C test (test-dimensions mode)
            // Use actual constants so the test works in both default and test-dimensions modes
            let n_lora_q = N_LORA_Q as usize;
            let n_embd = N_EMBD as usize;
            let n_indexer_head = N_INDEXER_HEAD as usize;
            let indexer_head_dim = N_INDEXER_HEAD_DIM as usize;
            let index_q_dim = n_indexer_head * indexer_head_dim; // 32
            // n_comp must exceed N_INDEXER_TOP_K to exercise the top-k selection path
            let n_comp = (N_INDEXER_TOP_K + 4) as u32;
            let top_k = N_INDEXER_TOP_K as usize;

            // Create synthetic F16 tensors (all ones)
            let (_q_b_buf, q_b_tensor) = make_f16_ones_tensor(n_lora_q * index_q_dim);
            let (_proj_buf, proj_tensor) = make_f16_ones_tensor(n_embd * n_indexer_head);

            // Create empty tensors for unused fields
            let empty = Tensor {
                name: String::new(),
                tensor_type: 0,
                data: std::ptr::null(),
                elements: 0,
                bytes: 0,
                dims: vec![],
            };

            let layer = LayerWeights {
                hc_attn_fn: empty.clone(),
                hc_attn_scale: empty.clone(),
                hc_attn_base: empty.clone(),
                attn_norm: empty.clone(),
                attn_q_a: empty.clone(),
                attn_q_a_norm: empty.clone(),
                attn_q_b: empty.clone(),
                attn_kv: empty.clone(),
                attn_kv_a_norm: empty.clone(),
                attn_sinks: empty.clone(),
                attn_output_a: empty.clone(),
                attn_output_b: empty.clone(),
                attn_compressor_ape: empty.clone(),
                attn_compressor_kv: empty.clone(),
                attn_compressor_gate: empty.clone(),
                attn_compressor_norm: empty.clone(),
                indexer_attn_q_b: q_b_tensor,
                indexer_proj: proj_tensor,
                indexer_compressor_ape: empty.clone(),
                indexer_compressor_kv: empty.clone(),
                indexer_compressor_gate: empty.clone(),
                indexer_compressor_norm: empty.clone(),
                hc_ffn_fn: empty.clone(),
                hc_ffn_scale: empty.clone(),
                hc_ffn_base: empty.clone(),
                ffn_norm: empty.clone(),
                ffn_gate_tid2eid: empty.clone(),
                ffn_gate_inp: empty.clone(),
                ffn_exp_probs_b: empty.clone(),
                ffn_gate_exps: empty.clone(),
                ffn_up_exps: empty.clone(),
                ffn_down_exps: empty.clone(),
                ffn_gate_shexp: empty.clone(),
                ffn_up_shexp: empty.clone(),
                ffn_down_shexp: empty.clone(),
            };

            // Inputs matching C test
            let qr_norm: Vec<f32> = vec![1.0f32; n_lora_q];
            let attn_norm: Vec<f32> = vec![1.0f32; n_embd];

            // Compressed KV rows: row c has values [c*2; 8]
            let mut index_comp_kv = vec![0.0f32; n_comp as usize * indexer_head_dim];
            for c in 0..n_comp as usize {
                for d in 0..indexer_head_dim {
                    index_comp_kv[c * indexer_head_dim + d] = (c * 2) as f32;
                }
            }

            let allowed = indexer_allowed_decode_one(
                &layer,
                &attn_norm,
                &qr_norm,
                &index_comp_kv,
                n_comp,
                2,  // il = 2 (ratio-4 layer)
                0,  // pos = 0
            );

            // C output pattern: scores increase with row index (row c has value c*2).
            // The top top_k rows are allowed; the bottom (n_comp - top_k) are not.
            assert_eq!(allowed.len(), n_comp as usize);
            let cutoff = n_comp as usize - top_k;
            for c in 0..cutoff {
                assert!(!allowed[c], "row {} should not be allowed (below cutoff)", c);
            }
            for c in cutoff..n_comp as usize {
                assert!(allowed[c], "row {} should be allowed (top {})", c, top_k);
            }

            // Count: exactly top_k allowed
            let n_allowed = allowed.iter().filter(|&&a| a).count();
            assert_eq!(n_allowed, top_k, "exactly {} rows should be allowed", top_k);
        }
    }

    /// Test indexer with n_comp <= top_k: all rows should be allowed.
    #[test]
    fn test_indexer_all_short() {
        unsafe {
            // Use actual constants so the test works in both default and test-dimensions modes
            let n_lora_q = N_LORA_Q as usize;
            let n_embd = N_EMBD as usize;
            let n_indexer_head = N_INDEXER_HEAD as usize;
            let indexer_head_dim = N_INDEXER_HEAD_DIM as usize;
            let index_q_dim = n_indexer_head * indexer_head_dim;

            let (_q_b_buf, q_b_tensor) = make_f16_ones_tensor(n_lora_q * index_q_dim);
            let (_proj_buf, proj_tensor) = make_f16_ones_tensor(n_embd * n_indexer_head);

            let empty = Tensor {
                name: String::new(),
                tensor_type: 0,
                data: std::ptr::null(),
                elements: 0,
                bytes: 0,
                dims: vec![],
            };

            let layer = LayerWeights {
                hc_attn_fn: empty.clone(),
                hc_attn_scale: empty.clone(),
                hc_attn_base: empty.clone(),
                attn_norm: empty.clone(),
                attn_q_a: empty.clone(),
                attn_q_a_norm: empty.clone(),
                attn_q_b: empty.clone(),
                attn_kv: empty.clone(),
                attn_kv_a_norm: empty.clone(),
                attn_sinks: empty.clone(),
                attn_output_a: empty.clone(),
                attn_output_b: empty.clone(),
                attn_compressor_ape: empty.clone(),
                attn_compressor_kv: empty.clone(),
                attn_compressor_gate: empty.clone(),
                attn_compressor_norm: empty.clone(),
                indexer_attn_q_b: q_b_tensor,
                indexer_proj: proj_tensor,
                indexer_compressor_ape: empty.clone(),
                indexer_compressor_kv: empty.clone(),
                indexer_compressor_gate: empty.clone(),
                indexer_compressor_norm: empty.clone(),
                hc_ffn_fn: empty.clone(),
                hc_ffn_scale: empty.clone(),
                hc_ffn_base: empty.clone(),
                ffn_norm: empty.clone(),
                ffn_gate_tid2eid: empty.clone(),
                ffn_gate_inp: empty.clone(),
                ffn_exp_probs_b: empty.clone(),
                ffn_gate_exps: empty.clone(),
                ffn_up_exps: empty.clone(),
                ffn_down_exps: empty.clone(),
                ffn_gate_shexp: empty.clone(),
                ffn_up_shexp: empty.clone(),
                ffn_down_shexp: empty.clone(),
            };

            let qr_norm: Vec<f32> = vec![1.0f32; n_lora_q];
            let attn_norm: Vec<f32> = vec![1.0f32; n_embd];
            let index_comp_kv = vec![0.0f32; 8]; // 1 row
            let n_comp = 1u32;

            let allowed = indexer_allowed_decode_one(
                &layer, &attn_norm, &qr_norm, &index_comp_kv, n_comp, 2, 0,
            );

            // With 1 row and top_k=2, all should be allowed
            assert_eq!(allowed.len(), 1);
            assert!(allowed[0]);
        }
    }

    /// Test indexer with n_comp == 0: returns empty vec, no rows to consider.
    #[test]
    fn test_indexer_empty() {
        unsafe {
            // Use actual constants so the test works in both default and test-dimensions modes
            let n_lora_q = N_LORA_Q as usize;
            let n_embd = N_EMBD as usize;
            let n_indexer_head = N_INDEXER_HEAD as usize;
            let indexer_head_dim = N_INDEXER_HEAD_DIM as usize;
            let index_q_dim = n_indexer_head * indexer_head_dim;

            let (_q_b_buf, q_b_tensor) = make_f16_ones_tensor(n_lora_q * index_q_dim);
            let (_proj_buf, proj_tensor) = make_f16_ones_tensor(n_embd * n_indexer_head);

            let empty = Tensor {
                name: String::new(),
                tensor_type: 0,
                data: std::ptr::null(),
                elements: 0,
                bytes: 0,
                dims: vec![],
            };

            let layer = LayerWeights {
                hc_attn_fn: empty.clone(),
                hc_attn_scale: empty.clone(),
                hc_attn_base: empty.clone(),
                attn_norm: empty.clone(),
                attn_q_a: empty.clone(),
                attn_q_a_norm: empty.clone(),
                attn_q_b: empty.clone(),
                attn_kv: empty.clone(),
                attn_kv_a_norm: empty.clone(),
                attn_sinks: empty.clone(),
                attn_output_a: empty.clone(),
                attn_output_b: empty.clone(),
                attn_compressor_ape: empty.clone(),
                attn_compressor_kv: empty.clone(),
                attn_compressor_gate: empty.clone(),
                attn_compressor_norm: empty.clone(),
                indexer_attn_q_b: q_b_tensor,
                indexer_proj: proj_tensor,
                indexer_compressor_ape: empty.clone(),
                indexer_compressor_kv: empty.clone(),
                indexer_compressor_gate: empty.clone(),
                indexer_compressor_norm: empty.clone(),
                hc_ffn_fn: empty.clone(),
                hc_ffn_scale: empty.clone(),
                hc_ffn_base: empty.clone(),
                ffn_norm: empty.clone(),
                ffn_gate_tid2eid: empty.clone(),
                ffn_gate_inp: empty.clone(),
                ffn_exp_probs_b: empty.clone(),
                ffn_gate_exps: empty.clone(),
                ffn_up_exps: empty.clone(),
                ffn_down_exps: empty.clone(),
                ffn_gate_shexp: empty.clone(),
                ffn_up_shexp: empty.clone(),
                ffn_down_shexp: empty.clone(),
            };

            let qr_norm: Vec<f32> = vec![1.0f32; n_lora_q];
            let attn_norm: Vec<f32> = vec![1.0f32; n_embd];
            let index_comp_kv: Vec<f32> = vec![];
            let n_comp = 0u32;

            let allowed = indexer_allowed_decode_one(
                &layer, &attn_norm, &qr_norm, &index_comp_kv, n_comp, 2, 0,
            );

            assert_eq!(allowed.len(), 0);
        }
    }

    /// Test that batched prefill produces identical logits and KV cache state
    /// as sequential prefill, verifying causal masking and parallel correctness.
    #[test]
    fn test_batched_prefill_matches_sequential() {
        let model = crate::model::build_test_model_with_value(0.5);
        let weights = &model.weights;
        let n_vocab = N_VOCAB as usize;
        let ctx_size = 32usize;

        // Create two independent KV caches
        let mut cache_seq = KvCache::new(ctx_size);
        let mut cache_batch = KvCache::new(ctx_size);

        // Prompt: 5 tokens (exercises causal masking with 5 positions)
        let tokens: Vec<i32> = vec![0, 1, 2, 3, 4];

        // Sequential prefill
        let mut logits_seq = vec![0.0f32; n_vocab];
        forward_prefill(&mut logits_seq, weights, &mut cache_seq, &tokens);

        // Batched prefill
        let mut logits_batch = vec![0.0f32; n_vocab];
        forward_prefill_batched(&mut logits_batch, weights, &mut cache_batch, &tokens);

        // Compare logits
        let mut max_diff = 0.0f32;
        let mut max_diff_idx = 0usize;
        for i in 0..n_vocab {
            let diff = (logits_seq[i] - logits_batch[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                max_diff_idx = i;
            }
        }
        assert!(max_diff < 1e-3,
            "logit mismatch: max_diff={:.6} at idx={}, seq={:.6}, batch={:.6}",
            max_diff, max_diff_idx, logits_seq[max_diff_idx], logits_batch[max_diff_idx]);

        // Compare KV cache state layer by layer
        for il in 0..N_LAYER as usize {
            let l_seq = &cache_seq.layers[il];
            let l_batch = &cache_batch.layers[il];
            assert_eq!(l_seq.n_raw, l_batch.n_raw,
                "layer {} n_raw: seq={}, batch={}", il, l_seq.n_raw, l_batch.n_raw);
            assert_eq!(l_seq.n_comp, l_batch.n_comp,
                "layer {} n_comp: seq={}, batch={}", il, l_seq.n_comp, l_batch.n_comp);

            // Compare raw KV entries
            let n_raw = l_seq.n_raw as usize;
            let head_dim = N_HEAD_DIM as usize;
            let raw_len = n_raw * head_dim;
            let mut kv_max_diff = 0.0f32;
            for i in 0..raw_len {
                let diff = (l_seq.raw_kv[i] - l_batch.raw_kv[i]).abs();
                if diff > kv_max_diff { kv_max_diff = diff; }
            }
            assert!(kv_max_diff < 1e-3,
                "layer {} raw_kv mismatch: max_diff={:.6}", il, kv_max_diff);

            // Compare compressed KV entries if any
            if l_seq.n_comp > 0 {
                let comp_len = l_seq.n_comp as usize * head_dim;
                let mut comp_max_diff = 0.0f32;
                for i in 0..comp_len {
                    let diff = (l_seq.attn_comp_kv[i] - l_batch.attn_comp_kv[i]).abs();
                    if diff > comp_max_diff { comp_max_diff = diff; }
                }
                assert!(comp_max_diff < 1e-3,
                    "layer {} comp_kv mismatch: max_diff={:.6}", il, comp_max_diff);
            }
        }
    }

    /// Test that batched prefill matches sequential prefill with a single token
    /// (edge case where there's only one token to prefill).
    #[test]
    fn test_batched_prefill_single_token() {
        let model = crate::model::build_test_model_with_value(0.5);
        let weights = &model.weights;
        let n_vocab = N_VOCAB as usize;
        let ctx_size = 32usize;

        let mut cache_seq = KvCache::new(ctx_size);
        let mut cache_batch = KvCache::new(ctx_size);

        // Single token
        let tokens: Vec<i32> = vec![7];

        let mut logits_seq = vec![0.0f32; n_vocab];
        forward_prefill(&mut logits_seq, weights, &mut cache_seq, &tokens);

        let mut logits_batch = vec![0.0f32; n_vocab];
        forward_prefill_batched(&mut logits_batch, weights, &mut cache_batch, &tokens);

        let mut max_diff = 0.0f32;
        for i in 0..n_vocab {
            let diff = (logits_seq[i] - logits_batch[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }
        assert!(max_diff < 1e-3,
            "single-token logit mismatch: max_diff={:.6}", max_diff);
    }
}
