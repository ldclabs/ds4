// Model architecture definition and tensor binding.
// Binds GGUF tensors to the fixed DeepSeek V4 Flash model layout.

use crate::f16_to_f32;
use crate::gguf::GgufModel;
use anyhow::{bail, Result};

/// A bound tensor reference into the mmap'd GGUF data.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub tensor_type: u32,
    pub data: *const u8,
    pub elements: u64,
    pub bytes: u64,
    pub dims: Vec<u64>,
}

// Safe because the data points into an mmap that lives as long as GgufModel.
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Tensor {
    /// Check if tensor has data (non-null, non-zero bytes).
    pub fn has_data(&self) -> bool {
        !self.data.is_null() && self.bytes > 0
    }

    /// Get tensor data as f32 slice (assumes F32 type).
    pub fn as_f32(&self) -> &[f32] {
        let len = self.bytes as usize / 4;
        unsafe { std::slice::from_raw_parts(self.data as *const f32, len) }
    }

    /// Get tensor data as f32, auto-converting from F16 if the tensor is F16 type.
    /// Returns an owned Vec because F16 conversion requires allocation.
    pub fn as_f32_auto(&self) -> Vec<f32> {
        if self.tensor_type == 1 {
            let len = self.bytes as usize / 2;
            let f16_data = unsafe { std::slice::from_raw_parts(self.data as *const u16, len) };
            f16_data.iter().map(|&h| f16_to_f32(h)).collect()
        } else {
            self.as_f32().to_vec()
        }
    }

    /// Get tensor data as u8 slice.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data, self.bytes as usize) }
    }

    /// Get tensor data as f16 (u16) slice.
    pub fn as_f16(&self) -> &[u16] {
        let len = self.bytes as usize / 2;
        unsafe { std::slice::from_raw_parts(self.data as *const u16, len) }
    }

    /// Get tensor data as i32 slice.
    pub fn as_i32(&self) -> &[i32] {
        let len = self.bytes as usize / 4;
        unsafe { std::slice::from_raw_parts(self.data as *const i32, len) }
    }
}

/// Per-layer weight bindings.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub hc_attn_fn: Tensor,
    pub hc_attn_scale: Tensor,
    pub hc_attn_base: Tensor,
    pub attn_norm: Tensor,
    pub attn_q_a: Tensor,
    pub attn_q_a_norm: Tensor,
    pub attn_q_b: Tensor,
    pub attn_kv: Tensor,
    pub attn_kv_a_norm: Tensor,
    pub attn_sinks: Tensor,
    pub attn_output_a: Tensor,
    pub attn_output_b: Tensor,
    pub attn_compressor_ape: Tensor,
    pub attn_compressor_kv: Tensor,
    pub attn_compressor_gate: Tensor,
    pub attn_compressor_norm: Tensor,
    pub indexer_attn_q_b: Tensor,
    pub indexer_proj: Tensor,
    pub indexer_compressor_ape: Tensor,
    pub indexer_compressor_kv: Tensor,
    pub indexer_compressor_gate: Tensor,
    pub indexer_compressor_norm: Tensor,
    pub hc_ffn_fn: Tensor,
    pub hc_ffn_scale: Tensor,
    pub hc_ffn_base: Tensor,
    pub ffn_norm: Tensor,
    pub ffn_gate_tid2eid: Tensor,
    pub ffn_gate_inp: Tensor,
    pub ffn_exp_probs_b: Tensor,
    pub ffn_gate_exps: Tensor,
    pub ffn_up_exps: Tensor,
    pub ffn_down_exps: Tensor,
    pub ffn_gate_shexp: Tensor,
    pub ffn_up_shexp: Tensor,
    pub ffn_down_shexp: Tensor,
}

/// All model weights.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub token_embd: Tensor,
    pub output_hc_base: Tensor,
    pub output_hc_fn: Tensor,
    pub output_hc_scale: Tensor,
    pub output_norm: Tensor,
    pub output: Tensor,
    pub layers: Vec<LayerWeights>,
}

/// Helper to bind a GGUF tensor by name, returning a Tensor.
fn bind_tensor(model: &GgufModel, name: &str) -> Result<Tensor> {
    let t = model.tensor(name)
        .ok_or_else(|| anyhow::anyhow!("missing tensor: {}", name))?;
    let data = model.tensor_data(name)
        .ok_or_else(|| anyhow::anyhow!("no data for tensor: {}", name))?;
    Ok(Tensor {
        name: name.to_string(),
        tensor_type: t.tensor_type,
        data: data.as_ptr(),
        elements: t.elements,
        bytes: t.bytes,
        dims: t.dims.clone(),
    })
}

/// Bind a tensor that may not exist in the model (returns a zeroed tensor if missing).
fn bind_tensor_optional(model: &GgufModel, name: &str) -> Tensor {
    match model.tensor(name) {
        Some(t) => {
            let data = model.tensor_data(name).unwrap_or(&[]);
            Tensor {
                name: name.to_string(),
                tensor_type: t.tensor_type,
                data: data.as_ptr(),
                elements: t.elements,
                bytes: t.bytes,
                dims: t.dims.clone(),
            }
        }
        None => Tensor {
            name: name.to_string(),
            tensor_type: 0,
            data: std::ptr::null(),
            elements: 0,
            bytes: 0,
            dims: vec![],
        },
    }
}

/// Validate the model configuration against fixed DS4 architecture.
pub fn validate_model(model: &GgufModel) -> Result<()> {
    validate_model_inner(model, false)
}

/// Validate the model, optionally allowing non-standard dimensions.
pub fn validate_model_lenient(model: &GgufModel) -> Result<()> {
    validate_model_inner(model, true)
}

fn validate_model_inner(model: &GgufModel, lenient: bool) -> Result<()> {
    let n_layer = model.get_u32("ds4.n_layer")
        .unwrap_or(model.get_u32("llama.block_count").unwrap_or(0));
    if !lenient && n_layer != crate::N_LAYER {
        bail!("expected {} layers, got {}", crate::N_LAYER, n_layer);
    }
    if lenient && n_layer == 0 {
        bail!("need at least 1 layer, got {}", n_layer);
    }

    let n_embd = model.get_u32("ds4.n_embd")
        .unwrap_or(model.get_u32("llama.embedding_length").unwrap_or(0));
    if !lenient && n_embd != crate::N_EMBD {
        bail!("expected embedding dim {}, got {}", crate::N_EMBD, n_embd);
    }
    if lenient && n_embd == 0 {
        bail!("embedding dimension is zero");
    }

    let vocab_size = model.get_u32("ds4.vocab_size")
        .unwrap_or(model.get_u32("llama.vocab_size").unwrap_or(0));
    if !lenient && vocab_size != crate::N_VOCAB {
        bail!("expected vocab size {}, got {}", crate::N_VOCAB, vocab_size);
    }
    if lenient && vocab_size == 0 {
        bail!("vocab size is zero");
    }

    let n_head = model.get_u32("ds4.attn_n_head")
        .unwrap_or(model.get_u32("llama.attention.head_count").unwrap_or(0));
    if !lenient && n_head != crate::N_HEAD {
        bail!("expected {} heads, got {}", crate::N_HEAD, n_head);
    }

    let n_head_kv = model.get_u32("ds4.attn_n_head_kv")
        .unwrap_or(model.get_u32("llama.attention.head_count_kv").unwrap_or(0));
    if !lenient && n_head_kv != crate::N_HEAD_KV {
        bail!("expected {} KV head, got {}", crate::N_HEAD_KV, n_head_kv);
    }

    Ok(())
}

/// Bind all model weights from GGUF tensors (strict validation).
pub fn bind_weights(model: &GgufModel) -> Result<ModelWeights> {
    validate_model(model)?;
    bind_weights_inner(model)
}

/// Bind weights without strict validation (for test models with custom dims).
pub fn bind_weights_unchecked(model: &GgufModel) -> Result<ModelWeights> {
    bind_weights_inner(model)
}

fn bind_weights_inner(model: &GgufModel) -> Result<ModelWeights> {
    let token_embd = bind_tensor(model, "token_embd.weight")?;
    let output_hc_base = bind_tensor(model, "output_hc_base.weight")?;
    let output_hc_fn = bind_tensor(model, "output_hc_fn.weight")?;
    let output_hc_scale = bind_tensor(model, "output_hc_scale.weight")?;
    let output_norm = bind_tensor(model, "output_norm.weight")?;
    let output = bind_tensor(model, "output.weight")?;

    // Detect layer count from GGUF metadata
    let n_layer = model.get_u32("ds4.n_layer")
        .unwrap_or(model.get_u32("llama.block_count").unwrap_or(crate::N_LAYER));

    let mut layers = Vec::with_capacity(n_layer as usize);
    for i in 0..n_layer {
        let prefix = format!("blk.{}.", i);
        layers.push(LayerWeights {
            hc_attn_fn:       bind_tensor(model, &format!("{}hc_attn_fn.weight", prefix))?,
            hc_attn_scale:    bind_tensor(model, &format!("{}hc_attn_scale.weight", prefix))?,
            hc_attn_base:     bind_tensor(model, &format!("{}hc_attn_base.weight", prefix))?,
            attn_norm:        bind_tensor(model, &format!("{}attn_norm.weight", prefix))?,
            attn_q_a:         bind_tensor(model, &format!("{}attn_q_a.weight", prefix))?,
            attn_q_a_norm:    bind_tensor(model, &format!("{}attn_q_a_norm.weight", prefix))?,
            attn_q_b:         bind_tensor(model, &format!("{}attn_q_b.weight", prefix))?,
            attn_kv:          bind_tensor(model, &format!("{}attn_kv.weight", prefix))?,
            attn_kv_a_norm:   bind_tensor(model, &format!("{}attn_kv_a_norm.weight", prefix))?,
            attn_sinks:       bind_tensor(model, &format!("{}attn_sinks.weight", prefix))?,
            attn_output_a:    bind_tensor(model, &format!("{}attn_output_a.weight", prefix))?,
            attn_output_b:    bind_tensor(model, &format!("{}attn_output_b.weight", prefix))?,
            attn_compressor_ape:  bind_tensor_optional(model, &format!("{}attn_compressor_ape.weight", prefix)),
            attn_compressor_kv:   bind_tensor_optional(model, &format!("{}attn_compressor_kv.weight", prefix)),
            attn_compressor_gate: bind_tensor_optional(model, &format!("{}attn_compressor_gate.weight", prefix)),
            attn_compressor_norm: bind_tensor_optional(model, &format!("{}attn_compressor_norm.weight", prefix)),
            indexer_attn_q_b:     bind_tensor_optional(model, &format!("{}indexer.attn_q_b.weight", prefix)),
            indexer_proj:         bind_tensor_optional(model, &format!("{}indexer.proj.weight", prefix)),
            indexer_compressor_ape:   bind_tensor_optional(model, &format!("{}indexer_compressor_ape.weight", prefix)),
            indexer_compressor_kv:    bind_tensor_optional(model, &format!("{}indexer_compressor_kv.weight", prefix)),
            indexer_compressor_gate:  bind_tensor_optional(model, &format!("{}indexer_compressor_gate.weight", prefix)),
            indexer_compressor_norm:  bind_tensor_optional(model, &format!("{}indexer_compressor_norm.weight", prefix)),
            hc_ffn_fn:           bind_tensor(model, &format!("{}hc_ffn_fn.weight", prefix))?,
            hc_ffn_scale:        bind_tensor(model, &format!("{}hc_ffn_scale.weight", prefix))?,
            hc_ffn_base:         bind_tensor(model, &format!("{}hc_ffn_base.weight", prefix))?,
            ffn_norm:            bind_tensor(model, &format!("{}ffn_norm.weight", prefix))?,
            ffn_gate_tid2eid:    bind_tensor_optional(model, &format!("{}ffn_gate_tid2eid.weight", prefix)),
            ffn_gate_inp:        bind_tensor(model, &format!("{}ffn_gate_inp.weight", prefix))?,
            ffn_exp_probs_b:     bind_tensor_optional(model, &format!("{}exp_probs_b.bias", prefix)),
            ffn_gate_exps:       bind_tensor(model, &format!("{}ffn_gate_exps.weight", prefix))?,
            ffn_up_exps:         bind_tensor(model, &format!("{}ffn_up_exps.weight", prefix))?,
            ffn_down_exps:       bind_tensor(model, &format!("{}ffn_down_exps.weight", prefix))?,
            ffn_gate_shexp:      bind_tensor(model, &format!("{}ffn_gate_shexp.weight", prefix))?,
            ffn_up_shexp:        bind_tensor(model, &format!("{}ffn_up_shexp.weight", prefix))?,
            ffn_down_shexp:      bind_tensor(model, &format!("{}ffn_down_shexp.weight", prefix))?,
        });
    }

    Ok(ModelWeights {
        token_embd,
        output_hc_base,
        output_hc_fn,
        output_hc_scale,
        output_norm,
        output,
        layers,
    })
}

// ============================================================================
// In-memory test model construction (does not require GGUF file)
// ============================================================================

/// Owns all tensor data buffers for a ModelWeights. Must outlive the weights.
pub struct OwnedModelWeights {
    _buffers: Vec<Vec<u8>>,
    pub weights: ModelWeights,
}

/// Build test ModelWeights with all weights set to f16(1.0) / f32(1.0).
pub fn build_test_model() -> OwnedModelWeights {
    build_test_model_with_value(1.0f32)
}

/// Build test ModelWeights with a custom fill value for all weights.
pub fn build_test_model_with_value(fill: f32) -> OwnedModelWeights {
    let fill_f16 = crate::f32_to_f16(fill);
    let n_layer = crate::N_LAYER as usize;
    let n_embd = crate::N_EMBD as usize;
    let n_vocab = crate::N_VOCAB as usize;
    let n_head = crate::N_HEAD as usize;
    let head_dim = crate::N_HEAD_DIM as usize;
    let n_hc = crate::N_HC as usize;
    let n_expert = crate::N_EXPERT as usize;
    let n_ff_exp = crate::N_FF_EXP as usize;
    let n_lora_q = crate::N_LORA_Q as usize;
    let n_lora_o = crate::N_LORA_O as usize;
    let n_out_group = crate::N_OUT_GROUP as usize;
    let n_group_heads = n_head / n_out_group;
    let n_exp_used = crate::N_EXPERT_USED as usize;
    let n_indexer_head = crate::N_INDEXER_HEAD as usize;
    let indexer_head_dim = crate::N_INDEXER_HEAD_DIM as usize;

    let n_hc_mix = 2 * n_hc + n_hc * n_hc;
    let hc_dim = n_hc * n_embd;
    let q_dim = n_head * head_dim;
    let group_dim = n_group_heads * head_dim;
    let o_a_dim = n_out_group * n_lora_o;
    let o_b_dim = n_embd;

    let mut buffers: Vec<Vec<u8>> = Vec::new();
    let tf = |b: &mut Vec<Vec<u8>>, buf: Vec<u8>, name: &str, tt: u32| -> Tensor {
        let bytes = buf.len() as u64;
        let e = bytes / if tt == 0 || tt == 22 { 4 } else { 2 };
        let p = buf.as_ptr();
        b.push(buf);
        Tensor { name: name.to_string(), tensor_type: tt, data: p,
            elements: e, bytes, dims: vec![e] }
    };

    // Output tensors: track which are f32 (type 0), f16 (type 1), i32 (type 22)
    let token_embd = tf(&mut buffers, alloc_f16_buf(n_vocab*n_embd, fill_f16), "token_embd.weight", 1);
    let output_hc_base = tf(&mut buffers, alloc_f32_buf(n_hc_mix, fill), "output.hc_base", 0);
    let output_hc_fn = tf(&mut buffers, alloc_f32_buf(n_hc_mix*hc_dim, fill), "output.hc_fn.weight", 0);
    let output_hc_scale = tf(&mut buffers, alloc_f32_buf(3, fill), "output.hc_scale", 0);
    let output_norm = tf(&mut buffers, alloc_f32_buf(n_embd, fill), "output.norm.weight", 0);
    let output = tf(&mut buffers, alloc_f16_buf(n_vocab*n_embd, fill_f16), "output.weight", 1);

    let mut layers = Vec::with_capacity(n_layer);
    for i in 0..n_layer {
        let p = format!("blk.{}.", i);
        layers.push(LayerWeights {
            hc_attn_fn:       tf(&mut buffers, alloc_f32_buf(n_hc_mix*hc_dim, fill), &format!("{}hc_attn_fn.weight", p), 0),
            hc_attn_scale:    tf(&mut buffers, alloc_f32_buf(3, fill), &format!("{}hc_attn_scale", p), 0),
            hc_attn_base:     tf(&mut buffers, alloc_f32_buf(n_hc_mix, fill), &format!("{}hc_attn_base", p), 0),
            attn_norm:        tf(&mut buffers, alloc_f32_buf(n_embd, fill), &format!("{}attn_norm.weight", p), 0),
            attn_q_a:         tf(&mut buffers, alloc_f16_buf(n_lora_q*n_embd, fill_f16), &format!("{}attn_q_a.weight", p), 1),
            attn_q_a_norm:    tf(&mut buffers, alloc_f32_buf(n_lora_q, fill), &format!("{}attn_q_a_norm.weight", p), 0),
            attn_q_b:         tf(&mut buffers, alloc_f16_buf(q_dim*n_lora_q, fill_f16), &format!("{}attn_q_b.weight", p), 1),
            attn_kv:          tf(&mut buffers, alloc_f16_buf(head_dim*n_embd, fill_f16), &format!("{}attn_kv.weight", p), 1),
            attn_kv_a_norm:   tf(&mut buffers, alloc_f32_buf(n_embd, fill), &format!("{}attn_kv_a_norm.weight", p), 0),
            attn_sinks:       tf(&mut buffers, alloc_f16_buf(1*head_dim, fill_f16), &format!("{}attn_sinks.weight", p), 1),
            attn_output_a:    tf(&mut buffers, alloc_f16_buf(o_a_dim*group_dim, fill_f16), &format!("{}attn_output_a.weight", p), 1),
            attn_output_b:    tf(&mut buffers, alloc_f32_buf(o_b_dim*o_a_dim, fill), &format!("{}attn_output_b.weight", p), 0),
            attn_compressor_ape:  tf(&mut buffers, alloc_f32_buf(n_head*head_dim, fill), &format!("{}attn_compressor_ape.weight", p), 0),
            attn_compressor_kv:   tf(&mut buffers, alloc_f16_buf(n_head*head_dim*n_embd, fill_f16), &format!("{}attn_compressor_kv.weight", p), 1),
            attn_compressor_gate: tf(&mut buffers, alloc_f32_buf(n_head*head_dim, fill), &format!("{}attn_compressor_gate.weight", p), 0),
            attn_compressor_norm: tf(&mut buffers, alloc_f32_buf(n_head*head_dim, fill), &format!("{}attn_compressor_norm.weight", p), 0),
            indexer_attn_q_b:     tf(&mut buffers, alloc_f16_buf(n_indexer_head*indexer_head_dim*n_embd, fill_f16), &format!("{}indexer.attn_q_b.weight", p), 1),
            indexer_proj:         tf(&mut buffers, alloc_f16_buf(n_head*n_indexer_head*head_dim*indexer_head_dim, fill_f16), &format!("{}indexer.proj.weight", p), 1),
            indexer_compressor_ape:   tf(&mut buffers, alloc_f32_buf(n_indexer_head*indexer_head_dim, fill), &format!("{}indexer_compressor_ape.weight", p), 0),
            indexer_compressor_kv:    tf(&mut buffers, alloc_f16_buf(n_indexer_head*indexer_head_dim*n_embd, fill_f16), &format!("{}indexer_compressor_kv.weight", p), 1),
            indexer_compressor_gate:  tf(&mut buffers, alloc_f32_buf(n_indexer_head*indexer_head_dim, fill), &format!("{}indexer_compressor_gate.weight", p), 0),
            indexer_compressor_norm:  tf(&mut buffers, alloc_f32_buf(n_indexer_head*indexer_head_dim, fill), &format!("{}indexer_compressor_norm.weight", p), 0),
            hc_ffn_fn:           tf(&mut buffers, alloc_f32_buf(n_hc_mix*hc_dim, fill), &format!("{}hc_ffn_fn.weight", p), 0),
            hc_ffn_scale:        tf(&mut buffers, alloc_f32_buf(3, fill), &format!("{}hc_ffn_scale.weight", p), 0),
            hc_ffn_base:         tf(&mut buffers, alloc_f32_buf(n_hc_mix, fill), &format!("{}hc_ffn_base.weight", p), 0),
            ffn_norm:            tf(&mut buffers, alloc_f32_buf(n_embd, fill), &format!("{}ffn_norm.weight", p), 0),
            ffn_gate_tid2eid:    tf(&mut buffers, alloc_i32_zero_buf(n_vocab*n_exp_used), &format!("{}ffn_gate_tid2eid.weight", p), 22),
            ffn_gate_inp:        tf(&mut buffers, alloc_f16_buf(n_expert*n_embd, fill_f16), &format!("{}ffn_gate_inp.weight", p), 1),
            ffn_exp_probs_b:     tf(&mut buffers, alloc_f32_buf(q_dim, fill), &format!("{}ffn_exp_probs_b.weight", p), 0),
            ffn_gate_exps:       tf(&mut buffers, alloc_f16_buf(n_expert*n_ff_exp*n_embd, fill_f16), &format!("{}ffn_gate_exps.weight", p), 1),
            ffn_up_exps:         tf(&mut buffers, alloc_f16_buf(n_expert*n_ff_exp*n_embd, fill_f16), &format!("{}ffn_up_exps.weight", p), 1),
            ffn_down_exps:       tf(&mut buffers, alloc_f16_buf(n_embd*n_expert*n_ff_exp, fill_f16), &format!("{}ffn_down_exps.weight", p), 1),
            ffn_gate_shexp:      tf(&mut buffers, alloc_f16_buf(n_ff_exp*n_embd, fill_f16), &format!("{}ffn_gate_shexp.weight", p), 1),
            ffn_up_shexp:        tf(&mut buffers, alloc_f16_buf(n_ff_exp*n_embd, fill_f16), &format!("{}ffn_up_shexp.weight", p), 1),
            ffn_down_shexp:      tf(&mut buffers, alloc_f16_buf(n_embd*n_ff_exp, fill_f16), &format!("{}ffn_down_shexp.weight", p), 1),
        });
    }

    OwnedModelWeights {
        _buffers: buffers,
        weights: ModelWeights { token_embd, output_hc_base, output_hc_fn,
            output_hc_scale, output_norm, output, layers },
    }
}

fn alloc_f16_buf(elems: usize, fill: u16) -> Vec<u8> {
    let mut buf = Vec::with_capacity(elems * 2);
    let fb = fill.to_le_bytes();
    for _ in 0..elems { buf.extend_from_slice(&fb); }
    buf
}

fn alloc_f32_buf(elems: usize, fill: f32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(elems * 4);
    let fb = fill.to_le_bytes();
    for _ in 0..elems { buf.extend_from_slice(&fb); }
    buf
}

fn alloc_i32_zero_buf(elems: usize) -> Vec<u8> {
    vec![0u8; elems * 4]
}

/// Add a tensor to the buffer list and return a Tensor struct.
/// tensor_type: 0=f32, 1=f16, 22=i32 (matching GGUF type indices)
#[allow(dead_code)]
fn mk_tensor(buffers: &mut Vec<Vec<u8>>, buf: Vec<u8>, name: &str, tensor_type: u32) -> Tensor {
    let bytes = buf.len() as u64;
    let elems_per = if tensor_type == 0 || tensor_type == 22 { 4 } else { 2 };
    let elems = bytes / elems_per;
    let ptr = buf.as_ptr();
    buffers.push(buf);
    Tensor {
        name: name.to_string(),
        tensor_type,
        data: ptr,
        elements: elems,
        bytes,
        dims: vec![elems],
    }
}
