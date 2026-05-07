// Generate a minimal test GGUF model for ds4.rs integration testing.
//
// Usage: cargo run --features test-dimensions --bin gen_test_gguf -- [output_path]
//
// Produces a valid GGUF v3 file with the test-dimensions architecture
// (1 layer, small vocab/embedding) and all weights set to f16(1.0)/f32(1.0).

use ds4::gguf::{GgufWriter, GgufValue, GgufArray, gguf_type_index};
use ds4::constants::*;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let out_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "test_ds4.gguf".to_string()
    };

    eprintln!("Generating test GGUF model: {}", out_path);
    eprintln!("  N_LAYER={} N_EMBD={} N_VOCAB={} N_HEAD={} N_HC={}",
        N_LAYER, N_EMBD, N_VOCAB, N_HEAD, N_HC);

    let fill_f16 = ds4::f32_to_f16(1.0);
    let fill_f32 = 1.0f32;

    let n_layer = N_LAYER as usize;
    let n_embd = N_EMBD as usize;
    let n_vocab = N_VOCAB as usize;
    let n_head = N_HEAD as usize;
    let head_dim = N_HEAD_DIM as usize;
    let n_hc = N_HC as usize;
    let n_expert = N_EXPERT as usize;
    let n_ff_exp = N_FF_EXP as usize;
    let n_lora_q = N_LORA_Q as usize;
    let n_lora_o = N_LORA_O as usize;
    let n_out_group = N_OUT_GROUP as usize;
    let n_group_heads = n_head / n_out_group;
    let n_exp_used = N_EXPERT_USED as usize;
    let n_indexer_head = N_INDEXER_HEAD as usize;
    let indexer_head_dim = N_INDEXER_HEAD_DIM as usize;

    let n_hc_mix = 2 * n_hc + n_hc * n_hc;
    let hc_dim = n_hc * n_embd;
    let q_dim = n_head * head_dim;
    let group_dim = n_group_heads * head_dim;
    let o_a_dim = n_out_group * n_lora_o;
    let o_b_dim = n_embd;

    let f16_type = gguf_type_index("f16").expect("f16 type not found") as u32;
    let f32_type = gguf_type_index("f32").expect("f32 type not found") as u32;
    let i32_type = gguf_type_index("i32").expect("i32 type not found") as u32;

    let mut writer = GgufWriter::create_file(&out_path, 3, 32)
        .expect("Failed to create GGUF file");

    // Metadata
    writer.add_meta("general.architecture", GgufValue::String("ds4".to_string()));
    writer.add_meta("general.name", GgufValue::String("test_model".to_string()));
    writer.add_meta("ds4.n_layer", GgufValue::Uint32(N_LAYER));
    writer.add_meta("ds4.n_embd", GgufValue::Uint32(N_EMBD));
    writer.add_meta("ds4.vocab_size", GgufValue::Uint32(N_VOCAB));
    writer.add_meta("ds4.attn_n_head", GgufValue::Uint32(N_HEAD));
    writer.add_meta("ds4.attn_n_head_kv", GgufValue::Uint32(N_HEAD_KV));
    writer.add_meta("general.file_type", GgufValue::Uint32(1));

    // Tokenizer metadata (minimal, for vocab loading tests)
    let mut tokens_arr = Vec::new();
    // Add some common tokens so vocab tests pass
    let common_tokens = ["<unk>", "<s>", "</s>", "the", "Hello", "a", "is", "of", "and", "to", "world", " ", "H", "e", "l", "o", "w", "r", "d"];
    for tok in &common_tokens {
        tokens_arr.push(GgufValue::String(tok.to_string()));
    }
    for i in tokens_arr.len()..N_VOCAB as usize {
        tokens_arr.push(GgufValue::String(format!("<tok_{}>", i)));
    }
    writer.add_meta("tokenizer.ggml.tokens",
        GgufValue::Array(GgufArray { element_type: 8, elements: tokens_arr }));
    writer.add_meta("tokenizer.ggml.bos_token_id", GgufValue::Uint32(0));
    writer.add_meta("tokenizer.ggml.eos_token_id", GgufValue::Uint32(1));

    // Helper to create filled tensor data
    let f16_data = |elems: usize| -> Vec<u8> {
        let mut v = Vec::with_capacity(elems * 2);
        let fb = fill_f16.to_le_bytes();
        for _ in 0..elems { v.extend_from_slice(&fb); }
        v
    };
    let f32_data = |elems: usize| -> Vec<u8> {
        let mut v = Vec::with_capacity(elems * 4);
        let fb = fill_f32.to_le_bytes();
        for _ in 0..elems { v.extend_from_slice(&fb); }
        v
    };
    let i32_zero = |elems: usize| -> Vec<u8> { vec![0u8; elems * 4] };

    // --- Output tensors ---
    // f16: token_embd, output
    // f32: output_hc_base, output_hc_fn, output_hc_scale, output_norm
    writer.add_tensor("token_embd.weight", vec![n_vocab as u64, n_embd as u64], f16_type, f16_data(n_vocab * n_embd));
    writer.add_tensor("output_hc_base.weight", vec![n_hc as u64], f32_type, f32_data(n_hc));
    writer.add_tensor("output_hc_fn.weight", vec![hc_dim as u64, n_hc as u64], f32_type, f32_data(hc_dim * n_hc));
    writer.add_tensor("output_hc_scale.weight", vec![1], f32_type, f32_data(1));
    writer.add_tensor("output_norm.weight", vec![n_embd as u64], f32_type, f32_data(n_embd));
    writer.add_tensor("output.weight", vec![n_vocab as u64, n_embd as u64], f16_type, f16_data(n_vocab * n_embd));

    // --- Layer tensors ---
    for i in 0..n_layer {
        let p = format!("blk.{}.", i);

        // HC attn: f32 fn/scale/base
        writer.add_tensor(&format!("{}hc_attn_fn.weight", p), vec![hc_dim as u64, n_hc_mix as u64], f32_type, f32_data(hc_dim * n_hc_mix));
        writer.add_tensor(&format!("{}hc_attn_scale.weight", p), vec![3], f32_type, f32_data(3));
        writer.add_tensor(&format!("{}hc_attn_base.weight", p), vec![n_hc_mix as u64], f32_type, f32_data(n_hc_mix));

        // Attn: norm f32, q_a/q_b f16, q_a_norm f32, kv f16, kv_norm f32, sinks f16, o_a f16, o_b f32
        writer.add_tensor(&format!("{}attn_norm.weight", p), vec![n_embd as u64], f32_type, f32_data(n_embd));
        writer.add_tensor(&format!("{}attn_q_a.weight", p), vec![n_lora_q as u64, n_embd as u64], f16_type, f16_data(n_lora_q * n_embd));
        writer.add_tensor(&format!("{}attn_q_a_norm.weight", p), vec![n_lora_q as u64], f32_type, f32_data(n_lora_q));
        writer.add_tensor(&format!("{}attn_q_b.weight", p), vec![q_dim as u64, n_lora_q as u64], f16_type, f16_data(q_dim * n_lora_q));
        writer.add_tensor(&format!("{}attn_kv.weight", p), vec![head_dim as u64, n_embd as u64], f16_type, f16_data(head_dim * n_embd));
        writer.add_tensor(&format!("{}attn_kv_a_norm.weight", p), vec![n_embd as u64], f32_type, f32_data(n_embd));
        writer.add_tensor(&format!("{}attn_sinks.weight", p), vec![1, head_dim as u64], f16_type, f16_data(1 * head_dim));
        writer.add_tensor(&format!("{}attn_output_a.weight", p), vec![o_a_dim as u64, group_dim as u64], f16_type, f16_data(o_a_dim * group_dim));
        writer.add_tensor(&format!("{}attn_output_b.weight", p), vec![o_b_dim as u64, o_a_dim as u64], f32_type, f32_data(o_b_dim * o_a_dim));

        // Compressor: f32 ape/gate/norm, f16 kv
        writer.add_tensor(&format!("{}attn_compressor_ape.weight", p), vec![(n_head * head_dim) as u64], f32_type, f32_data(n_head * head_dim));
        writer.add_tensor(&format!("{}attn_compressor_kv.weight", p), vec![(n_head * head_dim) as u64, n_embd as u64], f16_type, f16_data(n_head * head_dim * n_embd));
        writer.add_tensor(&format!("{}attn_compressor_gate.weight", p), vec![(n_head * head_dim) as u64], f32_type, f32_data(n_head * head_dim));
        writer.add_tensor(&format!("{}attn_compressor_norm.weight", p), vec![(n_head * head_dim) as u64], f32_type, f32_data(n_head * head_dim));

        // Indexer: f16 q_b/proj/kv, f32 compressor ape/gate/norm
        writer.add_tensor(&format!("{}indexer.attn_q_b.weight", p), vec![(n_indexer_head * indexer_head_dim) as u64, n_embd as u64], f16_type, f16_data(n_indexer_head * indexer_head_dim * n_embd));
        writer.add_tensor(&format!("{}indexer.proj.weight", p), vec![(n_head * n_indexer_head) as u64, (head_dim * indexer_head_dim) as u64], f16_type, f16_data(n_head * n_indexer_head * head_dim * indexer_head_dim));
        writer.add_tensor(&format!("{}indexer_compressor_ape.weight", p), vec![(n_indexer_head * indexer_head_dim) as u64], f32_type, f32_data(n_indexer_head * indexer_head_dim));
        writer.add_tensor(&format!("{}indexer_compressor_kv.weight", p), vec![(n_indexer_head * indexer_head_dim) as u64, n_embd as u64], f16_type, f16_data(n_indexer_head * indexer_head_dim * n_embd));
        writer.add_tensor(&format!("{}indexer_compressor_gate.weight", p), vec![(n_indexer_head * indexer_head_dim) as u64], f32_type, f32_data(n_indexer_head * indexer_head_dim));
        writer.add_tensor(&format!("{}indexer_compressor_norm.weight", p), vec![(n_indexer_head * indexer_head_dim) as u64], f32_type, f32_data(n_indexer_head * indexer_head_dim));

        // FFN HC: f32 fn/scale/base
        writer.add_tensor(&format!("{}hc_ffn_fn.weight", p), vec![hc_dim as u64, n_hc_mix as u64], f32_type, f32_data(hc_dim * n_hc_mix));
        writer.add_tensor(&format!("{}hc_ffn_scale.weight", p), vec![3], f32_type, f32_data(3));
        writer.add_tensor(&format!("{}hc_ffn_base.weight", p), vec![n_hc_mix as u64], f32_type, f32_data(n_hc_mix));

        // FFN: norm f32
        writer.add_tensor(&format!("{}ffn_norm.weight", p), vec![n_embd as u64], f32_type, f32_data(n_embd));

        // MoE: routing i32, gate_inp f16, exp_probs_b f32, experts f16
        writer.add_tensor(&format!("{}ffn_gate_tid2eid.weight", p), vec![n_vocab as u64, n_exp_used as u64], i32_type, i32_zero(n_vocab * n_exp_used));
        writer.add_tensor(&format!("{}ffn_gate_inp.weight", p), vec![n_expert as u64, n_embd as u64], f16_type, f16_data(n_expert * n_embd));
        writer.add_tensor(&format!("{}exp_probs_b.bias", p), vec![q_dim as u64], f32_type, f32_data(q_dim));
        writer.add_tensor(&format!("{}ffn_gate_exps.weight", p), vec![(n_expert * n_ff_exp) as u64, n_embd as u64], f16_type, f16_data(n_expert * n_ff_exp * n_embd));
        writer.add_tensor(&format!("{}ffn_up_exps.weight", p), vec![(n_expert * n_ff_exp) as u64, n_embd as u64], f16_type, f16_data(n_expert * n_ff_exp * n_embd));
        writer.add_tensor(&format!("{}ffn_down_exps.weight", p), vec![n_embd as u64, (n_expert * n_ff_exp) as u64], f16_type, f16_data(n_embd * n_expert * n_ff_exp));

        // Shared expert: f16
        writer.add_tensor(&format!("{}ffn_gate_shexp.weight", p), vec![n_ff_exp as u64, n_embd as u64], f16_type, f16_data(n_ff_exp * n_embd));
        writer.add_tensor(&format!("{}ffn_up_shexp.weight", p), vec![n_ff_exp as u64, n_embd as u64], f16_type, f16_data(n_ff_exp * n_embd));
        writer.add_tensor(&format!("{}ffn_down_shexp.weight", p), vec![n_embd as u64, n_ff_exp as u64], f16_type, f16_data(n_embd * n_ff_exp));
    }

    eprintln!("Writing GGUF file...");
    writer.finish().expect("Failed to write GGUF");
    eprintln!("Done: {}", out_path);
}
