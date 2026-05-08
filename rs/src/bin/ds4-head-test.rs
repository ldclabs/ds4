// ds4-head-test: Mirror of ds4_engine_head_test from ds4.c
//
// Loads a GGUF model, runs a single token through the first layer,
// and prints per-submodule statistics (attn_pre, q, kv, attn_heads,
// attn_out, after_attn_hc, after_ffn_hc, logits).
//
// Usage:
//   cargo run --features test-dimensions --bin ds4-head-test -- /tmp/test_ds4.gguf
//   cargo run --bin ds4-head-test -- ds4flash.gguf

use std::env;
use std::process;

use ds4::gguf::GgufModel;
use ds4::model::{bind_weights_unchecked, ModelWeights};
use ds4::forward::{
    embed_token_f16, hc_from_plain_embedding,
    hc_attn_pre, layer_q_projection, layer_kv_projection,
    rope_tail_layer_inplace, layer_attention_rows_one, layer_grouped_out,
    layer_ffn_one, output_logits, KvCache,
};
use ds4::{
    fp8_kv_quantize_row_inplace, f16_round_inplace, hc_post_one,
};
use ds4::constants::*;

fn print_vec_stats(label: &str, v: &[f32]) {
    let n = v.len();
    if n == 0 {
        println!("{} len=0", label);
        return;
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for &x in v {
        if x < min { min = x; }
        if x > max { max = x; }
        sum += x as f64;
        sum_sq += (x as f64) * (x as f64);
    }
    let mean = sum / n as f64;
    let var = sum_sq / n as f64 - mean * mean;
    println!(
        "{} n={} min={:.6} max={:.6} mean={:.6} std={:.6}",
        label, n, min, max, mean, var.abs().sqrt()
    );
}

fn run_head_test(weights: &ModelWeights, token: i32, pos: u32) {
    let n_embd = N_EMBD as usize;
    let n_hc = N_HC as usize;
    let n_head = N_HEAD as usize;
    let head_dim = N_HEAD_DIM as usize;
    let n_vocab = N_VOCAB as usize;

    // Embed token
    let mut plain = vec![0.0f32; n_embd];
    embed_token_f16(weights, token, &mut plain);
    print_vec_stats("token_embd", &plain);

    // Layer 0: attn_pre
    let mut cur = vec![0.0f32; n_hc * n_embd];
    hc_from_plain_embedding(&mut cur, &plain);

    let layer = &weights.layers[0];

    let mut attn_cur = vec![0.0f32; n_embd];
    let mut residual_hc = vec![0.0f32; n_hc * n_embd];
    let mut post = [0.0f32; 4];
    let mut comb = [0.0f32; 16];
    hc_attn_pre(&mut attn_cur, &mut residual_hc, &mut post, &mut comb, &cur, layer);
    print_vec_stats("blk.0 attn_pre", &attn_cur);

    // RMS Norm
    let mut attn_norm = vec![0.0f32; n_embd];
    let norm_w = layer.attn_norm.as_f32();
    ds4::rms_norm_weighted(&mut attn_norm, &attn_cur, norm_w, n_embd, RMS_EPS);

    // Q projection
    let q_dim = n_head * head_dim;
    let mut q = vec![0.0f32; q_dim];
    layer_q_projection(&mut q, &attn_norm, layer);
    print_vec_stats("blk.0 q", &q);

    // KV projection
    let mut kv = vec![0.0f32; head_dim];
    layer_kv_projection(&mut kv, &attn_norm, layer);
    print_vec_stats("blk.0 kv", &kv);

    // RoPE on Q and KV (layer 0, inverse=false)
    rope_tail_layer_inplace(&mut q, n_head, head_dim, N_ROT as usize, pos as usize, 0, false);
    rope_tail_layer_inplace(&mut kv, N_HEAD_KV as usize, head_dim, N_ROT as usize, pos as usize, 0, false);

    // Quantize KV
    fp8_kv_quantize_row_inplace(&mut kv, head_dim, N_ROT as usize);
    f16_round_inplace(&mut kv, head_dim);

    // For head test, use single-token attention (self-attention only)
    let mut kv_cache = KvCache::new(4096);
    kv_cache.push_raw(0, &kv);

    // Attention (self-attention, n_kv=1 — use raw SWA rows)
    let sinks = layer.attn_sinks.as_f32_auto();
    let mut attn_heads = vec![0.0f32; q_dim];
    let lc = &kv_cache.layers[0];
    layer_attention_rows_one(&mut attn_heads, &q, &lc.raw_kv, lc.n_raw, &sinks);
    print_vec_stats("blk.0 attn_heads", &attn_heads);

    // RoPE on attn output (inverse=true, deskew)
    rope_tail_layer_inplace(&mut attn_heads, n_head, head_dim, N_ROT as usize, pos as usize, 0, true);

    // Grouped output
    let mut attn_out = vec![0.0f32; n_embd];
    layer_grouped_out(&mut attn_out, &attn_heads, layer);
    print_vec_stats("blk.0 attn_out", &attn_out);

    // HC post
    let mut after_attn_hc = vec![0.0f32; n_hc * n_embd];
    hc_post_one(&mut after_attn_hc, &attn_out, &residual_hc, &post, &comb, n_embd, n_hc);
    print_vec_stats("blk.0 after_attn_hc", &after_attn_hc);

    // FFN
    let mut after_ffn_hc = vec![0.0f32; n_hc * n_embd];
    layer_ffn_one(&mut after_ffn_hc, &after_attn_hc, layer, 0, token, true);
    print_vec_stats("blk.0 after_ffn_hc", &after_ffn_hc);

    // Output logits
    let mut logits = vec![0.0f32; n_vocab];
    output_logits(&mut logits, &after_ffn_hc, weights);
    print_vec_stats("logits", &logits);

    // Top-8 tokens
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\ntop logits after native blk.0 slice:");
    for i in 0..8.min(indexed.len()) {
        println!("  {:6}  {:9.4}", indexed[i].0, indexed[i].1);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <gguf_path> <token_id> [position] [--full]", args[0]);
        eprintln!("  position: token position for RoPE (default: 0)");
        eprintln!("  --full: run all layers (first-token test), not just blk.0 slice");
        eprintln!("Example: {} /tmp/test_ds4.gguf 128821 9", args[0]);
        process::exit(1);
    }

    let path = &args[1];
    let token: i32 = args[2].parse().expect("token_id must be an integer");
    let pos: u32 = if args.len() > 3 && !args[3].starts_with("--") {
        args[3].parse().expect("position must be an integer")
    } else {
        0
    };
    let full = args.iter().any(|a| a == "--full");

    println!("Loading GGUF: {}", path);
    let gguf = GgufModel::open(path).expect("Failed to open GGUF");
    println!("  version={}, tensors={}", gguf.version, gguf.tensors.len());

    let weights = bind_weights_unchecked(&gguf).expect("Failed to bind weights");
    println!("  layers={}", weights.layers.len());

    if full {
        run_first_token_test(&weights, token);
    } else {
        run_head_test(&weights, token, pos);
    }
}

/// Full first-token test: run through ALL layers and output final logits.
/// Matches ds4_engine_first_token_test from ds4.c.
fn run_first_token_test(weights: &ModelWeights, token: i32) {
    use ds4::forward::{forward_one_token_debug, KvCache};

    let n_vocab = N_VOCAB as usize;
    let n_hc = N_HC as usize;
    let n_embd = N_EMBD as usize;
    let mut kv_cache = KvCache::new(4096);
    let mut logits = vec![0.0f32; n_vocab];
    let mut final_hc = vec![0.0f32; n_hc * n_embd];

    use std::time::Instant;
    let start = Instant::now();
    forward_one_token_debug(&mut logits, Some(&mut final_hc), weights, &mut kv_cache, token, 0);
    let elapsed = start.elapsed();

    print_vec_stats("final_hc", &final_hc);
    print_vec_stats("logits", &logits);

    // Top-8
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\ntop logits after whole-model CPU pass:");
    for i in 0..8.min(indexed.len()) {
        println!("  {:6}  {:9.4}", indexed[i].0, indexed[i].1);
    }

    println!("\nfirst-token time: {:?} ({:.1} ms)", elapsed, elapsed.as_secs_f64() * 1000.0);
}
