//! Binary: debug KV cache and attention for forward pass
use std::env;
use ds4::gguf::GgufModel;
use ds4::model::bind_weights_unchecked;
use ds4::forward::{forward_one_token_debug, KvCache};
use ds4::tokenizer::Vocab;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_path = &args[1];
    let prompt_str = &args[2];
    
    let gguf = GgufModel::open(model_path).expect("load");
    let weights = bind_weights_unchecked(&gguf).expect("bind");
    let vocab = Vocab::load(&gguf).expect("vocab");
    
    let tokens = vocab.encode(prompt_str);
    println!("# Tokens: {:?}", tokens);
    
    let ctx_size = 4096;
    let mut kv_cache = KvCache::new(ctx_size);
    let n_vocab = ds4::N_VOCAB as usize;
    let hc_dim = (ds4::N_HC * ds4::N_EMBD) as usize;
    let mut prev_hc = vec![0.0f32; hc_dim];
    
    for (i, &token) in tokens.iter().enumerate() {
        let mut logits = vec![0.0f32; n_vocab];
        let mut hc_out = vec![0.0f32; (ds4::N_HC * ds4::N_EMBD) as usize];
        let in_hc: Option<&[f32]> = if i == 0 { None } else { Some(&prev_hc) };
        forward_one_token_debug(&mut logits, in_hc, Some(&mut hc_out), &weights, &mut kv_cache, token, i);
        prev_hc.copy_from_slice(&hc_out);
        
        let lc = &kv_cache.layers[0];
        println!("# pos={} token={} n_raw={}", i, token, lc.n_raw);
        
        // Print min/max/rms of logits
        let n = logits.len();
        let mut lmin = f32::INFINITY;
        let mut lmax = f32::NEG_INFINITY;
        let mut ssq = 0.0f64;
        for &v in &logits {
            if v < lmin { lmin = v; }
            if v > lmax { lmax = v; }
            ssq += (v as f64) * (v as f64);
        }
        let rms = (ssq / n as f64).sqrt();
        println!("  logits: min={:.4} max={:.4} rms={:.4}", lmin, lmax, rms);
        
        // HC out stats
        let hn = hc_out.len();
        let mut hmin = f32::INFINITY;
        let mut hmax = f32::NEG_INFINITY;
        let mut hssq = 0.0f64;
        for &v in &hc_out {
            if v < hmin { hmin = v; }
            if v > hmax { hmax = v; }
            hssq += (v as f64) * (v as f64);
        }
        let hrms = (hssq / hn as f64).sqrt();
        println!("  hc_out: min={:.4} max={:.4} rms={:.4}", hmin, hmax, hrms);
    }
}
