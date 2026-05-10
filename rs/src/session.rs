// Session management: maintains the live KV cache and logits for inference.

use crate::forward::{KvCache, forward_one_token, forward_prefill, forward_prefill_batched};
use crate::model::ModelWeights;
use crate::N_VOCAB;

/// A mutable inference session: owns the KV cache and current logits.
pub struct Session {
    pub kv_cache: KvCache,
    pub logits: Vec<f32>,
    pub tokens: Vec<i32>,   // all tokens in the session
    pub ctx_size: usize,
}

impl Session {
    /// Create a new session with the given context size.
    pub fn new(ctx_size: usize) -> Self {
        Session {
            kv_cache: KvCache::new(ctx_size),
            logits: vec![0.0f32; N_VOCAB as usize],
            tokens: Vec::new(),
            ctx_size,
        }
    }

    /// Number of tokens currently in the session.
    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Get the largest common prefix length between this session's tokens and a prompt.
    pub fn common_prefix_len(&self, prompt: &[i32]) -> usize {
        let n = self.tokens.len().min(prompt.len());
        for i in 0..n {
            if self.tokens[i] != prompt[i] {
                return i;
            }
        }
        n
    }

    /// Sync the session to a full prompt. If the session already has a prefix
    /// of this prompt, only the suffix is processed.
    /// When `batched` is true, uses the layer-major parallel prefill for speed.
    pub fn sync(&mut self, weights: &ModelWeights, prompt: &[i32], batched: bool) {
        if prompt.is_empty() {
            return;
        }

        let common = self.common_prefix_len(prompt);

        if common == 0 {
            // Full mismatch: rebuild from scratch
            self.kv_cache = KvCache::new(self.ctx_size);
            self.tokens.clear();
            if batched && prompt.len() > 1 {
                forward_prefill_batched(&mut self.logits, weights, &mut self.kv_cache, prompt);
            } else {
                forward_prefill(&mut self.logits, weights, &mut self.kv_cache, prompt);
            }
            self.tokens.extend_from_slice(prompt);
        } else if common < prompt.len() {
            // Extend with suffix
            let suffix = &prompt[common..];
            // Truncate cached tokens
            self.tokens.truncate(common);
            // Process suffix
            for &token in suffix {
                forward_one_token(&mut self.logits, weights, &mut self.kv_cache, token, self.tokens.len());
                self.tokens.push(token);
            }
        }
        // If common == prompt.len(), we already have this prefix — just return
    }

    /// Evaluate one additional token.
    pub fn eval(&mut self, weights: &ModelWeights, token: i32) {
        let pos = self.tokens.len();
        forward_one_token(&mut self.logits, weights, &mut self.kv_cache, token, pos);
        self.tokens.push(token);
    }

    /// Get the argmax token from current logits.
    pub fn argmax(&self) -> i32 {
        let mut best = 0i32;
        let mut best_val = f32::NEG_INFINITY;
        for i in 0..self.logits.len() {
            if self.logits[i] > best_val {
                best_val = self.logits[i];
                best = i as i32;
            }
        }
        best
    }

    /// Sample a token from the logits with temperature.
    pub fn sample(&self, temperature: f32, top_k: usize, top_p: f32, rng: &mut u64) -> i32 {
        if temperature <= 0.0 || temperature < 0.001 {
            return self.argmax();
        }

        let n = self.logits.len();
        let mut logits = self.logits.clone();

        // Apply temperature
        let inv_temp = 1.0 / temperature;
        for l in &mut logits {
            *l *= inv_temp;
        }

        // Top-K filtering
        if top_k > 0 && top_k < n {
            let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
                .map(|(i, &v)| (i, v))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = indexed[top_k.min(indexed.len()).saturating_sub(1)].1;
            for l in &mut logits {
                if *l < threshold { *l = f32::NEG_INFINITY; }
            }
        }

        // Top-P (nucleus) filtering
        if top_p > 0.0 && top_p < 1.0 {
            let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
                .filter(|(_, &v)| v > f32::NEG_INFINITY / 2.0)
                .map(|(i, &v)| (i, v))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut probs: Vec<f32> = indexed.iter().map(|(_, v)| v.exp()).collect();
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for p in &mut probs { *p /= sum; }

                let mut cumsum = 0.0f32;
                let mut cutoff = indexed.len();
                for (k, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if cumsum >= top_p {
                        cutoff = k + 1;
                        break;
                    }
                }

                for i in cutoff..indexed.len() {
                    logits[indexed[i].0] = f32::NEG_INFINITY;
                }
            }
        }

        // Softmax + sample
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter()
            .map(|&v| if v > max_val - 50.0 { (v - max_val).exp() } else { 0.0 })
            .collect();
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs { *p /= sum; }

            // Xorshift RNG
            *rng ^= *rng << 13;
            *rng ^= *rng >> 17;
            *rng ^= *rng << 5;
            let r = (*rng as f64) / (u64::MAX as f64);

            let mut cumsum = 0.0f64;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p as f64;
                if r < cumsum {
                    return i as i32;
                }
            }
        }

        // Fallback: argmax
        self.argmax()
    }

    /// Get top-k logprobs.
    pub fn top_logprobs(&self, out: &mut [(i32, f32)], k: usize) {
        let _n = self.logits.len();
        let mut indexed: Vec<(usize, f32)> = self.logits.iter().enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let max_val = indexed.first().map(|&(_, v)| v).unwrap_or(0.0);
        let mut probs: Vec<f32> = indexed.iter()
            .map(|&(_, v)| (v - max_val).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs { if sum > 0.0 { *p /= sum; } }

        let m = k.min(indexed.len()).min(out.len());
        for i in 0..m {
            out[i] = (indexed[i].0 as i32, probs[i].ln());
        }
    }

    /// Invalidate the session (force re-prefill on next sync).
    pub fn invalidate(&mut self) {
        self.tokens.clear();
        self.kv_cache = KvCache::new(self.ctx_size);
        self.logits.fill(0.0);
    }

    /// Rewind the session to a specific position.
    pub fn rewind(&mut self, pos: usize) {
        if pos < self.tokens.len() {
            self.tokens.truncate(pos);
            // Rebuild KV cache from tokens
            // In a complete implementation, we'd rebuild the KV cache state
        }
    }

    /// Generate tokens autoregressively. Returns the generated token IDs
    /// (not including the prompt). Stops on EOS or when max_tokens is reached.
    pub fn generate(
        &mut self,
        weights: &ModelWeights,
        eos_token: i32,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        rng: &mut u64,
    ) -> Vec<i32> {
        let mut generated = Vec::new();

        for _ in 0..max_tokens {
            let token = if temperature > 0.001 {
                self.sample(temperature, top_k, top_p, rng)
            } else {
                self.argmax()
            };

            if token == eos_token {
                break;
            }

            self.eval(weights, token);
            generated.push(token);
        }

        generated
    }

    /// Generate tokens and return the decoded text string.
    pub fn generate_text(
        &mut self,
        weights: &ModelWeights,
        vocab: &crate::tokenizer::Vocab,
        eos_token: i32,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        rng: &mut u64,
    ) -> String {
        let tokens = self.generate(weights, eos_token, max_tokens, temperature, top_k, top_p, rng);
        vocab.decode(&tokens)
    }
}
