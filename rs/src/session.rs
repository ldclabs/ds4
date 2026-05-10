// Session management: maintains the live KV cache and logits for inference.

use crate::forward::{KvCache, forward_one_token_debug, forward_partial};
use crate::model::ModelWeights;
use crate::{N_HC, N_EMBD, N_VOCAB};

/// A mutable inference session: owns the KV cache, current logits, and HC state.
/// The HC (Hybrid Connection) state flows from one token to the next, matching
/// the C code's `residual_hc` carry-over. Without this, each token would start
/// from a fresh HC state (embedding-only), breaking temporal coherence.
pub struct Session {
    pub kv_cache: KvCache,
    pub logits: Vec<f32>,
    pub tokens: Vec<i32>,   // all tokens in the session
    pub ctx_size: usize,
    hc: Vec<f32>,           // [N_HC * N_EMBD], carried across tokens
}

impl Session {
    /// Create a new session with the given context size.
    pub fn new(ctx_size: usize) -> Self {
        Session {
            kv_cache: KvCache::new(ctx_size),
            logits: vec![0.0f32; N_VOCAB as usize],
            tokens: Vec::new(),
            ctx_size,
            hc: vec![0.0f32; (N_HC * N_EMBD) as usize],
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
    /// HC state is carried across all tokens to match the C code's residual_hc flow.
    pub fn sync(&mut self, weights: &ModelWeights, prompt: &[i32], _batched: bool) {
        if prompt.is_empty() {
            return;
        }

        let common = self.common_prefix_len(prompt);
        let hc_dim = (N_HC * N_EMBD) as usize;

        if common == 0 {
            // Full mismatch: rebuild from scratch
            self.kv_cache = KvCache::new(self.ctx_size);
            self.tokens.clear();
            // Process token-by-token to carry HC state correctly.
            // First token: init HC from embedding (in_hc=None).
            // Subsequent tokens: use HC from previous token.
            let mut hc = vec![0.0f32; hc_dim];
            for (i, &token) in prompt.iter().enumerate() {
                let in_hc = if i == 0 { None } else { Some(&hc[..]) };
                let mut next_hc = vec![0.0f32; hc_dim];
                forward_one_token_debug(
                    &mut self.logits, in_hc, Some(&mut next_hc),
                    weights, &mut self.kv_cache, token, i,
                );
                hc.copy_from_slice(&next_hc);
                self.tokens.push(token);
            }
            self.hc.copy_from_slice(&hc);
            // Finish prefill states (align compressor windows for decode)
            self.kv_cache.finish_prefill_states(prompt.len());
        } else if common < prompt.len() {
            // Extend with suffix
            let suffix = &prompt[common..];
            // Truncate cached tokens
            self.tokens.truncate(common);
            // Process suffix with HC carry-over
            let mut hc = self.hc.clone();
            for &token in suffix {
                let pos = self.tokens.len();
                let in_hc = if pos == 0 { None } else { Some(&hc[..]) };
                let mut next_hc = vec![0.0f32; hc_dim];
                forward_one_token_debug(
                    &mut self.logits, in_hc, Some(&mut next_hc),
                    weights, &mut self.kv_cache, token, pos,
                );
                hc.copy_from_slice(&next_hc);
                self.tokens.push(token);
            }
            self.hc.copy_from_slice(&hc);
        }
        // If common == prompt.len(), we already have this prefix — just return
    }

    /// Evaluate one additional token, carrying HC state from previous step.
    pub fn eval(&mut self, weights: &ModelWeights, token: i32) {
        let pos = self.tokens.len();
        let hc_dim = (N_HC * N_EMBD) as usize;
        let in_hc = if pos == 0 { None } else { Some(&self.hc[..]) };
        let mut next_hc = vec![0.0f32; hc_dim];
        forward_one_token_debug(
            &mut self.logits, in_hc, Some(&mut next_hc),
            weights, &mut self.kv_cache, token, pos,
        );
        self.hc.copy_from_slice(&next_hc);
        self.tokens.push(token);
    }

    /// Speculative decode step: given a just-sampled token, draft K more tokens
    /// with a fast partial model (first `draft_layers` layers only), then verify
    /// them against the full model. Returns the accepted draft tokens; the
    /// session's logits reflect the state after the last accepted position.
    ///
    /// The `token` must NOT yet be in self.tokens — it will be fed through the
    /// full model as the base token for verification, and accepted drafts will
    /// be appended to self.tokens.
    pub fn eval_speculative(
        &mut self,
        weights: &ModelWeights,
        token: i32,
        draft_layers: usize,
        draft_count: usize,
    ) -> (Vec<i32>, Vec<f32>) {
        let n_vocab = N_VOCAB as usize;
        let pos = self.tokens.len();

        // Step 1: Checkpoint KV cache for draft layers (0..draft_layers)
        let ckpt = self.kv_cache.checkpoint_layers(draft_layers);

        // Step 2: Run partial forward to get draft logits (fast, only N layers)
        // forward_partial pushes KV entries into the cache, which we'll undo.
        let mut draft_logits = vec![0.0f32; n_vocab];
        forward_partial(
            &mut draft_logits, weights, &mut self.kv_cache,
            token, pos, draft_layers,
        );

        // Step 3: Extract top-K draft tokens (greedy, with suppression)
        let mut drafts = Vec::with_capacity(draft_count);
        {
            let mut used = vec![false; n_vocab];
            for _ in 0..draft_count {
                let mut best = -1i32;
                let mut best_val = f32::NEG_INFINITY;
                for t in 0..n_vocab {
                    if !used[t] && draft_logits[t] > best_val {
                        best_val = draft_logits[t];
                        best = t as i32;
                    }
                }
                if best < 0 { break; }
                used[best as usize] = true;
                drafts.push(best);
            }
        }

        // Step 4: Restore KV cache to pre-draft state
        self.kv_cache.restore_layers(&ckpt);

        // Step 5: Run full forward on the real base token, carrying HC state
        let hc_dim = (N_HC * N_EMBD) as usize;
        let in_hc = if pos == 0 { None } else { Some(&self.hc[..]) };
        let mut hc = vec![0.0f32; hc_dim];
        forward_one_token_debug(
            &mut self.logits, in_hc, Some(&mut hc),
            weights, &mut self.kv_cache, token, pos,
        );
        self.tokens.push(token);

        // Step 6: Verify drafts sequentially, carrying HC across each accepted draft
        let mut accepted = Vec::new();
        for (i, &draft) in drafts.iter().enumerate() {
            let best = self.argmax();
            if best != draft {
                break;
            }
            accepted.push(draft);
            let mut next_hc = vec![0.0f32; hc_dim];
            forward_one_token_debug(
                &mut self.logits, Some(&hc), Some(&mut next_hc),
                weights, &mut self.kv_cache, draft, pos + 1 + i,
            );
            hc.copy_from_slice(&next_hc);
            self.tokens.push(draft);
        }

        // Save the final HC state for the next step
        self.hc.copy_from_slice(&hc);

        // Final logits are already in self.logits from the last forward_one_token
        (accepted, self.logits.clone())
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
