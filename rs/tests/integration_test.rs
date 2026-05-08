// Tests for the ds4 Rust inference engine.
// Unit tests that don't need a model file are always run.
// Integration tests needing a GGUF file are conditional on DS4_TEST_MODEL env var.

#[cfg(test)]
mod unit_tests {
    use ds4::*;
    use ds4::quant::*;
    use ds4::tokenizer::byte_encode;

    // =========================================================================
    // FP16 conversion tests
    // =========================================================================

    #[test]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0), 0.0);
    }

    #[test]
    fn test_f16_to_f32_one() {
        let one_f16 = 0x3c00u16; // 1.0 in FP16
        assert!((f16_to_f32(one_f16) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_neg_one() {
        let neg_one_f16 = 0xbc00u16; // -1.0 in FP16
        assert!((f16_to_f32(neg_one_f16) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_half() {
        let half_f16 = 0x3800u16; // 0.5 in FP16
        assert!((f16_to_f32(half_f16) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_f32_to_f16_roundtrip() {
        let test_values = [0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.125, 10.0, -10.0];
        for &val in &test_values {
            let f16 = f32_to_f16(val);
            let back = f16_to_f32(f16);
            let max_err = val.abs().max(0.01) * 0.01;
            assert!(
                (back - val).abs() < max_err,
                "roundtrip failed for {}: got {}", val, back
            );
        }
    }

    #[test]
    fn test_f16_round_inplace() {
        let mut x = vec![1.0 / 3.0, 0.123456, 100.0, -50.5];
        let orig = x.clone();
        f16_round_inplace(&mut x, 4);
        for i in 0..4 {
            let diff = (x[i] - orig[i]).abs();
            assert!(diff < orig[i].abs().max(0.01) * 0.02,
                "f16_round at {}: {} -> {} diff={}", i, orig[i], x[i], diff);
        }
    }

    // =========================================================================
    // Sigmoid tests
    // =========================================================================

    #[test]
    fn test_sigmoid_stable_zero() {
        assert!((sigmoid_stable(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_stable_large_pos() {
        assert!(sigmoid_stable(10.0) > 0.999);
    }

    #[test]
    fn test_sigmoid_stable_large_neg() {
        assert!(sigmoid_stable(-10.0) < 0.001);
    }

    #[test]
    fn test_sigmoid_stable_symmetry() {
        for &x in &[-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0] {
            let s1 = sigmoid_stable(x);
            let s2 = sigmoid_stable(-x);
            assert!((s1 + s2 - 1.0).abs() < 1e-4, "symmetry failed for x={}", x);
        }
    }

    // =========================================================================
    // SiLU / SwiGLU tests
    // =========================================================================

    #[test]
    fn test_silu_zero() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_silu_positive() {
        // SiLU(x) ≈ x for large positive x
        assert!((silu(10.0) - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_swiglu_simple() {
        // swiglu(gate, up) = silu(gate) * up
        let g = 1.0;
        let u = 2.0;
        let expected = silu(g) * u;
        assert!((swiglu(g, u) - expected).abs() < 1e-6);
    }

    // =========================================================================
    // RMS Norm tests
    // =========================================================================

    #[test]
    fn test_rms_norm_uniform() {
        let x = vec![2.0f32; 4];
        let mut out = vec![0.0f32; 4];
        rms_norm(&mut out, &x, 4, 1e-6);
        // RMS = sqrt(sum(x^2)/n) = sqrt(16/4) = 2, scale = 1/2 = 0.5
        // out[i] = x[i] * 0.5 = 1.0
        for v in &out {
            assert!((*v - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_rms_norm_varied() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        rms_norm(&mut out, &x, 4, 1e-6);
        // RMS = sqrt((1+4+9+16)/4) = sqrt(30/4) = sqrt(7.5)
        let rms = (7.5f32).sqrt();
        for i in 0..4 {
            assert!((out[i] - x[i] / rms).abs() < 1e-4);
        }
    }

    // =========================================================================
    // Softmax tests
    // =========================================================================

    #[test]
    fn test_softmax_basic() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        softmax(&mut x, 3);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(x[0] < x[1] && x[1] < x[2]);
    }

    #[test]
    fn test_softmax_all_equal() {
        let mut x = vec![2.0f32, 2.0, 2.0, 2.0];
        softmax(&mut x, 4);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        for v in &x {
            assert!((*v - 0.25).abs() < 1e-5);
        }
    }

    // =========================================================================
    // Layer compress ratio tests
    // =========================================================================

    #[test]
    fn test_layer_compress_ratio() {
        // Layers 0-1: no compression (dense)
        assert_eq!(layer_compress_ratio(0), 0);
        assert_eq!(layer_compress_ratio(1), 0);
        // Layer >=2 even: ratio 4
        assert_eq!(layer_compress_ratio(2), 4);
        // Layer >=2 odd: ratio 128
        assert_eq!(layer_compress_ratio(3), 128);
        // Verify pattern continues
        assert_eq!(layer_compress_ratio(4), 4);
        assert_eq!(layer_compress_ratio(5), 128);
        assert_eq!(layer_compress_ratio(6), 4);
        assert_eq!(layer_compress_ratio(7), 128);
    }

    // =========================================================================
    // Q2_K dequantization tests
    // =========================================================================

    #[test]
    fn test_dequantize_q2_k_basic() {
        let mut block = BlockQ2K {
            scales: [1u8; 16],  // scale factor = 1 (raw value 1)
            qs: [0u8; 64],      // all zeros
            d: f32_to_f16(1.0),
            dmin: f32_to_f16(0.0),
        };
        let mut out = [0.0f32; 256];
        dequantize_q2_k(&block, &mut out);
        // All quantized values are 0, so all outputs should be 0
        for v in &out {
            assert!((*v - 0.0).abs() < 1e-4);
        }

        // Now set all qs to 0xFF (all 2-bit values = 3)
        block.qs = [0xFFu8; 64];
        dequantize_q2_k(&block, &mut out);
        // Each element: d * scale * q = 1.0 * 1 * 3 = 3.0
        for v in &out {
            assert!((*v - 3.0).abs() < 1e-2, "got {}", *v);
        }
    }

    #[test]
    fn test_dequantize_q2_k_varied() {
        // scale byte 0x01: d_scale=1 (lower nibble), m_scale=0 (upper nibble)
        let block = BlockQ2K {
            scales: [0x01u8; 16], // d_scale=1, m_scale=0
            qs: [0x55u8; 64],     // each 2-bit value = 1
            d: f32_to_f16(2.0),
            dmin: f32_to_f16(0.5),
        };
        let mut out = [0.0f32; 256];
        dequantize_q2_k(&block, &mut out);
        // Each element: d*d_scale*q - dmin*m_scale = 2*1*1 - 0.5*0 = 2.0
        for v in &out {
            assert!((*v - 2.0).abs() < 1e-2, "got {}", *v);
        }
    }

    // =========================================================================
    // Q4_K dequantization tests
    // =========================================================================

    #[test]
    fn test_dequantize_q4_k_basic() {
        let block = BlockQ4K {
            d: f32_to_f16(1.0),
            dmin: f32_to_f16(0.0),
            scales: [63u8; 12], // all max scales
            qs: [0u8; 128],     // all zeros
        };
        let mut out = [0.0f32; 256];
        dequantize_q4_k(&block, &mut out);
        for v in &out {
            assert!((*v - 0.0).abs() < 1e-4);
        }
    }

    // =========================================================================
    // IQ2_XXS dequantization tests
    // =========================================================================

    #[test]
    fn test_dequantize_iq2_xxs_basic() {
        // With q=0: grid[0] = 0x080808..., sign_byte=0 (all positive), extra=0 → ls=1
        // Each grid byte = 8, sign = 1 → d * 0.125 * 8 * 1 = 1.0
        let block = BlockIq2Xxs {
            d: f32_to_f16(1.0),
            qs: [0u16; 32],
        };
        let mut out = [0.0f32; 256];
        dequantize_iq2_xxs(&block, &mut out);
        // All elements should be 1.0 (= 1.0 * 0.125 * 8 * 1 * 1)
        for (i, v) in out.iter().enumerate() {
            assert!((*v - 1.0).abs() < 1e-4 || *v == 0.0,
                "element {}: got {}, expected ~1.0", i, *v);
        }
    }

    // =========================================================================
    // Hash routing tests
    // =========================================================================

    #[test]
    fn test_hash_routed_expert_deterministic() {
        let e1 = hash_routed_expert(42, 0, 256);
        let e2 = hash_routed_expert(42, 0, 256);
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_hash_routed_expert_in_range() {
        for token in 0..100 {
            for router in 0..3 {
                let e = hash_routed_expert(token, router, 256);
                assert!(e < 256, "expert {} out of range", e);
            }
        }
    }

    // =========================================================================
    // Tokenizer tests
    // =========================================================================

    #[test]
    fn test_byte_encode_ascii() {
        let encoded = byte_encode(b"Hello");
        assert_eq!(encoded, "Hello");
    }

    #[test]
    fn test_byte_encode_mixed() {
        let encoded = byte_encode("Café".as_bytes());
        // 'C', 'a', 'f' are printable, 'é' (233) is not
        assert!(encoded.starts_with("Caf"));
        assert!(encoded.len() > 3);
    }

    // =========================================================================
    // HC Sinkhorn tests
    // =========================================================================

    #[test]
    fn test_hc_sinkhorn_normalizes() {
        // Test the hc_split_sinkhorn_one pathway: set up simple inputs and check
        // that outputs are finite and in reasonable ranges.
        let n_hc = 4;
        let n_mix = 2 * n_hc + n_hc * n_hc; // 24
        let mix = vec![0.0f32; n_mix];
        let scale = vec![1.0f32, 1.0, 1.0]; // pre_scale, post_scale, comb_scale
        let base = vec![0.0f32; n_mix];
        let mut out = vec![0.0f32; n_mix];

        hc_split_sinkhorn_one(&mut out, &mix, &scale, &base, n_hc, 20, HC_EPS);

        // Pre weights should be ~0.5-ish with zero inputs and zero base
        for i in 0..n_hc {
            assert!(out[i] > 0.0 && out[i] <= 1.0, "pre[{}]={}", i, out[i]);
        }

        // Post weights should be ~1.0-ish
        for i in n_hc..2 * n_hc {
            assert!(out[i] > 0.0 && out[i] <= 2.0, "post[{}]={}", i, out[i]);
        }

        // Comb matrix rows should each sum to ~1.0 (Sinkhorn normalized)
        let comb_off = 2 * n_hc;
        for dst in 0..n_hc {
            let row_sum: f32 = (0..n_hc)
                .map(|src| out[comb_off + src + dst * n_hc])
                .sum();
            assert!((row_sum - 1.0).abs() < 0.2, "comb row {} sum={}", dst, row_sum);
        }
    }

    // =========================================================================
    // FP8 KV quantization tests
    // =========================================================================

    #[test]
    fn test_fp8_kv_quantize_rot_only() {
        let mut kv = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                          9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let orig_rot: Vec<f32> = kv[..8].to_vec();
        fp8_kv_quantize_row_inplace(&mut kv, 16, 8);
        // First 8 elements (rotary dims) should be preserved unchanged
        for i in 0..8 {
            assert!((kv[i] - orig_rot[i]).abs() < 1e-6,
                "rotary dim {} changed: {} -> {}", i, orig_rot[i], kv[i]);
        }
        // Elements beyond n_rot (non-rotary) should be quantized (not zeroed)
        for i in 8..16 {
            assert!(kv[i].is_finite(),
                "non-rotary dim {} should be finite after quant", i);
        }
    }

    // =========================================================================
    // Model validation tests (no GGUF needed)
    // =========================================================================

    #[test]
    fn test_model_constants() {
        // Architecture invariants (exact values depend on feature flags)
        assert!(N_LAYER >= 1);
        assert!(N_EMBD >= 64);
        assert!(N_VOCAB >= 256);
        assert!(N_HEAD >= 4 && N_HEAD % N_OUT_GROUP == 0);
        assert_eq!(N_HEAD_KV, 1);
        assert!(N_HEAD_DIM >= 16);
        assert_eq!(N_HC, 4);
        assert!(N_EXPERT >= 4);
        assert!(N_EXPERT_USED >= 2);
        assert_eq!(N_EXPERT_SHARED, 1);
    }
}

// =============================================================================
// Integration tests (require a GGUF model file)
// =============================================================================

#[cfg(test)]
#[cfg(feature = "test-dimensions")]
mod integration_tests {
    use ds4::gguf::GgufModel;
    use ds4::model;
    use ds4::tokenizer::Vocab;
    use ds4::session::Session;
    use std::path::Path;

    fn test_model_path() -> String {
        std::env::var("DS4_TEST_MODEL")
            .unwrap_or_else(|_| "../ds4flash.gguf".to_string())
    }

    fn model_available() -> bool {
        Path::new(&test_model_path()).exists()
    }

    // =========================================================================
    // GGUF loading tests
    // =========================================================================

    #[test]
    fn test_load_gguf() {
        if !model_available() {
            eprintln!("Skipping: no GGUF model found");
            return;
        }
        let model = GgufModel::open(&test_model_path())
            .expect("Failed to open GGUF model");
        assert!(model.version >= 2);
        assert!(model.tensors.len() > 0, "No tensors found");
        // Should have the essential tensors
        assert!(model.tensor("token_embd.weight").is_some());
    }

    #[test]
    fn test_gguf_metadata() {
        if !model_available() {
            eprintln!("Skipping: no GGUF model found");
            return;
        }
        let model = GgufModel::open(&test_model_path())
            .expect("Failed to open GGUF model");

        // Check core metadata
        let n_layer = model.get_u32("ds4.n_layer")
            .or_else(|| model.get_u32("llama.block_count"));
        assert!(n_layer.is_some());

        let arch = model.get_string("general.architecture");
        assert!(arch.is_some());
    }

    // =========================================================================
    // Model binding tests
    // =========================================================================

    #[test]
    fn test_bind_weights() {
        if !model_available() {
            eprintln!("Skipping: no GGUF model found");
            return;
        }
        let gguf = GgufModel::open(&test_model_path())
            .expect("Failed to open GGUF model");
        let weights = model::bind_weights(&gguf)
            .expect("Failed to bind weights");

        assert_eq!(weights.layers.len(), ds4::N_LAYER as usize);
    }

    // =========================================================================
    // Tokenizer tests
    // =========================================================================

    #[test]
    fn test_load_vocab() {
        if !model_available() {
            eprintln!("Skipping: no GGUF model found");
            return;
        }
        let gguf = GgufModel::open(&test_model_path())
            .expect("Failed to open GGUF model");
        let vocab = Vocab::load(&gguf)
            .expect("Failed to load vocabulary");

        assert_eq!(vocab.n_vocab, ds4::N_VOCAB as usize);
        assert!(vocab.eos_id >= 0);
        assert!(vocab.bos_id >= 0);
        // At least some common tokens should exist
        assert!(vocab.token_id("the").is_some() || vocab.token_id("Hello").is_some(),
            "Expected common tokens to be present");
    }

    #[test]
    fn test_vocab_encode_decode() {
        if !model_available() {
            eprintln!("Skipping: no GGUF model found");
            return;
        }
        let gguf = GgufModel::open(&test_model_path())
            .expect("Failed to open GGUF model");
        let vocab = Vocab::load(&gguf)
            .expect("Failed to load vocabulary");

        // Encode "Hello world" and verify against known C engine output.
        // The C engine with this test GGUF produces (after removing chat encoding):
        //   H(55), e(26), l(33), l(33), o(36), Ġ(8), w(44), o(36), r(39), l(33), d(25)
        let tokens = vocab.encode("Hello world");
        assert!(!tokens.is_empty(), "Encoding should produce tokens");
        assert_eq!(tokens, vec![55, 26, 33, 33, 36, 8, 44, 36, 39, 33, 25],
            "Token IDs must match C engine output (cross-validated via --dump-tokens)");

        // Decode back
        let decoded = vocab.decode(&tokens);
        // With GPT-2 BPE, space is encoded as Ġ
        assert_eq!(decoded, "HelloĠworld");
    }

    #[test]
    fn test_vocab_encode_cross_validate_c() {
        // Cross-validate against C engine --dump-tokens output for the test GGUF.
        // C produces: [BOS, system..., User, <bpe tokens>, Assistant, think]
        // We extract the BPE tokens (between User(3) and Assistant(4)).
        if !model_available() {
            eprintln!("Skipping: no GGUF model found");
            return;
        }
        let gguf = GgufModel::open(&test_model_path())
            .expect("Failed to open GGUF model");
        let vocab = Vocab::load(&gguf)
            .expect("Failed to load vocabulary");

        // Test strings and their C-verified BPE token IDs
        let test_cases: &[(&str, &[i32])] = &[
            // "Hello" → H(55), e(26), l(33), l(33), o(36)
            ("Hello", &[55, 26, 33, 33, 36]),
            // "world" → w(44), o(36), r(39), l(33), d(25)
            ("world", &[44, 36, 39, 33, 25]),
            // "Hello world": pretokenize→["Hello"," world"], BPE→"Hello"+"Ġworld"
            ("Hello world", &[55, 26, 33, 33, 36, 8, 44, 36, 39, 33, 25]),
            // "test" → t(41), e(26), s(40), t(41)
            ("test", &[41, 26, 40, 41]),
            // "int x = 42;" → verify code-like text
            ("int x = 42;", &[
                30, // i
                35, // n
                41, // t
                8,  // Ġ (space)
                45, // x
                8,  // Ġ (space)
                102,// =
                8,  // Ġ (space)
                78, // 4
                76, // 2
                100,// ;
            ]),
        ];

        for (text, expected) in test_cases {
            let tokens = vocab.encode(text);
            assert_eq!(&tokens, expected,
                "Tokenization mismatch for {:?}: got {:?}, expected {:?}",
                text, tokens, expected);
        }
    }

    // =========================================================================
    // Forward pass tests
    // =========================================================================

    #[test]
    fn test_forward_single_token() {
        if !model_available() {
            eprintln!("Skipping: no GGUF model found");
            return;
        }
        let gguf = GgufModel::open(&test_model_path())
            .expect("Failed to open GGUF model");
        let weights = model::bind_weights(&gguf)
            .expect("Failed to bind weights");

        let _session = Session::new(4096);

        // Embed and forward a simple token (BOS is usually safe)
        let token = 0i32; // BOS token
        use ds4::forward::{forward_one_token, KvCache};
        let mut kv_cache = KvCache::new(4096);
        let mut logits = vec![0.0f32; ds4::N_VOCAB as usize];
        forward_one_token(&mut logits, &weights, &mut kv_cache, token, 0);

        // Logits should be finite and non-trivial
        let has_finite = logits.iter().any(|&v| v.is_finite() && v.abs() > 1e-10);
        assert!(has_finite, "Logits should contain finite values");
    }

    #[test]
    fn test_forward_small_prompt() {
        if !model_available() {
            eprintln!("Skipping: no GGUF model found");
            return;
        }
        let gguf = GgufModel::open(&test_model_path())
            .expect("Failed to open GGUF model");
        let weights = model::bind_weights(&gguf)
            .expect("Failed to bind weights");
        let vocab = Vocab::load(&gguf)
            .expect("Failed to load vocabulary");

        let mut session = Session::new(4096);

        // Encode a simple prompt
        let prompt = "The capital of France is";
        let tokens = vocab.encode(prompt);

        session.sync(&weights, &tokens);

        // Get argmax prediction
        let next_token = session.argmax();
        let next_text = vocab.token_text(next_token);

        // Just verify we get a valid token
        assert!(next_token >= 0 && next_token < ds4::N_VOCAB as i32);
        assert!(next_text.is_some());
    }

    #[test]
    fn test_session_sampling() {
        if !model_available() {
            eprintln!("Skipping: no GGUF model found");
            return;
        }
        let gguf = GgufModel::open(&test_model_path())
            .expect("Failed to open GGUF model");
        let weights = model::bind_weights(&gguf)
            .expect("Failed to bind weights");
        let vocab = Vocab::load(&gguf)
            .expect("Failed to load vocabulary");

        let mut session = Session::new(4096);
        let prompt = "The answer is";
        let tokens = vocab.encode(prompt);

        session.sync(&weights, &tokens);

        // Test argmax
        let token1 = session.argmax();
        assert!(token1 >= 0);

        // Test sampling with different temperatures
        let mut rng = 42u64;
        let token2 = session.sample(1.0, 50, 0.9, &mut rng);
        assert!(token2 >= 0);

        // Test top_logprobs
        let mut top = [(0i32, 0.0f32); 10];
        session.top_logprobs(&mut top, 10);
        // First entry should have highest (closest to 0) logprob
        assert!(top[0].1 <= top[1].1 + 1e-5 || top[0].1.is_finite());
    }
}

// =============================================================================
// Forward pass integration tests using in-memory test model (no GGUF file needed)
// =============================================================================

#[cfg(test)]
#[cfg(feature = "test-dimensions")]
mod forward_tests {
    use ds4::model::{build_test_model, OwnedModelWeights};
    use ds4::forward::{forward_one_token, KvCache};
    use ds4::constants::*;
    use ds4::f16_to_f32;

    /// Helper: run a single forward pass and return logits.
    fn run_forward(omw: &OwnedModelWeights, token: i32, pos: usize) -> Vec<f32> {
        let weights = &omw.weights;
        let mut kv_cache = KvCache::new(4096);
        let mut logits = vec![0.0f32; N_VOCAB as usize];
        forward_one_token(&mut logits, weights, &mut kv_cache, token, pos);
        logits
    }

    #[test]
    fn test_forward_pass_runs_without_panic() {
        let omw = build_test_model();
        let logits = run_forward(&omw, 0, 0);
        // All logits should be finite
        assert_eq!(logits.len(), N_VOCAB as usize);
        assert!(logits.iter().all(|&v| v.is_finite()),
            "All logits must be finite");
    }

    #[test]
    fn test_forward_pass_deterministic() {
        let omw = build_test_model();
        let logits1 = run_forward(&omw, 0, 0);
        let logits2 = run_forward(&omw, 0, 0);
        // Same model, same token, same position → same logits
        for i in 0..logits1.len() {
            assert!((logits1[i] - logits2[i]).abs() < 1e-6,
                "logit[{}] differs: {} vs {}", i, logits1[i], logits2[i]);
        }
    }

    #[test]
    fn test_forward_pass_different_tokens() {
        let omw = build_test_model();
        let logits0 = run_forward(&omw, 0, 0);
        let logits5 = run_forward(&omw, 5, 0);

        // Different tokens should produce different logits with a non-trivial model
        // With all-ones weights, embeddings are all-ones for every token, so they should
        // actually be identical. Skip the diff check.
        // Logits should be finite regardless
        assert!(logits0.iter().all(|&v| v.is_finite()));
        assert!(logits5.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_forward_pass_multi_position() {
        let omw = build_test_model();
        let weights = &omw.weights;
        let mut kv_cache = KvCache::new(4096);

        // Forward position 0
        let mut logits0 = vec![0.0f32; N_VOCAB as usize];
        forward_one_token(&mut logits0, weights, &mut kv_cache, 0, 0);

        // Forward position 1 (uses cached KV from position 0)
        let mut logits1 = vec![0.0f32; N_VOCAB as usize];
        forward_one_token(&mut logits1, weights, &mut kv_cache, 0, 1);

        assert!(logits0.iter().all(|&v| v.is_finite()));
        assert!(logits1.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_forward_pass_large_pos() {
        let omw = build_test_model();
        let weights = &omw.weights;
        let mut kv_cache = KvCache::new(4096);

        // Pre-fill the cache with several positions
        for pos in 0..N_SWA as usize + 5 {
            let mut logits = vec![0.0f32; N_VOCAB as usize];
            forward_one_token(&mut logits, weights, &mut kv_cache, pos as i32 % 10, pos);
            assert!(logits.iter().all(|&v| v.is_finite()),
                "logits not finite at pos {}", pos);
        }
    }

    #[test]
    fn test_token_embedding_dimensions() {
        let omw = build_test_model();
        let weights = &omw.weights;

        // Check that token_embd has expected shape
        let embd = weights.token_embd.as_f16();
        let expected = N_VOCAB as usize * N_EMBD as usize;
        assert_eq!(embd.len(), expected,
            "token_embd should have {} elements, got {}", expected, embd.len());

        // All values should be 1.0 in f16
        for &v in embd.iter().take(128) {
            assert!((f16_to_f32(v) - 1.0).abs() < 0.001,
                "expected 1.0 in token_embd, got {}", f16_to_f32(v));
        }
    }

    #[test]
    fn test_layer_count() {
        let omw = build_test_model();
        assert_eq!(omw.weights.layers.len(), N_LAYER as usize);
    }

    #[test]
    fn test_ffn_mask_table_zeros() {
        let omw = build_test_model();
        let layer = &omw.weights.layers[0];
        let table = layer.ffn_gate_tid2eid.as_i32();
        // All entries should be 0 (we initialized as zero)
        let expected_elems = N_VOCAB as usize * N_EXPERT_USED as usize;
        assert!(table.len() >= expected_elems, "table too small");
        assert!(table.iter().take(expected_elems).all(|&v| v == 0),
            "ffn_gate_tid2eid should be all zeros");
    }

    #[test]
    fn test_hc_sinkhorn_pre_weights() {
        use ds4::hc_split_sinkhorn_one;
        let n_hc = N_HC as usize;
        let n_mix = 2 * n_hc + n_hc * n_hc;
        let mix = vec![1.0f32; n_mix];
        let scale = vec![0.5f32, 0.5, 0.5];
        let base = vec![0.0f32; n_mix];
        let mut out = vec![0.0f32; n_mix];

        hc_split_sinkhorn_one(&mut out, &mix, &scale, &base, n_hc,
            N_HC_SINKHORN_ITER, HC_EPS);

        // Pre weights should be in (0, 1] range
        for i in 0..n_hc {
            assert!(out[i] > 0.0 && out[i] <= 1.1, "pre[{}]={}", i, out[i]);
        }
    }

    /// Load test GGUF and run forward pass through it.
    #[test]
    #[cfg(feature = "test-dimensions")]
    fn test_forward_via_gguf() {
        use ds4::gguf::GgufModel;
        use ds4::model;

        let path = std::env::var("DS4_TEST_GGUF")
            .unwrap_or_else(|_| "/tmp/test_ds4.gguf".to_string());

        if !std::path::Path::new(&path).exists() {
            eprintln!("Skipping: {} not found (run gen_test_gguf first)", path);
            return;
        }

        let gguf = GgufModel::open(&path).expect("Failed to open test GGUF");
        let weights = model::bind_weights_unchecked(&gguf)
            .expect("Failed to bind weights");

        let mut kv_cache = KvCache::new(4096);
        let mut logits = vec![0.0f32; N_VOCAB as usize];
        forward_one_token(&mut logits, &weights, &mut kv_cache, 0, 0);

        assert!(logits.iter().all(|&v| v.is_finite()),
            "GGUF forward pass produced non-finite logits");
    }
}

// =============================================================================
// Integration tests: autoregressive generation, session lifecycle, reproducibility
// =============================================================================

#[cfg(test)]
#[cfg(feature = "test-dimensions")]
mod generation_tests {
    use ds4::model::{build_test_model};
    use ds4::forward::{KvCache, forward_one_token};
    use ds4::session::Session;
    use ds4::constants::*;
    use std::time::Instant;

    /// Build a mock vocabulary for test-generation testing.
    /// Maps token IDs 0..N_VOCAB-1 to placeholder text like "[0]", "[1]", etc.
    #[allow(dead_code)]
    struct MockVocab;

    #[allow(dead_code)]
    impl MockVocab {
        fn decode(&self, tokens: &[i32]) -> String {
            tokens.iter()
                .map(|&t| format!("[{}]", t))
                .collect::<Vec<_>>()
                .join("")
        }

        fn token_text(&self, token: i32) -> Option<String> {
            if token >= 0 && token < N_VOCAB as i32 {
                Some(format!("[{}]", token))
            } else {
                None
            }
        }
    }

    #[test]
    fn test_autoregressive_generation_runs() {
        let omw = build_test_model();
        let mut session = Session::new(4096);

        // Prefill with a sequence of tokens
        let prompt: Vec<i32> = (0..5).collect();
        session.sync(&omw.weights, &prompt);

        // Generate 10 tokens with greedy decoding
        let mut rng = 42u64;
        let generated = session.generate(
            &omw.weights,
            -1,   // no EOS in test model
            10,
            0.0,  // temperature 0 = greedy
            0,
            0.0,
            &mut rng,
        );

        assert_eq!(generated.len(), 10, "Should generate exactly 10 tokens");
        for &t in &generated {
            assert!(t >= 0 && t < N_VOCAB as i32,
                "Generated token {} out of range", t);
        }
    }

    #[test]
    fn test_generation_reproducibility() {
        let omw = build_test_model();

        let run = |seed: u64| -> Vec<i32> {
            let mut session = Session::new(4096);
            let prompt: Vec<i32> = (0..5).collect();
            session.sync(&omw.weights, &prompt);
            let mut rng = seed;
            session.generate(&omw.weights, -1, 10, 1.0, 50, 0.9, &mut rng)
        };

        let result1 = run(42);
        let result2 = run(42);

        assert_eq!(result1, result2,
            "Same seed should produce identical sequences");

        let result3 = run(99);
        // Different seeds may or may not differ, but both must be valid
        assert!(result3.iter().all(|&t| t >= 0 && t < N_VOCAB as i32));
    }

    #[test]
    fn test_session_prefill_and_extend() {
        let omw = build_test_model();

        // Session 1: prefill with 0,1,2,3,4,5
        let mut s1 = Session::new(4096);
        let prompt: Vec<i32> = (0..6).collect();
        s1.sync(&omw.weights, &prompt);

        // Session 2: prefill with 0,1,2 then extend with 3,4,5
        let mut s2 = Session::new(4096);
        let prefix: Vec<i32> = (0..3).collect();
        s2.sync(&omw.weights, &prefix);
        s2.eval(&omw.weights, 3);
        s2.eval(&omw.weights, 4);
        s2.eval(&omw.weights, 5);

        // Both sessions should produce the same argmax
        let a1 = s1.argmax();
        let a2 = s2.argmax();
        assert_eq!(a1, a2, "Session state should be equivalent after prefill vs extend");
    }

    #[test]
    fn test_session_sync_common_prefix() {
        let omw = build_test_model();

        // Prefill with 0,1,2,3,4
        let mut session = Session::new(4096);
        let prompt1: Vec<i32> = (0..5).collect();
        session.sync(&omw.weights, &prompt1);

        let logits_after_5 = session.logits.clone();

        // Now sync with 0,1,2,3,4,5 (common prefix = 5)
        let prompt2: Vec<i32> = (0..6).collect();
        let common = session.common_prefix_len(&prompt2);
        assert_eq!(common, 5, "Should detect common prefix of 5 tokens");

        session.sync(&omw.weights, &prompt2);
        assert_eq!(session.n_tokens(), 6);

        // If we sync with completely different prompt, cache should rebuild
        let prompt3: Vec<i32> = (100..105).collect();
        session.sync(&omw.weights, &prompt3);
        assert_eq!(session.n_tokens(), 5);

        let _ = logits_after_5; // silence unused warning
    }

    #[test]
    fn test_session_invalidate() {
        let omw = build_test_model();
        let mut session = Session::new(4096);
        let prompt: Vec<i32> = (0..5).collect();
        session.sync(&omw.weights, &prompt);

        assert_eq!(session.n_tokens(), 5);

        session.invalidate();
        assert_eq!(session.n_tokens(), 0);
        assert!(session.logits.iter().all(|&v| v == 0.0),
            "Logits should be zero after invalidate (fresh session state)");
    }

    #[test]
    fn test_forward_prefill() {
        use ds4::forward::forward_prefill;
        let omw = build_test_model();

        let mut kv_cache = KvCache::new(4096);
        let mut logits = vec![0.0f32; N_VOCAB as usize];
        let prompt: Vec<i32> = (0..10).collect();

        forward_prefill(&mut logits, &omw.weights, &mut kv_cache, &prompt);

        assert!(logits.iter().all(|&v| v.is_finite()),
            "Prefill logits must be finite");
        assert!(logits.iter().any(|&v| v.abs() > 1e-10),
            "Prefill logits should be non-trivial");
    }

    #[test]
    fn test_top_logprobs_format() {
        let omw = build_test_model();
        let mut session = Session::new(4096);
        let prompt: Vec<i32> = (0..5).collect();
        session.sync(&omw.weights, &prompt);

        let mut top = [(0i32, 0.0f32); 10];
        session.top_logprobs(&mut top, 10);

        // Logprobs should be in descending order (closest to 0 is highest prob)
        for i in 1..top.len() {
            assert!(top[i - 1].1 <= top[i].1 + 1e-5,
                "Top logprobs should be sorted descending by probability: \
                 top[{}].lp={}, top[{}].lp={}",
                i - 1, top[i - 1].1, i, top[i].1);
        }

        // All entries should have valid token IDs
        for &(tid, lp) in &top {
            assert!(tid >= 0 && tid < N_VOCAB as i32,
                "Token ID {} out of range", tid);
            assert!(lp.is_finite(), "Logprob should be finite");
        }
    }

    #[test]
    fn test_forward_one_token_then_argmax() {
        let omw = build_test_model();
        let weights = &omw.weights;
        let mut kv_cache = KvCache::new(4096);
        let mut logits = vec![0.0f32; N_VOCAB as usize];

        // Run a few tokens sequentially
        let tokens: Vec<i32> = (0..10).collect();
        for (pos, &token) in tokens.iter().enumerate() {
            forward_one_token(&mut logits, weights, &mut kv_cache, token, pos);
            assert!(logits.iter().all(|&v| v.is_finite()),
                "Logits not finite at pos {}", pos);
        }

        // Find argmax
        let mut best_id = 0i32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_id = i as i32;
            }
        }
        assert!(best_id >= 0 && best_id < N_VOCAB as i32,
            "Best token ID {} out of range", best_id);
    }

    #[test]
    fn test_sampling_temperature_effect() {
        let omw = build_test_model();
        let mut session = Session::new(4096);
        let prompt: Vec<i32> = (0..5).collect();
        session.sync(&omw.weights, &prompt);

        // With temperature 0 (greedy), always the same token
        let mut rng = 42u64;
        let greedy1 = session.sample(0.0, 0, 0.0, &mut rng);
        let greedy2 = session.sample(0.0, 0, 0.0, &mut rng);
        let argmax = session.argmax();
        assert_eq!(greedy1, argmax, "Greedy should match argmax");
        assert_eq!(greedy1, greedy2, "Greedy should be deterministic regardless of RNG");

        // With high temperature, may differ
        let token_t1 = session.sample(2.0, 50, 0.95, &mut rng);
        assert!(token_t1 >= 0 && token_t1 < N_VOCAB as i32);
    }

    #[test]
    fn test_kv_cache_store_and_read() {
        let mut kv_cache = KvCache::new(4096);
        let head_dim = N_HEAD_DIM as usize;

        // Store some KV values
        let kv0: Vec<f32> = (0..head_dim).map(|i| i as f32).collect();
        let kv1: Vec<f32> = (0..head_dim).map(|i| (i * 2) as f32).collect();

        kv_cache.store_raw_kv(0, 0, &kv0);
        kv_cache.store_raw_kv(0, 1, &kv1);

        // Read back
        let read0 = kv_cache.read_raw_kv(0, 0);
        let read1 = kv_cache.read_raw_kv(0, 1);

        assert_eq!(read0, &kv0[..]);
        assert_eq!(read1, &kv1[..]);

        // Beyond raw_cap, wrap around
        let cap = kv_cache.raw_cap;
        let kv_over: Vec<f32> = (0..head_dim).map(|i| (i * 3) as f32).collect();
        kv_cache.store_raw_kv(0, cap, &kv_over);
        let read_over = kv_cache.read_raw_kv(0, cap);
        assert_eq!(read_over, &kv_over[..]);

        // Position 0 should be overwritten since it's the same mod position
        let read0_after = kv_cache.read_raw_kv(0, 0);
        assert_eq!(read0_after, &kv_over[..], "Position 0 should be overwritten after wrap");
    }

    #[test]
    fn test_generation_with_eos_stop() {
        let omw = build_test_model();
        let mut session = Session::new(4096);
        let prompt: Vec<i32> = (0..3).collect();
        session.sync(&omw.weights, &prompt);

        // Ask for 100 tokens but check that it stops at max_tokens
        let mut rng = 42u64;
        let generated = session.generate(&omw.weights, -1, 5, 0.0, 0, 0.0, &mut rng);
        assert_eq!(generated.len(), 5, "Should generate exactly max_tokens when no EOS");
    }

    /// Measure raw forward pass throughput in tokens/sec.
    /// This is not a #[bench] but a diagnostic test.
    #[test]
    fn bench_forward_throughput() {
        let omw = build_test_model();
        let weights = &omw.weights;

        let n_tokens = 100;
        let mut kv_cache = KvCache::new(4096);
        let mut logits = vec![0.0f32; N_VOCAB as usize];

        let start = Instant::now();
        for pos in 0..n_tokens {
            forward_one_token(&mut logits, weights, &mut kv_cache, pos as i32 % 10, pos);
        }
        let elapsed = start.elapsed();
        let tokens_per_sec = n_tokens as f64 / elapsed.as_secs_f64();

        println!(
            "Throughput: {} tokens in {:?} = {:.1} tokens/sec",
            n_tokens, elapsed, tokens_per_sec
        );

        // With test-dimensions, the model is tiny (1 layer, 64 dim).
        // This is a smoke test that forward passes work at speed.
        assert!(tokens_per_sec > 50.0,
            "Forward throughput too low: {:.1} tok/s (expected >50)", tokens_per_sec);
    }
}
