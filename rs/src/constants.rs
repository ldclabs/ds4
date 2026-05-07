// Model architecture constants for DeepSeek V4 Flash.
//
// In production mode these match the real DS4 architecture (43 layers, 4096-dim, 129280 vocab).
// With the `test-dimensions` feature, they shrink to tiny values suitable for fast unit tests
// without requiring an 81 GB model file.

// ============================================================================
// Production constants (full DS4)
// ============================================================================
#[cfg(not(feature = "test-dimensions"))]
mod prod {
    pub const N_LAYER: u32 = 43;
    pub const N_EMBD: u32 = 4096;
    pub const N_VOCAB: u32 = 129280;
    pub const N_HEAD: u32 = 64;
    pub const N_HEAD_KV: u32 = 1;
    pub const N_HEAD_DIM: u32 = 512;
    pub const N_VALUE_DIM: u32 = 512;
    pub const N_ROT: u32 = 64;
    pub const N_OUT_GROUP: u32 = 8;
    pub const N_LORA_Q: u32 = 1024;
    pub const N_LORA_O: u32 = 1024;
    pub const N_EXPERT: u32 = 256;
    pub const N_EXPERT_USED: u32 = 6;
    pub const N_EXPERT_SHARED: u32 = 1;
    pub const N_FF_EXP: u32 = 2048;
    pub const N_HASH_LAYER: u32 = 3;
    pub const N_SWA: u32 = 128;
    pub const N_INDEXER_HEAD: u32 = 64;
    pub const N_INDEXER_HEAD_DIM: u32 = 128;
    pub const N_INDEXER_TOP_K: u32 = 512;
    pub const N_HC: u32 = 4;
    pub const N_HC_SINKHORN_ITER: u32 = 20;
}

// ============================================================================
// Test constants (tiny, for fast unit/integration tests)
// ============================================================================
#[cfg(feature = "test-dimensions")]
mod prod {
    pub const N_LAYER: u32 = 1;
    pub const N_EMBD: u32 = 64;
    pub const N_VOCAB: u32 = 256;
    pub const N_HEAD: u32 = 4;
    pub const N_HEAD_KV: u32 = 1;
    pub const N_HEAD_DIM: u32 = 16;
    pub const N_VALUE_DIM: u32 = 16;
    pub const N_ROT: u32 = 8;
    pub const N_OUT_GROUP: u32 = 2;
    pub const N_LORA_Q: u32 = 16;
    pub const N_LORA_O: u32 = 16;
    pub const N_EXPERT: u32 = 4;
    pub const N_EXPERT_USED: u32 = 2;
    pub const N_EXPERT_SHARED: u32 = 1;
    pub const N_FF_EXP: u32 = 32;
    pub const N_HASH_LAYER: u32 = 2;
    pub const N_SWA: u32 = 8;
    pub const N_INDEXER_HEAD: u32 = 4;
    pub const N_INDEXER_HEAD_DIM: u32 = 8;
    pub const N_INDEXER_TOP_K: u32 = 16;
    pub const N_HC: u32 = 4;
    pub const N_HC_SINKHORN_ITER: u32 = 5;
}

// Re-export
pub use prod::*;

// Numerical constants (same for both modes)
pub const NEG_INF: f32 = -1.0e30;
pub const POS_INF: f32 = 1.0e30;
pub const RMS_EPS: f32 = 1.0e-6;
pub const HC_EPS: f32 = 1.0e-6;
pub const EXPERT_WEIGHT_SCALE: f32 = 1.5;
pub const SWIGLU_CLAMP_EXP: f32 = 10.0;
pub const ROPE_FREQ_BASE: f32 = 10000.0;
pub const ROPE_SCALE_FACTOR: f32 = 16.0;
pub const ROPE_YARN_BETA_FAST: f32 = 32.0;
pub const ROPE_YARN_BETA_SLOW: f32 = 1.0;
pub const COMPRESS_ROPE_FREQ_BASE: f32 = 160000.0;
pub const ROPE_ORIG_CTX: u64 = 65536;

pub const REASONING_EFFORT_MAX_PREFIX: &str =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n\
     You MUST be very thorough in your thinking and comprehensively decompose \
     the problem to resolve the root cause, rigorously stress-testing your logic \
     against all potential paths, edge cases, and adversarial scenarios.\n\
     Explicitly write out your entire deliberation process, documenting every \
     intermediate step, considered alternative, and rejected hypothesis to \
     ensure absolutely no assumption is left unchecked.\n\n";

pub const THINK_MAX_MIN_CONTEXT: u32 = 393216;
