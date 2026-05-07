// Tokenizer: GPT-2 style byte-level BPE with JoyAI pre-tokenizer.
// Loads token strings and merge ranks from GGUF metadata.

use crate::gguf::{GgufModel, GgufValue};
use anyhow::{bail, Result};
use std::collections::HashMap;

/// A loaded vocabulary.
pub struct Vocab {
    /// Token strings by ID.
    pub tokens: Vec<String>,
    /// Token count.
    pub n_vocab: usize,
    /// Special token IDs.
    pub bos_id: i32,
    pub eos_id: i32,
    pub user_id: i32,
    pub assistant_id: i32,
    pub think_start_id: i32,
    pub think_end_id: i32,
    pub dsml_id: i32,
    /// Token string to ID lookup.
    token_to_id: HashMap<String, i32>,
    /// BPE merge rank lookup.
    merge_ranks: HashMap<(String, String), i32>,
    /// Decoding: ID to text.
    id_to_text: Vec<String>,
}

impl Vocab {
    /// Load vocabulary from GGUF model metadata.
    pub fn load(model: &GgufModel) -> Result<Self> {
        let n_vocab = model.get_u32("ds4.vocab_size")
            .or_else(|| model.get_u32("llama.vocab_size"))
            .ok_or_else(|| anyhow::anyhow!("missing vocabulary size"))? as usize;

        // Read token strings
        let mut tokens = Vec::with_capacity(n_vocab);
        let mut token_to_id = HashMap::new();

        // In GGUF, token strings are stored as metadata array under "tokenizer.ggml.tokens"
        let token_arr = match model.kv.get("tokenizer.ggml.tokens") {
            Some(GgufValue::Array(arr)) => {
                arr.elements.iter()
                    .filter_map(|v| match v {
                        GgufValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            }
            _ => bail!("missing tokenizer.ggml.tokens in GGUF metadata"),
        };

        if token_arr.len() != n_vocab {
            bail!("token count mismatch: expected {}, got {}", n_vocab, token_arr.len());
        }

        for (i, tok) in token_arr.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as i32);
            tokens.push(tok.clone());
        }

        // Read BPE merge ranks
        let mut merge_ranks = HashMap::new();
        let merges_arr = match model.kv.get("tokenizer.ggml.merges") {
            Some(GgufValue::Array(arr)) => {
                arr.elements.iter()
                    .filter_map(|v| match v {
                        GgufValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            }
            _ => Vec::new(),
        };

        for (rank, merge) in merges_arr.iter().enumerate() {
            let parts: Vec<&str> = merge.split(' ').collect();
            if parts.len() == 2 {
                merge_ranks.insert(
                    (parts[0].to_string(), parts[1].to_string()),
                    rank as i32,
                );
            }
        }

        // Special tokens
        let bos_id = model.get_u32("tokenizer.ggml.bos_token_id")
            .unwrap_or(0) as i32;
        let eos_id = model.get_u32("tokenizer.ggml.eos_token_id")
            .unwrap_or(1) as i32;

        // Find special token IDs by name
        let user_id = token_to_id.get("的真实问题").copied()
            .unwrap_or_else(|| token_to_id.get("<|User|>").copied().unwrap_or(-1));
        let assistant_id = token_to_id.get("的真实回答").copied()
            .unwrap_or_else(|| token_to_id.get("<|Assistant|>").copied().unwrap_or(-1));
        let think_start_id = token_to_id.get("比如").copied()
            .unwrap_or(-1);
        let think_end_id = token_to_id.get("比如还有").copied()
            .unwrap_or(-1);
        let dsml_id = token_to_id.get("在校期间及").copied()
            .unwrap_or(-1);

        // Build ID-to-text for decoding
        let id_to_text = tokens.clone();

        Ok(Vocab {
            tokens,
            n_vocab,
            bos_id,
            eos_id,
            user_id,
            assistant_id,
            think_start_id,
            think_end_id,
            dsml_id,
            token_to_id,
            merge_ranks,
            id_to_text,
        })
    }

    /// Look up token ID for a single token string.
    pub fn token_id(&self, token: &str) -> Option<i32> {
        self.token_to_id.get(token).copied()
    }

    /// Look up token text by ID.
    pub fn token_text(&self, id: i32) -> Option<&str> {
        if id < 0 || id >= self.id_to_text.len() as i32 {
            return None;
        }
        Some(&self.id_to_text[id as usize])
    }

    /// Decode a sequence of token IDs to text.
    pub fn decode(&self, ids: &[i32]) -> String {
        let mut result = String::new();
        for &id in ids {
            if let Some(text) = self.token_text(id) {
                result.push_str(text);
            }
        }
        result
    }

    /// Encode text to token IDs using byte-level BPE with JoyAI-style pre-tokenization.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        encode_bpe(self, text)
    }
}

/// GPT-2 byte-to-unicode mapping.
fn gpt2_byte_to_unicode(b: u8) -> char {
    if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174 {
        return b as char;
    }
    let mut n = 0u32;
    for x in 0u8..=255 {
        if (x >= 33 && x <= 126) || (x >= 161 && x <= 172) || x >= 174 {
            continue;
        }
        if x == b { return char::from_u32(256 + n).unwrap(); }
        n += 1;
    }
    b as char
}

/// Byte-encode: map raw bytes to unicode codepoints for BPE.
pub fn byte_encode(s: &str) -> String {
    s.bytes().map(|b| gpt2_byte_to_unicode(b)).collect()
}

/// UTF-8 character length from first byte.
fn utf8_len(c: u8) -> usize {
    if c < 0x80 { 1 }
    else if (c & 0xe0) == 0xc0 { 2 }
    else if (c & 0xf0) == 0xe0 { 3 }
    else if (c & 0xf8) == 0xf0 { 4 }
    else { 1 }
}

/// Decode one UTF-8 codepoint from a byte slice.
fn utf8_decode_one(s: &[u8], pos: usize) -> (u32, usize) {
    let c0 = s[pos];
    let n = utf8_len(c0);
    if pos + n > s.len() { return (c0 as u32, 1); }

    match n {
        1 => (c0 as u32, 1),
        2 => (
            ((c0 as u32 & 0x1f) << 6) | (s[pos + 1] as u32 & 0x3f),
            2,
        ),
        3 => (
            ((c0 as u32 & 0x0f) << 12) | ((s[pos + 1] as u32 & 0x3f) << 6) | (s[pos + 2] as u32 & 0x3f),
            3,
        ),
        _ => (
            ((c0 as u32 & 0x07) << 18) | ((s[pos + 1] as u32 & 0x3f) << 12) | ((s[pos + 2] as u32 & 0x3f) << 6) | (s[pos + 3] as u32 & 0x3f),
            4,
        ),
    }
}

fn is_ascii_alpha(c: u8) -> bool {
    (c >= b'A' && c <= b'Z') || (c >= b'a' && c <= b'z')
}

fn is_ascii_digit(c: u8) -> bool {
    c >= b'0' && c <= b'9'
}

fn is_ascii_space(c: u8) -> bool {
    c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' || c == 0x0b || c == 0x0c
}

fn is_ascii_newline(c: u8) -> bool {
    c == b'\n' || c == b'\r'
}

fn is_ascii_punct(c: u8) -> bool {
    (c >= b'!' && c <= b'/') ||
    (c >= b':' && c <= b'@') ||
    (c >= b'[' && c <= b'`') ||
    (c >= b'{' && c <= b'~')
}

fn is_cjk_hira_kata(cp: u32) -> bool {
    (cp >= 0x4e00 && cp <= 0x9fa5) ||
    (cp >= 0x3040 && cp <= 0x309f) ||
    (cp >= 0x30a0 && cp <= 0x30ff)
}

/// JoyAI-style pre-tokenizer: split text into pieces, then apply BPE.
fn encode_bpe(vocab: &Vocab, text: &str) -> Vec<i32> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;

    while pos < len {
        let c = bytes[pos];

        // Skip spaces
        if is_ascii_space(c) && !is_ascii_newline(c) {
            pos += 1;
            continue;
        }

        // Newlines
        if is_ascii_newline(c) {
            pos += 1;
            continue;
        }

        // CJK / Hiragana / Katakana
        if c >= 128 {
            let (cp, adv) = utf8_decode_one(bytes, pos);
            if is_cjk_hira_kata(cp) {
                let piece = String::from_utf8_lossy(&bytes[pos..pos + adv]).to_string();
                bpe_emit_piece(vocab, &piece, &mut out);
                pos += adv;
                continue;
            }
            // Non-CJK: fall through to letter processing
        }

        // ASCII letters / digits / other non-CJK unicode
        if (c < 128 && (is_ascii_alpha(c) || is_ascii_digit(c))) || c >= 128 {
            // Consume a run of letters/digits/unicode
            let start = pos;
            while pos < len {
                let c2 = bytes[pos];
                if c2 < 128 {
                    if !is_ascii_alpha(c2) && !is_ascii_digit(c2) { break; }
                    pos += 1;
                } else {
                    let (cp2, _) = utf8_decode_one(bytes, pos);
                    if is_cjk_hira_kata(cp2) { break; }
                    pos += utf8_len(c2);
                }
            }
            let piece = String::from_utf8_lossy(&bytes[start..pos]).to_string();
            bpe_emit_piece(vocab, &piece, &mut out);
            continue;
        }

        // Punctuation/symbols
        if c < 128 && is_ascii_punct(c) {
            pos += 1;
            // Single char as piece
            let piece = String::from_utf8_lossy(&bytes[pos - 1..pos]).to_string();
            bpe_emit_piece(vocab, &piece, &mut out);
            continue;
        }

        // Fallback: advance one char
        pos += 1;
    }

    out
}

/// Apply BPE to one pre-tokenized piece and emit token IDs.
fn bpe_emit_piece(vocab: &Vocab, piece: &str, out: &mut Vec<i32>) {
    let encoded = byte_encode(piece);
    let chars: Vec<char> = encoded.chars().collect();
    let mut symbols: Vec<String> = chars.iter().map(|c| c.to_string()).collect();

    // Apply BPE merges
    loop {
        let mut best_rank = i32::MAX;
        let mut best_i = None;

        for i in 0..symbols.len().saturating_sub(1) {
            let key = (symbols[i].clone(), symbols[i + 1].clone());
            if let Some(&rank) = vocab.merge_ranks.get(&key) {
                if rank < best_rank {
                    best_rank = rank;
                    best_i = Some(i);
                }
            }
        }

        match best_i {
            Some(i) => {
                let merged = format!("{}{}", symbols[i], symbols[i + 1]);
                symbols[i] = merged;
                symbols.remove(i + 1);
            }
            None => break,
        }
    }

    // Emit token IDs for final symbols
    for sym in &symbols {
        if let Some(&id) = vocab.token_to_id.get(sym.as_str()) {
            out.push(id);
        } else {
            // Fallback: emit byte-level tokens
            for b in sym.bytes() {
                let byte_token = format!("<0x{:02X}>", b);
                if let Some(&id) = vocab.token_to_id.get(&byte_token) {
                    out.push(id);
                } else {
                    // Try single-char lookup
                    let ch = b as char;
                    if let Some(&id) = vocab.token_to_id.get(&ch.to_string()) {
                        out.push(id);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_encode_simple() {
        let encoded = byte_encode("Hello");
        assert_eq!(encoded, "Hello");
    }

    #[test]
    fn test_utf8_len() {
        assert_eq!(utf8_len(b'a'), 1);
        assert_eq!(utf8_len(0xc2), 2); // Â
    }
}
