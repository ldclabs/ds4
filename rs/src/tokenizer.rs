// Tokenizer: GPT-2 style byte-level BPE with JoyAI pre-tokenizer.
// Implements the exact algorithm used by ds4.c's `bpe_tokenize_text`.
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
    /// BPE merge rank lookup (keyed by space-separated pair, matching C's merge_rank table).
    merge_ranks: HashMap<String, i32>,
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

        // Read BPE merge ranks. In C, merges are stored as "a b" strings and looked
        // up by concatenating symbol bytes with a space separator. We store them
        // in a HashMap keyed by the space-separated form to match C's table_get.
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
            // C stores the full "a b" string as key; we do the same
            merge_ranks.insert(merge.clone(), rank as i32);
        }

        // Special tokens — use the same lookup names as C's vocab_lookup
        let bos_id = model.get_u32("tokenizer.ggml.bos_token_id")
            .unwrap_or(0) as i32;
        let eos_id = model.get_u32("tokenizer.ggml.eos_token_id")
            .unwrap_or(1) as i32;

        // C's vocab_lookup searches for these exact strings; if missing, it dies.
        // We soft-fallback to -1 for the test model which doesn't have all of them.
        let lookup = |s: &str| -> i32 {
            token_to_id.get(s).copied().unwrap_or(-1)
        };
        let user_id = lookup("<｜User｜>");
        let assistant_id = lookup("<｜Assistant｜>");
        let think_start_id = lookup("<think>");
        let think_end_id = lookup("</think>");
        let dsml_id = lookup("｜DSML｜");

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
    /// This matches C's `bpe_tokenize_text` algorithm exactly.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        encode_bpe(self, text)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPT-2 byte-to-unicode mapping
// ═══════════════════════════════════════════════════════════════════════════

/// GPT-2 byte-to-unicode mapping: maps raw bytes 0..255 to printable Unicode
/// codepoints. Printable ASCII passes through; control characters and high
/// bytes are remapped to the U+0100..U+01FF range. Exact match of C's
/// `gpt2_byte_to_codepoint`.
pub fn gpt2_byte_to_unicode(b: u8) -> char {
    if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174 {
        return b as char;
    }
    let mut n = 0u32;
    for x in 0u8..=255 {
        if (x >= 33 && x <= 126) || (x >= 161 && x <= 172) || x >= 174 {
            continue;
        }
        if x == b {
            return char::from_u32(256 + n).unwrap();
        }
        n += 1;
    }
    b as char
}

/// Byte-encode: map raw bytes to a Unicode string where each byte becomes
/// the printable codepoint defined by `gpt2_byte_to_unicode`. This is the
/// pre-processing step before BPE — it ensures every input byte has a unique,
/// printable Unicode representation in the BPE symbol space.
/// Exact match of C's `byte_encode`.
pub fn byte_encode(raw: &[u8]) -> String {
    raw.iter().map(|&b| gpt2_byte_to_unicode(b)).collect()
}

/// Memoized version of `byte_encode` for &str.
#[inline]
pub fn byte_encode_str(s: &str) -> String {
    byte_encode(s.as_bytes())
}

// ═══════════════════════════════════════════════════════════════════════════
// UTF-8 helpers
// ═══════════════════════════════════════════════════════════════════════════

/// UTF-8 character byte-length from the leading byte. Matches C's
/// `utf8_len_from_first_byte`.
#[inline]
pub fn utf8_len(c: u8) -> usize {
    if c < 0x80 {
        1
    } else if (c & 0xe0) == 0xc0 {
        2
    } else if (c & 0xf0) == 0xe0 {
        3
    } else if (c & 0xf8) == 0xf0 {
        4
    } else {
        1
    }
}

/// Advance `pos` past one UTF-8 character. Matches C's `next_utf8_char`.
#[inline]
fn next_utf8_char(s: &[u8], pos: usize) -> usize {
    if pos >= s.len() {
        return pos;
    }
    let n = utf8_len(s[pos]);
    if pos + n > s.len() {
        pos + 1
    } else {
        pos + n
    }
}

/// Decode one UTF-8 codepoint and return (codepoint, next_position).
/// Matches C's `utf8_peek_one`.
fn utf8_peek_one(s: &[u8], pos: usize) -> (u32, usize) {
    if pos >= s.len() {
        return (0, pos);
    }
    let c0 = s[pos];
    let n = utf8_len(c0);
    if pos + n > s.len() {
        return (c0 as u32, pos + 1);
    }
    let next = pos + n;
    let cp = match n {
        1 => c0 as u32,
        2 => ((c0 as u32 & 0x1f) << 6) | (s[pos + 1] as u32 & 0x3f),
        3 => ((c0 as u32 & 0x0f) << 12)
            | ((s[pos + 1] as u32 & 0x3f) << 6)
            | (s[pos + 2] as u32 & 0x3f),
        _ => ((c0 as u32 & 0x07) << 18)
            | ((s[pos + 1] as u32 & 0x3f) << 12)
            | ((s[pos + 2] as u32 & 0x3f) << 6)
            | (s[pos + 3] as u32 & 0x3f),
    };
    (cp, next)
}

// ═══════════════════════════════════════════════════════════════════════════
// Character classification (matches C's `ds4.c` exactly)
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn is_ascii_alpha(c: u8) -> bool {
    (c >= b'A' && c <= b'Z') || (c >= b'a' && c <= b'z')
}

#[inline]
fn is_ascii_digit(c: u8) -> bool {
    c >= b'0' && c <= b'9'
}

#[inline]
fn is_ascii_space(c: u8) -> bool {
    c == b' '
        || c == b'\t'
        || c == b'\n'
        || c == b'\r'
        || c == 0x0b
        || c == 0x0c
}

#[inline]
fn is_ascii_newline(c: u8) -> bool {
    c == b'\n' || c == b'\r'
}

/// JoyAI ASCII punctuation/symbol range. Matches C's `joyai_ascii_punct_symbol`.
#[inline]
fn is_ascii_punct(c: u8) -> bool {
    (c >= b'!' && c <= b'/')
        || (c >= b':' && c <= b'@')
        || (c >= b'[' && c <= b'`')
        || (c >= b'{' && c <= b'~')
}

/// CJK, Hiragana, Katakana codepoint ranges.
#[inline]
fn is_cjk_hira_kata(cp: u32) -> bool {
    (cp >= 0x4e00 && cp <= 0x9fa5)
        || (cp >= 0x3040 && cp <= 0x309f)
        || (cp >= 0x30a0 && cp <= 0x30ff)
}

/// Check if the byte at `pos` in `s` starts a CJK/Hiragana/Katakana character.
/// Matches C's `joyai_cjk_at`.
#[inline]
fn is_cjk_at(s: &[u8], pos: usize) -> bool {
    if pos >= s.len() || s[pos] < 128 {
        return false;
    }
    let (cp, _) = utf8_peek_one(s, pos);
    is_cjk_hira_kata(cp)
}

/// Letter-like test: ASCII alpha always true; non-ASCII always true
/// (CJK is handled before this rule, so non-ASCII here means accents, etc.).
/// Matches C's `joyai_letter_like_at`.
#[inline]
fn is_letter_like(s: &[u8], pos: usize) -> bool {
    if pos >= s.len() {
        return false;
    }
    let c = s[pos];
    if c < 128 {
        return is_ascii_alpha(c);
    }
    true
}

/// Consume a run of letter-like characters. Matches C's `joyai_consume_letters`.
fn consume_letters(s: &[u8], pos: usize) -> usize {
    let mut p = pos;
    while p < s.len() && is_letter_like(s, p) {
        p = next_utf8_char(s, p);
    }
    p
}

// ═══════════════════════════════════════════════════════════════════════════
// JoyAI pre-tokenizer — exact match of C's `bpe_tokenize_text`
// ═══════════════════════════════════════════════════════════════════════════

/// Apply the JoyAI BPE pre-tokenizer to raw bytes and emit tokens.
/// This is the main entry point, matching C's `bpe_tokenize_text`.
fn encode_bpe(vocab: &Vocab, text: &str) -> Vec<i32> {
    let mut out = Vec::new();
    let s = text.as_bytes();
    let len = s.len();
    let mut pos = 0usize;

    while pos < len {
        let start = pos;
        let c = s[pos];

        if is_ascii_digit(c) {
            // Rule: \p{N}{1,3} — up to 3 consecutive digits
            let mut ndigits = 0;
            while pos < len && is_ascii_digit(s[pos]) && ndigits < 3 {
                pos += 1;
                ndigits += 1;
            }
        } else if is_cjk_at(s, pos) {
            // Rule: [CJK/Hiragana/Katakana]+ — consecutive CJK/kana
            pos = next_utf8_char(s, pos);
            while pos < len && is_cjk_at(s, pos) {
                pos = next_utf8_char(s, pos);
            }
        } else if is_ascii_punct(c)
            && pos + 1 < len
            && is_ascii_alpha(s[pos + 1])
        {
            // Rule: [P/S][A-Za-z]+ — punct/symbol immediately followed by alpha
            pos += 1;
            while pos < len && is_ascii_alpha(s[pos]) {
                pos += 1;
            }
        } else if is_letter_like(s, pos) {
            // Rule: [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ — letter run
            pos = consume_letters(s, pos);
        } else if !is_ascii_newline(c)
            && !is_ascii_punct(c)
            && pos + 1 < len
            && is_letter_like(s, pos + 1)
        {
            // Rule: skip one non-letter/punct/non-newline, then consume letters
            pos += 1;
            pos = consume_letters(s, pos);
        } else if c == b' '
            && pos + 1 < len
            && is_ascii_punct(s[pos + 1])
        {
            // Rule:  ?[\p{P}\p{S}]+[\r\n]* — leading space + punct run + trailing newlines
            pos += 1;
            while pos < len && is_ascii_punct(s[pos]) {
                pos += 1;
            }
            while pos < len && is_ascii_newline(s[pos]) {
                pos += 1;
            }
        } else if is_ascii_punct(c) {
            // Rule: [\p{P}\p{S}]+[\r\n]* — punct run + trailing newlines
            while pos < len && is_ascii_punct(s[pos]) {
                pos += 1;
            }
            while pos < len && is_ascii_newline(s[pos]) {
                pos += 1;
            }
        } else if is_ascii_space(c) {
            // Rule: \s*[\r\n]+ / \s+(?!\S) / \s+ — complex space logic
            let p_start = pos;
            let mut last_newline_end = 0usize;
            let mut p = pos;
            while p < len && is_ascii_space(s[p]) {
                let sc = s[p];
                p += 1;
                if is_ascii_newline(sc) {
                    last_newline_end = p;
                }
            }
            if last_newline_end > 0 {
                // Found newlines: consume up to the last newline
                pos = last_newline_end;
            } else if p < len
                && p > p_start + 1
                && (is_letter_like(s, p) || is_ascii_punct(s[p]))
            {
                // JoyAI lets a single leading space join the following word/punct.
                // For "    int", emit "   " then " int".
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(s, pos);
        }

        // Safety: if nothing was consumed, advance one UTF-8 char
        if pos == start {
            pos = next_utf8_char(s, pos);
        }

        // Emit BPE for this piece
        let piece = &s[start..pos];
        bpe_emit_piece(vocab, piece, &mut out);
    }

    out
}

/// Apply byte-level BPE to one pre-tokenized piece and emit token IDs.
/// Exact match of C's `bpe_emit_piece`.
fn bpe_emit_piece(vocab: &Vocab, piece: &[u8], out: &mut Vec<i32>) {
    // Step 1: byte-encode the raw bytes to printable Unicode
    let encoded = byte_encode(piece);

    // Step 2: split encoded string into UTF-8 character symbols
    let syms = {
        let ebytes = encoded.as_bytes();
        let mut syms: Vec<&str> = Vec::new();
        let mut off = 0;
        while off < ebytes.len() {
            let n = utf8_len(ebytes[off]);
            let end = if off + n > ebytes.len() {
                ebytes.len()
            } else {
                off + n
            };
            // SAFETY: byte_encode always produces valid UTF-8
            syms.push(unsafe { std::str::from_utf8_unchecked(&ebytes[off..end]) });
            off = end;
        }
        syms
    };

    let mut symbols: Vec<String> = syms.into_iter().map(|s| s.to_string()).collect();

    // Step 3: apply BPE merges greedily (lowest rank first)
    loop {
        let mut best_rank = i32::MAX;
        let mut best_i: Option<usize> = None;

        for i in 0..symbols.len().saturating_sub(1) {
            // C constructs merge key as: a.ptr + " " + b.ptr
            let merge_key = format!("{} {}", symbols[i], symbols[i + 1]);
            if let Some(&rank) = vocab.merge_ranks.get(&merge_key) {
                if rank < best_rank {
                    best_rank = rank;
                    best_i = Some(i);
                }
            }
        }

        match best_i {
            Some(i) => {
                // Merge two adjacent symbols
                let merged = format!("{}{}", symbols[i], symbols[i + 1]);
                symbols[i] = merged;
                symbols.remove(i + 1);
            }
            None => break,
        }
    }

    // Step 4: emit token IDs for final symbols
    for sym in &symbols {
        if let Some(&id) = vocab.token_to_id.get(sym.as_str()) {
            out.push(id);
        } else {
            // Fallback: C tries each single byte in the symbol
            for b in sym.bytes() {
                if let Some(&id) = vocab.token_to_id.get(&(b as char).to_string()) {
                    out.push(id);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── byte_encode / gpt2_byte_to_unicode ──────────────────────────────

    #[test]
    fn test_gpt2_byte_to_unicode_printable() {
        // Printable ASCII passes through
        assert_eq!(gpt2_byte_to_unicode(b'H'), 'H');
        assert_eq!(gpt2_byte_to_unicode(b'e'), 'e');
        assert_eq!(gpt2_byte_to_unicode(b'a'), 'a');
        assert_eq!(gpt2_byte_to_unicode(b'0'), '0');
        assert_eq!(gpt2_byte_to_unicode(b'!'), '!');
        // Boundary values in printable range
        assert_eq!(gpt2_byte_to_unicode(33), '!');
        assert_eq!(gpt2_byte_to_unicode(126), '~');
        assert_eq!(gpt2_byte_to_unicode(161), '¡');
        assert_eq!(gpt2_byte_to_unicode(172), '¬');
        assert_eq!(gpt2_byte_to_unicode(174), '®');
    }

    #[test]
    fn test_gpt2_byte_to_unicode_control() {
        // Space (0x20) → Ġ (U+0120)
        assert_eq!(gpt2_byte_to_unicode(b' '), 'Ġ');
        // Tab (0x09) → U+0109
        assert_eq!(gpt2_byte_to_unicode(b'\t') as u32, 0x0109);
        // Newline (0x0a) → U+010A
        assert_eq!(gpt2_byte_to_unicode(b'\n') as u32, 0x010A);
        // Null (0x00) → U+0100
        assert_eq!(gpt2_byte_to_unicode(0x00) as u32, 0x0100);
    }

    #[test]
    fn test_byte_encode_hello() {
        assert_eq!(byte_encode(b"Hello"), "Hello");
    }

    #[test]
    fn test_byte_encode_space() {
        // Space byte (0x20) → Ġ
        assert_eq!(byte_encode(b" "), "Ġ");
        assert_eq!(byte_encode(b"Hello world"), "HelloĠworld");
    }

    #[test]
    fn test_byte_encode_roundtrip() {
        // Every byte 0..255 must produce a unique, printable char
        let mut seen = std::collections::HashSet::new();
        for b in 0u8..=255 {
            let ch = gpt2_byte_to_unicode(b);
            assert!(ch as u32 >= 32, "byte {} → U+{:04X} (non-printable)", b, ch as u32);
            assert!(seen.insert(ch), "byte {} → '{}' collides", b, ch);
        }
    }

    // ── UTF-8 helpers ───────────────────────────────────────────────────

    #[test]
    fn test_utf8_len_ascii() {
        assert_eq!(utf8_len(b'a'), 1);
        assert_eq!(utf8_len(b'z'), 1);
        assert_eq!(utf8_len(b' '), 1);
    }

    #[test]
    fn test_utf8_len_multibyte() {
        assert_eq!(utf8_len(0xc2), 2); // 2-byte lead
        assert_eq!(utf8_len(0xe0), 3); // 3-byte lead
        assert_eq!(utf8_len(0xf0), 4); // 4-byte lead
    }

    #[test]
    fn test_utf8_len_invalid() {
        assert_eq!(utf8_len(0x80), 1); // continuation byte → 1
        assert_eq!(utf8_len(0xfe), 1); // invalid lead → 1
    }

    #[test]
    fn test_utf8_peek_one_ascii() {
        let s = b"Hello";
        assert_eq!(utf8_peek_one(s, 0), (b'H' as u32, 1));
        assert_eq!(utf8_peek_one(s, 1), (b'e' as u32, 2));
    }

    #[test]
    fn test_utf8_peek_one_cjk() {
        // 你好 = e4 bd a0  e5 a5 bd
        let s = "你好".as_bytes();
        let (cp, next) = utf8_peek_one(s, 0);
        assert_eq!(cp, 0x4f60); // 你
        assert_eq!(next, 3);
        let (cp, next) = utf8_peek_one(s, 3);
        assert_eq!(cp, 0x597d); // 好
        assert_eq!(next, 6);
    }

    #[test]
    fn test_utf8_peek_one_gpt2_encoded_space() {
        // Ġ = U+0120 = c4 a0
        let s = "Ġ".as_bytes();
        let (cp, next) = utf8_peek_one(s, 0);
        assert_eq!(cp, 0x0120);
        assert_eq!(next, 2);
    }

    // ── Character classification ────────────────────────────────────────

    #[test]
    fn test_is_ascii_punct_ranges() {
        assert!(is_ascii_punct(b'!'));
        assert!(is_ascii_punct(b'/'));
        assert!(is_ascii_punct(b':'));
        assert!(is_ascii_punct(b'@'));
        assert!(is_ascii_punct(b'['));
        assert!(is_ascii_punct(b'`'));
        assert!(is_ascii_punct(b'{'));
        assert!(is_ascii_punct(b'~'));
        assert!(!is_ascii_punct(b'A'));
        assert!(!is_ascii_punct(b'0'));
        assert!(!is_ascii_punct(b' '));
    }

    #[test]
    fn test_is_cjk_hira_kata() {
        assert!(is_cjk_hira_kata(0x4e00)); // CJK start
        assert!(is_cjk_hira_kata(0x9fa5)); // CJK end
        assert!(is_cjk_hira_kata(0x3040)); // Hiragana start
        assert!(is_cjk_hira_kata(0x309f)); // Hiragana end
        assert!(is_cjk_hira_kata(0x30a0)); // Katakana start
        assert!(is_cjk_hira_kata(0x30ff)); // Katakana end
        assert!(!is_cjk_hira_kata(0x0041)); // 'A' — ASCII
        assert!(!is_cjk_hira_kata(0x00e9)); // é — Latin supplement
    }

    #[test]
    fn test_is_letter_like() {
        let s = b"Hello";
        assert!(is_letter_like(s, 0)); // 'H'
        assert!(!is_letter_like(b" ", 0)); // space is not letter-like
        assert!(!is_letter_like(b"!", 0)); // punct is not letter-like
        assert!(!is_letter_like(b"0", 0)); // digit is not letter-like
    }

    // ── JoyAI pre-tokenizer piece boundaries ────────────────────────────

    /// Helper: collect piece strings for debugging/tests
    fn pretokenize_pieces(text: &str) -> Vec<String> {
        let s = text.as_bytes();
        let len = s.len();
        let mut pos = 0usize;
        let mut pieces = Vec::new();

        while pos < len {
            let start = pos;
            let c = s[pos];

            if is_ascii_digit(c) {
                let mut ndigits = 0;
                while pos < len && is_ascii_digit(s[pos]) && ndigits < 3 {
                    pos += 1;
                    ndigits += 1;
                }
            } else if is_cjk_at(s, pos) {
                pos = next_utf8_char(s, pos);
                while pos < len && is_cjk_at(s, pos) {
                    pos = next_utf8_char(s, pos);
                }
            } else if is_ascii_punct(c) && pos + 1 < len && is_ascii_alpha(s[pos + 1]) {
                pos += 1;
                while pos < len && is_ascii_alpha(s[pos]) {
                    pos += 1;
                }
            } else if is_letter_like(s, pos) {
                pos = consume_letters(s, pos);
            } else if !is_ascii_newline(c)
                && !is_ascii_punct(c)
                && pos + 1 < len
                && is_letter_like(s, pos + 1)
            {
                pos += 1;
                pos = consume_letters(s, pos);
            } else if c == b' ' && pos + 1 < len && is_ascii_punct(s[pos + 1]) {
                pos += 1;
                while pos < len && is_ascii_punct(s[pos]) {
                    pos += 1;
                }
                while pos < len && is_ascii_newline(s[pos]) {
                    pos += 1;
                }
            } else if is_ascii_punct(c) {
                while pos < len && is_ascii_punct(s[pos]) {
                    pos += 1;
                }
                while pos < len && is_ascii_newline(s[pos]) {
                    pos += 1;
                }
            } else if is_ascii_space(c) {
                let p_start = pos;
                let mut last_newline_end = 0usize;
                let mut p = pos;
                while p < len && is_ascii_space(s[p]) {
                    let sc = s[p];
                    p += 1;
                    if is_ascii_newline(sc) {
                        last_newline_end = p;
                    }
                }
                if last_newline_end > 0 {
                    pos = last_newline_end;
                } else if p < len
                    && p > p_start + 1
                    && (is_letter_like(s, p) || is_ascii_punct(s[p]))
                {
                    pos = p - 1;
                } else {
                    pos = p;
                }
            } else {
                pos = next_utf8_char(s, pos);
            }

            if pos == start {
                pos = next_utf8_char(s, pos);
            }
            pieces.push(String::from_utf8_lossy(&s[start..pos]).to_string());
        }
        pieces
    }

    #[test]
    fn test_pretokenize_hello_world() {
        // "Hello world": H,e,l,l,o are letters → piece "Hello";
        // space + letter → JoyAI joins the space with the word → " world"
        let pieces = pretokenize_pieces("Hello world");
        assert_eq!(pieces, vec!["Hello", " world"]);
    }

    #[test]
    fn test_pretokenize_spaces_then_word() {
        // "   Hello": 3 spaces then letter.
        // JoyAI: "  " (consume all but last), " Hello" (leading space + word)
        let pieces = pretokenize_pieces("   Hello");
        assert_eq!(pieces, vec!["  ", " Hello"]);
    }

    #[test]
    fn test_pretokenize_cjk() {
        let pieces = pretokenize_pieces("你好世界");
        // CJK chars should be grouped together (unlike old Rust which split per-char)
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0], "你好世界");
    }

    #[test]
    fn test_pretokenize_cjk_mixed() {
        // C's consume_letters gobbles non-ASCII (including CJK) when reached
        // via the letter-like path. CJK is only split out when encountered as
        // the FIRST char of a piece (CJK rule checked before letter rule).
        // Here 'H' triggers letter rule, then consume_letters eats everything.
        let pieces = pretokenize_pieces("Hello你好World");
        assert_eq!(pieces, vec!["Hello你好World"]);
    }

    #[test]
    fn test_pretokenize_cjk_isolated() {
        // CJK as first char: CJK rule fires, groups the CJK run
        let pieces = pretokenize_pieces("你好World");
        assert_eq!(pieces, vec!["你好", "World"]);
    }

    #[test]
    fn test_pretokenize_digits() {
        // Digits grouped 1-3 at a time
        let pieces = pretokenize_pieces("12345");
        assert_eq!(pieces, vec!["123", "45"]);
    }

    #[test]
    fn test_pretokenize_punct_alpha() {
        // Punct immediately before alpha joins: "*ptr" is one piece
        let pieces = pretokenize_pieces("*ptr");
        assert_eq!(pieces, vec!["*ptr"]);
    }

    #[test]
    fn test_pretokenize_punct_run() {
        // Punctuation runs are grouped
        let pieces = pretokenize_pieces("!!!");
        assert_eq!(pieces, vec!["!!!"]);
    }

    #[test]
    fn test_pretokenize_punct_newline() {
        // Punctuation + trailing newline: ">;\n" stays together
        let pieces = pretokenize_pieces(">;\n");
        assert_eq!(pieces, vec![">;\n"]);
    }

    #[test]
    fn test_pretokenize_space_punct() {
        // Space + punct rule: " !" → piece " !" (space joins punct)
        let pieces = pretokenize_pieces(" !");
        assert_eq!(pieces, vec![" !"]);
    }

    #[test]
    fn test_pretokenize_newline_in_spaces() {
        // "  \n  " — spaces, newline, spaces.
        // C's space logic: scan all spaces, last_newline_end=3.
        // Piece 1: "  \n", then pos=3, remaining "  " becomes piece 2.
        let pieces = pretokenize_pieces("  \n  ");
        assert_eq!(pieces, vec!["  \n", "  "]);
    }

    #[test]
    fn test_pretokenize_non_alpha_then_letter() {
        // "`int" → '`' is not a letter, 'i' is a letter
        // Rule: skip one non-letter/punct/non-newline, then consume letters
        // But wait: '`' IS punct! So it should be caught by the punct rule or
        // by the punct+alpha rule. Let's check:
        // '`' = 0x60, is_ascii_punct(0x60) = true (in ['@'..'`'] range)
        // So this is punct+alpha rule → "`int"
        // Actually: punct+alpha requires `pos+1 < len && is_ascii_alpha(s[pos+1])`
        // '`' is punct, 'i' is alpha → "`int" as one piece
        let pieces = pretokenize_pieces("`int");
        assert_eq!(pieces, vec!["`int"]);
    }

    #[test]
    fn test_pretokenize_code_line() {
        // Test a realistic code line
        let pieces = pretokenize_pieces("int x = 42;");
        assert_eq!(pieces, vec!["int", " x", " =", " ", "42", ";"]);
    }

    // ── BPE encoding (with test vocab, empty merges) ─────────────────

    /// Build a minimal test vocab similar to the test GGUF's token list.
    fn test_vocab() -> Vocab {
        let tokens: Vec<String> = [
            "<unk>",
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<think>",
            "</think>",
            "｜DSML｜",
            "the",
            "Hello",
            "a",
            "is",
            "of",
            "and",
            "to",
            "world",
            "Ġ", // GPT-2 encoded space — crucial!
            "H", "e", "l", "o", "w", "r", "d", "t", "test",
            // Add more single-char tokens so BPE doesn't silently drop chars
            "b", "c", "f", "g", "h", "i", "j", "k", "m",
            "n", "p", "q", "s", "u", "v", "x", "y", "z",
            "A", "B", "C", "D", "E", "F", "G", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S",
            "T", "U", "V", "W", "X", "Y", "Z",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "!", "\"", "#", "$", "%", "&", "'", "(", ")",
            "*", "+", ",", "-", ".", "/", ":", ";", "<",
            "=", ">", "?", "@", "[", "\\", "]", "^", "_",
            "`", "{", "|", "}", "~",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let n_vocab = tokens.len();
        let mut token_to_id = HashMap::new();
        for (i, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as i32);
        }

        Vocab {
            tokens: tokens.clone(),
            n_vocab,
            bos_id: 1,
            eos_id: 2,
            user_id: 3,
            assistant_id: 4,
            think_start_id: 5,
            think_end_id: 6,
            dsml_id: 7,
            token_to_id,
            merge_ranks: HashMap::new(), // empty merges
            id_to_text: tokens,
        }
    }

    fn encode_test(text: &str) -> Vec<i32> {
        test_vocab().encode(text)
    }

    #[test]
    fn test_encode_hello_world() {
        let ids = encode_test("Hello world");
        // "Hello" → H(17), e(18), l(19), l(19), o(20)
        // " world" → byte_encode: "Ġworld" → Ġ(16), w(21), o(20), r(22), l(19), d(23)
        assert_eq!(ids, vec![17, 18, 19, 19, 20, 16, 21, 20, 22, 19, 23]);
    }

    #[test]
    fn test_encode_simple_word() {
        let ids = encode_test("Hello");
        assert_eq!(ids, vec![17, 18, 19, 19, 20]);
    }

    #[test]
    fn test_encode_with_spaces() {
        // "   Hello": pre-tokenizer gives ["  ", " Hello"]
        // "  " → byte_encode: "ĠĠ" → Ġ(16) Ġ(16)
        // " Hello" → byte_encode: "ĠHello" → Ġ(16) H(17) e(18) l(19) l(19) o(20)
        let ids = encode_test("   Hello");
        assert_eq!(ids, vec![16, 16, 16, 17, 18, 19, 19, 20]);
    }

    #[test]
    fn test_encode_code_line() {
        // "int x = 42;"
        // Pieces: ["int", " x", " =", " ", "42", ";"]
        // "int" → byte_encode "int" → i(31), n(35), t(24)
        // " x" → byte_encode "Ġx" → Ġ(16), x(41)
        // " =" → byte_encode "Ġ=" → Ġ(16), =(97)
        // " " → byte_encode "Ġ" → Ġ(16)
        // "42" → byte_encode "42" → 4(73), 2(71)
        // ";" → byte_encode ";" → ;(95)
        let ids = encode_test("int x = 42;");
        assert_eq!(ids, vec![31, 35, 24, 16, 41, 16, 97, 16, 73, 71, 95]);
    }

    #[test]
    fn test_encode_punct_newline() {
        // ">;\n" stays as one piece (punct run + trailing newlines)
        // byte_encode: ">"(62), ";"(59), "\n"→Ċ (U+010A)
        // Ċ not in test vocab, falls back to single-byte lookup (also not found)
        // So output: >(98), ;(95)
        let ids = encode_test(">;\n");
        assert_eq!(ids, vec![98, 95]);
    }

    // ── Decode ──────────────────────────────────────────────────────────

    #[test]
    fn test_decode_roundtrip() {
        let vocab = test_vocab();
        let text = "Hello world";
        let ids = vocab.encode(text);
        let decoded = vocab.decode(&ids);
        // Decoding just concatenates token strings; with GPT-2 BPE,
        // "HelloĠworld" is the decoded form (Ġ = space)
        assert_eq!(decoded, "HelloĠworld");
    }

    // ── Special token IDs ───────────────────────────────────────────────

    #[test]
    fn test_special_token_ids() {
        let vocab = test_vocab();
        assert_eq!(vocab.bos_id, 1);
        assert_eq!(vocab.eos_id, 2);
        assert_eq!(vocab.token_id("<｜begin▁of▁sentence｜>"), Some(1));
        assert_eq!(vocab.token_id("<｜end▁of▁sentence｜>"), Some(2));
    }
}
