"""Oracle implementation of the C JoyAI BPE tokenizer from ds4.c.
Used to generate expected token IDs for integration tests.

Usage: python3 tools/tokenizer_oracle.py /tmp/test_ds4.gguf "Hello world"
"""

import struct
import sys

# ─── GGUF reader ───────────────────────────────────────────────────────────

def read_gguf_kv(f):
    """Read GGUF metadata key-value pairs. Returns dict of key->value."""
    kv = {}
    magic = f.read(4)
    assert magic == b'GGUF', f"Bad magic: {magic}"
    version = struct.unpack('<I', f.read(4))[0]
    ntensors = struct.unpack('<Q', f.read(8))[0]
    nkv = struct.unpack('<Q', f.read(8))[0]

    for _ in range(nkv):
        klen = struct.unpack('<Q', f.read(8))[0]
        key = f.read(klen).decode()
        vtype = struct.unpack('<I', f.read(4))[0]

        if vtype == 8:  # string
            slen = struct.unpack('<Q', f.read(8))[0]
            kv[key] = ('str', f.read(slen).decode(errors='replace'))
        elif vtype == 4:  # u32
            kv[key] = ('u32', struct.unpack('<I', f.read(4))[0])
        elif vtype == 5:  # i32
            kv[key] = ('i32', struct.unpack('<i', f.read(4))[0])
        elif vtype == 9:  # array
            atype = struct.unpack('<I', f.read(4))[0]
            alen = struct.unpack('<Q', f.read(8))[0]
            kv[key] = ('array', atype, alen)
        elif vtype == 0:  # u8
            kv[key] = ('u8', f.read(1)[0])
        elif vtype == 10:
            kv[key] = ('u64', struct.unpack('<Q', f.read(8))[0])
        else:
            # Skip unknown
            pass
    return kv

def read_array_data(f, kv, key):
    """Read array data from GGUF."""
    info = kv[key]
    atype, alen = info[1], info[2]
    if atype != 8:  # only care about string arrays
        return []
    result = []
    for _ in range(alen):
        slen = struct.unpack('<Q', f.read(8))[0]
        result.append(f.read(slen).decode(errors='replace'))
    return result

# ─── GPT-2 byte-to-unicode ─────────────────────────────────────────────────

def gpt2_byte_to_codepoint(b):
    if (33 <= b <= 126) or (161 <= b <= 172) or b >= 174:
        return b
    n = 0
    for x in range(256):
        if (33 <= x <= 126) or (161 <= x <= 172) or x >= 174:
            continue
        if x == b:
            return 256 + n
        n += 1
    return b

def utf8_put(cp):
    """Encode a Unicode codepoint as UTF-8 bytes."""
    if cp <= 0x7f:
        return bytes([cp])
    elif cp <= 0x7ff:
        return bytes([0xc0 | (cp >> 6), 0x80 | (cp & 0x3f)])
    elif cp <= 0xffff:
        return bytes([0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f)])
    else:
        return bytes([0xf0 | (cp >> 18), 0x80 | ((cp >> 12) & 0x3f),
                      0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f)])

def byte_encode(raw_bytes):
    """GPT-2 byte-level encoding: map each byte to a printable Unicode codepoint."""
    result = b''
    for b in raw_bytes:
        result += utf8_put(gpt2_byte_to_codepoint(b))
    return result

# ─── UTF-8 helpers ─────────────────────────────────────────────────────────

def utf8_len_from_first_byte(c):
    if c < 0x80: return 1
    if (c & 0xe0) == 0xc0: return 2
    if (c & 0xf0) == 0xe0: return 3
    if (c & 0xf8) == 0xf0: return 4
    return 1

def next_utf8_char(s, pos):
    if pos >= len(s): return pos
    n = utf8_len_from_first_byte(s[pos])
    if pos + n > len(s): n = 1
    return pos + n

def utf8_peek_one(s, pos):
    """Return (codepoint, next_pos)."""
    if pos >= len(s): return (0, pos)
    c0 = s[pos]
    n = utf8_len_from_first_byte(c0)
    if pos + n > len(s): n = 1
    next_pos = pos + n
    if n == 1:
        return (c0, next_pos)
    elif n == 2:
        cp = ((c0 & 0x1f) << 6) | (s[pos + 1] & 0x3f)
        return (cp, next_pos)
    elif n == 3:
        cp = ((c0 & 0x0f) << 12) | ((s[pos + 1] & 0x3f) << 6) | (s[pos + 2] & 0x3f)
        return (cp, next_pos)
    else:
        cp = ((c0 & 0x07) << 18) | ((s[pos + 1] & 0x3f) << 12) | \
             ((s[pos + 2] & 0x3f) << 6) | (s[pos + 3] & 0x3f)
        return (cp, next_pos)

# ─── Character classification ──────────────────────────────────────────────

def ascii_alpha(c):
    return (65 <= c <= 90) or (97 <= c <= 122)

def ascii_digit(c):
    return 48 <= c <= 57

def ascii_space(c):
    return c in (32, 9, 10, 13, 11, 12)  # ' ', \t, \n, \r, \v, \f

def ascii_newline(c):
    return c in (10, 13)  # \n, \r

def joyai_ascii_punct_symbol(c):
    return ((33 <= c <= 47) or (58 <= c <= 64) or
            (91 <= c <= 96) or (123 <= c <= 126))

def utf8_is_cjk_hira_kata(cp):
    return ((0x4e00 <= cp <= 0x9fa5) or
            (0x3040 <= cp <= 0x309f) or
            (0x30a0 <= cp <= 0x30ff))

def joyai_letter_like_at(s, pos):
    if pos >= len(s): return False
    c = s[pos]
    if c < 128: return ascii_alpha(c)
    return True  # non-ASCII always letter-like (CJK handled separately)

def joyai_cjk_at(s, pos):
    if pos >= len(s) or s[pos] < 128:
        return False
    cp, _ = utf8_peek_one(s, pos)
    return utf8_is_cjk_hira_kata(cp)

# ─── JoyAI pre-tokenizer (exact match of C's bpe_tokenize_text) ────────────

def joyai_pretokenize(text_bytes):
    """Return list of (start, end) byte offsets for each pre-tokenized piece."""
    pieces = []
    pos = 0
    n = len(text_bytes)

    while pos < n:
        start = pos
        c = text_bytes[pos]

        if ascii_digit(c):
            ndigits = 0
            while pos < n and ascii_digit(text_bytes[pos]) and ndigits < 3:
                pos += 1
                ndigits += 1
        elif joyai_cjk_at(text_bytes, pos):
            pos = next_utf8_char(text_bytes, pos)
            while pos < n and joyai_cjk_at(text_bytes, pos):
                pos = next_utf8_char(text_bytes, pos)
        elif (joyai_ascii_punct_symbol(c) and pos + 1 < n and
              ascii_alpha(text_bytes[pos + 1])):
            pos += 1
            while pos < n and ascii_alpha(text_bytes[pos]):
                pos += 1
        elif joyai_letter_like_at(text_bytes, pos):
            while pos < n and joyai_letter_like_at(text_bytes, pos):
                pos = next_utf8_char(text_bytes, pos)
        elif (not ascii_newline(c) and not joyai_ascii_punct_symbol(c) and
              pos + 1 < n and joyai_letter_like_at(text_bytes, pos + 1)):
            pos += 1
            while pos < n and joyai_letter_like_at(text_bytes, pos):
                pos = next_utf8_char(text_bytes, pos)
        elif (c == 32 and pos + 1 < n and
              joyai_ascii_punct_symbol(text_bytes[pos + 1])):
            pos += 1
            while pos < n and joyai_ascii_punct_symbol(text_bytes[pos]):
                pos += 1
            while pos < n and ascii_newline(text_bytes[pos]):
                pos += 1
        elif joyai_ascii_punct_symbol(c):
            while pos < n and joyai_ascii_punct_symbol(text_bytes[pos]):
                pos += 1
            while pos < n and ascii_newline(text_bytes[pos]):
                pos += 1
        elif ascii_space(c):
            p = pos
            last_newline_end = 0
            while p < n and ascii_space(text_bytes[p]):
                sc = text_bytes[p]
                p += 1
                if ascii_newline(sc):
                    last_newline_end = p
            if last_newline_end:
                pos = last_newline_end
            elif (p < n and p > pos + 1 and
                  (joyai_letter_like_at(text_bytes, p) or
                   joyai_ascii_punct_symbol(text_bytes[p]))):
                pos = p - 1
            else:
                pos = p
        else:
            pos = next_utf8_char(text_bytes, pos)

        if pos == start:
            pos = next_utf8_char(text_bytes, pos)
        pieces.append((start, pos))

    return pieces


# ─── BPE ───────────────────────────────────────────────────────────────────

def bpe_tokenize(vocab, token_to_id, merge_rank, text):
    """Tokenize text using the pre-loaded vocab. Returns list of token IDs."""
    text_bytes = text.encode('utf-8')
    pieces = joyai_pretokenize(text_bytes)
    out = []

    for start, end in pieces:
        raw_piece = text_bytes[start:end]
        encoded = byte_encode(raw_piece)

        # Split encoded into UTF-8 characters
        syms = []
        off = 0
        while off < len(encoded):
            n = utf8_len_from_first_byte(encoded[off])
            if off + n > len(encoded): n = 1
            syms.append(encoded[off:off + n])
            off += n

        # Apply BPE merges (greedy, lowest rank first)
        while True:
            best_i = -1
            best_rank = 2**31 - 1
            for i in range(len(syms) - 1):
                merge_key = syms[i] + b' ' + syms[i + 1]
                rank = merge_rank.get(merge_key, -1)
                if rank >= 0 and rank < best_rank:
                    best_rank = rank
                    best_i = i
            if best_i < 0:
                break
            merged = syms[best_i] + syms[best_i + 1]
            syms[best_i] = merged
            del syms[best_i + 1]

        # Emit token IDs
        for sym in syms:
            tid = token_to_id.get(sym, None)
            if tid is not None:
                out.append(tid)
            else:
                # Fallback: try each single byte
                for j in range(len(sym)):
                    tid2 = token_to_id.get(sym[j:j+1], None)
                    if tid2 is not None:
                        out.append(tid2)
    return out


def main():
    if len(sys.argv) < 3:
        print("Usage: tokenizer_oracle.py <gguf_path> <text> [more_text...]")
        sys.exit(1)

    gguf_path = sys.argv[1]

    with open(gguf_path, 'rb') as f:
        kv = read_gguf_kv(f)

        # Read tokens array
        token_count = kv.get('ds4.vocab_size', ('u32', 0))[1]
        tokens = read_array_data(f, kv, 'tokenizer.ggml.tokens')
        merges = read_array_data(f, kv, 'tokenizer.ggml.merges')

    assert len(tokens) == token_count, f"Token count mismatch: {len(tokens)} vs {token_count}"

    # Build lookup tables
    token_to_id = {}
    for i, tok in enumerate(tokens):
        token_to_id[tok.encode('utf-8')] = i

    merge_rank = {}
    for rank, merge in enumerate(merges):
        merge_rank[merge.encode('utf-8')] = rank

    # Tokenize each text
    for text in sys.argv[2:]:
        ids = bpe_tokenize(tokens, token_to_id, merge_rank, text)
        print(f"{text!r} -> {ids}")


if __name__ == '__main__':
    main()
