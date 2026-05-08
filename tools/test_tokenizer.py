"""Generate expected BPE token IDs for the test GGUF model.
The test model has 256 tokens with empty merges, so BPE is just byte-level.
"""

# Token list from gen_test_gguf.rs (indices 0-26, rest are <tok_N>)
TOKENS = [
    "<unk>", "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
    "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>", "｜DSML｜",
    "the", "Hello", "a", "is", "of", "and", "to", "world", " ",
    "H", "e", "l", "o", "w", "r", "d", "t", "test",
]
# Remaining 27..255 are "<tok_{i}>"

# Build lookup
token_to_id = {}
for i, tok in enumerate(TOKENS):
    token_to_id[tok] = i
# For tokens 27-255, they're "<tok_N>" — we won't match these in normal text

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
    result = b''
    for b in raw_bytes:
        result += utf8_put(gpt2_byte_to_codepoint(b))
    return result

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

def ascii_alpha(c):
    return (65 <= c <= 90) or (97 <= c <= 122)

def ascii_digit(c):
    return 48 <= c <= 57

def ascii_space(c):
    return c in (32, 9, 10, 13, 11, 12)

def ascii_newline(c):
    return c in (10, 13)

def joyai_ascii_punct_symbol(c):
    return ((33 <= c <= 47) or (58 <= c <= 64) or
            (91 <= c <= 96) or (123 <= c <= 126))

def joyai_letter_like_at(s, pos):
    if pos >= len(s): return False
    c = s[pos]
    if c < 128: return ascii_alpha(c)
    return True

def joyai_pretokenize(text_bytes):
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


def bpe_tokenize(text):
    """Tokenize using test model vocab (empty merges)."""
    text_bytes = text.encode('utf-8')
    pieces = joyai_pretokenize(text_bytes)
    out = []

    for start, end in pieces:
        raw_piece = text_bytes[start:end]
        encoded = byte_encode(raw_piece)

        syms = []
        off = 0
        while off < len(encoded):
            n = utf8_len_from_first_byte(encoded[off])
            if off + n > len(encoded): n = 1
            syms.append(encoded[off:off + n])
            off += n

        # No merges (empty merge table), skip BPE loop

        for sym in syms:
            tid = token_to_id.get(sym.decode('utf-8'), None)
            if tid is not None:
                out.append(tid)

    return out


def pretokenize_debug(text):
    """Print pre-tokenization pieces for debugging."""
    text_bytes = text.encode('utf-8')
    pieces = joyai_pretokenize(text_bytes)
    print(f"Text: {text!r}")
    for start, end in pieces:
        piece = text_bytes[start:end]
        print(f"  piece [{start}:{end}]: {piece!r} -> BPE: {bpe_tokenize(text[start:end])}")
    print(f"  All tokens: {bpe_tokenize(text)}")


if __name__ == '__main__':
    import sys
    tests = [
        "Hello world",
        "Hello",
        "world",
        "test",
        "the quick brown fox",
        "123",
        "  spaces  ",
        "Hello, world!",
        "int x = 42;",
        ">;\n",
    ]
    if len(sys.argv) > 1:
        tests = sys.argv[1:]

    for t in tests:
        pretokenize_debug(t)
        print()
