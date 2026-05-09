#!/usr/bin/env python3
"""Inspect GPT-2 BPE tokenizer vocabulary in a GGUF file.

Usage:
    python3 tools/inspect_gguf_vocab.py <path/to/model.gguf> [command]

Commands:
    (none)    — Print vocabulary summary and first/last tokens
    find <s>  — Find tokens containing substring <s>
    decode <ids> — Decode comma-separated token IDs (e.g., "16,17,18")
    garbled   — Search for tokens matching the known garbled patterns
    check     — Verify byte-to-unicode roundtrip and decode correctness
"""

import struct
import sys
import os


# ── GPT-2 byte ↔ Unicode mapping (exact match of ds4.c) ──────────────

def gpt2_byte_to_unicode(b: int) -> int:
    """Map raw byte 0..255 to printable Unicode codepoint."""
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


def gpt2_unicode_to_byte(cp: int) -> int | None:
    """Reverse of gpt2_byte_to_unicode. Returns None if not in range."""
    if (33 <= cp <= 126) or (161 <= cp <= 172) or (174 <= cp <= 255):
        return cp
    if cp >= 256:
        n = 0
        for b in range(256):
            if (33 <= b <= 126) or (161 <= b <= 172) or b >= 174:
                continue
            if cp == 256 + n:
                return b
            n += 1
    return None


def token_to_bytes(token_text: str) -> bytes:
    """Decode a single GPT-2 BPE token string to raw bytes."""
    # Literal special tokens (containing U+FF5C) pass through as UTF-8
    if '\uff5c' in token_text:
        return token_text.encode('utf-8')
    result = bytearray()
    for ch in token_text:
        b = gpt2_unicode_to_byte(ord(ch))
        if b is not None:
            result.append(b)
    return bytes(result)


# ── GGUF reader ─────────────────────────────────────────────────────

def read_gguf(path):
    """Read GGUF metadata. Returns dict of kv pairs."""
    with open(path, 'rb') as f:
        data = f.read()

    pos = 0
    magic = struct.unpack_from('<I', data, pos)[0]; pos += 4
    if magic != 0x46554747:
        raise ValueError(f"Not a GGUF file (magic: 0x{magic:08X})")

    version = struct.unpack_from('<I', data, pos)[0]; pos += 4
    n_tensors = struct.unpack_from('<Q', data, pos)[0]; pos += 8
    n_kv = struct.unpack_from('<Q', data, pos)[0]; pos += 8

    print(f"GGUF v{version}: {n_tensors} tensors, {n_kv} metadata entries")

    def read_u64():
        nonlocal pos
        v = struct.unpack_from('<Q', data, pos)[0]
        pos += 8
        return v

    def read_u32():
        nonlocal pos
        v = struct.unpack_from('<I', data, pos)[0]
        pos += 4
        return v

    def read_string():
        nonlocal pos
        length = read_u64()
        s = data[pos:pos+length].decode('utf-8', errors='replace')
        pos += length
        return s

    def read_value(val_type, depth=0):
        nonlocal pos
        if depth > 8:
            raise ValueError("Array nesting too deep")
        if val_type == 0:  # uint8
            v = data[pos]; pos += 1; return v
        elif val_type == 1:  # int8
            v = data[pos]; pos += 1; return v - 256 if v >= 128 else v
        elif val_type in (2, 3):  # uint16, int16
            v = struct.unpack_from('<H', data, pos)[0]; pos += 2; return v
        elif val_type in (4, 5):  # uint32, int32
            v = struct.unpack_from('<I', data, pos)[0]; pos += 4; return v
        elif val_type == 6:  # float32
            v = struct.unpack_from('<f', data, pos)[0]; pos += 4; return v
        elif val_type == 7:  # bool
            v = data[pos] != 0; pos += 1; return v
        elif val_type == 8:  # string
            return read_string()
        elif val_type == 9:  # array
            elem_type = read_u32()
            arr_len = read_u64()
            if elem_type == 8:  # string array (tokens/merges)
                result = []
                for _ in range(arr_len):
                    result.append(read_string())
                return result
            else:
                # Skip non-string arrays
                for _ in range(arr_len):
                    read_value(elem_type, depth+1)
                return f"<array[{arr_len}] of type {elem_type}>"
        elif val_type in (10, 11):  # uint64, int64
            v = struct.unpack_from('<Q', data, pos)[0]; pos += 8; return v
        elif val_type == 12:  # float64
            v = struct.unpack_from('<d', data, pos)[0]; pos += 8; return v
        else:
            raise ValueError(f"Unknown value type {val_type} at pos {pos}")

    kv = {}
    for _ in range(n_kv):
        key = read_string()
        val_type = read_u32()
        # Only load tokenizer keys and common config keys
        if (key.startswith('tokenizer.') or
            key.startswith('ds4.') or
            key.startswith('deepseek4.') or
            key.startswith('general.') or
            key in ('llama.vocab_size', 'llama.block_count', 'llama.embedding_length')):
            kv[key] = read_value(val_type)
        else:
            # Skip
            _skip_value(data, pos, val_type)
            pos = _skip_value.__wrapped_pos__

    return kv


# ── Main commands ────────────────────────────────────────────────────

def cmd_summary(kv):
    """Print vocabulary summary."""
    tokens = kv.get('tokenizer.ggml.tokens', [])
    merges = kv.get('tokenizer.ggml.merges', [])

    n_vocab = (kv.get('ds4.vocab_size') or
               kv.get('deepseek4.vocab_size') or
               kv.get('llama.vocab_size') or
               len(tokens))

    print(f"\nVocabulary: {len(tokens)} tokens (declared: {n_vocab})")
    print(f"Merges: {len(merges)} entries")

    if len(tokens) != n_vocab:
        print(f"⚠️  Token count mismatch! Declared {n_vocab}, actual {len(tokens)}")

    # Special token IDs
    for name in ['bos_token_id', 'eos_token_id']:
        key = f'tokenizer.ggml.{name}'
        if key in kv:
            val = kv[key]
            if isinstance(val, int) and 0 <= val < len(tokens):
                print(f"  {name}: {val} -> {tokens[val]!r}")
            else:
                print(f"  {name}: {val}")

    # First 20 tokens
    print(f"\nFirst 20 tokens:")
    for i, tok in enumerate(tokens[:20]):
        decoded = token_to_bytes(tok)
        print(f"  [{i:6d}] {tok!r:20s} → {decoded!r}")

    # Last 5 tokens
    if len(tokens) > 20:
        print(f"\nLast 5 tokens:")
        for i in range(max(20, len(tokens)-5), len(tokens)):
            tok = tokens[i]
            decoded = token_to_bytes(tok)
            print(f"  [{i:6d}] {tok!r:30s} → {decoded!r}")

    # Statistics
    byte_counts = {}
    for tok in tokens:
        decoded = token_to_bytes(tok)
        byte_counts[len(decoded)] = byte_counts.get(len(decoded), 0) + 1

    print(f"\nToken byte-length distribution:")
    for blen in sorted(byte_counts):
        if byte_counts[blen] > 5:
            print(f"  {blen:2d} bytes: {byte_counts[blen]:6d} tokens")
    # Small counts
    small = [(blen, cnt) for blen, cnt in byte_counts.items() if cnt <= 5]
    if small:
        print(f"  (rare lengths: {', '.join(f'{blen}:{cnt}' for blen, cnt in small)})")


def cmd_find(tokens, substring):
    """Find tokens containing a substring."""
    matches = []
    for i, tok in enumerate(tokens):
        if substring in tok:
            decoded = token_to_bytes(tok)
            matches.append((i, tok, decoded))

    print(f"\nTokens containing {substring!r}: {len(matches)} found")
    for i, tok, decoded in matches[:50]:
        print(f"  [{i:6d}] {tok!r:40s} → {decoded!r}")
    if len(matches) > 50:
        print(f"  ... and {len(matches)-50} more")


def cmd_decode(tokens, ids_str):
    """Decode a comma-separated list of token IDs."""
    ids = [int(x.strip()) for x in ids_str.split(',')]
    raw_tokens = []
    raw_bytes = bytearray()
    for tid in ids:
        if 0 <= tid < len(tokens):
            tok = tokens[tid]
            raw_tokens.append(tok)
            raw_bytes.extend(token_to_bytes(tok))
        else:
            raw_tokens.append(f"<OOB:{tid}>")

    print(f"\nDecoding {len(ids)} token IDs:")
    print(f"  Raw token text: {''.join(raw_tokens)!r}")
    print(f"  Decoded bytes:  {bytes(raw_bytes)!r}")
    print(f"  As UTF-8:       {bytes(raw_bytes).decode('utf-8', errors='replace')!r}")


def cmd_garbled(tokens):
    """Search for tokens matching known garbled patterns."""
    # Pattern 1: æĢ³ = U+00E6 U+0122 U+00B3
    garbled_str = 'æĢ³'
    # Pattern 2: tokens that decode to the same byte sequence
    garbled_bytes = bytes([0xE6, 0x80, 0xB3])  # UTF-8 for 怳

    print("\n=== Garbled Pattern Analysis ===")

    # Find token(s) that produce the garbled byte sequence
    for i, tok in enumerate(tokens):
        decoded = token_to_bytes(tok)
        if decoded and len(decoded) <= 4:
            # Check if this token's bytes contain the garbled pattern
            if garbled_bytes in decoded:
                print(f"  Token [{i}] {tok!r} decodes to {decoded!r} (contains garbled bytes)")
            # Also check if consecutive tokens could produce it
            if decoded == garbled_bytes:
                print(f"  EXACT MATCH: Token [{i}] {tok!r} → {decoded!r} = '{decoded.decode('utf-8', errors='replace')}'")

    # Look for the exact token string "æĢ³"
    print(f"\nSearching for exact string 'æĢ³' in token texts...")
    found = []
    for i, tok in enumerate(tokens):
        if garbled_str in tok:
            found.append((i, tok))
    if found:
        for i, tok in found[:20]:
            print(f"  Token [{i}] = {tok!r} → {token_to_bytes(tok)!r}")
    else:
        print("  Not found as substring of any token text.")
        print("  This means 'æĢ³' is composed from MULTIPLE tokens concatenated.")

    # Check if there are tokens with just 'æ', 'Ģ', or '³'
    for ch in ['æ', 'Ģ', '³']:
        matches = []
        for i, tok in enumerate(tokens):
            if ch in tok:
                decoded = token_to_bytes(tok)
                matches.append((i, tok, decoded))
        if matches:
            print(f"\n  Tokens containing '{ch}' (U+{ord(ch):04X}):")
            for i, tok, decoded in matches[:10]:
                print(f"    [{i:6d}] {tok!r:30s} → {decoded!r}")


def cmd_check(tokens):
    """Verify byte-to-unicode roundtrip and basic decode correctness."""
    print("\n=== Roundtrip Verification ===")
    errors = 0
    for b in range(256):
        cp = gpt2_byte_to_unicode(b)
        back = gpt2_unicode_to_byte(cp)
        if back != b:
            print(f"  FAIL: byte {b} → U+{cp:04X} → byte {back}")
            errors += 1
    if errors == 0:
        print("  ✅ All 256 bytes roundtrip correctly")

    # Check all tokens can be decoded
    print(f"\n=== Token Decode Verification ({len(tokens)} tokens) ===")
    decode_errors = 0
    for i, tok in enumerate(tokens):
        try:
            decoded = token_to_bytes(tok)
            # Verify decoded bytes produce valid-ish output
            text = decoded.decode('utf-8', errors='replace')
        except Exception as e:
            print(f"  FAIL token [{i}]: {tok!r} → {e}")
            decode_errors += 1
    if decode_errors == 0:
        print(f"  ✅ All {len(tokens)} tokens decode without errors")

    # Verify a known encode-decode roundtrip
    test_text = "Hello world"
    print(f"\n=== Encode/Decode Test ===")
    # Manual encode: byte-encode, tokenize
    # For now, just check that known tokens decode correctly
    print("  (Full encode/decode roundtrip requires BPE merge ranks)")
    merges = tokens  # dummy — we don't have kv in scope here
    print(f"  Available merges: N/A (see summary)")

    # Test specific token IDs if they make sense
    if len(tokens) > 10:
        # Decode first 10 non-special tokens
        print(f"\n  First 10 regular tokens decoded:")
        for i in range(min(10, len(tokens))):
            tok = tokens[i]
            decoded = token_to_bytes(tok)
            text = decoded.decode('utf-8', errors='replace')
            print(f"    [{i}] {tok!r:25s} → {decoded!r:20s} → '{text}'")


# ── _skip_value helper (hack: uses closure-over-nonlocal) ───────────

def _skip_value(data, pos, val_type, depth=0):
    """Skip a GGUF value, returning new position."""
    if depth > 8:
        raise ValueError("Array nesting too deep")

    if val_type in (0, 1, 7):  # uint8, int8, bool
        pos += 1
    elif val_type in (2, 3):  # uint16, int16
        pos += 2
    elif val_type in (4, 5, 6):  # uint32, int32, float32
        pos += 4
    elif val_type in (10, 11, 12):  # uint64, int64, float64
        pos += 8
    elif val_type == 8:  # string
        length = struct.unpack_from('<Q', data, pos)[0]
        pos += 8 + length
    elif val_type == 9:  # array
        elem_type = struct.unpack_from('<I', data, pos)[0]; pos += 4
        arr_len = struct.unpack_from('<Q', data, pos)[0]; pos += 8
        for _ in range(arr_len):
            pos = _skip_value(data, pos, elem_type, depth+1)
    else:
        raise ValueError(f"Unknown value type {val_type}")
    return pos


# ── Main ─────────────────────────────────────────────────────────────

def _patch_read_gguf():
    """Monkey-patch _skip_value to return the new position"""
    global _skip_value
    # We actually need a different approach: use a class
    pass


class GGUFParser:
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = f.read()
        self.pos = 0
        self._parse()

    def _read_u64(self):
        v = struct.unpack_from('<Q', self.data, self.pos)[0]
        self.pos += 8
        return v

    def _read_u32(self):
        v = struct.unpack_from('<I', self.data, self.pos)[0]
        self.pos += 4
        return v

    def _read_string(self):
        length = self._read_u64()
        s = self.data[self.pos:self.pos+length].decode('utf-8', errors='replace')
        self.pos += length
        return s

    def _skip_value(self, val_type, depth=0):
        if depth > 8:
            return
        if val_type in (0, 1, 7):
            self.pos += 1
        elif val_type in (2, 3):
            self.pos += 2
        elif val_type in (4, 5, 6):
            self.pos += 4
        elif val_type in (10, 11, 12):
            self.pos += 8
        elif val_type == 8:
            length = self._read_u64()
            self.pos += length
        elif val_type == 9:
            elem_type = self._read_u32()
            arr_len = self._read_u64()
            for _ in range(arr_len):
                self._skip_value(elem_type, depth+1)

    def _read_value(self, val_type, depth=0):
        if depth > 8:
            raise ValueError("Nesting too deep")
        if val_type == 0:
            v = self.data[self.pos]; self.pos += 1; return v
        elif val_type == 1:
            v = self.data[self.pos]; self.pos += 1; return v - 256 if v >= 128 else v
        elif val_type in (2, 3):
            v = struct.unpack_from('<H', self.data, self.pos)[0]; self.pos += 2; return v
        elif val_type in (4, 5):
            v = struct.unpack_from('<I', self.data, self.pos)[0]; self.pos += 4; return v
        elif val_type == 6:
            v = struct.unpack_from('<f', self.data, self.pos)[0]; self.pos += 4; return v
        elif val_type == 7:
            v = self.data[self.pos] != 0; self.pos += 1; return v
        elif val_type == 8:
            return self._read_string()
        elif val_type == 9:
            elem_type = self._read_u32()
            arr_len = self._read_u64()
            if elem_type == 8:
                result = []
                for _ in range(arr_len):
                    result.append(self._read_string())
                return result
            else:
                for _ in range(arr_len):
                    self._skip_value(elem_type, depth+1)
                return f"<array[{arr_len}] type={elem_type}>"
        elif val_type in (10, 11):
            v = struct.unpack_from('<Q', self.data, self.pos)[0]; self.pos += 8; return v
        elif val_type == 12:
            v = struct.unpack_from('<d', self.data, self.pos)[0]; self.pos += 8; return v
        else:
            raise ValueError(f"Unknown type {val_type}")

    def _parse(self):
        magic = struct.unpack_from('<I', self.data, self.pos)[0]; self.pos += 4
        if magic != 0x46554747:
            raise ValueError(f"Not GGUF: 0x{magic:08X}")
        version = struct.unpack_from('<I', self.data, self.pos)[0]; self.pos += 4
        n_tensors = self._read_u64()
        n_kv = self._read_u64()

        print(f"GGUF v{version}: {n_tensors} tensors, {n_kv} KV pairs")

        self.kv = {}
        for _ in range(n_kv):
            key = self._read_string()
            val_type = self._read_u32()
            if (key.startswith('tokenizer.') or
                key.startswith('ds4.') or
                key.startswith('deepseek4.') or
                key.startswith('general.') or
                key in ('llama.vocab_size', 'llama.block_count', 'llama.embedding_length')):
                self.kv[key] = self._read_value(val_type)
            else:
                self._skip_value(val_type)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    parser = GGUFParser(path)
    kv = parser.kv

    tokens = kv.get('tokenizer.ggml.tokens', [])
    merges = kv.get('tokenizer.ggml.merges', [])

    cmd = sys.argv[2] if len(sys.argv) > 2 else 'summary'

    if cmd == 'summary':
        cmd_summary(kv)
    elif cmd == 'find':
        if len(sys.argv) < 4:
            print("Usage: inspect_gguf_vocab.py <gguf> find <substring>")
            sys.exit(1)
        cmd_find(tokens, sys.argv[3])
    elif cmd == 'decode':
        if len(sys.argv) < 4:
            print("Usage: inspect_gguf_vocab.py <gguf> decode <id1,id2,...>")
            sys.exit(1)
        cmd_decode(tokens, sys.argv[3])
    elif cmd == 'garbled':
        cmd_garbled(tokens)
    elif cmd == 'check':
        cmd_check(tokens)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()
