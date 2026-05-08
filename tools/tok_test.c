/* Tokenizer smoke test: load GGUF, tokenize known strings, print token IDs.
 * Compile: cc -O1 -DDS4_TEST_DIMENSIONS -DDS4_NO_METAL -o /tmp/tok_test
 *          tok_test.c ds4.c -lm -lpthread
 * Usage:   /tmp/tok_test /tmp/test_ds4.gguf "Hello world" "test" "你好"
 *
 * This program ONLY tests the tokenizer (vocab_load + bpe_tokenize_text).
 * It does NOT run the model or allocate GPU memory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Minimal GGUF reader — just enough to read metadata arrays. */
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ===== Start of copied ds4.c tokenizer code ===== */

#define GGUF_VALUE_STRING 8
#define DS4_N_VOCAB 256

typedef struct { const char *ptr; uint64_t len; } ds4_str;

static int ds4_str_eq(ds4_str a, ds4_str b) {
    return a.len == b.len && memcmp(a.ptr, b.ptr, a.len) == 0;
}

static uint64_t hash_bytes(const char *ptr, uint64_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (uint64_t i = 0; i < len; i++) {
        h ^= (unsigned char)ptr[i];
        h *= 1099511628211ULL;
    }
    return h;
}

typedef struct {
    ds4_str key;
    int value;
    int used;
} str_i32_entry;

typedef struct {
    str_i32_entry *entry;
    uint64_t cap;
    uint64_t used;
} str_i32_table;

static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

static void *xmalloc(size_t sz) {
    void *p = malloc(sz);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

static void *xrealloc(void *p, size_t sz) {
    p = realloc(p, sz);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

static void table_init(str_i32_table *t, uint64_t expected) {
    t->cap = next_pow2(expected * 2 + 16);
    t->used = 0;
    t->entry = xcalloc((size_t)t->cap, sizeof(t->entry[0]));
}

static int table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value) {
    if (t->cap == 0) return 0;
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(ptr, len) & mask;
    while (t->entry[i].used) {
        ds4_str key = t->entry[i].key;
        if (key.len == len && memcmp(key.ptr, ptr, len) == 0) {
            *value = t->entry[i].value;
            return 1;
        }
        i = (i + 1) & mask;
    }
    return 0;
}

static void table_put(str_i32_table *t, ds4_str key, int value) {
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(key.ptr, key.len) & mask;
    while (t->entry[i].used) {
        if (ds4_str_eq(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }
    t->entry[i].used = 1;
    t->entry[i].key = key;
    t->entry[i].value = value;
    t->used++;
}

/* token vector */
typedef struct {
    int *v;
    int len;
    int cap;
} token_vec;

static void token_vec_push(token_vec *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = xrealloc(tv->v, (size_t)tv->cap * sizeof(tv->v[0]));
    }
    tv->v[tv->len++] = token;
}

typedef struct {
    ds4_str *token;
    int n_vocab;
    str_i32_table token_to_id;
    str_i32_table merge_rank;
} ds4_vocab;

/* UTF-8 helpers */
static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174))
        return b;
    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174))
            continue;
        if (x == b) return 256 + n;
        n++;
    }
    return b;
}

static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) { *(*p)++ = (char)cp; }
    else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static char *byte_encode(ds4_str in, uint64_t *out_len) {
    char *out = xmalloc((size_t)in.len * 4 + 1);
    char *p = out;
    for (uint64_t i = 0; i < in.len; i++)
        utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

static int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

typedef struct { char *ptr; uint64_t len; } owned_str;

static owned_str owned_copy(const char *ptr, uint64_t len) {
    owned_str s;
    s.ptr = xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

static int bpe_rank(const ds4_vocab *vocab, const owned_str *a, const owned_str *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : xmalloc((size_t)len);
    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);
    int rank = -1;
    table_get(&vocab->merge_rank, buf, len, &rank);
    if (buf != stack) free(buf);
    return rank;
}

static void bpe_emit_piece(const ds4_vocab *vocab, ds4_str raw_piece, token_vec *out) {
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_piece, &encoded_len);
    int n_sym = 0, cap_sym = 32;
    owned_str *sym = xcalloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1, best_rank = INT32_MAX;
        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) { best_rank = rank; best_i = i; }
        }
        if (best_i < 0) break;
        owned_str merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = xmalloc((size_t)merged.len);
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);
        free(sym[best_i].ptr); free(sym[best_i + 1].ptr);
        sym[best_i] = merged;
        for (int j = best_i + 1; j + 1 < n_sym; j++) sym[j] = sym[j + 1];
        n_sym--;
    }

    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token))
            token_vec_push(out, token);
        else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token))
                    token_vec_push(out, token);
            }
        }
        free(sym[i].ptr);
    }
    free(sym); free(encoded);
}

static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static int ascii_alpha(uint8_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
static int ascii_digit(uint8_t c) { return c >= '0' && c <= '9'; }
static int ascii_space(uint8_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }
static int ascii_newline(uint8_t c) { return c == '\n' || c == '\r'; }
static int joyai_ascii_punct_symbol(uint8_t c) {
    return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
}

static uint32_t utf8_peek_one(const char *s, uint64_t len, uint64_t pos, uint64_t *next) {
    uint8_t c0 = (uint8_t)s[pos];
    int n = utf8_len_from_first_byte(c0);
    if (pos + (uint64_t)n > len) n = 1;
    *next = pos + (uint64_t)n;
    if (n == 1) return c0;
    if (n == 2) return ((uint32_t)(c0 & 0x1f) << 6) | ((uint32_t)((uint8_t)s[pos+1] & 0x3f));
    if (n == 3) return ((uint32_t)(c0 & 0x0f) << 12) | ((uint32_t)((uint8_t)s[pos+1] & 0x3f) << 6) | ((uint32_t)((uint8_t)s[pos+2] & 0x3f));
    return ((uint32_t)(c0 & 0x07) << 18) | ((uint32_t)((uint8_t)s[pos+1] & 0x3f) << 12) | ((uint32_t)((uint8_t)s[pos+2] & 0x3f) << 6) | ((uint32_t)((uint8_t)s[pos+3] & 0x3f));
}

static int utf8_is_cjk_hira_kata(uint32_t cp) {
    return (cp >= 0x4e00 && cp <= 0x9fa5) || (cp >= 0x3040 && cp <= 0x309f) || (cp >= 0x30a0 && cp <= 0x30ff);
}

static int joyai_letter_like_at(const char *s, uint64_t len, uint64_t pos) {
    (void)len;
    uint8_t c = (uint8_t)s[pos];
    if (c < 128) return ascii_alpha(c);
    return 1;
}

static uint64_t joyai_consume_letters(const char *s, uint64_t len, uint64_t pos) {
    while (pos < len && joyai_letter_like_at(s, len, pos))
        pos = next_utf8_char(s, len, pos);
    return pos;
}

static int joyai_cjk_at(const char *s, uint64_t len, uint64_t pos) {
    if ((uint8_t)s[pos] < 128) return 0;
    uint64_t next;
    return utf8_is_cjk_hira_kata(utf8_peek_one(s, len, pos, &next));
}

static void bpe_tokenize_text(const ds4_vocab *vocab, const char *text, token_vec *out) {
    uint64_t len = strlen(text);
    uint64_t pos = 0;
    while (pos < len) {
        uint64_t start = pos;
        uint8_t c = (uint8_t)text[pos];
        if (ascii_digit(c)) {
            int ndigits = 0;
            while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) { pos++; ndigits++; }
        } else if (joyai_cjk_at(text, len, pos)) {
            do { pos = next_utf8_char(text, len, pos); }
            while (pos < len && joyai_cjk_at(text, len, pos));
        } else if (joyai_ascii_punct_symbol(c) && pos + 1 < len && ascii_alpha((uint8_t)text[pos+1])) {
            pos++;
            while (pos < len && ascii_alpha((uint8_t)text[pos])) pos++;
        } else if (joyai_letter_like_at(text, len, pos)) {
            pos = joyai_consume_letters(text, len, pos);
        } else if (!ascii_newline(c) && !joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len && joyai_letter_like_at(text, len, pos + 1)) {
            pos++;
            pos = joyai_consume_letters(text, len, pos);
        } else if (c == ' ' && pos + 1 < len && joyai_ascii_punct_symbol((uint8_t)text[pos+1])) {
            pos++;
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (joyai_ascii_punct_symbol(c)) {
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (ascii_space(c)) {
            uint64_t p = pos, last_newline_end = 0;
            while (p < len && ascii_space((uint8_t)text[p])) {
                if (ascii_newline((uint8_t)text[p])) last_newline_end = p + 1;
                p++;
            }
            if (last_newline_end) pos = last_newline_end;
            else if (p < len && p > pos + 1 &&
                     (joyai_letter_like_at(text, len, p) || joyai_ascii_punct_symbol((uint8_t)text[p])))
                pos = p - 1;
            else pos = p;
        } else {
            pos = next_utf8_char(text, len, pos);
        }
        if (pos == start) pos = next_utf8_char(text, len, pos);
        bpe_emit_piece(vocab, (ds4_str){text + start, pos - start}, out);
    }
}

/* Minimal GGUF reader — just enough for vocabulary */
typedef struct {
    uint8_t *data;
    uint64_t size;
    uint64_t off;
} gguf_reader;

static uint32_t ru32(gguf_reader *r) {
    uint32_t v; memcpy(&v, r->data + r->off, 4); r->off += 4; return v;
}
static uint64_t ru64(gguf_reader *r) {
    uint64_t v; memcpy(&v, r->data + r->off, 8); r->off += 8; return v;
}
static ds4_str rstr(gguf_reader *r) {
    uint64_t len = ru64(r);
    ds4_str s = {(char*)r->data + r->off, len};
    r->off += len;
    return s;
}

static void vocab_load_gguf(ds4_vocab *vocab, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    struct stat st; fstat(fd, &st);
    uint8_t *data = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) { perror("mmap"); exit(1); }

    gguf_reader r = {data, (uint64_t)st.st_size, 0};

    /* Skip magic + version */
    r.off += 8;

    /* Skip ntensors + nkv_table */
    ru64(&r);
    uint64_t nkv = ru64(&r);

    int got_tokens = 0, got_merges = 0;
    for (uint64_t i = 0; i < nkv; i++) {
        ds4_str key = rstr(&r);
        uint32_t vtype = ru32(&r);

        if (vtype == 8) { /* string */ rstr(&r); }
        else if (vtype == 9) { /* array — data follows INLINE */
            uint32_t atype = ru32(&r);
            uint64_t alen = ru64(&r);
            char *k = strndup(key.ptr, key.len);
            int is_tokens = !strcmp(k, "tokenizer.ggml.tokens");
            int is_merges = !strcmp(k, "tokenizer.ggml.merges");
            free(k);

            if (is_tokens && atype == 8) {
                vocab->n_vocab = (int)alen;
                vocab->token = xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
                table_init(&vocab->token_to_id, alen);
                for (uint64_t j = 0; j < alen; j++) {
                    vocab->token[j] = rstr(&r);
                    table_put(&vocab->token_to_id, vocab->token[j], (int)j);
                }
                got_tokens = 1;
            } else if (is_merges && atype == 8) {
                table_init(&vocab->merge_rank, alen);
                for (uint64_t j = 0; j < alen; j++) {
                    ds4_str merge = rstr(&r);
                    table_put(&vocab->merge_rank, merge, (int)j);
                }
                got_merges = 1;
            } else {
                /* Skip unknown array: for strings, skip alen strings */
                if (atype == 8) {
                    for (uint64_t j = 0; j < alen; j++) rstr(&r);
                } else {
                    /* fixed-size elements — skip */
                    size_t esz = (atype == 0 || atype == 1 || atype == 7) ? 1 :
                                (atype == 2 || atype == 3) ? 2 :
                                (atype == 4 || atype == 5 || atype == 6) ? 4 :
                                (atype == 10 || atype == 11 || atype == 12) ? 8 : 1;
                    r.off += (uint64_t)esz * alen;
                }
            }
        } else if (vtype == 4 || vtype == 5) { ru32(&r); }
        else if (vtype == 0) { r.off++; }
        else if (vtype == 10 || vtype == 11) { ru64(&r); }
        else if (vtype == 6) { r.off += 4; }
        else if (vtype == 7) { r.off++; }
        else { fprintf(stderr, "unknown KV type %u\n", vtype); exit(1); }
    }

    if (!got_tokens || !got_merges) {
        fprintf(stderr, "Missing tokenizer metadata\n"); exit(1);
    }
    munmap(data, (size_t)st.st_size);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <gguf_path> <text>...\n", argv[0]);
        return 1;
    }

    ds4_vocab vocab;
    memset(&vocab, 0, sizeof(vocab));
    vocab_load_gguf(&vocab, argv[1]);

    for (int i = 2; i < argc; i++) {
        token_vec out = {0};
        bpe_tokenize_text(&vocab, argv[i], &out);
        printf("%s -> [", argv[i]);
        for (int j = 0; j < out.len; j++) {
            if (j) printf(", ");
            printf("%d", out.v[j]);
        }
        printf("]\n");
        free(out.v);
    }
    return 0;
}
