// GGUF file loading for ds4.
// Parses GGUF metadata and provides tensor data access from mmap'd files.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::path::Path;

use anyhow::{bail, Context, Result};
use memmap2::Mmap;

/// GGUF metadata value types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

/// Tensor quantization type info.
#[derive(Debug, Clone)]
pub struct GgufTypeInfo {
    pub name: &'static str,
    pub block_elems: u32,
    pub block_bytes: u32,
}

/// All known GGUF tensor types (matching ds4.c gguf_types table exactly).
/// Index by the on-disk type number.
pub static GGUF_TYPES: &[GgufTypeInfo] = &[
    // 0: F32
    GgufTypeInfo { name: "f32",       block_elems: 1,   block_bytes: 4 },
    // 1: F16
    GgufTypeInfo { name: "f16",       block_elems: 1,   block_bytes: 2 },
    // 2: Q4_0
    GgufTypeInfo { name: "q4_0",      block_elems: 32,  block_bytes: 18 },
    // 3: Q4_1
    GgufTypeInfo { name: "q4_1",      block_elems: 32,  block_bytes: 20 },
    // 4-5: unknown
    GgufTypeInfo { name: "unk4",      block_elems: 0,   block_bytes: 0 },
    GgufTypeInfo { name: "unk5",      block_elems: 0,   block_bytes: 0 },
    // 6: Q5_0
    GgufTypeInfo { name: "q5_0",      block_elems: 32,  block_bytes: 22 },
    // 7: Q5_1
    GgufTypeInfo { name: "q5_1",      block_elems: 32,  block_bytes: 24 },
    // 8: Q8_0
    GgufTypeInfo { name: "q8_0",      block_elems: 32,  block_bytes: 34 },
    // 9: Q8_1
    GgufTypeInfo { name: "q8_1",      block_elems: 32,  block_bytes: 40 },
    // 10: Q2_K
    GgufTypeInfo { name: "q2_k",      block_elems: 256, block_bytes: 84 },
    // 11: Q3_K
    GgufTypeInfo { name: "q3_k",      block_elems: 256, block_bytes: 110 },
    // 12: Q4_K
    GgufTypeInfo { name: "q4_k",      block_elems: 256, block_bytes: 144 },
    // 13: Q5_K
    GgufTypeInfo { name: "q5_k",      block_elems: 256, block_bytes: 176 },
    // 14: Q6_K
    GgufTypeInfo { name: "q6_k",      block_elems: 256, block_bytes: 210 },
    // 15: Q8_K
    GgufTypeInfo { name: "q8_k",      block_elems: 256, block_bytes: 292 },
    // 16: IQ2_XXS
    GgufTypeInfo { name: "iq2_xxs",   block_elems: 256, block_bytes: 66 },
    // 17: IQ2_XS
    GgufTypeInfo { name: "iq2_xs",    block_elems: 256, block_bytes: 74 },
    // 18: IQ3_XXS
    GgufTypeInfo { name: "iq3_xxs",   block_elems: 256, block_bytes: 98 },
    // 19: IQ1_S
    GgufTypeInfo { name: "iq1_s",     block_elems: 256, block_bytes: 110 },
    // 20: IQ4_NL
    GgufTypeInfo { name: "iq4_nl",    block_elems: 256, block_bytes: 50 },
    // 21: IQ3_S
    GgufTypeInfo { name: "iq3_s",     block_elems: 256, block_bytes: 110 },
    // 22: IQ2_S
    GgufTypeInfo { name: "iq2_s",     block_elems: 256, block_bytes: 82 },
    // 23: IQ4_XS
    GgufTypeInfo { name: "iq4_xs",    block_elems: 256, block_bytes: 136 },
    // 24: I8
    GgufTypeInfo { name: "i8",        block_elems: 1,   block_bytes: 1 },
    // 25: I16
    GgufTypeInfo { name: "i16",       block_elems: 1,   block_bytes: 2 },
    // 26: I32
    GgufTypeInfo { name: "i32",       block_elems: 1,   block_bytes: 4 },
    // 27: I64
    GgufTypeInfo { name: "i64",       block_elems: 1,   block_bytes: 8 },
    // 28: F64
    GgufTypeInfo { name: "f64",       block_elems: 1,   block_bytes: 8 },
    // 29: IQ1_M
    GgufTypeInfo { name: "iq1_m",     block_elems: 256, block_bytes: 56 },
    // 30: BF16
    GgufTypeInfo { name: "bf16",      block_elems: 1,   block_bytes: 2 },
];

/// Convenience: look up type info safely.
pub fn gguf_type_info(tensor_type: u32) -> Option<&'static GgufTypeInfo> {
    let idx = tensor_type as usize;
    if idx >= GGUF_TYPES.len() {
        return None;
    }
    let info = &GGUF_TYPES[idx];
    if info.block_elems == 0 {
        None
    } else {
        Some(info)
    }
}

/// Get GGUF type index by name.
pub fn gguf_type_index(name: &str) -> Option<usize> {
    GGUF_TYPES.iter().position(|t| t.name == name)
}

/// Compute bytes for a tensor given elements and type.
pub fn tensor_byte_size(elements: u64, tensor_type: u32) -> Option<u64> {
    let info = gguf_type_info(tensor_type)?;
    let blocks = (elements + info.block_elems as u64 - 1) / info.block_elems as u64;
    blocks.checked_mul(info.block_bytes as u64)
}

// ============================================================================
// Metadata values
// ============================================================================

#[derive(Debug, Clone)]
pub struct GgufKv {
    pub key: String,
    pub value: GgufValue,
}

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Array(GgufArray),
}

#[derive(Debug, Clone)]
pub struct GgufArray {
    pub element_type: u32,
    pub elements: Vec<GgufValue>,
}

impl fmt::Display for GgufValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufValue::Uint8(v) => write!(f, "{}", v),
            GgufValue::Int8(v) => write!(f, "{}", v),
            GgufValue::Uint16(v) => write!(f, "{}", v),
            GgufValue::Int16(v) => write!(f, "{}", v),
            GgufValue::Uint32(v) => write!(f, "{}", v),
            GgufValue::Int32(v) => write!(f, "{}", v),
            GgufValue::Float32(v) => write!(f, "{}", v),
            GgufValue::Bool(v) => write!(f, "{}", v),
            GgufValue::String(v) => write!(f, "\"{}\"", v),
            GgufValue::Uint64(v) => write!(f, "{}", v),
            GgufValue::Int64(v) => write!(f, "{}", v),
            GgufValue::Float64(v) => write!(f, "{}", v),
            GgufValue::Array(arr) => write!(f, "[{} elements]", arr.elements.len()),
        }
    }
}

// ============================================================================
// Tensor descriptor
// ============================================================================

#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: String,
    pub ndim: u32,
    pub dims: Vec<u64>,
    pub tensor_type: u32,
    pub offset: u64,   // absolute file offset of tensor data
    pub elements: u64,
    pub bytes: u64,
}

// ============================================================================
// GgufModel — the loaded GGUF
// ============================================================================

pub struct GgufModel {
    pub file: File,
    pub mmap: Mmap,
    pub version: u32,
    pub alignment: u64,
    pub tensor_data_offset: u64,
    pub kv: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensor>,
    pub tensor_by_name: HashMap<String, usize>,
}

// ============================================================================
// Low-level cursor read helpers
// ============================================================================

fn read_bytes<'a>(data: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8]> {
    if *pos + n > data.len() {
        bail!("unexpected end of GGUF data at position {}", *pos);
    }
    let slice = &data[*pos..*pos + n];
    *pos += n;
    Ok(slice)
}

fn read_u16_le(data: &[u8], pos: &mut usize) -> Result<u16> {
    let bytes = read_bytes(data, pos, 2)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32_le(data: &[u8], pos: &mut usize) -> Result<u32> {
    let bytes = read_bytes(data, pos, 4)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u64_le(data: &[u8], pos: &mut usize) -> Result<u64> {
    let bytes = read_bytes(data, pos, 8)?;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn read_f32_le(data: &[u8], pos: &mut usize) -> Result<f32> {
    let bytes = read_bytes(data, pos, 4)?;
    Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_f64_le(data: &[u8], pos: &mut usize) -> Result<f64> {
    let bytes = read_bytes(data, pos, 8)?;
    Ok(f64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn read_string(data: &[u8], pos: &mut usize) -> Result<String> {
    let len = read_u64_le(data, pos)? as usize;
    let bytes = read_bytes(data, pos, len)?;
    Ok(String::from_utf8_lossy(bytes).into_owned())
}

fn skip_gguf_value(data: &[u8], pos: &mut usize, value_type: u32, depth: u32) -> Result<()> {
    if depth > 8 {
        bail!("GGUF metadata array nesting too deep");
    }
    match value_type {
        0 | 1 | 7 => { *pos += 1; }
        2 | 3 => { *pos += 2; }
        4 | 5 | 6 => { *pos += 4; }
        8 => {
            let len = read_u64_le(data, pos)? as usize;
            *pos += len;
        }
        9 => {
            let elem_type = read_u32_le(data, pos)?;
            let len = read_u64_le(data, pos)? as usize;
            for _ in 0..len {
                skip_gguf_value(data, pos, elem_type, depth + 1)?;
            }
        }
        10 | 11 | 12 => { *pos += 8; }
        _ => bail!("unknown GGUF value type: {}", value_type),
    }
    Ok(())
}

fn read_gguf_value(data: &[u8], pos: &mut usize, value_type: u32, depth: u32) -> Result<GgufValue> {
    if depth > 8 {
        bail!("GGUF metadata array nesting too deep");
    }
    Ok(match value_type {
        0 => GgufValue::Uint8(read_bytes(data, pos, 1)?[0]),
        1 => GgufValue::Int8(read_bytes(data, pos, 1)?[0] as i8),
        2 => GgufValue::Uint16(read_u16_le(data, pos)?),
        3 => GgufValue::Int16(read_u16_le(data, pos)? as i16),
        4 => GgufValue::Uint32(read_u32_le(data, pos)?),
        5 => {
            let bytes = read_bytes(data, pos, 4)?;
            GgufValue::Int32(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        }
        6 => GgufValue::Float32(read_f32_le(data, pos)?),
        7 => GgufValue::Bool(read_bytes(data, pos, 1)?[0] != 0),
        8 => GgufValue::String(read_string(data, pos)?),
        9 => {
            let elem_type = read_u32_le(data, pos)?;
            let len = read_u64_le(data, pos)? as usize;
            let mut elements = Vec::with_capacity(len);
            for _ in 0..len {
                elements.push(read_gguf_value(data, pos, elem_type, depth + 1)?);
            }
            GgufValue::Array(GgufArray { element_type: elem_type, elements })
        }
        10 => GgufValue::Uint64(read_u64_le(data, pos)?),
        11 => {
            let bytes = read_bytes(data, pos, 8)?;
            GgufValue::Int64(i64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
            ]))
        }
        12 => GgufValue::Float64(read_f64_le(data, pos)?),
        _ => bail!("unknown GGUF value type: {}", value_type),
    })
}

// ============================================================================
// GgufModel implementation
// ============================================================================

impl GgufModel {
    /// Open and parse a GGUF file. The file is mmap'd for zero-copy tensor access.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("failed to open GGUF: {}", path.as_ref().display()))?;

        let mmap = unsafe {
            Mmap::map(&file)
                .with_context(|| "failed to mmap GGUF file")?
        };

        let data: &[u8] = &mmap;

        if data.len() < 40 {
            bail!("GGUF file too small");
        }

        let mut pos: usize = 0;

        // Magic: "GGUF" = 0x46554747
        let magic = read_u32_le(data, &mut pos)?;
        if magic != 0x46554747 {
            bail!("not a valid GGUF file (bad magic: 0x{:08X})", magic);
        }

        // Version
        let version = read_u32_le(data, &mut pos)?;
        if version < 2 || version > 3 {
            bail!("unsupported GGUF version: {}", version);
        }

        // n_tensors and n_kv
        let n_tensors = read_u64_le(data, &mut pos)?;
        let n_kv = read_u64_le(data, &mut pos)?;

        // --- Read metadata key-value pairs ---
        // We store only scalars and strings useful for model configuration.
        // Large arrays (e.g. tokenizer vocab) are skipped to save memory.
        let mut kv: HashMap<String, GgufValue> = HashMap::new();
        let small_kv_keys: &[&str] = &[
            "general.architecture",
            "general.name",
            "general.alignment",
            "general.file_type",
            "ds4.n_layer",
            "ds4.n_embd",
            "ds4.vocab_size",
            "ds4.attn_n_head",
            "ds4.attn_n_head_kv",
            "ds4.attn_n_head_exp",
            "ds4.n_ff_exp",
            "ds4.n_expert",
            "ds4.n_expert_used",
            "ds4.n_lora_q",
            "ds4.n_lora_o",
            "ds4.n_out_group",
            "ds4.n_hc",
            "ds4.n_indexer_head",
            "ds4.n_indexer_head_dim",
            "ds4.n_train_ctx",
            "ds4.n_expert_per_token",
            "ds4.layer_compress_ratio",
            "tokenizer.ggml.model",
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.merges",
            // llama.cpp compatibility aliases
            "llama.block_count",
            "llama.embedding_length",
            "llama.vocab_size",
            "llama.attention.head_count",
            "llama.attention.head_count_kv",
            "llama.context_length",
        ];

        let mut alignment: u64 = 32;

        for _ki in 0..n_kv {
            let key = read_string(data, &mut pos)?;
            let value_type = read_u32_le(data, &mut pos)?;

            // Only fully decode values for small/config keys; skip the rest
            let should_store = small_kv_keys.iter().any(|k| *k == key)
                || key.starts_with("general.")
                || key.starts_with("ds4.")
                || key.starts_with("deepseek4.")
                || key.starts_with("llama.")
                || key.starts_with("tokenizer.");

            if should_store {
                let value = read_gguf_value(data, &mut pos, value_type, 0)?;

                // Extract alignment early
                if key == "general.alignment" {
                    if let GgufValue::Uint32(a) = &value {
                        if *a > 0 {
                            alignment = *a as u64;
                        }
                    }
                }

                kv.insert(key, value);
            } else {
                // Skip the value
                skip_gguf_value(data, &mut pos, value_type, 0)?;
            }
        }

        // --- Read tensor infos ---
        let mut tensor_infos: Vec<(String, u32, Vec<u64>, u32, u64, u64)> =
            Vec::with_capacity(n_tensors as usize);

        for _ti in 0..n_tensors {
            let name = read_string(data, &mut pos)?;
            let ndim = read_u32_le(data, &mut pos)?;
            if ndim == 0 || ndim > 8 {
                bail!("tensor '{}' has unsupported ndim={}", name, ndim);
            }

            let mut dims = Vec::with_capacity(ndim as usize);
            let mut elements: u64 = 1;
            for _d in 0..ndim {
                let dim = read_u64_le(data, &mut pos)?;
                if dim != 0 {
                    elements = elements.checked_mul(dim)
                        .ok_or_else(|| anyhow::anyhow!("tensor '{}' element count overflow", name))?;
                }
                dims.push(dim);
            }

            let tensor_type = read_u32_le(data, &mut pos)?;

            // GGUF v3: relative offset for tensor data
            let rel_offset = read_u64_le(data, &mut pos)?;

            let bytes = if elements == 0 {
                0
            } else {
                tensor_byte_size(elements, tensor_type)
                    .unwrap_or_else(|| {
                        // Unknown type: treat as raw bytes
                        0
                    })
            };

            tensor_infos.push((name, ndim, dims, tensor_type, rel_offset, bytes));
        }

        // Data section starts after all tensor infos, aligned
        let data_start = ((pos as u64) + alignment - 1) & !(alignment - 1);

        // Build tensor table
        let mut tensors: Vec<GgufTensor> = Vec::with_capacity(n_tensors as usize);
        let mut tensor_by_name: HashMap<String, usize> = HashMap::new();

        for (name, ndim, dims, tensor_type, rel_offset, bytes) in tensor_infos {
            let offset = data_start + rel_offset;
            let elements: u64 = dims.iter().product();

            let idx = tensors.len();
            tensor_by_name.insert(name.clone(), idx);
            tensors.push(GgufTensor {
                name,
                ndim,
                dims,
                tensor_type,
                offset,
                elements,
                bytes,
            });
        }

        Ok(GgufModel {
            file,
            mmap,
            version,
            alignment,
            tensor_data_offset: data_start,
            kv,
            tensors,
            tensor_by_name,
        })
    }

    /// Get the tensor info by name.
    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.tensor_by_name.get(name).map(|&idx| &self.tensors[idx])
    }

    /// Get raw tensor data as a byte slice.
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let tensor = self.tensor(name)?;
        let start = tensor.offset as usize;
        let end = start + tensor.bytes as usize;
        if end > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[start..end])
    }

    /// Get string metadata value.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.kv.get(key) {
            Some(GgufValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get u32 metadata value.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.kv.get(key) {
            Some(GgufValue::Uint32(v)) => Some(*v),
            Some(GgufValue::Uint64(v)) => Some(*v as u32),
            _ => None,
        }
    }

    /// Get u64 metadata value.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.kv.get(key) {
            Some(GgufValue::Uint64(v)) => Some(*v),
            Some(GgufValue::Uint32(v)) => Some(*v as u64),
            _ => None,
        }
    }

    /// Get i32 metadata value.
    pub fn get_i32(&self, key: &str) -> Option<i32> {
        match self.kv.get(key) {
            Some(GgufValue::Int32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get bool metadata value.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.kv.get(key) {
            Some(GgufValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get float32 metadata value.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.kv.get(key) {
            Some(GgufValue::Float32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get an array metadata value.
    pub fn get_array(&self, key: &str) -> Option<&GgufArray> {
        match self.kv.get(key) {
            Some(GgufValue::Array(arr)) => Some(arr),
            _ => None,
        }
    }
}

// ============================================================================
// GGUF Writer (for gen_test_gguf and other test utilities)
// ============================================================================

use std::fs;
use std::io::{self, Write};

/// A builder for writing GGUF files.
pub struct GgufWriter<W: Write> {
    writer: W,
    version: u32,
    alignment: u64,
    kv: Vec<(String, GgufValue)>,
    tensor_infos: Vec<(String, Vec<u64>, u32, Vec<u8>)>, // name, dims, tensor_type, data
}

impl GgufWriter<io::BufWriter<fs::File>> {
    /// Create a GGUF writer that writes to a file.
    pub fn create_file(path: impl AsRef<Path>, version: u32, alignment: u64) -> Result<Self> {
        let file = fs::File::create(path.as_ref())
            .with_context(|| format!("failed to create GGUF: {}", path.as_ref().display()))?;
        let writer = io::BufWriter::new(file);
        Ok(GgufWriter {
            writer,
            version,
            alignment,
            kv: Vec::new(),
            tensor_infos: Vec::new(),
        })
    }
}

impl<W: Write> GgufWriter<W> {
    fn write_u32_le(&mut self, v: u32) -> io::Result<()> {
        self.writer.write_all(&v.to_le_bytes())
    }

    fn write_u64_le(&mut self, v: u64) -> io::Result<()> {
        self.writer.write_all(&v.to_le_bytes())
    }

    fn write_string_raw(&mut self, s: &str) -> io::Result<()> {
        self.write_u64_le(s.len() as u64)?;
        self.writer.write_all(s.as_bytes())
    }

    fn write_value(&mut self, value: &GgufValue) -> io::Result<()> {
        match value {
            GgufValue::Uint8(v)  => self.writer.write_all(&[*v])?,
            GgufValue::Int8(v)   => self.writer.write_all(&[*v as u8])?,
            GgufValue::Uint16(v) => self.writer.write_all(&v.to_le_bytes())?,
            GgufValue::Int16(v)  => self.writer.write_all(&v.to_le_bytes())?,
            GgufValue::Uint32(v) => self.write_u32_le(*v)?,
            GgufValue::Int32(v)  => self.writer.write_all(&v.to_le_bytes())?,
            GgufValue::Float32(v) => self.writer.write_all(&v.to_le_bytes())?,
            GgufValue::Bool(v)   => self.writer.write_all(&[if *v { 1u8 } else { 0u8 }])?,
            GgufValue::String(s) => self.write_string_raw(s)?,
            GgufValue::Uint64(v) => self.write_u64_le(*v)?,
            GgufValue::Int64(v)  => self.writer.write_all(&v.to_le_bytes())?,
            GgufValue::Float64(v) => self.writer.write_all(&v.to_le_bytes())?,
            GgufValue::Array(arr) => {
                self.write_u32_le(arr.element_type)?;
                self.write_u64_le(arr.elements.len() as u64)?;
                for elem in &arr.elements {
                    self.write_value(elem)?;
                }
            }
        }
        Ok(())
    }

    /// Add a metadata key-value pair.
    pub fn add_meta(&mut self, key: &str, value: GgufValue) {
        self.kv.push((key.to_string(), value));
    }

    /// Add a tensor with its name, dimensions, GGUF type, and raw data bytes.
    pub fn add_tensor(&mut self, name: &str, dims: Vec<u64>, tensor_type: u32, data: Vec<u8>) {
        self.tensor_infos.push((name.to_string(), dims, tensor_type, data));
    }

    /// Write the complete GGUF file and return the writer.
    pub fn finish(mut self) -> io::Result<W> {
        let alignment = self.alignment.max(32);
        let n_tensors = self.tensor_infos.len() as u64;
        let n_kv = self.kv.len() as u64;

        // Take ownership of kv and tensor_infos to avoid borrow conflicts
        let kv = std::mem::take(&mut self.kv);
        let tensor_infos = std::mem::take(&mut self.tensor_infos);

        // --- Pre-compute all byte positions ---
        // Header: magic(4) + version(4) + n_tensors(8) + n_kv(8) = 24 bytes
        let mut pos: u64 = 24;

        // Metadata
        for (_key, value) in &kv {
            pos += 8 + _key.len() as u64; // string
            pos += 4; // type code
            pos += value_byte_size(value);
        }

        // Tensor infos
        for (name, dims, _tensor_type, _data) in &tensor_infos {
            pos += 8 + name.len() as u64; // name string
            pos += 4; // ndim
            pos += (dims.len() as u64) * 8; // dims
            pos += 4; // tensor_type
            pos += 8; // rel_offset
        }

        // Data section
        let data_start = (pos + alignment - 1) & !(alignment - 1);

        // Compute absolute offsets and rel_offsets for each tensor
        let mut abs_offsets: Vec<u64> = Vec::with_capacity(tensor_infos.len());
        let mut rel_offsets: Vec<u64> = Vec::with_capacity(tensor_infos.len());
        {
            let mut data_off = data_start;
            for (_name, _dims, _tensor_type, data) in &tensor_infos {
                let padded = (data_off + alignment - 1) & !(alignment - 1);
                abs_offsets.push(padded);
                rel_offsets.push(padded - data_start);
                data_off = padded + data.len() as u64;
            }
        }

        // --- Write everything in order ---
        // Header
        self.writer.write_all(&0x46554747u32.to_le_bytes())?;
        self.writer.write_all(&self.version.to_le_bytes())?;
        self.writer.write_all(&n_tensors.to_le_bytes())?;
        self.writer.write_all(&n_kv.to_le_bytes())?;

        // Metadata
        for (key, value) in &kv {
            self.write_string_raw(key)?;
            let type_code: u32 = match value {
                GgufValue::Uint8(_) => 0,
                GgufValue::Int8(_) => 1,
                GgufValue::Uint16(_) => 2,
                GgufValue::Int16(_) => 3,
                GgufValue::Uint32(_) => 4,
                GgufValue::Int32(_) => 5,
                GgufValue::Float32(_) => 6,
                GgufValue::Bool(_) => 7,
                GgufValue::String(_) => 8,
                GgufValue::Array(_) => 9,
                GgufValue::Uint64(_) => 10,
                GgufValue::Int64(_) => 11,
                GgufValue::Float64(_) => 12,
            };
            self.write_u32_le(type_code)?;
            self.write_value(value)?;
        }

        // Tensor infos with correct rel_offsets
        for (i, (name, dims, tensor_type, _data)) in tensor_infos.iter().enumerate() {
            self.write_string_raw(name)?;
            self.write_u32_le(dims.len() as u32)?;
            for &d in dims {
                self.write_u64_le(d)?;
            }
            self.write_u32_le(*tensor_type)?;
            self.write_u64_le(rel_offsets[i])?;
        }

        // Padding to data section (use pre-computed data_start)
        // stream_position should equal pos at this point; write padding directly
        let pad = data_start.saturating_sub(pos);
        if pad > 0 {
            self.writer.write_all(&vec![0u8; pad as usize])?;
        }

        // Write tensor data
        let mut cur_pos = data_start;
        for (i, (_name, _dims, _tensor_type, data)) in tensor_infos.iter().enumerate() {
            let target = abs_offsets[i];
            if target > cur_pos {
                let pad_bytes = (target - cur_pos) as usize;
                self.writer.write_all(&vec![0u8; pad_bytes])?;
            }
            self.writer.write_all(data)?;
            cur_pos = target + data.len() as u64;
        }

        self.writer.flush()?;
        Ok(self.writer)
    }
}

/// Compute byte size of a GGUF value (for file layout calculations).
fn value_byte_size(value: &GgufValue) -> u64 {
    match value {
        GgufValue::Uint8(_) | GgufValue::Int8(_) | GgufValue::Bool(_) => 1,
        GgufValue::Uint16(_) | GgufValue::Int16(_) => 2,
        GgufValue::Uint32(_) | GgufValue::Int32(_) | GgufValue::Float32(_) => 4,
        GgufValue::Uint64(_) | GgufValue::Int64(_) | GgufValue::Float64(_) => 8,
        GgufValue::String(s) => 8 + s.len() as u64,
        GgufValue::Array(arr) => {
            let mut size: u64 = 12; // element_type(4) + len(8)
            for elem in &arr.elements {
                size += value_byte_size(elem);
            }
            size
        }
    }
}
