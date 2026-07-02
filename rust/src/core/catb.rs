//! `.catb` v4 ("CATB0001") bitstream telemetry container reader.
//!
//! Implemented from the CC0 .catb v4 format specification
//! (codec-analyzer docs/catb-v4-format.md); no GPL code consulted.
//!
//! The container stores per-frame decoder telemetry with a 10-entry section
//! directory, an interned string table, a JSON meta blob, and fixed-size
//! little-endian records (FRAME 80 B, BLOCK 156 B, REF 68 B). The file is
//! mmap'd; FRAME records are parsed eagerly (small), BLOCK/REF records are
//! parsed lazily per frame/block. The `syntax`, `cabac`, `transforms`,
//! `coeffs`, and `frame_aux` sections are NOT parsed in M0 — only their
//! directory entries are retained.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

/// File magic: first 8 bytes of every `.catb` file.
pub const CATB_MAGIC: &[u8; 8] = b"CATB0001";
/// Container version implemented by this reader.
pub const CATB_VERSION: u32 = 4;

/// Fixed header size: magic(8) + version(4) + frame_count(4) + 10×(u64,u64).
pub const HEADER_SIZE: usize = 176;
/// FRAME record size in bytes.
pub const FRAME_RECORD_SIZE: usize = 80;
/// BLOCK record size in bytes.
pub const BLOCK_RECORD_SIZE: usize = 156;
/// REF record size in bytes.
pub const REF_RECORD_SIZE: usize = 68;

// Section directory indices (fixed, normative order).
pub const SEC_STRINGS: usize = 0;
pub const SEC_META: usize = 1;
pub const SEC_FRAMES: usize = 2;
pub const SEC_BLOCKS: usize = 3;
pub const SEC_SYNTAX: usize = 4;
pub const SEC_CABAC: usize = 5;
pub const SEC_TRANSFORMS: usize = 6;
pub const SEC_REFS: usize = 7;
pub const SEC_COEFFS: usize = 8;
pub const SEC_FRAME_AUX: usize = 9;

// BLOCK `mv_flags` presence bits (§6.1).
pub const MV_HAS_MVP: u32 = 1;
pub const MV_HAS_MVD: u32 = 2;
pub const MV_HAS_MV: u32 = 4;
pub const MV_HAS_LIST_INDEX: u32 = 8;
pub const MV_HAS_POC: u32 = 16;
pub const MV_HAS_FRAME: u32 = 32;
pub const MV_HAS_LONG_TERM: u32 = 64;
pub const MV_LONG_TERM_VALUE: u32 = 128;

// REF `ref_flags` presence bits (§10.1).
pub const REF_HAS_MVP: u32 = 1;
pub const REF_HAS_MVD: u32 = 2;
pub const REF_HAS_MV: u32 = 4;
pub const REF_HAS_FRAME: u32 = 8;
pub const REF_HAS_LONG_TERM: u32 = 16;
pub const REF_LONG_TERM_VALUE: u32 = 32;
pub const REF_HAS_PU_PART: u32 = 64;

/// One directory entry: absolute byte offset + byte size of a section.
#[derive(Debug, Clone, Copy, Default)]
pub struct CatbSection {
    pub offset: u64,
    pub size: u64,
}

/// Decoded meta blob (section 1). Only the fields M0 needs; the raw
/// `parameter_sets` array is kept as JSON for downstream inspection.
#[derive(Debug, Clone)]
pub struct CatbMeta {
    pub schema_version: Option<i64>,
    /// `decoder.codec` — e.g. "hevc"/"HEVC", "avc", "av1". Empty if absent.
    pub codec: String,
    /// `decoder.contract` version string; empty if absent.
    pub contract: String,
    /// `decoder.capture_level`; "" means full capture.
    pub capture_level: String,
    /// Raw `parameter_sets` JSON array (Null when the key is absent).
    pub parameter_sets: serde_json::Value,
    /// Whether the `frames_meta` parallel array is present.
    pub has_frames_meta: bool,
}

/// One FRAME record (80 bytes), with string ids resolved and −1 sentinels
/// mapped to `None` (§5, §13).
#[derive(Debug, Clone)]
pub struct CatbFrame {
    /// Decode-order frame index (unique).
    pub index: i32,
    /// Picture order count (AV1: order-hint equivalent).
    pub poc: i32,
    /// Resolved frame type label ("IDR", "CRA", "B", ... — §14.1). "" if absent.
    pub frame_type: String,
    /// Frame is output/displayed.
    pub output: bool,
    pub vps_id: Option<i32>,
    pub sps_id: Option<i32>,
    pub pps_id: Option<i32>,
    pub slice_bits: Option<i64>,
    pub header_bits: Option<i64>,
    pub data_bits: Option<i64>,
    /// Absolute BLOCK record index of this frame's first block.
    pub block_off: u64,
    /// Number of BLOCK records for this frame.
    pub block_n: i32,
    /// Byte offset into `frame_aux` of this frame's aux JSON (unused in M0).
    pub aux_off: u64,
    /// Byte length of this frame's aux JSON (0 = none; unused in M0).
    pub aux_n: i64,
}

/// One BLOCK record (156 bytes) with presence bits and string ids resolved
/// (§6). Optional MV pairs / reference fields follow `mv_flags` (§6.1).
#[derive(Debug, Clone, PartialEq)]
pub struct BsBlock {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub ctu_address: i64,
    /// Block QP. Spec §13: NO sentinel — an absent QP is stored as 0 and is
    /// indistinguishable from QP 0; the stored value is surfaced as-is.
    pub qp: i32,
    /// Partition shape label (e.g. "64x64", "2Nx2N"); "" if absent.
    pub partition: String,
    /// Prediction mode label (e.g. "Intra", "Inter", "Skip"); "" if absent.
    pub prediction_mode: String,
    pub bits: i64,
    pub bit_offset: i64,
    pub exactness_flags: i32,
    // Sub-record ranges (absolute record indices into sections 4–7).
    pub syntax_off: u64,
    pub syntax_n: i32,
    pub cabac_off: u64,
    pub cabac_n: i32,
    pub tx_off: u64,
    pub tx_n: i32,
    pub ref_off: u64,
    pub ref_n: i32,
    /// Raw presence bitmask (§6.1); kept for long-term bit inspection.
    pub mv_flags: u32,
    /// MV predictor, quarter-pel luma units (§14.5).
    pub mvp: Option<(i32, i32)>,
    /// MV difference, quarter-pel luma units.
    pub mvd: Option<(i32, i32)>,
    /// Final MV, quarter-pel luma units.
    pub mv: Option<(i32, i32)>,
    /// Reference summary label, e.g. "L0[0]", "Bi(L0[0],L1[1])" (§14.4).
    pub reference: Option<String>,
    /// "L0" / "L1".
    pub reference_list: Option<String>,
    /// Decoder-resolved reference label.
    pub reference_label: Option<String>,
    pub reference_list_index: Option<i32>,
    pub reference_poc: Option<i32>,
    pub reference_frame: Option<i32>,
    /// `reference_long_term` boolean (present iff `MV_HAS_LONG_TERM`).
    pub reference_long_term: Option<bool>,
}

/// One REF record (68 bytes): a normalized per-PU reference row (§10).
#[derive(Debug, Clone, PartialEq)]
pub struct BsRef {
    /// "L0" or "L1"; `None` when string id 0.
    pub list: Option<String>,
    pub list_index: i32,
    pub reference_poc: i32,
    /// Decoder-resolved reference label; `None` when string id 0.
    pub label: Option<String>,
    pub pu_x: i32,
    pub pu_y: i32,
    pub pu_w: i32,
    pub pu_h: i32,
    pub ref_flags: u32,
    pub mvp: Option<(i32, i32)>,
    pub mvd: Option<(i32, i32)>,
    pub mv: Option<(i32, i32)>,
    pub reference_frame: Option<i32>,
    pub long_term: Option<bool>,
    pub pu_part_index: Option<i32>,
}

/// An open, header-validated `.catb` v4 file.
///
/// FRAME records are parsed eagerly; BLOCK and REF records are parsed on
/// demand via [`CatbFile::blocks_for_frame`] / [`CatbFile::refs_for_block`].
pub struct CatbFile {
    mmap: Mmap,
    pub frame_count: u32,
    pub sections: [CatbSection; 10],
    pub strings: Vec<String>,
    pub meta: CatbMeta,
    pub frames: Vec<CatbFrame>,
}

impl std::fmt::Debug for CatbFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CatbFile")
            .field("frame_count", &self.frame_count)
            .field("strings", &self.strings.len())
            .field("meta", &self.meta)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Bounds-checked little-endian readers (no panics on malformed input)
// ---------------------------------------------------------------------------

fn get_bytes<'a>(data: &'a [u8], off: usize, len: usize, what: &str) -> Result<&'a [u8], String> {
    let end = off
        .checked_add(len)
        .ok_or_else(|| format!("catb: offset overflow reading {what}"))?;
    data.get(off..end)
        .ok_or_else(|| format!("catb: truncated file reading {what} at byte {off} (+{len})"))
}

fn read_u32(data: &[u8], off: usize, what: &str) -> Result<u32, String> {
    let b = get_bytes(data, off, 4, what)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_i32(data: &[u8], off: usize, what: &str) -> Result<i32, String> {
    let b = get_bytes(data, off, 4, what)?;
    Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_u64(data: &[u8], off: usize, what: &str) -> Result<u64, String> {
    let b = get_bytes(data, off, 8, what)?;
    Ok(u64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

fn read_i64(data: &[u8], off: usize, what: &str) -> Result<i64, String> {
    let b = get_bytes(data, off, 8, what)?;
    Ok(i64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

/// −1 sentinel → `None` (§13: values are otherwise ≥ 0; any negative is
/// treated as absent).
fn opt_i32(v: i32) -> Option<i32> {
    if v < 0 {
        None
    } else {
        Some(v)
    }
}

fn opt_i64(v: i64) -> Option<i64> {
    if v < 0 {
        None
    } else {
        Some(v)
    }
}

impl CatbFile {
    /// Open and validate a `.catb` v4 file (mmap-based).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| format!("catb: cannot open {}: {e}", path.display()))?;
        // SAFETY: read-only mapping of a regular file via the memmap2 API.
        // The file may in principle be truncated by another process while
        // mapped; all subsequent accesses are bounds-checked slices.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("catb: mmap failed for {}: {e}", path.display()))?;
        Self::from_mmap(mmap)
    }

    fn from_mmap(mmap: Mmap) -> Result<Self, String> {
        let data: &[u8] = &mmap;
        // §7 fixed wording (UX spec 04): magic/version mismatch surfaces this
        // exact string in the sidebar panel — do not reword.
        const UNSUPPORTED: &str =
            "Unsupported .catb (need v4) — regenerate with codec-analyzer ≥0.8.3";
        let magic = get_bytes(data, 0, 8, "magic")?;
        if magic != CATB_MAGIC {
            return Err(UNSUPPORTED.to_string());
        }
        let version = read_u32(data, 8, "version")?;
        if version != CATB_VERSION {
            return Err(UNSUPPORTED.to_string());
        }
        let frame_count = read_u32(data, 12, "frame_count")?;

        // Directory: 10 × (u64 offset, u64 size), bounds-checked against file size.
        let file_len = data.len() as u64;
        let mut sections = [CatbSection::default(); 10];
        for (i, sec) in sections.iter_mut().enumerate() {
            let base = 16 + i * 16;
            let offset = read_u64(data, base, "directory offset")?;
            let size = read_u64(data, base + 8, "directory size")?;
            let end = offset
                .checked_add(size)
                .ok_or_else(|| format!("catb: directory entry {i} offset+size overflows"))?;
            if end > file_len {
                return Err(format!(
                    "catb: directory entry {i} out of bounds \
                     (offset {offset} + size {size} > file size {file_len})"
                ));
            }
            *sec = CatbSection { offset, size };
        }

        let strings = parse_string_table(section_slice(data, &sections[SEC_STRINGS]))?;
        let meta = parse_meta(section_slice(data, &sections[SEC_META]))?;

        // Eager FRAME parse: frame_count × 80 bytes (small).
        let frames_bytes = section_slice(data, &sections[SEC_FRAMES]);
        let need = (frame_count as usize)
            .checked_mul(FRAME_RECORD_SIZE)
            .ok_or_else(|| "catb: frame_count overflows".to_string())?;
        if frames_bytes.len() < need {
            return Err(format!(
                "catb: frames section too small ({} bytes for {frame_count} frames, need {need})",
                frames_bytes.len()
            ));
        }
        let mut frames = Vec::with_capacity(frame_count as usize);
        for i in 0..frame_count as usize {
            frames.push(parse_frame_record(frames_bytes, i * FRAME_RECORD_SIZE, &strings)?);
        }

        Ok(Self {
            mmap,
            frame_count,
            sections,
            strings,
            meta,
            frames,
        })
    }

    /// Raw bytes of a directory section (validated at open time).
    pub fn section_bytes(&self, idx: usize) -> &[u8] {
        section_slice(&self.mmap, &self.sections[idx])
    }

    /// Resolve a string id: id 0 and out-of-range ids yield `""` (§3).
    pub fn resolve_str(&self, id: i32) -> &str {
        resolve_str(&self.strings, id)
    }

    /// Lazily parse frame `frame_idx`'s BLOCK records (decode order).
    pub fn blocks_for_frame(&self, frame_idx: usize) -> Result<Vec<BsBlock>, String> {
        let frame = self
            .frames
            .get(frame_idx)
            .ok_or_else(|| format!("catb: frame index {frame_idx} out of range"))?;
        let blocks_bytes = self.section_bytes(SEC_BLOCKS);
        let n = frame.block_n.max(0) as usize;
        // Reject counts a malformed/hostile file cannot back with actual
        // section bytes *before* allocating — an unchecked
        // `Vec::with_capacity(2^31)` of ~200 B records would abort on OOM.
        let max_records = blocks_bytes.len() / BLOCK_RECORD_SIZE;
        if n > max_records {
            return Err(format!(
                "catb: frame {frame_idx} claims {n} blocks but the BLOCK \
                 section only holds {max_records} records"
            ));
        }
        let mut out = Vec::with_capacity(n);
        for j in 0..n {
            let rec_idx = (frame.block_off as usize)
                .checked_add(j)
                .ok_or_else(|| "catb: block record index overflows".to_string())?;
            let byte_off = rec_idx
                .checked_mul(BLOCK_RECORD_SIZE)
                .ok_or_else(|| "catb: block byte offset overflows".to_string())?;
            out.push(parse_block_record(blocks_bytes, byte_off, &self.strings)?);
        }
        Ok(out)
    }

    /// Lazily parse a block's REF records (per-PU reference rows).
    pub fn refs_for_block(&self, block: &BsBlock) -> Result<Vec<BsRef>, String> {
        let refs_bytes = self.section_bytes(SEC_REFS);
        let n = block.ref_n.max(0) as usize;
        // Same OOM guard as `blocks_for_frame`: never allocate more record
        // slots than the REF section can physically contain.
        let max_records = refs_bytes.len() / REF_RECORD_SIZE;
        if n > max_records {
            return Err(format!(
                "catb: block claims {n} refs but the REF section only holds \
                 {max_records} records"
            ));
        }
        let mut out = Vec::with_capacity(n);
        for j in 0..n {
            let rec_idx = (block.ref_off as usize)
                .checked_add(j)
                .ok_or_else(|| "catb: ref record index overflows".to_string())?;
            let byte_off = rec_idx
                .checked_mul(REF_RECORD_SIZE)
                .ok_or_else(|| "catb: ref byte offset overflows".to_string())?;
            out.push(parse_ref_record(refs_bytes, byte_off, &self.strings)?);
        }
        Ok(out)
    }
}

/// Section slice — directory entries are validated against the file size at
/// open time, so this cannot go out of bounds.
fn section_slice<'a>(data: &'a [u8], sec: &CatbSection) -> &'a [u8] {
    let start = sec.offset as usize;
    let end = start.saturating_add(sec.size as usize);
    data.get(start..end).unwrap_or(&[])
}

fn resolve_str(strings: &[String], id: i32) -> &str {
    if id <= 0 {
        return "";
    }
    strings.get(id as usize).map(|s| s.as_str()).unwrap_or("")
}

/// `None` for id 0 / out-of-range (absent), `Some(label)` otherwise (§13).
fn opt_str(strings: &[String], id: i32) -> Option<String> {
    let s = resolve_str(strings, id);
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

fn parse_string_table(bytes: &[u8]) -> Result<Vec<String>, String> {
    if bytes.is_empty() {
        // Legal only in degenerate files; treat as a table with just id 0.
        return Ok(vec![String::new()]);
    }
    let count = read_u32(bytes, 0, "string table count")? as usize;
    let mut strings = Vec::with_capacity(count.min(1 << 20));
    let mut off = 4usize;
    for i in 0..count {
        let len = read_u32(bytes, off, "string length")? as usize;
        off += 4;
        let raw = get_bytes(bytes, off, len, "string data")?;
        off += len;
        // UTF-8 lossy: tolerate decoder-emitted non-UTF-8 labels.
        strings.push(String::from_utf8_lossy(raw).into_owned());
        if i == 0 && !strings[0].is_empty() {
            return Err("catb: string table id 0 is not the empty string".to_string());
        }
    }
    if strings.is_empty() {
        strings.push(String::new());
    }
    Ok(strings)
}

fn parse_meta(bytes: &[u8]) -> Result<CatbMeta, String> {
    let v: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| format!("catb: invalid meta JSON: {e}"))?;
    let decoder = v.get("decoder").cloned().unwrap_or(serde_json::Value::Null);
    let str_of = |obj: &serde_json::Value, key: &str| -> String {
        obj.get(key)
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string()
    };
    Ok(CatbMeta {
        schema_version: v.get("schema_version").and_then(|x| x.as_i64()),
        codec: str_of(&decoder, "codec"),
        contract: str_of(&decoder, "contract"),
        capture_level: str_of(&decoder, "capture_level"),
        parameter_sets: v
            .get("parameter_sets")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        has_frames_meta: v.get("frames_meta").map(|x| x.is_array()).unwrap_or(false),
    })
}

/// Parse one 80-byte FRAME record at `off` (§5).
fn parse_frame_record(bytes: &[u8], off: usize, strings: &[String]) -> Result<CatbFrame, String> {
    let rec = get_bytes(bytes, off, FRAME_RECORD_SIZE, "FRAME record")?;
    Ok(CatbFrame {
        index: read_i32(rec, 0, "frame.index")?,
        poc: read_i32(rec, 4, "frame.poc")?,
        frame_type: resolve_str(strings, read_i32(rec, 8, "frame.frame_type")?).to_string(),
        output: read_i32(rec, 12, "frame.output")? != 0,
        vps_id: opt_i32(read_i32(rec, 16, "frame.vps_id")?),
        sps_id: opt_i32(read_i32(rec, 20, "frame.sps_id")?),
        pps_id: opt_i32(read_i32(rec, 24, "frame.pps_id")?),
        slice_bits: opt_i64(read_i64(rec, 28, "frame.slice_bits")?),
        header_bits: opt_i64(read_i64(rec, 36, "frame.header_bits")?),
        data_bits: opt_i64(read_i64(rec, 44, "frame.data_bits")?),
        block_off: read_u64(rec, 52, "frame.block_off")?,
        block_n: read_i32(rec, 60, "frame.block_n")?,
        aux_off: read_u64(rec, 64, "frame.aux_off")?,
        aux_n: read_i64(rec, 72, "frame.aux_n")?,
    })
}

/// Parse one 156-byte BLOCK record at `off` (§6).
fn parse_block_record(bytes: &[u8], off: usize, strings: &[String]) -> Result<BsBlock, String> {
    let rec = get_bytes(bytes, off, BLOCK_RECORD_SIZE, "BLOCK record")?;
    let mv_flags = read_u32(rec, 104, "block.mv_flags")?;
    let pair = |base: usize, name: &str| -> Result<(i32, i32), String> {
        Ok((read_i32(rec, base, name)?, read_i32(rec, base + 4, name)?))
    };
    Ok(BsBlock {
        x: read_i32(rec, 0, "block.x")?.max(0) as u32,
        y: read_i32(rec, 4, "block.y")?.max(0) as u32,
        w: read_i32(rec, 8, "block.w")?.max(0) as u32,
        h: read_i32(rec, 12, "block.h")?.max(0) as u32,
        ctu_address: read_i64(rec, 16, "block.ctu_address")?,
        // §13: no sentinel — absent QP is stored as 0; surface as stored.
        qp: read_i32(rec, 24, "block.qp")?,
        partition: resolve_str(strings, read_i32(rec, 28, "block.partition")?).to_string(),
        prediction_mode: resolve_str(strings, read_i32(rec, 32, "block.prediction_mode")?)
            .to_string(),
        bits: read_i64(rec, 36, "block.bits")?,
        bit_offset: read_i64(rec, 44, "block.bit_offset")?,
        exactness_flags: read_i32(rec, 52, "block.exactness_flags")?,
        syntax_off: read_u64(rec, 56, "block.syntax_off")?,
        syntax_n: read_i32(rec, 64, "block.syntax_n")?,
        cabac_off: read_u64(rec, 68, "block.cabac_off")?,
        cabac_n: read_i32(rec, 76, "block.cabac_n")?,
        tx_off: read_u64(rec, 80, "block.tx_off")?,
        tx_n: read_i32(rec, 88, "block.tx_n")?,
        ref_off: read_u64(rec, 92, "block.ref_off")?,
        ref_n: read_i32(rec, 100, "block.ref_n")?,
        mv_flags,
        mvp: if mv_flags & MV_HAS_MVP != 0 {
            Some(pair(108, "block.mvp")?)
        } else {
            None
        },
        mvd: if mv_flags & MV_HAS_MVD != 0 {
            Some(pair(116, "block.mvd")?)
        } else {
            None
        },
        mv: if mv_flags & MV_HAS_MV != 0 {
            Some(pair(124, "block.mv")?)
        } else {
            None
        },
        reference: opt_str(strings, read_i32(rec, 132, "block.reference")?),
        reference_list: opt_str(strings, read_i32(rec, 136, "block.reference_list")?),
        reference_label: opt_str(strings, read_i32(rec, 140, "block.reference_label")?),
        reference_list_index: if mv_flags & MV_HAS_LIST_INDEX != 0 {
            Some(read_i32(rec, 144, "block.reference_list_index")?)
        } else {
            None
        },
        reference_poc: if mv_flags & MV_HAS_POC != 0 {
            Some(read_i32(rec, 148, "block.reference_poc")?)
        } else {
            None
        },
        reference_frame: if mv_flags & MV_HAS_FRAME != 0 {
            Some(read_i32(rec, 152, "block.reference_frame")?)
        } else {
            None
        },
        reference_long_term: if mv_flags & MV_HAS_LONG_TERM != 0 {
            Some(mv_flags & MV_LONG_TERM_VALUE != 0)
        } else {
            None
        },
    })
}

/// Parse one 68-byte REF record at `off` (§10).
fn parse_ref_record(bytes: &[u8], off: usize, strings: &[String]) -> Result<BsRef, String> {
    let rec = get_bytes(bytes, off, REF_RECORD_SIZE, "REF record")?;
    let ref_flags = read_u32(rec, 32, "ref.ref_flags")?;
    let pair = |base: usize, name: &str| -> Result<(i32, i32), String> {
        Ok((read_i32(rec, base, name)?, read_i32(rec, base + 4, name)?))
    };
    Ok(BsRef {
        list: opt_str(strings, read_i32(rec, 0, "ref.list")?),
        list_index: read_i32(rec, 4, "ref.list_index")?,
        reference_poc: read_i32(rec, 8, "ref.reference_poc")?,
        label: opt_str(strings, read_i32(rec, 12, "ref.label")?),
        pu_x: read_i32(rec, 16, "ref.pu_x")?,
        pu_y: read_i32(rec, 20, "ref.pu_y")?,
        pu_w: read_i32(rec, 24, "ref.pu_w")?,
        pu_h: read_i32(rec, 28, "ref.pu_h")?,
        ref_flags,
        mvp: if ref_flags & REF_HAS_MVP != 0 {
            Some(pair(36, "ref.mvp")?)
        } else {
            None
        },
        mvd: if ref_flags & REF_HAS_MVD != 0 {
            Some(pair(44, "ref.mvd")?)
        } else {
            None
        },
        mv: if ref_flags & REF_HAS_MV != 0 {
            Some(pair(52, "ref.mv")?)
        } else {
            None
        },
        reference_frame: if ref_flags & REF_HAS_FRAME != 0 {
            Some(read_i32(rec, 60, "ref.reference_frame")?)
        } else {
            None
        },
        long_term: if ref_flags & REF_HAS_LONG_TERM != 0 {
            Some(ref_flags & REF_LONG_TERM_VALUE != 0)
        } else {
            None
        },
        pu_part_index: if ref_flags & REF_HAS_PU_PART != 0 {
            Some(read_i32(rec, 64, "ref.pu_part_index")?)
        } else {
            None
        },
    })
}
