//! `.catb` v4 ("CATB0001") bitstream telemetry container reader.
//!
//! Implemented from the CC0 .catb v4 format specification
//! (codec-analyzer docs/catb-v4-format.md); no GPL code consulted.
//!
//! The container stores per-frame decoder telemetry with a 10-entry section
//! directory, an interned string table, a JSON meta blob, and fixed-size
//! little-endian records (FRAME 80 B, BLOCK 156 B, REF 68 B). The file is
//! mmap'd; FRAME records are parsed eagerly (small), BLOCK/REF records are
//! parsed lazily per frame/block, and SYNTAX (28 B), CABAC (28 B), and
//! TRANSFORM (104 B) records are parsed lazily per block (M4 / M-B).
//! Coefficient detail is resolved per-TU from the `coeffs` i32 section
//! (§11 indirect rule), and `frame_aux` (per-frame loop-filter / SAO JSON
//! blobs, §12) is parsed lazily per frame (M-E).

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
/// SYNTAX record size in bytes (§7).
pub const SYNTAX_RECORD_SIZE: usize = 28;
/// CABAC record size in bytes (§8).
pub const CABAC_RECORD_SIZE: usize = 28;
/// TRANSFORM (TX) record size in bytes (§9).
pub const TX_RECORD_SIZE: usize = 104;

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

// TX `coeff_flags` detail/presence bits (§9.1).
pub const COEFF_HAS_DETAIL: u32 = 1;
pub const COEFF_DETAIL_DROPPED: u32 = 2;
pub const COEFF_HAS_NONZERO: u32 = 4;
pub const COEFF_HAS_LAST_SIG_X: u32 = 8;
pub const COEFF_HAS_LAST_SIG_Y: u32 = 16;
pub const COEFF_HAS_ABS_SUM: u32 = 32;

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
    /// Structured `parameter_sets` (M-A Structure tab). Lenient: entries
    /// that are not objects are skipped; missing key → empty.
    pub parameter_set_infos: Vec<ParameterSetInfo>,
    /// Structured `frames_meta`, parallel to FRAME records (§4). Lenient:
    /// missing/mistyped elements decay to empty defaults, never an error.
    pub frames_meta: Vec<CatbFrameMeta>,
}

/// One `meta.parameter_sets` entry: `{kind, id, nal_index, fields{...}}`
/// (fields observed in the fixtures: VPS/SPS/PPS with plain scalar values).
#[derive(Debug, Clone, Default)]
pub struct ParameterSetInfo {
    /// "VPS" / "SPS" / "PPS" (AV1: sequence header kind string).
    pub kind: String,
    pub id: i64,
    pub nal_index: Option<i64>,
    /// Sorted (name, rendered value) pairs of the decoder-defined `fields`
    /// object (serde_json objects iterate key-sorted).
    pub fields: Vec<(String, String)>,
}

/// One `frames_meta[i].slice_headers[]` row. Observed fixture keys:
/// `nal_index`, `nal_unit_name`, `payload_bits`, `parsed_bits`, `data_bits`,
/// `slice_pic_parameter_set_id`, `slice_type`, `slice_type_label`,
/// `ref_list0` / `ref_list1` (only when present), `syntax` (rows with
/// name/value/coding/bit_offset/bits).
#[derive(Debug, Clone, Default)]
pub struct SliceHeaderInfo {
    /// `nal_unit_name`, e.g. "IDR_N_LP", "TRAIL_R". Empty if absent.
    pub nal_unit_name: String,
    /// `slice_type_label`, e.g. "I"/"P"/"B". Empty if absent.
    pub slice_type_label: String,
    pub nal_index: Option<i64>,
    /// Sorted (name, rendered value) pairs of every scalar top-level key
    /// (arrays/objects like ref lists and syntax are surfaced separately).
    pub fields: Vec<(String, String)>,
    /// Decoded slice-header syntax rows as (name, value, bits) in stored
    /// order (bits feed the Stats tab's "(slice)" aggregate rows; 0 when
    /// the decoder did not attribute per-element bits).
    pub syntax: Vec<(String, String, i64)>,
    pub ref_list0: Vec<RefListEntry>,
    pub ref_list1: Vec<RefListEntry>,
}

/// One reference-list entry: `{poc, long_term, label}` (observed keys).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RefListEntry {
    pub poc: i64,
    pub long_term: bool,
    /// e.g. "L0[0] POC 0". Empty if absent.
    pub label: String,
}

/// One `frames_meta[i].dpb[]` row — DPB state after the picture. Observed
/// keys: `poc`, `slot`, `used_for_reference`, `long_term`, `output_mark`,
/// `label`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DpbRow {
    pub poc: i64,
    pub slot: Option<i64>,
    /// `used_for_reference`; false = held for output only ("hold").
    pub used_for_reference: bool,
    pub long_term: bool,
    pub output_mark: bool,
    /// e.g. "slot[0] POC 0". Empty if absent.
    pub label: String,
}

/// Structured `frames_meta[i]` element (parallel to FRAME record `i`).
#[derive(Debug, Clone, Default)]
pub struct CatbFrameMeta {
    pub slice_headers: Vec<SliceHeaderInfo>,
    pub dpb: Vec<DpbRow>,
    /// Per-block missing-exactness label string, parallel to the frame's
    /// BLOCK records ("" = nothing missing).
    pub exactness_missing: Vec<String>,
    /// Per-block dropped telemetry row count, parallel to BLOCK records.
    pub block_dropped_rows: Vec<i64>,
    /// M-C: decoder-run stage image paths, sorted (key, path) pairs.
    /// Observed keys: `residual`, `prediction`, `recon_unfiltered`,
    /// `final_recon`; observed values: file names relative to the .catb
    /// directory (decoder-run writes both into the same workdir).
    pub stage_images: Vec<(String, String)>,
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

/// One SYNTAX record (28 bytes): a decoded syntax-element row (§7).
///
/// Only the fields the M4 visualization layers need are surfaced. Note that
/// per §7 the element's `value` is a **string id**, not a numeric field —
/// values are interned strings exactly as the decoder emitted them, so the
/// resolved string is kept and numeric interpretation is left to consumers
/// (see `bitstream_stats::extract_intra_modes`).
#[derive(Debug, Clone, PartialEq)]
pub struct SyntaxRow {
    /// Resolved element name, e.g. `"intra_luma_pred_mode"`; `""` if absent.
    pub name: String,
    /// Resolved value string, e.g. `"26"`; `""` if absent.
    pub value: String,
    /// Bits consumed by the element.
    pub bits: i64,
}

/// One TRANSFORM (TX) record (104 bytes) with presence bits and string ids
/// resolved (§9). Scalar fields follow `coeff_flags` (§9.1): a clear
/// presence bit means *absent*, never 0.
#[derive(Debug, Clone, PartialEq)]
pub struct TxRow {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    /// Transform tree depth.
    pub depth: i32,
    /// Resolved transform type label (e.g. "DCT-II",
    /// "AVC 4x4 transform"); "" if absent.
    pub tx_type: String,
    /// Coded-block flags: bit 0 = cbf_luma, bit 1 = cbf_cb, bit 2 = cbf_cr.
    pub cbf: u32,
    /// TU bit delta.
    pub bits: i64,
    /// Bit position of TU residual decode start.
    pub bit_offset: i64,
    /// Present iff `coeff_flags & COEFF_HAS_NONZERO`.
    pub nonzero_coeff_count: Option<i32>,
    /// Present iff `coeff_flags & COEFF_HAS_LAST_SIG_X`.
    pub last_sig_coeff_x: Option<i32>,
    /// Present iff `coeff_flags & COEFF_HAS_LAST_SIG_Y`.
    pub last_sig_coeff_y: Option<i32>,
    /// Present iff `coeff_flags & COEFF_HAS_ABS_SUM`.
    pub coeff_abs_sum: Option<i64>,
    /// Raw detail/presence bitmask (§9.1).
    pub coeff_flags: u32,
    /// Number of coefficient levels (valid iff [`TxRow::has_detail`]).
    pub coeff_count: i32,
    /// Coefficient group grid width (valid iff has_detail).
    pub coeff_group_width: i32,
    /// Coefficient group grid height (valid iff has_detail).
    pub coeff_group_height: i32,
    /// Number of group flags (valid iff has_detail).
    pub coeff_group_count: i32,
    /// Resolved component label, e.g. "Y" (valid iff has_detail).
    pub coeff_component: String,
    /// Resolved level-kind label, e.g. "decoded_transform_coeff_level"
    /// (valid iff has_detail).
    pub coeff_level_kind: String,
    /// Base element index into the `coeffs` section (valid iff has_detail).
    pub coeff_off: u64,
}

impl TxRow {
    /// Full coefficient detail (levels + scan positions + group flags) is
    /// present in the `coeffs` section (§9.1 `_COEFF_HAS_DETAIL`).
    pub fn has_detail(&self) -> bool {
        self.coeff_flags & COEFF_HAS_DETAIL != 0
    }

    /// The decoder captured the TU but intentionally dropped its
    /// coefficient arrays (§11, e.g. `capture_level` below full).
    pub fn detail_dropped(&self) -> bool {
        self.coeff_flags & COEFF_DETAIL_DROPPED != 0
    }

    pub fn cbf_luma(&self) -> bool {
        self.cbf & 1 != 0
    }

    pub fn cbf_cb(&self) -> bool {
        self.cbf & 2 != 0
    }

    pub fn cbf_cr(&self) -> bool {
        self.cbf & 4 != 0
    }

    /// Any coded-block flag set (cbf == 0 TUs render dashed in the TU layer).
    pub fn any_cbf(&self) -> bool {
        self.cbf & 0b111 != 0
    }
}

/// One `frame_aux.loop_filters[]` deblocking-edge row (§12, M-E). Fixture-
/// observed keys (hevc_* fixtures): `x/y/w/h` (a 1×4 or 4×1 px edge
/// segment), `orientation` ("vertical"/"horizontal"), `boundary_strength`
/// (0–2), `filter_strength` ("strong"/"weak"/"none"), `reason`
/// ("boundary_strength_zero"/"threshold"/""), `qp`, `beta`, `tc`, plus
/// `ctu_address/ctu_x/ctu_y/segment/no_p/no_q/pcm_or_bypass/
/// filter_slice_edges/beta_offset/tc_offset/telemetry_source` (not
/// surfaced). Lenient: missing/mistyped keys decay to defaults.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LoopFilterRow {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    /// `true` = vertical edge (the segment spans `y..y+h` at `x`).
    pub vertical: bool,
    pub boundary_strength: i32,
    /// "strong" / "weak" / "none" (fixture-observed values).
    pub filter_strength: String,
    /// Why the filter did not apply: "boundary_strength_zero" /
    /// "threshold" / "" (applied).
    pub reason: String,
    pub qp: i32,
}

/// One `frame_aux.sao[]` row (§12, M-E). Fixtures carry per-CTU rows whose
/// `type_*` are all `"not_applied"`; parsed for completeness, no overlay
/// consumes them yet.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SaoRow {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub type_y: String,
    pub type_cb: String,
    pub type_cr: String,
    pub merge_left: bool,
    pub merge_up: bool,
}

impl SaoRow {
    /// Any component carries an applied SAO type.
    pub fn applied(&self) -> bool {
        [&self.type_y, &self.type_cb, &self.type_cr]
            .iter()
            .any(|t| !t.is_empty() && *t != "not_applied")
    }
}

/// One frame's `frame_aux` blob, lazily parsed (§12).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FrameAux {
    pub loop_filters: Vec<LoopFilterRow>,
    pub sao: Vec<SaoRow>,
}

/// One CABAC record (28 bytes): a decoded bin row (§8).
#[derive(Debug, Clone, PartialEq)]
pub struct CabacRow {
    /// Resolved bin/context name, e.g. "split_cu_flag"; "" if absent.
    pub name: String,
    /// Context index.
    pub ctx: i32,
    /// Decoded bin value.
    pub bin: i32,
    /// Bit position.
    pub bit_offset: i64,
    /// Bits consumed.
    pub bits: i64,
}

/// A TU's coefficient detail resolved from the `coeffs` section (§11):
/// three consecutive i32 sub-arrays at `coeff_off`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CoeffDetail {
    /// Coefficient levels, `coeff_count` entries. Fixture observation
    /// (hevc_intra/hevc_bslice): stored in **raster position order** over
    /// the full w×h TU (`coeff_count == w·h` for luma).
    pub levels: Vec<i32>,
    /// Scan positions, parallel to `levels`. Fixture observation: the
    /// scan-order index of that raster position, −1 where insignificant.
    pub scan: Vec<i32>,
    /// Coefficient-group coded flags on the
    /// `coeff_group_width × coeff_group_height` grid.
    pub group_flags: Vec<i32>,
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

    /// Lazily parse a block's SYNTAX records (decoded syntax-element rows).
    ///
    /// Same OOM guard pattern as `blocks_for_frame` / `refs_for_block`
    /// (0.12.1 F1): `syntax_n` is clamped against the physical SYNTAX
    /// section size *before* any allocation.
    pub fn syntax_for_block(&self, block: &BsBlock) -> Result<Vec<SyntaxRow>, String> {
        let syntax_bytes = self.section_bytes(SEC_SYNTAX);
        let n = block.syntax_n.max(0) as usize;
        let max_records = syntax_bytes.len() / SYNTAX_RECORD_SIZE;
        if n > max_records {
            return Err(format!(
                "catb: block claims {n} syntax rows but the SYNTAX section \
                 only holds {max_records} records"
            ));
        }
        let mut out = Vec::with_capacity(n);
        for j in 0..n {
            let rec_idx = (block.syntax_off as usize)
                .checked_add(j)
                .ok_or_else(|| "catb: syntax record index overflows".to_string())?;
            let byte_off = rec_idx
                .checked_mul(SYNTAX_RECORD_SIZE)
                .ok_or_else(|| "catb: syntax byte offset overflows".to_string())?;
            out.push(parse_syntax_record(syntax_bytes, byte_off, &self.strings)?);
        }
        Ok(out)
    }

    /// Lazily parse a block's TRANSFORM records (per-TU rows, M-B).
    ///
    /// Same OOM guard pattern as `blocks_for_frame` / `syntax_for_block`:
    /// `tx_n` is clamped against the physical TRANSFORMS section size
    /// *before* any allocation.
    pub fn tx_for_block(&self, block: &BsBlock) -> Result<Vec<TxRow>, String> {
        let tx_bytes = self.section_bytes(SEC_TRANSFORMS);
        let n = block.tx_n.max(0) as usize;
        let max_records = tx_bytes.len() / TX_RECORD_SIZE;
        if n > max_records {
            return Err(format!(
                "catb: block claims {n} transforms but the TRANSFORMS \
                 section only holds {max_records} records"
            ));
        }
        let mut out = Vec::with_capacity(n);
        for j in 0..n {
            let rec_idx = (block.tx_off as usize)
                .checked_add(j)
                .ok_or_else(|| "catb: tx record index overflows".to_string())?;
            let byte_off = rec_idx
                .checked_mul(TX_RECORD_SIZE)
                .ok_or_else(|| "catb: tx byte offset overflows".to_string())?;
            out.push(parse_tx_record(tx_bytes, byte_off, &self.strings)?);
        }
        Ok(out)
    }

    /// Lazily parse a block's CABAC records (decoded bin rows, M-B).
    pub fn cabac_for_block(&self, block: &BsBlock) -> Result<Vec<CabacRow>, String> {
        let cabac_bytes = self.section_bytes(SEC_CABAC);
        let n = block.cabac_n.max(0) as usize;
        // Same OOM guard: never allocate more record slots than the CABAC
        // section can physically contain.
        let max_records = cabac_bytes.len() / CABAC_RECORD_SIZE;
        if n > max_records {
            return Err(format!(
                "catb: block claims {n} cabac bins but the CABAC section \
                 only holds {max_records} records"
            ));
        }
        let mut out = Vec::with_capacity(n);
        for j in 0..n {
            let rec_idx = (block.cabac_off as usize)
                .checked_add(j)
                .ok_or_else(|| "catb: cabac record index overflows".to_string())?;
            let byte_off = rec_idx
                .checked_mul(CABAC_RECORD_SIZE)
                .ok_or_else(|| "catb: cabac byte offset overflows".to_string())?;
            out.push(parse_cabac_record(cabac_bytes, byte_off, &self.strings)?);
        }
        Ok(out)
    }

    /// Resolve a TU's coefficient detail from the `coeffs` section (§11
    /// indirect rule): `levels[coeff_count] + scan[coeff_count] +
    /// group_flags[coeff_group_count]` i32 elements at `coeff_off`.
    ///
    /// Returns `Ok(None)` when the TU carries no detail
    /// (`_COEFF_HAS_DETAIL` clear — including the detail-dropped case).
    /// OOM guard: the requested element span is bounds-checked against the
    /// physical `coeffs` section size *before* any allocation, so a hostile
    /// `coeff_count`/`coeff_group_count` errors instead of aborting.
    pub fn coeffs_for_tx(&self, tx: &TxRow) -> Result<Option<CoeffDetail>, String> {
        if !tx.has_detail() {
            return Ok(None);
        }
        let bytes = self.section_bytes(SEC_COEFFS);
        let total_elems = bytes.len() / 4;
        let cc = usize::try_from(tx.coeff_count.max(0)).unwrap_or(0);
        let gc = usize::try_from(tx.coeff_group_count.max(0)).unwrap_or(0);
        let span = cc
            .checked_mul(2)
            .and_then(|v| v.checked_add(gc))
            .ok_or_else(|| "catb: coeff span overflows".to_string())?;
        let base = usize::try_from(tx.coeff_off)
            .map_err(|_| "catb: coeff_off overflows".to_string())?;
        let end = base
            .checked_add(span)
            .ok_or_else(|| "catb: coeff range overflows".to_string())?;
        if end > total_elems {
            return Err(format!(
                "catb: TU claims {span} coeff elements at {base} but the \
                 coeffs section only holds {total_elems}"
            ));
        }
        let read_i32s = |start: usize, n: usize| -> Vec<i32> {
            (0..n)
                .map(|k| {
                    let off = (start + k) * 4;
                    i32::from_le_bytes([
                        bytes[off],
                        bytes[off + 1],
                        bytes[off + 2],
                        bytes[off + 3],
                    ])
                })
                .collect()
        };
        Ok(Some(CoeffDetail {
            levels: read_i32s(base, cc),
            scan: read_i32s(base + cc, cc),
            group_flags: read_i32s(base + 2 * cc, gc),
        }))
    }

    /// Lazily parse frame `frame_idx`'s `frame_aux` blob (§12, M-E): the
    /// per-frame compact JSON `{"loop_filters": [...], "sao": [...]}`
    /// addressed by the FRAME record's `(aux_off, aux_n)`.
    ///
    /// `aux_n <= 0` (spec: "no aux") and an out-of-section range both yield
    /// an empty [`FrameAux`]; truncated/invalid JSON is an `Err` (never a
    /// panic); missing/mistyped rows inside valid JSON decay to defaults.
    /// Callers cache the result (rows outnumber blocks 12–15×; the
    /// `BitstreamFile` layer keeps a 1-frame LRU).
    pub fn frame_aux_for_frame(&self, frame_idx: usize) -> Result<FrameAux, String> {
        let frame = self
            .frames
            .get(frame_idx)
            .ok_or_else(|| format!("catb: frame index {frame_idx} out of range"))?;
        if frame.aux_n <= 0 {
            return Ok(FrameAux::default());
        }
        let aux_bytes = self.section_bytes(SEC_FRAME_AUX);
        let start = usize::try_from(frame.aux_off)
            .map_err(|_| "catb: aux_off overflows".to_string())?;
        let len = usize::try_from(frame.aux_n).unwrap_or(0);
        let end = start
            .checked_add(len)
            .ok_or_else(|| "catb: aux range overflows".to_string())?;
        if end > aux_bytes.len() {
            return Err(format!(
                "catb: frame {frame_idx} aux range {start}+{len} exceeds the \
                 frame_aux section ({} bytes)",
                aux_bytes.len()
            ));
        }
        let v: serde_json::Value = serde_json::from_slice(&aux_bytes[start..end])
            .map_err(|e| format!("catb: frame {frame_idx} aux JSON invalid: {e}"))?;
        Ok(parse_frame_aux(&v))
    }

    /// Lightweight per-block TX aggregate for the CoeffEnergy /
    /// NonzeroCoeffs fills (M-B): `(Σ coeff_abs_sum, Σ nonzero_coeff_count)`
    /// over the block's TX records, honouring the §9.1 presence bits
    /// (absent scalars contribute 0). Reads only the needed fields — no
    /// string resolution, no `TxRow` allocation.
    pub fn tx_aggregate_for_block(&self, block: &BsBlock) -> Result<(i64, i64), String> {
        let tx_bytes = self.section_bytes(SEC_TRANSFORMS);
        let n = block.tx_n.max(0) as usize;
        let max_records = tx_bytes.len() / TX_RECORD_SIZE;
        if n > max_records {
            return Err(format!(
                "catb: block claims {n} transforms but the TRANSFORMS \
                 section only holds {max_records} records"
            ));
        }
        let mut abs_sum = 0i64;
        let mut nonzero = 0i64;
        for j in 0..n {
            let rec_idx = (block.tx_off as usize)
                .checked_add(j)
                .ok_or_else(|| "catb: tx record index overflows".to_string())?;
            let byte_off = rec_idx
                .checked_mul(TX_RECORD_SIZE)
                .ok_or_else(|| "catb: tx byte offset overflows".to_string())?;
            let rec = get_bytes(tx_bytes, byte_off, TX_RECORD_SIZE, "TX record")?;
            let flags = read_u32(rec, 68, "tx.coeff_flags")?;
            if flags & COEFF_HAS_ABS_SUM != 0 {
                abs_sum = abs_sum.saturating_add(read_i64(rec, 60, "tx.coeff_abs_sum")?.max(0));
            }
            if flags & COEFF_HAS_NONZERO != 0 {
                nonzero =
                    nonzero.saturating_add(i64::from(read_i32(rec, 44, "tx.nonzero")?.max(0)));
            }
        }
        Ok((abs_sum, nonzero))
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
    let parameter_sets = v
        .get("parameter_sets")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let parameter_set_infos = parse_parameter_sets(&parameter_sets);
    let frames_meta_raw = v.get("frames_meta");
    let has_frames_meta = frames_meta_raw.map(|x| x.is_array()).unwrap_or(false);
    let frames_meta = frames_meta_raw
        .and_then(|x| x.as_array())
        .map(|arr| arr.iter().map(parse_frame_meta).collect())
        .unwrap_or_default();
    Ok(CatbMeta {
        schema_version: v.get("schema_version").and_then(|x| x.as_i64()),
        codec: str_of(&decoder, "codec"),
        contract: str_of(&decoder, "contract"),
        capture_level: str_of(&decoder, "capture_level"),
        parameter_sets,
        has_frames_meta,
        parameter_set_infos,
        frames_meta,
    })
}

/// Render a scalar JSON value for the (name, value) field grids. Non-scalar
/// values yield `None` (they are surfaced through dedicated structures).
fn scalar_text(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::Null => Some("null".to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// Sorted (name, value) pairs of an object's scalar members. serde_json
/// objects iterate key-sorted (BTreeMap), so the result is name-ordered.
fn scalar_fields(obj: &serde_json::Value) -> Vec<(String, String)> {
    obj.as_object()
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| scalar_text(v).map(|t| (k.clone(), t)))
                .collect()
        })
        .unwrap_or_default()
}

/// Lenient `meta.parameter_sets` → structured list (non-objects skipped).
fn parse_parameter_sets(sets: &serde_json::Value) -> Vec<ParameterSetInfo> {
    let Some(arr) = sets.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .filter(|s| s.is_object())
        .map(|s| ParameterSetInfo {
            kind: s
                .get("kind")
                .and_then(|k| k.as_str())
                .unwrap_or("")
                .to_string(),
            id: s.get("id").and_then(|i| i.as_i64()).unwrap_or(0),
            nal_index: s.get("nal_index").and_then(|i| i.as_i64()),
            fields: s
                .get("fields")
                .map(scalar_fields)
                .unwrap_or_default(),
        })
        .collect()
}

/// Lenient reference-list array (`ref_list0` / `ref_list1`) parser.
fn parse_ref_list(v: Option<&serde_json::Value>) -> Vec<RefListEntry> {
    v.and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .map(|e| RefListEntry {
                    poc: e.get("poc").and_then(|p| p.as_i64()).unwrap_or(0),
                    long_term: e
                        .get("long_term")
                        .and_then(|l| l.as_bool())
                        .unwrap_or(false),
                    label: e
                        .get("label")
                        .and_then(|l| l.as_str())
                        .unwrap_or("")
                        .to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Lenient `frame_aux` blob parser (M-E §12): rows that are not objects are
/// skipped; missing keys decay to defaults.
fn parse_frame_aux(v: &serde_json::Value) -> FrameAux {
    let i32_of = |o: &serde_json::Value, k: &str| -> i32 {
        o.get(k).and_then(|x| x.as_i64()).unwrap_or(0) as i32
    };
    let str_of = |o: &serde_json::Value, k: &str| -> String {
        o.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string()
    };
    let bool_of = |o: &serde_json::Value, k: &str| -> bool {
        o.get(k).and_then(|x| x.as_bool()).unwrap_or(false)
    };
    let loop_filters = v
        .get("loop_filters")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|r| r.is_object())
                .map(|r| LoopFilterRow {
                    x: i32_of(r, "x"),
                    y: i32_of(r, "y"),
                    w: i32_of(r, "w"),
                    h: i32_of(r, "h"),
                    vertical: str_of(r, "orientation") == "vertical",
                    boundary_strength: i32_of(r, "boundary_strength"),
                    filter_strength: str_of(r, "filter_strength"),
                    reason: str_of(r, "reason"),
                    qp: i32_of(r, "qp"),
                })
                .collect()
        })
        .unwrap_or_default();
    let sao = v
        .get("sao")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|r| r.is_object())
                .map(|r| SaoRow {
                    x: i32_of(r, "x"),
                    y: i32_of(r, "y"),
                    w: i32_of(r, "w"),
                    h: i32_of(r, "h"),
                    type_y: str_of(r, "type_y"),
                    type_cb: str_of(r, "type_cb"),
                    type_cr: str_of(r, "type_cr"),
                    merge_left: bool_of(r, "merge_left"),
                    merge_up: bool_of(r, "merge_up"),
                })
                .collect()
        })
        .unwrap_or_default();
    FrameAux { loop_filters, sao }
}

/// Lenient `frames_meta[i]` element parser (any missing/mistyped member
/// decays to its empty default).
fn parse_frame_meta(v: &serde_json::Value) -> CatbFrameMeta {
    let slice_headers = v
        .get("slice_headers")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .map(|sh| SliceHeaderInfo {
                    nal_unit_name: sh
                        .get("nal_unit_name")
                        .and_then(|s| s.as_str())
                        .unwrap_or("")
                        .to_string(),
                    slice_type_label: sh
                        .get("slice_type_label")
                        .and_then(|s| s.as_str())
                        .unwrap_or("")
                        .to_string(),
                    nal_index: sh.get("nal_index").and_then(|i| i.as_i64()),
                    fields: scalar_fields(sh),
                    syntax: sh
                        .get("syntax")
                        .and_then(|x| x.as_array())
                        .map(|rows| {
                            rows.iter()
                                .filter_map(|r| {
                                    let name = r.get("name")?.as_str()?.to_string();
                                    let value = r
                                        .get("value")
                                        .and_then(scalar_text)
                                        .unwrap_or_default();
                                    let bits = r
                                        .get("bits")
                                        .and_then(|b| b.as_i64())
                                        .unwrap_or(0);
                                    Some((name, value, bits))
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                    ref_list0: parse_ref_list(sh.get("ref_list0")),
                    ref_list1: parse_ref_list(sh.get("ref_list1")),
                })
                .collect()
        })
        .unwrap_or_default();
    let dpb = v
        .get("dpb")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .map(|r| DpbRow {
                    poc: r.get("poc").and_then(|p| p.as_i64()).unwrap_or(0),
                    slot: r.get("slot").and_then(|s| s.as_i64()),
                    used_for_reference: r
                        .get("used_for_reference")
                        .and_then(|b| b.as_bool())
                        .unwrap_or(false),
                    long_term: r
                        .get("long_term")
                        .and_then(|b| b.as_bool())
                        .unwrap_or(false),
                    output_mark: r
                        .get("output_mark")
                        .and_then(|b| b.as_bool())
                        .unwrap_or(false),
                    label: r
                        .get("label")
                        .and_then(|l| l.as_str())
                        .unwrap_or("")
                        .to_string(),
                })
                .collect()
        })
        .unwrap_or_default();
    let exactness_missing = v
        .get("exactness_missing")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .map(|e| e.as_str().unwrap_or("").to_string())
                .collect()
        })
        .unwrap_or_default();
    let block_dropped_rows = v
        .get("block_dropped_rows")
        .and_then(|x| x.as_array())
        .map(|arr| arr.iter().map(|e| e.as_i64().unwrap_or(0)).collect())
        .unwrap_or_default();
    // M-C: stage image map — scalar (key, path) pairs, key-sorted like every
    // other field grid (serde_json objects iterate key-sorted).
    let stage_images = v
        .get("stage_images")
        .map(scalar_fields)
        .unwrap_or_default();
    CatbFrameMeta {
        slice_headers,
        dpb,
        exactness_missing,
        block_dropped_rows,
        stage_images,
    }
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

/// Parse one 28-byte SYNTAX record at `off` (§7): `<3iqq` — name(string id),
/// value(string id), coding(string id), bit_offset(i64), bits(i64). Only
/// name/value/bits are surfaced (coding and bit_offset are unused by M4).
fn parse_syntax_record(bytes: &[u8], off: usize, strings: &[String]) -> Result<SyntaxRow, String> {
    let rec = get_bytes(bytes, off, SYNTAX_RECORD_SIZE, "SYNTAX record")?;
    Ok(SyntaxRow {
        name: resolve_str(strings, read_i32(rec, 0, "syntax.name")?).to_string(),
        value: resolve_str(strings, read_i32(rec, 4, "syntax.value")?).to_string(),
        bits: read_i64(rec, 20, "syntax.bits")?,
    })
}

/// Parse one 104-byte TRANSFORM record at `off` (§9): `<6iIqq4iqI6iQ` —
/// x/y/w/h/depth(i32), type(string id), cbf(u32), bits/bit_offset(i64),
/// nonzero/last_sig_x/last_sig_y/_pad(i32), coeff_abs_sum(i64),
/// coeff_flags(u32), coeff_count/group_w/group_h/group_count/component/
/// level_kind(i32), coeff_off(u64). Presence bits per §9.1.
fn parse_tx_record(bytes: &[u8], off: usize, strings: &[String]) -> Result<TxRow, String> {
    let rec = get_bytes(bytes, off, TX_RECORD_SIZE, "TX record")?;
    let coeff_flags = read_u32(rec, 68, "tx.coeff_flags")?;
    let has = |bit: u32| coeff_flags & bit != 0;
    let has_detail = has(COEFF_HAS_DETAIL);
    Ok(TxRow {
        x: read_i32(rec, 0, "tx.x")?,
        y: read_i32(rec, 4, "tx.y")?,
        w: read_i32(rec, 8, "tx.w")?,
        h: read_i32(rec, 12, "tx.h")?,
        depth: read_i32(rec, 16, "tx.depth")?,
        tx_type: resolve_str(strings, read_i32(rec, 20, "tx.type")?).to_string(),
        cbf: read_u32(rec, 24, "tx.cbf")?,
        bits: read_i64(rec, 28, "tx.bits")?,
        bit_offset: read_i64(rec, 36, "tx.bit_offset")?,
        nonzero_coeff_count: has(COEFF_HAS_NONZERO)
            .then(|| read_i32(rec, 44, "tx.nonzero"))
            .transpose()?,
        last_sig_coeff_x: has(COEFF_HAS_LAST_SIG_X)
            .then(|| read_i32(rec, 48, "tx.last_sig_x"))
            .transpose()?,
        last_sig_coeff_y: has(COEFF_HAS_LAST_SIG_Y)
            .then(|| read_i32(rec, 52, "tx.last_sig_y"))
            .transpose()?,
        // offset 56 is the explicit `_pad` field (always 0) — skipped.
        coeff_abs_sum: has(COEFF_HAS_ABS_SUM)
            .then(|| read_i64(rec, 60, "tx.coeff_abs_sum"))
            .transpose()?,
        coeff_flags,
        coeff_count: read_i32(rec, 72, "tx.coeff_count")?,
        coeff_group_width: read_i32(rec, 76, "tx.coeff_group_width")?,
        coeff_group_height: read_i32(rec, 80, "tx.coeff_group_height")?,
        coeff_group_count: read_i32(rec, 84, "tx.coeff_group_count")?,
        // §11: component / level_kind are meaningless without detail.
        coeff_component: if has_detail {
            resolve_str(strings, read_i32(rec, 88, "tx.coeff_component")?).to_string()
        } else {
            String::new()
        },
        coeff_level_kind: if has_detail {
            resolve_str(strings, read_i32(rec, 92, "tx.coeff_level_kind")?).to_string()
        } else {
            String::new()
        },
        coeff_off: read_u64(rec, 96, "tx.coeff_off")?,
    })
}

/// Parse one 28-byte CABAC record at `off` (§8): `<3iqq` — name(string id),
/// ctx(i32), bin(i32), bit_offset(i64), bits(i64).
fn parse_cabac_record(bytes: &[u8], off: usize, strings: &[String]) -> Result<CabacRow, String> {
    let rec = get_bytes(bytes, off, CABAC_RECORD_SIZE, "CABAC record")?;
    Ok(CabacRow {
        name: resolve_str(strings, read_i32(rec, 0, "cabac.name")?).to_string(),
        ctx: read_i32(rec, 4, "cabac.ctx")?,
        bin: read_i32(rec, 8, "cabac.bin")?,
        bit_offset: read_i64(rec, 12, "cabac.bit_offset")?,
        bits: read_i64(rec, 20, "cabac.bits")?,
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
