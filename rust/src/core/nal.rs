//! Annex-B NAL unit scanner for raw H.264 / H.265 elementary streams (M-E).
//!
//! Implemented from public standards knowledge only (ITU-T H.264 §7.3.1 /
//! Table 7-1, ITU-T H.265 §7.3.1.1-7.3.1.2 / Table 7-1); no GPL code
//! consulted. This module is `.catb`-independent: it reads the *original*
//! bitstream file (mmap, single O(n) pass, no panics on malformed input).
//!
//! Start-code convention: a unit begins at `00 00 01` (3-byte) or
//! `00 00 00 01` (4-byte, when the 3-byte pattern is not matched first at
//! the same scan position). `size` counts the NAL header + payload bytes
//! (start code excluded), i.e. `next_start − (offset + start_code_len)`.

use std::path::{Path, PathBuf};

use memmap2::Mmap;

/// Which codec's `nal_unit_header` the units were parsed with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NalCodec {
    /// H.265/HEVC: 2-byte header — forbidden_zero_bit(1) +
    /// nal_unit_type(6) + nuh_layer_id(6) + nuh_temporal_id_plus1(3).
    Hevc,
    /// H.264/AVC: 1-byte header — forbidden_zero_bit(1) +
    /// nal_ref_idc(2) + nal_unit_type(5).
    Avc,
}

impl NalCodec {
    pub fn label(self) -> &'static str {
        match self {
            NalCodec::Hevc => "HEVC",
            NalCodec::Avc => "AVC",
        }
    }
}

/// One scanned NAL unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NalUnit {
    /// Byte offset of the start code in the file.
    pub offset: u64,
    /// Start-code length: 3 or 4.
    pub start_code_len: u8,
    /// NAL header + payload bytes (start code excluded).
    pub size: u64,
    /// Raw `nal_unit_type` (HEVC 6-bit / AVC 5-bit).
    pub nal_type: u8,
    /// HEVC `nuh_temporal_id_plus1 − 1`; 0 for AVC.
    pub temporal_id: u8,
    /// HEVC `nuh_layer_id`; 0 for AVC.
    pub layer_id: u8,
    /// AVC `nal_ref_idc`; 0 for HEVC.
    pub nal_ref_idc: u8,
    /// `forbidden_zero_bit` was set (malformed / misdetected codec).
    pub forbidden: bool,
    /// VCL (slice-carrying) unit: HEVC type ≤ 31, AVC type 1–5.
    pub is_vcl: bool,
    /// Estimated access-unit start. HEVC: VCL unit whose
    /// `first_slice_segment_in_pic_flag` (first payload bit after the
    /// header) is 1. AVC: VCL unit whose first slice-header bit is 1
    /// (`first_mb_in_slice` ue(v) leading '1' ⇒ value 0) — a heuristic;
    /// full §7.4.1.2.4 AU detection needs parameter-set state.
    pub au_start: bool,
    /// Estimated access-unit index (0-based, stream order). Non-VCL units
    /// that per spec start a new AU (AUD / parameter sets / prefix SEI)
    /// are attributed to the *following* AU; suffix SEI / EOS / EOB /
    /// filler stay with the current one.
    pub au_id: u32,
}

/// Scan result: codec + units + the mapped file bytes (HEX view source).
pub struct NalScan {
    pub codec: NalCodec,
    pub units: Vec<NalUnit>,
    pub file_size: u64,
    /// The scan hit [`MAX_UNITS`] and stopped early (hostile/garbage input).
    pub truncated: bool,
    /// Backing mmap; `None` for in-memory scans (tests).
    mmap: Option<Mmap>,
    /// In-memory fallback (tests / tiny buffers).
    buf: Vec<u8>,
}

impl std::fmt::Debug for NalScan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NalScan")
            .field("codec", &self.codec)
            .field("units", &self.units.len())
            .field("file_size", &self.file_size)
            .field("truncated", &self.truncated)
            .finish()
    }
}

/// Unit-count ceiling: a garbage file of repeating `00 00 01` would other-
/// wise produce one unit per 3 bytes (hundreds of millions on large files).
pub const MAX_UNITS: usize = 2_000_000;

/// Extensions treated as raw Annex-B bitstreams (source inference + codec
/// hint). `bin` is accepted with codec auto-detection.
pub const BITSTREAM_EXTS: [&str; 7] = ["h265", "265", "hevc", "h264", "264", "avc", "bin"];

impl NalScan {
    /// mmap + codec detection + single-pass scan. Never panics; malformed
    /// or non-Annex-B content simply yields few/zero units.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|e| format!("nal: cannot open {}: {e}", path.display()))?;
        // SAFETY: read-only mapping of a regular file; all accesses are
        // bounds-checked slices (same contract as the .catb reader).
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("nal: mmap failed for {}: {e}", path.display()))?;
        let hint = codec_hint_from_ext(path);
        let codec = detect_codec(&mmap, hint);
        let (units, truncated) = scan_annexb(&mmap, codec);
        if units.is_empty() {
            return Err(format!(
                "nal: no Annex-B start codes found in {} — not a raw \
                 .h264/.h265 elementary stream?",
                path.display()
            ));
        }
        Ok(Self {
            codec,
            units,
            file_size: mmap.len() as u64,
            truncated,
            mmap: Some(mmap),
            buf: Vec::new(),
        })
    }

    /// In-memory scan with an explicit codec (tests / synthetic streams).
    pub fn from_bytes(bytes: Vec<u8>, codec: NalCodec) -> Self {
        let (units, truncated) = scan_annexb(&bytes, codec);
        Self {
            codec,
            units,
            file_size: bytes.len() as u64,
            truncated,
            mmap: None,
            buf: bytes,
        }
    }

    /// The scanned bytes (HEX view source).
    pub fn data(&self) -> &[u8] {
        match &self.mmap {
            Some(m) => m,
            None => &self.buf,
        }
    }

    /// Bytes of unit `idx` **including its start code** (what the HEX view
    /// shows). Empty slice on out-of-range.
    pub fn unit_bytes(&self, idx: usize) -> &[u8] {
        let Some(u) = self.units.get(idx) else {
            return &[];
        };
        let start = u.offset as usize;
        let end = start
            .saturating_add(u.start_code_len as usize)
            .saturating_add(u.size as usize);
        self.data().get(start..end.min(self.data().len())).unwrap_or(&[])
    }

    /// Number of VCL units (decode-order picture count upper bound).
    pub fn vcl_count(&self) -> usize {
        self.units.iter().filter(|u| u.is_vcl).count()
    }
}

/// Codec hint from a file extension (`None` = score both parses).
pub fn codec_hint_from_ext(path: &Path) -> Option<NalCodec> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "h265" | "265" | "hevc" => Some(NalCodec::Hevc),
        "h264" | "264" | "avc" => Some(NalCodec::Avc),
        _ => None,
    }
}

/// Header-validity score: parse the stream both ways and pick the codec
/// with fewer violations (forbidden bit, reserved/unspecified types, HEVC
/// `nuh_temporal_id_plus1 == 0`). Ties fall back to the extension hint,
/// then HEVC (arbitrary but stable — the UI shows the chosen codec).
pub fn detect_codec(data: &[u8], hint: Option<NalCodec>) -> NalCodec {
    let hevc = violation_score(data, NalCodec::Hevc);
    let avc = violation_score(data, NalCodec::Avc);
    match hevc.cmp(&avc) {
        std::cmp::Ordering::Less => NalCodec::Hevc,
        std::cmp::Ordering::Greater => NalCodec::Avc,
        std::cmp::Ordering::Equal => hint.unwrap_or(NalCodec::Hevc),
    }
}

/// Violation score over the first units (bounded — detection never needs
/// the whole file).
fn violation_score(data: &[u8], codec: NalCodec) -> u32 {
    const SAMPLE_UNITS: usize = 64;
    let mut score = 0u32;
    for (off, scl) in StartCodeIter::new(data).take(SAMPLE_UNITS) {
        let p = &data[off + scl..];
        match codec {
            NalCodec::Hevc => {
                if p.len() < 2 {
                    score += 2;
                    continue;
                }
                if p[0] & 0x80 != 0 {
                    score += 2; // forbidden_zero_bit
                }
                let ty = (p[0] >> 1) & 0x3f;
                if ty >= 48 {
                    score += 2; // unspecified
                } else if (41..=47).contains(&ty) {
                    score += 1; // reserved non-VCL
                }
                if p[1] & 0x07 == 0 {
                    score += 2; // nuh_temporal_id_plus1 must be ≥ 1
                }
            }
            NalCodec::Avc => {
                if p.is_empty() {
                    score += 2;
                    continue;
                }
                if p[0] & 0x80 != 0 {
                    score += 2; // forbidden_zero_bit
                }
                let ty = p[0] & 0x1f;
                if ty == 0 || ty >= 24 {
                    score += 2; // unspecified
                } else if (17..=18).contains(&ty) || (22..=23).contains(&ty) {
                    score += 1; // reserved
                }
            }
        }
    }
    score
}

/// Iterator over `(start_code_offset, start_code_len)` pairs. The 3-byte
/// pattern is matched first at each position, so `00 00 00 01` yields a
/// 4-byte code (the leading zero is consumed as part of it).
struct StartCodeIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> StartCodeIter<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl Iterator for StartCodeIter<'_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let d = self.data;
        let mut i = self.pos;
        while i + 3 <= d.len() {
            if d[i + 2] > 1 {
                // No start code can begin at i, i+1 or i+2: this byte would
                // be code byte 2 (must be 1), 1 (0) or 0 (0) respectively.
                i += 3;
            } else if d[i] == 0 && d[i + 1] == 0 {
                if d[i + 2] == 1 {
                    self.pos = i + 3;
                    return Some((i, 3));
                }
                // d[i+2] == 0: a 4-byte code if the next byte is 1;
                // otherwise a longer zero run — only i is ruled out.
                if i + 4 <= d.len() && d[i + 3] == 1 {
                    self.pos = i + 4;
                    return Some((i, 4));
                }
                i += 1;
            } else {
                i += 1;
            }
        }
        self.pos = d.len();
        None
    }
}

/// Single-pass Annex-B scan. Returns the units and whether [`MAX_UNITS`]
/// truncated the scan.
pub fn scan_annexb(data: &[u8], codec: NalCodec) -> (Vec<NalUnit>, bool) {
    let starts: Vec<(usize, usize)> = {
        let mut v = Vec::new();
        for sc in StartCodeIter::new(data) {
            v.push(sc);
            if v.len() >= MAX_UNITS {
                break;
            }
        }
        v
    };
    let truncated = starts.len() >= MAX_UNITS;
    let mut units = Vec::with_capacity(starts.len());
    // au = index of the AU the *last VCL unit* belonged to.
    let mut au: i64 = -1;
    for (k, &(off, scl)) in starts.iter().enumerate() {
        let end = starts.get(k + 1).map(|&(o, _)| o).unwrap_or(data.len());
        let payload = &data[(off + scl).min(end)..end];
        let mut u = parse_unit(codec, payload);
        u.offset = off as u64;
        u.start_code_len = scl as u8;
        u.size = payload.len() as u64;
        // AU attribution (documented approximation, header-only).
        if u.is_vcl {
            if u.au_start || au < 0 {
                au += 1;
            }
            u.au_id = au.max(0) as u32;
        } else if starts_new_au_nonvcl(codec, u.nal_type) {
            u.au_id = (au + 1).max(0) as u32;
        } else {
            u.au_id = au.max(0) as u32;
        }
        units.push(u);
    }
    (units, truncated)
}

/// Parse one unit's header/flags from its payload (header at payload[0..]).
fn parse_unit(codec: NalCodec, p: &[u8]) -> NalUnit {
    let mut u = NalUnit {
        offset: 0,
        start_code_len: 0,
        size: 0,
        nal_type: 0,
        temporal_id: 0,
        layer_id: 0,
        nal_ref_idc: 0,
        forbidden: false,
        is_vcl: false,
        au_start: false,
        au_id: 0,
    };
    match codec {
        NalCodec::Hevc => {
            if p.len() < 2 {
                return u; // short/truncated unit — all-zero fields
            }
            u.forbidden = p[0] & 0x80 != 0;
            u.nal_type = (p[0] >> 1) & 0x3f;
            u.layer_id = ((p[0] & 1) << 5) | (p[1] >> 3);
            u.temporal_id = (p[1] & 0x07).saturating_sub(1);
            u.is_vcl = u.nal_type <= 31;
            // first_slice_segment_in_pic_flag: first bit after the 2-byte
            // header (H.265 §7.3.6.1) — EPB cannot occur this early.
            u.au_start = u.is_vcl && p.get(2).is_some_and(|b| b & 0x80 != 0);
        }
        NalCodec::Avc => {
            if p.is_empty() {
                return u;
            }
            u.forbidden = p[0] & 0x80 != 0;
            u.nal_ref_idc = (p[0] >> 5) & 0x03;
            u.nal_type = p[0] & 0x1f;
            u.is_vcl = (1..=5).contains(&u.nal_type);
            // first_mb_in_slice ue(v): a leading '1' bit encodes value 0
            // (H.264 §7.3.3) ⇒ the slice starts the picture.
            u.au_start = u.is_vcl && p.get(1).is_some_and(|b| b & 0x80 != 0);
        }
    }
    u
}

/// Non-VCL types that start a new access unit when they follow a VCL unit
/// (H.265 §7.4.2.4.4 / H.264 §7.4.1.2.3 first-of-AU sets).
fn starts_new_au_nonvcl(codec: NalCodec, ty: u8) -> bool {
    match codec {
        // AUD, VPS, SPS, PPS, prefix SEI (and reserved 41–44 per spec).
        NalCodec::Hevc => matches!(ty, 32..=35 | 39 | 41..=44),
        // AUD, SPS, PPS, SEI, SPS-ext, prefix NAL, subset SPS, DPS.
        NalCodec::Avc => matches!(ty, 6..=9 | 13..=16),
    }
}

/// H.265 Table 7-1 type names.
pub fn hevc_type_name(ty: u8) -> &'static str {
    match ty {
        0 => "TRAIL_N",
        1 => "TRAIL_R",
        2 => "TSA_N",
        3 => "TSA_R",
        4 => "STSA_N",
        5 => "STSA_R",
        6 => "RADL_N",
        7 => "RADL_R",
        8 => "RASL_N",
        9 => "RASL_R",
        10 | 12 | 14 => "RSV_VCL_N",
        11 | 13 | 15 => "RSV_VCL_R",
        16 => "BLA_W_LP",
        17 => "BLA_W_RADL",
        18 => "BLA_N_LP",
        19 => "IDR_W_RADL",
        20 => "IDR_N_LP",
        21 => "CRA_NUT",
        22 | 23 => "RSV_IRAP_VCL",
        24..=31 => "RSV_VCL",
        32 => "VPS_NUT",
        33 => "SPS_NUT",
        34 => "PPS_NUT",
        35 => "AUD_NUT",
        36 => "EOS_NUT",
        37 => "EOB_NUT",
        38 => "FD_NUT",
        39 => "PREFIX_SEI_NUT",
        40 => "SUFFIX_SEI_NUT",
        41..=47 => "RSV_NVCL",
        _ => "UNSPEC",
    }
}

/// H.264 Table 7-1 type names.
pub fn avc_type_name(ty: u8) -> &'static str {
    match ty {
        1 => "Slice (non-IDR)",
        2 => "Slice partition A",
        3 => "Slice partition B",
        4 => "Slice partition C",
        5 => "Slice (IDR)",
        6 => "SEI",
        7 => "SPS",
        8 => "PPS",
        9 => "AUD",
        10 => "End of sequence",
        11 => "End of stream",
        12 => "Filler",
        13 => "SPS extension",
        14 => "Prefix NAL",
        15 => "Subset SPS",
        16 => "DPS",
        17 | 18 | 22 | 23 => "Reserved",
        19 => "Aux slice",
        20 => "Slice extension",
        21 => "Slice ext (depth)",
        _ => "Unspecified",
    }
}

/// Type name for a unit under `codec`.
pub fn nal_type_name(codec: NalCodec, ty: u8) -> &'static str {
    match codec {
        NalCodec::Hevc => hevc_type_name(ty),
        NalCodec::Avc => avc_type_name(ty),
    }
}

/// Locate the original bitstream next to a loaded `.catb` (M-E §B):
/// 1. `.catb` inside a decoder-run workdir `<name>.catb-run/` → the sibling
///    `<name>` of the workdir (decoder_run::derive_workdir convention).
/// 2. Else a same-stem sibling with a known bitstream extension.
pub fn infer_bitstream_source(catb_path: &Path) -> Option<PathBuf> {
    if let Some(parent) = catb_path.parent() {
        if let Some(dir_name) = parent.file_name().and_then(|n| n.to_str()) {
            if let Some(stream_name) = dir_name.strip_suffix(".catb-run") {
                if !stream_name.is_empty() {
                    let candidate = parent
                        .parent()
                        .map(|gp| gp.join(stream_name))
                        .unwrap_or_else(|| PathBuf::from(stream_name));
                    if candidate.is_file() {
                        return Some(candidate);
                    }
                }
            }
        }
    }
    for ext in BITSTREAM_EXTS {
        let candidate = catb_path.with_extension(ext);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}
