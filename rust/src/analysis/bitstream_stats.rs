//! Bitstream telemetry analysis layer over the `.catb` v4 reader.
//!
//! Implemented from the CC0 .catb v4 format specification
//! (codec-analyzer docs/catb-v4-format.md); no GPL code consulted.
//!
//! Wraps [`CatbFile`] with:
//! - a **CVS-aware display-order map** (viewer YUV frames are in display
//!   order; `.catb` FRAME records are in decode order),
//! - resolution resolution (`meta.parameter_sets` SPS width/height first,
//!   block bounding extent fallback),
//! - a per-frame LRU cache of parsed [`BsBlock`] lists.

use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, Mutex};

use lru::LruCache;

use crate::core::catb::{BsBlock, CabacRow, CatbFile, FrameAux, SyntaxRow};

/// How many frames' block lists to keep parsed in memory.
const BLOCK_CACHE_FRAMES: usize = 32;

/// Sanity ceiling for the resolved stream dimensions. Malformed/hostile
/// files can carry arbitrary u32 SPS sizes or block coordinates; anything
/// beyond this (double 8K) is rejected instead of feeding `rasterize_blocks`
/// / `refresh_derived` a cols×rows allocation in the 10^17 range.
const MAX_STREAM_DIMENSION: u32 = 16384;

/// Where the reported resolution came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionSource {
    /// `meta.parameter_sets` SPS-like entry carried width/height.
    ParameterSets,
    /// Bounding extent (max x+w, max y+h) over all frames' blocks.
    BlockExtent,
    /// Neither method produced a value (no parameter sets, no blocks).
    Unknown,
}

/// Per-frame summary for the status bar.
#[derive(Debug, Clone)]
pub struct FrameSummary {
    pub poc: i32,
    pub frame_type: String,
    /// Decode-order index of this display frame in the `.catb`.
    pub decode_idx: usize,
    pub slice_bits: Option<i64>,
    pub block_count: usize,
}

/// A loaded `.catb` with display-order mapping, resolution, and block cache.
pub struct BitstreamFile {
    pub catb: CatbFile,
    /// Path the `.catb` was opened from — anchor for resolving relative
    /// stage-image paths (M-C: decoder-run writes the BMPs next to it).
    pub path: std::path::PathBuf,
    /// `display_map[display_idx] = decode_idx`. Only `output == true` frames
    /// appear; built CVS-segment by CVS-segment (POC resets at IDR/BLA).
    pub display_map: Vec<usize>,
    pub width: u32,
    pub height: u32,
    pub resolution_source: ResolutionSource,
    cache: Mutex<LruCache<usize, Arc<Vec<BsBlock>>>>,
    /// 1-frame `frame_aux` LRU (M-E): loop-filter rows outnumber blocks
    /// 12–15× (fixtures: 228–457 KB JSON per frame), so only the frame on
    /// screen stays parsed.
    aux_cache: Mutex<Option<(usize, Arc<FrameAux>)>>,
}

impl std::fmt::Debug for BitstreamFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitstreamFile")
            .field("catb", &self.catb)
            .field("display_map", &self.display_map)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("resolution_source", &self.resolution_source)
            .finish()
    }
}

impl BitstreamFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let catb = CatbFile::open(path)?;
        let display_map = build_display_map(&catb);
        let (width, height, resolution_source) = resolve_resolution(&catb)?;
        Ok(Self {
            catb,
            path: path.to_path_buf(),
            display_map,
            width,
            height,
            resolution_source,
            cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(BLOCK_CACHE_FRAMES).expect("nonzero cache size"),
            )),
            aux_cache: Mutex::new(None),
        })
    }

    /// Number of displayable (output) frames.
    pub fn frame_count(&self) -> usize {
        self.display_map.len()
    }

    /// Decode-order index for a display-order index.
    pub fn decode_idx(&self, display_idx: usize) -> Option<usize> {
        self.display_map.get(display_idx).copied()
    }

    /// Cached block list for a **decode-order** frame index.
    pub fn blocks_decode(&self, decode_idx: usize) -> Result<Arc<Vec<BsBlock>>, String> {
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(hit) = cache.get(&decode_idx) {
                return Ok(Arc::clone(hit));
            }
        }
        let blocks = Arc::new(self.catb.blocks_for_frame(decode_idx)?);
        if let Ok(mut cache) = self.cache.lock() {
            cache.put(decode_idx, Arc::clone(&blocks));
        }
        Ok(blocks)
    }

    /// Cached `frame_aux` (loop-filter / SAO rows) for a **decode-order**
    /// frame index — lazy parse + 1-frame LRU (M-E).
    pub fn frame_aux_decode(&self, decode_idx: usize) -> Result<Arc<FrameAux>, String> {
        if let Ok(cache) = self.aux_cache.lock() {
            if let Some((idx, aux)) = cache.as_ref() {
                if *idx == decode_idx {
                    return Ok(Arc::clone(aux));
                }
            }
        }
        let aux = Arc::new(self.catb.frame_aux_for_frame(decode_idx)?);
        if let Ok(mut cache) = self.aux_cache.lock() {
            *cache = Some((decode_idx, Arc::clone(&aux)));
        }
        Ok(aux)
    }

    /// Cached block list for a **display-order** frame index.
    pub fn blocks_display(&self, display_idx: usize) -> Result<Arc<Vec<BsBlock>>, String> {
        let decode_idx = self
            .decode_idx(display_idx)
            .ok_or_else(|| format!("catb: display index {display_idx} out of range"))?;
        self.blocks_decode(decode_idx)
    }

    /// Status-bar summary for a display-order frame index.
    pub fn frame_summary(&self, display_idx: usize) -> Option<FrameSummary> {
        let decode_idx = self.decode_idx(display_idx)?;
        let frame = self.catb.frames.get(decode_idx)?;
        Some(FrameSummary {
            poc: frame.poc,
            frame_type: frame.frame_type.clone(),
            decode_idx,
            slice_bits: frame.slice_bits,
            block_count: frame.block_n.max(0) as usize,
        })
    }
}

/// Build the CVS-aware display-order map (§14.1 frame_type conventions).
///
/// POC resets at IDR/BLA (HEVC), IDR (AVC), and KEY (AV1) — a global POC sort
/// breaks on multi-IDR streams. Procedure: split the decode-order sequence
/// into CVS segments at those boundaries (a stream-leading CRA also starts a
/// segment, trivially), sort each segment's `output == true` frames by POC
/// (stable, so decode order breaks ties), then concatenate segments in order.
pub fn build_display_map(catb: &CatbFile) -> Vec<usize> {
    let mut segments: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for (i, frame) in catb.frames.iter().enumerate() {
        // Case-insensitive: fixture decoders emit e.g. codec "HEVC"/"avc";
        // frame_type is uppercase by convention but tolerate variants.
        let ft = frame.frame_type.to_ascii_uppercase();
        let boundary = matches!(ft.as_str(), "IDR" | "BLA" | "KEY");
        if boundary && !current.is_empty() {
            segments.push(std::mem::take(&mut current));
        }
        current.push(i);
    }
    if !current.is_empty() {
        segments.push(current);
    }

    let mut map = Vec::with_capacity(catb.frames.len());
    for segment in segments {
        let mut output_frames: Vec<usize> = segment
            .into_iter()
            .filter(|&i| catb.frames[i].output)
            .collect();
        output_frames.sort_by_key(|&i| catb.frames[i].poc);
        map.extend(output_frames);
    }
    map
}

/// Resolve the coded resolution: `meta.parameter_sets` first, block extent
/// fallback. Returns `(width, height, source)`.
pub fn resolve_resolution(catb: &CatbFile) -> Result<(u32, u32, ResolutionSource), String> {
    if let Some((w, h)) = resolution_from_parameter_sets(catb) {
        return Ok((w, h, ResolutionSource::ParameterSets));
    }
    if let Some((w, h)) = resolution_from_block_extent(catb)? {
        return Ok((w, h, ResolutionSource::BlockExtent));
    }
    Ok((0, 0, ResolutionSource::Unknown))
}

/// Scan `meta.parameter_sets` for an SPS-like entry with usable dimensions.
///
/// The spec (§4) leaves `fields` decoder-defined; observed keys are plain
/// `width`/`height` (instrumented FFmpeg for both HEVC and AVC). HEVC
/// spec-style names are also tried as a fallback.
fn resolution_from_parameter_sets(catb: &CatbFile) -> Option<(u32, u32)> {
    let sets = catb.meta.parameter_sets.as_array()?;
    for set in sets {
        let kind = set.get("kind").and_then(|k| k.as_str()).unwrap_or("");
        let kind_lc = kind.to_ascii_lowercase();
        // SPS (HEVC/AVC) or AV1 sequence header.
        if !(kind_lc.contains("sps") || kind_lc.contains("sequence")) {
            continue;
        }
        let fields = set.get("fields")?;
        let dim = |names: &[&str]| -> Option<u32> {
            for name in names {
                if let Some(v) = fields.get(*name).and_then(|x| x.as_u64()) {
                    if v > 0 && v <= MAX_STREAM_DIMENSION as u64 {
                        return u32::try_from(v).ok();
                    }
                }
            }
            None
        };
        let w = dim(&["width", "pic_width_in_luma_samples"]);
        let h = dim(&["height", "pic_height_in_luma_samples"]);
        if let (Some(w), Some(h)) = (w, h) {
            return Some((w, h));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// M1: viewer-frame ↔ catb display-frame offset mapping
// ---------------------------------------------------------------------------

/// Map a viewer (display-order YUV) frame index to a catb display-order index
/// with the user's manual frame offset applied (R4 trim compensation):
/// `catb_display = viewer + offset`. Returns `None` when the result is
/// negative — the viewer frame precedes the telemetry.
pub fn viewer_to_catb_display(viewer_idx: usize, offset: i32) -> Option<usize> {
    let v = viewer_idx as i64 + offset as i64;
    if v < 0 {
        None
    } else {
        Some(v as usize)
    }
}

// ---------------------------------------------------------------------------
// M-A: filmstrip reference graph (arrows + reference-frequency dots)
// ---------------------------------------------------------------------------

/// One reference-arrow edge of a viewer frame (filmstrip, VQA Thumbnails
/// view analogue): `to` is the referenced *viewer* frame index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefEdge {
    pub to: usize,
    /// 0 = ref_list0 (orange), 1 = ref_list1 (purple).
    pub list: u8,
    /// Long-term reference (green in the strip).
    pub long_term: bool,
}

/// Per-viewer-frame reference data derived once per (file, offset, total).
#[derive(Debug, Clone, Default)]
pub struct FilmstripRefs {
    /// `edges[v]` = the frames viewer frame `v` references (slice ref lists).
    pub edges: Vec<Vec<RefEdge>>,
    /// `counts[v]` = how many times viewer frame `v` appears in any other
    /// frame's slice ref list (reference-usage frequency dot).
    pub counts: Vec<u32>,
    /// `exactness[v]` = the frame has exactness problems (any non-empty
    /// `exactness_missing` entry or dropped telemetry rows).
    pub exactness: Vec<bool>,
}

/// Nearest *preceding* (decode order) frame whose POC equals `poc`.
///
/// Reference pictures are always already-decoded frames, so only
/// `i < from_decode` is searched. POC values can repeat across CVS segments
/// (multi-IDR streams); restricting to the decode past both picks the
/// current segment's frame and never maps a ref to a future/next-segment
/// frame that could not possibly be in the DPB yet.
pub fn nearest_poc_match(frame_pocs: &[i32], from_decode: usize, poc: i32) -> Option<usize> {
    let end = from_decode.min(frame_pocs.len());
    frame_pocs[..end].iter().rposition(|&p| p == poc)
}

/// Pure edge/count derivation over plain slices (unit-testable without a
/// `.catb` file):
/// - `frame_pocs[d]` — POC of decode frame `d`,
/// - `slice_refs[d]` — flattened slice ref-list rows `(poc, list, long_term)`
///   of decode frame `d`,
/// - `display_map[disp] = decode idx` (output frames, display order),
/// - viewer `v` ↔ catb display `v + offset`.
///
/// References whose POC has no decode-order match, or whose target falls
/// outside the viewer range after offset mapping, are dropped (§C: 매핑 밖
/// 참조는 생략).
pub fn derive_ref_edges(
    frame_pocs: &[i32],
    slice_refs: &[Vec<(i32, u8, bool)>],
    display_map: &[usize],
    offset: i32,
    viewer_total: usize,
) -> (Vec<Vec<RefEdge>>, Vec<u32>) {
    // decode idx → display idx (position in display_map).
    let mut decode_to_display = vec![usize::MAX; frame_pocs.len()];
    for (disp, &dec) in display_map.iter().enumerate() {
        if let Some(slot) = decode_to_display.get_mut(dec) {
            *slot = disp;
        }
    }
    // display idx → viewer idx (inverse offset mapping).
    let display_to_viewer = |disp: usize| -> Option<usize> {
        let v = disp as i64 - offset as i64;
        (v >= 0 && (v as usize) < viewer_total).then_some(v as usize)
    };
    let mut edges: Vec<Vec<RefEdge>> = vec![Vec::new(); viewer_total];
    let mut counts = vec![0u32; viewer_total];
    for (v, edge_list) in edges.iter_mut().enumerate() {
        let Some(disp) = viewer_to_catb_display(v, offset) else {
            continue;
        };
        let Some(&dec) = display_map.get(disp) else {
            continue;
        };
        let Some(refs) = slice_refs.get(dec) else {
            continue;
        };
        for &(poc, list, long_term) in refs {
            let Some(target_dec) = nearest_poc_match(frame_pocs, dec, poc) else {
                continue;
            };
            let target_disp = decode_to_display[target_dec];
            if target_disp == usize::MAX {
                continue; // non-output frame: not on the strip
            }
            let Some(to) = display_to_viewer(target_disp) else {
                continue;
            };
            edge_list.push(RefEdge {
                to,
                list,
                long_term,
            });
            counts[to] += 1;
        }
    }
    (edges, counts)
}

/// Build the filmstrip reference data for a loaded `.catb` (one pass over
/// `meta.frames_meta`; called once per (file, offset, total) cache key).
pub fn build_filmstrip_refs(
    file: &BitstreamFile,
    offset: i32,
    viewer_total: usize,
) -> FilmstripRefs {
    let frames = &file.catb.frames;
    let frame_pocs: Vec<i32> = frames.iter().map(|f| f.poc).collect();
    let meta = &file.catb.meta.frames_meta;
    let slice_refs: Vec<Vec<(i32, u8, bool)>> = (0..frames.len())
        .map(|d| {
            meta.get(d)
                .map(|m| {
                    let mut rows = Vec::new();
                    for sh in &m.slice_headers {
                        for e in &sh.ref_list0 {
                            rows.push((e.poc as i32, 0u8, e.long_term));
                        }
                        for e in &sh.ref_list1 {
                            rows.push((e.poc as i32, 1u8, e.long_term));
                        }
                    }
                    rows
                })
                .unwrap_or_default()
        })
        .collect();
    let (edges, counts) = derive_ref_edges(
        &frame_pocs,
        &slice_refs,
        &file.display_map,
        offset,
        viewer_total,
    );
    let mut exactness = vec![false; viewer_total];
    for (v, flag) in exactness.iter_mut().enumerate() {
        let Some(disp) = viewer_to_catb_display(v, offset) else {
            continue;
        };
        let Some(&dec) = file.display_map.get(disp) else {
            continue;
        };
        if let Some(m) = meta.get(dec) {
            *flag = m.exactness_missing.iter().any(|s| !s.is_empty())
                || m.block_dropped_rows.iter().any(|&n| n > 0);
        }
    }
    FilmstripRefs {
        edges,
        counts,
        exactness,
    }
}

/// Reference-frequency dot radius tier (§C: 3단계): 0 → none, 1 → small,
/// 2–3 → medium, ≥4 → large.
pub fn ref_count_tier(count: u32) -> u8 {
    match count {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        _ => 3,
    }
}

// ---------------------------------------------------------------------------
// M1: frame-type classification (filmstrip colours)
// ---------------------------------------------------------------------------

/// Coarse frame-type class for filmstrip colouring: I=red / P=blue / B=green.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameTypeClass {
    I,
    P,
    B,
    Other,
}

impl FrameTypeClass {
    /// Classify a `.catb` frame_type label ("IDR", "CRA", "BLA", "I", "P",
    /// "B", "KEY", …). Case-insensitive.
    pub fn from_label(label: &str) -> Self {
        let up = label.to_ascii_uppercase();
        match up.as_str() {
            "I" | "IDR" | "CRA" | "BLA" | "KEY" => FrameTypeClass::I,
            "P" => FrameTypeClass::P,
            "B" => FrameTypeClass::B,
            _ => {
                // Tolerate decoder-specific variants like "IDR_W_RADL",
                // "BLA_W_LP", "CRA_NUT".
                if up.starts_with("IDR") || up.starts_with("BLA") || up.starts_with("CRA") {
                    FrameTypeClass::I
                } else {
                    FrameTypeClass::Other
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// M1: prediction-mode classification
// ---------------------------------------------------------------------------

/// Discrete prediction-mode class for the Mode fill (§5 legend swatches).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeClass {
    Unknown,
    Intra,
    Inter,
    Skip,
    Merge,
}

impl ModeClass {
    pub fn from_label(label: &str) -> Self {
        let lc = label.to_ascii_lowercase();
        // Order matters: "skip"/"merge" labels may embed "inter"
        // (e.g. "Inter (Skip)").
        if lc.contains("skip") {
            ModeClass::Skip
        } else if lc.contains("merge") {
            ModeClass::Merge
        } else if lc.contains("intra") {
            ModeClass::Intra
        } else if lc.contains("inter") {
            ModeClass::Inter
        } else {
            ModeClass::Unknown
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            ModeClass::Unknown => "?",
            ModeClass::Intra => "Intra",
            ModeClass::Inter => "Inter",
            ModeClass::Skip => "Skip",
            ModeClass::Merge => "Merge",
        }
    }

    fn index(&self) -> usize {
        match self {
            ModeClass::Unknown => 0,
            ModeClass::Intra => 1,
            ModeClass::Inter => 2,
            ModeClass::Skip => 3,
            ModeClass::Merge => 4,
        }
    }

    fn from_index(i: usize) -> Self {
        match i {
            1 => ModeClass::Intra,
            2 => ModeClass::Inter,
            3 => ModeClass::Skip,
            4 => ModeClass::Merge,
            _ => ModeClass::Unknown,
        }
    }
}

// ---------------------------------------------------------------------------
// M4: intra prediction directions from SYNTAX rows
// ---------------------------------------------------------------------------

/// One extracted intra luma prediction mode of a block (M4 Intra layer).
///
/// `angle_deg` is the prediction line orientation in **math convention**
/// (degrees CCW from +x, y up); renderers negate y for screen space. A line
/// through the (sub)block centre at this angle visualizes the direction —
/// only the orientation matters, so θ and θ+180° draw identically.
/// `None` = non-angular mode (Planar / DC / plane): no line, badge instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntraDir {
    /// Raw mode value as stored in the SYNTAX row.
    pub mode: i32,
    pub angle_deg: Option<f32>,
    /// Badge for non-angular modes: `"P"` (Planar/Plane), `"DC"`, else `""`.
    pub badge: &'static str,
}

/// HEVC intra luma mode → line angle (degrees, math convention).
///
/// Standard public knowledge (H.265 §8.4.4.2.6 intraPredAngle table /
/// Figure 8-1 mode directions): 0 = Planar, 1 = DC (non-angular);
/// 2..=34 angular, uniformly spanning 225°→45° so that
/// `angle = 225 − (mode − 2) · 180/32`. Anchors: 2→225°, 10→180°
/// (horizontal), 18→135°, 26→90° (vertical), 34→45°.
pub fn intra_angle_hevc(mode: i32) -> Option<f32> {
    if (2..=34).contains(&mode) {
        Some(225.0 - (mode - 2) as f32 * (180.0 / 32.0))
    } else {
        None
    }
}

/// AVC 4x4 / 8x8 intra mode → line angle (degrees, math convention).
///
/// Standard public knowledge (H.264 §8.3.1.2.x, Figure 8-4 prediction
/// directions; intermediate modes use the ±½-slope directions, i.e.
/// atan(1/2) ≈ 26.565° from an axis):
/// 0 Vertical 90°, 1 Horizontal 0°, 2 DC (none),
/// 3 Diagonal-Down-Left 45°, 4 Diagonal-Down-Right 135°,
/// 5 Vertical-Right 116.565°, 6 Horizontal-Down 153.435°,
/// 7 Vertical-Left 63.435°, 8 Horizontal-Up 26.565°.
/// Values 9..=11 are the decoder's edge DC variants
/// (Left-DC / Top-DC / DC-128, observed in the fixture string tables) —
/// non-angular like DC.
pub fn intra_angle_avc_nxn(mode: i32) -> Option<f32> {
    const HALF_SLOPE: f32 = 26.565_05; // atan(1/2) in degrees
    match mode {
        0 => Some(90.0),
        1 => Some(0.0),
        3 => Some(45.0),
        4 => Some(135.0),
        5 => Some(90.0 + HALF_SLOPE),
        6 => Some(180.0 - HALF_SLOPE),
        7 => Some(90.0 - HALF_SLOPE),
        8 => Some(HALF_SLOPE),
        _ => None, // 2 = DC, 9..=11 = DC variants, out-of-range
    }
}

/// AVC 16x16 intra mode → line angle (degrees, math convention).
///
/// The fixture decoder emits FFmpeg-internal 8x8/16x16 pred enum values,
/// **not** the H.264 spec 0..3 order — confirmed against the oracle JSON
/// label pairs (value 2 = "Vertical", 3 = "Plane", 6 = "DC128"):
/// 0 DC, 1 Horizontal 0°, 2 Vertical 90°, 3 Plane (non-angular),
/// 4/5/6 Left-DC / Top-DC / DC-128 (non-angular). The angular directions
/// themselves are the standard H.264 16x16 vertical/horizontal modes.
pub fn intra_angle_avc_16x16(mode: i32) -> Option<f32> {
    match mode {
        1 => Some(0.0),
        2 => Some(90.0),
        _ => None, // 0/4/5/6 DC-class, 3 Plane
    }
}

/// Badge for a non-angular mode (empty for angular ones).
fn intra_badge(name: &str, mode: i32) -> &'static str {
    match name {
        "intra_luma_pred_mode" => match mode {
            0 => "P",
            1 => "DC",
            _ => "",
        },
        "intra4x4_pred_mode" | "intra8x8_pred_mode" => match mode {
            2 | 9..=11 => "DC",
            _ => "",
        },
        "intra16x16_pred_mode" => match mode {
            0 | 4..=6 => "DC",
            3 => "P",
            _ => "",
        },
        _ => "",
    }
}

/// Extract the intra **luma** prediction modes of a block from its SYNTAX
/// rows (names observed in the fixture string tables:
/// `intra_luma_pred_mode` for HEVC, `intra4x4_pred_mode` /
/// `intra8x8_pred_mode` / `intra16x16_pred_mode` for AVC; chroma modes are
/// deliberately excluded).
///
/// Multi-mode blocks (HEVC NxN = 4 rows, AVC 8x8 = 4 rows, AVC 4x4 = 16
/// rows) return the rows **in stored (decode) order**. For AVC 4x4 that is
/// H.264 §6.4.3 z-scan (`luma4x4BlkIdx`), *not* raster: the fixture proves
/// it — `avc_cavlc.catb` block 3 (x=48, y=0, top frame row) stores its
/// LEFT_DC (mode 9, "top unavailable") rows at positions {0,1,4,5}, which
/// is exactly the top 4×4 row in z-scan and would be {0,1,2,3} in raster.
/// The 2×2 grids (HEVC NxN, AVC 8x8) have identical z-scan/raster order.
/// Renderers must map list position → sub-grid cell via
/// [`intra_subblock_pos`]; a single mode sits at the block centre. Rows
/// whose value string is not an integer are skipped.
pub fn extract_intra_modes(rows: &[SyntaxRow]) -> Vec<IntraDir> {
    let mut out = Vec::new();
    for row in rows {
        let angle_fn: fn(i32) -> Option<f32> = match row.name.as_str() {
            "intra_luma_pred_mode" => intra_angle_hevc,
            "intra4x4_pred_mode" | "intra8x8_pred_mode" => intra_angle_avc_nxn,
            "intra16x16_pred_mode" => intra_angle_avc_16x16,
            _ => continue,
        };
        let Ok(mode) = row.value.trim().parse::<i32>() else {
            continue;
        };
        out.push(IntraDir {
            mode,
            angle_deg: angle_fn(mode),
            badge: intra_badge(&row.name, mode),
        });
    }
    out
}

/// Sub-grid dimension for `n` extracted intra dirs: 1 (single), 2 (HEVC NxN
/// / AVC 8x8), or 4 (AVC 4x4). Stored order is decode order (see
/// [`extract_intra_modes`]); use [`intra_subblock_pos`] to place entries.
pub fn intra_grid_dim(n: usize) -> u32 {
    match n {
        0..=1 => 1,
        2..=4 => 2,
        _ => 4,
    }
}

/// Map the `k`-th extracted intra dir to its `(col, row)` cell on the
/// `dim`×`dim` sub-grid.
///
/// For `dim == 4` (AVC intra4x4) the stored order is H.264 §6.4.3 z-scan
/// (`luma4x4BlkIdx`): four 8×8 quadrants in raster order, each holding four
/// 4×4 blocks in raster order. Remap z→raster with `q = k >> 2` (quadrant),
/// `z = k & 3` (block within quadrant):
/// `raster = (q>>1)*8 + (z>>1)*4 + (q&1)*2 + (z&1)`.
/// For `dim <= 2` z-scan and raster coincide, so `k` is used directly.
/// Returns `None` when `k` falls outside the grid.
pub fn intra_subblock_pos(k: usize, dim: u32) -> Option<(u32, u32)> {
    let raster = if dim == 4 {
        let (q, z) = (k >> 2, k & 3);
        ((q >> 1) * 8 + (z >> 1) * 4 + (q & 1) * 2 + (z & 1)) as u32
    } else {
        k as u32
    };
    let (col, row) = (raster % dim.max(1), raster / dim.max(1));
    (row < dim).then_some((col, row))
}

// ---------------------------------------------------------------------------
// M-B: per-block TRANSFORM aggregates (CoeffEnergy / NonzeroCoeffs fills)
// ---------------------------------------------------------------------------

/// Per-block TX aggregate, parallel to a frame's block list: total
/// `coeff_abs_sum` and total `nonzero_coeff_count` over the block's TX
/// records (absent §9.1 scalars contribute 0).
///
/// A block without TX rows (skip CU, capture level below `full`… ) yields
/// `(0, 0)` — **0-값 취급, coverage 제외 아님**: zero residual energy is the
/// true coded value for skip/uncoded blocks, and excluding them would bias
/// every correlation toward residual-carrying blocks only. Capture levels
/// that drop the TRANSFORMS section entirely degrade to an all-zero (thus
/// zero-variance → `r = None`) metric rather than a misleading partial one.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BlockTxAgg {
    pub abs_sum: f64,
    pub nonzero: f64,
}

/// Aggregate every block's TX rows (one cheap fixed-field pass; parse
/// errors on a malformed sub-range decay to 0 — one bad block must not kill
/// the whole fill).
pub fn aggregate_block_tx(file: &BitstreamFile, blocks: &[BsBlock]) -> Vec<BlockTxAgg> {
    blocks
        .iter()
        .map(|b| {
            if b.tx_n > 0 {
                file.catb
                    .tx_aggregate_for_block(b)
                    .map(|(a, n)| BlockTxAgg {
                        abs_sum: a as f64,
                        nonzero: n as f64,
                    })
                    .unwrap_or_default()
            } else {
                BlockTxAgg::default()
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// M-B: Stats tab aggregation (VQA Syntax Stats / Picture Stats analogue)
// ---------------------------------------------------------------------------

/// One aggregated syntax-element row of the Stats tab.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntaxAgg {
    pub name: String,
    pub count: usize,
    pub bits: i64,
}

/// Aggregate `(name, bits)` syntax rows by name, sorted by total bits
/// descending (ties: name ascending, deterministic).
pub fn aggregate_syntax_rows<'a, I>(rows: I) -> Vec<SyntaxAgg>
where
    I: IntoIterator<Item = (&'a str, i64)>,
{
    let mut map: std::collections::HashMap<&'a str, (usize, i64)> =
        std::collections::HashMap::new();
    for (name, bits) in rows {
        let e = map.entry(name).or_insert((0, 0));
        e.0 += 1;
        e.1 += bits;
    }
    let mut out: Vec<SyntaxAgg> = map
        .into_iter()
        .map(|(name, (count, bits))| SyntaxAgg {
            name: name.to_string(),
            count,
            bits,
        })
        .collect();
    out.sort_by(|a, b| b.bits.cmp(&a.bits).then_with(|| a.name.cmp(&b.name)));
    out
}

/// One aggregated CABAC row of the Stats tab.
#[derive(Debug, Clone, PartialEq)]
pub struct CabacAgg {
    pub name: String,
    /// Number of decoded bins.
    pub bins: usize,
    /// Total bits attributed to the bins.
    pub bits: i64,
    /// Number of distinct context indices used.
    pub ctx_count: usize,
}

impl CabacAgg {
    /// Mean bits per bin (0 when no bins).
    pub fn bits_per_bin(&self) -> f64 {
        if self.bins == 0 {
            0.0
        } else {
            self.bits as f64 / self.bins as f64
        }
    }
}

/// Aggregate CABAC bin rows by name, sorted by bin count descending
/// (ties: name ascending).
pub fn aggregate_cabac_rows(rows: &[CabacRow]) -> Vec<CabacAgg> {
    use std::collections::{HashMap, HashSet};
    let mut map: HashMap<&str, (usize, i64, HashSet<i32>)> = HashMap::new();
    for r in rows {
        let e = map.entry(r.name.as_str()).or_default();
        e.0 += 1;
        e.1 += r.bits;
        e.2.insert(r.ctx);
    }
    let mut out: Vec<CabacAgg> = map
        .into_iter()
        .map(|(name, (bins, bits, ctxs))| CabacAgg {
            name: name.to_string(),
            bins,
            bits,
            ctx_count: ctxs.len(),
        })
        .collect();
    out.sort_by(|a, b| b.bins.cmp(&a.bins).then_with(|| a.name.cmp(&b.name)));
    out
}

/// One MV scatter point of the Stats tab (px units, §14.5 quarter-pel / 4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvPoint {
    pub x: f64,
    pub y: f64,
    /// 0 = L0 (orange), 1 = L1 (purple).
    pub list: u8,
    /// Block origin, for the hover tooltip.
    pub block_x: u32,
    pub block_y: u32,
}

/// Collect every PU-level MV of a frame as scatter points: per-PU REF rows
/// when present, block-level MV fallback otherwise (draw_mv_layer rule).
pub fn collect_mv_points(file: &BitstreamFile, blocks: &[BsBlock]) -> Vec<MvPoint> {
    let mut out = Vec::new();
    for b in blocks {
        let mut any_pu = false;
        if b.ref_n > 0 {
            if let Ok(refs) = file.catb.refs_for_block(b) {
                for r in &refs {
                    let Some((mx, my)) = r.mv else { continue };
                    any_pu = true;
                    let list = u8::from(
                        r.list
                            .as_deref()
                            .is_some_and(|l| l.eq_ignore_ascii_case("l1")),
                    );
                    out.push(MvPoint {
                        x: mx as f64 / 4.0,
                        y: my as f64 / 4.0,
                        list,
                        block_x: b.x,
                        block_y: b.y,
                    });
                }
            }
        }
        if !any_pu {
            if let Some((mx, my)) = b.mv {
                let list = u8::from(
                    b.reference_list
                        .as_deref()
                        .is_some_and(|l| l.eq_ignore_ascii_case("l1")),
                );
                out.push(MvPoint {
                    x: mx as f64 / 4.0,
                    y: my as f64 / 4.0,
                    list,
                    block_x: b.x,
                    block_y: b.y,
                });
            }
        }
    }
    out
}

/// Everything the Stats tab shows for one frame (built once per
/// (file, display frame) — see the window's cache).
#[derive(Debug, Clone, Default)]
pub struct FrameStatsData {
    pub blocks: usize,
    pub tus: usize,
    pub syntax_rows: usize,
    pub cabac_bins: usize,
    /// Frame `data_bits` when the FRAME record carries it, else the block
    /// bits sum (percentage base for the syntax table).
    pub total_bits: i64,
    /// Block-level syntax aggregate + slice-header "(slice)" rows,
    /// bits-descending.
    pub syntax: Vec<SyntaxAgg>,
    pub cabac: Vec<CabacAgg>,
    pub mv_points: Vec<MvPoint>,
}

/// Build the Stats tab data for a catb **display-order** frame index.
pub fn compute_frame_stats(file: &BitstreamFile, display_idx: usize) -> Option<FrameStatsData> {
    let decode_idx = file.decode_idx(display_idx)?;
    let frame = file.catb.frames.get(decode_idx)?;
    let blocks = file.blocks_display(display_idx).ok()?;

    // Syntax rows over all blocks (owned strings collected once so the
    // aggregate can borrow them), plus slice-header rows with the
    // "(slice) " prefix (§C-1).
    let mut syntax_rows: Vec<(String, i64)> = Vec::new();
    let mut cabac_rows: Vec<CabacRow> = Vec::new();
    let mut tus = 0usize;
    for b in blocks.iter() {
        if b.syntax_n > 0 {
            if let Ok(rows) = file.catb.syntax_for_block(b) {
                syntax_rows.extend(rows.into_iter().map(|r| (r.name, r.bits)));
            }
        }
        if b.cabac_n > 0 {
            if let Ok(rows) = file.catb.cabac_for_block(b) {
                cabac_rows.extend(rows);
            }
        }
        tus += b.tx_n.max(0) as usize;
    }
    let block_syntax_rows = syntax_rows.len();
    if let Some(m) = file.catb.meta.frames_meta.get(decode_idx) {
        for sh in &m.slice_headers {
            for (name, _value, bits) in &sh.syntax {
                syntax_rows.push((format!("(slice) {name}"), *bits));
            }
        }
    }
    let total_bits = frame.data_bits.unwrap_or_else(|| {
        blocks.iter().map(|b| b.bits.max(0)).sum()
    });
    let syntax = aggregate_syntax_rows(syntax_rows.iter().map(|(n, b)| (n.as_str(), *b)));
    let cabac_bins = cabac_rows.len();
    let cabac = aggregate_cabac_rows(&cabac_rows);
    let mv_points = collect_mv_points(file, &blocks);
    Some(FrameStatsData {
        blocks: blocks.len(),
        tus,
        syntax_rows: block_syntax_rows,
        cabac_bins,
        total_bits,
        syntax,
        cabac,
        mv_points,
    })
}

// ---------------------------------------------------------------------------
// M1: L1 rasterization — variable CU rects → fixed-cell grid
// ---------------------------------------------------------------------------

/// Fixed-cell rasterization of a frame's variable-size CU rects (§ L1).
///
/// Values are **area-weighted**: a CU overlapping a cell contributes to that
/// cell proportionally to the overlap area. `bpp` distributes each block's
/// bits area-proportionally, so a cell's bpp is "bits per pixel of the
/// covered area". `mode` is the class with the largest covered area.
#[derive(Debug, Clone)]
pub struct BitstreamGrid {
    /// Cell edge in stream pixels (8 for L1, 64/32 for the LOD level).
    pub cell: u32,
    pub cols: u32,
    pub rows: u32,
    /// Area-weighted mean QP per cell (0.0 where uncovered).
    pub qp: Vec<f32>,
    /// Bits per covered pixel per cell (0.0 where uncovered).
    pub bpp: Vec<f32>,
    /// Area-weighted mean L1-norm MV magnitude in pixels (02 §1: the
    /// quarter-pel L1 norm `|x|+|y|` divided by 4).
    pub mv_mag: Vec<f32>,
    /// Majority (largest-area) mode class per cell.
    pub mode: Vec<ModeClass>,
    /// Coefficient energy per covered pixel (M-B): each block's TX
    /// `coeff_abs_sum` total distributed area-proportionally, like `bpp`.
    /// All-zero when no TX aggregate was supplied (see [`BlockTxAgg`]).
    pub coeff_energy: Vec<f32>,
    /// Nonzero-coefficient density per covered pixel (M-B), same rule.
    pub nz_density: Vec<f32>,
    /// Covered area / full cell area ∈ [0, 1].
    pub coverage: Vec<f32>,
}

impl BitstreamGrid {
    pub fn is_empty(&self) -> bool {
        self.cols == 0 || self.rows == 0
    }
}

/// LOD aggregate cell size for a codec (§2 Grid layer / LOD rule):
/// HEVC CTU = 64 px, otherwise (AVC & co.) 32 px.
pub fn lod_cell_size(codec: &str) -> u32 {
    if codec.to_ascii_lowercase().contains("hevc") || codec.to_ascii_lowercase().contains("265") {
        64
    } else {
        32
    }
}

/// Below this zoom the canvas renders the LOD aggregate grid instead of the
/// per-CU rect list (native_ui rule, 01 §4).
pub const LOD_ZOOM_THRESHOLD: f32 = 1.5;

/// True when the given zoom should render the LOD aggregate level.
pub fn use_lod(zoom: f32) -> bool {
    zoom < LOD_ZOOM_THRESHOLD
}

/// Rasterize variable-size blocks into a `cell`-px grid over a
/// `width`×`height` stream, area-weighting every value. No TX aggregate:
/// `coeff_energy` / `nz_density` come out all-zero.
pub fn rasterize_blocks(blocks: &[BsBlock], width: u32, height: u32, cell: u32) -> BitstreamGrid {
    rasterize_blocks_tx(blocks, None, width, height, cell)
}

/// Like [`rasterize_blocks`] but with an optional per-block TX aggregate
/// (parallel to `blocks`, see [`aggregate_block_tx`]) feeding the
/// `coeff_energy` / `nz_density` cells (M-B).
pub fn rasterize_blocks_tx(
    blocks: &[BsBlock],
    tx: Option<&[BlockTxAgg]>,
    width: u32,
    height: u32,
    cell: u32,
) -> BitstreamGrid {
    let cell = cell.max(1);
    let cols = width.div_ceil(cell);
    let rows = height.div_ceil(cell);
    let n = (cols as usize) * (rows as usize);
    if n == 0 {
        return BitstreamGrid {
            cell,
            cols,
            rows,
            qp: Vec::new(),
            bpp: Vec::new(),
            mv_mag: Vec::new(),
            mode: Vec::new(),
            coeff_energy: Vec::new(),
            nz_density: Vec::new(),
            coverage: Vec::new(),
        };
    }

    let mut area = vec![0.0_f64; n];
    let mut qp_acc = vec![0.0_f64; n];
    let mut bits_acc = vec![0.0_f64; n];
    let mut mv_acc = vec![0.0_f64; n];
    let mut coeff_acc = vec![0.0_f64; n];
    let mut nz_acc = vec![0.0_f64; n];
    // Per-cell covered area per mode class (5 classes).
    let mut mode_area = vec![[0.0_f32; 5]; n];

    for (bi, b) in blocks.iter().enumerate() {
        if b.w == 0 || b.h == 0 {
            continue;
        }
        let block_area = (b.w as f64) * (b.h as f64);
        let bits_per_px = b.bits.max(0) as f64 / block_area;
        let agg = tx.and_then(|t| t.get(bi)).copied().unwrap_or_default();
        let coeff_per_px = agg.abs_sum.max(0.0) / block_area;
        let nz_per_px = agg.nonzero.max(0.0) / block_area;
        // Quarter-pel L1 norm / 4 → pixels (02 §1), so every consumer —
        // heatmap legend, class table, scatter, CSV — shares one unit.
        let mv_mag = b
            .mv
            .map(|(x, y)| (x.abs() + y.abs()) as f64 / 4.0)
            .unwrap_or(0.0);
        let mode = ModeClass::from_label(&b.prediction_mode);

        let x1 = b.x.saturating_add(b.w).min(width);
        let y1 = b.y.saturating_add(b.h).min(height);
        if b.x >= x1 || b.y >= y1 {
            continue;
        }
        let c0 = b.x / cell;
        let c1 = (x1 - 1) / cell;
        let r0 = b.y / cell;
        let r1 = (y1 - 1) / cell;
        for r in r0..=r1.min(rows - 1) {
            let cy0 = r * cell;
            let cy1 = (cy0 + cell).min(height);
            let oy = (y1.min(cy1) as i64 - b.y.max(cy0) as i64).max(0) as f64;
            for c in c0..=c1.min(cols - 1) {
                let cx0 = c * cell;
                let cx1 = (cx0 + cell).min(width);
                let ox = (x1.min(cx1) as i64 - b.x.max(cx0) as i64).max(0) as f64;
                let a = ox * oy;
                if a <= 0.0 {
                    continue;
                }
                let idx = (r * cols + c) as usize;
                area[idx] += a;
                qp_acc[idx] += b.qp as f64 * a;
                bits_acc[idx] += bits_per_px * a;
                mv_acc[idx] += mv_mag * a;
                coeff_acc[idx] += coeff_per_px * a;
                nz_acc[idx] += nz_per_px * a;
                mode_area[idx][mode.index()] += a as f32;
            }
        }
    }

    let full_cell = (cell as f64) * (cell as f64);
    let mut qp = vec![0.0_f32; n];
    let mut bpp = vec![0.0_f32; n];
    let mut mv = vec![0.0_f32; n];
    let mut coeff_energy = vec![0.0_f32; n];
    let mut nz_density = vec![0.0_f32; n];
    let mut mode = vec![ModeClass::Unknown; n];
    let mut coverage = vec![0.0_f32; n];
    for i in 0..n {
        if area[i] > 0.0 {
            qp[i] = (qp_acc[i] / area[i]) as f32;
            bpp[i] = (bits_acc[i] / area[i]) as f32;
            mv[i] = (mv_acc[i] / area[i]) as f32;
            coeff_energy[i] = (coeff_acc[i] / area[i]) as f32;
            nz_density[i] = (nz_acc[i] / area[i]) as f32;
            coverage[i] = (area[i] / full_cell) as f32;
            let best = mode_area[i]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| k)
                .unwrap_or(0);
            mode[i] = ModeClass::from_index(best);
        }
    }

    BitstreamGrid {
        cell,
        cols,
        rows,
        qp,
        bpp,
        mv_mag: mv,
        mode,
        coeff_energy,
        nz_density,
        coverage,
    }
}

// ---------------------------------------------------------------------------
// M1: L0 minimum-area hit test
// ---------------------------------------------------------------------------

/// Index of the **smallest-area** block containing stream point `(x, y)`
/// (V12: an 8×8 CU inside a 64×64 CTU rect must win).
pub fn hit_test_min_area(blocks: &[BsBlock], x: u32, y: u32) -> Option<usize> {
    let mut best: Option<(usize, u64)> = None;
    for (i, b) in blocks.iter().enumerate() {
        if x >= b.x && x < b.x.saturating_add(b.w) && y >= b.y && y < b.y.saturating_add(b.h) {
            let a = b.w as u64 * b.h as u64;
            if best.map(|(_, ba)| a < ba).unwrap_or(true) {
                best = Some((i, a));
            }
        }
    }
    best.map(|(i, _)| i)
}

/// Bounding extent (max x+w, max y+h) over all frames' blocks.
fn resolution_from_block_extent(catb: &CatbFile) -> Result<Option<(u32, u32)>, String> {
    let mut max_w = 0u32;
    let mut max_h = 0u32;
    for i in 0..catb.frames.len() {
        for block in catb.blocks_for_frame(i)? {
            max_w = max_w.max(block.x.saturating_add(block.w));
            max_h = max_h.max(block.y.saturating_add(block.h));
        }
    }
    if max_w == 0 || max_h == 0 {
        Ok(None)
    } else if max_w > MAX_STREAM_DIMENSION || max_h > MAX_STREAM_DIMENSION {
        // Malformed block coordinates — refuse rather than derive a
        // resolution that makes every grid allocation explode.
        Err(format!(
            "catb: block extent {max_w}x{max_h} exceeds the sane maximum \
             ({MAX_STREAM_DIMENSION})"
        ))
    } else {
        Ok(Some((max_w, max_h)))
    }
}

#[cfg(test)]
mod raster_tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn block(x: u32, y: u32, w: u32, h: u32, qp: i32, bits: i64, mode: &str, mv: Option<(i32, i32)>) -> BsBlock {
        BsBlock {
            x,
            y,
            w,
            h,
            ctu_address: 0,
            qp,
            partition: String::new(),
            prediction_mode: mode.to_string(),
            bits,
            bit_offset: 0,
            exactness_flags: 0,
            syntax_off: 0,
            syntax_n: 0,
            cabac_off: 0,
            cabac_n: 0,
            tx_off: 0,
            tx_n: 0,
            ref_off: 0,
            ref_n: 0,
            mv_flags: if mv.is_some() { crate::core::catb::MV_HAS_MV } else { 0 },
            mvp: None,
            mvd: None,
            mv,
            reference: None,
            reference_list: None,
            reference_label: None,
            reference_list_index: None,
            reference_poc: None,
            reference_frame: None,
            reference_long_term: None,
        }
    }

    #[test]
    fn rasterize_single_full_cell() {
        // One 8×8 block exactly covering cell (0,0) of a 16×16 frame.
        let blocks = vec![block(0, 0, 8, 8, 32, 128, "Inter", Some((4, -2)))];
        let g = rasterize_blocks(&blocks, 16, 16, 8);
        assert_eq!((g.cols, g.rows), (2, 2));
        assert!((g.qp[0] - 32.0).abs() < 1e-5);
        // 128 bits over 64 px = 2.0 bpp.
        assert!((g.bpp[0] - 2.0).abs() < 1e-5);
        // (|4| + |−2|) quarter-pel / 4 = 1.5 px.
        assert!((g.mv_mag[0] - 1.5).abs() < 1e-5);
        assert_eq!(g.mode[0], ModeClass::Inter);
        assert!((g.coverage[0] - 1.0).abs() < 1e-5);
        // Untouched cells stay zeroed / uncovered.
        assert_eq!(g.coverage[1], 0.0);
        assert_eq!(g.mode[3], ModeClass::Unknown);
    }

    #[test]
    fn rasterize_area_weighted_split_cell() {
        // Two 8×4 blocks each covering half of cell (0,0): QP 20 and QP 40.
        let blocks = vec![
            block(0, 0, 8, 4, 20, 64, "Intra", None),
            block(0, 4, 8, 4, 40, 192, "Skip", None),
        ];
        let g = rasterize_blocks(&blocks, 8, 8, 8);
        assert_eq!((g.cols, g.rows), (1, 1));
        // Equal areas → mean QP = 30.
        assert!((g.qp[0] - 30.0).abs() < 1e-4);
        // (64 + 192) bits / 64 px = 4.0 bpp.
        assert!((g.bpp[0] - 4.0).abs() < 1e-4);
        assert!((g.coverage[0] - 1.0).abs() < 1e-5);
        // Equal areas tie: max_by keeps the later (Skip, index 3) since
        // strictly-greater comparison is required to replace. Either winner
        // is acceptable for a tie; pin current behaviour.
        assert!(matches!(g.mode[0], ModeClass::Intra | ModeClass::Skip));
    }

    #[test]
    fn rasterize_partial_coverage_below_one() {
        // A 4×4 block in an 8-px cell: coverage 16/64 = 0.25.
        let blocks = vec![block(0, 0, 4, 4, 24, 16, "Intra", None)];
        let g = rasterize_blocks(&blocks, 8, 8, 8);
        assert!((g.coverage[0] - 0.25).abs() < 1e-5);
        // Values are still means over the *covered* area.
        assert!((g.qp[0] - 24.0).abs() < 1e-5);
        assert!((g.bpp[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rasterize_block_spanning_cells_distributes_bits_by_area() {
        // A 16×8 block spanning two 8-px cells with 128 bits → 64 bits/cell,
        // 1.0 bpp each.
        let blocks = vec![block(0, 0, 16, 8, 30, 128, "Inter", None)];
        let g = rasterize_blocks(&blocks, 16, 8, 8);
        assert_eq!((g.cols, g.rows), (2, 1));
        for c in 0..2 {
            assert!((g.bpp[c] - 1.0).abs() < 1e-5);
            assert!((g.qp[c] - 30.0).abs() < 1e-5);
            assert!((g.coverage[c] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn rasterize_majority_mode_wins() {
        // Cell (0,0): 3/4 Intra, 1/4 Skip → Intra.
        let blocks = vec![
            block(0, 0, 8, 4, 20, 0, "Intra", None),
            block(0, 4, 4, 4, 20, 0, "Intra", None),
            block(4, 4, 4, 4, 20, 0, "Skip", None),
        ];
        let g = rasterize_blocks(&blocks, 8, 8, 8);
        assert_eq!(g.mode[0], ModeClass::Intra);
    }

    #[test]
    fn rasterize_lod_64px_aggregate() {
        // Same rasterizer at the LOD cell size: one 64×64 CTU covers the cell.
        let blocks = vec![block(0, 0, 64, 64, 36, 4096, "Inter", Some((8, 0)))];
        let g = rasterize_blocks(&blocks, 64, 64, 64);
        assert_eq!((g.cols, g.rows), (1, 1));
        assert!((g.bpp[0] - 1.0).abs() < 1e-5);
        assert!((g.qp[0] - 36.0).abs() < 1e-5);
    }

    #[test]
    fn lod_threshold_and_cell_size() {
        assert!(use_lod(1.0));
        assert!(use_lod(1.4999));
        assert!(!use_lod(1.5));
        assert!(!use_lod(3.0));
        assert_eq!(lod_cell_size("hevc"), 64);
        assert_eq!(lod_cell_size("HEVC"), 64);
        assert_eq!(lod_cell_size("h265"), 64);
        assert_eq!(lod_cell_size("avc"), 32);
        assert_eq!(lod_cell_size("av1"), 32);
    }

    #[test]
    fn hit_test_prefers_minimum_area() {
        // V12: an 8×8 CU nested inside a 64×64 CTU rect wins the hit test.
        let blocks = vec![
            block(0, 0, 64, 64, 30, 0, "Inter", None),
            block(8, 8, 8, 8, 40, 0, "Intra", None),
        ];
        assert_eq!(hit_test_min_area(&blocks, 10, 10), Some(1));
        assert_eq!(hit_test_min_area(&blocks, 40, 40), Some(0));
        assert_eq!(hit_test_min_area(&blocks, 64, 0), None);
        assert_eq!(hit_test_min_area(&blocks, 63, 63), Some(0));
    }

    #[test]
    fn viewer_offset_mapping() {
        // V5: offset +2 → viewer#0 reads catb display slot #2.
        assert_eq!(viewer_to_catb_display(0, 2), Some(2));
        assert_eq!(viewer_to_catb_display(5, 0), Some(5));
        // Negative offset: viewer frames before the telemetry map to None.
        assert_eq!(viewer_to_catb_display(0, -1), None);
        assert_eq!(viewer_to_catb_display(3, -3), Some(0));
    }

    #[test]
    fn frame_type_classification() {
        assert_eq!(FrameTypeClass::from_label("IDR"), FrameTypeClass::I);
        assert_eq!(FrameTypeClass::from_label("IDR_W_RADL"), FrameTypeClass::I);
        assert_eq!(FrameTypeClass::from_label("CRA"), FrameTypeClass::I);
        assert_eq!(FrameTypeClass::from_label("BLA_W_LP"), FrameTypeClass::I);
        assert_eq!(FrameTypeClass::from_label("I"), FrameTypeClass::I);
        assert_eq!(FrameTypeClass::from_label("KEY"), FrameTypeClass::I);
        assert_eq!(FrameTypeClass::from_label("p"), FrameTypeClass::P);
        assert_eq!(FrameTypeClass::from_label("B"), FrameTypeClass::B);
        assert_eq!(FrameTypeClass::from_label("RASL"), FrameTypeClass::Other);
        assert_eq!(FrameTypeClass::from_label(""), FrameTypeClass::Other);
    }

    #[test]
    fn hevc_intra_angle_anchors() {
        // 45°-spaced anchors of the uniform 2..34 span (M4 task spec).
        for (mode, deg) in [(2, 225.0), (10, 180.0), (18, 135.0), (26, 90.0), (34, 45.0)] {
            let a = intra_angle_hevc(mode).unwrap();
            assert!((a - deg).abs() < 1e-4, "mode {mode}: {a} != {deg}");
        }
        // Step between adjacent modes is 180/32 = 5.625°.
        let step = intra_angle_hevc(2).unwrap() - intra_angle_hevc(3).unwrap();
        assert!((step - 5.625).abs() < 1e-4);
        // Planar / DC / out-of-range: non-angular.
        assert_eq!(intra_angle_hevc(0), None);
        assert_eq!(intra_angle_hevc(1), None);
        assert_eq!(intra_angle_hevc(35), None);
        assert_eq!(intra_angle_hevc(-1), None);
    }

    #[test]
    fn avc_intra_angles() {
        // 4x4/8x8: vertical / horizontal / diagonals (H.264 Fig. 8-4).
        assert_eq!(intra_angle_avc_nxn(0), Some(90.0));
        assert_eq!(intra_angle_avc_nxn(1), Some(0.0));
        assert_eq!(intra_angle_avc_nxn(3), Some(45.0));
        assert_eq!(intra_angle_avc_nxn(4), Some(135.0));
        // DC and the decoder's edge-DC variants: non-angular.
        for m in [2, 9, 10, 11, 12, -1] {
            assert_eq!(intra_angle_avc_nxn(m), None, "mode {m}");
        }
        // 16x16 (FFmpeg enum order, oracle-confirmed): 1=H, 2=V; DC/Plane none.
        assert_eq!(intra_angle_avc_16x16(1), Some(0.0));
        assert_eq!(intra_angle_avc_16x16(2), Some(90.0));
        for m in [0, 3, 4, 5, 6] {
            assert_eq!(intra_angle_avc_16x16(m), None, "mode {m}");
        }
    }

    #[test]
    fn extract_intra_modes_maps_names_and_badges() {
        let row = |name: &str, value: &str| SyntaxRow {
            name: name.to_string(),
            value: value.to_string(),
            bits: 1,
        };
        let rows = vec![
            row("split_cu_flag", "1"),          // not an intra mode → skipped
            row("intra_luma_pred_mode", "0"),   // HEVC Planar
            row("intra_luma_pred_mode", "1"),   // HEVC DC
            row("intra_luma_pred_mode", "26"),  // HEVC vertical
            row("intra4x4_pred_mode", "9"),     // AVC Left-DC variant
            row("intra16x16_pred_mode", "3"),   // AVC Plane
            row("chroma_pred_mode", "2"),       // chroma → excluded
            row("intra_luma_pred_mode", "x"),   // unparsable → skipped
        ];
        let dirs = extract_intra_modes(&rows);
        assert_eq!(dirs.len(), 5);
        assert_eq!((dirs[0].angle_deg, dirs[0].badge), (None, "P"));
        assert_eq!((dirs[1].angle_deg, dirs[1].badge), (None, "DC"));
        assert_eq!((dirs[2].angle_deg, dirs[2].badge), (Some(90.0), ""));
        assert_eq!((dirs[3].angle_deg, dirs[3].badge), (None, "DC"));
        assert_eq!((dirs[4].angle_deg, dirs[4].badge), (None, "P"));
        // Sub-grid dims: 1 mode → 1×1, 4 → 2×2, 16 → 4×4.
        assert_eq!(intra_grid_dim(0), 1);
        assert_eq!(intra_grid_dim(1), 1);
        assert_eq!(intra_grid_dim(4), 2);
        assert_eq!(intra_grid_dim(16), 4);
    }

    #[test]
    fn intra_subblock_pos_zscan_to_raster() {
        // dim=4 (AVC 4x4): stored order is z-scan (H.264 §6.4.3). The top
        // 4×4 row of the MB is z-scan positions {0,1,4,5} — the pattern the
        // avc_cavlc fixture shows at frame row y=0 (LEFT_DC substitution).
        let full: Vec<_> = (0..16).map(|k| intra_subblock_pos(k, 4)).collect();
        assert_eq!(
            full,
            [
                // z-scan k → raster (col,row)
                (0, 0), (1, 0), (0, 1), (1, 1), // quadrant 0 (top-left)
                (2, 0), (3, 0), (2, 1), (3, 1), // quadrant 1 (top-right)
                (0, 2), (1, 2), (0, 3), (1, 3), // quadrant 2 (bottom-left)
                (2, 2), (3, 2), (2, 3), (3, 3), // quadrant 3 (bottom-right)
            ]
            .map(Some)
        );
        // Every cell covered exactly once (bijection).
        let mut seen: Vec<_> = full.into_iter().flatten().collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 16);
        // dim<=2: z-scan == raster, k used directly.
        assert_eq!(intra_subblock_pos(0, 1), Some((0, 0)));
        assert_eq!(intra_subblock_pos(3, 2), Some((1, 1)));
        // Out-of-grid entries are rejected, not wrapped.
        assert_eq!(intra_subblock_pos(1, 1), None);
        assert_eq!(intra_subblock_pos(4, 2), None);
        assert_eq!(intra_subblock_pos(16, 4), None);
    }

    #[test]
    fn nearest_poc_match_prefers_decode_distance() {
        // Two CVS segments with repeating POCs: 0,3,1,2 | 0,1 (decode order).
        let pocs = [0, 3, 1, 2, 0, 1];
        // From decode#5 (2nd segment), POC 0 must resolve to decode#4, not #0.
        assert_eq!(nearest_poc_match(&pocs, 5, 0), Some(4));
        // From decode#1 (1st segment), POC 0 → decode#0.
        assert_eq!(nearest_poc_match(&pocs, 1, 0), Some(0));
        // An IDR has no decode-order past: nothing to reference.
        assert_eq!(nearest_poc_match(&pocs, 0, 0), None);
        // Unknown POC → None.
        assert_eq!(nearest_poc_match(&pocs, 2, 9), None);
    }

    #[test]
    fn nearest_poc_match_never_future() {
        // multi-IDR: 0,1,2 | IDR,1,2 (decode order). From decode#2 (POC 2),
        // ref POC 0 must resolve backwards to decode#0, never to the next
        // segment's IDR at decode#3 (closer by distance, but not decoded yet).
        let pocs = [0, 1, 2, 0, 1, 2];
        assert_eq!(nearest_poc_match(&pocs, 2, 0), Some(0));
        assert_eq!(nearest_poc_match(&pocs, 2, 1), Some(1));
        // Second segment resolves within itself.
        assert_eq!(nearest_poc_match(&pocs, 5, 0), Some(3));
        // from_decode beyond the slice is clamped, not a panic.
        assert_eq!(nearest_poc_match(&pocs, 99, 2), Some(5));
    }

    #[test]
    fn ref_edges_bslice_pattern() {
        // hevc_bslice-like: decode IDR(0) P(3) B(1) B(2); display = 0,2,3,1.
        let pocs = [0, 3, 1, 2];
        let slice_refs = vec![
            vec![],                                   // IDR
            vec![(0, 0, false)],                      // P: L0 → POC 0
            vec![(0, 0, false), (3, 1, false)],       // B: L0 → 0, L1 → 3
            vec![(0, 0, false), (3, 1, true)],        // B: L1 long-term
        ];
        let display_map = vec![0, 2, 3, 1]; // POC sort: 0,1,2,3
        let (edges, counts) = derive_ref_edges(&pocs, &slice_refs, &display_map, 0, 4);
        // Viewer 0 = decode 0 (IDR): no refs.
        assert!(edges[0].is_empty());
        // Viewer 3 = decode 1 (P, POC 3): one L0 edge → viewer 0.
        assert_eq!(
            edges[3],
            vec![RefEdge { to: 0, list: 0, long_term: false }]
        );
        // Viewer 1 = decode 2 (B, POC 1): L0 → viewer 0, L1 → viewer 3.
        assert_eq!(
            edges[1],
            vec![
                RefEdge { to: 0, list: 0, long_term: false },
                RefEdge { to: 3, list: 1, long_term: false },
            ]
        );
        // Long-term flag survives (viewer 2 = decode 3).
        assert_eq!(edges[2][1], RefEdge { to: 3, list: 1, long_term: true });
        // Reference frequency: POC 0 used by 3 frames, POC 3 by 2.
        assert_eq!(counts, vec![3, 0, 0, 2]);
    }

    #[test]
    fn ref_edges_offset_and_range_clipping() {
        let pocs = [0, 1, 2];
        let slice_refs = vec![
            vec![],
            vec![(0, 0, false)],
            vec![(1, 0, false), (0, 0, false)],
        ];
        let display_map = vec![0, 1, 2];
        // Offset +1: viewer 0 → display 1 (decode 1). Its ref target POC 0 is
        // display 0 → viewer −1: outside the strip → edge dropped.
        let (edges, counts) = derive_ref_edges(&pocs, &slice_refs, &display_map, 1, 2);
        assert!(edges[0].is_empty());
        // Viewer 1 → decode 2: POC 1 → display 1 → viewer 0 kept; POC 0
        // clipped away.
        assert_eq!(edges[1], vec![RefEdge { to: 0, list: 0, long_term: false }]);
        assert_eq!(counts, vec![1, 0]);
        // Unknown ref POC never produces an edge.
        let bad_refs = vec![vec![(7, 0, false)], vec![]];
        let (e2, c2) = derive_ref_edges(&[0, 1], &bad_refs, &[0, 1], 0, 2);
        assert!(e2[0].is_empty() && e2[1].is_empty());
        assert_eq!(c2, vec![0, 0]);
    }

    #[test]
    fn ref_count_tiers() {
        assert_eq!(ref_count_tier(0), 0);
        assert_eq!(ref_count_tier(1), 1);
        assert_eq!(ref_count_tier(2), 2);
        assert_eq!(ref_count_tier(3), 2);
        assert_eq!(ref_count_tier(4), 3);
        assert_eq!(ref_count_tier(100), 3);
    }

    #[test]
    fn mode_classification() {
        assert_eq!(ModeClass::from_label("Intra"), ModeClass::Intra);
        assert_eq!(ModeClass::from_label("INTER"), ModeClass::Inter);
        assert_eq!(ModeClass::from_label("Inter (Skip)"), ModeClass::Skip);
        assert_eq!(ModeClass::from_label("merge"), ModeClass::Merge);
        assert_eq!(ModeClass::from_label(""), ModeClass::Unknown);
    }
}
