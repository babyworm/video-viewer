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

use crate::core::catb::{BsBlock, CatbFile};

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
    /// `display_map[display_idx] = decode_idx`. Only `output == true` frames
    /// appear; built CVS-segment by CVS-segment (POC resets at IDR/BLA).
    pub display_map: Vec<usize>,
    pub width: u32,
    pub height: u32,
    pub resolution_source: ResolutionSource,
    cache: Mutex<LruCache<usize, Arc<Vec<BsBlock>>>>,
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
        let catb = CatbFile::open(path)?;
        let display_map = build_display_map(&catb);
        let (width, height, resolution_source) = resolve_resolution(&catb)?;
        Ok(Self {
            catb,
            display_map,
            width,
            height,
            resolution_source,
            cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(BLOCK_CACHE_FRAMES).expect("nonzero cache size"),
            )),
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
/// `width`×`height` stream, area-weighting every value.
pub fn rasterize_blocks(blocks: &[BsBlock], width: u32, height: u32, cell: u32) -> BitstreamGrid {
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
            coverage: Vec::new(),
        };
    }

    let mut area = vec![0.0_f64; n];
    let mut qp_acc = vec![0.0_f64; n];
    let mut bits_acc = vec![0.0_f64; n];
    let mut mv_acc = vec![0.0_f64; n];
    // Per-cell covered area per mode class (5 classes).
    let mut mode_area = vec![[0.0_f32; 5]; n];

    for b in blocks {
        if b.w == 0 || b.h == 0 {
            continue;
        }
        let block_area = (b.w as f64) * (b.h as f64);
        let bits_per_px = b.bits.max(0) as f64 / block_area;
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
                mode_area[idx][mode.index()] += a as f32;
            }
        }
    }

    let full_cell = (cell as f64) * (cell as f64);
    let mut qp = vec![0.0_f32; n];
    let mut bpp = vec![0.0_f32; n];
    let mut mv = vec![0.0_f32; n];
    let mut mode = vec![ModeClass::Unknown; n];
    let mut coverage = vec![0.0_f32; n];
    for i in 0..n {
        if area[i] > 0.0 {
            qp[i] = (qp_acc[i] / area[i]) as f32;
            bpp[i] = (bits_acc[i] / area[i]) as f32;
            mv[i] = (mv_acc[i] / area[i]) as f32;
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
    fn mode_classification() {
        assert_eq!(ModeClass::from_label("Intra"), ModeClass::Intra);
        assert_eq!(ModeClass::from_label("INTER"), ModeClass::Inter);
        assert_eq!(ModeClass::from_label("Inter (Skip)"), ModeClass::Skip);
        assert_eq!(ModeClass::from_label("merge"), ModeClass::Merge);
        assert_eq!(ModeClass::from_label(""), ModeClass::Unknown);
    }
}
