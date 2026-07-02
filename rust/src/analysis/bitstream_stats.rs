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
                    if v > 0 {
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
    } else {
        Ok(Some((max_w, max_h)))
    }
}
