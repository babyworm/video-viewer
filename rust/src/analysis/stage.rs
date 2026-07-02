//! Decoder-run stage images + block PSNR/SSIM (M-C, parity plan G10/G11/
//! G15/D6 — VQAnalyzer "Reconstruction/YUV pictures" + "Info Overlays
//! PSNR/SSIM" analogue).
//!
//! Stage images are BMPs the external `codec-analyzer decoder-run` writes
//! next to the `.catb` (observed: `frameNNNNNN-<stage>.bmp`, 24-bit
//! bottom-up, stream resolution). The `.catb` meta's
//! `frames_meta[i].stage_images` object maps stage key → path (observed:
//! bare file names, i.e. relative to the .catb directory).
//!
//! Quality metrics are computed on **BT.601 luma** (0.299 R + 0.587 G +
//! 0.114 B): codec-side PSNR/SSIM is conventionally reported on the Y
//! plane, and the SD-class fixture streams carry BT.601 content — full-RGB
//! MSE would triple-count chroma upsampling error instead. For grey
//! content (R=G=B) the block PSNR is bit-identical to
//! `metrics::calculate_psnr` (differential-tested).

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lru::LruCache;

use crate::analysis::bitstream_stats::BitstreamFile;
use crate::core::bmp::{load_bmp, BmpImage};
use crate::core::catb::BsBlock;

// ---------------------------------------------------------------------------
// Stage kinds + path resolution
// ---------------------------------------------------------------------------

/// One decoder pipeline stage image (M-A observed `stage_images` keys).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageKind {
    FinalRecon,
    ReconUnfiltered,
    Prediction,
    Residual,
}

impl StageKind {
    pub const ALL: [StageKind; 4] = [
        StageKind::FinalRecon,
        StageKind::ReconUnfiltered,
        StageKind::Prediction,
        StageKind::Residual,
    ];

    /// `frames_meta[i].stage_images` object key.
    pub fn key(self) -> &'static str {
        match self {
            StageKind::FinalRecon => "final_recon",
            StageKind::ReconUnfiltered => "recon_unfiltered",
            StageKind::Prediction => "prediction",
            StageKind::Residual => "residual",
        }
    }

    /// Toolbar label (VQA "Pic" combo analogue).
    pub fn label(self) -> &'static str {
        match self {
            StageKind::FinalRecon => "Final recon",
            StageKind::ReconUnfiltered => "Recon unfiltered",
            StageKind::Prediction => "Prediction",
            StageKind::Residual => "Residual",
        }
    }
}

/// The stage-image path string recorded for `decode_idx`, if any.
pub fn stage_rel_path(file: &BitstreamFile, decode_idx: usize, kind: StageKind) -> Option<&str> {
    let meta = file.catb.meta.frames_meta.get(decode_idx)?;
    meta.stage_images
        .iter()
        .find(|(k, _)| k == kind.key())
        .map(|(_, v)| v.as_str())
}

/// Resolve a recorded stage-image path against the `.catb` location:
/// 1. absolute path → itself (if it exists)
/// 2. relative → joined onto the .catb directory (observed layout)
/// 3. fallback: bare file name searched next to the .catb (sidecar — the
///    run directory may have been moved)
pub fn resolve_stage_path(catb_path: &Path, recorded: &str) -> Option<PathBuf> {
    if recorded.is_empty() {
        return None;
    }
    let p = Path::new(recorded);
    if p.is_absolute() {
        if p.is_file() {
            return Some(p.to_path_buf());
        }
    } else if let Some(dir) = catb_path.parent() {
        let joined = dir.join(p);
        if joined.is_file() {
            return Some(joined);
        }
    }
    // Sidecar fallback: basename next to the .catb.
    let name = p.file_name()?;
    let sidecar = catb_path.parent()?.join(name);
    if sidecar.is_file() && Path::new(recorded) != sidecar {
        return Some(sidecar);
    }
    None
}

/// True when a loadable stage image is recorded for `decode_idx` (path
/// resolution only — no pixel I/O; drives the Picture combo enable state).
pub fn stage_available(file: &BitstreamFile, decode_idx: usize, kind: StageKind) -> bool {
    stage_rel_path(file, decode_idx, kind)
        .and_then(|rel| resolve_stage_path(&file.path, rel))
        .is_some()
}

// ---------------------------------------------------------------------------
// Lazy per-frame LRU cache
// ---------------------------------------------------------------------------

/// Frames × stages kept decoded: 4 stages of a handful of frames — stage
/// BMPs are stream-sized RGB (~6 MB each at 1080p).
const STAGE_CACHE_ENTRIES: usize = 16;

/// Lazy stage-image loader with a small LRU, keyed by file identity
/// (`Arc::as_ptr`) + decode index + stage. Load failures are cached as
/// `None` so a missing BMP is stat'ed once, not on every repaint.
pub struct StageCache {
    file_ptr: usize,
    lru: LruCache<(usize, StageKind), Option<Arc<BmpImage>>>,
}

impl Default for StageCache {
    fn default() -> Self {
        Self::new()
    }
}

impl StageCache {
    pub fn new() -> Self {
        Self {
            file_ptr: 0,
            lru: LruCache::new(NonZeroUsize::new(STAGE_CACHE_ENTRIES).expect("nonzero")),
        }
    }

    /// The stage image for `decode_idx`, loading it on first use. `None`
    /// when the frame has no recorded/resolvable/parsable image. A `.catb`
    /// loaded over another one resets the cache (pointer-key convention).
    pub fn get(
        &mut self,
        file: &Arc<BitstreamFile>,
        decode_idx: usize,
        kind: StageKind,
    ) -> Option<Arc<BmpImage>> {
        let ptr = Arc::as_ptr(file) as usize;
        if ptr != self.file_ptr {
            self.lru.clear();
            self.file_ptr = ptr;
        }
        if let Some(hit) = self.lru.get(&(decode_idx, kind)) {
            return hit.clone();
        }
        let loaded = stage_rel_path(file, decode_idx, kind)
            .and_then(|rel| resolve_stage_path(&file.path, rel))
            .and_then(|path| match load_bmp(&path) {
                Ok(img) => Some(Arc::new(img)),
                Err(e) => {
                    log::warn!("stage image load failed: {e}");
                    None
                }
            });
        self.lru.put((decode_idx, kind), loaded.clone());
        loaded
    }
}

// ---------------------------------------------------------------------------
// Luma + block PSNR / SSIM
// ---------------------------------------------------------------------------

/// BT.601 luma plane of a tightly-packed RGB8 buffer (see module docs for
/// why BT.601 and why luma).
pub fn luma_bt601(rgb: &[u8], pixels: usize) -> Vec<f32> {
    let n = pixels.min(rgb.len() / 3);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        out.push(0.299 * r + 0.587 * g + 0.114 * b);
    }
    out
}

/// PSNR of one rectangle of two equally-sized luma planes.
/// `f32::INFINITY` on a bit-exact match (rendered as `e`, VQA convention).
pub fn region_psnr(
    a: &[f32],
    b: &[f32],
    width: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> f32 {
    let mut se = 0.0_f64;
    let mut n = 0u64;
    for row in y..y + h {
        for col in x..x + w {
            let i = (row as usize) * (width as usize) + col as usize;
            if let (Some(va), Some(vb)) = (a.get(i), b.get(i)) {
                let d = (*va - *vb) as f64;
                se += d * d;
                n += 1;
            }
        }
    }
    if n == 0 {
        return 0.0;
    }
    let mse = se / n as f64;
    if mse == 0.0 {
        f32::INFINITY
    } else {
        (10.0 * (255.0_f64 * 255.0 / mse).log10()) as f32
    }
}

/// Global-statistics SSIM of one rectangle (the standard single-window
/// form: means, variances, covariance over the whole block — the same
/// formula `metrics::calculate_ssim` falls back to for images smaller than
/// its 11×11 window; per-block that degenerate form *is* the block SSIM).
pub fn region_ssim(
    a: &[f32],
    b: &[f32],
    width: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> f32 {
    const C1: f64 = 6.5025; // (0.01·255)²
    const C2: f64 = 58.5225; // (0.03·255)²
    let (mut s1, mut s2, mut s11, mut s22, mut s12) = (0.0_f64, 0.0, 0.0, 0.0, 0.0);
    let mut n = 0.0_f64;
    for row in y..y + h {
        for col in x..x + w {
            let i = (row as usize) * (width as usize) + col as usize;
            if let (Some(&va), Some(&vb)) = (a.get(i), b.get(i)) {
                let (va, vb) = (va as f64, vb as f64);
                s1 += va;
                s2 += vb;
                s11 += va * va;
                s22 += vb * vb;
                s12 += va * vb;
                n += 1.0;
            }
        }
    }
    if n == 0.0 {
        return 0.0;
    }
    let mu1 = s1 / n;
    let mu2 = s2 / n;
    let var1 = (s11 / n - mu1 * mu1).max(0.0);
    let var2 = (s22 / n - mu2 * mu2).max(0.0);
    let cov = s12 / n - mu1 * mu2;
    let num = (2.0 * mu1 * mu2 + C1) * (2.0 * cov + C2);
    let den = (mu1 * mu1 + mu2 * mu2 + C1) * (var1 + var2 + C2);
    (num / den) as f32
}

/// Why per-block quality cannot be computed for the current frame — shown
/// verbatim in the legend slot (D: the fill stays selectable, the reason
/// replaces the scale).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualityUnavailable {
    NoViewerFrame,
    NoStageImage,
    ResolutionMismatch {
        stream: (u32, u32),
        viewer: (u32, u32),
    },
}

impl QualityUnavailable {
    pub fn message(&self) -> String {
        match self {
            QualityUnavailable::NoViewerFrame => {
                "PSNR/SSIM: open the source video in the main window".to_string()
            }
            QualityUnavailable::NoStageImage => {
                "PSNR/SSIM: stage images not found next to the .catb — re-run decoder-run"
                    .to_string()
            }
            QualityUnavailable::ResolutionMismatch { stream, viewer } => format!(
                "PSNR/SSIM: resolution mismatch — recon {}x{} vs viewer {}x{}",
                stream.0, stream.1, viewer.0, viewer.1
            ),
        }
    }
}

/// Per-L0-block PSNR/SSIM of `final_recon` vs the viewer's current frame,
/// parallel to the block list. Infinite PSNR (bit-exact block) is kept as
/// `f32::INFINITY`; the legend range uses the finite min/max.
#[derive(Debug, Clone)]
pub struct BlockQuality {
    pub psnr: Vec<f32>,
    pub ssim: Vec<f32>,
    /// Finite PSNR range (0,0 when every block is bit-exact).
    pub psnr_min: f32,
    pub psnr_max: f32,
    pub ssim_min: f32,
    pub ssim_max: f32,
}

/// Compute per-block quality. `viewer_luma`/`recon_luma` must both be
/// `width × height` planes (the caller verifies the resolutions match).
pub fn compute_block_quality(
    blocks: &[BsBlock],
    viewer_luma: &[f32],
    recon_luma: &[f32],
    width: u32,
    height: u32,
) -> BlockQuality {
    let mut psnr = Vec::with_capacity(blocks.len());
    let mut ssim = Vec::with_capacity(blocks.len());
    let (mut pmin, mut pmax) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut smin, mut smax) = (f32::INFINITY, f32::NEG_INFINITY);
    for b in blocks {
        // Clamp to the frame: edge blocks may extend past the picture.
        let x = b.x.min(width);
        let y = b.y.min(height);
        let w = b.w.min(width.saturating_sub(x));
        let h = b.h.min(height.saturating_sub(y));
        let p = region_psnr(viewer_luma, recon_luma, width, x, y, w, h);
        let s = region_ssim(viewer_luma, recon_luma, width, x, y, w, h);
        if p.is_finite() {
            pmin = pmin.min(p);
            pmax = pmax.max(p);
        }
        smin = smin.min(s);
        smax = smax.max(s);
        psnr.push(p);
        ssim.push(s);
    }
    if !pmin.is_finite() || !pmax.is_finite() {
        (pmin, pmax) = (0.0, 0.0);
    }
    if !smin.is_finite() || !smax.is_finite() {
        (smin, smax) = (0.0, 0.0);
    }
    BlockQuality {
        psnr,
        ssim,
        psnr_min: pmin,
        psnr_max: pmax,
        ssim_min: smin,
        ssim_max: smax,
    }
}

// ---------------------------------------------------------------------------
// PSNR at the correlation G grid (ReconPsnr Y metric)
// ---------------------------------------------------------------------------

/// Cap applied to bit-exact G cells so a mixed frame stays usable for
/// Pearson/Spearman (∞ would poison both). 99 dB sits above any real lossy
/// cell (8-bit MSE ≥ 1/g² ⇒ PSNR ≤ ~85 dB at g=64).
pub const PSNR_CAP_DB: f32 = 99.0;

/// Per-G-cell luma PSNR grid (row-major, ceil(w/g) × ceil(h/g) — the same
/// cell layout as `aggregate_bitstream_to_g`). Bit-exact cells are capped
/// at [`PSNR_CAP_DB`] instead of ∞.
pub fn psnr_to_g(
    viewer_luma: &[f32],
    recon_luma: &[f32],
    width: u32,
    height: u32,
    g: u32,
) -> Vec<f32> {
    let g = g.max(1);
    let cols = width.div_ceil(g);
    let rows = height.div_ceil(g);
    let mut out = Vec::with_capacity((cols as usize) * (rows as usize));
    for r in 0..rows {
        for c in 0..cols {
            let x = c * g;
            let y = r * g;
            let w = g.min(width - x);
            let h = g.min(height - y);
            let p = region_psnr(viewer_luma, recon_luma, width, x, y, w, h);
            out.push(if p.is_finite() { p } else { PSNR_CAP_DB });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::metrics::{calculate_psnr, calculate_ssim};
    use std::io::Write;

    fn gray_rgb(vals: &[u8]) -> Vec<u8> {
        vals.iter().flat_map(|&v| [v, v, v]).collect()
    }

    // -- path resolution ----------------------------------------------------

    #[test]
    fn resolve_absolute_relative_sidecar_missing() {
        let dir = tempfile::tempdir().unwrap();
        let catb = dir.path().join("run.catb");
        std::fs::File::create(&catb).unwrap();
        let bmp = dir.path().join("frame000000-final_recon.bmp");
        std::fs::File::create(&bmp).unwrap().write_all(b"BM").unwrap();

        // 1. absolute
        let abs = bmp.to_string_lossy().into_owned();
        assert_eq!(resolve_stage_path(&catb, &abs), Some(bmp.clone()));
        // 2. relative to the catb directory
        assert_eq!(
            resolve_stage_path(&catb, "frame000000-final_recon.bmp"),
            Some(bmp.clone())
        );
        // 3. sidecar: recorded path points elsewhere, basename exists here
        assert_eq!(
            resolve_stage_path(&catb, "/nonexistent/run/frame000000-final_recon.bmp"),
            Some(bmp.clone())
        );
        assert_eq!(
            resolve_stage_path(&catb, "old/subdir/frame000000-final_recon.bmp"),
            Some(bmp)
        );
        // 4. missing → None
        assert_eq!(resolve_stage_path(&catb, "frame000001-residual.bmp"), None);
        assert_eq!(resolve_stage_path(&catb, ""), None);
    }

    // -- block PSNR vs metrics.rs (differential) ------------------------------

    #[test]
    fn full_frame_block_psnr_matches_metrics_on_gray() {
        // Grey content: R=G=B ⇒ RGB-MSE PSNR == luma PSNR for any luma
        // weights, so the two implementations must agree exactly.
        let w = 8u32;
        let h = 6u32;
        let a: Vec<u8> = (0..(w * h) as usize).map(|i| (i * 5 % 256) as u8).collect();
        let b: Vec<u8> = a.iter().map(|v| v.wrapping_add(3)).collect();
        let (rgb_a, rgb_b) = (gray_rgb(&a), gray_rgb(&b));
        let expected = calculate_psnr(&rgb_a, &rgb_b, w, h);
        let la = luma_bt601(&rgb_a, (w * h) as usize);
        let lb = luma_bt601(&rgb_b, (w * h) as usize);
        let got = region_psnr(&la, &lb, w, 0, 0, w, h);
        assert!(
            (got as f64 - expected).abs() < 1e-3,
            "block {got} vs metrics {expected}"
        );
    }

    #[test]
    fn full_frame_block_ssim_matches_metrics_on_small_gray() {
        // Images below the 11×11 window make calculate_ssim use its global
        // form — the same statistic region_ssim computes per block.
        let w = 8u32;
        let h = 8u32;
        let a: Vec<u8> = (0..(w * h) as usize).map(|i| (i * 3 % 200) as u8).collect();
        let b: Vec<u8> = a.iter().map(|v| v.wrapping_add(7)).collect();
        let (rgb_a, rgb_b) = (gray_rgb(&a), gray_rgb(&b));
        let expected = calculate_ssim(&rgb_a, &rgb_b, w, h);
        let la = luma_bt601(&rgb_a, (w * h) as usize);
        let lb = luma_bt601(&rgb_b, (w * h) as usize);
        let got = region_ssim(&la, &lb, w, 0, 0, w, h);
        assert!(
            (got as f64 - expected).abs() < 1e-6,
            "block {got} vs metrics {expected}"
        );
    }

    #[test]
    fn identical_block_is_infinite_psnr() {
        let l: Vec<f32> = (0..64).map(|i| i as f32).collect();
        assert_eq!(region_psnr(&l, &l, 8, 0, 0, 8, 8), f32::INFINITY);
        // SSIM of identical content is 1.
        assert!((region_ssim(&l, &l, 8, 0, 0, 8, 8) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn block_quality_ranges_and_partial_blocks() {
        // 8x4 frame, two 4x4 blocks: left identical (∞), right differs.
        let w = 8u32;
        let h = 4u32;
        let a: Vec<f32> = vec![100.0; (w * h) as usize];
        let mut b = a.clone();
        for row in 0..h as usize {
            for col in 4..8 {
                b[row * w as usize + col] = 110.0;
            }
        }
        let blocks = vec![
            BsBlock { x: 0, y: 0, w: 4, h: 4, ..block_zero() },
            BsBlock { x: 4, y: 0, w: 4, h: 4, ..block_zero() },
            // Edge block extending past the frame: must clamp, not panic.
            BsBlock { x: 6, y: 2, w: 4, h: 4, ..block_zero() },
        ];
        let q = compute_block_quality(&blocks, &a, &b, w, h);
        assert_eq!(q.psnr[0], f32::INFINITY);
        assert!(q.psnr[1].is_finite() && q.psnr[1] > 0.0);
        // Legend range excludes the infinite block.
        assert_eq!(q.psnr_min, q.psnr.iter().copied().filter(|p| p.is_finite()).fold(f32::INFINITY, f32::min));
        assert!(q.psnr_max.is_finite());
        assert!(q.ssim[0] > q.ssim[1]);
    }

    #[test]
    fn psnr_to_g_caps_identical_cells() {
        let w = 16u32;
        let h = 8u32;
        let a: Vec<f32> = (0..(w * h) as usize).map(|i| (i % 251) as f32).collect();
        let mut b = a.clone();
        b[0] += 20.0; // only the first 8x8 cell differs
        let grid = psnr_to_g(&a, &b, w, h, 8);
        assert_eq!(grid.len(), 2);
        assert!(grid[0] < PSNR_CAP_DB);
        assert_eq!(grid[1], PSNR_CAP_DB);
    }

    fn block_zero() -> BsBlock {
        BsBlock {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
            ctu_address: 0,
            qp: 0,
            partition: String::new(),
            prediction_mode: String::new(),
            bits: 0,
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
            mv_flags: 0,
            mvp: None,
            mvd: None,
            mv: None,
            reference: None,
            reference_list: None,
            reference_label: None,
            reference_list_index: None,
            reference_poc: None,
            reference_frame: None,
            reference_long_term: None,
        }
    }
}
