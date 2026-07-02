//! Per-block dominant gradient orientation (02 §3) — the analysis-side
//! metric behind the future intra-direction correlation.
//!
//! Pipeline: BT.709 luma (shared [`pixel_value`] path) → 3×3 Sobel gx/gy →
//! per-pixel **edge orientation** → magnitude-weighted 16-bin histogram per
//! block → dominant angle + purity (winning-bin weight fraction).
//!
//! **Angle convention**: the stored angle is the *edge* orientation, i.e.
//! the direction **perpendicular to the luma gradient**, in degrees within
//! `[0, 180)`. 0° = a horizontal edge (luma varies vertically, gy-dominant),
//! 90° = a vertical edge (luma varies horizontally, gx-dominant). This is
//! the direction an angular intra predictor would extrapolate along, which
//! is what the HEVC mode-angle mapping (M4) needs.

use crate::analysis::block_stats::{pixel_value, BlockMetric};

/// Histogram bins over [0°, 180°): 11.25° per bin.
pub const ORIENTATION_BINS: usize = 16;

/// Gradient magnitudes below this (8-bit luma units) are treated as flat and
/// excluded from the histogram — sensor noise must not fabricate structure.
const MAG_EPSILON: f64 = 1.0;

/// Per-block dominant orientation over a frame. Same row-major
/// `{block_size, cols, rows, Vec<f32>}` shape as BlockStats/MotionStats, so
/// it plugs straight into the correlation resamplers.
#[derive(Debug, Clone)]
pub struct OrientationStats {
    pub block_size: u32,
    /// ceil(width / block_size).
    pub cols: u32,
    /// ceil(height / block_size).
    pub rows: u32,
    /// Dominant edge orientation per block, degrees in [0, 180). 0 when the
    /// block is flat (see `purity`).
    pub angles: Vec<f32>,
    /// Winning-bin weight / total weight ∈ [0, 1]; 0 for flat blocks —
    /// always gate on purity before trusting an angle.
    pub purity: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl OrientationStats {
    pub fn empty() -> Self {
        Self {
            block_size: 8,
            cols: 0,
            rows: 0,
            angles: Vec::new(),
            purity: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.cols == 0 || self.rows == 0
    }
}

/// Compute per-block dominant edge orientation + purity.
///
/// `rgb` is tightly packed 8-bit RGB. Frame-border pixels (where the 3×3
/// Sobel window is incomplete) are excluded from the histograms. Returns
/// [`OrientationStats::empty`] for degenerate input.
pub fn compute_orientation_stats(
    rgb: &[u8],
    width: u32,
    height: u32,
    block_size: u32,
) -> OrientationStats {
    let bs = block_size.max(1);
    if width == 0 || height == 0 {
        return OrientationStats::empty();
    }
    let w = width as usize;
    let h = height as usize;
    if rgb.len() < w * h * 3 {
        return OrientationStats::empty();
    }

    // Luma plane once — Sobel reads each pixel up to 8 times.
    let mut luma = vec![0.0_f64; w * h];
    for (i, l) in luma.iter_mut().enumerate() {
        let p = i * 3;
        *l = pixel_value(BlockMetric::Y, rgb[p], rgb[p + 1], rgb[p + 2]);
    }

    let cols = width.div_ceil(bs);
    let rows = height.div_ceil(bs);
    let n_blocks = (cols as usize) * (rows as usize);
    // Per-block, per-bin: (weight, weight·angle) so the dominant angle is the
    // magnitude-weighted mean of the winning bin, not just its centre.
    let mut hist_w = vec![[0.0_f64; ORIENTATION_BINS]; n_blocks];
    let mut hist_wa = vec![[0.0_f64; ORIENTATION_BINS]; n_blocks];
    let mut total_w = vec![0.0_f64; n_blocks];

    for y in 1..h.saturating_sub(1) {
        let by = (y as u32) / bs;
        for x in 1..w - 1 {
            let bx = (x as u32) / bs;
            let bidx = (by as usize) * (cols as usize) + bx as usize;
            let l = |dx: isize, dy: isize| {
                luma[(y as isize + dy) as usize * w + (x as isize + dx) as usize]
            };
            // Sobel: gx = ∂/∂x (columns), gy = ∂/∂y (rows).
            let gx = (l(1, -1) + 2.0 * l(1, 0) + l(1, 1))
                - (l(-1, -1) + 2.0 * l(-1, 0) + l(-1, 1));
            let gy = (l(-1, 1) + 2.0 * l(0, 1) + l(1, 1))
                - (l(-1, -1) + 2.0 * l(0, -1) + l(1, -1));
            let mag = (gx * gx + gy * gy).sqrt();
            if mag < MAG_EPSILON {
                continue;
            }
            // Edge orientation = gradient angle + 90°, folded into [0, 180).
            let angle = (gy.atan2(gx).to_degrees() + 90.0).rem_euclid(180.0);
            let bin = ((angle / 180.0 * ORIENTATION_BINS as f64) as usize)
                .min(ORIENTATION_BINS - 1);
            hist_w[bidx][bin] += mag;
            hist_wa[bidx][bin] += mag * angle;
            total_w[bidx] += mag;
        }
    }

    let mut angles = vec![0.0_f32; n_blocks];
    let mut purity = vec![0.0_f32; n_blocks];
    for i in 0..n_blocks {
        if total_w[i] <= 0.0 {
            continue; // flat block: angle 0, purity 0
        }
        let (best_bin, best_w) = hist_w[i]
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0));
        if best_w > 0.0 {
            angles[i] = (hist_wa[i][best_bin] / best_w) as f32;
            purity[i] = (best_w / total_w[i]) as f32;
        }
    }

    OrientationStats {
        block_size: bs,
        cols,
        rows,
        angles,
        purity,
        width,
        height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Grey image from a per-pixel luma function.
    fn grey_image<F: Fn(usize, usize) -> u8>(w: usize, h: usize, f: F) -> Vec<u8> {
        let mut rgb = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let v = f(x, y);
                rgb.extend_from_slice(&[v, v, v]);
            }
        }
        rgb
    }

    #[test]
    fn horizontal_edge_reports_zero_degrees() {
        // Top half dark, bottom half bright: luma varies with y only →
        // gradient is vertical → edge orientation 0° (horizontal edge).
        let rgb = grey_image(16, 16, |_, y| if y < 8 { 0 } else { 255 });
        let s = compute_orientation_stats(&rgb, 16, 16, 16);
        assert_eq!((s.cols, s.rows), (1, 1));
        let a = s.angles[0];
        // 0 and 180 are the same orientation.
        assert!(!(3.0..=177.0).contains(&a), "angle={a}");
        assert!(s.purity[0] > 0.9, "purity={}", s.purity[0]);
    }

    #[test]
    fn vertical_edge_reports_ninety_degrees() {
        let rgb = grey_image(16, 16, |x, _| if x < 8 { 0 } else { 255 });
        let s = compute_orientation_stats(&rgb, 16, 16, 16);
        assert!((s.angles[0] - 90.0).abs() < 3.0, "angle={}", s.angles[0]);
        assert!(s.purity[0] > 0.9);
    }

    #[test]
    fn diagonal_edge_reports_forty_five_degrees() {
        // Bright above-right of the x = y line: the edge runs along 45°.
        let rgb = grey_image(16, 16, |x, y| if x > y { 255 } else { 0 });
        let s = compute_orientation_stats(&rgb, 16, 16, 16);
        assert!((s.angles[0] - 45.0).abs() < 8.0, "angle={}", s.angles[0]);
        assert!(s.purity[0] > 0.5, "purity={}", s.purity[0]);
    }

    #[test]
    fn flat_block_has_zero_purity() {
        let rgb = grey_image(16, 16, |_, _| 128);
        let s = compute_orientation_stats(&rgb, 16, 16, 8);
        assert_eq!(s.angles.len(), 4);
        for i in 0..4 {
            assert_eq!(s.purity[i], 0.0, "flat block {i} must have purity 0");
            assert_eq!(s.angles[i], 0.0);
        }
    }

    #[test]
    fn noise_block_has_lower_purity_than_edge_block() {
        // Left 8px column of blocks: clean vertical edge; right: noise.
        let mut seed = 0xdead_beef_u32;
        let mut noise = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 24) as u8
        };
        let mut vals = [0u8; 16 * 8];
        for y in 0..8 {
            for x in 0..16 {
                vals[y * 16 + x] = if x < 8 {
                    if x < 4 { 0 } else { 255 } // vertical edge at x=4
                } else {
                    noise()
                };
            }
        }
        let rgb = grey_image(16, 8, |x, y| vals[y * 16 + x]);
        let s = compute_orientation_stats(&rgb, 16, 8, 8);
        assert_eq!((s.cols, s.rows), (2, 1));
        assert!(
            s.purity[0] > s.purity[1],
            "edge purity {} should exceed noise purity {}",
            s.purity[0],
            s.purity[1]
        );
        assert!((s.angles[0] - 90.0).abs() < 6.0);
    }

    #[test]
    fn partial_blocks_are_computed_without_panic() {
        // 20×12 with block 16 → 2×1 grid; right block is 4px wide (partial).
        let rgb = grey_image(20, 12, |x, _| if x < 10 { 0 } else { 255 });
        let s = compute_orientation_stats(&rgb, 20, 12, 16);
        assert_eq!((s.cols, s.rows), (2, 1));
        assert_eq!(s.angles.len(), 2);
        // The edge at x=10 lives in block 0 → vertical edge ≈ 90°.
        assert!((s.angles[0] - 90.0).abs() < 6.0);
    }

    #[test]
    fn degenerate_inputs_yield_empty() {
        assert!(compute_orientation_stats(&[], 0, 0, 8).is_empty());
        // Truncated buffer.
        let small = vec![0u8; 10];
        assert!(compute_orientation_stats(&small, 8, 8, 8).is_empty());
    }
}
