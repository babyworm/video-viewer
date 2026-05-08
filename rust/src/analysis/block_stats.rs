//! Per-block luma statistics for spatial structure visualization.
//!
//! Computes the mean and variance of luma for fixed-size square blocks
//! tiling the image. Two scalar metrics are supported:
//!
//! - [`BlockMetric::Y`]  — BT.709 luma (Y' = 0.2126 R + 0.7152 G + 0.0722 B).
//! - [`BlockMetric::Ms`] — `(6Y + Cb + Cr) / 8`, the chroma-aware perceptual
//!   weighting also used by the Video Diff diff_stats readout.
//!
//! Both metrics are 8-bit-range scalars so the heatmap colourisation in the
//! Block tab can use the same min/max bounds either way.

/// Which scalar to compute per pixel before aggregating block statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockMetric {
    /// BT.709 luma: `Y = 0.2126 R + 0.7152 G + 0.0722 B`.
    Y,
    /// Chroma-aware perceptual weighting: `(6 Y + Cb + Cr) / 8`,
    /// matching the comparison-view diff_stats MS definition.
    Ms,
}

impl BlockMetric {
    pub fn label(self) -> &'static str {
        match self {
            BlockMetric::Y => "Y",
            BlockMetric::Ms => "MS",
        }
    }
}

/// Per-block luma statistics over an image.
#[derive(Debug, Clone)]
pub struct BlockStats {
    pub block_size: u32,
    /// Which scalar metric was aggregated (Y or MS).
    pub metric: BlockMetric,
    /// Number of block columns: ceil(width / block_size).
    pub cols: u32,
    /// Number of block rows: ceil(height / block_size).
    pub rows: u32,
    /// Block means in row-major order. Range [0.0, 255.0] (both metrics).
    pub means: Vec<f32>,
    /// Block variances in row-major order. Non-negative.
    pub vars: Vec<f32>,
    /// Source image width in pixels (for rendering aspect).
    pub width: u32,
    /// Source image height in pixels.
    pub height: u32,
    /// Whole-frame minimum of the chosen metric. 0 when image empty.
    pub frame_min: f32,
    /// Whole-frame maximum of the chosen metric.
    pub frame_max: f32,
    /// Whole-frame mean of the chosen metric.
    pub frame_mean: f32,
    /// Whole-frame variance of the chosen metric (E[X²] − E[X]²).
    pub frame_var: f32,
}

impl BlockStats {
    pub fn empty() -> Self {
        Self {
            block_size: 32,
            metric: BlockMetric::Y,
            cols: 0,
            rows: 0,
            means: Vec::new(),
            vars: Vec::new(),
            width: 0,
            height: 0,
            frame_min: 0.0,
            frame_max: 0.0,
            frame_mean: 0.0,
            frame_var: 0.0,
        }
    }

    /// Maximum variance across all blocks; 0 if empty.
    pub fn max_var(&self) -> f32 {
        self.vars.iter().copied().fold(0.0_f32, f32::max)
    }
}

/// BT.709 luma coefficient applied to an 8-bit RGB pixel.
#[inline]
fn rgb_to_luma(r: u8, g: u8, b: u8) -> f64 {
    0.2126 * r as f64 + 0.7152 * g as f64 + 0.0722 * b as f64
}

/// BT.709 YCbCr conversion → MS = (6Y + Cb + Cr) / 8. Cb/Cr are returned in
/// the centred range [-128, 127] before mixing, so MS sits roughly in [0, 255]
/// for natural images.
#[inline]
fn rgb_to_ms(r: u8, g: u8, b: u8) -> f64 {
    const KR: f64 = 0.2126;
    const KG: f64 = 0.7152;
    const KB: f64 = 0.0722;
    let r = r as f64;
    let g = g as f64;
    let b = b as f64;
    let y = KR * r + KG * g + KB * b;
    let cb = -KR / (2.0 * (1.0 - KB)) * r
        - KG / (2.0 * (1.0 - KB)) * g
        + (1.0 - KB) / (2.0 * (1.0 - KB)) * b;
    let cr = (1.0 - KR) / (2.0 * (1.0 - KR)) * r
        - KG / (2.0 * (1.0 - KR)) * g
        - KB / (2.0 * (1.0 - KR)) * b;
    (6.0 * y + cb + cr) / 8.0
}

#[inline]
fn pixel_value(metric: BlockMetric, r: u8, g: u8, b: u8) -> f64 {
    match metric {
        BlockMetric::Y => rgb_to_luma(r, g, b),
        BlockMetric::Ms => rgb_to_ms(r, g, b),
    }
}

/// Compute per-block mean and variance of the chosen `metric`.
///
/// Returns `BlockStats` describing each `block_size × block_size` tile of
/// the image (right/bottom edges may be partial). Uses single-pass running
/// sums of X and X² and computes variance as `E[X²] − (E[X])²`.
pub fn compute_block_stats(
    rgb: &[u8],
    width: u32,
    height: u32,
    block_size: u32,
    metric: BlockMetric,
) -> BlockStats {
    let bs = block_size.max(1);
    if width == 0 || height == 0 {
        return BlockStats::empty();
    }
    let cols = width.div_ceil(bs);
    let rows = height.div_ceil(bs);
    let n_blocks = (cols as usize) * (rows as usize);
    let mut sum_y = vec![0.0_f64; n_blocks];
    let mut sum_y2 = vec![0.0_f64; n_blocks];
    let mut counts = vec![0_u32; n_blocks];

    let stride = (width as usize) * 3;
    let need = stride * height as usize;
    if rgb.len() < need {
        return BlockStats::empty();
    }

    // Whole-frame accumulators — computed in the same pass for free.
    let mut frame_min = f64::INFINITY;
    let mut frame_max = f64::NEG_INFINITY;
    let mut frame_sum = 0.0_f64;
    let mut frame_sum_sq = 0.0_f64;

    for y in 0..height {
        let by = y / bs;
        let row_off = (y as usize) * stride;
        for x in 0..width {
            let bx = x / bs;
            let bidx = (by * cols + bx) as usize;
            let pi = row_off + (x as usize) * 3;
            let yv = pixel_value(metric, rgb[pi], rgb[pi + 1], rgb[pi + 2]);
            sum_y[bidx] += yv;
            sum_y2[bidx] += yv * yv;
            counts[bidx] += 1;
            if yv < frame_min { frame_min = yv; }
            if yv > frame_max { frame_max = yv; }
            frame_sum += yv;
            frame_sum_sq += yv * yv;
        }
    }

    let means: Vec<f32> = (0..n_blocks)
        .map(|i| {
            if counts[i] == 0 {
                0.0
            } else {
                (sum_y[i] / counts[i] as f64) as f32
            }
        })
        .collect();
    let vars: Vec<f32> = (0..n_blocks)
        .map(|i| {
            if counts[i] == 0 {
                0.0
            } else {
                let n = counts[i] as f64;
                let mean = sum_y[i] / n;
                ((sum_y2[i] / n) - mean * mean).max(0.0) as f32
            }
        })
        .collect();

    let n_pixels = (width as u64) * (height as u64);
    let (frame_mean, frame_var, frame_min_out, frame_max_out) = if n_pixels == 0 {
        (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32)
    } else {
        let n = n_pixels as f64;
        let m = frame_sum / n;
        let v = ((frame_sum_sq / n) - m * m).max(0.0);
        (m as f32, v as f32, frame_min as f32, frame_max as f32)
    };

    BlockStats {
        block_size: bs,
        metric,
        cols,
        rows,
        means,
        vars,
        width,
        height,
        frame_min: frame_min_out,
        frame_max: frame_max_out,
        frame_mean,
        frame_var,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a uniform RGB image of `(r, g, b)`.
    fn uniform(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let n = (w * h) as usize;
        let mut v = Vec::with_capacity(n * 3);
        for _ in 0..n {
            v.extend_from_slice(&[r, g, b]);
        }
        v
    }

    #[test]
    fn empty_image_yields_empty_stats() {
        let s = compute_block_stats(&[], 0, 0, 32, BlockMetric::Y);
        assert_eq!(s.cols, 0);
        assert_eq!(s.rows, 0);
        assert!(s.means.is_empty());
        // Frame-wide stats are zeros for an empty image.
        assert_eq!(s.frame_min, 0.0);
        assert_eq!(s.frame_max, 0.0);
        assert_eq!(s.frame_mean, 0.0);
        assert_eq!(s.frame_var, 0.0);
    }

    #[test]
    fn frame_stats_match_uniform_image() {
        let rgb = uniform(8, 8, 100, 100, 100);
        let s = compute_block_stats(&rgb, 8, 8, 8, BlockMetric::Y);
        // Achromatic 100 → BT.709 luma == 100.
        assert!((s.frame_min - 100.0).abs() < 0.5);
        assert!((s.frame_max - 100.0).abs() < 0.5);
        assert!((s.frame_mean - 100.0).abs() < 0.5);
        assert!(s.frame_var.abs() < 1e-3);
    }

    #[test]
    fn frame_var_matches_two_value_split() {
        // Half pixels at Y=0, half at Y=255 → variance = (255/2)² = 16256.25.
        let mut rgb = vec![0u8; 16 * 8 * 3];
        for y in 0..8 {
            for x in 0..8 {
                let p = (y * 16 + 8 + x) * 3;
                rgb[p] = 255;
                rgb[p + 1] = 255;
                rgb[p + 2] = 255;
            }
        }
        let s = compute_block_stats(&rgb, 16, 8, 8, BlockMetric::Y);
        assert!((s.frame_min - 0.0).abs() < 0.5);
        assert!((s.frame_max - 255.0).abs() < 0.5);
        assert!((s.frame_mean - 127.5).abs() < 0.5);
        assert!((s.frame_var - 16256.25).abs() < 5.0, "frame_var={}", s.frame_var);
    }

    #[test]
    fn uniform_image_has_zero_variance() {
        let rgb = uniform(32, 32, 128, 128, 128);
        let s = compute_block_stats(&rgb, 32, 32, 32, BlockMetric::Y);
        assert_eq!(s.cols, 1);
        assert_eq!(s.rows, 1);
        // Y = 0.2126*128 + 0.7152*128 + 0.0722*128 = 128
        assert!((s.means[0] - 128.0).abs() < 0.5);
        assert!(s.vars[0] < 1e-3);
    }

    #[test]
    fn block_grid_dimensions_round_up() {
        let rgb = uniform(33, 33, 0, 0, 0);
        let s = compute_block_stats(&rgb, 33, 33, 32, BlockMetric::Y);
        assert_eq!(s.cols, 2);
        assert_eq!(s.rows, 2);
        assert_eq!(s.means.len(), 4);
    }

    #[test]
    fn means_recover_block_brightness() {
        // 16×16 image with two halves: left = black, right = white.
        let mut rgb = vec![0u8; 16 * 16 * 3];
        for y in 0..16 {
            for x in 8..16 {
                let p = (y * 16 + x) * 3;
                rgb[p] = 255;
                rgb[p + 1] = 255;
                rgb[p + 2] = 255;
            }
        }
        let s = compute_block_stats(&rgb, 16, 16, 8, BlockMetric::Y);
        assert_eq!(s.cols, 2);
        assert_eq!(s.rows, 2);
        // Left blocks: ~0, right blocks: ~255
        assert!(s.means[0] < 1.0);
        assert!((s.means[1] - 255.0).abs() < 1.0);
        assert!(s.means[2] < 1.0);
        assert!((s.means[3] - 255.0).abs() < 1.0);
    }

    #[test]
    fn bt709_weights_are_asymmetric_per_channel() {
        // A pure red, green, blue input must produce luma equal to the BT.709
        // coefficient ×255. This pins the implementation to BT.709 specifically
        // — BT.601 (e.g. 0.587 for green) would fail this test.
        let red = uniform(8, 8, 255, 0, 0);
        let green = uniform(8, 8, 0, 255, 0);
        let blue = uniform(8, 8, 0, 0, 255);

        let r = compute_block_stats(&red, 8, 8, 8, BlockMetric::Y).means[0];
        let g = compute_block_stats(&green, 8, 8, 8, BlockMetric::Y).means[0];
        let b = compute_block_stats(&blue, 8, 8, 8, BlockMetric::Y).means[0];

        assert!((r - 0.2126 * 255.0).abs() < 0.5, "red Y: {}", r);
        assert!((g - 0.7152 * 255.0).abs() < 0.5, "green Y: {}", g);
        assert!((b - 0.0722 * 255.0).abs() < 0.5, "blue Y: {}", b);
        // BT.709 weights should not be the BT.601 weights.
        assert!((g - 0.587 * 255.0).abs() > 5.0, "green should be BT.709");
    }

    #[test]
    fn checkerboard_variance_matches_analytical_value() {
        // 8×8 luma checkerboard (black / white) at block_size=8.
        // Half pixels at Y=0, half at Y=255 → mean = 127.5,
        // variance = E[Y²] − (E[Y])² = 127.5² = 16256.25.
        let mut rgb = vec![0u8; 8 * 8 * 3];
        for y in 0_usize..8 {
            for x in 0_usize..8 {
                if (x + y).is_multiple_of(2) {
                    let p = (y * 8 + x) * 3;
                    rgb[p] = 255;
                    rgb[p + 1] = 255;
                    rgb[p + 2] = 255;
                }
            }
        }
        let s = compute_block_stats(&rgb, 8, 8, 8, BlockMetric::Y);
        assert_eq!(s.cols, 1);
        assert_eq!(s.rows, 1);
        let mean = s.means[0];
        let var = s.vars[0];
        assert!((mean - 127.5).abs() < 0.5, "mean {}", mean);
        // Allow ±5 absolute tolerance on the variance (rounded BT.709 luma).
        assert!((var - 16256.25).abs() < 5.0, "variance {}", var);
    }

    #[test]
    fn partial_edge_blocks_are_handled() {
        // 12×7 image with block_size 8 → 2×1 grid; the bottom-right block
        // is 4 wide and 7 tall (clipped). Set the right column to mid-grey
        // so the partial block has a non-trivial mean.
        let mut rgb = vec![0u8; 12 * 7 * 3];
        for y in 0..7 {
            for x in 8..12 {
                let p = (y * 12 + x) * 3;
                rgb[p] = 128;
                rgb[p + 1] = 128;
                rgb[p + 2] = 128;
            }
        }
        let s = compute_block_stats(&rgb, 12, 7, 8, BlockMetric::Y);
        assert_eq!(s.cols, 2);
        assert_eq!(s.rows, 1);
        assert!(s.means[0] < 1.0, "left full block should be black");
        assert!((s.means[1] - 128.0).abs() < 0.5, "right partial: {}", s.means[1]);
        // Both blocks are uniform inside their footprint → zero variance.
        assert!(s.vars[0] < 1e-3 && s.vars[1] < 1e-3);
    }

    #[test]
    fn block_size_one_produces_per_pixel_stats() {
        // block_size = 1 means each pixel is its own block; variance must be 0.
        let rgb = uniform(4, 4, 200, 50, 100);
        let s = compute_block_stats(&rgb, 4, 4, 1, BlockMetric::Y);
        assert_eq!(s.cols, 4);
        assert_eq!(s.rows, 4);
        assert_eq!(s.means.len(), 16);
        for v in &s.vars {
            assert!(*v < 1e-6);
        }
        // Each block's mean equals the BT.709 luma of the constant pixel.
        let expected = 0.2126 * 200.0 + 0.7152 * 50.0 + 0.0722 * 100.0;
        for m in &s.means {
            assert!((*m as f64 - expected).abs() < 0.5);
        }
    }

    #[test]
    fn block_size_zero_is_clamped_to_one() {
        // We never want to divide by zero; bs = 0 should fall back to bs = 1.
        let rgb = uniform(4, 4, 0, 0, 0);
        let s = compute_block_stats(&rgb, 4, 4, 0, BlockMetric::Y);
        assert_eq!(s.block_size, 1);
        assert_eq!(s.means.len(), 16);
    }

    #[test]
    fn max_var_returns_largest_block_variance() {
        // Two blocks: one uniform (variance 0), one black/white split
        // (variance > 0). max_var must return the larger one.
        let mut rgb = vec![128u8; 16 * 8 * 3];
        for y in 0..8 {
            for x in 0..4 {
                let p = (y * 16 + 8 + x) * 3;
                rgb[p] = 0;
                rgb[p + 1] = 0;
                rgb[p + 2] = 0;
            }
            for x in 4..8 {
                let p = (y * 16 + 8 + x) * 3;
                rgb[p] = 255;
                rgb[p + 1] = 255;
                rgb[p + 2] = 255;
            }
        }
        let s = compute_block_stats(&rgb, 16, 8, 8, BlockMetric::Y);
        let mv = s.max_var();
        assert!(mv > 1000.0, "expected non-trivial max_var, got {}", mv);
        assert_eq!(mv, s.vars.iter().copied().fold(0.0_f32, f32::max));
    }

    #[test]
    fn truncated_buffer_returns_empty_stats() {
        // RGB buffer too small for the declared dimensions: must fall back
        // to empty stats instead of indexing past the slice.
        let too_small = vec![0u8; 10]; // declared 8×8 = 192 bytes
        let s = compute_block_stats(&too_small, 8, 8, 8, BlockMetric::Y);
        assert_eq!(s.cols, 0);
        assert_eq!(s.rows, 0);
        assert!(s.means.is_empty());
    }
}
