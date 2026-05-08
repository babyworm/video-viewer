//! Per-block luma statistics for spatial structure visualization.
//!
//! Computes the mean and variance of luma (BT.709) for fixed-size square
//! blocks tiling the image. Used by the Frame Analysis "Block" tab to show
//! brightness distribution and texture/noise distribution side-by-side.

/// Per-block luma statistics over an image.
#[derive(Debug, Clone)]
pub struct BlockStats {
    pub block_size: u32,
    /// Number of block columns: ceil(width / block_size).
    pub cols: u32,
    /// Number of block rows: ceil(height / block_size).
    pub rows: u32,
    /// Block luma means in row-major order. Range [0.0, 255.0].
    pub means: Vec<f32>,
    /// Block luma variances in row-major order. Non-negative.
    pub vars: Vec<f32>,
    /// Source image width in pixels (for rendering aspect).
    pub width: u32,
    /// Source image height in pixels.
    pub height: u32,
}

impl BlockStats {
    pub fn empty() -> Self {
        Self {
            block_size: 32,
            cols: 0,
            rows: 0,
            means: Vec::new(),
            vars: Vec::new(),
            width: 0,
            height: 0,
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

/// Compute per-block luma mean and variance.
///
/// Returns `BlockStats` describing each `block_size × block_size` tile of
/// the image (right/bottom edges may be partial). Uses single-pass running
/// sums of Y and Y² and computes variance as `E[Y²] − (E[Y])²`.
pub fn compute_block_stats(
    rgb: &[u8],
    width: u32,
    height: u32,
    block_size: u32,
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

    for y in 0..height {
        let by = y / bs;
        let row_off = (y as usize) * stride;
        for x in 0..width {
            let bx = x / bs;
            let bidx = (by * cols + bx) as usize;
            let pi = row_off + (x as usize) * 3;
            let yv = rgb_to_luma(rgb[pi], rgb[pi + 1], rgb[pi + 2]);
            sum_y[bidx] += yv;
            sum_y2[bidx] += yv * yv;
            counts[bidx] += 1;
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

    BlockStats {
        block_size: bs,
        cols,
        rows,
        means,
        vars,
        width,
        height,
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
        let s = compute_block_stats(&[], 0, 0, 32);
        assert_eq!(s.cols, 0);
        assert_eq!(s.rows, 0);
        assert!(s.means.is_empty());
    }

    #[test]
    fn uniform_image_has_zero_variance() {
        let rgb = uniform(32, 32, 128, 128, 128);
        let s = compute_block_stats(&rgb, 32, 32, 32);
        assert_eq!(s.cols, 1);
        assert_eq!(s.rows, 1);
        // Y = 0.2126*128 + 0.7152*128 + 0.0722*128 = 128
        assert!((s.means[0] - 128.0).abs() < 0.5);
        assert!(s.vars[0] < 1e-3);
    }

    #[test]
    fn block_grid_dimensions_round_up() {
        let rgb = uniform(33, 33, 0, 0, 0);
        let s = compute_block_stats(&rgb, 33, 33, 32);
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
        let s = compute_block_stats(&rgb, 16, 16, 8);
        assert_eq!(s.cols, 2);
        assert_eq!(s.rows, 2);
        // Left blocks: ~0, right blocks: ~255
        assert!(s.means[0] < 1.0);
        assert!((s.means[1] - 255.0).abs() < 1.0);
        assert!(s.means[2] < 1.0);
        assert!((s.means[3] - 255.0).abs() < 1.0);
    }
}
