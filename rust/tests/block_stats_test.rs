//! Integration tests for `analysis::block_stats`. These exercise the public
//! API as it appears to consumers (the Block tab in the analysis viewport).

use video_viewer::analysis::block_stats::{compute_block_stats, BlockMetric, BlockStats};

/// Build a uniform RGB buffer of dimensions `(w × h)` with the given colour.
fn uniform(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
    let n = (w * h) as usize;
    let mut v = Vec::with_capacity(n * 3);
    for _ in 0..n {
        v.extend_from_slice(&[r, g, b]);
    }
    v
}

/// Vertical bars: every `period` columns the colour alternates between
/// `colour_a` and `colour_b`.
fn vertical_bars(
    w: u32,
    h: u32,
    period: u32,
    colour_a: [u8; 3],
    colour_b: [u8; 3],
) -> Vec<u8> {
    let mut v = vec![0u8; (w * h) as usize * 3];
    for y in 0..h {
        for x in 0..w {
            let p = ((y * w + x) as usize) * 3;
            let c = if (x / period).is_multiple_of(2) {
                colour_a
            } else {
                colour_b
            };
            v[p] = c[0];
            v[p + 1] = c[1];
            v[p + 2] = c[2];
        }
    }
    v
}

#[test]
fn test_block_stats_default_block_size_32_grid_for_full_hd() {
    // 1920×1080 default grid → 60 × 34 (rows is ceil(1080/32) = 34).
    let rgb = uniform(1920, 1080, 0, 0, 0);
    let s = compute_block_stats(&rgb, 1920, 1080, 32, BlockMetric::Y);
    assert_eq!(s.cols, 60);
    assert_eq!(s.rows, 34);
    assert_eq!(s.means.len() as u32, s.cols * s.rows);
    assert_eq!(s.vars.len(), s.means.len());
}

#[test]
fn test_block_stats_block_size_32_with_uniform_image() {
    let rgb = uniform(64, 64, 200, 200, 200);
    let s = compute_block_stats(&rgb, 64, 64, 32, BlockMetric::Y);
    assert_eq!(s.cols, 2);
    assert_eq!(s.rows, 2);
    for m in &s.means {
        assert!((m - 200.0).abs() < 0.5);
    }
    for v in &s.vars {
        assert!(*v < 1e-3);
    }
}

#[test]
fn test_block_stats_vertical_bars_isolated_to_left_right() {
    // 32×16 image with 16-pixel period → left half black, right half mid-grey.
    // Block size 16 → 2×1 grid. Left block dark, right block mid-grey.
    let rgb = vertical_bars(32, 16, 16, [0, 0, 0], [128, 128, 128]);
    let s = compute_block_stats(&rgb, 32, 16, 16, BlockMetric::Y);
    assert_eq!(s.cols, 2);
    assert_eq!(s.rows, 1);
    assert!(s.means[0] < 1.0);
    assert!((s.means[1] - 128.0).abs() < 0.5);
}

#[test]
fn test_block_stats_variance_inside_block_with_alternating_pixels() {
    // 4×4 alternating pixel checkerboard, block_size = 4 → single block
    // with mean 127.5 and variance 127.5² = 16256.25.
    let mut rgb = vec![0u8; 4 * 4 * 3];
    for y in 0..4 {
        for x in 0..4 {
            if (x + y) % 2 == 0 {
                let p = (y * 4 + x) * 3;
                rgb[p] = 255;
                rgb[p + 1] = 255;
                rgb[p + 2] = 255;
            }
        }
    }
    let s = compute_block_stats(&rgb, 4, 4, 4, BlockMetric::Y);
    assert!((s.means[0] - 127.5).abs() < 0.5);
    assert!((s.vars[0] - 16256.25).abs() < 5.0);
}

#[test]
fn test_block_stats_block_size_larger_than_image() {
    // 8×8 image with block_size 64: should produce a single 1×1 block
    // covering the whole image.
    let rgb = uniform(8, 8, 60, 60, 60);
    let s = compute_block_stats(&rgb, 8, 8, 64, BlockMetric::Y);
    assert_eq!(s.cols, 1);
    assert_eq!(s.rows, 1);
    assert!((s.means[0] - 60.0).abs() < 0.5);
    assert!(s.vars[0] < 1e-3);
}

#[test]
fn test_block_stats_max_var_returns_zero_for_uniform_image() {
    let rgb = uniform(16, 16, 100, 100, 100);
    let s = compute_block_stats(&rgb, 16, 16, 8, BlockMetric::Y);
    assert!(s.max_var() < 1e-3);
}

#[test]
fn test_block_stats_empty_helper() {
    let e = BlockStats::empty();
    assert_eq!(e.cols, 0);
    assert_eq!(e.rows, 0);
    assert!(e.means.is_empty());
    assert_eq!(e.max_var(), 0.0);
}

#[test]
fn test_block_stats_ms_metric_matches_y_for_achromatic_input() {
    // Achromatic R=G=B → dCb = dCr = 0, so MS = (6Y + 0 + 0)/8 = 0.75 Y.
    let rgb = uniform(8, 8, 200, 200, 200);
    let s_y = compute_block_stats(&rgb, 8, 8, 8, BlockMetric::Y);
    let s_ms = compute_block_stats(&rgb, 8, 8, 8, BlockMetric::Ms);
    // Y mean = 200; MS mean ≈ 0.75 * 200 = 150.
    assert!((s_y.frame_mean - 200.0).abs() < 0.5, "y mean {}", s_y.frame_mean);
    assert!((s_ms.frame_mean - 150.0).abs() < 0.5, "ms mean {}", s_ms.frame_mean);
    // Both metrics record the chosen kind.
    assert_eq!(s_y.metric, BlockMetric::Y);
    assert_eq!(s_ms.metric, BlockMetric::Ms);
}

#[test]
fn test_block_stats_ms_metric_diverges_from_y_for_chromatic_input() {
    // Pure blue 200 → Y = 0.0722*200 ≈ 14.44; MS adds positive Cb (≈+50)
    // and small negative Cr, ending around 22.2 — so Y and MS diverge by
    // ~7.8 here, large enough to verify the MS code path is being taken.
    let rgb = uniform(8, 8, 0, 0, 200);
    let s_y = compute_block_stats(&rgb, 8, 8, 8, BlockMetric::Y);
    let s_ms = compute_block_stats(&rgb, 8, 8, 8, BlockMetric::Ms);
    assert!(
        (s_y.frame_mean - s_ms.frame_mean).abs() > 5.0,
        "Y mean {} vs MS mean {} should differ for chromatic input",
        s_y.frame_mean, s_ms.frame_mean
    );
}

#[test]
fn test_block_stats_horizontal_bands() {
    // 8×16 image with two horizontal bands (top dark, bottom bright).
    // Block size 8 → 1×2 grid: top block dark, bottom block bright.
    let mut rgb = vec![0u8; 8 * 16 * 3];
    for y in 8..16 {
        for x in 0..8 {
            let p = (y * 8 + x) * 3;
            rgb[p] = 200;
            rgb[p + 1] = 200;
            rgb[p + 2] = 200;
        }
    }
    let s = compute_block_stats(&rgb, 8, 16, 8, BlockMetric::Y);
    assert_eq!(s.cols, 1);
    assert_eq!(s.rows, 2);
    assert!(s.means[0] < 1.0, "top mean {}", s.means[0]);
    assert!((s.means[1] - 200.0).abs() < 0.5, "bottom mean {}", s.means[1]);
}
