//! Integration tests for `analysis::metrics::compute_diff_stats`.
//! Verifies the four whole-frame diff scalars used in the Video Diff
//! header (avg(Y), var(Y), avg(MS), var(MS) where MS = (6Y+Cb+Cr)/8).

use video_viewer::analysis::metrics::{compute_diff_stats, DiffStats};

fn uniform(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
    let n = (w * h) as usize;
    let mut v = Vec::with_capacity(n * 3);
    for _ in 0..n {
        v.extend_from_slice(&[r, g, b]);
    }
    v
}

#[test]
fn test_diff_stats_zero_for_identical_frames() {
    let rgb = uniform(8, 8, 100, 150, 200);
    let s = compute_diff_stats(&rgb, &rgb, 8, 8).unwrap();
    assert!(s.avg_y.abs() < 1e-3);
    assert!(s.var_y.abs() < 1e-3);
    assert!(s.avg_ms.abs() < 1e-3);
    assert!(s.var_ms.abs() < 1e-3);
}

#[test]
fn test_diff_stats_returns_none_on_size_mismatch() {
    let r = uniform(4, 4, 0, 0, 0);
    let c = uniform(4, 5, 0, 0, 0);
    assert!(compute_diff_stats(&r, &c, 4, 4).is_none());
}

#[test]
fn test_diff_stats_returns_none_on_buffer_too_small() {
    let r = vec![0u8; 4]; // far below 4×4×3 = 48 bytes
    let c = vec![0u8; 4];
    assert!(compute_diff_stats(&r, &c, 4, 4).is_none());
}

#[test]
fn test_diff_stats_pure_blue_offset_carries_chroma_only() {
    // Reference: pure blue 200, current: pure blue 100, R/G unchanged.
    // dB = 100 → dY = 0.0722*100 ≈ 7.22, and chroma diff is dominant.
    let mut r = vec![0u8; 4 * 4 * 3];
    let mut c = vec![0u8; 4 * 4 * 3];
    for i in 0..16 {
        r[i * 3 + 2] = 200;
        c[i * 3 + 2] = 100;
    }
    let s = compute_diff_stats(&r, &c, 4, 4).unwrap();
    assert!((s.avg_y - 7.22).abs() < 0.05, "avg_y {}", s.avg_y);
    // avg_ms must differ from avg_y (chroma terms are non-zero).
    assert!(
        (s.avg_ms - s.avg_y).abs() > 0.5,
        "avg_ms {} too close to avg_y {}",
        s.avg_ms,
        s.avg_y
    );
    // Uniform offset → variance of both channels is zero.
    assert!(s.var_y.abs() < 1e-3);
    assert!(s.var_ms.abs() < 1e-3);
}

#[test]
fn test_diff_stats_achromatic_offset_zeroes_chroma_terms() {
    // Identical R, G, B offsets → dCb = dCr = 0 by construction.
    // Therefore avg_ms = (6 * avg_y + 0 + 0) / 8 = 0.75 * avg_y.
    let r = uniform(8, 8, 220, 220, 220);
    let c = uniform(8, 8, 80, 80, 80);
    let s = compute_diff_stats(&r, &c, 8, 8).unwrap();
    assert!((s.avg_y - 140.0).abs() < 0.05);
    let expected_ms = 0.75 * s.avg_y;
    assert!(
        (s.avg_ms - expected_ms).abs() < 0.05,
        "avg_ms {} expected {}",
        s.avg_ms,
        expected_ms
    );
}

#[test]
fn test_diff_stats_per_pixel_variation_yields_positive_variance() {
    // Two halves: left dY=0, right dY=80. avg_y ≈ 40, var_y ≈ 1600.
    let mut r = vec![0u8; 8 * 4 * 3];
    let c = vec![0u8; 8 * 4 * 3];
    for y in 0..4 {
        for x in 4..8 {
            let p = (y * 8 + x) * 3;
            r[p] = 80;
            r[p + 1] = 80;
            r[p + 2] = 80;
        }
    }
    let s = compute_diff_stats(&r, &c, 8, 4).unwrap();
    assert!((s.avg_y - 40.0).abs() < 0.5);
    assert!(
        (s.var_y - 1600.0).abs() < 20.0,
        "var_y {} expected ~1600",
        s.var_y
    );
}

#[test]
fn test_diff_stats_sign_distinguishes_brighter_vs_darker_reference() {
    // Reference brighter than current → positive avg_y; reverse → negative.
    let bright = uniform(4, 4, 200, 200, 200);
    let dark = uniform(4, 4, 80, 80, 80);
    let s_pos = compute_diff_stats(&bright, &dark, 4, 4).unwrap();
    let s_neg = compute_diff_stats(&dark, &bright, 4, 4).unwrap();
    assert!(s_pos.avg_y > 100.0);
    assert!(s_neg.avg_y < -100.0);
    // Variance is invariant under sign flip.
    assert!((s_pos.var_y - s_neg.var_y).abs() < 1e-3);
    assert!((s_pos.var_ms - s_neg.var_ms).abs() < 1e-3);
}

#[test]
fn test_diff_stats_ms_formula_holds_for_known_input() {
    // Synthesize a single-pixel-per-block delta with known YCbCr diff so
    // we can verify dMS = (6 dY + dCb + dCr) / 8 to within rounding.
    // R only: dR = 80, dG = dB = 0
    //   dY  = 0.2126 * 80  = 17.008
    //   dCb = -0.1146 * 80 ≈ -9.165
    //   dCr =  0.5    * 80 = 40.000
    //   dMS = (6 * 17.008 + (-9.165) + 40.0) / 8 = (102.048 + 30.835)/8 ≈ 16.610
    let mut r = vec![0u8; 4 * 4 * 3];
    let mut c = vec![0u8; 4 * 4 * 3];
    for i in 0..16 {
        r[i * 3] = 80;
        c[i * 3] = 0;
    }
    let s = compute_diff_stats(&r, &c, 4, 4).unwrap();
    let expected_dy = 0.2126 * 80.0;
    let expected_dcb = -0.2126 / (2.0 * (1.0 - 0.0722)) * 80.0;
    let expected_dcr = 0.5 * 80.0;
    let expected_dms = (6.0 * expected_dy + expected_dcb + expected_dcr) / 8.0;
    assert!((s.avg_y as f64 - expected_dy).abs() < 0.05);
    assert!(
        (s.avg_ms as f64 - expected_dms).abs() < 0.1,
        "avg_ms {} expected {}",
        s.avg_ms,
        expected_dms
    );
}

#[test]
fn test_diff_stats_struct_layout_is_stable() {
    // Pin the public DiffStats shape so downstream code (the comparison view)
    // can rely on the four named fields. If you intentionally rename or add
    // fields, update this test alongside.
    let s = DiffStats {
        avg_y: 1.0,
        var_y: 2.0,
        avg_ms: 3.0,
        var_ms: 4.0,
    };
    assert_eq!(s.avg_y, 1.0);
    assert_eq!(s.var_y, 2.0);
    assert_eq!(s.avg_ms, 3.0);
    assert_eq!(s.var_ms, 4.0);
}
