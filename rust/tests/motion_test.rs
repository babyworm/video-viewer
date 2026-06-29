//! Integration tests for `analysis::motion` — per-block inter-frame motion
//! classification as consumed by the Motion tab in the analysis viewport.

use video_viewer::analysis::block_stats::BlockMetric;
use video_viewer::analysis::motion::{
    compute_motion_stats, MotionClass, MotionMethod, MotionThresholds,
};

/// Uniform achromatic RGB buffer (R=G=B=`v`) so BT.709 luma == `v`.
fn uniform(w: u32, h: u32, v: u8) -> Vec<u8> {
    vec![v; (w * h) as usize * 3]
}

/// 8×8 achromatic checkerboard alternating 0 / 255 by `(x+y)` parity.
/// `invert` flips which parity is white. Mean=127.5, std=127.5 either way.
fn checker8(invert: bool) -> Vec<u8> {
    let mut v = vec![0u8; 8 * 8 * 3];
    for y in 0..8usize {
        for x in 0..8usize {
            let white = ((x + y) % 2 == 0) ^ invert;
            if white {
                let p = (y * 8 + x) * 3;
                v[p] = 255;
                v[p + 1] = 255;
                v[p + 2] = 255;
            }
        }
    }
    v
}

const DEF: MotionThresholds = MotionThresholds {
    slight: MotionThresholds::DEFAULT_SLIGHT,
    much: MotionThresholds::DEFAULT_MUCH,
    full: MotionThresholds::DEFAULT_FULL,
};

#[test]
fn identical_frames_have_no_motion() {
    let f = uniform(32, 32, 100);
    let s = compute_motion_stats(&f, &f, 32, 32, 8, BlockMetric::Y, MotionMethod::PixelDiff, DEF);
    assert!(!s.is_empty());
    assert_eq!(s.cols, 4);
    assert_eq!(s.rows, 4);
    assert_eq!(s.classes.len(), 16);
    assert!(s.classes.iter().all(|&c| c == MotionClass::None));
    assert!(s.frame_score.abs() < 1e-3, "frame_score {}", s.frame_score);
    assert_eq!(s.class_counts, [16, 0, 0, 0]);
}

#[test]
fn pixel_diff_and_stats_diff_disagree_on_rearranged_block() {
    // The decisive test: a checkerboard vs its inverse. Every pixel flips
    // 0↔255 (PixelDiff = 255 → Full) but the mean (127.5) and std (127.5)
    // are unchanged (StatsDiff = 0 → None).
    let prev = checker8(false);
    let cur = checker8(true);

    let pd = compute_motion_stats(&prev, &cur, 8, 8, 8, BlockMetric::Y, MotionMethod::PixelDiff, DEF);
    assert_eq!(pd.cols, 1);
    assert_eq!(pd.rows, 1);
    assert!((pd.scores[0] - 255.0).abs() < 0.5, "pixel-diff score {}", pd.scores[0]);
    assert_eq!(pd.classes[0], MotionClass::Full);

    let sd = compute_motion_stats(&prev, &cur, 8, 8, 8, BlockMetric::Y, MotionMethod::StatsDiff, DEF);
    assert!(sd.scores[0] < 1e-3, "stats-diff score {} should be ~0", sd.scores[0]);
    assert_eq!(sd.classes[0], MotionClass::None);
}

#[test]
fn uniform_brightness_shift_scores_equal_for_both_methods() {
    // A flat brightness step changes the mean but not the spread, so PixelDiff
    // (per-pixel |Δ|) and StatsDiff (|Δmean| + |Δstd|, Δstd=0) must agree.
    let prev = uniform(16, 16, 100);
    let cur = uniform(16, 16, 120); // +20 luma everywhere

    let pd = compute_motion_stats(&prev, &cur, 16, 16, 16, BlockMetric::Y, MotionMethod::PixelDiff, DEF);
    let sd = compute_motion_stats(&prev, &cur, 16, 16, 16, BlockMetric::Y, MotionMethod::StatsDiff, DEF);
    assert!((pd.scores[0] - 20.0).abs() < 0.2, "pixel {}", pd.scores[0]);
    assert!((sd.scores[0] - 20.0).abs() < 0.2, "stats {}", sd.scores[0]);
    assert_eq!(pd.classes[0], MotionClass::Much); // 20 ∈ [10,30)
    assert_eq!(sd.classes[0], MotionClass::Much);
}

#[test]
fn classification_bands_cover_all_four_classes() {
    // PixelDiff on uniform shifts lands the score exactly on the shift amount.
    let cases = [
        (0u8, MotionClass::None),    // 0   < 2
        (5u8, MotionClass::Slight),  // 5   ∈ [2,10)
        (20u8, MotionClass::Much),   // 20  ∈ [10,30)
        (60u8, MotionClass::Full),   // 60  ≥ 30
    ];
    let base = uniform(8, 8, 30);
    for (shift, expected) in cases {
        let cur = uniform(8, 8, 30 + shift);
        let s = compute_motion_stats(&base, &cur, 8, 8, 8, BlockMetric::Y, MotionMethod::PixelDiff, DEF);
        assert_eq!(s.classes[0], expected, "shift {} → {:?}", shift, s.classes[0]);
        assert!((s.scores[0] - shift as f32).abs() < 0.2, "shift {} score {}", shift, s.scores[0]);
    }
}

#[test]
fn motion_is_localized_to_changed_block() {
    // 16×16, block 8 → 2×2 grid. Change only the top-left block.
    let prev = uniform(16, 16, 50);
    let mut cur = uniform(16, 16, 50);
    for y in 0..8 {
        for x in 0..8 {
            let p = (y * 16 + x) * 3;
            cur[p] = 200;
            cur[p + 1] = 200;
            cur[p + 2] = 200;
        }
    }
    let s = compute_motion_stats(&prev, &cur, 16, 16, 8, BlockMetric::Y, MotionMethod::PixelDiff, DEF);
    assert_eq!(s.classes[0], MotionClass::Full); // |200-50| = 150
    assert_eq!(s.classes[1], MotionClass::None);
    assert_eq!(s.classes[2], MotionClass::None);
    assert_eq!(s.classes[3], MotionClass::None);
    assert_eq!(s.class_counts[MotionClass::None.index()], 3);
    assert_eq!(s.class_counts[MotionClass::Full.index()], 1);
}

#[test]
fn class_counts_sum_to_block_total() {
    let prev = uniform(40, 24, 80);
    let cur = uniform(40, 24, 130);
    let s = compute_motion_stats(&prev, &cur, 40, 24, 8, BlockMetric::Y, MotionMethod::PixelDiff, DEF);
    let total: u32 = s.class_counts.iter().sum();
    assert_eq!(total, s.cols * s.rows);
    assert_eq!(total as usize, s.classes.len());
}

#[test]
fn frame_score_is_mean_and_max_is_peak() {
    // Two blocks side by side: left unchanged, right shifted by 40.
    let prev = uniform(16, 8, 60);
    let mut cur = uniform(16, 8, 60);
    for y in 0..8 {
        for x in 8..16 {
            let p = (y * 16 + x) * 3;
            cur[p] = 100;
            cur[p + 1] = 100;
            cur[p + 2] = 100;
        }
    }
    let s = compute_motion_stats(&prev, &cur, 16, 8, 8, BlockMetric::Y, MotionMethod::PixelDiff, DEF);
    assert_eq!(s.cols, 2);
    assert_eq!(s.rows, 1);
    // left=0, right=40 → mean 20, max 40.
    assert!((s.frame_score - 20.0).abs() < 0.2, "frame_score {}", s.frame_score);
    assert!((s.max_score - 40.0).abs() < 0.2, "max_score {}", s.max_score);
}

#[test]
fn block_size_eight_is_the_finest_grid() {
    // 8×8 px block is the new minimum supported by the Motion tab.
    let prev = uniform(32, 16, 10);
    let cur = uniform(32, 16, 10);
    let s = compute_motion_stats(&prev, &cur, 32, 16, 8, BlockMetric::Y, MotionMethod::PixelDiff, DEF);
    assert_eq!(s.block_size, 8);
    assert_eq!(s.cols, 4);
    assert_eq!(s.rows, 2);
}

#[test]
fn mismatched_buffers_return_empty() {
    let prev = uniform(8, 8, 0);
    let cur = uniform(4, 4, 0); // different length
    let s = compute_motion_stats(&prev, &cur, 8, 8, 8, BlockMetric::Y, MotionMethod::PixelDiff, DEF);
    assert!(s.is_empty());
    assert_eq!(s.cols, 0);
}

#[test]
fn zero_dimensions_return_empty() {
    let empty: Vec<u8> = Vec::new();
    let s = compute_motion_stats(&empty, &empty, 0, 0, 8, BlockMetric::Y, MotionMethod::PixelDiff, DEF);
    assert!(s.is_empty());
}

#[test]
fn stats_diff_detects_variance_change_without_mean_change() {
    // prev: uniform 128 (std 0). cur: 8×8 checkerboard 0/255 (mean 127.5,
    // std 127.5). Means nearly equal but std jumps → StatsDiff flags Full.
    let prev = uniform(8, 8, 128);
    let cur = checker8(false);
    let s = compute_motion_stats(&prev, &cur, 8, 8, 8, BlockMetric::Y, MotionMethod::StatsDiff, DEF);
    // |Δmean| = |127.5-128| = 0.5, |Δstd| = |127.5-0| = 127.5 → score = 128.0.
    // Pin the magnitude (not just >100) so a variance-for-std regression
    // (which would give ~16256) is caught.
    assert!((s.scores[0] - 128.0).abs() < 1.0, "stats score {}", s.scores[0]);
    assert_eq!(s.classes[0], MotionClass::Full);
}

#[test]
fn partial_edge_blocks_are_classified() {
    // 12×8 with block 8 → 2×1 grid; right block is a 4-wide partial. A shift
    // applied only to the right column must classify the partial block.
    let prev = uniform(12, 8, 40);
    let mut cur = uniform(12, 8, 40);
    for y in 0..8 {
        for x in 8..12 {
            let p = (y * 12 + x) * 3;
            cur[p] = 90;
            cur[p + 1] = 90;
            cur[p + 2] = 90;
        }
    }
    let s = compute_motion_stats(&prev, &cur, 12, 8, 8, BlockMetric::Y, MotionMethod::PixelDiff, DEF);
    assert_eq!(s.cols, 2);
    assert_eq!(s.rows, 1);
    assert_eq!(s.classes[0], MotionClass::None); // left block unchanged
    // Right partial block shifted by 50 → Full.
    assert_eq!(s.classes[1], MotionClass::Full);
    assert!((s.scores[1] - 50.0).abs() < 0.5, "partial block score {}", s.scores[1]);
}

#[test]
fn ms_metric_scores_use_the_ms_scalar() {
    // Achromatic R=G=B → Cb=Cr=0, so MS = 6Y/8 = 0.75*v. A +40 luma shift
    // therefore yields an MS PixelDiff of 0.75*40 = 30 (not 40), pinning the
    // MS code path to a distinct, easily-checked value.
    let prev = uniform(8, 8, 100);
    let cur = uniform(8, 8, 140);
    let s = compute_motion_stats(&prev, &cur, 8, 8, 8, BlockMetric::Ms, MotionMethod::PixelDiff, DEF);
    assert!((s.scores[0] - 30.0).abs() < 0.2, "ms score {}", s.scores[0]);
    assert_eq!(s.classes[0], MotionClass::Full); // 30 ≥ default full (30)
    // Sanity: the same input under Y gives 40 (the raw luma shift), proving the
    // metric argument actually selects a different scalar.
    let s_y = compute_motion_stats(&prev, &cur, 8, 8, 8, BlockMetric::Y, MotionMethod::PixelDiff, DEF);
    assert!((s_y.scores[0] - 40.0).abs() < 0.2, "y score {}", s_y.scores[0]);
}

#[test]
fn thresholds_argument_actually_drives_classification() {
    // Same input (PixelDiff score ≈ 20) classified under two threshold sets:
    // the passed thresholds — not hardcoded defaults — must decide the class.
    let prev = uniform(8, 8, 30);
    let cur = uniform(8, 8, 50); // +20 luma

    let s_def = compute_motion_stats(&prev, &cur, 8, 8, 8, BlockMetric::Y, MotionMethod::PixelDiff, DEF);
    assert_eq!(s_def.classes[0], MotionClass::Much); // 20 ∈ [10,30) under defaults
    assert_eq!(s_def.thresholds, DEF); // echo contract

    let tight = MotionThresholds { slight: 1.0, much: 5.0, full: 15.0 };
    let s_tight = compute_motion_stats(&prev, &cur, 8, 8, 8, BlockMetric::Y, MotionMethod::PixelDiff, tight);
    assert_eq!(s_tight.classes[0], MotionClass::Full); // 20 ≥ 15 → Full
    assert_eq!(s_tight.thresholds, tight); // echo contract
    // Score itself is independent of thresholds.
    assert!((s_def.scores[0] - s_tight.scores[0]).abs() < 1e-6);
}
