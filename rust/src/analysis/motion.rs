//! Per-block inter-frame motion classification.
//!
//! Compares the previous and current frame block-by-block and assigns each
//! block one of four motion classes (None / Slight / Much / Full) based on a
//! tunable set of thresholds. Two scoring methods are offered:
//!
//! - [`MotionMethod::PixelDiff`] — mean absolute per-pixel difference (MAD)
//!   of the chosen scalar metric inside each block. Sensitive to *any* pixel
//!   movement even when the block's average and spread are unchanged.
//! - [`MotionMethod::StatsDiff`] — compares the per-block mean and standard
//!   deviation between the two frames: `score = |Δmean| + |Δstd|`. Blind to
//!   spatial rearrangement that preserves a block's first/second moments.
//!
//! Both scores are expressed in 8-bit luma units so the same threshold scale
//! applies to either method. The scalar metric (Y or MS) is shared with the
//! Block tab via [`crate::analysis::block_stats`].

use crate::analysis::block_stats::{pixel_value, BlockMetric};

/// Scoring strategy for per-block motion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionMethod {
    /// Mean absolute per-pixel difference inside each block.
    PixelDiff,
    /// Difference of per-block mean and std-dev: `|Δmean| + |Δstd|`.
    StatsDiff,
}

impl MotionMethod {
    pub fn label(self) -> &'static str {
        match self {
            MotionMethod::PixelDiff => "Pixel diff",
            // "Mean+Std" (a sum) rather than "Avg·Std" (which reads as a product).
            MotionMethod::StatsDiff => "Mean+Std",
        }
    }
}

/// Four-level motion classification, ordered from least to most change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionClass {
    /// Essentially identical to the previous frame (no motion).
    None,
    /// Slightly different.
    Slight,
    /// Substantially different.
    Much,
    /// Completely different (e.g. a scene cut).
    Full,
}

impl MotionClass {
    /// Short human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            MotionClass::None => "None",
            MotionClass::Slight => "Slight",
            MotionClass::Much => "Much",
            MotionClass::Full => "Full",
        }
    }

    /// Stable index 0..=3 (None=0 … Full=3), used to bucket `class_counts`.
    pub fn index(self) -> usize {
        match self {
            MotionClass::None => 0,
            MotionClass::Slight => 1,
            MotionClass::Much => 2,
            MotionClass::Full => 3,
        }
    }

    /// All four classes in ascending order (handy for legends/iteration).
    pub fn all() -> [MotionClass; 4] {
        [
            MotionClass::None,
            MotionClass::Slight,
            MotionClass::Much,
            MotionClass::Full,
        ]
    }

    /// Classify a score against the thresholds. Uses a descending cascade so
    /// the result is deterministic even if thresholds are not strictly
    /// ordered (the UI keeps them ordered, but the function is robust anyway).
    pub fn from_score(score: f32, t: MotionThresholds) -> MotionClass {
        if score >= t.full {
            MotionClass::Full
        } else if score >= t.much {
            MotionClass::Much
        } else if score >= t.slight {
            MotionClass::Slight
        } else {
            MotionClass::None
        }
    }
}

/// Score boundaries between the four motion classes. A block scoring
/// `>= full` is `Full`; `>= much` is `Much`; `>= slight` is `Slight`;
/// otherwise `None`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionThresholds {
    pub slight: f32,
    pub much: f32,
    pub full: f32,
}

impl MotionThresholds {
    /// Sensible starting points in 8-bit luma units. For consecutive frames a
    /// static scene (sensor noise only) typically scores below 2; small object
    /// or camera motion lands in the 2–10 band; large motion 10–30; a scene
    /// cut usually exceeds 30.
    pub const DEFAULT_SLIGHT: f32 = 2.0;
    pub const DEFAULT_MUCH: f32 = 10.0;
    pub const DEFAULT_FULL: f32 = 30.0;
}

impl Default for MotionThresholds {
    fn default() -> Self {
        Self {
            slight: Self::DEFAULT_SLIGHT,
            much: Self::DEFAULT_MUCH,
            full: Self::DEFAULT_FULL,
        }
    }
}

/// Per-block motion result over a frame pair.
#[derive(Debug, Clone)]
pub struct MotionStats {
    pub block_size: u32,
    pub metric: BlockMetric,
    pub method: MotionMethod,
    /// Number of block columns: ceil(width / block_size).
    pub cols: u32,
    /// Number of block rows: ceil(height / block_size).
    pub rows: u32,
    pub width: u32,
    pub height: u32,
    /// Per-block motion score in row-major order. Non-negative.
    pub scores: Vec<f32>,
    /// Per-block class in row-major order (aligned with `scores`).
    pub classes: Vec<MotionClass>,
    /// Thresholds used to derive `classes` (echoed for rendering/summary).
    pub thresholds: MotionThresholds,
    /// Count of blocks in each class, indexed by [`MotionClass::index`].
    pub class_counts: [u32; 4],
    /// Mean score across all blocks (whole-frame motion level).
    pub frame_score: f32,
    /// Maximum block score (hottest region).
    pub max_score: f32,
}

impl MotionStats {
    pub fn empty() -> Self {
        Self {
            block_size: 32,
            metric: BlockMetric::Y,
            method: MotionMethod::PixelDiff,
            cols: 0,
            rows: 0,
            width: 0,
            height: 0,
            scores: Vec::new(),
            classes: Vec::new(),
            thresholds: MotionThresholds::default(),
            class_counts: [0; 4],
            frame_score: 0.0,
            max_score: 0.0,
        }
    }

    /// True when no motion data is available (no/mismatched frames).
    pub fn is_empty(&self) -> bool {
        self.cols == 0 || self.rows == 0
    }
}

/// Compute per-block motion between `prev_rgb` and `cur_rgb`.
///
/// Both buffers must be tightly packed 8-bit RGB of size `width * height * 3`
/// and equal length. Returns [`MotionStats::empty`] when the dimensions are
/// zero or the buffers are mismatched/too small (the caller renders a "needs
/// a previous frame" hint in that case).
#[allow(clippy::too_many_arguments)]
pub fn compute_motion_stats(
    prev_rgb: &[u8],
    cur_rgb: &[u8],
    width: u32,
    height: u32,
    block_size: u32,
    metric: BlockMetric,
    method: MotionMethod,
    thresholds: MotionThresholds,
) -> MotionStats {
    let bs = block_size.max(1);
    if width == 0 || height == 0 {
        return MotionStats::empty();
    }
    let stride = (width as usize) * 3;
    let need = stride * height as usize;
    if prev_rgb.len() != cur_rgb.len() || prev_rgb.len() < need {
        return MotionStats::empty();
    }

    let cols = width.div_ceil(bs);
    let rows = height.div_ceil(bs);
    let n_blocks = (cols as usize) * (rows as usize);

    // Per-block accumulators. A single pass over the frame feeds every
    // statistic both methods need: sums/sums-of-squares for each frame (mean
    // and variance) plus the running absolute pixel difference (MAD).
    let mut sum_cur = vec![0.0_f64; n_blocks];
    let mut sum_cur2 = vec![0.0_f64; n_blocks];
    let mut sum_prev = vec![0.0_f64; n_blocks];
    let mut sum_prev2 = vec![0.0_f64; n_blocks];
    let mut sum_absdiff = vec![0.0_f64; n_blocks];
    let mut counts = vec![0_u32; n_blocks];

    for y in 0..height {
        let by = y / bs;
        let row_off = (y as usize) * stride;
        for x in 0..width {
            let bx = x / bs;
            // Index in usize to match n_blocks; avoids u32 overflow on
            // pathological frame sizes (cols*rows can exceed u32::MAX).
            let bidx = (by as usize) * (cols as usize) + bx as usize;
            let pi = row_off + (x as usize) * 3;
            let cv = pixel_value(metric, cur_rgb[pi], cur_rgb[pi + 1], cur_rgb[pi + 2]);
            let pv = pixel_value(metric, prev_rgb[pi], prev_rgb[pi + 1], prev_rgb[pi + 2]);
            sum_cur[bidx] += cv;
            sum_cur2[bidx] += cv * cv;
            sum_prev[bidx] += pv;
            sum_prev2[bidx] += pv * pv;
            sum_absdiff[bidx] += (cv - pv).abs();
            counts[bidx] += 1;
        }
    }

    let mut scores = vec![0.0_f32; n_blocks];
    let mut classes = vec![MotionClass::None; n_blocks];
    let mut class_counts = [0_u32; 4];
    let mut score_sum = 0.0_f64;
    let mut max_score = 0.0_f32;

    for i in 0..n_blocks {
        let n = counts[i];
        let score = if n == 0 {
            0.0_f32
        } else {
            let nf = n as f64;
            match method {
                MotionMethod::PixelDiff => (sum_absdiff[i] / nf) as f32,
                MotionMethod::StatsDiff => {
                    let mean_cur = sum_cur[i] / nf;
                    let mean_prev = sum_prev[i] / nf;
                    let var_cur = ((sum_cur2[i] / nf) - mean_cur * mean_cur).max(0.0);
                    let var_prev = ((sum_prev2[i] / nf) - mean_prev * mean_prev).max(0.0);
                    let d_mean = (mean_cur - mean_prev).abs();
                    let d_std = (var_cur.sqrt() - var_prev.sqrt()).abs();
                    (d_mean + d_std) as f32
                }
            }
        };
        let class = MotionClass::from_score(score, thresholds);
        scores[i] = score;
        classes[i] = class;
        class_counts[class.index()] += 1;
        score_sum += score as f64;
        if score > max_score {
            max_score = score;
        }
    }

    let frame_score = if n_blocks == 0 {
        0.0
    } else {
        (score_sum / n_blocks as f64) as f32
    };

    MotionStats {
        block_size: bs,
        metric,
        method,
        cols,
        rows,
        width,
        height,
        scores,
        classes,
        thresholds,
        class_counts,
        frame_score,
        max_score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_score_cascade_picks_the_right_band() {
        let t = MotionThresholds::default(); // (2, 10, 30)
        assert_eq!(MotionClass::from_score(0.0, t), MotionClass::None);
        assert_eq!(MotionClass::from_score(1.9, t), MotionClass::None);
        assert_eq!(MotionClass::from_score(2.0, t), MotionClass::Slight);
        assert_eq!(MotionClass::from_score(9.99, t), MotionClass::Slight);
        assert_eq!(MotionClass::from_score(10.0, t), MotionClass::Much);
        assert_eq!(MotionClass::from_score(29.9, t), MotionClass::Much);
        assert_eq!(MotionClass::from_score(30.0, t), MotionClass::Full);
        assert_eq!(MotionClass::from_score(1000.0, t), MotionClass::Full);
    }

    #[test]
    fn default_thresholds_are_documented_constants() {
        let t = MotionThresholds::default();
        assert_eq!(t.slight, MotionThresholds::DEFAULT_SLIGHT);
        assert_eq!(t.much, MotionThresholds::DEFAULT_MUCH);
        assert_eq!(t.full, MotionThresholds::DEFAULT_FULL);
        assert!(t.slight < t.much && t.much < t.full);
    }

    #[test]
    fn class_index_and_all_are_consistent() {
        let all = MotionClass::all();
        for (i, c) in all.iter().enumerate() {
            assert_eq!(c.index(), i);
        }
    }

    #[test]
    fn empty_stats_report_is_empty() {
        let e = MotionStats::empty();
        assert!(e.is_empty());
        assert_eq!(e.class_counts, [0; 4]);
        assert_eq!(e.frame_score, 0.0);
    }
}
