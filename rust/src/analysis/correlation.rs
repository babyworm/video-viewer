//! Correlation analysis engine (M2, 02 §1–§2) — pure functions, egui-free.
//!
//! Aligns the analysis-side grids (BlockStats / MotionStats /
//! OrientationStats, block size `B`) and the bitstream-side canonical L1
//! grid ([`BitstreamGrid`], 8 px) onto a user-selected common grid
//! `G ∈ {8, 16, 32, 64}` and computes Pearson r / Spearman ρ (tie-average
//! ranks), the conditional motion-class table, and the CSV dump.
//!
//! Resample rules (02 §1):
//! - analysis `B < G`: area-weighted mean; variances compose by the total
//!   variance law `Var = E[var_i] + Var(mean_i)` (pixel-count weighted)
//! - analysis `B > G`: value replication (the same law degenerates to it)
//! - orientation angles are circular ([0, 180), 0 ≡ 180): purity-weighted
//!   doubled-angle vector mean, flat (purity 0) blocks excluded
//! - bitstream L1 → G: mean (qp, mv_mag), sum→renormalize (bpp), covered-area
//!   Intra fraction (intra_ratio)
//! - right/bottom partial G cells and cells with bitstream coverage < 1 are
//!   masked `valid = false` so edge effects never pollute the statistics.

use crate::analysis::bitstream_stats::{BitstreamGrid, ModeClass};
use crate::analysis::block_stats::{compute_block_stats, BlockMetric, BlockStats};
use crate::analysis::motion::{
    compute_motion_stats, MotionClass, MotionMethod, MotionStats, MotionThresholds,
};
use crate::analysis::orientation::{compute_orientation_stats, OrientationStats};

/// Analysis-side (X axis) metric choices (02 §2 view 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XMetric {
    Variance,
    Mean,
    MotionScore,
    MotionClass,
    Orientation,
}

impl XMetric {
    pub const ALL: [XMetric; 5] = [
        XMetric::Variance,
        XMetric::Mean,
        XMetric::MotionScore,
        XMetric::MotionClass,
        XMetric::Orientation,
    ];

    pub fn label(self) -> &'static str {
        match self {
            XMetric::Variance => "variance",
            XMetric::Mean => "mean",
            XMetric::MotionScore => "motion score",
            XMetric::MotionClass => "motion class",
            XMetric::Orientation => "orientation",
        }
    }

    /// True when the metric needs the previous frame (motion pair).
    pub fn needs_previous_frame(self) -> bool {
        matches!(self, XMetric::MotionScore | XMetric::MotionClass)
    }
}

/// Bitstream-side (Y axis) metric choices (02 §2 view 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YMetric {
    Qp,
    Bpp,
    MvMag,
    IntraRatio,
}

impl YMetric {
    pub const ALL: [YMetric; 4] = [YMetric::Qp, YMetric::Bpp, YMetric::MvMag, YMetric::IntraRatio];

    pub fn label(self) -> &'static str {
        match self {
            YMetric::Qp => "QP",
            YMetric::Bpp => "bpp",
            YMetric::MvMag => "|MV|",
            YMetric::IntraRatio => "intra %",
        }
    }
}

/// Representative correlation-pair presets (02 §2). The M2 Y-metric set has
/// no skip-rate / intra-direction axis yet, so the two presets that need
/// them (`motion_class↔skip률`, `orientation↔intra 방향`) are approximated
/// with the closest available Y metric and revisited in M3/M4.
pub const PRESET_PAIRS: [(&str, XMetric, YMetric); 5] = [
    ("variance ↔ bits", XMetric::Variance, YMetric::Bpp),
    ("variance ↔ QP", XMetric::Variance, YMetric::Qp),
    ("motion ↔ |MV|", XMetric::MotionScore, YMetric::MvMag),
    ("motion ↔ bits", XMetric::MotionScore, YMetric::Bpp),
    ("orientation ↔ intra %", XMetric::Orientation, YMetric::IntraRatio),
];

/// The G grid sizes offered by the Correlation tab.
pub const G_SIZES: [u32; 4] = [8, 16, 32, 64];

// ---------------------------------------------------------------------------
// Grids
// ---------------------------------------------------------------------------

/// A scalar grid resampled to the common cell size `g` (analysis side).
#[derive(Debug, Clone)]
pub struct GGrid {
    pub g: u32,
    pub cols: u32,
    pub rows: u32,
    pub values: Vec<f32>,
    /// False for right/bottom partial cells (and cells with no source data).
    pub valid: Vec<bool>,
}

/// Bitstream metrics aggregated from the L1 grid to the common cell size.
#[derive(Debug, Clone)]
pub struct BitstreamG {
    pub g: u32,
    pub cols: u32,
    pub rows: u32,
    pub qp: Vec<f32>,
    /// Bits per covered pixel (sum of bits / covered area — renormalized).
    pub bpp: Vec<f32>,
    pub mv_mag: Vec<f32>,
    /// Covered-area fraction classified Intra ∈ [0, 1].
    pub intra_ratio: Vec<f32>,
    /// False when partially covered (coverage < 1) or a partial edge cell.
    pub valid: Vec<bool>,
}

/// L2 aligned pair (02 §1): one row-major grid of (a, b, valid) samples.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedPair {
    pub g: u32,
    pub cols: u32,
    pub rows: u32,
    /// Analysis-side values.
    pub a: Vec<f32>,
    /// Bitstream-side values.
    pub b: Vec<f32>,
    /// Combined mask: both sides valid.
    pub valid: Vec<bool>,
}

impl AlignedPair {
    /// Number of valid samples.
    pub fn n_valid(&self) -> usize {
        self.valid.iter().filter(|v| **v).count()
    }

    /// Valid fraction ∈ [0, 1] (0 when empty).
    pub fn valid_fraction(&self) -> f32 {
        if self.valid.is_empty() {
            0.0
        } else {
            self.n_valid() as f32 / self.valid.len() as f32
        }
    }
}

#[inline]
fn g_dims(width: u32, height: u32, g: u32) -> (u32, u32) {
    (width.div_ceil(g), height.div_ceil(g))
}

/// Full-cell validity: a G cell is valid only when it lies entirely inside
/// the frame (02 §1: right/bottom partial blocks are masked on both sides).
#[inline]
fn cell_fully_inside(col: u32, row: u32, g: u32, width: u32, height: u32) -> bool {
    (col + 1) * g <= width && (row + 1) * g <= height
}

/// Shared accumulation walk: distribute every source block's clipped
/// footprint over the overlapped G cells, area-weighted. Calls
/// `add(g_index, src_index, overlap_area_px)`.
fn accumulate_overlaps<F: FnMut(usize, usize, f64)>(
    b: u32,
    cols: u32,
    rows: u32,
    width: u32,
    height: u32,
    g: u32,
    mut add: F,
) {
    let (gcols, grows) = g_dims(width, height, g);
    if gcols == 0 || grows == 0 {
        return;
    }
    for br in 0..rows {
        let y0 = br * b;
        let y1 = (y0 + b).min(height);
        if y0 >= y1 {
            continue;
        }
        for bc in 0..cols {
            let x0 = bc * b;
            let x1 = (x0 + b).min(width);
            if x0 >= x1 {
                continue;
            }
            let src = (br as usize) * (cols as usize) + bc as usize;
            let c0 = x0 / g;
            let c1 = (x1 - 1) / g;
            let r0 = y0 / g;
            let r1 = (y1 - 1) / g;
            for gr in r0..=r1.min(grows - 1) {
                let cy0 = (gr * g).max(y0);
                let cy1 = ((gr + 1) * g).min(y1);
                for gc in c0..=c1.min(gcols - 1) {
                    let cx0 = (gc * g).max(x0);
                    let cx1 = ((gc + 1) * g).min(x1);
                    let area = (cx1.saturating_sub(cx0) as f64) * (cy1.saturating_sub(cy0) as f64);
                    if area > 0.0 {
                        let gi = (gr as usize) * (gcols as usize) + gc as usize;
                        add(gi, src, area);
                    }
                }
            }
        }
    }
}

/// Resample a row-major scalar grid (block size `b`) to G by area-weighted
/// mean (`b < G`) / value replication (`b > G`) — one formula covers both.
pub fn resample_scalar_to_g(
    values: &[f32],
    b: u32,
    cols: u32,
    rows: u32,
    width: u32,
    height: u32,
    g: u32,
) -> GGrid {
    let b = b.max(1);
    let g = g.max(1);
    let (gcols, grows) = g_dims(width, height, g);
    let n = (gcols as usize) * (grows as usize);
    let mut acc_w = vec![0.0_f64; n];
    let mut acc_v = vec![0.0_f64; n];
    if values.len() >= (cols as usize) * (rows as usize) {
        accumulate_overlaps(b, cols, rows, width, height, g, |gi, src, area| {
            acc_w[gi] += area;
            acc_v[gi] += values[src] as f64 * area;
        });
    }
    finish_grid(g, gcols, grows, width, height, &acc_w, |i| {
        (acc_v[i] / acc_w[i]) as f32
    })
}

/// Resample per-block variances to G with the total variance law
/// (02 §1): `Var_G = E_w[var_i] + (E_w[mean_i²] − E_w[mean_i]²)` where the
/// weights are the overlap pixel counts. For `b > G` this degenerates to
/// replication (`Var_G = var_i`).
#[allow(clippy::too_many_arguments)]
pub fn resample_variance_to_g(
    means: &[f32],
    vars: &[f32],
    b: u32,
    cols: u32,
    rows: u32,
    width: u32,
    height: u32,
    g: u32,
) -> GGrid {
    let b = b.max(1);
    let g = g.max(1);
    let (gcols, grows) = g_dims(width, height, g);
    let n = (gcols as usize) * (grows as usize);
    let mut acc_w = vec![0.0_f64; n];
    let mut acc_var = vec![0.0_f64; n];
    let mut acc_mean = vec![0.0_f64; n];
    let mut acc_mean2 = vec![0.0_f64; n];
    let src_n = (cols as usize) * (rows as usize);
    if means.len() >= src_n && vars.len() >= src_n {
        accumulate_overlaps(b, cols, rows, width, height, g, |gi, src, area| {
            let m = means[src] as f64;
            acc_w[gi] += area;
            acc_var[gi] += vars[src] as f64 * area;
            acc_mean[gi] += m * area;
            acc_mean2[gi] += m * m * area;
        });
    }
    finish_grid(g, gcols, grows, width, height, &acc_w, |i| {
        let w = acc_w[i];
        let e_var = acc_var[i] / w;
        let e_mean = acc_mean[i] / w;
        let var_mean = (acc_mean2[i] / w - e_mean * e_mean).max(0.0);
        (e_var + var_mean) as f32
    })
}

/// Resample per-block edge orientations to G. Angles live on the circular
/// domain [0, 180) (0 ≡ 180), so an arithmetic mean is wrong: 178° and 2°
/// (both near-horizontal edges) would average to 90° (vertical). Instead the
/// doubled angles are combined as vectors — `Σ w·e^{i·2θ}` — and the half
/// angle of the resultant is the cell orientation. The weight per source
/// block is `overlap_area × purity`, which simultaneously
/// (a) gates flat blocks out entirely (purity 0 ⇒ weight 0 — the
///     [`crate::analysis::orientation`] contract says never trust an angle
///     without its purity), and
/// (b) lets confident edges dominate mixed cells.
/// Cells with no oriented contribution (all-flat) are `valid = false`.
#[allow(clippy::too_many_arguments)]
pub fn resample_orientation_to_g(
    angles: &[f32],
    purity: &[f32],
    b: u32,
    cols: u32,
    rows: u32,
    width: u32,
    height: u32,
    g: u32,
) -> GGrid {
    let b = b.max(1);
    let g = g.max(1);
    let (gcols, grows) = g_dims(width, height, g);
    let n = (gcols as usize) * (grows as usize);
    let mut acc_w = vec![0.0_f64; n];
    let mut acc_cos = vec![0.0_f64; n];
    let mut acc_sin = vec![0.0_f64; n];
    let src_n = (cols as usize) * (rows as usize);
    if angles.len() >= src_n && purity.len() >= src_n {
        accumulate_overlaps(b, cols, rows, width, height, g, |gi, src, area| {
            let w = area * purity[src].max(0.0) as f64;
            if w <= 0.0 {
                return; // flat block: no orientation evidence
            }
            let two_theta = 2.0 * (angles[src] as f64).to_radians();
            acc_w[gi] += w;
            acc_cos[gi] += w * two_theta.cos();
            acc_sin[gi] += w * two_theta.sin();
        });
    }
    finish_grid(g, gcols, grows, width, height, &acc_w, |i| {
        // atan2 of the resultant vector, halved back to [0, 180).
        (acc_sin[i].atan2(acc_cos[i]).to_degrees() / 2.0).rem_euclid(180.0) as f32
    })
}

/// Resample per-block motion classes to G by largest-overlap-area majority
/// vote. Returns the class list plus the same classes as `f32` indices
/// (ordinal scatter axis).
pub fn resample_classes_to_g(
    classes: &[MotionClass],
    b: u32,
    cols: u32,
    rows: u32,
    width: u32,
    height: u32,
    g: u32,
) -> (Vec<MotionClass>, GGrid) {
    let b = b.max(1);
    let g = g.max(1);
    let (gcols, grows) = g_dims(width, height, g);
    let n = (gcols as usize) * (grows as usize);
    let mut votes = vec![[0.0_f64; 4]; n];
    let mut acc_w = vec![0.0_f64; n];
    if classes.len() >= (cols as usize) * (rows as usize) {
        accumulate_overlaps(b, cols, rows, width, height, g, |gi, src, area| {
            votes[gi][classes[src].index()] += area;
            acc_w[gi] += area;
        });
    }
    let mut out_classes = vec![MotionClass::None; n];
    for (i, v) in votes.iter().enumerate() {
        if acc_w[i] > 0.0 {
            let best = v
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| k)
                .unwrap_or(0);
            out_classes[i] = MotionClass::all()[best];
        }
    }
    let grid = finish_grid(g, gcols, grows, width, height, &acc_w, |i| {
        out_classes[i].index() as f32
    });
    (out_classes, grid)
}

/// Common tail: values from `f(i)` where covered, full-cell validity mask.
fn finish_grid<F: Fn(usize) -> f32>(
    g: u32,
    gcols: u32,
    grows: u32,
    width: u32,
    height: u32,
    acc_w: &[f64],
    f: F,
) -> GGrid {
    let n = (gcols as usize) * (grows as usize);
    let mut values = vec![0.0_f32; n];
    let mut valid = vec![false; n];
    for gr in 0..grows {
        for gc in 0..gcols {
            let i = (gr as usize) * (gcols as usize) + gc as usize;
            if acc_w[i] > 0.0 {
                values[i] = f(i);
                valid[i] = cell_fully_inside(gc, gr, g, width, height);
            }
        }
    }
    GGrid {
        g,
        cols: gcols,
        rows: grows,
        values,
        valid,
    }
}

/// Aggregate the canonical bitstream L1 grid to G (02 §1 L1 → G rule):
/// covered-area weighted mean for qp / mv_mag, bits sum renormalized over the
/// covered area for bpp, covered-area Intra fraction for intra_ratio. Cells
/// touching bitstream coverage < 1 or hanging over the frame edge are
/// `valid = false`.
pub fn aggregate_bitstream_to_g(l1: &BitstreamGrid, width: u32, height: u32, g: u32) -> BitstreamG {
    let g = g.max(1);
    let (gcols, grows) = g_dims(width, height, g);
    let n = (gcols as usize) * (grows as usize);
    let mut acc_a = vec![0.0_f64; n]; // covered px
    let mut acc_qp = vec![0.0_f64; n];
    let mut acc_bits = vec![0.0_f64; n]; // bpp·area = bits
    let mut acc_mv = vec![0.0_f64; n];
    let mut acc_intra = vec![0.0_f64; n];
    let cell = l1.cell.max(1);
    let cell_area = (cell as f64) * (cell as f64);
    accumulate_overlaps(cell, l1.cols, l1.rows, width, height, g, |gi, src, area| {
        // `area` is the in-frame footprint overlap with this G cell.
        // `coverage` is covered px / full cell area (cell²), so the covered
        // pixels inside the overlap are coverage·cell² scaled by the overlap
        // share of the cell's in-frame footprint.
        let c = (src % l1.cols as usize) as u32;
        let r = (src / l1.cols as usize) as u32;
        let fw = ((c + 1) * cell).min(width).saturating_sub(c * cell) as f64;
        let fh = ((r + 1) * cell).min(height).saturating_sub(r * cell) as f64;
        let footprint = fw * fh;
        if footprint <= 0.0 {
            return;
        }
        let covered = l1.coverage[src] as f64 * cell_area * (area / footprint);
        if covered <= 0.0 {
            return;
        }
        acc_a[gi] += covered;
        acc_qp[gi] += l1.qp[src] as f64 * covered;
        acc_bits[gi] += l1.bpp[src] as f64 * covered;
        acc_mv[gi] += l1.mv_mag[src] as f64 * covered;
        if l1.mode[src] == ModeClass::Intra {
            acc_intra[gi] += covered;
        }
    });

    let mut qp = vec![0.0_f32; n];
    let mut bpp = vec![0.0_f32; n];
    let mut mv = vec![0.0_f32; n];
    let mut intra = vec![0.0_f32; n];
    let mut valid = vec![false; n];
    let full = (g as f64) * (g as f64);
    for gr in 0..grows {
        for gc in 0..gcols {
            let i = (gr as usize) * (gcols as usize) + gc as usize;
            let a = acc_a[i];
            if a > 0.0 {
                qp[i] = (acc_qp[i] / a) as f32;
                bpp[i] = (acc_bits[i] / a) as f32;
                mv[i] = (acc_mv[i] / a) as f32;
                intra[i] = (acc_intra[i] / a) as f32;
                // Fully covered (coverage == 1 everywhere) ⇔ covered px == g².
                valid[i] = cell_fully_inside(gc, gr, g, width, height) && a >= full * 0.999;
            }
        }
    }
    BitstreamG {
        g,
        cols: gcols,
        rows: grows,
        qp,
        bpp,
        mv_mag: mv,
        intra_ratio: intra,
        valid,
    }
}

/// Combine an analysis G grid and a bitstream G aggregate into the L2
/// [`AlignedPair`]. Grid dimensions may differ (R5 resolution mismatch);
/// the intersection is used.
pub fn align(x: &GGrid, y: &BitstreamG, y_metric: YMetric) -> AlignedPair {
    debug_assert_eq!(x.g, y.g);
    let cols = x.cols.min(y.cols);
    let rows = x.rows.min(y.rows);
    let n = (cols as usize) * (rows as usize);
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    let mut valid = Vec::with_capacity(n);
    let y_vals: &[f32] = match y_metric {
        YMetric::Qp => &y.qp,
        YMetric::Bpp => &y.bpp,
        YMetric::MvMag => &y.mv_mag,
        YMetric::IntraRatio => &y.intra_ratio,
    };
    for r in 0..rows as usize {
        for c in 0..cols as usize {
            let xi = r * x.cols as usize + c;
            let yi = r * y.cols as usize + c;
            a.push(x.values[xi]);
            b.push(y_vals[yi]);
            valid.push(x.valid[xi] && y.valid[yi]);
        }
    }
    AlignedPair {
        g: x.g,
        cols,
        rows,
        a,
        b,
        valid,
    }
}

// ---------------------------------------------------------------------------
// Statistics (02 §2 view 2)
// ---------------------------------------------------------------------------

/// Pearson correlation over the valid samples. `None` when fewer than two
/// valid samples or either side has zero variance.
pub fn pearson_r(a: &[f32], b: &[f32], valid: &[bool]) -> Option<f64> {
    let mut n = 0.0_f64;
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0_f64, 0.0, 0.0, 0.0, 0.0);
    for i in 0..a.len().min(b.len()).min(valid.len()) {
        if !valid[i] {
            continue;
        }
        let (x, y) = (a[i] as f64, b[i] as f64);
        n += 1.0;
        sa += x;
        sb += y;
        saa += x * x;
        sbb += y * y;
        sab += x * y;
    }
    if n < 2.0 {
        return None;
    }
    let cov = sab - sa * sb / n;
    let va = saa - sa * sa / n;
    let vb = sbb - sb * sb / n;
    if va <= 0.0 || vb <= 0.0 {
        return None;
    }
    Some(cov / (va.sqrt() * vb.sqrt()))
}

/// Tie-average (fractional) ranks, 1-based: equal values receive the mean of
/// the ranks they span — required for stepped variables like QP.
pub fn ranks_tie_average(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| vals[i].partial_cmp(&vals[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && vals[order[j + 1]] == vals[order[i]] {
            j += 1;
        }
        // Ranks i+1 ..= j+1 (1-based) share the average.
        let avg = (i + 1 + j + 1) as f64 / 2.0;
        for k in i..=j {
            ranks[order[k]] = avg;
        }
        i = j + 1;
    }
    ranks
}

/// Spearman ρ with tie-average ranks: Pearson correlation of the rank
/// vectors over the valid samples.
pub fn spearman_rho(a: &[f32], b: &[f32], valid: &[bool]) -> Option<f64> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..a.len().min(b.len()).min(valid.len()) {
        if valid[i] {
            xs.push(a[i] as f64);
            ys.push(b[i] as f64);
        }
    }
    if xs.len() < 2 {
        return None;
    }
    let rx: Vec<f32> = ranks_tie_average(&xs).into_iter().map(|v| v as f32).collect();
    let ry: Vec<f32> = ranks_tie_average(&ys).into_iter().map(|v| v as f32).collect();
    let all = vec![true; rx.len()];
    pearson_r(&rx, &ry, &all)
}

// ---------------------------------------------------------------------------
// Conditional class table (02 §2 view 3)
// ---------------------------------------------------------------------------

/// One row of the motion-class × bitstream table.
#[derive(Debug, Clone, Copy)]
pub struct ClassRow {
    pub class: MotionClass,
    pub cells: usize,
    pub mean_qp: f32,
    pub mean_bpp: f32,
    pub mean_mv: f32,
    pub intra_ratio: f32,
}

/// Per-motion-class means of the bitstream metrics over the valid
/// intersection cells: "정적으로 분류한 블록에 인코더가 bits를 쓰는가".
pub fn class_table(
    classes: &[MotionClass],
    class_valid: &[bool],
    class_cols: u32,
    class_rows: u32,
    bs: &BitstreamG,
) -> [ClassRow; 4] {
    let cols = class_cols.min(bs.cols) as usize;
    let rows = class_rows.min(bs.rows) as usize;
    let mut count = [0usize; 4];
    let mut qp = [0.0_f64; 4];
    let mut bpp = [0.0_f64; 4];
    let mut mv = [0.0_f64; 4];
    let mut intra = [0.0_f64; 4];
    for r in 0..rows {
        for c in 0..cols {
            let ci = r * class_cols as usize + c;
            let bi = r * bs.cols as usize + c;
            if ci >= classes.len() || ci >= class_valid.len() {
                continue;
            }
            if !class_valid[ci] || !bs.valid[bi] {
                continue;
            }
            let k = classes[ci].index();
            count[k] += 1;
            qp[k] += bs.qp[bi] as f64;
            bpp[k] += bs.bpp[bi] as f64;
            mv[k] += bs.mv_mag[bi] as f64;
            intra[k] += bs.intra_ratio[bi] as f64;
        }
    }
    let mut out = [ClassRow {
        class: MotionClass::None,
        cells: 0,
        mean_qp: 0.0,
        mean_bpp: 0.0,
        mean_mv: 0.0,
        intra_ratio: 0.0,
    }; 4];
    for (k, class) in MotionClass::all().into_iter().enumerate() {
        let n = count[k].max(1) as f64;
        out[k] = ClassRow {
            class,
            cells: count[k],
            mean_qp: if count[k] > 0 { (qp[k] / n) as f32 } else { 0.0 },
            mean_bpp: if count[k] > 0 { (bpp[k] / n) as f32 } else { 0.0 },
            mean_mv: if count[k] > 0 { (mv[k] / n) as f32 } else { 0.0 },
            intra_ratio: if count[k] > 0 { (intra[k] / n) as f32 } else { 0.0 },
        };
    }
    out
}

// ---------------------------------------------------------------------------
// Per-frame analysis grids (root / scan-thread side)
// ---------------------------------------------------------------------------

/// Canonical analysis block size used for the correlation source grids: the
/// same 8 px as the bitstream L1 grid, so the resamplers only ever aggregate
/// upward in the UI path (`B ≤ G`); `B > G` stays unit-tested for generality.
pub const ANALYSIS_BLOCK: u32 = 8;

/// The analysis-side grids for one frame, computed at [`ANALYSIS_BLOCK`] px.
#[derive(Debug, Clone)]
pub struct AnalysisFrameGrids {
    pub width: u32,
    pub height: u32,
    pub block: BlockStats,
    /// `None` when there is no previous sequential frame.
    pub motion: Option<MotionStats>,
    pub orientation: OrientationStats,
}

/// Compute all analysis-side grids for one RGB frame (BT.709 luma metric,
/// PixelDiff motion scoring with the default thresholds).
pub fn compute_analysis_grids(
    rgb: &[u8],
    prev_rgb: Option<&[u8]>,
    width: u32,
    height: u32,
) -> AnalysisFrameGrids {
    let block = compute_block_stats(rgb, width, height, ANALYSIS_BLOCK, BlockMetric::Y);
    let motion = prev_rgb.and_then(|prev| {
        if prev.len() == rgb.len() {
            let m = compute_motion_stats(
                prev,
                rgb,
                width,
                height,
                ANALYSIS_BLOCK,
                BlockMetric::Y,
                MotionMethod::PixelDiff,
                MotionThresholds::default(),
            );
            (!m.is_empty()).then_some(m)
        } else {
            None
        }
    });
    let orientation = compute_orientation_stats(rgb, width, height, ANALYSIS_BLOCK);
    AnalysisFrameGrids {
        width,
        height,
        block,
        motion,
        orientation,
    }
}

/// Resolve the selected X metric to a G grid. `None` when the metric needs
/// motion data and no previous frame was available.
pub fn x_grid(grids: &AnalysisFrameGrids, x: XMetric, g: u32) -> Option<GGrid> {
    let bl = &grids.block;
    match x {
        XMetric::Variance => Some(resample_variance_to_g(
            &bl.means, &bl.vars, bl.block_size, bl.cols, bl.rows, grids.width, grids.height, g,
        )),
        XMetric::Mean => Some(resample_scalar_to_g(
            &bl.means, bl.block_size, bl.cols, bl.rows, grids.width, grids.height, g,
        )),
        XMetric::MotionScore => grids.motion.as_ref().map(|m| {
            resample_scalar_to_g(
                &m.scores, m.block_size, m.cols, m.rows, grids.width, grids.height, g,
            )
        }),
        XMetric::MotionClass => grids.motion.as_ref().map(|m| {
            resample_classes_to_g(
                &m.classes, m.block_size, m.cols, m.rows, grids.width, grids.height, g,
            )
            .1
        }),
        XMetric::Orientation => {
            let o = &grids.orientation;
            Some(resample_orientation_to_g(
                &o.angles, &o.purity, o.block_size, o.cols, o.rows, grids.width, grids.height, g,
            ))
        }
    }
}

/// Motion classes resampled to G for the class table. `None` without motion.
pub fn classes_at_g(grids: &AnalysisFrameGrids, g: u32) -> Option<(Vec<MotionClass>, GGrid)> {
    grids.motion.as_ref().map(|m| {
        resample_classes_to_g(
            &m.classes, m.block_size, m.cols, m.rows, grids.width, grids.height, g,
        )
    })
}

// ---------------------------------------------------------------------------
// Frame-range scan plumbing (root background thread ↔ window)
// ---------------------------------------------------------------------------

/// Child → root request to scan a frame range on a background thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorrScanRequest {
    pub start: usize,
    pub end: usize,
    pub g: u32,
    pub x: XMetric,
    pub y: YMetric,
}

/// Background scan output: per-frame aligned pairs (kept per frame so the
/// CSV dump can carry the frame column).
#[derive(Debug, Clone)]
pub struct CorrScanResult {
    pub request: CorrScanRequest,
    pub frames: Vec<(usize, AlignedPair)>,
    pub error: Option<String>,
}

impl CorrScanResult {
    /// Concatenated (a, b, valid) across all scanned frames.
    pub fn concat(&self) -> (Vec<f32>, Vec<f32>, Vec<bool>) {
        let n: usize = self.frames.iter().map(|(_, p)| p.a.len()).sum();
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut valid = Vec::with_capacity(n);
        for (_, p) in &self.frames {
            a.extend_from_slice(&p.a);
            b.extend_from_slice(&p.b);
            valid.extend_from_slice(&p.valid);
        }
        (a, b, valid)
    }
}

// ---------------------------------------------------------------------------
// CSV export (02 §2)
// ---------------------------------------------------------------------------

pub const CSV_HEADER: &str = "frame,cell_x,cell_y,a,b,valid";

/// Append one frame's pair as CSV rows. `valid` is written as 1/0. Floats
/// use Rust's shortest-roundtrip `Display`, so re-parsing is lossless.
pub fn csv_push(out: &mut String, frame: usize, pair: &AlignedPair) {
    use std::fmt::Write;
    for r in 0..pair.rows as usize {
        for c in 0..pair.cols as usize {
            let i = r * pair.cols as usize + c;
            let _ = writeln!(
                out,
                "{},{},{},{},{},{}",
                frame,
                c,
                r,
                pair.a[i],
                pair.b[i],
                if pair.valid[i] { 1 } else { 0 },
            );
        }
    }
}

/// Full CSV dump (header + rows) for a set of frames.
pub fn csv_dump(frames: &[(usize, &AlignedPair)]) -> String {
    let mut out = String::from(CSV_HEADER);
    out.push('\n');
    for (frame, pair) in frames {
        csv_push(&mut out, *frame, pair);
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn all_valid(n: usize) -> Vec<bool> {
        vec![true; n]
    }

    #[test]
    fn pearson_perfect_inverse_and_none() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [10.0_f32, 20.0, 30.0, 40.0];
        let r = pearson_r(&a, &b, &all_valid(4)).unwrap();
        assert!((r - 1.0).abs() < 1e-9, "r={r}");
        let inv = [4.0_f32, 3.0, 2.0, 1.0];
        let r = pearson_r(&a, &inv, &all_valid(4)).unwrap();
        assert!((r + 1.0).abs() < 1e-9, "r={r}");
        // Zero variance on one side → None.
        let flat = [5.0_f32; 4];
        assert!(pearson_r(&a, &flat, &all_valid(4)).is_none());
        // Fewer than 2 valid samples → None.
        assert!(pearson_r(&a, &b, &[true, false, false, false]).is_none());
    }

    #[test]
    fn pearson_uncorrelated_is_near_zero() {
        // Orthogonal pattern: b is symmetric around the mean w.r.t. a.
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [1.0_f32, -1.0, -1.0, 1.0];
        let r = pearson_r(&a, &b, &all_valid(4)).unwrap();
        assert!(r.abs() < 1e-9, "r={r}");
    }

    #[test]
    fn pearson_respects_valid_mask() {
        // The masked-out outlier would destroy the correlation if counted.
        let a = [1.0_f32, 2.0, 3.0, 100.0];
        let b = [1.0_f32, 2.0, 3.0, -100.0];
        let r = pearson_r(&a, &b, &[true, true, true, false]).unwrap();
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ranks_tie_average_known_values() {
        // [10, 20, 20, 30] → ranks [1, 2.5, 2.5, 4]
        let r = ranks_tie_average(&[10.0, 20.0, 20.0, 30.0]);
        assert_eq!(r, vec![1.0, 2.5, 2.5, 4.0]);
        // All equal → everyone gets the middle rank.
        let r = ranks_tie_average(&[7.0, 7.0, 7.0]);
        assert_eq!(r, vec![2.0, 2.0, 2.0]);
        // Unsorted input keeps positional mapping.
        let r = ranks_tie_average(&[30.0, 10.0, 20.0]);
        assert_eq!(r, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn spearman_with_ties_matches_hand_computed_value() {
        // a = [1, 2, 2, 3] → ranks [1, 2.5, 2.5, 4]
        // b = [1, 3, 2, 4] → ranks [1, 3, 2, 4]
        // Pearson of ranks = 4.5 / sqrt(4.5 · 5) = 0.9486832…
        let a = [1.0_f32, 2.0, 2.0, 3.0];
        let b = [1.0_f32, 3.0, 2.0, 4.0];
        let rho = spearman_rho(&a, &b, &all_valid(4)).unwrap();
        assert!((rho - 0.948_683_298).abs() < 1e-6, "rho={rho}");
        // Monotone but non-linear stays exactly 1 (rank invariance).
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let y = [1.0_f32, 8.0, 27.0, 64.0];
        let rho = spearman_rho(&x, &y, &all_valid(4)).unwrap();
        assert!((rho - 1.0).abs() < 1e-9);
    }

    #[test]
    fn variance_composition_matches_direct_block_stats() {
        // Numerically verify the total variance law: resampling 8px block
        // stats to G=16 must equal computing block stats directly at 16px.
        // Deterministic pseudo-random 16×16 grey image.
        let (w, h) = (16u32, 16u32);
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        let mut seed = 0x1234_5678_u32;
        for _ in 0..(w * h) {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = (seed >> 24) as u8;
            rgb.extend_from_slice(&[v, v, v]);
        }
        let s8 = compute_block_stats(&rgb, w, h, 8, BlockMetric::Y);
        let s16 = compute_block_stats(&rgb, w, h, 16, BlockMetric::Y);
        let g = resample_variance_to_g(&s8.means, &s8.vars, 8, s8.cols, s8.rows, w, h, 16);
        assert_eq!((g.cols, g.rows), (1, 1));
        assert!(
            (g.values[0] - s16.vars[0]).abs() < 0.05,
            "law {} vs direct {}",
            g.values[0],
            s16.vars[0]
        );
        // Analytic cross-check with two equal-weight halves:
        // Var = (v1+v2)/2 + ((m1−m2)/2)².
        let m = resample_scalar_to_g(&s8.means, 8, s8.cols, s8.rows, w, h, 16);
        assert!((m.values[0] - s16.means[0]).abs() < 1e-3);
    }

    #[test]
    fn variance_law_two_halves_analytic() {
        // One 16px G cell from two 8×16 columns... build from raw values:
        // means [10, 30], vars [4, 8], equal areas.
        // Var = (4+8)/2 + ((10−30)/2)² = 6 + 100 = 106.
        let g = resample_variance_to_g(&[10.0, 30.0], &[4.0, 8.0], 8, 2, 1, 16, 8, 16);
        // 16×8 frame at G=16: one column cell rows: height 8 < 16 → partial →
        // value still computed, valid false.
        assert_eq!((g.cols, g.rows), (1, 1));
        assert!((g.values[0] - 106.0).abs() < 1e-4, "{}", g.values[0]);
        assert!(!g.valid[0], "8px-high frame can't fill a 16px cell");
    }

    #[test]
    fn resample_b_less_than_g_averages() {
        // 4 blocks of 8px → one 16px cell, equal areas → plain mean.
        let vals = [1.0_f32, 3.0, 5.0, 7.0];
        let g = resample_scalar_to_g(&vals, 8, 2, 2, 16, 16, 16);
        assert_eq!((g.cols, g.rows), (1, 1));
        assert!((g.values[0] - 4.0).abs() < 1e-6);
        assert!(g.valid[0]);
    }

    #[test]
    fn resample_b_greater_than_g_replicates() {
        // One 16px block → four 8px cells, all carrying the block's value.
        let vals = [42.0_f32];
        let g = resample_scalar_to_g(&vals, 16, 1, 1, 16, 16, 8);
        assert_eq!((g.cols, g.rows), (2, 2));
        for i in 0..4 {
            assert!((g.values[i] - 42.0).abs() < 1e-6);
            assert!(g.valid[i]);
        }
        // Variance replicates too (Var of a single mean is 0).
        let gv = resample_variance_to_g(&[10.0], &[3.5], 16, 1, 1, 16, 16, 8);
        for i in 0..4 {
            assert!((gv.values[i] - 3.5).abs() < 1e-6);
        }
    }

    #[test]
    fn resample_g_equals_b_is_identity() {
        let vals = [1.0_f32, 2.0, 3.0, 4.0];
        let g = resample_scalar_to_g(&vals, 8, 2, 2, 16, 16, 8);
        assert_eq!(g.values, vals.to_vec());
        assert!(g.valid.iter().all(|v| *v));
    }

    #[test]
    fn partial_edge_cells_are_invalid() {
        // 20×12 frame, G=8 → 3×2 cells; only (0,0) and (1,0) fit fully.
        let vals = vec![1.0_f32; 6];
        let g = resample_scalar_to_g(&vals, 8, 3, 2, 20, 12, 8);
        assert_eq!((g.cols, g.rows), (3, 2));
        assert_eq!(
            g.valid,
            vec![true, true, false, false, false, false],
            "only full 8px cells are valid"
        );
    }

    #[test]
    fn orientation_resample_is_circular() {
        // Two 8px blocks at 178° and 2° — the same near-horizontal edge on
        // both sides of the wrap. The arithmetic mean would be 90°
        // (vertical, i.e. orthogonal!); the circular mean must stay at
        // 0°/180°. 16×8 frame → one 16px G cell (partial in height, so
        // check the value, not validity).
        let g = resample_orientation_to_g(&[178.0, 2.0], &[1.0, 1.0], 8, 2, 1, 16, 8, 16);
        assert_eq!((g.cols, g.rows), (1, 1));
        let a = g.values[0];
        let dist = (a.min(180.0 - a)).abs();
        assert!(dist < 1.0, "circular mean of 178° and 2° must be ≈0/180, got {a}");
        // Same-side angles behave like a plain mean: 40° and 60° → 50°.
        let g = resample_orientation_to_g(&[40.0, 60.0], &[1.0, 1.0], 8, 2, 1, 16, 8, 16);
        assert!((g.values[0] - 50.0).abs() < 1e-3, "{}", g.values[0]);
    }

    #[test]
    fn orientation_resample_gates_on_purity() {
        // Block 0 is flat (purity 0, angle 0 is a placeholder), block 1 has
        // a confident 90° edge: the cell must report 90°, not the 45° an
        // unweighted mean of the placeholder would give.
        let g = resample_orientation_to_g(&[0.0, 90.0], &[0.0, 1.0], 8, 2, 1, 16, 8, 16);
        assert!((g.values[0] - 90.0).abs() < 1e-3, "{}", g.values[0]);
        // All-flat cell: no orientation evidence → invalid.
        let g = resample_orientation_to_g(&[0.0; 4], &[0.0; 4], 8, 2, 2, 16, 16, 16);
        assert!(!g.valid[0], "all-flat cell must be invalid");
        // Identity case (B == G): angles pass through, flat cells invalid.
        let g = resample_orientation_to_g(&[30.0, 0.0], &[0.5, 0.0], 8, 2, 1, 16, 8, 8);
        assert!((g.values[0] - 30.0).abs() < 1e-3);
        assert!(g.valid[0]);
        assert!(!g.valid[1]);
    }

    #[test]
    fn class_majority_resample() {
        use MotionClass::*;
        // 2×2 blocks of 8px → one 16px cell: 3 None vs 1 Full → None.
        let (classes, grid) = resample_classes_to_g(&[None, None, None, Full], 8, 2, 2, 16, 16, 16);
        assert_eq!(classes, vec![None]);
        assert_eq!(grid.values, vec![0.0]);
        // 2 Much + 1 None + 1 Slight → Much (index 2).
        let (classes, grid) =
            resample_classes_to_g(&[Much, Much, None, Slight], 8, 2, 2, 16, 16, 16);
        assert_eq!(classes, vec![Much]);
        assert_eq!(grid.values, vec![2.0]);
    }

    #[test]
    fn bitstream_aggregate_means_and_renormalized_bpp() {
        use crate::analysis::bitstream_stats::rasterize_blocks;
        use crate::core::catb::BsBlock;
        let block = |x: u32, y: u32, qp: i32, bits: i64, mode: &str| BsBlock {
            x,
            y,
            w: 8,
            h: 8,
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
        };
        // 16×16 frame, four 8×8 blocks: QP 20/40 mix, bits 64 each,
        // top-left Intra, rest Inter.
        let blocks = vec![
            block(0, 0, 20, 64, "Intra"),
            block(8, 0, 40, 64, "Inter"),
            block(0, 8, 20, 64, "Inter"),
            block(8, 8, 40, 64, "Inter"),
        ];
        let l1 = rasterize_blocks(&blocks, 16, 16, 8);
        let bg = aggregate_bitstream_to_g(&l1, 16, 16, 16);
        assert_eq!((bg.cols, bg.rows), (1, 1));
        assert!((bg.qp[0] - 30.0).abs() < 1e-4);
        // 256 bits over 256 px = 1.0 bpp (sum → renormalize).
        assert!((bg.bpp[0] - 1.0).abs() < 1e-4);
        // 1 of 4 equal-area cells Intra → 0.25.
        assert!((bg.intra_ratio[0] - 0.25).abs() < 1e-4);
        assert!(bg.valid[0]);
    }

    #[test]
    fn bitstream_partial_coverage_invalidates_cell() {
        use crate::analysis::bitstream_stats::rasterize_blocks;
        use crate::core::catb::BsBlock;
        // Only one 8×8 block in a 16×16 frame → the 16px G cell is 25%
        // covered → invalid.
        let b = BsBlock {
            x: 0,
            y: 0,
            w: 8,
            h: 8,
            ctu_address: 0,
            qp: 32,
            partition: String::new(),
            prediction_mode: "Intra".into(),
            bits: 64,
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
        };
        let l1 = rasterize_blocks(&[b], 16, 16, 8);
        let bg = aggregate_bitstream_to_g(&l1, 16, 16, 16);
        assert!(!bg.valid[0], "coverage < 1 must invalidate");
        // Values are still the covered-area means.
        assert!((bg.qp[0] - 32.0).abs() < 1e-4);
    }

    #[test]
    fn align_intersects_and_combines_masks() {
        let x = GGrid {
            g: 16,
            cols: 2,
            rows: 1,
            values: vec![1.0, 2.0],
            valid: vec![true, false],
        };
        let y = BitstreamG {
            g: 16,
            cols: 3,
            rows: 1,
            qp: vec![30.0, 31.0, 32.0],
            bpp: vec![0.5, 0.6, 0.7],
            mv_mag: vec![0.0; 3],
            intra_ratio: vec![0.0; 3],
            valid: vec![true, true, true],
        };
        let p = align(&x, &y, YMetric::Qp);
        assert_eq!((p.cols, p.rows), (2, 1));
        assert_eq!(p.a, vec![1.0, 2.0]);
        assert_eq!(p.b, vec![30.0, 31.0]);
        assert_eq!(p.valid, vec![true, false]);
        assert_eq!(p.n_valid(), 1);
        assert!((p.valid_fraction() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn csv_roundtrip() {
        let pair = AlignedPair {
            g: 16,
            cols: 2,
            rows: 2,
            a: vec![1.25, -3.5, 0.001, 7.0],
            b: vec![30.0, 31.5, 0.0, -1.0],
            valid: vec![true, false, true, true],
        };
        let csv = csv_dump(&[(3, &pair)]);
        let mut lines = csv.lines();
        assert_eq!(lines.next(), Some(CSV_HEADER));
        let rows: Vec<&str> = lines.collect();
        assert_eq!(rows.len(), 4, "rows = cols × rows");
        // Parse back and compare field-for-field.
        let mut got_a = vec![0.0_f32; 4];
        let mut got_b = vec![0.0_f32; 4];
        let mut got_v = vec![false; 4];
        for row in rows {
            let f: Vec<&str> = row.split(',').collect();
            assert_eq!(f.len(), 6);
            assert_eq!(f[0], "3");
            let (cx, cy): (usize, usize) = (f[1].parse().unwrap(), f[2].parse().unwrap());
            let i = cy * 2 + cx;
            got_a[i] = f[3].parse().unwrap();
            got_b[i] = f[4].parse().unwrap();
            got_v[i] = f[5] == "1";
        }
        assert_eq!(got_a, pair.a);
        assert_eq!(got_b, pair.b);
        assert_eq!(got_v, pair.valid);
    }

    #[test]
    fn class_table_conditional_means() {
        use MotionClass::*;
        let classes = vec![None, Full];
        let bs = BitstreamG {
            g: 16,
            cols: 2,
            rows: 1,
            qp: vec![30.0, 40.0],
            bpp: vec![0.5, 2.0],
            mv_mag: vec![0.0, 12.0],
            intra_ratio: vec![1.0, 0.0],
            valid: vec![true, true],
        };
        let t = class_table(&classes, &[true, true], 2, 1, &bs);
        assert_eq!(t[0].cells, 1);
        assert!((t[0].mean_qp - 30.0).abs() < 1e-5);
        assert!((t[0].intra_ratio - 1.0).abs() < 1e-5);
        assert_eq!(t[3].cells, 1);
        assert!((t[3].mean_mv - 12.0).abs() < 1e-5);
        // Empty classes report zero cells.
        assert_eq!(t[1].cells, 0);
        assert_eq!(t[2].cells, 0);
    }

    #[test]
    fn x_grid_motion_requires_previous_frame() {
        let rgb = vec![128u8; 16 * 16 * 3];
        let grids = compute_analysis_grids(&rgb, None, 16, 16);
        assert!(grids.motion.is_none());
        assert!(x_grid(&grids, XMetric::MotionScore, 16).is_none());
        assert!(x_grid(&grids, XMetric::MotionClass, 16).is_none());
        assert!(x_grid(&grids, XMetric::Variance, 16).is_some());
        assert!(x_grid(&grids, XMetric::Mean, 16).is_some());
        assert!(x_grid(&grids, XMetric::Orientation, 16).is_some());
        // With a previous frame, motion grids appear.
        let prev = vec![120u8; 16 * 16 * 3];
        let grids = compute_analysis_grids(&rgb, Some(&prev), 16, 16);
        assert!(grids.motion.is_some());
        assert!(x_grid(&grids, XMetric::MotionScore, 16).is_some());
        assert!(classes_at_g(&grids, 16).is_some());
    }
}
