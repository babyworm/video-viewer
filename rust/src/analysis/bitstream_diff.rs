//! M-D: Dual `.catb` Δ comparison (VQAnalyzer "Dual view" analogue, G27).
//!
//! Pure, egui-free diff engine over two rasterized [`BitstreamGrid`]s. The
//! two streams encode the same source with different settings (e.g.
//! preprocessing on/off), so their CU partitions differ — per-rect diffing
//! is meaningless. Both frames are rasterized onto the same fixed-cell grid
//! (L1 8 px canonical, LOD aggregate for far zoom) and compared cell-wise:
//! scalar metrics as signed `A − B`, prediction mode as an agreement mask
//! (VQA Δ convention: identical → grey, different → the cell keeps A's
//! mode colour).

use crate::analysis::bitstream_stats::{BitstreamGrid, ModeClass};

/// Scalar Δ metric selector (UI-independent mirror of the diffable fills).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffMetric {
    Qp,
    Bpp,
    MvMag,
    CoeffEnergy,
    NzDensity,
}

/// Cell-wise signed difference of two same-geometry [`BitstreamGrid`]s.
///
/// `valid[i]` requires **both** sides to have coverage in cell `i` — a cell
/// only one stream codes (possible on trailing partial frames) must not
/// masquerade as a difference of `value − 0`.
#[derive(Debug, Clone)]
pub struct DiffGrid {
    pub cell: u32,
    pub cols: u32,
    pub rows: u32,
    /// `qp_A − qp_B` per cell (0.0 where invalid).
    pub d_qp: Vec<f32>,
    pub d_bpp: Vec<f32>,
    pub d_mv: Vec<f32>,
    pub d_coeff: Vec<f32>,
    pub d_nz: Vec<f32>,
    /// A's majority mode per cell (Δ Mode rendering colours by A).
    pub mode_a: Vec<ModeClass>,
    /// B's majority mode per cell (tooltip / agreement).
    pub mode_b: Vec<ModeClass>,
    pub valid: Vec<bool>,
}

impl DiffGrid {
    pub fn is_empty(&self) -> bool {
        self.cols == 0 || self.rows == 0
    }

    /// Δ values for one scalar metric.
    pub fn values(&self, m: DiffMetric) -> &[f32] {
        match m {
            DiffMetric::Qp => &self.d_qp,
            DiffMetric::Bpp => &self.d_bpp,
            DiffMetric::MvMag => &self.d_mv,
            DiffMetric::CoeffEnergy => &self.d_coeff,
            DiffMetric::NzDensity => &self.d_nz,
        }
    }

    /// Symmetric legend scale: max |Δ| over the valid cells (0.0 when none).
    pub fn max_abs(&self, m: DiffMetric) -> f32 {
        self.values(m)
            .iter()
            .zip(self.valid.iter())
            .filter(|(_, v)| **v)
            .fold(0.0_f32, |acc, (d, _)| acc.max(d.abs()))
    }

    /// True when both sides agree on the cell's majority mode.
    pub fn mode_same(&self, i: usize) -> bool {
        self.mode_a.get(i) == self.mode_b.get(i)
    }
}

/// Build the cell-wise Δ of two grids. Returns `None` when the geometries
/// disagree (different cell size or dimensions) — the load-time resolution
/// gate makes this unreachable in the UI, but a malformed pair must degrade
/// to "no Δ", never to a misaligned subtraction.
pub fn diff_grids(a: &BitstreamGrid, b: &BitstreamGrid) -> Option<DiffGrid> {
    if a.cell != b.cell || a.cols != b.cols || a.rows != b.rows {
        return None;
    }
    let n = (a.cols as usize) * (a.rows as usize);
    let mut out = DiffGrid {
        cell: a.cell,
        cols: a.cols,
        rows: a.rows,
        d_qp: vec![0.0; n],
        d_bpp: vec![0.0; n],
        d_mv: vec![0.0; n],
        d_coeff: vec![0.0; n],
        d_nz: vec![0.0; n],
        mode_a: vec![ModeClass::Unknown; n],
        mode_b: vec![ModeClass::Unknown; n],
        valid: vec![false; n],
    };
    for i in 0..n {
        let ok = a.coverage[i] > 0.0 && b.coverage[i] > 0.0;
        out.valid[i] = ok;
        if !ok {
            continue;
        }
        out.d_qp[i] = a.qp[i] - b.qp[i];
        out.d_bpp[i] = a.bpp[i] - b.bpp[i];
        out.d_mv[i] = a.mv_mag[i] - b.mv_mag[i];
        out.d_coeff[i] = a.coeff_energy[i] - b.coeff_energy[i];
        out.d_nz[i] = a.nz_density[i] - b.nz_density[i];
        out.mode_a[i] = a.mode[i];
        out.mode_b[i] = b.mode[i];
    }
    Some(out)
}

/// Load-time compatibility gate for the comparison `.catb` (B): the streams
/// must share the coded resolution — a Δ over different geometries is
/// meaningless (the fixed-cell grids would not correspond). Dimensions of 0
/// (unknown resolution) are not comparable either.
pub fn validate_b_resolution(a: (u32, u32), b: (u32, u32)) -> Result<(), String> {
    if a.0 == 0 || a.1 == 0 || b.0 == 0 || b.1 == 0 {
        return Err(format!(
            "comparison rejected — unknown stream resolution (A {}×{}, B {}×{})",
            a.0, a.1, b.0, b.1
        ));
    }
    if a != b {
        return Err(format!(
            "comparison rejected — resolution mismatch: A {}×{} vs B {}×{}",
            a.0, a.1, b.0, b.1
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(cell: u32, cols: u32, rows: u32, qp: &[f32], cov: &[f32]) -> BitstreamGrid {
        let n = (cols * rows) as usize;
        BitstreamGrid {
            cell,
            cols,
            rows,
            qp: qp.to_vec(),
            bpp: vec![0.0; n],
            mv_mag: vec![0.0; n],
            mode: vec![ModeClass::Unknown; n],
            coeff_energy: vec![0.0; n],
            nz_density: vec![0.0; n],
            coverage: cov.to_vec(),
        }
    }

    #[test]
    fn diff_rejects_geometry_mismatch() {
        let a = grid(8, 2, 1, &[30.0, 30.0], &[1.0, 1.0]);
        let b8 = grid(8, 1, 1, &[30.0], &[1.0]);
        let b16 = grid(16, 2, 1, &[30.0, 30.0], &[1.0, 1.0]);
        assert!(diff_grids(&a, &b8).is_none());
        assert!(diff_grids(&a, &b16).is_none());
    }

    #[test]
    fn diff_requires_both_covered() {
        let a = grid(8, 2, 1, &[30.0, 40.0], &[1.0, 1.0]);
        let b = grid(8, 2, 1, &[28.0, 99.0], &[1.0, 0.0]);
        let d = diff_grids(&a, &b).unwrap();
        assert!(d.valid[0] && !d.valid[1]);
        assert!((d.d_qp[0] - 2.0).abs() < 1e-6);
        assert_eq!(d.d_qp[1], 0.0, "uncovered cell carries no fake Δ");
        assert!((d.max_abs(DiffMetric::Qp) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn validate_b_resolution_rules() {
        assert!(validate_b_resolution((64, 64), (64, 64)).is_ok());
        assert!(validate_b_resolution((64, 64), (128, 64)).is_err());
        assert!(validate_b_resolution((0, 0), (64, 64)).is_err());
    }
}
