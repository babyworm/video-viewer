//! Bitstream Analysis window — a separate OS viewport (UX spec 04, M1).
//!
//! Follows the sidebar.rs Frame Analysis window protocol exactly:
//! `show_viewport_immediate` (deferred viewports starve during playback),
//! `Arc<Mutex<BitstreamShared>>` for root↔child data, ColorImage handed
//! across the viewport boundary and texturized inside the child
//! (TextureHandle cannot cross viewports), `close_requested` flag,
//! `request_repaint_of` in both directions, and child self-repaint while
//! the root is playing.
//!
//! The child never seeks by itself: filmstrip/transport/keyboard writes a
//! `seek_request` into the shared state and the root executes `goto_frame`
//! (one-way pull model, §4).

use std::sync::Arc;

use eframe::egui;
use parking_lot::Mutex;

use crate::analysis::bitstream_diff::{diff_grids, DiffGrid, DiffMetric};
use crate::analysis::bitstream_panel::format_bits;
use crate::analysis::bitstream_stats::{
    aggregate_block_tx, build_filmstrip_refs, compute_frame_stats, hit_test_min_area,
    lod_cell_size, rasterize_blocks_tx, ref_count_tier, use_lod, viewer_to_catb_display,
    BitstreamFile, BitstreamGrid, BlockTxAgg, FilmstripRefs, FrameStatsData, FrameTypeClass,
    IntraDir, ModeClass,
};
use crate::analysis::correlation::{
    aggregate_bitstream_to_g, align, class_table, classes_at_g, csv_dump, opportunity_grid,
    pearson_r, spearman_rho, subtract_bitstream_g, top_n_ranking, x_grid, AlignedPair,
    AnalysisFrameGrids, BitstreamG, CorrScanRequest, CorrScanResult, XMetric, YMetric, G_SIZES,
    PRESET_PAIRS,
};
use crate::analysis::motion::MotionClass;
use crate::analysis::stage::{
    compute_block_quality, psnr_to_g, stage_available, BlockQuality, QualityUnavailable,
    StageCache, StageKind,
};
use crate::core::catb::{BsBlock, BsRef, TxRow};
use crate::core::dropped::{classify_dropped_file, DroppedKind};
use crate::ui::bitstream_overlay::{
    build_intra, build_refs, build_tx, draw_intra_layer, draw_mv_layer, draw_part_layer,
    draw_tu_layer, LayerGeom, MvSource, MV_SOURCES,
};
use crate::ui::settings::BitstreamViewSettings;
use crate::ui::sideband_overlay::diverging_colormap;

// ---------------------------------------------------------------------------
// Fill / layer / preset model (§2, §3) — pure data, unit-tested below.
// ---------------------------------------------------------------------------

/// Exclusive fill layer (§2). Opportunity (M3) renders the correlation
/// z-mismatch on G cells, not L0 rects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillMode {
    None,
    Qp,
    Bpp,
    Mode,
    MvHeat,
    Opportunity,
    /// M-B: TX coeff_abs_sum energy per pixel (frame-max normalized,
    /// orange→red ramp).
    CoeffEnergy,
    /// M-B: nonzero-coefficient density per pixel.
    NonzeroCoeffs,
    /// M-C: per-block luma PSNR — `final_recon` stage image vs the viewer's
    /// source frame (VQA Info-Overlay PSNR analogue). Like Opportunity it
    /// is not a [`FillSample`] colour: the values live in the per-frame
    /// [`crate::analysis::stage::BlockQuality`] and render in a dedicated
    /// pass.
    BlockPsnr,
    /// M-C: per-block luma SSIM, same data path as [`FillMode::BlockPsnr`].
    /// Combo-only (no shortcut key — 1–9 are taken).
    BlockSsim,
}

/// All fill modes in §8 shortcut order (keys 1–9; BlockSSIM is combo-only).
pub const FILL_MODES: [FillMode; 10] = [
    FillMode::None,
    FillMode::Qp,
    FillMode::Bpp,
    FillMode::Mode,
    FillMode::MvHeat,
    FillMode::Opportunity,
    FillMode::CoeffEnergy,
    FillMode::NonzeroCoeffs,
    FillMode::BlockPsnr,
    FillMode::BlockSsim,
];

impl FillMode {
    pub fn label(&self) -> &'static str {
        match self {
            FillMode::None => "None",
            FillMode::Qp => "QP",
            FillMode::Bpp => "bpp",
            FillMode::Mode => "Mode",
            FillMode::MvHeat => "MV-heat",
            FillMode::Opportunity => "Opportunity",
            FillMode::CoeffEnergy => "CoeffEnergy",
            FillMode::NonzeroCoeffs => "NonzeroCoeffs",
            FillMode::BlockPsnr => "BlockPSNR",
            FillMode::BlockSsim => "BlockSSIM",
        }
    }

    pub fn from_label(s: &str) -> Self {
        match s {
            "QP" => FillMode::Qp,
            "bpp" => FillMode::Bpp,
            "Mode" => FillMode::Mode,
            "MV-heat" => FillMode::MvHeat,
            "Opportunity" => FillMode::Opportunity,
            "CoeffEnergy" => FillMode::CoeffEnergy,
            "NonzeroCoeffs" => FillMode::NonzeroCoeffs,
            "BlockPSNR" => FillMode::BlockPsnr,
            "BlockSSIM" => FillMode::BlockSsim,
            _ => FillMode::None,
        }
    }

    /// True for the M-C quality fills, which render from [`BlockQuality`]
    /// in their own pass (like Opportunity renders from the aligned pair).
    pub fn is_quality(&self) -> bool {
        matches!(self, FillMode::BlockPsnr | FillMode::BlockSsim)
    }

    /// M-D: fills the dual-`.catb` Δ mode can render. Scalar fills map to a
    /// signed cell difference, Mode to the agreement mask. Opportunity and
    /// the quality fills draw from A-only data paths (aligned pair /
    /// BlockQuality) — no B counterpart exists, so the Δ toggle disables.
    pub fn supports_diff(&self) -> bool {
        self.diff_metric().is_some() || *self == FillMode::Mode
    }

    /// The [`DiffMetric`] a scalar fill diffs on (`None` for Mode and the
    /// unsupported fills).
    pub fn diff_metric(&self) -> Option<DiffMetric> {
        match self {
            FillMode::Qp => Some(DiffMetric::Qp),
            FillMode::Bpp => Some(DiffMetric::Bpp),
            FillMode::MvHeat => Some(DiffMetric::MvMag),
            FillMode::CoeffEnergy => Some(DiffMetric::CoeffEnergy),
            FillMode::NonzeroCoeffs => Some(DiffMetric::NzDensity),
            _ => None,
        }
    }
}

/// The §2 toolbar tuple: fill + independent toggle layers + fill-only opacity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewConfig {
    pub fill: FillMode,
    /// MV arrows (M4): per-PU arrows from REF geometry, block fallback.
    pub mv: bool,
    /// Partition outlines (M4): CU on every block + per-PU boundaries.
    pub part: bool,
    /// TU outlines (M-B): yellow, depth-shaded; cbf=0 dashed;
    /// transform_skip shaded.
    pub tu: bool,
    /// Intra direction lines + P/DC badges (M4).
    pub intra: bool,
    /// Per-block value text, rendered at zoom ≥ 2x only.
    pub label: bool,
    /// CTU (HEVC 64 px) / 4-MB (AVC) boundary grid.
    pub grid: bool,
    /// Selected-block highlight.
    pub sel: bool,
    /// Fill alpha only (lines/text always full contrast). Default 0.6.
    pub opacity: f32,
}

impl Default for ViewConfig {
    fn default() -> Self {
        Preset::QpMap.config()
    }
}

impl ViewConfig {
    pub fn from_settings(s: &BitstreamViewSettings) -> Self {
        Self {
            fill: FillMode::from_label(&s.fill),
            mv: s.layer_mv,
            part: s.layer_part,
            tu: s.layer_tu,
            intra: s.layer_intra,
            label: s.layer_label,
            grid: s.layer_grid,
            sel: s.layer_sel,
            opacity: s.opacity.clamp(0.0, 1.0),
        }
    }

    pub fn to_settings(
        self,
        show_loupe: bool,
        inspector_collapsed: bool,
        mv_source: MvSource,
    ) -> BitstreamViewSettings {
        BitstreamViewSettings {
            fill: self.fill.label().to_string(),
            layer_mv: self.mv,
            layer_part: self.part,
            layer_tu: self.tu,
            layer_intra: self.intra,
            layer_label: self.label,
            layer_grid: self.grid,
            layer_sel: self.sel,
            opacity: self.opacity,
            show_loupe,
            inspector_collapsed,
            mv_source: mv_source.label().to_string(),
        }
    }
}

/// One-click configurations (§3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    Rate,
    QpMap,
    Motion,
    Mode,
    Opportunity,
    Clean,
}

pub const PRESETS: [Preset; 6] = [
    Preset::Rate,
    Preset::QpMap,
    Preset::Motion,
    Preset::Mode,
    Preset::Opportunity,
    Preset::Clean,
];

impl Preset {
    pub fn label(&self) -> &'static str {
        match self {
            Preset::Rate => "Rate",
            Preset::QpMap => "QP Map",
            Preset::Motion => "Motion",
            Preset::Mode => "Mode",
            Preset::Opportunity => "Opportunity",
            Preset::Clean => "Clean",
        }
    }

    /// §3 table. Presets are starting points — free edits switch the combo
    /// to "Custom". `sel` stays on except for Clean (screenshot mode).
    pub fn config(&self) -> ViewConfig {
        let base = ViewConfig {
            fill: FillMode::None,
            mv: false,
            part: false,
            tu: false,
            intra: false,
            label: false,
            grid: false,
            sel: true,
            opacity: 0.6,
        };
        match self {
            Preset::Rate => ViewConfig {
                fill: FillMode::Bpp,
                grid: true,
                ..base
            },
            Preset::QpMap => ViewConfig {
                fill: FillMode::Qp,
                grid: true,
                label: true,
                ..base
            },
            // 04 §3 M4 intent: Motion = MV-heat fill + live MV arrows.
            Preset::Motion => ViewConfig {
                fill: FillMode::MvHeat,
                mv: true,
                grid: true,
                ..base
            },
            // 04 §3 M4 intent: Mode = Mode fill + Part outlines + Intra dirs.
            Preset::Mode => ViewConfig {
                fill: FillMode::Mode,
                part: true,
                intra: true,
                grid: true,
                ..base
            },
            // §3: {fill=Opportunity, Grid, Sel} — ranked-cell traversal.
            Preset::Opportunity => ViewConfig {
                fill: FillMode::Opportunity,
                grid: true,
                ..base
            },
            Preset::Clean => ViewConfig {
                sel: false,
                ..base
            },
        }
    }
}

/// Which preset (if any) matches the current config — `None` renders as
/// "Custom" in the combo.
pub fn matching_preset(cfg: &ViewConfig) -> Option<Preset> {
    PRESETS.into_iter().find(|p| {
        let c = p.config();
        c.fill == cfg.fill
            && c.mv == cfg.mv
            && c.part == cfg.part
            && c.tu == cfg.tu
            && c.intra == cfg.intra
            && c.label == cfg.label
            && c.grid == cfg.grid
            && c.sel == cfg.sel
            && (c.opacity - cfg.opacity).abs() < 1e-3
    })
}

/// Tab-key preset cycling (§8): Custom → Rate, otherwise the next preset.
pub fn next_preset(cfg: &ViewConfig) -> Preset {
    match matching_preset(cfg) {
        None => Preset::Rate,
        Some(p) => {
            let i = PRESETS.iter().position(|q| *q == p).unwrap_or(0);
            PRESETS[(i + 1) % PRESETS.len()]
        }
    }
}

// ---------------------------------------------------------------------------
// View math helpers — pure, unit-tested.
// ---------------------------------------------------------------------------

/// New pan that keeps the image point under `anchor` (canvas-relative)
/// fixed across `old_zoom → new_zoom`. Same derivation as the sidebar
/// heatmap panes: `(anchor − pan) / zoom` invariant.
pub fn zoom_anchor_pan(
    pan: egui::Vec2,
    old_zoom: f32,
    new_zoom: f32,
    anchor: egui::Vec2,
) -> egui::Vec2 {
    let ratio = new_zoom / old_zoom;
    anchor * (1.0 - ratio) + pan * ratio
}

/// Normalize `v` into [0,1] over [min,max]; degenerate range → 1.0 so a
/// uniform frame still renders visibly.
pub fn normalize(v: f32, min: f32, max: f32) -> f32 {
    if max - min <= f32::EPSILON {
        1.0
    } else {
        ((v - min) / (max - min)).clamp(0.0, 1.0)
    }
}

/// Filmstrip cell colour (§4): I=red / P=blue / B=green; no data = dark grey.
pub fn filmstrip_color(class: Option<FrameTypeClass>) -> egui::Color32 {
    match class {
        Some(FrameTypeClass::I) => egui::Color32::from_rgb(205, 62, 62),
        Some(FrameTypeClass::P) => egui::Color32::from_rgb(72, 112, 220),
        Some(FrameTypeClass::B) => egui::Color32::from_rgb(70, 175, 92),
        Some(FrameTypeClass::Other) => egui::Color32::from_rgb(120, 120, 120),
        None => egui::Color32::from_rgb(52, 52, 56),
    }
}

/// Discrete Mode-fill swatch (§5 legend): Intra cyan / Inter blue /
/// Skip yellow / Merge green.
pub fn mode_color(m: ModeClass) -> egui::Color32 {
    match m {
        ModeClass::Intra => egui::Color32::from_rgb(0, 190, 190),
        ModeClass::Inter => egui::Color32::from_rgb(70, 105, 225),
        ModeClass::Skip => egui::Color32::from_rgb(228, 195, 50),
        ModeClass::Merge => egui::Color32::from_rgb(70, 180, 90),
        ModeClass::Unknown => egui::Color32::from_rgb(110, 110, 110),
    }
}

/// Per-frame value ranges used by the fills and the legend (real min/max,
/// recomputed each frame so the legend always matches the colours).
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameFillStats {
    pub qp_min: f32,
    pub qp_max: f32,
    pub bpp_max: f32,
    /// Max L1-norm |MV| in **pixels** (quarter-pel / 4) — the one unit every
    /// consumer shares (L1 grid, LOD grid, labels, loupe, legend).
    pub mv_max: f32,
    /// Max TX coeff_abs_sum energy per pixel (M-B CoeffEnergy fill).
    pub coeff_max: f32,
    /// Max nonzero-coefficient density per pixel (M-B NonzeroCoeffs fill).
    pub nz_max: f32,
}

/// One fill-value sample: everything a fill mode can colour/label, whether
/// it came from an L0 block or an L1/LOD grid cell (M-B: the growing
/// per-sample tuple outgrew positional parameters).
#[derive(Debug, Clone, Copy)]
pub struct FillSample {
    pub qp: f32,
    pub bpp: f32,
    pub mode: ModeClass,
    /// |MV| in pixels.
    pub mv: f32,
    /// TX coeff_abs_sum energy per pixel.
    pub coeff: f32,
    /// Nonzero-coefficient density per pixel.
    pub nz: f32,
}

impl Default for FillSample {
    fn default() -> Self {
        Self {
            qp: 0.0,
            bpp: 0.0,
            mode: ModeClass::Unknown,
            mv: 0.0,
            coeff: 0.0,
            nz: 0.0,
        }
    }
}

impl FillSample {
    /// Sample of one L0 block (+ its optional TX aggregate).
    pub fn from_block(b: &BsBlock, tx: Option<&BlockTxAgg>) -> Self {
        let area = (b.w as f64 * b.h as f64).max(1.0);
        let agg = tx.copied().unwrap_or_default();
        Self {
            qp: b.qp as f32,
            bpp: (b.bits.max(0) as f64 / area) as f32,
            mode: ModeClass::from_label(&b.prediction_mode),
            mv: block_mv_px(b),
            coeff: (agg.abs_sum.max(0.0) / area) as f32,
            nz: (agg.nonzero.max(0.0) / area) as f32,
        }
    }

    /// Sample of one L1/LOD grid cell.
    pub fn from_grid(g: &BitstreamGrid, i: usize) -> Self {
        Self {
            qp: g.qp[i],
            bpp: g.bpp[i],
            mode: g.mode[i],
            mv: g.mv_mag[i],
            coeff: g.coeff_energy[i],
            nz: g.nz_density[i],
        }
    }
}

/// L1-norm MV magnitude of a block in **pixels** (quarter-pel `|x|+|y|` / 4)
/// — same unit as `BitstreamGrid::mv_mag`, so the L0 rect path and the
/// L1/LOD grid path colour and label identically.
pub fn block_mv_px(b: &BsBlock) -> f32 {
    b.mv
        .map(|(x, y)| (x.abs() + y.abs()) as f32 / 4.0)
        .unwrap_or(0.0)
}

/// Compute fill statistics over a frame's L0 block list (+ its optional
/// per-block TX aggregate, parallel — see [`BlockTxAgg`]).
pub fn frame_fill_stats(blocks: &[BsBlock], tx: Option<&[BlockTxAgg]>) -> FrameFillStats {
    let mut s = FrameFillStats {
        qp_min: f32::INFINITY,
        qp_max: f32::NEG_INFINITY,
        bpp_max: 0.0,
        mv_max: 0.0,
        coeff_max: 0.0,
        nz_max: 0.0,
    };
    for (i, b) in blocks.iter().enumerate() {
        let qp = b.qp as f32;
        s.qp_min = s.qp_min.min(qp);
        s.qp_max = s.qp_max.max(qp);
        let area = (b.w as f64 * b.h as f64).max(1.0);
        s.bpp_max = s.bpp_max.max(b.bits.max(0) as f32 / area as f32);
        s.mv_max = s.mv_max.max(block_mv_px(b));
        if let Some(agg) = tx.and_then(|t| t.get(i)) {
            s.coeff_max = s.coeff_max.max((agg.abs_sum.max(0.0) / area) as f32);
            s.nz_max = s.nz_max.max((agg.nonzero.max(0.0) / area) as f32);
        }
    }
    if !s.qp_min.is_finite() {
        s.qp_min = 0.0;
        s.qp_max = 0.0;
    }
    s
}

/// Fill colour for a sample under the current fill mode. `opacity` applies
/// to the fill only (§2).
pub fn fill_color(
    fill: FillMode,
    s: &FillSample,
    stats: &FrameFillStats,
    opacity: f32,
) -> Option<egui::Color32> {
    let a255 = |t: f32| ((opacity * t).clamp(0.0, 1.0) * 255.0) as u8;
    let ramp = |v: f32, max: f32| if max > 0.0 { (v / max).clamp(0.0, 1.0) } else { 0.0 };
    match fill {
        // Opportunity is a G-cell layer drawn from the aligned pair, and
        // the M-C quality fills draw from BlockQuality — neither is a
        // per-sample colour (see `opportunity_cell_color` /
        // `quality_fill_color`).
        FillMode::None
        | FillMode::Opportunity
        | FillMode::BlockPsnr
        | FillMode::BlockSsim => None,
        FillMode::Qp => {
            let t = normalize(s.qp, stats.qp_min, stats.qp_max);
            Some(egui::Color32::from_rgba_unmultiplied(
                225,
                40,
                40,
                a255(0.15 + 0.85 * t),
            ))
        }
        FillMode::Bpp => {
            let t = ramp(s.bpp, stats.bpp_max);
            Some(egui::Color32::from_rgba_unmultiplied(
                255,
                150,
                20,
                a255(0.10 + 0.90 * t),
            ))
        }
        FillMode::Mode => {
            let c = mode_color(s.mode);
            Some(egui::Color32::from_rgba_unmultiplied(
                c.r(),
                c.g(),
                c.b(),
                a255(1.0),
            ))
        }
        FillMode::MvHeat => {
            let t = ramp(s.mv, stats.mv_max);
            Some(egui::Color32::from_rgba_unmultiplied(
                55,
                125,
                255,
                a255(0.10 + 0.90 * t),
            ))
        }
        // M-B: orange→red ramp (hue shifts with t, frame-max normalized).
        FillMode::CoeffEnergy => {
            let t = ramp(s.coeff, stats.coeff_max);
            Some(egui::Color32::from_rgba_unmultiplied(
                255,
                (140.0 - 110.0 * t) as u8,
                10,
                a255(0.10 + 0.90 * t),
            ))
        }
        FillMode::NonzeroCoeffs => {
            let t = ramp(s.nz, stats.nz_max);
            Some(egui::Color32::from_rgba_unmultiplied(
                210,
                70,
                220,
                a255(0.10 + 0.90 * t),
            ))
        }
    }
}

/// Short value text for tooltips / labels / loupe under a fill mode.
pub fn fill_value_text(fill: FillMode, s: &FillSample) -> String {
    match fill {
        // Opportunity z lives on G cells and the quality values on
        // BlockQuality, not blocks — fall back to QP for block-level text
        // (labels / loupe; the quality label layer overrides this with
        // `quality_value_text`).
        FillMode::None
        | FillMode::Qp
        | FillMode::Opportunity
        | FillMode::BlockPsnr
        | FillMode::BlockSsim => format!("{:.0}", s.qp),
        FillMode::Bpp => format!("{:.2}", s.bpp),
        FillMode::Mode => s.mode.label().to_string(),
        FillMode::MvHeat => format!("{:.1}", s.mv),
        FillMode::CoeffEnergy => format!("{:.1}", s.coeff),
        FillMode::NonzeroCoeffs => format!("{:.2}", s.nz),
    }
}

/// Diverging fill colour for an opportunity z value on the symmetric scale
/// −zmax..+zmax (blue = negative, white = 0, red = positive). `opacity`
/// applies to the fill only (§2).
pub fn opportunity_cell_color(z: f32, zmax: f32, opacity: f32) -> egui::Color32 {
    let t = if zmax > 0.0 {
        (0.5 + z / (2.0 * zmax)).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let c = diverging_colormap(t as f64);
    egui::Color32::from_rgba_unmultiplied(
        c.r(),
        c.g(),
        c.b(),
        (opacity.clamp(0.0, 1.0) * 255.0) as u8,
    )
}

/// Quarter-pel MV pair display, e.g. `(-3.25, 0.5)`.
pub fn qpel(pair: (i32, i32)) -> String {
    format!("({}, {})", pair.0 as f32 / 4.0, pair.1 as f32 / 4.0)
}

// ---------------------------------------------------------------------------
// M-C: Picture selector + block-quality fills
// ---------------------------------------------------------------------------

/// Which picture the Viewer-tab canvas shows as the background texture
/// (VQA "Pic" selector analogue). Session-only, resets to Source on file
/// change. Window-only by design: the main-canvas overlay mirror keeps the
/// viewer's own frame — the root pipeline (pixel inspector, analysis tabs,
/// comparison) reads `current_rgb`, and swapping its texture to a decoder
/// stage would desynchronize every one of those readouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PictureSource {
    /// The viewer's own video frame (default).
    Source,
    /// A decoder-run stage image (`final_recon` / `recon_unfiltered` /
    /// `prediction` / `residual`).
    Stage(StageKind),
}

impl PictureSource {
    pub const ALL: [PictureSource; 5] = [
        PictureSource::Source,
        PictureSource::Stage(StageKind::FinalRecon),
        PictureSource::Stage(StageKind::ReconUnfiltered),
        PictureSource::Stage(StageKind::Prediction),
        PictureSource::Stage(StageKind::Residual),
    ];

    pub fn label(self) -> &'static str {
        match self {
            PictureSource::Source => "Source",
            PictureSource::Stage(k) => k.label(),
        }
    }
}

/// Fill colour for one block's quality value (worst = most opaque, so bad
/// blocks pull the eye like the other "cost" ramps). Bit-exact blocks
/// (`psnr = ∞`, the `e` blocks) render fully transparent — nothing to fix.
pub fn quality_fill_color(
    fill: FillMode,
    v: f32,
    q: &BlockQuality,
    opacity: f32,
) -> Option<egui::Color32> {
    if !v.is_finite() {
        return None;
    }
    let (lo, hi) = match fill {
        FillMode::BlockPsnr => (q.psnr_min, q.psnr_max),
        FillMode::BlockSsim => (q.ssim_min, q.ssim_max),
        _ => return None,
    };
    // High PSNR/SSIM = good = faint; low = bad = strong red.
    let t = 1.0 - normalize(v, lo, hi);
    let a = ((opacity * (0.10 + 0.90 * t)).clamp(0.0, 1.0) * 255.0) as u8;
    Some(egui::Color32::from_rgba_unmultiplied(230, 50, 50, a))
}

/// Block label / tooltip text for a quality value. Bit-exact PSNR renders
/// as `e` (VQA convention).
pub fn quality_value_text(fill: FillMode, v: f32) -> String {
    match fill {
        FillMode::BlockPsnr => {
            if v.is_finite() {
                format!("{v:.1}")
            } else {
                "e".to_string()
            }
        }
        FillMode::BlockSsim => format!("{v:.3}"),
        _ => String::new(),
    }
}

/// BT.601 luma plane of an egui `ColorImage` — the same weights as
/// [`crate::analysis::stage::luma_bt601`], so viewer-side and BMP-side
/// planes are directly comparable.
fn color_image_luma(img: &egui::ColorImage) -> Vec<f32> {
    img.pixels
        .iter()
        .map(|p| 0.299 * p.r() as f32 + 0.587 * p.g() as f32 + 0.114 * p.b() as f32)
        .collect()
}

// ---------------------------------------------------------------------------
// Shared state (root ↔ child viewport)
// ---------------------------------------------------------------------------

/// One point of the Frame Graph tab (display order).
#[derive(Debug, Clone, Copy)]
pub struct FrameGraphPoint {
    pub display_idx: usize,
    pub bits: f64,
    pub avg_qp: f64,
}

/// Data shared between the root window and the bitstream viewport.
/// Root writes frame data + playback state; child writes requests back.
pub struct BitstreamShared {
    /// The loaded telemetry, shared so the child can read blocks/refs itself.
    pub file: Option<Arc<BitstreamFile>>,
    /// Basename of the loaded .catb for the window title (§1 mockup).
    pub file_name: Option<String>,
    /// Current viewer frame as RGB (TextureHandle can't cross viewports).
    /// `Arc` so the child's snapshot shares it instead of cloning ~8 MB of
    /// pixels per 1080p frame during playback.
    pub frame_image: Option<Arc<egui::ColorImage>>,
    /// Bumped whenever `frame_image` changes (texture re-upload gate).
    pub generation: u64,
    pub viewer_frame: usize,
    pub viewer_total: usize,
    pub viewer_size: Option<(u32, u32)>,
    /// R4 manual frame offset (viewer + offset = catb display index).
    pub offset: i32,
    /// Mirrors root playback so the child self-repaints (0.8.1 lesson).
    pub is_playing: bool,
    /// Child → root: jump to this viewer frame.
    pub seek_request: Option<usize>,
    /// Child → root: toggle playback.
    pub toggle_play_request: bool,
    /// Child → root: set the frame offset (panel state is shared, §9).
    pub offset_request: Option<i32>,
    /// Child → root: window was closed.
    pub close_requested: bool,
    /// Child → root: a file was dropped on this window, pre-classified
    /// (0.15.0). Processing is root-owned (seek_request pattern): the root
    /// routes it exactly like a main-window drop — the child never loads
    /// anything itself.
    pub drop_request: Option<(std::path::PathBuf, DroppedKind)>,
    /// Frame Graph series, filled by a background scan thread on load.
    pub frame_graph: Option<Arc<Vec<FrameGraphPoint>>>,
    pub frame_graph_scanning: bool,
    /// M-D: the comparison stream (B) — root-published, never loaded here.
    pub file_b: Option<Arc<BitstreamFile>>,
    /// M-D: B's own viewer↔catb frame offset.
    pub offset_b: i32,
    /// M-D: B's Frame Graph series (own background scan on B load).
    pub frame_graph_b: Option<Arc<Vec<FrameGraphPoint>>>,
    pub frame_graph_b_scanning: bool,
    /// Child → root: the Correlation tab is visible, so the root computes
    /// analysis grids for the current frame (update_analysis convention).
    pub corr_active: bool,
    /// Root-pushed analysis-side grids (8 px) for `frame_idx`.
    pub corr_analysis: Option<Arc<CorrAnalysisData>>,
    /// Child → root: run a frame-range scan on a background thread.
    pub corr_scan_request: Option<CorrScanRequest>,
    /// Range-scan output (background thread, root-owned job).
    pub corr_scan: Option<Arc<CorrScanResult>>,
    pub corr_scanning: bool,
    /// (frames done, frames total) while scanning.
    pub corr_scan_progress: (usize, usize),
    /// Root-pushed scene-change frame indices (existing detection results
    /// only — the window never triggers a detection run). Timeline markers.
    pub scene_changes: Vec<usize>,
    /// Child → root: scatter-hovered opportunity cell `(col, row, g)` in
    /// stream px grid units — mirrored as a highlight on the main viewer
    /// canvas (02 §2 view 1). Kept until replaced, like the window's own
    /// marker; Esc clears it.
    pub corr_hover_cell: Option<(u32, u32, u32)>,
}

/// Analysis-side grids for one viewer frame, computed by the root
/// (`app.rs`) — the child never touches the decoder.
#[derive(Debug)]
pub struct CorrAnalysisData {
    pub frame_idx: usize,
    pub grids: AnalysisFrameGrids,
}

impl Default for BitstreamShared {
    fn default() -> Self {
        Self::new()
    }
}

impl BitstreamShared {
    pub fn new() -> Self {
        Self {
            file: None,
            file_name: None,
            frame_image: None,
            generation: 0,
            viewer_frame: 0,
            viewer_total: 0,
            viewer_size: None,
            offset: 0,
            is_playing: false,
            seek_request: None,
            toggle_play_request: false,
            offset_request: None,
            close_requested: false,
            drop_request: None,
            frame_graph: None,
            frame_graph_scanning: false,
            file_b: None,
            offset_b: 0,
            frame_graph_b: None,
            frame_graph_b_scanning: false,
            corr_active: false,
            corr_analysis: None,
            corr_scan_request: None,
            corr_scan: None,
            corr_scanning: false,
            corr_scan_progress: (0, 0),
            scene_changes: Vec::new(),
            corr_hover_cell: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BsTab {
    Viewer,
    Correlation,
    FrameGraph,
    /// M-A: parameter sets / slice headers / DPB / exactness (VQA Syntax
    /// panel + Unit Info DPB tab analogue).
    Structure,
    /// M-B: syntax-element aggregate + CABAC summary + MV scatter (VQA
    /// Stats tab analogue, current frame only).
    Stats,
}

/// Selection: catb display frame + index into that frame's block list.
/// Kept across frame changes (§5: selection persists; Esc / empty click
/// clears it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Selection {
    display_idx: usize,
    block_idx: usize,
}

/// Per-frame derived render data, cached by catb display index **and** file
/// identity — a `.catb` loaded over another one must not reuse the old
/// file's blocks for the same display index.
struct FrameDerived {
    file_ptr: usize,
    display_idx: usize,
    blocks: Arc<Vec<BsBlock>>,
    /// Per-block TX aggregate (M-B), parallel to `blocks` — eager because
    /// the L1/LOD grids bake it in (one cheap fixed-field pass over the
    /// frame's TX records; empty TRANSFORMS section ⇒ all zeros).
    tx_agg: Vec<BlockTxAgg>,
    stats: FrameFillStats,
    /// L1 canonical 8-px grid (loupe).
    grid_l1: BitstreamGrid,
    /// LOD aggregate grid (64 px HEVC / 32 px AVC) for zoom < 1.5.
    grid_lod: BitstreamGrid,
    /// Per-block REF rows, parallel to `blocks` — built lazily the first
    /// time the MV or Part layer needs them for this frame (M4).
    refs: Option<Vec<Vec<BsRef>>>,
    /// Per-block intra dirs, parallel to `blocks` — lazily for the Intra
    /// layer (M4).
    intra: Option<Vec<Vec<IntraDir>>>,
    /// Per-block TX rows, parallel to `blocks` — lazily for the TU layer
    /// and the Inspector TU table (M-B).
    tx: Option<Vec<Vec<TxRow>>>,
    /// M-C: per-block PSNR/SSIM of `final_recon` vs the viewer frame —
    /// lazily when a quality fill is active, keyed on the viewer image
    /// generation (the frame pixels can arrive after the block data).
    quality: Option<(u64, Result<BlockQuality, QualityUnavailable>)>,
}

/// Snapshot of the shared state taken at the top of the child pass so the
/// mutex is never held while rendering (immediate viewports run inside the
/// root update — re-locking under a held lock would deadlock).
struct Snapshot {
    file: Option<Arc<BitstreamFile>>,
    image: Option<Arc<egui::ColorImage>>,
    /// The current viewer frame pixels, always present when loaded (Arc
    /// clone — cheap). `image` above is texture-upload-gated; the M-C
    /// quality fills and the ReconPsnr Y metric need the pixels regardless
    /// of whether the texture already moved.
    frame_pixels: Option<Arc<egui::ColorImage>>,
    generation: u64,
    viewer_frame: usize,
    viewer_total: usize,
    viewer_size: Option<(u32, u32)>,
    offset: i32,
    is_playing: bool,
    frame_graph: Option<Arc<Vec<FrameGraphPoint>>>,
    frame_graph_scanning: bool,
    /// M-D: comparison stream + its offset (None when not loaded).
    file_b: Option<Arc<BitstreamFile>>,
    offset_b: i32,
    frame_graph_b: Option<Arc<Vec<FrameGraphPoint>>>,
    frame_graph_b_scanning: bool,
    corr_analysis: Option<Arc<CorrAnalysisData>>,
    corr_scan: Option<Arc<CorrScanResult>>,
    corr_scanning: bool,
    corr_scan_progress: (usize, usize),
    scene_changes: Vec<usize>,
}

pub struct BitstreamWindow {
    /// Whether the OS window is open (root toggles via menu/panel/CLI).
    pub open: bool,
    pub shared: Arc<Mutex<BitstreamShared>>,

    // UI state below is only touched inside the viewport closure, which the
    // root runs synchronously — plain fields, no lock needed.
    tab: BsTab,
    pub view: ViewConfig,
    /// Which vector the MV layer draws (M4). Deliberately outside
    /// `ViewConfig`: it must not break preset matching (a preset prescribes
    /// *that* MV arrows show, not which vector they trace).
    pub mv_source: MvSource,
    pub show_loupe: bool,
    pub inspector_collapsed: bool,
    /// Screen px per *viewer* image px; `None` = fit to canvas.
    zoom: Option<f32>,
    pan: egui::Vec2,
    selection: Option<Selection>,
    /// Frame texture inside the child viewport.
    texture: Option<egui::TextureHandle>,
    texture_gen: u64,
    derived: Option<FrameDerived>,
    /// Show per-frame bits / avg-QP series in the Frame Graph tab.
    graph_show_bits: bool,
    graph_show_qp: bool,

    // -- Correlation tab (M2) state --
    corr_x: XMetric,
    corr_y: YMetric,
    corr_g: u32,
    /// false = current frame, true = frame range.
    corr_range_mode: bool,
    corr_range: (usize, usize),
    corr_csv_path: String,
    /// Transient CSV save status line.
    corr_csv_status: Option<String>,
    /// Cached current-frame correlation derivation.
    corr_derived: Option<CorrDerived>,
    /// Cached range-scan statistics + scatter points (see
    /// [`Self::refresh_corr_scan_derived`]).
    corr_scan_derived: Option<CorrScanDerived>,

    // -- M3: opportunity map + bidirectional highlight state --
    /// Scatter-hovered G cell `(col, row, g)`: faint highlight on the Viewer
    /// canvas, kept until replaced (window-local, §6).
    corr_hover_cell: Option<(u32, u32, u32)>,
    /// Selected opportunity G cell `(col, row, g)`: Sel-layer highlight on
    /// the canvas, emphasized scatter point, Inspector raw values.
    opp_focus: Option<(u32, u32, u32)>,
    /// Stream-px point to centre the canvas on (Top-N [Jump], V22) — applied
    /// on the next canvas pass where the effective zoom is known.
    pending_center: Option<egui::Pos2>,

    /// Filmstrip frame-type classes, cached per (file, offset, total) — the
    /// strip repaints every frame during playback and recomputing a
    /// `frame_summary` (String clone) per cell per repaint is O(total)
    /// allocations at 30 fps.
    filmstrip_cache: Option<FilmstripCache>,
    /// M-A §C: filmstrip reference arrows + frequency dots toggle
    /// (session-only; default on — one arc set + tiny dots per repaint).
    show_refs: bool,
    /// M-A §B: Structure tab parameter-set field name filter.
    structure_filter: String,
    /// M-B: Inspector TU-table selection (index into the selected block's
    /// TX rows) — opens the coefficient detail section. Cleared whenever
    /// the block selection changes.
    selected_tu: Option<usize>,
    /// M-B: Stats tab per-frame aggregate, keyed on (file, display_idx).
    stats_cache: Option<(usize, usize, FrameStatsData)>,
    /// M-B: Stats tab syntax-element name filter.
    stats_filter: String,
    /// M-C: Viewer-tab background picture (VQA "Pic" selector). Session-
    /// only, reset to Source on file change — see [`PictureSource`].
    picture: PictureSource,
    /// M-C: lazy stage-image loader (small LRU, file-identity keyed).
    stage_cache: StageCache,
    /// M-C: uploaded stage texture, keyed on (file, decode_idx, kind).
    stage_texture: Option<StageTexture>,
    /// M-D: Δ mode — render the fill as the A−B cell difference (X key /
    /// toolbar Δ toggle). Session-only and B-bound: forced off whenever no
    /// comparison stream is loaded. Deliberately outside `ViewConfig`
    /// (would break preset matching and has no meaning without B).
    pub diff_mode: bool,
    /// M-D: per-frame Δ derivation cache (A/B grids diffed at L1 + LOD).
    diff: Option<DiffDerived>,
}

/// M-D: cached per-frame Δ data, keyed on both file identities, the viewer
/// frame and both offsets (any of them re-maps the A/B pair).
struct DiffDerived {
    file_a_ptr: usize,
    file_b_ptr: usize,
    viewer_frame: usize,
    offset: i32,
    offset_b: i32,
    /// B's blocks for the inspector's same-position hit test; `None` when B
    /// has no frame mapped to this viewer index.
    blocks_b: Option<Arc<Vec<BsBlock>>>,
    /// Cell-wise Δ at the canonical L1 (8 px) grid.
    grid_l1: Option<DiffGrid>,
    /// Cell-wise Δ at the LOD aggregate size (A codec's CTU/4-MB cell).
    grid_lod: Option<DiffGrid>,
    /// Why there is no Δ for this frame (legend slot), e.g. B frame missing.
    reason: Option<String>,
}

/// The stage image currently uploaded as a child-viewport texture (M-C).
struct StageTexture {
    handle: egui::TextureHandle,
    file_ptr: usize,
    decode_idx: usize,
    kind: StageKind,
}

/// Cached per-viewer-frame filmstrip classes + reference graph (M-A §C).
/// Keyed on (file, offset, total): `build_filmstrip_refs` scans every
/// frame's slice_headers once per key change, never per repaint.
struct FilmstripCache {
    file_ptr: usize,
    offset: i32,
    total: usize,
    classes: Vec<Option<FrameTypeClass>>,
    refs: FilmstripRefs,
}

/// Opportunity grid derived from the current frame's aligned pair (M3).
struct OppData {
    g: u32,
    cols: u32,
    rows: u32,
    /// `z_b − z_a` per G cell; `None` on invalid cells.
    grid: Vec<Option<f32>>,
    /// max |z| over valid cells — symmetric legend scale.
    zmax: f32,
}

/// Cached current-frame correlation data, keyed on everything that feeds it
/// (including the file identity — see [`FrameDerived`]).
struct CorrDerived {
    file_ptr: usize,
    frame_idx: usize,
    display_idx: usize,
    g: u32,
    x: XMetric,
    y: YMetric,
    /// M-D: (B file identity, B offset) the derivation used — (0, 0) for
    /// non-delta Y metrics. Any B change re-derives a delta pair.
    b_key: (usize, i32),
    /// `None` when the X metric needs a previous frame and there is none.
    pair: Option<AlignedPair>,
    /// Bitstream aggregate at G (drives the class table too).
    bs_g: BitstreamG,
    /// Motion classes at G (class table); `None` without a previous frame.
    classes: Option<(Vec<MotionClass>, Vec<bool>, u32, u32)>,
    r: Option<f64>,
    rho: Option<f64>,
    /// Opportunity map of `pair` (M3); `None` when `pair` is.
    opp: Option<OppData>,
}

/// Cached statistics and valid scatter points of a range-scan result. The
/// concat + Pearson + Spearman (rank sort!) over a long 8px scan is
/// O(N log N) with N in the millions on 1080p content — recomputing it every
/// repaint would stall playback, so it is keyed on the scan Arc identity
/// (the published `CorrScanResult` is immutable).
struct CorrScanDerived {
    /// `Arc::as_ptr` of the snapshot's scan result.
    scan_ptr: usize,
    r: Option<f64>,
    rho: Option<f64>,
    n: usize,
    frac: f32,
    /// Valid (a, b) samples, ready for the scatter plot.
    pts: Vec<[f64; 2]>,
}

impl Default for BitstreamWindow {
    fn default() -> Self {
        Self::new()
    }
}

impl BitstreamWindow {
    pub fn new() -> Self {
        Self {
            open: false,
            shared: Arc::new(Mutex::new(BitstreamShared::new())),
            tab: BsTab::Viewer,
            view: ViewConfig::default(),
            mv_source: MvSource::Mv,
            show_loupe: false,
            inspector_collapsed: false,
            zoom: None,
            pan: egui::Vec2::ZERO,
            selection: None,
            texture: None,
            texture_gen: u64::MAX,
            derived: None,
            graph_show_bits: true,
            graph_show_qp: true,
            corr_x: XMetric::Variance,
            corr_y: YMetric::Bpp,
            corr_g: 16,
            corr_range_mode: false,
            corr_range: (0, 0),
            corr_csv_path: String::new(),
            corr_csv_status: None,
            corr_derived: None,
            corr_scan_derived: None,
            corr_hover_cell: None,
            opp_focus: None,
            pending_center: None,
            filmstrip_cache: None,
            show_refs: true,
            structure_filter: String::new(),
            selected_tu: None,
            stats_cache: None,
            stats_filter: String::new(),
            picture: PictureSource::Source,
            stage_cache: StageCache::new(),
            stage_texture: None,
            diff_mode: false,
            diff: None,
        }
    }

    /// Load persisted toggle state (called once at app start).
    pub fn apply_settings(&mut self, s: &BitstreamViewSettings) {
        self.view = ViewConfig::from_settings(s);
        self.mv_source = MvSource::from_label(&s.mv_source);
        self.show_loupe = s.show_loupe;
        self.inspector_collapsed = s.inspector_collapsed;
    }

    /// Drop every piece of window state derived from (or pointing into) the
    /// loaded `.catb`. The root calls this when the file is unloaded **or
    /// replaced** — a selection/opp-focus/derived cache from file A must not
    /// be interpreted against file B.
    pub fn reset_file_state(&mut self) {
        self.selection = None;
        self.selected_tu = None;
        self.opp_focus = None;
        self.corr_hover_cell = None;
        self.pending_center = None;
        self.derived = None;
        self.corr_derived = None;
        self.corr_scan_derived = None;
        self.filmstrip_cache = None;
        self.stats_cache = None;
        // M-C: the Picture selector is session-only and file-bound.
        self.picture = PictureSource::Source;
        self.stage_texture = None;
        self.stage_cache = StageCache::new();
        // M-D: the Δ pair is anchored on A — an A change invalidates it.
        self.reset_b_state();
    }

    /// M-D: drop every piece of state derived from the comparison stream
    /// (B). Called when B is unloaded/replaced and from
    /// [`Self::reset_file_state`] (0.12.1 cache-reset convention).
    pub fn reset_b_state(&mut self) {
        self.diff_mode = false;
        self.diff = None;
        // A freed B Arc's address can be reused by the next load, so the
        // b_key pointer inside corr_derived cannot distinguish B1 from B2
        // on its own (same rule as bs_overlay_cache in 0.12.1).
        self.corr_derived = None;
    }

    /// Render the window. Returns the current view settings when they
    /// changed this frame (caller persists them to settings.toml).
    pub fn show(&mut self, ctx: &egui::Context) -> Option<BitstreamViewSettings> {
        // Child close → root cleans up on the next pass (sidebar protocol).
        {
            let mut shared = self.shared.lock();
            if shared.close_requested {
                shared.close_requested = false;
                self.open = false;
            }
        }
        if !self.open {
            return None;
        }

        let settings_before =
            self.view
                .to_settings(self.show_loupe, self.inspector_collapsed, self.mv_source);

        // Snapshot shared state; clone the frame image only when the
        // generation moved past the uploaded texture (upload gate).
        let (mut snap, title) = {
            let s = self.shared.lock();
            let title = match (&s.file, &s.file_name) {
                (Some(_), Some(name)) => format!("Bitstream Analysis — {name}"),
                (Some(_), None) => "Bitstream Analysis".to_string(),
                (None, _) => "Bitstream Analysis — (no .catb loaded)".to_string(),
            };
            let snap = Snapshot {
                file: s.file.clone(),
                image: if s.generation != self.texture_gen {
                    s.frame_image.clone()
                } else {
                    None
                },
                frame_pixels: s.frame_image.clone(),
                generation: s.generation,
                viewer_frame: s.viewer_frame,
                viewer_total: s.viewer_total,
                viewer_size: s.viewer_size,
                offset: s.offset,
                is_playing: s.is_playing,
                frame_graph: s.frame_graph.clone(),
                frame_graph_scanning: s.frame_graph_scanning,
                file_b: s.file_b.clone(),
                offset_b: s.offset_b,
                frame_graph_b: s.frame_graph_b.clone(),
                frame_graph_b_scanning: s.frame_graph_b_scanning,
                corr_analysis: s.corr_analysis.clone(),
                corr_scan: s.corr_scan.clone(),
                corr_scanning: s.corr_scanning,
                corr_scan_progress: s.corr_scan_progress,
                scene_changes: s.scene_changes.clone(),
            };
            (snap, title)
        };

        let shared = Arc::clone(&self.shared);
        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("bitstream_viewport"),
            egui::ViewportBuilder::default()
                .with_title(title)
                .with_inner_size([980.0, 700.0])
                .with_min_inner_size([640.0, 420.0]),
            |ctx, _class| {
                if ctx.input(|i| i.viewport().close_requested()) {
                    shared.lock().close_requested = true;
                    egui::CentralPanel::default().show(ctx, |_| {});
                    return;
                }
                // §5: hover tooltip after a 0.2 s delay. Style is
                // Context-global (shared with the root viewport), so stash
                // the original value and restore it at the end of this pass
                // — the main window's tooltips must keep their default delay.
                let root_tooltip_delay = ctx.style().interaction.tooltip_delay;
                ctx.style_mut(|s| s.interaction.tooltip_delay = 0.2);

                // Upload the frame texture when the generation moved —
                // done here (not per-tab) so switching to Correlation /
                // Frame Graph doesn't leave `texture_gen` behind and force
                // a multi-MB ColorImage clone on every repaint.
                if let Some(img) = snap.image.take() {
                    self.texture = Some(ctx.load_texture(
                        "bs_frame",
                        egui::ImageData::Color(img),
                        egui::TextureOptions::NEAREST,
                    ));
                    self.texture_gen = snap.generation;
                }

                self.handle_keys(ctx, &snap);

                self.ui_tab_bar(ctx, &snap);
                match self.tab {
                    BsTab::Viewer => self.ui_viewer_tab(ctx, &snap),
                    BsTab::Correlation => self.ui_correlation_tab(ctx, &snap),
                    BsTab::FrameGraph => self.ui_frame_graph_tab(ctx, &snap),
                    BsTab::Structure => self.ui_structure_tab(ctx, &snap),
                    BsTab::Stats => self.ui_stats_tab(ctx, &snap),
                }

                // Tell the root whether it must keep computing analysis
                // grids for the Correlation tab (update_analysis convention:
                // only pay for visible views).
                {
                    let mut s = shared.lock();
                    // M3: the Opportunity fill on the Viewer tab consumes
                    // the same analysis grids as the Correlation tab.
                    let active = self.tab == BsTab::Correlation
                        || (self.tab == BsTab::Viewer
                            && self.view.fill == FillMode::Opportunity);
                    if active && !s.corr_active {
                        // Freshly activated: the root computes on its next
                        // pass — wake it up.
                        ctx.request_repaint_of(egui::ViewportId::ROOT);
                    }
                    s.corr_active = active;
                    // M3 (02 §2 view 1): mirror the scatter-hover cell to
                    // the root so the main viewer canvas can highlight it
                    // too. Wake the root when it changes — the root only
                    // repaints on its own events otherwise.
                    if s.corr_hover_cell != self.corr_hover_cell {
                        s.corr_hover_cell = self.corr_hover_cell;
                        ctx.request_repaint_of(egui::ViewportId::ROOT);
                    }
                }

                // --- Drag & drop onto this window (0.15.0) ---
                // egui-winit routes DroppedFile/HoveredFile OS events by
                // window, so drops on this viewport arrive in *this*
                // viewport's RawInput. On backends that misroute them to
                // the root input instead (see comparison.rs drop notes),
                // the root's own drop handler classifies and routes the
                // file identically — so no root-input fallback here:
                // reading both would double-handle the same drop.
                let (dropped_path, drag_hover) = ctx.input(|i| {
                    (
                        i.raw.dropped_files.iter().find_map(|f| f.path.clone()),
                        !i.raw.hovered_files.is_empty(),
                    )
                });
                if drag_hover {
                    // Same hint style as the comparison panes
                    // (comparison.rs show_pane): translucent blue fill,
                    // border, centred label.
                    let rect = ctx.screen_rect();
                    let painter = ctx.layer_painter(egui::LayerId::new(
                        egui::Order::Foreground,
                        egui::Id::new("bs_drop_hint"),
                    ));
                    painter.rect_filled(
                        rect,
                        0.0,
                        egui::Color32::from_rgba_unmultiplied(40, 110, 220, 50),
                    );
                    painter.rect_stroke(
                        rect,
                        0.0,
                        egui::Stroke::new(2.0, egui::Color32::from_rgb(60, 140, 240)),
                        egui::StrokeKind::Inside,
                    );
                    painter.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "Drop .catb / bitstream to load",
                        egui::FontId::proportional(16.0),
                        egui::Color32::WHITE,
                    );
                }
                if let Some(path) = dropped_path {
                    let kind = classify_dropped_file(&path);
                    shared.lock().drop_request = Some((path, kind));
                    // The root polls drop_request on its next pass — wake it.
                    ctx.request_repaint_of(egui::ViewportId::ROOT);
                }

                // Keep repainting while the root plays back (see sidebar.rs:
                // a cross-viewport repaint alone is starved while the root
                // animates continuously).
                if snap.is_playing {
                    ctx.request_repaint();
                }

                // Restore the app-global tooltip delay (see above).
                ctx.style_mut(|s| s.interaction.tooltip_delay = root_tooltip_delay);
            },
        );

        let settings_after =
            self.view
                .to_settings(self.show_loupe, self.inspector_collapsed, self.mv_source);
        (settings_after != settings_before).then_some(settings_after)
    }

    // -- keyboard (§8) ------------------------------------------------------

    fn handle_keys(&mut self, ctx: &egui::Context, snap: &Snapshot) {
        // Suppress everything while a text field owns the keyboard
        // (app.rs:1490 dialog-guard convention).
        if ctx.wants_keyboard_input() {
            return;
        }
        struct Keys {
            space: bool,
            left: bool,
            right: bool,
            home: bool,
            end: bool,
            f: bool,
            m: bool,
            c: bool,
            g: bool,
            nums: [bool; 9],
            v: bool,
            p: bool,
            t: bool,
            d: bool,
            l: bool,
            s: bool,
            tab: bool,
            i_key: bool,
            x: bool,
            esc: bool,
        }
        let k = ctx.input_mut(|i| {
            let plain = !i.modifiers.ctrl && !i.modifiers.command;
            Keys {
                space: plain && i.key_pressed(egui::Key::Space),
                left: plain && i.key_pressed(egui::Key::ArrowLeft),
                right: plain && i.key_pressed(egui::Key::ArrowRight),
                home: plain && i.key_pressed(egui::Key::Home),
                end: plain && i.key_pressed(egui::Key::End),
                f: plain && i.key_pressed(egui::Key::F),
                m: plain && i.key_pressed(egui::Key::M),
                c: plain && i.key_pressed(egui::Key::C),
                g: plain && i.key_pressed(egui::Key::G),
                nums: [
                    plain && i.key_pressed(egui::Key::Num1),
                    plain && i.key_pressed(egui::Key::Num2),
                    plain && i.key_pressed(egui::Key::Num3),
                    plain && i.key_pressed(egui::Key::Num4),
                    plain && i.key_pressed(egui::Key::Num5),
                    plain && i.key_pressed(egui::Key::Num6),
                    plain && i.key_pressed(egui::Key::Num7),
                    plain && i.key_pressed(egui::Key::Num8),
                    plain && i.key_pressed(egui::Key::Num9),
                ],
                v: plain && i.key_pressed(egui::Key::V),
                p: plain && i.key_pressed(egui::Key::P),
                t: plain && i.key_pressed(egui::Key::T),
                d: plain && i.key_pressed(egui::Key::D),
                l: plain && i.key_pressed(egui::Key::L),
                s: plain && i.key_pressed(egui::Key::S),
                // Consume Tab so egui's focus traversal doesn't also react.
                tab: i.consume_key(egui::Modifiers::NONE, egui::Key::Tab),
                i_key: plain && i.key_pressed(egui::Key::I),
                x: plain && i.key_pressed(egui::Key::X),
                esc: plain && i.key_pressed(egui::Key::Escape),
            }
        });

        let mut wake_root = false;
        if k.space {
            self.shared.lock().toggle_play_request = true;
            wake_root = true;
        }
        let total = snap.viewer_total;
        let mut seek: Option<usize> = None;
        if k.left && snap.viewer_frame > 0 {
            seek = Some(snap.viewer_frame - 1);
        }
        if k.right && total > 0 && snap.viewer_frame + 1 < total {
            seek = Some(snap.viewer_frame + 1);
        }
        if k.home && total > 0 {
            seek = Some(0);
        }
        if k.end && total > 0 {
            seek = Some(total - 1);
        }
        if let Some(idx) = seek {
            self.shared.lock().seek_request = Some(idx);
            wake_root = true;
        }
        if k.f {
            self.zoom = None;
            self.pan = egui::Vec2::ZERO;
        }
        if k.m {
            self.show_loupe = !self.show_loupe;
        }
        if k.c {
            self.pan = egui::Vec2::ZERO;
        }
        if k.g {
            self.view.grid = !self.view.grid;
        }
        for (i, &pressed) in k.nums.iter().enumerate() {
            if pressed {
                if let Some(&f) = FILL_MODES.get(i) {
                    self.view.fill = f;
                }
            }
        }
        // M4: MV / Part / Intra layer toggles are live.
        if k.v {
            self.view.mv = !self.view.mv;
        }
        if k.p {
            self.view.part = !self.view.part;
        }
        if k.t {
            self.view.tu = !self.view.tu;
        }
        if k.d {
            self.view.intra = !self.view.intra;
        }
        if k.l {
            self.view.label = !self.view.label;
        }
        if k.s {
            self.view.sel = !self.view.sel;
        }
        if k.tab {
            self.view = next_preset(&self.view).config();
        }
        if k.i_key {
            self.inspector_collapsed = !self.inspector_collapsed;
        }
        // M-D: Δ toggle (X) — only when a comparison stream is loaded and
        // the current fill has a Δ rendering.
        if k.x && snap.file_b.is_some() && self.view.fill.supports_diff() {
            self.diff_mode = !self.diff_mode;
        }
        if k.esc {
            self.selection = None;
            self.selected_tu = None;
            self.opp_focus = None;
            self.corr_hover_cell = None;
        }
        if wake_root {
            ctx.request_repaint_of(egui::ViewportId::ROOT);
        }
    }

    // -- tab bar + sync readout (§1) ----------------------------------------

    fn sync_readout(&self, snap: &Snapshot) -> String {
        let summary = snap.file.as_ref().and_then(|f| {
            viewer_to_catb_display(snap.viewer_frame, snap.offset).and_then(|d| f.frame_summary(d))
        });
        match summary {
            Some(s) => format!(
                "viewer#{} ↔ POC {} ({}, decode#{})",
                snap.viewer_frame, s.poc, s.frame_type, s.decode_idx
            ),
            None => format!("viewer#{} ↔ (no catb frame)", snap.viewer_frame),
        }
    }

    fn ui_tab_bar(&mut self, ctx: &egui::Context, snap: &Snapshot) {
        egui::TopBottomPanel::top("bs_tabbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, BsTab::Viewer, "Viewer");
                ui.selectable_value(&mut self.tab, BsTab::Correlation, "Correlation");
                ui.selectable_value(&mut self.tab, BsTab::FrameGraph, "Frame Graph");
                ui.selectable_value(&mut self.tab, BsTab::Structure, "Structure");
                ui.selectable_value(&mut self.tab, BsTab::Stats, "Stats");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.monospace(self.sync_readout(snap));
                });
            });
        });
    }

    // -- toolbar (§2) --------------------------------------------------------

    fn ui_toolbar(&mut self, ctx: &egui::Context, snap: &Snapshot) {
        // M-C Picture combo enable state: a stage is offered only when the
        // current frame records a resolvable image (path probe, no pixel
        // I/O — see `stage_available`).
        let decode_idx = snap.file.as_ref().and_then(|f| {
            viewer_to_catb_display(snap.viewer_frame, snap.offset)
                .and_then(|d| f.decode_idx(d))
        });
        egui::TopBottomPanel::top("bs_toolbar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                // M-C: background picture selector (VQA "Pic").
                ui.label("Pic:");
                egui::ComboBox::from_id_salt("bs_picture")
                    .selected_text(self.picture.label())
                    .show_ui(ui, |ui| {
                        for p in PictureSource::ALL {
                            let available = match p {
                                PictureSource::Source => true,
                                PictureSource::Stage(kind) => {
                                    matches!((snap.file.as_ref(), decode_idx),
                                        (Some(f), Some(di)) if stage_available(f, di, kind))
                                }
                            };
                            let resp = ui.add_enabled(
                                available,
                                egui::SelectableLabel::new(self.picture == p, p.label()),
                            );
                            if available {
                                if resp.clicked() {
                                    self.picture = p;
                                }
                            } else {
                                resp.on_disabled_hover_text(
                                    "Stage image not found next to the .catb — \
                                     re-run decoder-run to regenerate stage BMPs",
                                );
                            }
                        }
                    })
                    .response
                    .on_hover_text(
                        "Which picture the canvas shows under the overlays: the \
                         viewer's source video or a decoder-run stage image \
                         (window-only; the main canvas keeps the source)",
                    );
                ui.separator();

                // Preset combo — shows "Custom" when the config was edited.
                let preset_label = matching_preset(&self.view)
                    .map(|p| p.label())
                    .unwrap_or("Custom");
                ui.label("Preset:");
                egui::ComboBox::from_id_salt("bs_preset")
                    .selected_text(preset_label)
                    .show_ui(ui, |ui| {
                        for p in PRESETS {
                            if ui
                                .selectable_label(matching_preset(&self.view) == Some(p), p.label())
                                .clicked()
                            {
                                self.view = p.config();
                            }
                        }
                    });

                ui.label("Fill:");
                egui::ComboBox::from_id_salt("bs_fill")
                    .selected_text(self.view.fill.label())
                    .show_ui(ui, |ui| {
                        for f in FILL_MODES {
                            ui.selectable_value(&mut self.view.fill, f, f.label());
                        }
                    });

                // M-D: Δ toggle — visible only while a comparison .catb (B)
                // is loaded; enabled only for fills with a Δ rendering.
                if snap.file_b.is_some() {
                    let supported = self.view.fill.supports_diff();
                    if !supported {
                        self.diff_mode = false;
                    }
                    let resp = ui.add_enabled(
                        supported,
                        egui::SelectableLabel::new(self.diff_mode, "Δ"),
                    );
                    let resp = resp.on_hover_text(
                        "Δ mode (X): fill shows the A−B difference on the \
                         8px comparison grid — scalar fills as signed \
                         blue-white-red, Mode as grey (same) / A's colour \
                         (different)",
                    );
                    if supported {
                        if resp.clicked() {
                            self.diff_mode = !self.diff_mode;
                        }
                    } else {
                        resp.on_disabled_hover_text(
                            "Δ is not defined for this fill (Opportunity / \
                             quality fills have no B counterpart) — pick \
                             QP, bpp, Mode, MV-heat or a coeff fill",
                        );
                    }
                }

                ui.separator();
                // M4 layers: MV arrows / Partition outlines / Intra dirs.
                if ui
                    .selectable_label(self.view.mv, "MV")
                    .on_hover_text(
                        "Motion arrows (V): per-PU when REF geometry exists, \
                         block-level otherwise. L0 orange / L1 purple, ref \
                         index > 0 dashed. Rendered at zoom ≥ 1.5x.",
                    )
                    .clicked()
                {
                    self.view.mv = !self.view.mv;
                }
                if self.view.mv {
                    // Vector source combo (design note: explicit combo, not a
                    // hidden 3-state cycle — the active source stays visible).
                    egui::ComboBox::from_id_salt("bs_mv_source")
                        .selected_text(self.mv_source.label())
                        .width(52.0)
                        .show_ui(ui, |ui| {
                            for s in MV_SOURCES {
                                ui.selectable_value(&mut self.mv_source, s, s.label());
                            }
                        })
                        .response
                        .on_hover_text("Which vector the arrows trace: final MV, predictor (MVP), or difference (MVD)");
                }
                if ui
                    .selectable_label(self.view.part, "Part")
                    .on_hover_text(
                        "Partition outlines (P): CU grey on every block, \
                         per-PU boundaries magenta where REF rows carry PU \
                         geometry.",
                    )
                    .clicked()
                {
                    self.view.part = !self.view.part;
                }
                if ui
                    .selectable_label(self.view.tu, "TU")
                    .on_hover_text(
                        "Transform-unit outlines (T): yellow, brighter at \
                         deeper TU depth; cbf=0 TUs dashed; transform_skip \
                         TUs shaded; Coeffs:n labels at zoom ≥ 2x. Rendered \
                         at zoom ≥ 1.5x.",
                    )
                    .clicked()
                {
                    self.view.tu = !self.view.tu;
                }
                if ui
                    .selectable_label(self.view.intra, "Intra")
                    .on_hover_text(
                        "Intra prediction directions (D): angle lines per \
                         (sub)block, P/DC badges at zoom ≥ 2x. Rendered at \
                         zoom ≥ 1.5x.",
                    )
                    .clicked()
                {
                    self.view.intra = !self.view.intra;
                }
                if ui
                    .selectable_label(self.view.label, "Label")
                    .on_hover_text("Per-block values (rendered at zoom ≥ 2x)")
                    .clicked()
                {
                    self.view.label = !self.view.label;
                }
                if ui
                    .selectable_label(self.view.grid, "Grid")
                    .on_hover_text("CTU (HEVC 64px) / 4-MB (AVC) boundaries")
                    .clicked()
                {
                    self.view.grid = !self.view.grid;
                }
                if ui
                    .selectable_label(self.view.sel, "Sel")
                    .on_hover_text("Selected-block highlight")
                    .clicked()
                {
                    self.view.sel = !self.view.sel;
                }

                ui.separator();
                let mut pct = self.view.opacity * 100.0;
                ui.add(
                    egui::Slider::new(&mut pct, 0.0..=100.0)
                        .text("Op")
                        .fixed_decimals(0)
                        .suffix("%"),
                )
                .on_hover_text("Fill opacity only — lines and text stay at full contrast");
                self.view.opacity = pct / 100.0;

                if ui
                    .selectable_label(self.show_loupe, "🔍")
                    .on_hover_text("Value loupe (M): magnified 8px cells with values")
                    .clicked()
                {
                    self.show_loupe = !self.show_loupe;
                }

                // Zoom readout — click = fit (Frame Analysis window
                // convention).
                let zoom_text = match self.zoom {
                    Some(z) => format!("{z:.1}x"),
                    None => "fit".to_string(),
                };
                if ui
                    .add(egui::Button::new(egui::RichText::new(zoom_text).monospace()).frame(false))
                    .on_hover_text("Click to fit (F)")
                    .clicked()
                {
                    self.zoom = None;
                    self.pan = egui::Vec2::ZERO;
                }
            });
        });
        let _ = ctx;
    }

    // -- viewer tab ----------------------------------------------------------

    fn ui_viewer_tab(&mut self, ctx: &egui::Context, snap: &Snapshot) {
        self.ui_toolbar(ctx, snap);
        self.ui_transport_and_status(ctx, snap, true);
        self.ui_filmstrip(ctx, snap);
        self.ui_inspector(ctx, snap);

        // Refresh per-frame derived data (blocks, stats, grids). The frame
        // texture itself is uploaded tab-independently in `show()`.
        self.refresh_derived(snap);
        self.ensure_layer_data(snap);
        // M-C: per-block PSNR/SSIM when a quality fill is selected.
        self.ensure_quality(snap);
        // Opportunity fill consumes the correlation derivation (M3).
        if self.view.fill == FillMode::Opportunity {
            self.refresh_corr_derived(snap);
        }
        // M-D: Δ mode is B-bound — force off when B disappeared, refresh
        // the A/B diff grids otherwise.
        if snap.file_b.is_none() {
            self.diff_mode = false;
            self.diff = None;
        } else if self.diff_mode {
            self.refresh_diff(snap);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            self.ui_canvas(ui, snap);
        });
    }

    fn refresh_derived(&mut self, snap: &Snapshot) {
        let Some(file) = snap.file.as_ref() else {
            self.derived = None;
            return;
        };
        let file_ptr = Arc::as_ptr(file) as usize;
        let Some(display_idx) = viewer_to_catb_display(snap.viewer_frame, snap.offset) else {
            self.derived = None;
            return;
        };
        if display_idx >= file.frame_count() {
            self.derived = None;
            return;
        }
        if self
            .derived
            .as_ref()
            .map(|d| d.display_idx == display_idx && d.file_ptr == file_ptr)
            .unwrap_or(false)
        {
            return; // still current
        }
        let Ok(blocks) = file.blocks_display(display_idx) else {
            self.derived = None;
            return;
        };
        let (w, h) = (file.width.max(1), file.height.max(1));
        let tx_agg = aggregate_block_tx(file, &blocks);
        let stats = frame_fill_stats(&blocks, Some(&tx_agg));
        let grid_l1 = rasterize_blocks_tx(&blocks, Some(&tx_agg), w, h, 8);
        let lod = lod_cell_size(&file.catb.meta.codec);
        let grid_lod = rasterize_blocks_tx(&blocks, Some(&tx_agg), w, h, lod);
        self.derived = Some(FrameDerived {
            file_ptr,
            display_idx,
            blocks,
            tx_agg,
            stats,
            grid_l1,
            grid_lod,
            refs: None,
            intra: None,
            tx: None,
            quality: None,
        });
    }

    /// M-D: refresh the per-frame Δ derivation. Both streams are rasterized
    /// onto the same fixed-cell grids (L1 8 px + A-codec LOD) and diffed
    /// cell-wise — CU partitions differ between the two encodes, so
    /// per-rect comparison is impossible by design. Keyed on both file
    /// identities, the viewer frame and both offsets.
    fn refresh_diff(&mut self, snap: &Snapshot) {
        let (Some(file_a), Some(file_b)) = (snap.file.as_ref(), snap.file_b.as_ref()) else {
            self.diff = None;
            return;
        };
        let key = (
            Arc::as_ptr(file_a) as usize,
            Arc::as_ptr(file_b) as usize,
            snap.viewer_frame,
            snap.offset,
            snap.offset_b,
        );
        if self.diff.as_ref().is_some_and(|d| {
            (d.file_a_ptr, d.file_b_ptr, d.viewer_frame, d.offset, d.offset_b) == key
        }) {
            return;
        }
        let mut out = DiffDerived {
            file_a_ptr: key.0,
            file_b_ptr: key.1,
            viewer_frame: key.2,
            offset: key.3,
            offset_b: key.4,
            blocks_b: None,
            grid_l1: None,
            grid_lod: None,
            reason: None,
        };
        // A side: reuse the Viewer tab's per-frame grids (refresh_derived
        // ran just before). Without A data there is nothing to diff.
        let Some(d) = self.derived.as_ref() else {
            out.reason = Some("no catb-A frame for this viewer frame".to_string());
            self.diff = Some(out);
            return;
        };
        // B side: viewer → B display map via B's own offset.
        let display_b = viewer_to_catb_display(snap.viewer_frame, snap.offset_b)
            .filter(|&db| db < file_b.frame_count());
        let Some(display_b) = display_b else {
            out.reason = Some(format!(
                "no catb-B frame for viewer#{} (B offset {:+})",
                snap.viewer_frame, snap.offset_b
            ));
            self.diff = Some(out);
            return;
        };
        let Ok(blocks_b) = file_b.blocks_display(display_b) else {
            out.reason = Some(format!("catb-B: failed to parse frame {display_b}"));
            self.diff = Some(out);
            return;
        };
        // Same grid geometry as A (resolution equality enforced at load;
        // diff_grids re-checks and degrades to a reason, never misaligns).
        let (sw, sh) = (file_a.width.max(1), file_a.height.max(1));
        let tx_agg_b = aggregate_block_tx(file_b, &blocks_b);
        let b_l1 = rasterize_blocks_tx(&blocks_b, Some(&tx_agg_b), sw, sh, 8);
        let lod = lod_cell_size(&file_a.catb.meta.codec);
        let b_lod = rasterize_blocks_tx(&blocks_b, Some(&tx_agg_b), sw, sh, lod);
        out.grid_l1 = diff_grids(&d.grid_l1, &b_l1);
        out.grid_lod = diff_grids(&d.grid_lod, &b_lod);
        if out.grid_l1.is_none() {
            out.reason = Some("A/B grid geometry mismatch — Δ unavailable".to_string());
        }
        out.blocks_b = Some(blocks_b);
        self.diff = Some(out);
    }

    /// M-C: lazily compute per-block PSNR/SSIM (`final_recon` vs the viewer
    /// frame) when a quality fill is active. Keyed on the viewer image
    /// generation inside the per-frame `derived` cache: a recomputation
    /// happens only when the pixels move, not per repaint.
    fn ensure_quality(&mut self, snap: &Snapshot) {
        if !self.view.fill.is_quality() {
            return;
        }
        let Some(file) = snap.file.as_ref() else { return };
        let Some(d) = self.derived.as_mut() else { return };
        if d.quality.as_ref().is_some_and(|(g, _)| *g == snap.generation) {
            return;
        }
        let result = Self::compute_quality(file, d, snap, &mut self.stage_cache);
        d.quality = Some((snap.generation, result));
    }

    /// The actual quality derivation — every M-C precondition surfaces as a
    /// [`QualityUnavailable`] reason for the legend slot.
    fn compute_quality(
        file: &Arc<BitstreamFile>,
        d: &FrameDerived,
        snap: &Snapshot,
        stage_cache: &mut StageCache,
    ) -> Result<BlockQuality, QualityUnavailable> {
        let img = snap
            .frame_pixels
            .as_ref()
            .ok_or(QualityUnavailable::NoViewerFrame)?;
        let decode_idx = file
            .decode_idx(d.display_idx)
            .ok_or(QualityUnavailable::NoStageImage)?;
        let recon = stage_cache
            .get(file, decode_idx, StageKind::FinalRecon)
            .ok_or(QualityUnavailable::NoStageImage)?;
        let (vw, vh) = (img.size[0] as u32, img.size[1] as u32);
        if (recon.width, recon.height) != (vw, vh) {
            return Err(QualityUnavailable::ResolutionMismatch {
                stream: (recon.width, recon.height),
                viewer: (vw, vh),
            });
        }
        let viewer_luma = color_image_luma(img);
        let recon_luma =
            crate::analysis::stage::luma_bt601(&recon.rgb, (vw as usize) * (vh as usize));
        Ok(compute_block_quality(
            &d.blocks,
            &viewer_luma,
            &recon_luma,
            vw,
            vh,
        ))
    }

    /// Lazily build the M4/M-B layer data (REF rows / intra dirs / TX rows)
    /// the current view needs. Called after `refresh_derived`; a toggle
    /// flipped later fills the missing piece on the next pass without
    /// re-deriving.
    fn ensure_layer_data(&mut self, snap: &Snapshot) {
        let Some(file) = snap.file.as_ref() else { return };
        let Some(d) = self.derived.as_mut() else { return };
        if (self.view.mv || self.view.part) && d.refs.is_none() {
            d.refs = Some(build_refs(file, &d.blocks));
        }
        if self.view.intra && d.intra.is_none() {
            d.intra = Some(build_intra(file, &d.blocks));
        }
        if self.view.tu && d.tx.is_none() {
            d.tx = Some(build_tx(file, &d.blocks));
        }
    }

    #[allow(clippy::too_many_lines)]
    fn ui_canvas(&mut self, ui: &mut egui::Ui, snap: &Snapshot) {
        let avail = ui.available_size();
        let (canvas_rect, response) =
            ui.allocate_exact_size(avail, egui::Sense::click_and_drag());
        let painter = ui.painter_at(canvas_rect);
        painter.rect_filled(canvas_rect, 0.0, ui.visuals().extreme_bg_color);

        let Some((vw, vh)) = snap.viewer_size else {
            painter.text(
                canvas_rect.center(),
                egui::Align2::CENTER_CENTER,
                "Open a video in the main window to see the frame here.",
                egui::FontId::proportional(14.0),
                ui.visuals().text_color(),
            );
            return;
        };
        let (vw_f, vh_f) = (vw.max(1) as f32, vh.max(1) as f32);
        let fit_zoom = (canvas_rect.width() / vw_f)
            .min(canvas_rect.height() / vh_f)
            .max(0.01);
        let mut zoom = self.zoom.unwrap_or(fit_zoom);

        // Cursor-anchored wheel zoom (0.8.2 logic, shared free function).
        if response.hovered() {
            let scroll = ui.input_mut(|i| {
                let d = i.smooth_scroll_delta.y;
                i.smooth_scroll_delta = egui::Vec2::ZERO;
                d
            });
            if scroll != 0.0 {
                if let Some(cursor) = response.hover_pos() {
                    let factor = 1.0 + (scroll * 0.002).clamp(-0.25, 0.25);
                    let new_zoom = (zoom * factor).clamp(fit_zoom * 0.25, 64.0);
                    if (new_zoom - zoom).abs() > f32::EPSILON {
                        // The render origin is canvas.min + centre(z) + pan.
                        // Fold the zoom-dependent centring offset into the
                        // pan so the shared, unit-tested `zoom_anchor_pan`
                        // keeps the image point under the cursor fixed.
                        let centre = |z: f32| {
                            egui::vec2(
                                (canvas_rect.width() - vw_f * z) * 0.5,
                                (canvas_rect.height() - vh_f * z) * 0.5,
                            )
                        };
                        let anchor = cursor - canvas_rect.min;
                        self.pan =
                            zoom_anchor_pan(self.pan + centre(zoom), zoom, new_zoom, anchor)
                                - centre(new_zoom);
                        zoom = new_zoom;
                        self.zoom = Some(new_zoom);
                    }
                }
            }
        }
        if response.dragged() {
            self.pan += response.drag_delta();
            if self.zoom.is_none() {
                self.zoom = Some(zoom); // panning implies leaving fit mode
            }
        }
        if response.double_clicked() {
            self.zoom = None;
            self.pan = egui::Vec2::ZERO;
            zoom = fit_zoom;
        }

        // M3 [Jump]: centre the canvas on a stream-px point (V22). Applied
        // here because only the canvas pass knows the effective zoom.
        if let Some(p) = self.pending_center.take() {
            let (sw0, sh0) = snap
                .file
                .as_ref()
                .map(|f| (f.width.max(1) as f32, f.height.max(1) as f32))
                .unwrap_or((vw_f, vh_f));
            let (zx0, zy0) = (zoom * vw_f / sw0, zoom * vh_f / sh0);
            // origin + p·z = canvas centre ⇔ pan = img/2 − p·z.
            self.pan = egui::vec2(
                vw_f * zoom * 0.5 - p.x * zx0,
                vh_f * zoom * 0.5 - p.y * zy0,
            );
            self.zoom = Some(zoom); // panning leaves fit mode
        }

        let img_w = vw_f * zoom;
        let img_h = vh_f * zoom;
        let origin = canvas_rect.min
            + egui::vec2(
                (canvas_rect.width() - img_w) * 0.5,
                (canvas_rect.height() - img_h) * 0.5,
            )
            + self.pan;
        let image_rect = egui::Rect::from_min_size(origin, egui::vec2(img_w, img_h));

        // M-C: background picture — the selected decoder-run stage image,
        // or the viewer's source frame (default). The stage BMP is stream-
        // resolution; stretching it into `image_rect` applies exactly the
        // stream→viewer scale the overlays use (zx/zy below).
        let mut stage_missing = false;
        let bg_tex: Option<egui::TextureId> = match self.picture {
            PictureSource::Source => self.texture.as_ref().map(|t| t.id()),
            PictureSource::Stage(kind) => {
                let key = snap.file.as_ref().zip(self.derived.as_ref()).and_then(
                    |(f, d)| {
                        f.decode_idx(d.display_idx)
                            .map(|di| (Arc::as_ptr(f) as usize, di))
                    },
                );
                if let (Some(file), Some((fp, di))) = (snap.file.as_ref(), key) {
                    let current = self.stage_texture.as_ref().is_some_and(|t| {
                        t.file_ptr == fp && t.decode_idx == di && t.kind == kind
                    });
                    if !current {
                        self.stage_texture =
                            self.stage_cache.get(file, di, kind).map(|img| {
                                let ci = egui::ColorImage::from_rgb(
                                    [img.width as usize, img.height as usize],
                                    &img.rgb,
                                );
                                StageTexture {
                                    handle: ui.ctx().load_texture(
                                        "bs_stage",
                                        egui::ImageData::Color(Arc::new(ci)),
                                        egui::TextureOptions::NEAREST,
                                    ),
                                    file_ptr: fp,
                                    decode_idx: di,
                                    kind,
                                }
                            });
                    }
                } else {
                    self.stage_texture = None;
                }
                if self.stage_texture.is_none() {
                    // This frame has no loadable stage image: show the
                    // source and say why (the combo disables items per
                    // frame, but playback can move onto a frame without
                    // one while a stage is selected).
                    stage_missing = true;
                }
                self.stage_texture
                    .as_ref()
                    .map(|t| t.handle.id())
                    .or_else(|| self.texture.as_ref().map(|t| t.id()))
            }
        };
        if let Some(id) = bg_tex {
            painter.image(
                id,
                image_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
        if stage_missing {
            painter.text(
                canvas_rect.min + egui::vec2(8.0, 8.0),
                egui::Align2::LEFT_TOP,
                format!(
                    "{} not found next to the .catb — re-run decoder-run",
                    self.picture.label()
                ),
                egui::FontId::proportional(13.0),
                egui::Color32::from_rgb(255, 190, 90),
            );
        }

        // Alt hold = peek: original picture only (§2).
        let peek = ui.input(|i| i.modifiers.alt);

        // Stream → viewer scale (R5: overlay scaled on resolution mismatch).
        let (sw, sh) = snap
            .file
            .as_ref()
            .map(|f| (f.width.max(1) as f32, f.height.max(1) as f32))
            .unwrap_or((vw_f, vh_f));
        let zx = zoom * (vw_f / sw);
        let zy = zoom * (vh_f / sh);
        let block_rect = |x: f32, y: f32, w: f32, h: f32| {
            egui::Rect::from_min_size(
                egui::pos2(origin.x + x * zx, origin.y + y * zy),
                egui::vec2(w * zx, h * zy),
            )
        };

        let derived = self.derived.as_ref();
        // M-D: Δ mode replaces the plain fill with the A−B cell difference.
        let diff_active = self.diff_mode && self.view.fill.supports_diff();
        if let (Some(d), false) = (derived, peek) {
            let opacity = self.view.opacity;
            let lod_mode = use_lod(zoom);

            // M-D Δ fill: cell-wise difference on the comparison grids
            // (L1 8px canonical, LOD aggregate below 1.5x — the Opportunity
            // paint-pass convention). Scalar fills → diverging
            // blue-white-red on the symmetric ±max|Δ| scale; Mode → grey
            // where both streams agree, A's mode colour where they differ
            // (VQA Δ convention).
            if diff_active {
                let grid = self.diff.as_ref().and_then(|dd| {
                    if lod_mode {
                        dd.grid_lod.as_ref()
                    } else {
                        dd.grid_l1.as_ref()
                    }
                });
                if let Some(g) = grid {
                    // One colour scale for both LOD levels: the canonical
                    // L1 max|Δ| (the legend shows the same number).
                    let scale = self
                        .diff
                        .as_ref()
                        .and_then(|dd| dd.grid_l1.as_ref())
                        .unwrap_or(g);
                    let metric = self.view.fill.diff_metric();
                    let dmax = metric.map(|m| scale.max_abs(m)).unwrap_or(0.0);
                    let cell = g.cell as f32;
                    let same_grey = egui::Color32::from_rgba_unmultiplied(
                        128,
                        128,
                        128,
                        ((opacity * 0.45).clamp(0.0, 1.0) * 255.0) as u8,
                    );
                    for r in 0..g.rows {
                        for c in 0..g.cols {
                            let i = (r as usize) * (g.cols as usize) + c as usize;
                            if !g.valid[i] {
                                continue;
                            }
                            let color = match metric {
                                Some(m) => {
                                    opportunity_cell_color(g.values(m)[i], dmax, opacity)
                                }
                                None => {
                                    // Mode agreement.
                                    if g.mode_same(i) {
                                        same_grey
                                    } else {
                                        let mc = mode_color(g.mode_a[i]);
                                        egui::Color32::from_rgba_unmultiplied(
                                            mc.r(),
                                            mc.g(),
                                            mc.b(),
                                            (opacity.clamp(0.0, 1.0) * 255.0) as u8,
                                        )
                                    }
                                }
                            };
                            let rect =
                                block_rect(c as f32 * cell, r as f32 * cell, cell, cell);
                            if rect.intersects(canvas_rect) {
                                painter.rect_filled(rect, 0.0, color);
                            }
                        }
                    }
                }
            }

            // Opportunity fill (M3): G-grid cells from the aligned pair —
            // the cell count is small, so no LOD is needed.
            if self.view.fill == FillMode::Opportunity {
                let opp = self
                    .corr_derived
                    .as_ref()
                    .filter(|cd| cd.display_idx == d.display_idx)
                    .and_then(|cd| cd.opp.as_ref());
                if let Some(o) = opp {
                    let cell = o.g as f32;
                    for r in 0..o.rows {
                        for c in 0..o.cols {
                            let i = (r as usize) * (o.cols as usize) + c as usize;
                            if let Some(z) = o.grid.get(i).copied().flatten() {
                                let rect = block_rect(
                                    c as f32 * cell,
                                    r as f32 * cell,
                                    cell,
                                    cell,
                                );
                                if rect.intersects(canvas_rect) {
                                    painter.rect_filled(
                                        rect,
                                        0.0,
                                        opportunity_cell_color(z, o.zmax, opacity),
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // M-C quality fills: per-block rects from BlockQuality (no LOD
            // variant — the values are block-native, and unlike bpp/QP they
            // do not aggregate meaningfully by area-mean).
            if self.view.fill.is_quality() {
                if let Some((_, Ok(q))) = d.quality.as_ref() {
                    let vals = if self.view.fill == FillMode::BlockPsnr {
                        &q.psnr
                    } else {
                        &q.ssim
                    };
                    for (b, &v) in d.blocks.iter().zip(vals.iter()) {
                        if let Some(color) =
                            quality_fill_color(self.view.fill, v, q, opacity)
                        {
                            let rect =
                                block_rect(b.x as f32, b.y as f32, b.w as f32, b.h as f32);
                            if rect.intersects(canvas_rect) {
                                painter.rect_filled(rect, 0.0, color);
                            }
                        }
                    }
                }
            }

            // Fill layer — LOD aggregate cells below 1.5x, per-CU rects above.
            // Suppressed in Δ mode: the Δ pass above owns the fill surface.
            if !diff_active && !matches!(
                self.view.fill,
                FillMode::None | FillMode::Opportunity | FillMode::BlockPsnr | FillMode::BlockSsim
            ) {
                if lod_mode {
                    let g = &d.grid_lod;
                    for r in 0..g.rows {
                        for c in 0..g.cols {
                            let i = (r * g.cols + c) as usize;
                            if g.coverage[i] <= 0.0 {
                                continue;
                            }
                            if let Some(color) = fill_color(
                                self.view.fill,
                                &FillSample::from_grid(g, i),
                                &d.stats,
                                opacity,
                            ) {
                                let cell = g.cell as f32;
                                let rect = block_rect(
                                    c as f32 * cell,
                                    r as f32 * cell,
                                    cell,
                                    cell,
                                );
                                if rect.intersects(canvas_rect) {
                                    painter.rect_filled(rect, 0.0, color);
                                }
                            }
                        }
                    }
                } else {
                    for (bi, b) in d.blocks.iter().enumerate() {
                        if let Some(color) = fill_color(
                            self.view.fill,
                            &FillSample::from_block(b, d.tx_agg.get(bi)),
                            &d.stats,
                            opacity,
                        ) {
                            let rect =
                                block_rect(b.x as f32, b.y as f32, b.w as f32, b.h as f32);
                            if rect.intersects(canvas_rect) {
                                painter.rect_filled(rect, 0.0, color);
                            }
                        }
                    }
                }
            }

            // Grid layer: CTU / 4-MB boundaries.
            if self.view.grid {
                let step = d.grid_lod.cell as f32;
                let stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(0, 220, 0, 130));
                let mut x = step;
                while x < sw {
                    let sx0 = origin.x + x * zx;
                    painter.line_segment(
                        [
                            egui::pos2(sx0, image_rect.min.y),
                            egui::pos2(sx0, image_rect.max.y),
                        ],
                        stroke,
                    );
                    x += step;
                }
                let mut y = step;
                while y < sh {
                    let sy0 = origin.y + y * zy;
                    painter.line_segment(
                        [
                            egui::pos2(image_rect.min.x, sy0),
                            egui::pos2(image_rect.max.x, sy0),
                        ],
                        stroke,
                    );
                    y += step;
                }
            }

            // M4 layers: Part / MV / Intra — shared renderers with the
            // main-canvas overlay (bitstream_overlay.rs). Inside the
            // Alt-peek gate like every other layer.
            let geom = LayerGeom {
                origin,
                zx,
                zy,
                clip: canvas_rect,
            };
            if self.view.part {
                if let Some(refs) = d.refs.as_ref() {
                    draw_part_layer(&painter, &geom, &d.blocks, refs);
                }
            }
            // M-B TU layer: yellow depth-shaded TU rects (+ Coeffs:n at
            // zoom ≥ 2x), drawn between Part (CU/PU) and MV so the arrows
            // stay on top.
            if self.view.tu {
                if let Some(tx) = d.tx.as_ref() {
                    draw_tu_layer(&painter, &geom, tx);
                }
            }
            if self.view.mv {
                if let Some(refs) = d.refs.as_ref() {
                    draw_mv_layer(&painter, &geom, &d.blocks, refs, self.mv_source);
                }
            }
            if self.view.intra {
                if let Some(intra) = d.intra.as_ref() {
                    draw_intra_layer(&painter, &geom, &d.blocks, intra);
                }
            }

            // Label layer: per-block value text at zoom ≥ 2x (§2).
            if self.view.label && zoom >= 2.0 && !lod_mode {
                for (bi, b) in d.blocks.iter().enumerate() {
                    let rect = block_rect(b.x as f32, b.y as f32, b.w as f32, b.h as f32);
                    if !rect.intersects(canvas_rect) || rect.width() < 18.0 {
                        continue;
                    }
                    // M-C: quality fills label the quality value ('e' for
                    // bit-exact blocks, VQA convention).
                    let quality_txt = self
                        .view
                        .fill
                        .is_quality()
                        .then(|| {
                            d.quality.as_ref().and_then(|(_, r)| r.as_ref().ok()).map(|q| {
                                let v = if self.view.fill == FillMode::BlockPsnr {
                                    q.psnr.get(bi)
                                } else {
                                    q.ssim.get(bi)
                                };
                                quality_value_text(
                                    self.view.fill,
                                    v.copied().unwrap_or(f32::NAN),
                                )
                            })
                        })
                        .flatten();
                    let txt = quality_txt.unwrap_or_else(|| {
                        fill_value_text(
                            self.view.fill,
                            &FillSample::from_block(b, d.tx_agg.get(bi)),
                        )
                    });
                    let font_size = (rect.height() * 0.4).clamp(8.0, 14.0);
                    painter.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        txt,
                        egui::FontId::monospace(font_size),
                        egui::Color32::WHITE,
                    );
                }
            }

            // Sel layer: yellow outline on the selected block.
            if self.view.sel {
                if let Some(sel) = self.selection {
                    if sel.display_idx == d.display_idx {
                        if let Some(b) = d.blocks.get(sel.block_idx) {
                            let rect =
                                block_rect(b.x as f32, b.y as f32, b.w as f32, b.h as f32);
                            painter.rect_stroke(
                                rect,
                                0.0,
                                egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 220, 40)),
                                egui::StrokeKind::Outside,
                            );
                        }
                    }
                }
                // M3: selected opportunity G cell (Top-N [Jump] / cell click).
                if let Some((c, r, g)) = self.opp_focus {
                    let gf = g as f32;
                    let rect = block_rect(c as f32 * gf, r as f32 * gf, gf, gf);
                    painter.rect_stroke(
                        rect,
                        0.0,
                        egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 220, 40)),
                        egui::StrokeKind::Outside,
                    );
                }
            }

            // M3: scatter-hovered cell — a faint marker distinct from the
            // Sel layer (§6: linger without stealing the selection look).
            if let Some((c, r, g)) = self.corr_hover_cell {
                let gf = g as f32;
                let rect = block_rect(c as f32 * gf, r as f32 * gf, gf, gf);
                painter.rect_stroke(
                    rect,
                    0.0,
                    egui::Stroke::new(
                        1.5,
                        egui::Color32::from_rgba_unmultiplied(0, 230, 230, 150),
                    ),
                    egui::StrokeKind::Outside,
                );
            }
        }

        // Click: minimum-area hit test → Inspector; empty area clears (§5).
        if response.clicked() && !response.double_clicked() {
            if let (Some(pos), Some(d)) = (response.interact_pointer_pos(), derived) {
                let sx = (pos.x - origin.x) / zx;
                let sy = (pos.y - origin.y) / zy;
                if sx >= 0.0 && sy >= 0.0 && sx < sw && sy < sh {
                    let new_sel = hit_test_min_area(&d.blocks, sx as u32, sy as u32)
                        .map(|block_idx| Selection {
                            display_idx: d.display_idx,
                            block_idx,
                        });
                    if new_sel != self.selection {
                        // M-B: the TU-table selection belongs to the block.
                        self.selected_tu = None;
                    }
                    self.selection = new_sel;
                    // M3: with the Opportunity fill, a click also selects the
                    // G cell — the Inspector shows its raw (a, b, z).
                    if self.view.fill == FillMode::Opportunity {
                        self.opp_focus = self
                            .corr_derived
                            .as_ref()
                            .filter(|cd| cd.display_idx == d.display_idx)
                            .and_then(|cd| cd.opp.as_ref())
                            .and_then(|o| {
                                let (c, r) = (sx as u32 / o.g, sy as u32 / o.g);
                                let i = (r as usize) * (o.cols as usize) + c as usize;
                                (c < o.cols
                                    && r < o.rows
                                    && o.grid.get(i).copied().flatten().is_some())
                                .then_some((c, r, o.g))
                            });
                    }
                } else {
                    self.selection = None;
                    self.selected_tu = None;
                    self.opp_focus = None;
                }
            }
        }

        // Hover: loupe or one-line tooltip summary (§5). Suppressed entirely
        // during Alt-peek — §2 says the original picture only.
        if !peek {
            if let (Some(pos), Some(d)) = (response.hover_pos(), self.derived.as_ref()) {
                let sx = (pos.x - origin.x) / zx;
                let sy = (pos.y - origin.y) / zy;
                if sx >= 0.0 && sy >= 0.0 && sx < sw && sy < sh {
                    // M-D: Δ readout of the hovered L1 cell (A / B / Δ 병기,
                    // tooltip + loupe alike).
                    let diff_cell = (diff_active)
                        .then(|| {
                            let g = self.diff.as_ref()?.grid_l1.as_ref()?;
                            let (c, r) = (sx as u32 / g.cell, sy as u32 / g.cell);
                            if c >= g.cols || r >= g.rows {
                                return None;
                            }
                            let i = (r as usize) * (g.cols as usize) + c as usize;
                            g.valid[i].then_some((g, i))
                        })
                        .flatten();
                    if self.show_loupe {
                        if diff_active {
                            if let Some(g) =
                                self.diff.as_ref().and_then(|dd| dd.grid_l1.as_ref())
                            {
                                draw_diff_loupe(
                                    ui,
                                    pos,
                                    g,
                                    self.view.fill,
                                    sx as u32,
                                    sy as u32,
                                );
                            }
                        } else {
                            draw_l1_loupe(
                                ui,
                                pos,
                                &d.grid_l1,
                                self.view.fill,
                                sx as u32,
                                sy as u32,
                            );
                        }
                    } else if let Some(bi) = hit_test_min_area(&d.blocks, sx as u32, sy as u32) {
                        let b = &d.blocks[bi];
                        let area = (b.w as f32 * b.h as f32).max(1.0);
                        let bpp = b.bits.max(0) as f32 / area;
                        let mode = if b.prediction_mode.is_empty() {
                            "?"
                        } else {
                            b.prediction_mode.as_str()
                        };
                        let mut text = format!(
                            "QP {} · {}b · {} · {}×{} @({},{}) · bpp {:.2}",
                            b.qp, b.bits, mode, b.w, b.h, b.x, b.y, bpp,
                        );
                        // §5: the tooltip leads with the current fill value —
                        // QP/bpp/mode are already in the summary; MV-heat /
                        // the M-B TX fills need their value prepended.
                        if self.view.fill == FillMode::MvHeat {
                            text = format!("|MV| {:.1}px · {text}", block_mv_px(b));
                        } else if matches!(
                            self.view.fill,
                            FillMode::CoeffEnergy | FillMode::NonzeroCoeffs
                        ) {
                            let s = FillSample::from_block(b, d.tx_agg.get(bi));
                            text = format!(
                                "energy {:.1}/px · nz {:.2}/px · {text}",
                                s.coeff, s.nz
                            );
                        } else if self.view.fill.is_quality() {
                            if let Some((_, Ok(q))) = d.quality.as_ref() {
                                let (p, s) = (
                                    q.psnr.get(bi).copied().unwrap_or(f32::NAN),
                                    q.ssim.get(bi).copied().unwrap_or(f32::NAN),
                                );
                                text = format!(
                                    "PSNR {} dB · SSIM {} · {text}",
                                    quality_value_text(FillMode::BlockPsnr, p),
                                    quality_value_text(FillMode::BlockSsim, s),
                                );
                            }
                        }
                        // M-D: lead with the hovered comparison cell's
                        // A / B / Δ (§ tooltip 병기). The A value comes from
                        // A's own L1 grid at the same cell; B = A − Δ.
                        if let Some((g, i)) = diff_cell {
                            match self.view.fill.diff_metric() {
                                Some(m) => {
                                    let a_s = FillSample::from_grid(&d.grid_l1, i);
                                    let av = match m {
                                        DiffMetric::Qp => a_s.qp,
                                        DiffMetric::Bpp => a_s.bpp,
                                        DiffMetric::MvMag => a_s.mv,
                                        DiffMetric::CoeffEnergy => a_s.coeff,
                                        DiffMetric::NzDensity => a_s.nz,
                                    };
                                    let dv = g.values(m)[i];
                                    text = format!(
                                        "Δ {dv:+.2} (A {av:.2} / B {:.2}) · {text}",
                                        av - dv,
                                    );
                                }
                                None => {
                                    text = format!(
                                        "mode A {} / B {} ({}) · {text}",
                                        g.mode_a[i].label(),
                                        g.mode_b[i].label(),
                                        if g.mode_same(i) { "=" } else { "≠" },
                                    );
                                }
                            }
                        }
                        response.clone().on_hover_text(text);
                    }
                }
            }
        }

        // Legend — always visible, bottom-left (§5): min/max re-normalize per
        // frame, so hiding it would make colours incomparable.
        if !peek {
            if diff_active {
                // M-D: symmetric ±max|Δ| legend (or the reason there is no
                // Δ for this frame — e.g. B carries no mapped frame).
                match self.diff.as_ref() {
                    Some(dd) => {
                        if let Some(reason) = &dd.reason {
                            draw_legend_note(&painter, canvas_rect, reason);
                        } else if let Some(g) = dd.grid_l1.as_ref() {
                            match self.view.fill.diff_metric() {
                                Some(m) => {
                                    let name = format!(
                                        "Δ{} (A−B)",
                                        self.view.fill.label()
                                    );
                                    draw_opportunity_legend(
                                        &painter,
                                        canvas_rect,
                                        &name,
                                        g.max_abs(m),
                                    );
                                }
                                None => {
                                    draw_legend_note(
                                        &painter,
                                        canvas_rect,
                                        "ΔMode: grey = same · colour = A's mode where different",
                                    );
                                }
                            }
                        }
                    }
                    None => {
                        draw_legend_note(&painter, canvas_rect, "Δ: no comparison data");
                    }
                }
            } else if self.view.fill == FillMode::Opportunity {
                // No-data guard: like the other fills (whose legend hides
                // when `derived` is None), skip the bar entirely instead of
                // rendering a meaningless "−0.00 .. +0.00" scale.
                if let Some(zmax) = self
                    .corr_derived
                    .as_ref()
                    .and_then(|cd| cd.opp.as_ref())
                    .map(|o| o.zmax)
                {
                    // Symmetric z scale; name states the direction
                    // ("z(Y)−z(X)") whenever X/Y stray from the default
                    // variance/bpp pair.
                    let name =
                        if self.corr_x == XMetric::Variance && self.corr_y == YMetric::Bpp {
                            "Opportunity".to_string()
                        } else {
                            format!("z({})−z({})", self.corr_y.label(), self.corr_x.label())
                        };
                    draw_opportunity_legend(&painter, canvas_rect, &name, zmax);
                }
            } else if self.view.fill.is_quality() {
                // M-C: finite min/max scale, or the unmet-precondition
                // reason in the legend slot (D: the fill stays selectable).
                if let Some((_, result)) = self.derived.as_ref().and_then(|d| d.quality.as_ref())
                {
                    draw_quality_legend(&painter, canvas_rect, self.view.fill, result);
                }
            } else if let Some(d) = self.derived.as_ref() {
                draw_legend(&painter, canvas_rect, self.view.fill, &d.stats);
            }
        }

        // Live zoom readout for the toolbar (fit resolves to a number here).
        if self.zoom.is_none() {
            // keep as fit — toolbar shows "fit"
        } else {
            self.zoom = Some(zoom);
        }
    }

    // -- inspector (§1 right dock) -------------------------------------------

    fn ui_inspector(&mut self, ctx: &egui::Context, snap: &Snapshot) {
        if self.inspector_collapsed {
            egui::SidePanel::right("bs_inspector_collapsed")
                .resizable(false)
                .exact_width(22.0)
                .show(ctx, |ui| {
                    if ui
                        .button("◀")
                        .on_hover_text("Expand Inspector (I)")
                        .clicked()
                    {
                        self.inspector_collapsed = false;
                    }
                });
            return;
        }
        egui::SidePanel::right("bs_inspector")
            .default_width(230.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Inspector");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .button("▶")
                            .on_hover_text("Collapse Inspector (I) — selection is kept")
                            .clicked()
                        {
                            self.inspector_collapsed = true;
                        }
                    });
                });
                ui.separator();

                // M3: raw values of the selected opportunity G cell
                // ((a, b, z) — the heatmap colour's provenance).
                if let Some((c, r, g)) = self.opp_focus {
                    let cell = self
                        .corr_derived
                        .as_ref()
                        .filter(|cd| cd.g == g)
                        .and_then(|cd| {
                            let p = cd.pair.as_ref()?;
                            let o = cd.opp.as_ref()?;
                            if c >= p.cols || r >= p.rows {
                                return None;
                            }
                            let i = (r as usize) * (p.cols as usize) + c as usize;
                            let z = o.grid.get(i).copied().flatten()?;
                            Some((p.a[i], p.b[i], z, cd.x, cd.y))
                        });
                    if let Some((a, b, z, x, y)) = cell {
                        ui.label(
                            egui::RichText::new(format!(
                                "Opportunity cell ({}, {})",
                                c * g,
                                r * g
                            ))
                            .strong(),
                        );
                        egui::Grid::new("bs_opp_cell_grid")
                            .spacing(egui::vec2(10.0, 3.0))
                            .show(ui, |ui| {
                                ui.label(x.label());
                                ui.monospace(format!("{a:.3}"));
                                ui.end_row();
                                ui.label(y.label());
                                ui.monospace(format!("{b:.3}"));
                                ui.end_row();
                                ui.label("z");
                                ui.monospace(format!("{z:+.3}"));
                                ui.end_row();
                            });
                        ui.separator();
                    }
                }

                let Some(file) = snap.file.as_ref() else {
                    ui.label("No .catb loaded.");
                    return;
                };
                let Some(sel) = self.selection else {
                    ui.label("Click a block on the canvas to inspect it.");
                    ui.weak("Esc or an empty-area click clears the selection.");
                    return;
                };
                let block = self
                    .derived
                    .as_ref()
                    .filter(|d| d.display_idx == sel.display_idx)
                    .and_then(|d| d.blocks.get(sel.block_idx).cloned())
                    .or_else(|| {
                        file.blocks_display(sel.display_idx)
                            .ok()
                            .and_then(|b| b.get(sel.block_idx).cloned())
                    });
                let Some(b) = block else {
                    ui.label("Selection is out of range.");
                    return;
                };

                ui.label(
                    egui::RichText::new(format!("Block ({}, {})", b.x, b.y)).strong(),
                );
                // M-A §B-4: per-block exactness from frames_meta (parallel
                // to the frame's BLOCK records). None when the .catb has no
                // frames_meta.
                let exactness: Option<String> = file
                    .decode_idx(sel.display_idx)
                    .and_then(|dec| file.catb.meta.frames_meta.get(dec))
                    .and_then(|m| m.exactness_missing.get(sel.block_idx).cloned());
                egui::Grid::new("bs_inspector_grid")
                    .spacing(egui::vec2(10.0, 3.0))
                    .show(ui, |ui| {
                        ui.label("size");
                        ui.monospace(format!("{}×{}", b.w, b.h));
                        ui.end_row();
                        ui.label("CTU#");
                        ui.monospace(format!("{}", b.ctu_address));
                        ui.end_row();
                        ui.label("part");
                        ui.monospace(if b.partition.is_empty() { "-" } else { &b.partition });
                        ui.end_row();
                        ui.label("mode");
                        ui.monospace(if b.prediction_mode.is_empty() {
                            "-"
                        } else {
                            &b.prediction_mode
                        });
                        ui.end_row();
                        ui.label("QP");
                        ui.monospace(format!("{}", b.qp));
                        ui.end_row();
                        ui.label("bits");
                        ui.monospace(format_bits(b.bits));
                        ui.end_row();
                        ui.label("bpp");
                        let area = (b.w as f64 * b.h as f64).max(1.0);
                        ui.monospace(format!("{:.2}", b.bits.max(0) as f64 / area));
                        ui.end_row();
                        if let Some(mv) = b.mv {
                            ui.label("MV");
                            ui.monospace(qpel(mv));
                            ui.end_row();
                        }
                        if let Some(mvp) = b.mvp {
                            ui.label("MVP");
                            ui.monospace(qpel(mvp));
                            ui.end_row();
                        }
                        if let Some(mvd) = b.mvd {
                            ui.label("MVD");
                            ui.monospace(qpel(mvd));
                            ui.end_row();
                        }
                        if let Some(ref r) = b.reference {
                            ui.label("ref");
                            ui.monospace(r);
                            ui.end_row();
                        }
                        if let Some(ref rl) = b.reference_label {
                            ui.label("ref label");
                            ui.monospace(rl);
                            ui.end_row();
                        }
                        if let Some(poc) = b.reference_poc {
                            ui.label("ref POC");
                            ui.monospace(format!("{poc}"));
                            ui.end_row();
                        }
                        if let Some(em) = &exactness {
                            ui.label("exact");
                            if em.is_empty() {
                                ui.monospace("✓");
                            } else {
                                ui.colored_label(
                                    egui::Color32::from_rgb(240, 200, 40),
                                    format!("missing: {em}"),
                                );
                            }
                            ui.end_row();
                        }
                    });

                // M-D: in Δ mode, the co-located B block (hit test at A's
                // block centre — B's partition differs, so index-pairing is
                // meaningless). Differing values are highlighted.
                if self.diff_mode {
                    let b_block = self
                        .diff
                        .as_ref()
                        .filter(|dd| dd.viewer_frame == snap.viewer_frame)
                        .and_then(|dd| dd.blocks_b.as_ref())
                        .and_then(|blocks_b| {
                            let cx = b.x + b.w / 2;
                            let cy = b.y + b.h / 2;
                            hit_test_min_area(blocks_b, cx, cy)
                                .and_then(|i| blocks_b.get(i).cloned())
                        });
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("B (same position)").strong());
                    match b_block {
                        Some(bb) => {
                            let hi = egui::Color32::from_rgb(255, 190, 90);
                            let row = |ui: &mut egui::Ui,
                                       name: &str,
                                       val: String,
                                       differs: bool| {
                                ui.label(name);
                                if differs {
                                    ui.colored_label(
                                        hi,
                                        egui::RichText::new(val).monospace(),
                                    );
                                } else {
                                    ui.monospace(val);
                                }
                                ui.end_row();
                            };
                            egui::Grid::new("bs_inspector_b_grid")
                                .spacing(egui::vec2(10.0, 3.0))
                                .show(ui, |ui| {
                                    row(
                                        ui,
                                        "block",
                                        format!(
                                            "{}×{}@({},{})",
                                            bb.w, bb.h, bb.x, bb.y
                                        ),
                                        (bb.x, bb.y, bb.w, bb.h)
                                            != (b.x, b.y, b.w, b.h),
                                    );
                                    row(
                                        ui,
                                        "QP",
                                        format!("{} (Δ{:+})", bb.qp, b.qp - bb.qp),
                                        bb.qp != b.qp,
                                    );
                                    row(
                                        ui,
                                        "bits",
                                        format!(
                                            "{} (Δ{:+})",
                                            format_bits(bb.bits),
                                            b.bits - bb.bits
                                        ),
                                        bb.bits != b.bits,
                                    );
                                    let mode_b = if bb.prediction_mode.is_empty() {
                                        "-"
                                    } else {
                                        &bb.prediction_mode
                                    };
                                    row(
                                        ui,
                                        "mode",
                                        mode_b.to_string(),
                                        bb.prediction_mode != b.prediction_mode,
                                    );
                                    if bb.mv.is_some() || b.mv.is_some() {
                                        row(
                                            ui,
                                            "MV",
                                            bb.mv
                                                .map(qpel)
                                                .unwrap_or_else(|| "-".to_string()),
                                            bb.mv != b.mv,
                                        );
                                    }
                                });
                            ui.weak("highlighted = differs from A");
                        }
                        None => {
                            ui.weak("No B block at this position.");
                        }
                    }
                }

                // REF records: per-PU reference rows (§10 of the format).
                if b.ref_n > 0 {
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("Per-PU references").strong());
                    match file.catb.refs_for_block(&b) {
                        Ok(refs) => {
                            egui::ScrollArea::vertical()
                                .max_height(180.0)
                                .show(ui, |ui| {
                                    egui::Grid::new("bs_ref_grid")
                                        .striped(true)
                                        .spacing(egui::vec2(8.0, 2.0))
                                        .show(ui, |ui| {
                                            ui.label(egui::RichText::new("list").small());
                                            ui.label(egui::RichText::new("PU").small());
                                            ui.label(egui::RichText::new("MV").small());
                                            ui.label(egui::RichText::new("POC").small());
                                            ui.end_row();
                                            for r in &refs {
                                                let list = r
                                                    .list
                                                    .clone()
                                                    .unwrap_or_else(|| "?".to_string());
                                                ui.monospace(format!(
                                                    "{}[{}]",
                                                    list, r.list_index
                                                ));
                                                ui.monospace(format!(
                                                    "{}×{}@({},{})",
                                                    r.pu_w, r.pu_h, r.pu_x, r.pu_y
                                                ));
                                                ui.monospace(
                                                    r.mv.map(qpel)
                                                        .unwrap_or_else(|| "-".to_string()),
                                                );
                                                ui.monospace(format!("{}", r.reference_poc));
                                                ui.end_row();
                                            }
                                        });
                                });
                        }
                        Err(e) => {
                            ui.colored_label(egui::Color32::RED, format!("REF parse: {e}"));
                        }
                    }
                }

                // M-B: TU table (VQA TU tab analogue). Row click opens the
                // coefficient detail below (Transform Detail stage 1 —
                // decoded levels; inverse quant/transform are out of scope).
                if b.tx_n > 0 {
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("Transform units").strong());
                    match file.catb.tx_for_block(&b) {
                        Ok(tx_rows) => {
                            self.ui_inspector_tu_table(ui, file, &tx_rows);
                        }
                        Err(e) => {
                            ui.colored_label(egui::Color32::RED, format!("TX parse: {e}"));
                        }
                    }
                }
            });
    }

    /// Inspector TU table + selected-TU coefficient grid (M-B).
    fn ui_inspector_tu_table(
        &mut self,
        ui: &mut egui::Ui,
        file: &BitstreamFile,
        tx_rows: &[TxRow],
    ) {
        egui::ScrollArea::vertical()
            .id_salt("bs_tu_table")
            .max_height(180.0)
            .show(ui, |ui| {
                egui::Grid::new("bs_tu_grid")
                    .striped(true)
                    .spacing(egui::vec2(8.0, 2.0))
                    .show(ui, |ui| {
                        for head in ["TU", "d", "cbf", "nz", "abs", "bits"] {
                            ui.label(egui::RichText::new(head).small());
                        }
                        ui.end_row();
                        for (k, t) in tx_rows.iter().enumerate() {
                            let selected = self.selected_tu == Some(k);
                            let label = format!("{}×{}@({},{})", t.w, t.h, t.x, t.y);
                            if ui.selectable_label(selected, egui::RichText::new(label).monospace())
                                .on_hover_text(format!(
                                    "type: {} · click for coefficients",
                                    if t.tx_type.is_empty() { "?" } else { &t.tx_type },
                                ))
                                .clicked()
                            {
                                self.selected_tu = (!selected).then_some(k);
                            }
                            ui.monospace(format!("{}", t.depth));
                            let cbf = format!(
                                "{}{}{}",
                                if t.cbf_luma() { "Y" } else { "-" },
                                if t.cbf_cb() { "b" } else { "-" },
                                if t.cbf_cr() { "r" } else { "-" },
                            );
                            ui.monospace(cbf);
                            ui.monospace(
                                t.nonzero_coeff_count
                                    .map(|n| n.to_string())
                                    .unwrap_or_else(|| "-".into()),
                            );
                            ui.monospace(
                                t.coeff_abs_sum
                                    .map(|n| n.to_string())
                                    .unwrap_or_else(|| "-".into()),
                            );
                            ui.monospace(format!("{}", t.bits));
                            ui.end_row();
                        }
                    });
            });

        // Selected TU → coefficient detail (levels grid, scan indices).
        let Some(k) = self.selected_tu else { return };
        let Some(t) = tx_rows.get(k) else {
            self.selected_tu = None;
            return;
        };
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(format!(
                "TU {k}: {}×{}@({},{}) coefficients",
                t.w, t.h, t.x, t.y
            ))
            .strong(),
        );
        let mut meta_line = format!(
            "type {} · last_sig ({}, {})",
            if t.tx_type.is_empty() { "?" } else { &t.tx_type },
            t.last_sig_coeff_x.map(|v| v.to_string()).unwrap_or_else(|| "-".into()),
            t.last_sig_coeff_y.map(|v| v.to_string()).unwrap_or_else(|| "-".into()),
        );
        if t.has_detail() {
            meta_line += &format!(
                " · {} {} · groups {}×{}",
                t.coeff_component, t.coeff_level_kind, t.coeff_group_width, t.coeff_group_height,
            );
        }
        ui.weak(meta_line);
        match file.catb.coeffs_for_tx(t) {
            Ok(Some(detail)) => {
                // Coeff grid width: the group grid is in 4×4 sub-blocks
                // (fixture-observed: cc == (4·gw)·(4·gh)); fall back to the
                // TU width when the shape disagrees.
                let gw4 = t.coeff_group_width.max(0) as usize * 4;
                let gh4 = t.coeff_group_height.max(0) as usize * 4;
                let cc = detail.levels.len();
                let cols = if gw4 > 0 && gw4 * gh4 == cc {
                    gw4
                } else if t.w > 0 && cc % (t.w as usize) == 0 {
                    t.w as usize
                } else {
                    cc.max(1)
                };
                egui::ScrollArea::both()
                    .id_salt("bs_tu_coeffs")
                    .max_height(200.0)
                    .show(ui, |ui| {
                        // Levels in raster order; nonzero cells carry the
                        // scan-order index as "level(scan)".
                        for (ri, row) in detail.levels.chunks(cols).take(64).enumerate() {
                            let base = ri * cols;
                            let line: String = row
                                .iter()
                                .enumerate()
                                .map(|(j, &v)| {
                                    if v != 0 {
                                        let scan = detail
                                            .scan
                                            .get(base + j)
                                            .copied()
                                            .unwrap_or(-1);
                                        format!("{v}({scan})")
                                    } else {
                                        "·".to_string()
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join(" ");
                            ui.monospace(line);
                        }
                        if !detail.group_flags.is_empty() {
                            ui.add_space(2.0);
                            ui.weak(format!(
                                "group flags ({}×{}): {}",
                                t.coeff_group_width,
                                t.coeff_group_height,
                                detail
                                    .group_flags
                                    .iter()
                                    .map(|g| g.to_string())
                                    .collect::<Vec<_>>()
                                    .join(" "),
                            ));
                        }
                    });
            }
            Ok(None) => {
                if t.detail_dropped() {
                    ui.weak(
                        "Coefficient detail dropped by the decoder — \
                         re-run with Deep telemetry (--telemetry-level full).",
                    );
                } else {
                    ui.weak("No coefficient detail for this TU.");
                }
            }
            Err(e) => {
                ui.colored_label(egui::Color32::RED, format!("coeffs: {e}"));
            }
        }
    }

    // -- filmstrip (§4) --------------------------------------------------------

    fn ui_filmstrip(&mut self, ctx: &egui::Context, snap: &Snapshot) {
        // M-A §C: reference arcs need headroom above the cells.
        let refs_on = self.show_refs && snap.file.is_some();
        let arc_band = if refs_on { 16.0_f32 } else { 0.0 };
        egui::TopBottomPanel::bottom("bs_filmstrip")
            .exact_height(30.0 + arc_band)
            .show(ctx, |ui| {
                let total = snap.viewer_total;
                if total == 0 {
                    ui.weak("filmstrip: no video loaded");
                    return;
                }
                let avail_w = ui.available_width();
                let cell_w = (avail_w / total as f32).max(3.0);
                let strip_w = cell_w * total as f32;
                let h = 24.0_f32 + arc_band;
                egui::ScrollArea::horizontal()
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        let (rect, resp) = ui.allocate_exact_size(
                            egui::vec2(strip_w.max(avail_w), h),
                            egui::Sense::click(),
                        );
                        let painter = ui.painter_at(rect);
                        // Frame-type classes are computed once per
                        // (file, offset, total) — the strip repaints every
                        // frame during playback, and a per-cell
                        // `frame_summary()` (String clone) per repaint is
                        // O(total) allocations at playback rate.
                        let file_ptr = snap
                            .file
                            .as_ref()
                            .map(|f| Arc::as_ptr(f) as usize)
                            .unwrap_or(0);
                        let cache_ok = self.filmstrip_cache.as_ref().is_some_and(|c| {
                            c.file_ptr == file_ptr
                                && c.offset == snap.offset
                                && c.total == total
                        });
                        if !cache_ok {
                            let classes = (0..total)
                                .map(|i| {
                                    snap.file.as_ref().and_then(|f| {
                                        viewer_to_catb_display(i, snap.offset)
                                            .and_then(|d| f.frame_summary(d))
                                            .map(|s| FrameTypeClass::from_label(&s.frame_type))
                                    })
                                })
                                .collect();
                            // M-A §C: one full slice_headers scan per key.
                            let refs = snap
                                .file
                                .as_ref()
                                .map(|f| build_filmstrip_refs(f, snap.offset, total))
                                .unwrap_or_default();
                            self.filmstrip_cache = Some(FilmstripCache {
                                file_ptr,
                                offset: snap.offset,
                                total,
                                classes,
                                refs,
                            });
                        }
                        let cache = self.filmstrip_cache.as_ref().expect("just built");
                        let classes = &cache.classes;
                        let cells_top = rect.min.y + arc_band;
                        let cell_h = 24.0 - 4.0;
                        // Only paint the cells inside the visible scroll clip.
                        let clip = painter.clip_rect();
                        let first = (((clip.min.x - rect.min.x) / cell_w).floor().max(0.0)) as usize;
                        let last =
                            ((((clip.max.x - rect.min.x) / cell_w).ceil() as usize) + 1).min(total);
                        for i in first..last {
                            let x0 = rect.min.x + i as f32 * cell_w;
                            let cell = egui::Rect::from_min_size(
                                egui::pos2(x0, cells_top + 2.0),
                                egui::vec2((cell_w - 1.0).max(1.0), cell_h),
                            );
                            let class = classes.get(i).copied().flatten();
                            painter.rect_filled(cell, 1.0, filmstrip_color(class));
                            if i == snap.viewer_frame {
                                painter.rect_stroke(
                                    cell.expand(1.0),
                                    1.0,
                                    egui::Stroke::new(2.0, egui::Color32::WHITE),
                                    egui::StrokeKind::Outside,
                                );
                            }
                            if refs_on {
                                // Reference-frequency dot (red, 3 size tiers).
                                let tier = ref_count_tier(
                                    cache.refs.counts.get(i).copied().unwrap_or(0),
                                );
                                if tier > 0 {
                                    let radius = match tier {
                                        1 => 1.2,
                                        2 => 2.0,
                                        _ => 2.8,
                                    };
                                    painter.circle_filled(
                                        egui::pos2(cell.center().x, cell.max.y - 3.0),
                                        radius,
                                        egui::Color32::from_rgb(235, 55, 55),
                                    );
                                }
                                // Exactness problem: thin yellow top marker.
                                if cache.refs.exactness.get(i).copied().unwrap_or(false) {
                                    painter.line_segment(
                                        [
                                            egui::pos2(cell.min.x, cell.min.y + 1.0),
                                            egui::pos2(cell.max.x, cell.min.y + 1.0),
                                        ],
                                        egui::Stroke::new(
                                            2.0,
                                            egui::Color32::from_rgb(240, 200, 40),
                                        ),
                                    );
                                }
                            }
                        }
                        // M-A §C: reference arcs of the *current* frame only
                        // (VQA Thumbnails view convention) — L0 orange /
                        // L1 purple / long-term green, arcing through the
                        // band above the cells.
                        if refs_on {
                            if let Some(edges) = cache.refs.edges.get(snap.viewer_frame) {
                                let cell_center_x = |i: usize| {
                                    rect.min.x + i as f32 * cell_w + cell_w * 0.5
                                };
                                let y0 = cells_top + 2.0;
                                for e in edges {
                                    let color = if e.long_term {
                                        egui::Color32::from_rgb(60, 200, 80)
                                    } else if e.list == 0 {
                                        egui::Color32::from_rgb(255, 150, 20)
                                    } else {
                                        egui::Color32::from_rgb(170, 90, 230)
                                    };
                                    let from = egui::pos2(
                                        cell_center_x(snap.viewer_frame),
                                        y0,
                                    );
                                    let to = egui::pos2(cell_center_x(e.to), y0);
                                    let ctrl = egui::pos2(
                                        (from.x + to.x) * 0.5,
                                        (rect.min.y + 2.0).max(y0 - arc_band + 2.0),
                                    );
                                    painter.add(
                                        egui::epaint::QuadraticBezierShape::from_points_stroke(
                                            [from, ctrl, to],
                                            false,
                                            egui::Color32::TRANSPARENT,
                                            egui::Stroke::new(1.4, color),
                                        ),
                                    );
                                    // Chevron head at the referenced frame.
                                    painter.line_segment(
                                        [to, to + egui::vec2(-3.0, -4.0)],
                                        egui::Stroke::new(1.4, color),
                                    );
                                    painter.line_segment(
                                        [to, to + egui::vec2(3.0, -4.0)],
                                        egui::Stroke::new(1.4, color),
                                    );
                                }
                            }
                        }
                        // Click → seek via the root (one-way pull model, §4).
                        if let (true, Some(pos)) =
                            (resp.clicked(), resp.interact_pointer_pos())
                        {
                            let idx =
                                (((pos.x - rect.min.x) / cell_w).floor() as usize).min(total - 1);
                            self.shared.lock().seek_request = Some(idx);
                            ctx.request_repaint_of(egui::ViewportId::ROOT);
                        }
                        // Hover tooltip: viewer#N · POC P · type · decode#D · bits.
                        if let Some(pos) = resp.hover_pos() {
                            let idx =
                                (((pos.x - rect.min.x) / cell_w).floor() as usize).min(total - 1);
                            let text = match snap.file.as_ref().and_then(|f| {
                                viewer_to_catb_display(idx, snap.offset)
                                    .and_then(|d| f.frame_summary(d))
                            }) {
                                Some(s) => format!(
                                    "viewer#{idx} · POC {} · {} · decode#{} · {} bits",
                                    s.poc,
                                    s.frame_type,
                                    s.decode_idx,
                                    s.slice_bits.map(format_bits).unwrap_or_else(|| "-".into()),
                                ),
                                None => format!("viewer#{idx} · (no catb frame)"),
                            };
                            resp.on_hover_text(text);
                        }
                    });
            });
    }

    // -- transport + status strip (§1, §7) --------------------------------------

    /// `with_refs_toggle`: show the filmstrip "Refs" toggle — pass `true`
    /// only from tabs that also call [`Self::ui_filmstrip`].
    fn ui_transport_and_status(
        &mut self,
        ctx: &egui::Context,
        snap: &Snapshot,
        with_refs_toggle: bool,
    ) {
        egui::TopBottomPanel::bottom("bs_transport").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let total = snap.viewer_total;
                let cur = snap.viewer_frame;
                let mut seek: Option<usize> = None;
                if ui.button("⏮").on_hover_text("First frame (Home)").clicked() {
                    seek = Some(0);
                }
                if ui.button("◀").on_hover_text("Previous frame (←)").clicked() && cur > 0 {
                    seek = Some(cur - 1);
                }
                let play_label = if snap.is_playing { "⏸" } else { "▶" };
                if ui.button(play_label).on_hover_text("Play/Pause (Space)").clicked() {
                    self.shared.lock().toggle_play_request = true;
                    ctx.request_repaint_of(egui::ViewportId::ROOT);
                }
                if ui.button("▶|").on_hover_text("Next frame (→)").clicked()
                    && total > 0
                    && cur + 1 < total
                {
                    seek = Some(cur + 1);
                }
                if ui.button("⏭").on_hover_text("Last frame (End)").clicked() && total > 0 {
                    seek = Some(total - 1);
                }
                if let Some(idx) = seek {
                    self.shared.lock().seek_request = Some(idx);
                    ctx.request_repaint_of(egui::ViewportId::ROOT);
                }
                ui.monospace(format!(
                    "Frame {}/{}",
                    cur,
                    total.saturating_sub(1)
                ));

                ui.separator();
                let mut offset = snap.offset;
                ui.label("offset:");
                ui.add(egui::DragValue::new(&mut offset).speed(0.05))
                    .on_hover_text("catb display frame = viewer frame + offset");
                if offset != snap.offset {
                    self.shared.lock().offset_request = Some(offset);
                    ctx.request_repaint_of(egui::ViewportId::ROOT);
                }

                // M-A §C: filmstrip reference-arrows toggle (small, next to
                // the strip it controls). Only offered on tabs that actually
                // render the filmstrip — the Correlation tab has none.
                if with_refs_toggle
                    && snap.file.is_some()
                    && ui
                        .selectable_label(self.show_refs, "Refs")
                        .on_hover_text(
                            "Filmstrip reference view: current frame's \
                             ref_list0 (orange) / ref_list1 (purple) arcs \
                             (long-term green), reference-usage dots, and \
                             exactness markers",
                        )
                        .clicked()
                {
                    self.show_refs = !self.show_refs;
                }

                // §7 status strip — fixed wording.
                if snap.offset != 0 {
                    ui.colored_label(
                        egui::Color32::LIGHT_BLUE,
                        format!("offset {:+}", snap.offset),
                    );
                }
                if let Some(f) = snap.file.as_ref() {
                    if snap.viewer_total > 0 && f.frame_count() != snap.viewer_total {
                        ui.colored_label(
                            egui::Color32::YELLOW,
                            format!(
                                "⚠ catb {} frames ≠ viewer {} — check offset",
                                f.frame_count(),
                                snap.viewer_total,
                            ),
                        );
                    }
                    if let Some((vw, vh)) = snap.viewer_size {
                        if f.width > 0 && f.height > 0 && (f.width != vw || f.height != vh) {
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                format!(
                                    "⚠ Stream {}×{} ≠ viewer {}×{} — overlay scaled, values unreliable",
                                    f.width, f.height, vw, vh,
                                ),
                            );
                        }
                    }
                }
            });
        });
    }

    // -- Correlation tab (M2, UX §6) --------------------------------------------

    /// Refresh the cached current-frame correlation derivation.
    fn refresh_corr_derived(&mut self, snap: &Snapshot) {
        // M-D: delta Y metrics are B-bound — auto-revert to bpp when the
        // comparison stream is gone (the combo hides them too).
        if self.corr_y.is_delta() && snap.file_b.is_none() {
            self.corr_y = YMetric::Bpp;
        }
        let Some(file) = snap.file.as_ref() else {
            self.corr_derived = None;
            return;
        };
        let file_ptr = Arc::as_ptr(file) as usize;
        let Some(display_idx) = viewer_to_catb_display(snap.viewer_frame, snap.offset) else {
            self.corr_derived = None;
            return;
        };
        if display_idx >= file.frame_count() {
            self.corr_derived = None;
            return;
        }
        let analysis = snap
            .corr_analysis
            .as_ref()
            .filter(|d| d.frame_idx == snap.viewer_frame);
        let Some(analysis) = analysis else {
            // Root hasn't pushed grids for this frame yet.
            self.corr_derived = None;
            return;
        };
        let b_key = if self.corr_y.is_delta() {
            (
                snap.file_b
                    .as_ref()
                    .map(|f| Arc::as_ptr(f) as usize)
                    .unwrap_or(0),
                snap.offset_b,
            )
        } else {
            (0, 0)
        };
        let current = self.corr_derived.as_ref().is_some_and(|d| {
            d.file_ptr == file_ptr
                && d.frame_idx == snap.viewer_frame
                && d.display_idx == display_idx
                && d.g == self.corr_g
                && d.x == self.corr_x
                && d.y == self.corr_y
                && d.b_key == b_key
        });
        if current {
            return;
        }
        // Bitstream side: reuse the Viewer tab's per-frame L1 grid.
        self.refresh_derived(snap);
        let Some(derived) = self.derived.as_ref() else {
            self.corr_derived = None;
            return;
        };
        let (sw, sh) = (file.width.max(1), file.height.max(1));
        let mut bs_g = aggregate_bitstream_to_g(&derived.grid_l1, sw, sh, self.corr_g);
        // M-C ReconPsnr: image-derived Y values — final_recon vs the viewer
        // frame at the stream resolution. Left empty (⇒ all cells invalid)
        // when the stage image is missing or the resolutions disagree.
        if self.corr_y == YMetric::ReconPsnr {
            let recon = file
                .decode_idx(display_idx)
                .and_then(|di| self.stage_cache.get(file, di, StageKind::FinalRecon));
            if let (Some(img), Some(recon)) = (snap.frame_pixels.as_ref(), recon) {
                let (vw, vh) = (img.size[0] as u32, img.size[1] as u32);
                if (recon.width, recon.height) == (vw, vh) && (vw, vh) == (sw, sh) {
                    let viewer_luma = color_image_luma(img);
                    let recon_luma = crate::analysis::stage::luma_bt601(
                        &recon.rgb,
                        (vw as usize) * (vh as usize),
                    );
                    bs_g.psnr = psnr_to_g(&viewer_luma, &recon_luma, sw, sh, self.corr_g);
                }
            }
        }
        // M-D delta metrics: subtract B's G aggregate (same grid geometry —
        // B's resolution equals A's by the load gate). A viewer frame
        // without a mapped B frame degrades to an all-invalid pair
        // (empty B aggregate), never to a fake A−0 difference.
        if self.corr_y.is_delta() {
            let bs_b = snap
                .file_b
                .as_ref()
                .and_then(|fb| {
                    let db = viewer_to_catb_display(snap.viewer_frame, snap.offset_b)
                        .filter(|&db| db < fb.frame_count())?;
                    let blocks_b = fb.blocks_display(db).ok()?;
                    let b_l1 = rasterize_blocks_tx(&blocks_b, None, sw, sh, 8);
                    Some(aggregate_bitstream_to_g(&b_l1, sw, sh, self.corr_g))
                })
                .unwrap_or_else(|| aggregate_bitstream_to_g(
                    &rasterize_blocks_tx(&[], None, sw, sh, 8),
                    sw,
                    sh,
                    self.corr_g,
                ));
            subtract_bitstream_g(&mut bs_g, &bs_b);
        }
        let pair = x_grid(&analysis.grids, self.corr_x, self.corr_g)
            .map(|xg| align(&xg, &bs_g, self.corr_y));
        let (r, rho) = pair
            .as_ref()
            .map(|p| {
                (
                    pearson_r(&p.a, &p.b, &p.valid),
                    spearman_rho(&p.a, &p.b, &p.valid),
                )
            })
            .unwrap_or((None, None));
        let classes = classes_at_g(&analysis.grids, self.corr_g)
            .map(|(classes, grid)| (classes, grid.valid, grid.cols, grid.rows));
        // M3: opportunity map of the same pair (cheap — O(cells)).
        let opp = pair.as_ref().map(|p| {
            let grid = opportunity_grid(p);
            let zmax = grid
                .iter()
                .flatten()
                .fold(0.0_f32, |m, z| m.max(z.abs()));
            OppData {
                g: self.corr_g,
                cols: p.cols,
                rows: p.rows,
                grid,
                zmax,
            }
        });
        self.corr_derived = Some(CorrDerived {
            file_ptr,
            frame_idx: snap.viewer_frame,
            display_idx,
            g: self.corr_g,
            x: self.corr_x,
            y: self.corr_y,
            b_key,
            pair,
            bs_g,
            classes,
            r,
            rho,
            opp,
        });
    }

    /// True when the stored scan was produced with different X/Y/G settings
    /// — or a different viewer↔catb frame offset — than the window currently
    /// shows. Its numbers must not be presented next to the current labels:
    /// an offset change re-maps every (viewer, catb) pair the scan was
    /// built on.
    fn corr_scan_stale(&self, scan: &CorrScanResult, snap: &Snapshot) -> bool {
        scan.request.g != self.corr_g
            || scan.request.x != self.corr_x
            || scan.request.y != self.corr_y
            || scan.request.offset != snap.offset
            // M-D: delta scans are additionally bound to B's offset — a
            // re-mapped B side invalidates every scanned pair too.
            || (scan.request.y.is_delta() && scan.request.offset_b != snap.offset_b)
    }

    /// Refresh the cached range-scan derivation (statistics + scatter
    /// points). Keyed on the scan Arc identity, so the O(N log N) work runs
    /// once per published scan instead of on every repaint.
    fn refresh_corr_scan_derived(&mut self, snap: &Snapshot) {
        let Some(scan) = snap.corr_scan.as_ref() else {
            self.corr_scan_derived = None;
            return;
        };
        let scan_ptr = Arc::as_ptr(scan) as usize;
        if self
            .corr_scan_derived
            .as_ref()
            .is_some_and(|d| d.scan_ptr == scan_ptr)
        {
            return;
        }
        let (a, b, valid) = scan.concat();
        let n = valid.iter().filter(|v| **v).count();
        let frac = if valid.is_empty() {
            0.0
        } else {
            n as f32 / valid.len() as f32
        };
        let mut pts = Vec::with_capacity(n);
        for i in 0..valid.len() {
            if valid[i] {
                pts.push([a[i] as f64, b[i] as f64]);
            }
        }
        self.corr_scan_derived = Some(CorrScanDerived {
            scan_ptr,
            r: pearson_r(&a, &b, &valid),
            rho: spearman_rho(&a, &b, &valid),
            n,
            frac,
            pts,
        });
    }

    /// Statistics readout source: scan result in range mode, else the
    /// current-frame pair.
    fn corr_stats_readout(&self, snap: &Snapshot) -> String {
        let (r, rho, n, frac) = if self.corr_range_mode {
            match snap.corr_scan.as_ref() {
                // §6: never pair the current combo labels with a previous
                // scan's numbers — flag staleness right in the readout.
                Some(scan) if self.corr_scan_stale(scan, snap) => {
                    return "r=– ρ=– N=– (stale scan — press Scan)".to_string();
                }
                Some(_) => match self.corr_scan_derived.as_ref() {
                    Some(d) => (d.r, d.rho, d.n, d.frac),
                    None => (None, None, 0, 0.0),
                },
                None => (None, None, 0, 0.0),
            }
        } else {
            match self.corr_derived.as_ref().and_then(|d| d.pair.as_ref()) {
                Some(p) => (
                    self.corr_derived.as_ref().and_then(|d| d.r),
                    self.corr_derived.as_ref().and_then(|d| d.rho),
                    p.n_valid(),
                    p.valid_fraction(),
                ),
                None => (None, None, 0, 0.0),
            }
        };
        let fmt = |v: Option<f64>| match v {
            Some(v) => format!("{v:+.3}"),
            None => "–".to_string(),
        };
        format!(
            "r={} ρ={} N={} (valid {:.0}%)",
            fmt(r),
            fmt(rho),
            n,
            frac * 100.0
        )
    }

    #[allow(clippy::too_many_lines)]
    fn ui_correlation_tab(&mut self, ctx: &egui::Context, snap: &Snapshot) {
        self.ui_transport_and_status(ctx, snap, false);
        // M3 timeline strip — added after the transport so it stacks above it.
        self.ui_corr_timeline(ctx, snap);
        self.refresh_corr_derived(snap);
        self.refresh_corr_scan_derived(snap);

        let total = snap.viewer_total;
        // Clamp the range to the shorter of the video / telemetry.
        let max_frame = total.saturating_sub(1);

        // -- controls strip (§6: everything in one line + live readout) --
        egui::TopBottomPanel::top("bs_corr_controls").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("X:");
                egui::ComboBox::from_id_salt("bs_corr_x")
                    .selected_text(self.corr_x.label())
                    .show_ui(ui, |ui| {
                        // Preset pairs above the separator (02 §2). M-D
                        // delta pairs only exist while B is loaded.
                        for (label, x, y) in PRESET_PAIRS {
                            if y.is_delta() && snap.file_b.is_none() {
                                continue;
                            }
                            if ui.selectable_label(false, label).clicked() {
                                self.corr_x = x;
                                self.corr_y = y;
                            }
                        }
                        ui.separator();
                        for x in XMetric::ALL {
                            ui.selectable_value(&mut self.corr_x, x, x.label());
                        }
                    });
                ui.label("Y:");
                egui::ComboBox::from_id_salt("bs_corr_y")
                    .selected_text(self.corr_y.label())
                    .show_ui(ui, |ui| {
                        for y in YMetric::ALL {
                            // M-D: Δbpp/ΔQP need the comparison stream.
                            if y.is_delta() && snap.file_b.is_none() {
                                continue;
                            }
                            ui.selectable_value(&mut self.corr_y, y, y.label());
                        }
                    });
                ui.label("G:");
                egui::ComboBox::from_id_salt("bs_corr_g")
                    .selected_text(format!("{}", self.corr_g))
                    .show_ui(ui, |ui| {
                        for g in G_SIZES {
                            ui.selectable_value(&mut self.corr_g, g, format!("{g}"));
                        }
                    });

                ui.separator();
                ui.label("Frames:");
                ui.radio_value(&mut self.corr_range_mode, false, "current");
                ui.radio_value(&mut self.corr_range_mode, true, "range");
                if self.corr_range_mode {
                    let (mut s, mut e) = self.corr_range;
                    ui.add(
                        egui::DragValue::new(&mut s)
                            .range(0..=max_frame)
                            .prefix("from "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut e)
                            .range(0..=max_frame)
                            .prefix("to "),
                    );
                    if e < s {
                        e = s;
                    }
                    self.corr_range = (s, e);
                    let can_scan =
                        !snap.is_playing && !snap.corr_scanning && snap.file.is_some() && total > 0;
                    let scan = ui
                        .add_enabled(can_scan, egui::Button::new("Scan"))
                        .on_hover_text("Accumulate the aligned pairs over the frame range on a background thread")
                        .on_disabled_hover_text(if snap.is_playing {
                            "Pause playback to scan a range"
                        } else {
                            "Scan unavailable"
                        });
                    if scan.clicked() {
                        self.shared.lock().corr_scan_request = Some(CorrScanRequest {
                            start: s,
                            end: e,
                            g: self.corr_g,
                            x: self.corr_x,
                            y: self.corr_y,
                            offset: snap.offset,
                            offset_b: snap.offset_b,
                        });
                        ctx.request_repaint_of(egui::ViewportId::ROOT);
                    }
                    if snap.corr_scanning {
                        ui.spinner();
                        let (done, of) = snap.corr_scan_progress;
                        ui.weak(format!("{done}/{of}"));
                    }
                }

                ui.separator();
                // §6: the statistics readout is always visible — a scatter
                // plot without its r/ρ/N invites misreading.
                ui.monospace(self.corr_stats_readout(snap));

                ui.separator();
                if self.corr_csv_path.is_empty() {
                    self.corr_csv_path = "correlation.csv".to_string();
                }
                ui.add(
                    egui::TextEdit::singleline(&mut self.corr_csv_path)
                        .desired_width(180.0)
                        .hint_text("CSV path"),
                );
                if ui
                    .button("CSV")
                    .on_hover_text("Dump the aligned pair grid (frame, cell_x, cell_y, a, b, valid)")
                    .clicked()
                {
                    self.corr_csv_status = Some(self.save_corr_csv(snap));
                }
                if let Some(status) = &self.corr_csv_status {
                    ui.weak(status);
                }
            });
        });

        // -- right dock: conditional class table (02 §2 view 3) + M3 Top-N --
        egui::SidePanel::right("bs_corr_classes")
            .default_width(300.0)
            .show(ctx, |ui| {
                ui.heading("Motion class × bitstream");
                ui.separator();
                // Owned Top-N snapshot first (small), so the [Jump] handler
                // below can mutate `self` without fighting the `d` borrow.
                // `(g, [(col, row, z)])` rows of the ranking table.
                type TopnRows = (u32, Vec<(u32, u32, f32)>);
                let topn: Option<TopnRows> =
                    self.corr_derived.as_ref().and_then(|d| {
                        let p = d.pair.as_ref()?;
                        let o = d.opp.as_ref()?;
                        let rows = top_n_ranking(&o.grid, 20)
                            .into_iter()
                            .map(|(i, z)| {
                                (i as u32 % p.cols, i as u32 / p.cols, z)
                            })
                            .collect();
                        Some((o.g, rows))
                    });
                if let Some(d) = self.corr_derived.as_ref() {
                    match &d.classes {
                        Some((classes, valid, cols, rows)) => {
                            let table = class_table(classes, valid, *cols, *rows, &d.bs_g);
                            egui::Grid::new("bs_corr_class_grid")
                                .striped(true)
                                .spacing(egui::vec2(10.0, 4.0))
                                .show(ui, |ui| {
                                    for head in
                                        ["class", "cells", "QP", "bpp", "|MV|px", "intra%"]
                                    {
                                        ui.label(egui::RichText::new(head).small().strong());
                                    }
                                    ui.end_row();
                                    for row in table {
                                        ui.label(row.class.label());
                                        ui.monospace(format!("{}", row.cells));
                                        if row.cells > 0 {
                                            ui.monospace(format!("{:.1}", row.mean_qp));
                                            ui.monospace(format!("{:.2}", row.mean_bpp));
                                            ui.monospace(format!("{:.1}", row.mean_mv));
                                            ui.monospace(format!(
                                                "{:.0}%",
                                                row.intra_ratio * 100.0
                                            ));
                                        } else {
                                            for _ in 0..4 {
                                                ui.monospace("–");
                                            }
                                        }
                                        ui.end_row();
                                    }
                                });
                            ui.add_space(4.0);
                            ui.weak("current frame · valid cells only");
                        }
                        None => {
                            ui.label("Motion classes need a previous frame —");
                            ui.label("step forward once (→) to populate.");
                        }
                    }
                } else {
                    ui.label(if snap.file.is_none() {
                        "No .catb loaded."
                    } else {
                        "No data for this frame."
                    });
                }

                // -- Opportunity Top-N (M3, 02 §2-4 / UX §6): z descending,
                // [Jump] pans the Viewer canvas to the cell (V22).
                ui.add_space(8.0);
                ui.separator();
                ui.heading("Opportunity Top-N");
                match topn {
                    Some((g, rows)) if !rows.is_empty() => {
                        let mut jump: Option<(u32, u32)> = None;
                        egui::ScrollArea::vertical()
                            .id_salt("bs_opp_topn")
                            .max_height(260.0)
                            .show(ui, |ui| {
                                egui::Grid::new("bs_opp_topn_grid")
                                    .striped(true)
                                    .spacing(egui::vec2(8.0, 2.0))
                                    .show(ui, |ui| {
                                        for head in ["#", "cell (px)", "z", ""] {
                                            ui.label(
                                                egui::RichText::new(head).small().strong(),
                                            );
                                        }
                                        ui.end_row();
                                        for (k, (c, r, z)) in rows.iter().enumerate() {
                                            ui.monospace(format!("{}", k + 1));
                                            ui.monospace(format!("({}, {})", c * g, r * g));
                                            ui.monospace(format!("{z:+.2}"));
                                            if ui
                                                .small_button("Jump")
                                                .on_hover_text(
                                                    "Viewer tab · pan to cell · Sel highlight",
                                                )
                                                .clicked()
                                            {
                                                jump = Some((*c, *r));
                                            }
                                            ui.end_row();
                                        }
                                    });
                            });
                        ui.weak("current frame · z = z(Y) − z(X), positive = bits above what X explains");
                        if let Some((c, r)) = jump {
                            self.tab = BsTab::Viewer;
                            self.view.sel = true; // V22: highlight must be visible
                            self.opp_focus = Some((c, r, g));
                            let gf = g as f32;
                            self.pending_center = Some(egui::pos2(
                                c as f32 * gf + gf * 0.5,
                                r as f32 * gf + gf * 0.5,
                            ));
                        }
                    }
                    _ => {
                        ui.weak(
                            "Needs current-frame aligned data — Opportunity ranks the \
                             current frame's z(Y)−z(X) mismatch.",
                        );
                    }
                }
            });

        // -- centre: scatter plot (02 §2 view 1) --
        egui::CentralPanel::default().show(ctx, |ui| {
            use egui_plot::{Plot, PlotPoints, Points};

            if snap.file.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label("Load a .catb to correlate analysis metrics with encoder decisions.");
                });
                return;
            }

            // Collect the plotted points. Range mode reuses the cached scan
            // derivation (borrowed, not rebuilt — the scan can hold millions
            // of samples).
            let mut frame_pts: Vec<[f64; 2]> = Vec::new();
            // Parallel G-cell coordinates for the M3 scatter↔canvas link
            // (current-frame mode only — scan samples have no cell identity
            // across frames worth linking).
            let mut frame_cells: Vec<(u32, u32)> = Vec::new();
            let mut note: Option<String> = None;
            if self.corr_range_mode {
                match snap.corr_scan.as_ref() {
                    Some(scan) => {
                        if self.corr_scan_stale(scan, snap) {
                            note = Some(
                                "Scan result is for different settings — press Scan again."
                                    .to_string(),
                            );
                        }
                        if let Some(err) = &scan.error {
                            note = Some(format!("Scan: {err}"));
                        }
                    }
                    None if snap.corr_scanning => {
                        note = Some("Scanning…".to_string());
                    }
                    None => {
                        note = Some("Set a range and press Scan.".to_string());
                    }
                }
            } else {
                match self.corr_derived.as_ref() {
                    Some(d) => match &d.pair {
                        Some(p) => {
                            for i in 0..p.valid.len() {
                                if p.valid[i] {
                                    frame_pts.push([p.a[i] as f64, p.b[i] as f64]);
                                    frame_cells.push((
                                        i as u32 % p.cols,
                                        i as u32 / p.cols,
                                    ));
                                }
                            }
                        }
                        None => {
                            note = Some(format!(
                                "{} requires a previous frame — step forward once (→).",
                                self.corr_x.label()
                            ));
                        }
                    },
                    None => {
                        note = Some("No aligned data for this frame.".to_string());
                    }
                }
            }
            let pts: &[[f64; 2]] = if self.corr_range_mode {
                self.corr_scan_derived
                    .as_ref()
                    .map_or(&[][..], |d| d.pts.as_slice())
            } else {
                &frame_pts
            };

            // Cap the drawn points so a long-range 8px scan cannot stall the
            // painter; statistics above always use the full data.
            const MAX_POINTS: usize = 20_000;
            let stride = pts.len().div_ceil(MAX_POINTS).max(1);
            let drawn: Vec<[f64; 2]> = pts.iter().copied().step_by(stride).collect();

            if let Some(note) = &note {
                ui.weak(note);
            }

            // M3 reverse link: the Viewer-selected opportunity cell shows as
            // an emphasized scatter point (current-frame mode, matching G).
            let sel_pt: Option<[f64; 2]> = (|| {
                if self.corr_range_mode {
                    return None;
                }
                let (c, r, g) = self.opp_focus?;
                if g != self.corr_g {
                    return None;
                }
                let p = self.corr_derived.as_ref()?.pair.as_ref()?;
                if c >= p.cols || r >= p.rows {
                    return None;
                }
                let i = (r as usize) * (p.cols as usize) + c as usize;
                p.valid[i].then(|| [p.a[i] as f64, p.b[i] as f64])
            })();

            let plot_resp = Plot::new("bs_corr_scatter")
                .x_axis_label(self.corr_x.label())
                .y_axis_label(self.corr_y.label())
                .allow_drag(true)
                .allow_zoom(true)
                .allow_scroll(true)
                .show(ui, |plot_ui| {
                    plot_ui.points(
                        Points::new(PlotPoints::new(drawn))
                            .radius(1.6)
                            .color(egui::Color32::from_rgb(120, 190, 255)),
                    );
                    if let Some(pt) = sel_pt {
                        plot_ui.points(
                            Points::new(PlotPoints::new(vec![pt]))
                                .radius(5.0)
                                .color(egui::Color32::from_rgb(255, 220, 40)),
                        );
                    }
                    (plot_ui.pointer_coordinate(), plot_ui.plot_bounds())
                });

            // M3 forward link: hovering near a scatter point marks its G
            // cell on the Viewer canvas (kept until replaced, §6) and shows
            // a coordinate tooltip.
            if !self.corr_range_mode && !frame_pts.is_empty() {
                let (pointer, bounds) = plot_resp.inner;
                if let Some(ptr) = pointer {
                    let bw = bounds.width().max(f64::EPSILON);
                    let bh = bounds.height().max(f64::EPSILON);
                    let mut best: Option<(usize, f64)> = None;
                    for (i, q) in frame_pts.iter().enumerate() {
                        let dx = (q[0] - ptr.x) / bw;
                        let dy = (q[1] - ptr.y) / bh;
                        let dd = dx * dx + dy * dy;
                        if best.map(|(_, bd)| dd < bd).unwrap_or(true) {
                            best = Some((i, dd));
                        }
                    }
                    if let Some((i, dd)) = best {
                        if dd.sqrt() < 0.02 {
                            let (c, r) = frame_cells[i];
                            self.corr_hover_cell = Some((c, r, self.corr_g));
                            let g = self.corr_g;
                            plot_resp.response.on_hover_text(format!(
                                "cell ({}, {}) px · {} {:.3} · {} {:.3}",
                                c * g,
                                r * g,
                                self.corr_x.label(),
                                frame_pts[i][0],
                                self.corr_y.label(),
                                frame_pts[i][1],
                            ));
                        }
                    }
                }
            }
        });
    }

    /// M3 (02 §2-5): per-frame r timeline under the Correlation tab — the
    /// scanned range's r per frame, the current-frame marker, and scene
    /// change markers (existing detection results only, never recomputed
    /// here). Click = seek via the root (filmstrip seek path).
    fn ui_corr_timeline(&mut self, ctx: &egui::Context, snap: &Snapshot) {
        egui::TopBottomPanel::bottom("bs_corr_timeline")
            .exact_height(130.0)
            .show(ctx, |ui| {
                use egui_plot::{Line, LineStyle, Plot, PlotPoints, VLine};

                let scan = snap
                    .corr_scan
                    .as_ref()
                    .filter(|s| !self.corr_scan_stale(s, snap));
                let Some(scan) = scan else {
                    ui.weak("Per-frame r timeline — run a range Scan to populate.");
                    return;
                };
                let pts: Vec<[f64; 2]> = scan
                    .per_frame
                    .iter()
                    .filter_map(|s| s.r.map(|r| [s.frame as f64, r]))
                    .collect();
                if pts.is_empty() {
                    ui.weak("No per-frame r in the scan (each frame needs ≥ 2 valid cells).");
                    return;
                }
                let total = snap.viewer_total;
                let resp = Plot::new("bs_corr_timeline_plot")
                    .x_axis_label("frame")
                    .y_axis_label("r")
                    .include_y(-1.0)
                    .include_y(1.0)
                    .allow_drag(false)
                    .allow_zoom(false)
                    .allow_scroll(false)
                    .show(ui, |plot_ui| {
                        plot_ui.line(
                            Line::new(PlotPoints::new(pts))
                                .color(egui::Color32::from_rgb(120, 190, 255))
                                .name("r"),
                        );
                        // Current frame (white), scene changes (grey dashed).
                        plot_ui.vline(
                            VLine::new(snap.viewer_frame as f64)
                                .color(egui::Color32::WHITE)
                                .width(1.0),
                        );
                        for &sc in &snap.scene_changes {
                            plot_ui.vline(
                                VLine::new(sc as f64)
                                    .color(egui::Color32::from_rgb(150, 150, 150))
                                    .style(LineStyle::Dashed { length: 6.0 }),
                            );
                        }
                        plot_ui.pointer_coordinate()
                    });
                if resp.response.clicked() && total > 0 {
                    if let Some(p) = resp.inner {
                        let idx = p.x.round().clamp(0.0, (total - 1) as f64) as usize;
                        self.shared.lock().seek_request = Some(idx);
                        ctx.request_repaint_of(egui::ViewportId::ROOT);
                    }
                }
            });
    }

    /// Write the CSV dump for the current mode. Returns a status line.
    fn save_corr_csv(&self, snap: &Snapshot) -> String {
        let csv = if self.corr_range_mode {
            match snap.corr_scan.as_ref() {
                Some(scan) => {
                    if self.corr_scan_stale(scan, snap) {
                        return "scan is stale (settings changed) — press Scan again".to_string();
                    }
                    let frames: Vec<(usize, &AlignedPair)> =
                        scan.frames.iter().map(|(f, p)| (*f, p)).collect();
                    csv_dump(&frames)
                }
                None => return "no scan result to export".to_string(),
            }
        } else {
            match self
                .corr_derived
                .as_ref()
                .and_then(|d| d.pair.as_ref().map(|p| (d.frame_idx, p)))
            {
                Some((frame, pair)) => csv_dump(&[(frame, pair)]),
                None => return "no aligned data to export".to_string(),
            }
        };
        match std::fs::write(&self.corr_csv_path, csv) {
            Ok(()) => format!("saved {}", self.corr_csv_path),
            Err(e) => format!("save failed: {e}"),
        }
    }

    // -- Structure tab (M-A) -----------------------------------------------
    //
    // VQAnalyzer parity: Syntax panel (VPS/SPS/PPS/Slice tabs) + Unit Info
    // "DPB info" tab. Layout: one scrollable column of CollapsingHeader
    // sections instead of a left-tree/right-detail split — the per-frame
    // data volume is small (a handful of parameter sets, 1–few slices,
    // ≤16 DPB rows), the CollapsingHeader idiom matches the app's sidebar
    // conventions, and a split would add selection state for no benefit.
    // Frame navigation reuses the transport + filmstrip (auto-refresh on
    // frame moves falls out of reading `snap` each pass).

    #[allow(clippy::too_many_lines)]
    fn ui_structure_tab(&mut self, ctx: &egui::Context, snap: &Snapshot) {
        self.ui_transport_and_status(ctx, snap, true);
        self.ui_filmstrip(ctx, snap);
        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(file) = snap.file.as_ref() else {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        "Load a .catb to browse parameter sets, slice headers \
                         and the DPB.",
                    );
                });
                return;
            };
            let meta = &file.catb.meta;
            // Current viewer frame → decode-order catb frame.
            let decode = viewer_to_catb_display(snap.viewer_frame, snap.offset)
                .and_then(|d| file.decode_idx(d));
            let fmeta = decode.and_then(|d| meta.frames_meta.get(d));

            egui::ScrollArea::vertical().show(ui, |ui| {
                // §D: capture level / decoder contract badge.
                let level = if meta.capture_level.is_empty() {
                    "n/a"
                } else {
                    meta.capture_level.as_str()
                };
                let contract = if meta.contract.is_empty() {
                    "n/a"
                } else {
                    meta.contract.as_str()
                };
                ui.weak(format!(
                    "capture level: {level} · decoder contract: {contract}"
                ));
                ui.separator();

                // -- 1. Parameter Sets (VPS/SPS/PPS) --
                egui::CollapsingHeader::new(format!(
                    "Parameter Sets ({})",
                    meta.parameter_set_infos.len()
                ))
                .default_open(true)
                .show(ui, |ui| {
                    if meta.parameter_set_infos.is_empty() {
                        ui.weak("No parameter sets in this .catb.");
                        return;
                    }
                    ui.horizontal(|ui| {
                        ui.label("Filter:");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.structure_filter)
                                .desired_width(180.0)
                                .hint_text("field name contains…"),
                        );
                        if !self.structure_filter.is_empty()
                            && ui.small_button("×").clicked()
                        {
                            self.structure_filter.clear();
                        }
                    });
                    let needle = self.structure_filter.to_ascii_lowercase();
                    for (k, ps) in meta.parameter_set_infos.iter().enumerate() {
                        let title = format!(
                            "{} {}{}",
                            if ps.kind.is_empty() { "?" } else { &ps.kind },
                            ps.id,
                            ps.nal_index
                                .map(|n| format!(" · nal#{n}"))
                                .unwrap_or_default(),
                        );
                        egui::CollapsingHeader::new(title)
                            .id_salt(("bs_ps", k))
                            .default_open(true)
                            .show(ui, |ui| {
                                let rows: Vec<&(String, String)> = ps
                                    .fields
                                    .iter()
                                    .filter(|(n, _)| {
                                        needle.is_empty()
                                            || n.to_ascii_lowercase().contains(&needle)
                                    })
                                    .collect();
                                if rows.is_empty() {
                                    ui.weak(if ps.fields.is_empty() {
                                        "no fields"
                                    } else {
                                        "no fields match the filter"
                                    });
                                    return;
                                }
                                egui::Grid::new(("bs_ps_grid", k))
                                    .striped(true)
                                    .spacing(egui::vec2(12.0, 2.0))
                                    .show(ui, |ui| {
                                        for (n, v) in rows {
                                            ui.label(n);
                                            ui.monospace(v);
                                            ui.end_row();
                                        }
                                    });
                            });
                    }
                });

                // -- 2. Slice Headers — current frame --
                let sh_title = match (decode, fmeta) {
                    (Some(d), Some(m)) => format!(
                        "Slice Headers — decode#{d} ({} slice{})",
                        m.slice_headers.len(),
                        if m.slice_headers.len() == 1 { "" } else { "s" },
                    ),
                    _ => "Slice Headers — current frame".to_string(),
                };
                egui::CollapsingHeader::new(sh_title)
                    .default_open(true)
                    .show(ui, |ui| {
                        let Some(m) = fmeta else {
                            ui.weak(if meta.frames_meta.is_empty() {
                                "No frames_meta in this .catb."
                            } else {
                                "No catb frame maps to this viewer frame — check the offset."
                            });
                            return;
                        };
                        if m.slice_headers.is_empty() {
                            ui.weak("No slice headers recorded for this frame.");
                            return;
                        }
                        for (si, sh) in m.slice_headers.iter().enumerate() {
                            let title = format!(
                                "Slice {si} — {}{}{}",
                                if sh.slice_type_label.is_empty() {
                                    "?"
                                } else {
                                    &sh.slice_type_label
                                },
                                if sh.nal_unit_name.is_empty() {
                                    String::new()
                                } else {
                                    format!(" · {}", sh.nal_unit_name)
                                },
                                sh.nal_index
                                    .map(|n| format!(" · nal#{n}"))
                                    .unwrap_or_default(),
                            );
                            egui::CollapsingHeader::new(title)
                                .id_salt(("bs_slice", si))
                                .default_open(si == 0)
                                .show(ui, |ui| {
                                    egui::Grid::new(("bs_slice_fields", si))
                                        .striped(true)
                                        .spacing(egui::vec2(12.0, 2.0))
                                        .show(ui, |ui| {
                                            for (n, v) in &sh.fields {
                                                ui.label(n);
                                                ui.monospace(v);
                                                ui.end_row();
                                            }
                                        });
                                    if !sh.syntax.is_empty() {
                                        ui.add_space(4.0);
                                        ui.label(
                                            egui::RichText::new("decoded syntax")
                                                .small()
                                                .strong(),
                                        );
                                        egui::Grid::new(("bs_slice_syntax", si))
                                            .striped(true)
                                            .spacing(egui::vec2(12.0, 2.0))
                                            .show(ui, |ui| {
                                                for (n, v, _) in &sh.syntax {
                                                    ui.label(n);
                                                    ui.monospace(v);
                                                    ui.end_row();
                                                }
                                            });
                                    }
                                    for (li, list) in
                                        [(0, &sh.ref_list0), (1, &sh.ref_list1)]
                                    {
                                        if list.is_empty() {
                                            continue;
                                        }
                                        ui.add_space(4.0);
                                        ui.label(
                                            egui::RichText::new(format!("ref_list{li}"))
                                                .small()
                                                .strong(),
                                        );
                                        egui::Grid::new(("bs_slice_refs", si, li))
                                            .striped(true)
                                            .spacing(egui::vec2(12.0, 2.0))
                                            .show(ui, |ui| {
                                                for head in ["idx", "POC", "LT", "label"]
                                                {
                                                    ui.label(
                                                        egui::RichText::new(head)
                                                            .small()
                                                            .strong(),
                                                    );
                                                }
                                                ui.end_row();
                                                for (ri, e) in list.iter().enumerate() {
                                                    ui.monospace(format!("{ri}"));
                                                    ui.monospace(format!("{}", e.poc));
                                                    ui.monospace(if e.long_term {
                                                        "LT"
                                                    } else {
                                                        "-"
                                                    });
                                                    ui.monospace(&e.label);
                                                    ui.end_row();
                                                }
                                            });
                                    }
                                });
                        }
                    });

                // -- 3. DPB — current frame --
                egui::CollapsingHeader::new("DPB — current frame")
                    .default_open(true)
                    .show(ui, |ui| {
                        let Some(m) = fmeta else {
                            ui.weak("No DPB data for this frame.");
                            return;
                        };
                        if m.dpb.is_empty() {
                            ui.weak("Empty DPB.");
                            return;
                        }
                        // Active refs: POCs the current frame's slice ref
                        // lists actually use.
                        let active: std::collections::HashSet<i64> = m
                            .slice_headers
                            .iter()
                            .flat_map(|sh| {
                                sh.ref_list0.iter().chain(sh.ref_list1.iter())
                            })
                            .map(|e| e.poc)
                            .collect();
                        let active_color = egui::Color32::from_rgb(255, 170, 40);
                        egui::Grid::new("bs_dpb_grid")
                            .striped(true)
                            .spacing(egui::vec2(12.0, 2.0))
                            .show(ui, |ui| {
                                for head in ["slot", "POC", "flags", "label"] {
                                    ui.label(
                                        egui::RichText::new(head).small().strong(),
                                    );
                                }
                                ui.end_row();
                                for row in &m.dpb {
                                    let is_active = active.contains(&row.poc);
                                    let mut flags: Vec<&str> = Vec::new();
                                    flags.push(if row.used_for_reference {
                                        "ref"
                                    } else {
                                        "hold"
                                    });
                                    if row.long_term {
                                        flags.push("LT");
                                    }
                                    if row.output_mark {
                                        flags.push("out");
                                    }
                                    let cells = [
                                        row.slot
                                            .map(|s| s.to_string())
                                            .unwrap_or_else(|| "-".to_string()),
                                        row.poc.to_string(),
                                        flags.join(" "),
                                        row.label.clone(),
                                    ];
                                    for cell in cells {
                                        if is_active {
                                            ui.colored_label(
                                                active_color,
                                                egui::RichText::new(cell).monospace(),
                                            );
                                        } else {
                                            ui.monospace(cell);
                                        }
                                    }
                                    ui.end_row();
                                }
                            });
                        ui.weak(
                            "orange = used by the current frame's active slice \
                             reference lists",
                        );
                    });

                // -- 4. Exactness — current frame --
                egui::CollapsingHeader::new("Exactness")
                    .default_open(true)
                    .show(ui, |ui| {
                        let Some(m) = fmeta else {
                            ui.weak("No exactness data for this frame.");
                            return;
                        };
                        if m.exactness_missing.is_empty()
                            && m.block_dropped_rows.is_empty()
                        {
                            ui.weak("No exactness data in this .catb.");
                            return;
                        }
                        let missing_blocks = m
                            .exactness_missing
                            .iter()
                            .filter(|s| !s.is_empty())
                            .count();
                        let dropped: i64 = m.block_dropped_rows.iter().sum();
                        if missing_blocks == 0 && dropped == 0 {
                            ui.colored_label(
                                egui::Color32::from_rgb(90, 200, 100),
                                format!(
                                    "✓ all {} blocks exact · no dropped telemetry rows",
                                    m.exactness_missing.len()
                                ),
                            );
                        } else {
                            if missing_blocks > 0 {
                                ui.colored_label(
                                    egui::Color32::from_rgb(240, 200, 40),
                                    format!(
                                        "⚠ {missing_blocks}/{} blocks with missing \
                                         exactness keys",
                                        m.exactness_missing.len()
                                    ),
                                );
                                let mut keys: Vec<&str> = m
                                    .exactness_missing
                                    .iter()
                                    .filter(|s| !s.is_empty())
                                    .flat_map(|s| s.split(','))
                                    .map(str::trim)
                                    .filter(|s| !s.is_empty())
                                    .collect();
                                keys.sort_unstable();
                                keys.dedup();
                                ui.weak(format!("missing keys: {}", keys.join(", ")));
                            }
                            if dropped > 0 {
                                ui.colored_label(
                                    egui::Color32::from_rgb(240, 200, 40),
                                    format!(
                                        "⚠ {dropped} telemetry rows dropped across {} \
                                         blocks",
                                        m.block_dropped_rows
                                            .iter()
                                            .filter(|&&n| n > 0)
                                            .count()
                                    ),
                                );
                            }
                        }
                        ui.weak(
                            "Per-block detail: select a block on the Viewer tab — \
                             the Inspector shows its missing keys.",
                        );
                    });
            });
        });
    }

    // -- Stats tab (M-B) --------------------------------------------------
    //
    // VQAnalyzer Stats-tab analogue (Syntax Stats + the Picture Stats
    // header line + the Motion vectors scatter), current frame only.

    #[allow(clippy::too_many_lines)]
    fn ui_stats_tab(&mut self, ctx: &egui::Context, snap: &Snapshot) {
        self.ui_transport_and_status(ctx, snap, true);
        self.ui_filmstrip(ctx, snap);

        // Refresh the per-frame aggregate cache (file identity + display
        // index key, FrameDerived convention).
        let file_ptr = snap
            .file
            .as_ref()
            .map(|f| Arc::as_ptr(f) as usize)
            .unwrap_or(0);
        let display_idx = viewer_to_catb_display(snap.viewer_frame, snap.offset);
        if let (Some(file), Some(display_idx)) = (snap.file.as_ref(), display_idx) {
            let current = self
                .stats_cache
                .as_ref()
                .is_some_and(|(fp, di, _)| *fp == file_ptr && *di == display_idx);
            if !current {
                self.stats_cache = compute_frame_stats(file, display_idx)
                    .map(|s| (file_ptr, display_idx, s));
            }
        } else {
            self.stats_cache = None;
        }

        // -- right dock: MV scatter (VQA "Motion vectors" chart) --
        egui::SidePanel::right("bs_stats_mv")
            .default_width(340.0)
            .show(ctx, |ui| {
                use egui_plot::{Plot, PlotPoints, Points};
                ui.heading("MV scatter");
                let Some((_, _, stats)) = self.stats_cache.as_ref() else {
                    ui.weak("No frame data.");
                    return;
                };
                if stats.mv_points.is_empty() {
                    ui.weak("No motion vectors in this frame (intra / no MV data).");
                    return;
                }
                let l0: Vec<[f64; 2]> = stats
                    .mv_points
                    .iter()
                    .filter(|p| p.list == 0)
                    .map(|p| [p.x, p.y])
                    .collect();
                let l1: Vec<[f64; 2]> = stats
                    .mv_points
                    .iter()
                    .filter(|p| p.list == 1)
                    .map(|p| [p.x, p.y])
                    .collect();
                // Symmetric axes around 0 (VQA quarter-pel 2D distribution
                // convention, shown here in px).
                let m = stats
                    .mv_points
                    .iter()
                    .fold(1.0_f64, |m, p| m.max(p.x.abs()).max(p.y.abs()))
                    * 1.1;
                let pts = &stats.mv_points;
                let resp = Plot::new("bs_stats_mv_plot")
                    .x_axis_label("mv_x (px)")
                    .y_axis_label("mv_y (px)")
                    .include_x(-m)
                    .include_x(m)
                    .include_y(-m)
                    .include_y(m)
                    .data_aspect(1.0)
                    .show(ui, |plot_ui| {
                        plot_ui.points(
                            Points::new(PlotPoints::new(l0))
                                .radius(2.0)
                                .name("L0")
                                .color(egui::Color32::from_rgb(255, 150, 20)),
                        );
                        plot_ui.points(
                            Points::new(PlotPoints::new(l1))
                                .radius(2.0)
                                .name("L1")
                                .color(egui::Color32::from_rgb(170, 90, 230)),
                        );
                        (plot_ui.pointer_coordinate(), plot_ui.plot_bounds())
                    });
                // Hover: nearest point → block-coordinate tooltip (corr
                // scatter convention).
                if let (Some(ptr), bounds) = resp.inner {
                    let bw = bounds.width().max(f64::EPSILON);
                    let bh = bounds.height().max(f64::EPSILON);
                    let mut best: Option<(usize, f64)> = None;
                    for (i, p) in pts.iter().enumerate() {
                        let dx = (p.x - ptr.x) / bw;
                        let dy = (p.y - ptr.y) / bh;
                        let dd = dx * dx + dy * dy;
                        if best.map(|(_, bd)| dd < bd).unwrap_or(true) {
                            best = Some((i, dd));
                        }
                    }
                    if let Some((i, dd)) = best {
                        if dd.sqrt() < 0.02 {
                            let p = &pts[i];
                            resp.response.on_hover_text(format!(
                                "{} ({:+.2}, {:+.2}) px · block ({}, {})",
                                if p.list == 0 { "L0" } else { "L1" },
                                p.x,
                                p.y,
                                p.block_x,
                                p.block_y,
                            ));
                        }
                    }
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            if snap.file.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label("Load a .catb to see per-frame syntax / CABAC statistics.");
                });
                return;
            }
            let Some((_, _, stats)) = self.stats_cache.as_ref() else {
                ui.weak("No catb frame maps to this viewer frame — check the offset.");
                return;
            };
            // -- summary line --
            ui.monospace(format!(
                "blocks {} · TUs {} · syntax rows {} · cabac bins {} · {} bits",
                stats.blocks,
                stats.tus,
                stats.syntax_rows,
                stats.cabac_bins,
                format_bits(stats.total_bits),
            ));
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                // -- 1. syntax element aggregate --
                egui::CollapsingHeader::new(format!(
                    "Syntax elements ({})",
                    stats.syntax.len()
                ))
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Filter:");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.stats_filter)
                                .desired_width(180.0)
                                .hint_text("name contains…"),
                        );
                        if !self.stats_filter.is_empty() && ui.small_button("×").clicked() {
                            self.stats_filter.clear();
                        }
                    });
                    let needle = self.stats_filter.to_ascii_lowercase();
                    let total = stats.total_bits.max(1) as f64;
                    egui::Grid::new("bs_stats_syntax_grid")
                        .striped(true)
                        .spacing(egui::vec2(14.0, 2.0))
                        .show(ui, |ui| {
                            for head in ["name", "count", "bits", "% of data"] {
                                ui.label(egui::RichText::new(head).small().strong());
                            }
                            ui.end_row();
                            for row in stats
                                .syntax
                                .iter()
                                .filter(|r| {
                                    needle.is_empty()
                                        || r.name.to_ascii_lowercase().contains(&needle)
                                })
                            {
                                ui.monospace(&row.name);
                                ui.monospace(format!("{}", row.count));
                                ui.monospace(format_bits(row.bits));
                                ui.monospace(format!(
                                    "{:.1}%",
                                    row.bits as f64 / total * 100.0
                                ));
                                ui.end_row();
                            }
                        });
                    ui.weak(
                        "\"(slice)\" rows come from the frame's slice headers \
                         (frames_meta); % is of frame data_bits.",
                    );
                });

                // -- 2. CABAC summary --
                egui::CollapsingHeader::new(format!("CABAC bins ({})", stats.cabac_bins))
                    .default_open(true)
                    .show(ui, |ui| {
                        if stats.cabac.is_empty() {
                            ui.weak(
                                "No CABAC bins captured for this frame — \
                                 re-run with Deep telemetry \
                                 (--telemetry-level full).",
                            );
                            return;
                        }
                        egui::Grid::new("bs_stats_cabac_grid")
                            .striped(true)
                            .spacing(egui::vec2(14.0, 2.0))
                            .show(ui, |ui| {
                                for head in ["name", "bins", "bits/bin", "ctx used"] {
                                    ui.label(egui::RichText::new(head).small().strong());
                                }
                                ui.end_row();
                                for row in &stats.cabac {
                                    ui.monospace(&row.name);
                                    ui.monospace(format!("{}", row.bins));
                                    ui.monospace(format!("{:.2}", row.bits_per_bin()));
                                    ui.monospace(format!("{}", row.ctx_count));
                                    ui.end_row();
                                }
                            });
                    });
            });
        });
    }

    // -- Frame Graph tab -------------------------------------------------------

    fn ui_frame_graph_tab(&mut self, ctx: &egui::Context, snap: &Snapshot) {
        self.ui_transport_and_status(ctx, snap, true);
        self.ui_filmstrip(ctx, snap);
        egui::CentralPanel::default().show(ctx, |ui| {
            use egui_plot::{Legend, Line, Plot, PlotPoints, VLine};

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.graph_show_bits, "frame bits");
                ui.checkbox(&mut self.graph_show_qp, "avg QP");
                if snap.frame_graph_scanning {
                    ui.spinner();
                    ui.weak("scanning frames…");
                }
                if snap.frame_graph_b_scanning {
                    ui.spinner();
                    ui.weak("scanning B…");
                }
                if snap.file_b.is_some() && snap.frame_graph_b.is_some() {
                    ui.weak("B = dashed");
                }
            });

            let Some(points) = snap.frame_graph.as_ref() else {
                if !snap.frame_graph_scanning {
                    ui.centered_and_justified(|ui| {
                        ui.label("Load a .catb to see per-frame bits / QP.");
                    });
                }
                return;
            };

            let bits: Vec<[f64; 2]> = points
                .iter()
                .map(|p| [p.display_idx as f64, p.bits])
                .collect();
            let qps: Vec<[f64; 2]> = points
                .iter()
                .map(|p| [p.display_idx as f64, p.avg_qp])
                .collect();
            // M-D: B's series (dashed, same hues) — plotted on B's own
            // display index like A, so both x-axes mean "display order".
            let (bits_b, qps_b): (Vec<[f64; 2]>, Vec<[f64; 2]>) = snap
                .frame_graph_b
                .as_ref()
                .map(|pb| {
                    (
                        pb.iter()
                            .map(|p| [p.display_idx as f64, p.bits])
                            .collect(),
                        pb.iter()
                            .map(|p| [p.display_idx as f64, p.avg_qp])
                            .collect(),
                    )
                })
                .unwrap_or_default();

            Plot::new("bs_frame_graph")
                .legend(Legend::default())
                .allow_drag(true)
                .allow_zoom(true)
                .allow_scroll(true)
                .x_axis_label("display order")
                .show(ui, |plot_ui| {
                    use egui_plot::LineStyle;
                    // The "(A)" suffix only means something once a
                    // comparison stream exists — single-stream sessions
                    // must look exactly as they did before M-D.
                    let has_b = snap.file_b.is_some();
                    if self.graph_show_bits {
                        plot_ui.line(
                            Line::new(PlotPoints::new(bits))
                                .name(if has_b { "frame bits (A)" } else { "frame bits" })
                                .color(egui::Color32::from_rgb(255, 150, 20)),
                        );
                        if !bits_b.is_empty() {
                            plot_ui.line(
                                Line::new(PlotPoints::new(bits_b))
                                    .name("frame bits (B)")
                                    .style(LineStyle::Dashed { length: 6.0 })
                                    .color(egui::Color32::from_rgb(255, 195, 110)),
                            );
                        }
                    }
                    if self.graph_show_qp {
                        plot_ui.line(
                            Line::new(PlotPoints::new(qps))
                                .name(if has_b { "avg QP (A)" } else { "avg QP" })
                                .color(egui::Color32::from_rgb(225, 60, 60)),
                        );
                        if !qps_b.is_empty() {
                            plot_ui.line(
                                Line::new(PlotPoints::new(qps_b))
                                    .name("avg QP (B)")
                                    .style(LineStyle::Dashed { length: 6.0 })
                                    .color(egui::Color32::from_rgb(245, 130, 130)),
                            );
                        }
                    }
                    // Current frame marker (offset-mapped display slot).
                    if let Some(d) = viewer_to_catb_display(snap.viewer_frame, snap.offset)
                    {
                        plot_ui.vline(
                            VLine::new(d as f64).color(egui::Color32::WHITE).width(1.0),
                        );
                    }
                });
        });
    }
}

// ---------------------------------------------------------------------------
// Free drawing helpers
// ---------------------------------------------------------------------------

/// Bottom-left legend (§5): gradient bar + measured min/max for scalar fills,
/// discrete swatches for Mode.
fn draw_legend(
    painter: &egui::Painter,
    canvas_rect: egui::Rect,
    fill: FillMode,
    stats: &FrameFillStats,
) {
    let pad = 8.0;
    let font = egui::FontId::monospace(11.0);
    let text_color = egui::Color32::from_rgb(235, 235, 235);
    let bg = egui::Color32::from_black_alpha(170);

    match fill {
        // Opportunity and the M-C quality fills have dedicated legends
        // (`draw_opportunity_legend` / `draw_quality_legend`).
        FillMode::None
        | FillMode::Opportunity
        | FillMode::BlockPsnr
        | FillMode::BlockSsim => {}
        FillMode::Mode => {
            let entries = [
                ModeClass::Intra,
                ModeClass::Inter,
                ModeClass::Skip,
                ModeClass::Merge,
            ];
            let w = 4.0 + entries.len() as f32 * 62.0;
            let rect = egui::Rect::from_min_size(
                egui::pos2(
                    canvas_rect.min.x + pad,
                    canvas_rect.max.y - pad - 22.0,
                ),
                egui::vec2(w, 22.0),
            );
            painter.rect_filled(rect, 3.0, bg);
            let mut x = rect.min.x + 4.0;
            for m in entries {
                let sw = egui::Rect::from_min_size(
                    egui::pos2(x, rect.min.y + 5.0),
                    egui::vec2(12.0, 12.0),
                );
                painter.rect_filled(sw, 2.0, mode_color(m));
                painter.text(
                    egui::pos2(x + 16.0, rect.center().y),
                    egui::Align2::LEFT_CENTER,
                    m.label(),
                    font.clone(),
                    text_color,
                );
                x += 62.0;
            }
        }
        _ => {
            let (name, lo, hi) = match fill {
                FillMode::Qp => ("QP", stats.qp_min, stats.qp_max),
                FillMode::Bpp => ("bpp", 0.0, stats.bpp_max),
                FillMode::MvHeat => ("|MV|px", 0.0, stats.mv_max),
                FillMode::CoeffEnergy => ("energy", 0.0, stats.coeff_max),
                FillMode::NonzeroCoeffs => ("nz/px", 0.0, stats.nz_max),
                _ => unreachable!(),
            };
            let bar_w = 90.0_f32;
            let rect = egui::Rect::from_min_size(
                egui::pos2(canvas_rect.min.x + pad, canvas_rect.max.y - pad - 22.0),
                egui::vec2(bar_w + 130.0, 22.0),
            );
            painter.rect_filled(rect, 3.0, bg);
            painter.text(
                egui::pos2(rect.min.x + 4.0, rect.center().y),
                egui::Align2::LEFT_CENTER,
                name,
                font.clone(),
                text_color,
            );
            let bar = egui::Rect::from_min_size(
                egui::pos2(rect.min.x + 48.0, rect.min.y + 6.0),
                egui::vec2(bar_w, 10.0),
            );
            let steps = 24;
            for s in 0..steps {
                let t = s as f32 / (steps - 1) as f32;
                let v = lo + t * (hi - lo);
                let mut sample = FillSample::default();
                match fill {
                    FillMode::Qp => sample.qp = v,
                    FillMode::Bpp => sample.bpp = v,
                    FillMode::MvHeat => sample.mv = v,
                    FillMode::CoeffEnergy => sample.coeff = v,
                    FillMode::NonzeroCoeffs => sample.nz = v,
                    _ => {}
                }
                let color = fill_color(fill, &sample, stats, 1.0)
                    .unwrap_or(egui::Color32::TRANSPARENT);
                let seg = egui::Rect::from_min_size(
                    egui::pos2(bar.min.x + t * (bar_w - bar_w / steps as f32), bar.min.y),
                    egui::vec2(bar_w / steps as f32 + 1.0, bar.height()),
                );
                painter.rect_filled(seg, 0.0, color);
            }
            let fmt = |v: f32| {
                if fill == FillMode::Qp {
                    format!("{v:.0}")
                } else {
                    format!("{v:.2}")
                }
            };
            painter.text(
                egui::pos2(bar.min.x - 3.0, rect.center().y),
                egui::Align2::RIGHT_CENTER,
                fmt(lo),
                font.clone(),
                text_color,
            );
            painter.text(
                egui::pos2(bar.max.x + 3.0, rect.center().y),
                egui::Align2::LEFT_CENTER,
                fmt(hi),
                font,
                text_color,
            );
        }
    }
}

/// Bottom-left legend for the Opportunity fill (M3): diverging gradient bar
/// on the symmetric scale −max|z| .. +max|z|.
fn draw_opportunity_legend(
    painter: &egui::Painter,
    canvas_rect: egui::Rect,
    name: &str,
    zmax: f32,
) {
    let pad = 8.0;
    let font = egui::FontId::monospace(11.0);
    let text_color = egui::Color32::from_rgb(235, 235, 235);
    let bg = egui::Color32::from_black_alpha(170);
    let bar_w = 90.0_f32;
    let name_w = 8.0 * name.chars().count() as f32;
    let rect = egui::Rect::from_min_size(
        egui::pos2(canvas_rect.min.x + pad, canvas_rect.max.y - pad - 22.0),
        egui::vec2(bar_w + name_w + 110.0, 22.0),
    );
    painter.rect_filled(rect, 3.0, bg);
    painter.text(
        egui::pos2(rect.min.x + 4.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        name,
        font.clone(),
        text_color,
    );
    let bar = egui::Rect::from_min_size(
        egui::pos2(rect.min.x + name_w + 48.0, rect.min.y + 6.0),
        egui::vec2(bar_w, 10.0),
    );
    let steps = 24;
    for s in 0..steps {
        let t = s as f32 / (steps - 1) as f32;
        let seg = egui::Rect::from_min_size(
            egui::pos2(bar.min.x + t * (bar_w - bar_w / steps as f32), bar.min.y),
            egui::vec2(bar_w / steps as f32 + 1.0, bar.height()),
        );
        painter.rect_filled(seg, 0.0, diverging_colormap(t as f64));
    }
    painter.text(
        egui::pos2(bar.min.x - 3.0, rect.center().y),
        egui::Align2::RIGHT_CENTER,
        format!("{:+.2}", -zmax),
        font.clone(),
        text_color,
    );
    painter.text(
        egui::pos2(bar.max.x + 3.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        format!("{:+.2}", zmax),
        font,
        text_color,
    );
}

/// Bottom-left legend for the M-C quality fills: finite min/max ramp, or
/// the unmet-precondition reason in the same slot. Bit-exact (`e`) blocks
/// sit outside the ramp by design — they render transparent.
fn draw_quality_legend(
    painter: &egui::Painter,
    canvas_rect: egui::Rect,
    fill: FillMode,
    result: &Result<BlockQuality, QualityUnavailable>,
) {
    let pad = 8.0;
    let font = egui::FontId::monospace(11.0);
    let text_color = egui::Color32::from_rgb(235, 235, 235);
    let bg = egui::Color32::from_black_alpha(170);
    match result {
        Err(reason) => {
            let msg = reason.message();
            let w = 8.0 * msg.chars().count() as f32 + 12.0;
            let rect = egui::Rect::from_min_size(
                egui::pos2(canvas_rect.min.x + pad, canvas_rect.max.y - pad - 22.0),
                egui::vec2(w, 22.0),
            );
            painter.rect_filled(rect, 3.0, bg);
            painter.text(
                egui::pos2(rect.min.x + 6.0, rect.center().y),
                egui::Align2::LEFT_CENTER,
                msg,
                font,
                egui::Color32::from_rgb(255, 190, 90),
            );
        }
        Ok(q) => {
            let (name, lo, hi) = match fill {
                FillMode::BlockPsnr => ("PSNR dB", q.psnr_min, q.psnr_max),
                _ => ("SSIM", q.ssim_min, q.ssim_max),
            };
            let bar_w = 90.0_f32;
            let rect = egui::Rect::from_min_size(
                egui::pos2(canvas_rect.min.x + pad, canvas_rect.max.y - pad - 22.0),
                egui::vec2(bar_w + 175.0, 22.0),
            );
            painter.rect_filled(rect, 3.0, bg);
            painter.text(
                egui::pos2(rect.min.x + 4.0, rect.center().y),
                egui::Align2::LEFT_CENTER,
                name,
                font.clone(),
                text_color,
            );
            let bar = egui::Rect::from_min_size(
                egui::pos2(rect.min.x + 62.0, rect.min.y + 6.0),
                egui::vec2(bar_w, 10.0),
            );
            let steps = 24;
            for s in 0..steps {
                let t = s as f32 / (steps - 1) as f32;
                let v = lo + t * (hi - lo);
                let color = quality_fill_color(fill, v, q, 1.0)
                    .unwrap_or(egui::Color32::TRANSPARENT);
                let seg = egui::Rect::from_min_size(
                    egui::pos2(bar.min.x + t * (bar_w - bar_w / steps as f32), bar.min.y),
                    egui::vec2(bar_w / steps as f32 + 1.0, bar.height()),
                );
                painter.rect_filled(seg, 0.0, color);
            }
            let fmt = |v: f32| {
                if fill == FillMode::BlockPsnr {
                    format!("{v:.1}")
                } else {
                    format!("{v:.3}")
                }
            };
            painter.text(
                egui::pos2(bar.min.x - 3.0, rect.center().y),
                egui::Align2::RIGHT_CENTER,
                fmt(lo),
                font.clone(),
                text_color,
            );
            painter.text(
                egui::pos2(bar.max.x + 3.0, rect.center().y),
                egui::Align2::LEFT_CENTER,
                // VQA convention: bit-exact blocks are 'e', outside the ramp.
                format!("{} (e=exact)", fmt(hi)),
                font,
                text_color,
            );
        }
    }
}

/// Value loupe (§5, M key): magnified grid of the L1 8-px cells around the
/// cursor with numeric values — the only way to read cell values at low zoom.
/// Ported from the sidebar's `draw_value_loupe`.
fn draw_l1_loupe(
    ui: &egui::Ui,
    cursor: egui::Pos2,
    grid: &BitstreamGrid,
    fill: FillMode,
    stream_x: u32,
    stream_y: u32,
) {
    if grid.is_empty() {
        return;
    }
    let center_col = (stream_x / grid.cell) as i32;
    let center_row = (stream_y / grid.cell) as i32;

    const HALF: i32 = 2; // 5×5 window
    const CELL: f32 = 34.0;
    let span = (2 * HALF + 1) as f32;
    let size = span * CELL;

    let screen = ui.ctx().screen_rect();
    let mut origin = egui::pos2(cursor.x + 24.0, cursor.y - size - 12.0);
    origin.x = origin.x.clamp(screen.left() + 4.0, (screen.right() - size - 4.0).max(4.0));
    origin.y = origin.y.clamp(screen.top() + 4.0, (screen.bottom() - size - 4.0).max(4.0));
    let loupe_rect = egui::Rect::from_min_size(origin, egui::vec2(size, size));

    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        ui.id().with("bs_value_loupe"),
    ));
    painter.rect_filled(loupe_rect.expand(2.0), 3.0, egui::Color32::from_black_alpha(230));
    painter.rect_stroke(
        loupe_rect.expand(2.0),
        3.0,
        egui::Stroke::new(1.0, egui::Color32::GRAY),
        egui::StrokeKind::Outside,
    );

    let stats = FrameFillStats {
        qp_min: grid.qp.iter().copied().fold(f32::INFINITY, f32::min),
        qp_max: grid.qp.iter().copied().fold(0.0, f32::max),
        bpp_max: grid.bpp.iter().copied().fold(0.0, f32::max),
        mv_max: grid.mv_mag.iter().copied().fold(0.0, f32::max),
        coeff_max: grid.coeff_energy.iter().copied().fold(0.0, f32::max),
        nz_max: grid.nz_density.iter().copied().fold(0.0, f32::max),
    };
    let font = egui::FontId::monospace(10.0);
    let empty_fill = egui::Color32::from_rgb(20, 22, 28);
    for dr in -HALF..=HALF {
        for dc in -HALF..=HALF {
            let col = center_col + dc;
            let row = center_row + dr;
            let cell_min = egui::pos2(
                origin.x + (dc + HALF) as f32 * CELL,
                origin.y + (dr + HALF) as f32 * CELL,
            );
            let cell_rect = egui::Rect::from_min_size(cell_min, egui::vec2(CELL, CELL));
            let in_range = col >= 0
                && row >= 0
                && (col as u32) < grid.cols
                && (row as u32) < grid.rows;
            if in_range {
                let i = (row as u32 * grid.cols + col as u32) as usize;
                // The loupe reads QP when the fill has no L1 value of its
                // own (None; Opportunity lives on G cells, not L1).
                let loupe_fill = if matches!(fill, FillMode::None | FillMode::Opportunity) {
                    FillMode::Qp
                } else {
                    fill
                };
                let sample = FillSample::from_grid(grid, i);
                let color = fill_color(loupe_fill, &sample, &stats, 1.0)
                    .unwrap_or(empty_fill);
                painter.rect_filled(cell_rect.shrink(0.5), 0.0, color);
                let txt = fill_value_text(loupe_fill, &sample);
                painter.text(
                    cell_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    txt,
                    font.clone(),
                    egui::Color32::WHITE,
                );
            } else {
                painter.rect_filled(cell_rect.shrink(0.5), 0.0, empty_fill);
            }
        }
    }
    // Centre (hovered) cell highlight.
    let center_rect = egui::Rect::from_min_size(
        egui::pos2(origin.x + HALF as f32 * CELL, origin.y + HALF as f32 * CELL),
        egui::vec2(CELL, CELL),
    );
    painter.rect_stroke(
        center_rect,
        0.0,
        egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 220, 40)),
        egui::StrokeKind::Inside,
    );
}

/// One-line text in the legend slot (M-D: Δ reason / ΔMode key). Same
/// geometry as the quality legend's unmet-precondition message.
fn draw_legend_note(painter: &egui::Painter, canvas_rect: egui::Rect, msg: &str) {
    let pad = 8.0;
    let font = egui::FontId::monospace(11.0);
    let bg = egui::Color32::from_black_alpha(170);
    let w = 8.0 * msg.chars().count() as f32 + 12.0;
    let rect = egui::Rect::from_min_size(
        egui::pos2(canvas_rect.min.x + pad, canvas_rect.max.y - pad - 22.0),
        egui::vec2(w, 22.0),
    );
    painter.rect_filled(rect, 3.0, bg);
    painter.text(
        egui::pos2(rect.min.x + 6.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        msg,
        font,
        egui::Color32::from_rgb(235, 235, 235),
    );
}

/// M-D value loupe: magnified L1 cells around the cursor showing the A−B
/// difference — diverging colours on the frame's ±max|Δ| scale for scalar
/// fills, `=` / A-mode label for the Mode agreement fill. Same 5×5 window
/// geometry as [`draw_l1_loupe`].
fn draw_diff_loupe(
    ui: &egui::Ui,
    cursor: egui::Pos2,
    grid: &DiffGrid,
    fill: FillMode,
    stream_x: u32,
    stream_y: u32,
) {
    if grid.is_empty() {
        return;
    }
    let metric = fill.diff_metric();
    let dmax = metric.map(|m| grid.max_abs(m)).unwrap_or(0.0);
    let center_col = (stream_x / grid.cell.max(1)) as i32;
    let center_row = (stream_y / grid.cell.max(1)) as i32;

    const HALF: i32 = 2; // 5×5 window
    const CELL: f32 = 34.0;
    let span = (2 * HALF + 1) as f32;
    let size = span * CELL;

    let screen = ui.ctx().screen_rect();
    let mut origin = egui::pos2(cursor.x + 24.0, cursor.y - size - 12.0);
    origin.x = origin.x.clamp(screen.left() + 4.0, (screen.right() - size - 4.0).max(4.0));
    origin.y = origin.y.clamp(screen.top() + 4.0, (screen.bottom() - size - 4.0).max(4.0));
    let loupe_rect = egui::Rect::from_min_size(origin, egui::vec2(size, size));

    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        ui.id().with("bs_diff_loupe"),
    ));
    painter.rect_filled(loupe_rect.expand(2.0), 3.0, egui::Color32::from_black_alpha(230));
    painter.rect_stroke(
        loupe_rect.expand(2.0),
        3.0,
        egui::Stroke::new(1.0, egui::Color32::GRAY),
        egui::StrokeKind::Outside,
    );

    let font = egui::FontId::monospace(10.0);
    let empty_fill = egui::Color32::from_rgb(20, 22, 28);
    for dr in -HALF..=HALF {
        for dc in -HALF..=HALF {
            let col = center_col + dc;
            let row = center_row + dr;
            let cell_min = egui::pos2(
                origin.x + (dc + HALF) as f32 * CELL,
                origin.y + (dr + HALF) as f32 * CELL,
            );
            let cell_rect = egui::Rect::from_min_size(cell_min, egui::vec2(CELL, CELL));
            let i = (row.max(0) as u32 * grid.cols + col.max(0) as u32) as usize;
            let in_range = col >= 0
                && row >= 0
                && (col as u32) < grid.cols
                && (row as u32) < grid.rows
                && grid.valid.get(i).copied().unwrap_or(false);
            if in_range {
                let (color, txt) = match metric {
                    Some(m) => {
                        let dv = grid.values(m)[i];
                        (
                            opportunity_cell_color(dv, dmax, 1.0),
                            format!("{dv:+.1}"),
                        )
                    }
                    None => {
                        if grid.mode_same(i) {
                            (egui::Color32::from_rgb(90, 90, 90), "=".to_string())
                        } else {
                            (
                                mode_color(grid.mode_a[i]),
                                grid.mode_a[i].label().to_string(),
                            )
                        }
                    }
                };
                painter.rect_filled(cell_rect.shrink(0.5), 0.0, color);
                // Text contrast: black on the light (near-white) middle of
                // the diverging ramp, white on the saturated ends.
                let lum = 0.299 * color.r() as f32
                    + 0.587 * color.g() as f32
                    + 0.114 * color.b() as f32;
                let text_color = if lum > 150.0 {
                    egui::Color32::BLACK
                } else {
                    egui::Color32::WHITE
                };
                painter.text(
                    cell_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    txt,
                    font.clone(),
                    text_color,
                );
            } else {
                painter.rect_filled(cell_rect.shrink(0.5), 0.0, empty_fill);
            }
        }
    }
    let center_rect = egui::Rect::from_min_size(
        egui::pos2(origin.x + HALF as f32 * CELL, origin.y + HALF as f32 * CELL),
        egui::vec2(CELL, CELL),
    );
    painter.rect_stroke(
        center_rect,
        0.0,
        egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 220, 40)),
        egui::StrokeKind::Inside,
    );
}

// ---------------------------------------------------------------------------
// Frame Graph background scan (root spawns this on load — never on the UI
// thread; app.rs scene-detect pattern).
// ---------------------------------------------------------------------------

/// Compute the per-display-frame bits / area-weighted average QP series.
/// Pure with respect to the file — runs on a background thread.
pub fn compute_frame_graph(file: &BitstreamFile) -> Vec<FrameGraphPoint> {
    let mut out = Vec::with_capacity(file.frame_count());
    for display_idx in 0..file.frame_count() {
        let Some(decode_idx) = file.decode_idx(display_idx) else {
            continue;
        };
        let frame = &file.catb.frames[decode_idx];
        // Prefer the direct block parse over blocks_decode: this avoids
        // churning the render-path LRU cache from the scan thread.
        let blocks = file.catb.blocks_for_frame(decode_idx).unwrap_or_default();
        let mut area_sum = 0.0_f64;
        let mut qp_sum = 0.0_f64;
        let mut bits_sum = 0.0_f64;
        for b in &blocks {
            let a = b.w as f64 * b.h as f64;
            area_sum += a;
            qp_sum += b.qp as f64 * a;
            bits_sum += b.bits.max(0) as f64;
        }
        let bits = frame
            .slice_bits
            .map(|b| b as f64)
            .unwrap_or(bits_sum);
        let avg_qp = if area_sum > 0.0 { qp_sum / area_sum } else { 0.0 };
        out.push(FrameGraphPoint {
            display_idx,
            bits,
            avg_qp,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Tests — pure logic only (presets, view math, colours never asserted).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_table_matches_spec() {
        // §3 table: fill + layer sets.
        let r = Preset::Rate.config();
        assert_eq!(r.fill, FillMode::Bpp);
        assert!(r.grid && !r.label);

        let q = Preset::QpMap.config();
        assert_eq!(q.fill, FillMode::Qp);
        assert!(q.grid && q.label);

        // M4: Motion carries live MV arrows.
        let m = Preset::Motion.config();
        assert_eq!(m.fill, FillMode::MvHeat);
        assert!(m.mv && m.grid && !m.part && !m.intra);

        // M4: Mode carries Part outlines + Intra directions.
        let md = Preset::Mode.config();
        assert_eq!(md.fill, FillMode::Mode);
        assert!(md.part && md.intra && md.grid && !md.mv);

        // M3 §3: Opportunity = {fill=Opportunity, Grid, Sel}.
        let o = Preset::Opportunity.config();
        assert_eq!(o.fill, FillMode::Opportunity);
        assert!(o.grid && o.sel && !o.label && !o.mv && !o.part && !o.intra);

        let c = Preset::Clean.config();
        assert_eq!(c.fill, FillMode::None);
        assert!(!c.mv && !c.part && !c.intra && !c.label && !c.grid && !c.sel);
    }

    #[test]
    fn preset_detection_and_custom() {
        for p in PRESETS {
            assert_eq!(matching_preset(&p.config()), Some(p), "{:?}", p);
        }
        // V10: manual edit switches to Custom (None).
        let mut cfg = Preset::Rate.config();
        cfg.label = true;
        assert_eq!(matching_preset(&cfg), None);
        // Re-selecting Rate restores the original layer set.
        assert!(!Preset::Rate.config().label);
    }

    #[test]
    fn preset_cycle_order() {
        assert_eq!(next_preset(&Preset::Rate.config()), Preset::QpMap);
        assert_eq!(next_preset(&Preset::Clean.config()), Preset::Rate);
        // From Custom → Rate.
        let mut cfg = Preset::Rate.config();
        cfg.opacity = 0.31;
        assert_eq!(next_preset(&cfg), Preset::Rate);
    }

    #[test]
    fn view_config_settings_roundtrip() {
        let mut cfg = Preset::Motion.config();
        cfg.opacity = 0.42;
        cfg.intra = true;
        let s = cfg.to_settings(true, false, MvSource::Mvd);
        assert_eq!(s.fill, "MV-heat");
        assert!(s.layer_mv && s.layer_grid && s.layer_intra && s.show_loupe);
        assert_eq!(s.mv_source, "MVD");
        let back = ViewConfig::from_settings(&s);
        assert_eq!(back, cfg);
        assert_eq!(MvSource::from_label(&s.mv_source), MvSource::Mvd);
    }

    #[test]
    fn mode_preset_matches_after_settings_roundtrip() {
        // A persisted Mode preset (Part+Intra on) must still be detected as
        // the Mode preset after a settings round-trip — layer_intra must
        // survive serialization.
        let s = Preset::Mode
            .config()
            .to_settings(false, false, MvSource::Mv);
        let back = ViewConfig::from_settings(&s);
        assert_eq!(matching_preset(&back), Some(Preset::Mode));
    }

    #[test]
    fn fill_mode_label_roundtrip() {
        for f in FILL_MODES {
            assert_eq!(FillMode::from_label(f.label()), f);
        }
        // Unknown labels (future fills read from an old settings.toml)
        // degrade to None rather than failing.
        assert_eq!(FillMode::from_label("Chroma-heat"), FillMode::None);
    }

    #[test]
    fn quality_fill_color_and_text() {
        let q = BlockQuality {
            psnr: vec![30.0, 45.0, f32::INFINITY],
            ssim: vec![0.8, 0.99, 1.0],
            psnr_min: 30.0,
            psnr_max: 45.0,
            ssim_min: 0.8,
            ssim_max: 1.0,
        };
        // Worst block is most opaque; best is faintest; bit-exact ('e')
        // renders no fill at all.
        let worst = quality_fill_color(FillMode::BlockPsnr, 30.0, &q, 1.0).unwrap();
        let best = quality_fill_color(FillMode::BlockPsnr, 45.0, &q, 1.0).unwrap();
        assert!(worst.a() > best.a());
        assert!(quality_fill_color(FillMode::BlockPsnr, f32::INFINITY, &q, 1.0).is_none());
        // Non-quality fills yield no colour from this path.
        assert!(quality_fill_color(FillMode::Qp, 30.0, &q, 1.0).is_none());
        // VQA 'e' convention for bit-exact PSNR; SSIM keeps its number.
        assert_eq!(quality_value_text(FillMode::BlockPsnr, f32::INFINITY), "e");
        assert_eq!(quality_value_text(FillMode::BlockPsnr, 41.234), "41.2");
        assert_eq!(quality_value_text(FillMode::BlockSsim, 0.98765), "0.988");
        // Keys 1–9 cover exactly the first nine fills; BlockSSIM stays
        // combo-only.
        assert_eq!(FILL_MODES[8], FillMode::BlockPsnr);
        assert_eq!(FILL_MODES[9], FillMode::BlockSsim);
    }

    #[test]
    fn picture_source_labels_unique() {
        // The combo relies on distinct labels for the 5 entries.
        let mut labels: Vec<&str> = PictureSource::ALL.iter().map(|p| p.label()).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), PictureSource::ALL.len());
        assert_eq!(PictureSource::Source.label(), "Source");
    }

    #[test]
    fn opportunity_cell_color_diverges_symmetrically() {
        // Negative → blue-dominant, zero → white, positive → red-dominant;
        // zmax 0 degenerates to the white midpoint (no division by zero).
        let neg = opportunity_cell_color(-2.0, 2.0, 1.0);
        let mid = opportunity_cell_color(0.0, 2.0, 1.0);
        let pos = opportunity_cell_color(2.0, 2.0, 1.0);
        assert!(neg.b() > neg.r());
        assert_eq!((mid.r(), mid.g(), mid.b()), (255, 255, 255));
        assert!(pos.r() > pos.b());
        let degen = opportunity_cell_color(1.0, 0.0, 1.0);
        assert_eq!((degen.r(), degen.g(), degen.b()), (255, 255, 255));
        // Opacity maps to alpha (fill-only rule, §2).
        assert_eq!(opportunity_cell_color(1.0, 1.0, 0.0).a(), 0);
        assert_eq!(opportunity_cell_color(1.0, 1.0, 1.0).a(), 255);
    }

    #[test]
    fn zoom_anchor_keeps_point_fixed() {
        let pan0 = egui::Vec2::new(-12.0, 7.0);
        let (z0, z1) = (1.0_f32, 3.5_f32);
        let anchor = egui::Vec2::new(140.0, 90.0);
        let pan1 = zoom_anchor_pan(pan0, z0, z1, anchor);
        let uv0 = (anchor - pan0) / z0;
        let uv1 = (anchor - pan1) / z1;
        assert!((uv0.x - uv1.x).abs() < 1e-3);
        assert!((uv0.y - uv1.y).abs() < 1e-3);
    }

    #[test]
    fn normalize_handles_degenerate_range() {
        assert_eq!(normalize(5.0, 5.0, 5.0), 1.0);
        assert!((normalize(30.0, 20.0, 40.0) - 0.5).abs() < 1e-6);
        assert_eq!(normalize(10.0, 20.0, 40.0), 0.0);
        assert_eq!(normalize(50.0, 20.0, 40.0), 1.0);
    }

    #[test]
    fn bpp_normalization_uses_frame_max() {
        let stats = FrameFillStats {
            qp_min: 0.0,
            qp_max: 0.0,
            bpp_max: 2.0,
            mv_max: 0.0,
            coeff_max: 0.0,
            nz_max: 0.0,
        };
        let sample = |bpp: f32| FillSample {
            bpp,
            ..Default::default()
        };
        let half = fill_color(FillMode::Bpp, &sample(1.0), &stats, 1.0).unwrap();
        let full = fill_color(FillMode::Bpp, &sample(2.0), &stats, 1.0).unwrap();
        assert!(full.a() > half.a());
        // Opacity 0 → fully transparent fill (V8).
        let zero = fill_color(FillMode::Bpp, &sample(2.0), &stats, 0.0).unwrap();
        assert_eq!(zero.a(), 0);
    }

    #[test]
    fn coeff_energy_fill_normalizes_and_labels() {
        // M-B: CoeffEnergy alpha ramps with frame-max normalization;
        // NonzeroCoeffs likewise. Zero max (no TX data) → base alpha.
        let stats = FrameFillStats {
            qp_min: 0.0,
            qp_max: 0.0,
            bpp_max: 0.0,
            mv_max: 0.0,
            coeff_max: 4.0,
            nz_max: 0.5,
        };
        let s = |coeff: f32, nz: f32| FillSample {
            coeff,
            nz,
            ..Default::default()
        };
        let lo = fill_color(FillMode::CoeffEnergy, &s(1.0, 0.0), &stats, 1.0).unwrap();
        let hi = fill_color(FillMode::CoeffEnergy, &s(4.0, 0.0), &stats, 1.0).unwrap();
        assert!(hi.a() > lo.a());
        // Orange→red ramp: green channel drops as t rises.
        assert!(hi.g() < lo.g());
        let nz_hi = fill_color(FillMode::NonzeroCoeffs, &s(0.0, 0.5), &stats, 1.0).unwrap();
        let nz_lo = fill_color(FillMode::NonzeroCoeffs, &s(0.0, 0.1), &stats, 1.0).unwrap();
        assert!(nz_hi.a() > nz_lo.a());
        // Value text arms.
        assert_eq!(fill_value_text(FillMode::CoeffEnergy, &s(2.5, 0.0)), "2.5");
        assert_eq!(fill_value_text(FillMode::NonzeroCoeffs, &s(0.0, 0.25)), "0.25");
        // Degenerate max → t = 0, still a colour (base alpha).
        let none_stats = FrameFillStats::default();
        assert!(fill_color(FillMode::CoeffEnergy, &s(1.0, 0.0), &none_stats, 1.0).is_some());
    }

    #[test]
    fn filmstrip_colors_by_frame_type() {
        use crate::analysis::bitstream_stats::FrameTypeClass as F;
        let i = filmstrip_color(Some(F::I));
        let p = filmstrip_color(Some(F::P));
        let b = filmstrip_color(Some(F::B));
        // I = red-dominant, P = blue-dominant, B = green-dominant (§4).
        assert!(i.r() > i.g() && i.r() > i.b());
        assert!(p.b() > p.r() && p.b() > p.g());
        assert!(b.g() > b.r() && b.g() > b.b());
        // No catb data → dark grey.
        let none = filmstrip_color(None);
        assert!(none.r() < 80 && none.g() < 80 && none.b() < 80);
    }

    #[test]
    fn qpel_formatting() {
        assert_eq!(qpel((-13, 2)), "(-3.25, 0.5)");
        assert_eq!(qpel((0, 0)), "(0, 0)");
        assert_eq!(qpel((4, -8)), "(1, -2)");
    }
}
