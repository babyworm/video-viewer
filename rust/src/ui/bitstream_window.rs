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

use crate::analysis::bitstream_panel::format_bits;
use crate::analysis::bitstream_stats::{
    hit_test_min_area, lod_cell_size, rasterize_blocks, use_lod, viewer_to_catb_display,
    BitstreamFile, BitstreamGrid, FrameTypeClass, ModeClass,
};
use crate::analysis::correlation::{
    aggregate_bitstream_to_g, align, class_table, classes_at_g, csv_dump, pearson_r,
    spearman_rho, x_grid, AlignedPair, AnalysisFrameGrids, BitstreamG, CorrScanRequest,
    CorrScanResult, XMetric, YMetric, G_SIZES, PRESET_PAIRS,
};
use crate::analysis::motion::MotionClass;
use crate::core::catb::BsBlock;
use crate::ui::settings::BitstreamViewSettings;

// ---------------------------------------------------------------------------
// Fill / layer / preset model (§2, §3) — pure data, unit-tested below.
// ---------------------------------------------------------------------------

/// Exclusive fill layer (§2). Opportunity arrives in M3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillMode {
    None,
    Qp,
    Bpp,
    Mode,
    MvHeat,
}

impl FillMode {
    pub fn label(&self) -> &'static str {
        match self {
            FillMode::None => "None",
            FillMode::Qp => "QP",
            FillMode::Bpp => "bpp",
            FillMode::Mode => "Mode",
            FillMode::MvHeat => "MV-heat",
        }
    }

    pub fn from_label(s: &str) -> Self {
        match s {
            "QP" => FillMode::Qp,
            "bpp" => FillMode::Bpp,
            "Mode" => FillMode::Mode,
            "MV-heat" => FillMode::MvHeat,
            _ => FillMode::None,
        }
    }
}

/// The §2 toolbar tuple: fill + independent toggle layers + fill-only opacity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewConfig {
    pub fill: FillMode,
    /// MV arrows — M4; toggle exists but is disabled in M1.
    pub mv: bool,
    /// Partition outlines — M4; toggle exists but is disabled in M1.
    pub part: bool,
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
            label: s.layer_label,
            grid: s.layer_grid,
            sel: s.layer_sel,
            opacity: s.opacity.clamp(0.0, 1.0),
        }
    }

    pub fn to_settings(self, show_loupe: bool, inspector_collapsed: bool) -> BitstreamViewSettings {
        BitstreamViewSettings {
            fill: self.fill.label().to_string(),
            layer_mv: self.mv,
            layer_part: self.part,
            layer_label: self.label,
            layer_grid: self.grid,
            layer_sel: self.sel,
            opacity: self.opacity,
            show_loupe,
            inspector_collapsed,
        }
    }
}

/// One-click configurations (§3). Opportunity preset arrives with M3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    Rate,
    QpMap,
    Motion,
    Mode,
    Clean,
}

pub const PRESETS: [Preset; 5] = [
    Preset::Rate,
    Preset::QpMap,
    Preset::Motion,
    Preset::Mode,
    Preset::Clean,
];

impl Preset {
    pub fn label(&self) -> &'static str {
        match self {
            Preset::Rate => "Rate",
            Preset::QpMap => "QP Map",
            Preset::Motion => "Motion",
            Preset::Mode => "Mode",
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
            Preset::Motion => ViewConfig {
                fill: FillMode::MvHeat,
                mv: true, // M4 layer — recorded now so the preset is stable
                grid: true,
                ..base
            },
            Preset::Mode => ViewConfig {
                fill: FillMode::Mode,
                part: true, // M4 layer
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
    pub mv_max: f32,
}

/// Compute fill statistics over a frame's L0 block list.
pub fn frame_fill_stats(blocks: &[BsBlock]) -> FrameFillStats {
    let mut s = FrameFillStats {
        qp_min: f32::INFINITY,
        qp_max: f32::NEG_INFINITY,
        bpp_max: 0.0,
        mv_max: 0.0,
    };
    for b in blocks {
        let qp = b.qp as f32;
        s.qp_min = s.qp_min.min(qp);
        s.qp_max = s.qp_max.max(qp);
        let area = (b.w as f32 * b.h as f32).max(1.0);
        s.bpp_max = s.bpp_max.max(b.bits.max(0) as f32 / area);
        if let Some((x, y)) = b.mv {
            s.mv_max = s.mv_max.max((x.abs() + y.abs()) as f32);
        }
    }
    if !s.qp_min.is_finite() {
        s.qp_min = 0.0;
        s.qp_max = 0.0;
    }
    s
}

/// Fill colour for a (qp, bpp, mode, mv) sample under the current fill mode.
/// `opacity` applies to the fill only (§2).
pub fn fill_color(
    fill: FillMode,
    qp: f32,
    bpp: f32,
    mode: ModeClass,
    mv: f32,
    stats: &FrameFillStats,
    opacity: f32,
) -> Option<egui::Color32> {
    let a255 = |t: f32| ((opacity * t).clamp(0.0, 1.0) * 255.0) as u8;
    match fill {
        FillMode::None => None,
        FillMode::Qp => {
            let t = normalize(qp, stats.qp_min, stats.qp_max);
            Some(egui::Color32::from_rgba_unmultiplied(
                225,
                40,
                40,
                a255(0.15 + 0.85 * t),
            ))
        }
        FillMode::Bpp => {
            let t = if stats.bpp_max > 0.0 { (bpp / stats.bpp_max).clamp(0.0, 1.0) } else { 0.0 };
            Some(egui::Color32::from_rgba_unmultiplied(
                255,
                150,
                20,
                a255(0.10 + 0.90 * t),
            ))
        }
        FillMode::Mode => {
            let c = mode_color(mode);
            Some(egui::Color32::from_rgba_unmultiplied(
                c.r(),
                c.g(),
                c.b(),
                a255(1.0),
            ))
        }
        FillMode::MvHeat => {
            let t = if stats.mv_max > 0.0 { (mv / stats.mv_max).clamp(0.0, 1.0) } else { 0.0 };
            Some(egui::Color32::from_rgba_unmultiplied(
                55,
                125,
                255,
                a255(0.10 + 0.90 * t),
            ))
        }
    }
}

/// Short value text for tooltips / labels / loupe under a fill mode.
pub fn fill_value_text(fill: FillMode, qp: f32, bpp: f32, mode: ModeClass, mv: f32) -> String {
    match fill {
        FillMode::None | FillMode::Qp => format!("{qp:.0}"),
        FillMode::Bpp => format!("{bpp:.2}"),
        FillMode::Mode => mode.label().to_string(),
        FillMode::MvHeat => format!("{mv:.1}"),
    }
}

/// Quarter-pel MV pair display, e.g. `(-3.25, 0.5)`.
pub fn qpel(pair: (i32, i32)) -> String {
    format!("({}, {})", pair.0 as f32 / 4.0, pair.1 as f32 / 4.0)
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
    pub frame_image: Option<egui::ColorImage>,
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
    /// Frame Graph series, filled by a background scan thread on load.
    pub frame_graph: Option<Arc<Vec<FrameGraphPoint>>>,
    pub frame_graph_scanning: bool,
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
            frame_graph: None,
            frame_graph_scanning: false,
            corr_active: false,
            corr_analysis: None,
            corr_scan_request: None,
            corr_scan: None,
            corr_scanning: false,
            corr_scan_progress: (0, 0),
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
}

/// Selection: catb display frame + index into that frame's block list.
/// Kept across frame changes (§5: selection persists; Esc / empty click
/// clears it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Selection {
    display_idx: usize,
    block_idx: usize,
}

/// Per-frame derived render data, cached by catb display index.
struct FrameDerived {
    display_idx: usize,
    blocks: Arc<Vec<BsBlock>>,
    stats: FrameFillStats,
    /// L1 canonical 8-px grid (loupe).
    grid_l1: BitstreamGrid,
    /// LOD aggregate grid (64 px HEVC / 32 px AVC) for zoom < 1.5.
    grid_lod: BitstreamGrid,
}

/// Snapshot of the shared state taken at the top of the child pass so the
/// mutex is never held while rendering (immediate viewports run inside the
/// root update — re-locking under a held lock would deadlock).
struct Snapshot {
    file: Option<Arc<BitstreamFile>>,
    image: Option<egui::ColorImage>,
    generation: u64,
    viewer_frame: usize,
    viewer_total: usize,
    viewer_size: Option<(u32, u32)>,
    offset: i32,
    is_playing: bool,
    frame_graph: Option<Arc<Vec<FrameGraphPoint>>>,
    frame_graph_scanning: bool,
    corr_analysis: Option<Arc<CorrAnalysisData>>,
    corr_scan: Option<Arc<CorrScanResult>>,
    corr_scanning: bool,
    corr_scan_progress: (usize, usize),
}

pub struct BitstreamWindow {
    /// Whether the OS window is open (root toggles via menu/panel/CLI).
    pub open: bool,
    pub shared: Arc<Mutex<BitstreamShared>>,

    // UI state below is only touched inside the viewport closure, which the
    // root runs synchronously — plain fields, no lock needed.
    tab: BsTab,
    pub view: ViewConfig,
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
    /// Transient status-strip hint ("MV/Part arrive in M4", …).
    hint: Option<(String, f64)>,
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
}

/// Cached current-frame correlation data, keyed on everything that feeds it.
struct CorrDerived {
    frame_idx: usize,
    display_idx: usize,
    g: u32,
    x: XMetric,
    y: YMetric,
    /// `None` when the X metric needs a previous frame and there is none.
    pair: Option<AlignedPair>,
    /// Bitstream aggregate at G (drives the class table too).
    bs_g: BitstreamG,
    /// Motion classes at G (class table); `None` without a previous frame.
    classes: Option<(Vec<MotionClass>, Vec<bool>, u32, u32)>,
    r: Option<f64>,
    rho: Option<f64>,
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
            show_loupe: false,
            inspector_collapsed: false,
            zoom: None,
            pan: egui::Vec2::ZERO,
            selection: None,
            texture: None,
            texture_gen: u64::MAX,
            derived: None,
            hint: None,
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
        }
    }

    /// Load persisted toggle state (called once at app start).
    pub fn apply_settings(&mut self, s: &BitstreamViewSettings) {
        self.view = ViewConfig::from_settings(s);
        self.show_loupe = s.show_loupe;
        self.inspector_collapsed = s.inspector_collapsed;
    }

    fn set_hint(&mut self, ctx: &egui::Context, text: &str) {
        self.hint = Some((text.to_string(), ctx.input(|i| i.time)));
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

        let settings_before = self
            .view
            .to_settings(self.show_loupe, self.inspector_collapsed);

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
                generation: s.generation,
                viewer_frame: s.viewer_frame,
                viewer_total: s.viewer_total,
                viewer_size: s.viewer_size,
                offset: s.offset,
                is_playing: s.is_playing,
                frame_graph: s.frame_graph.clone(),
                frame_graph_scanning: s.frame_graph_scanning,
                corr_analysis: s.corr_analysis.clone(),
                corr_scan: s.corr_scan.clone(),
                corr_scanning: s.corr_scanning,
                corr_scan_progress: s.corr_scan_progress,
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
                    self.texture =
                        Some(ctx.load_texture("bs_frame", img, egui::TextureOptions::NEAREST));
                    self.texture_gen = snap.generation;
                }

                self.handle_keys(ctx, &snap);

                self.ui_tab_bar(ctx, &snap);
                match self.tab {
                    BsTab::Viewer => self.ui_viewer_tab(ctx, &snap),
                    BsTab::Correlation => self.ui_correlation_tab(ctx, &snap),
                    BsTab::FrameGraph => self.ui_frame_graph_tab(ctx, &snap),
                }

                // Tell the root whether it must keep computing analysis
                // grids for the Correlation tab (update_analysis convention:
                // only pay for visible views).
                {
                    let mut s = shared.lock();
                    let active = self.tab == BsTab::Correlation;
                    if active && !s.corr_active {
                        // Freshly activated: the root computes on its next
                        // pass — wake it up.
                        ctx.request_repaint_of(egui::ViewportId::ROOT);
                    }
                    s.corr_active = active;
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

        let settings_after = self
            .view
            .to_settings(self.show_loupe, self.inspector_collapsed);
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
            nums: [bool; 6],
            v: bool,
            p: bool,
            l: bool,
            s: bool,
            tab: bool,
            i_key: bool,
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
                ],
                v: plain && i.key_pressed(egui::Key::V),
                p: plain && i.key_pressed(egui::Key::P),
                l: plain && i.key_pressed(egui::Key::L),
                s: plain && i.key_pressed(egui::Key::S),
                // Consume Tab so egui's focus traversal doesn't also react.
                tab: i.consume_key(egui::Modifiers::NONE, egui::Key::Tab),
                i_key: plain && i.key_pressed(egui::Key::I),
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
        let fills = [
            FillMode::None,
            FillMode::Qp,
            FillMode::Bpp,
            FillMode::Mode,
            FillMode::MvHeat,
        ];
        for (i, &pressed) in k.nums.iter().enumerate() {
            if pressed {
                if let Some(&f) = fills.get(i) {
                    self.view.fill = f;
                } else {
                    self.set_hint(ctx, "Opportunity fill arrives in M3");
                }
            }
        }
        if k.v || k.p {
            // MV / Part layers are disabled placeholders in M1.
            self.set_hint(ctx, "MV / Part layers arrive in M4");
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
        if k.esc {
            self.selection = None;
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
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.monospace(self.sync_readout(snap));
                });
            });
        });
    }

    // -- toolbar (§2) --------------------------------------------------------

    fn ui_toolbar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("bs_toolbar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
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
                        for f in [
                            FillMode::None,
                            FillMode::Qp,
                            FillMode::Bpp,
                            FillMode::Mode,
                            FillMode::MvHeat,
                        ] {
                            ui.selectable_value(&mut self.view.fill, f, f.label());
                        }
                    });

                ui.separator();
                // MV / Part: visible but disabled — the layer system's slots
                // are fixed now, the renderers arrive in M4.
                ui.add_enabled(false, egui::SelectableLabel::new(self.view.mv, "MV"))
                    .on_disabled_hover_text("MV arrows arrive in M4");
                ui.add_enabled(false, egui::SelectableLabel::new(self.view.part, "Part"))
                    .on_disabled_hover_text("Partition outlines arrive in M4");
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
        self.ui_toolbar(ctx);
        self.ui_transport_and_status(ctx, snap);
        self.ui_filmstrip(ctx, snap);
        self.ui_inspector(ctx, snap);

        // Refresh per-frame derived data (blocks, stats, grids). The frame
        // texture itself is uploaded tab-independently in `show()`.
        self.refresh_derived(snap);

        egui::CentralPanel::default().show(ctx, |ui| {
            self.ui_canvas(ui, snap);
        });
    }

    fn refresh_derived(&mut self, snap: &Snapshot) {
        let Some(file) = snap.file.as_ref() else {
            self.derived = None;
            return;
        };
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
            .map(|d| d.display_idx == display_idx)
            .unwrap_or(false)
        {
            return; // still current
        }
        let Ok(blocks) = file.blocks_display(display_idx) else {
            self.derived = None;
            return;
        };
        let (w, h) = (file.width.max(1), file.height.max(1));
        let stats = frame_fill_stats(&blocks);
        let grid_l1 = rasterize_blocks(&blocks, w, h, 8);
        let lod = lod_cell_size(&file.catb.meta.codec);
        let grid_lod = rasterize_blocks(&blocks, w, h, lod);
        self.derived = Some(FrameDerived {
            display_idx,
            blocks,
            stats,
            grid_l1,
            grid_lod,
        });
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

        let img_w = vw_f * zoom;
        let img_h = vh_f * zoom;
        let origin = canvas_rect.min
            + egui::vec2(
                (canvas_rect.width() - img_w) * 0.5,
                (canvas_rect.height() - img_h) * 0.5,
            )
            + self.pan;
        let image_rect = egui::Rect::from_min_size(origin, egui::vec2(img_w, img_h));

        if let Some(tex) = &self.texture {
            painter.image(
                tex.id(),
                image_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
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
        if let (Some(d), false) = (derived, peek) {
            let opacity = self.view.opacity;
            let lod_mode = use_lod(zoom);

            // Fill layer — LOD aggregate cells below 1.5x, per-CU rects above.
            if self.view.fill != FillMode::None {
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
                                g.qp[i],
                                g.bpp[i],
                                g.mode[i],
                                g.mv_mag[i],
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
                    for b in d.blocks.iter() {
                        let area = (b.w as f32 * b.h as f32).max(1.0);
                        let bpp = b.bits.max(0) as f32 / area;
                        let mv = b
                            .mv
                            .map(|(x, y)| (x.abs() + y.abs()) as f32)
                            .unwrap_or(0.0);
                        if let Some(color) = fill_color(
                            self.view.fill,
                            b.qp as f32,
                            bpp,
                            ModeClass::from_label(&b.prediction_mode),
                            mv,
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

            // Label layer: per-block value text at zoom ≥ 2x (§2).
            if self.view.label && zoom >= 2.0 && !lod_mode {
                for b in d.blocks.iter() {
                    let rect = block_rect(b.x as f32, b.y as f32, b.w as f32, b.h as f32);
                    if !rect.intersects(canvas_rect) || rect.width() < 18.0 {
                        continue;
                    }
                    let area = (b.w as f32 * b.h as f32).max(1.0);
                    let bpp = b.bits.max(0) as f32 / area;
                    let mv = b
                        .mv
                        .map(|(x, y)| (x.abs() + y.abs()) as f32)
                        .unwrap_or(0.0);
                    let txt = fill_value_text(
                        self.view.fill,
                        b.qp as f32,
                        bpp,
                        ModeClass::from_label(&b.prediction_mode),
                        mv,
                    );
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
            }
        }

        // Click: minimum-area hit test → Inspector; empty area clears (§5).
        if response.clicked() && !response.double_clicked() {
            if let (Some(pos), Some(d)) = (response.interact_pointer_pos(), derived) {
                let sx = (pos.x - origin.x) / zx;
                let sy = (pos.y - origin.y) / zy;
                if sx >= 0.0 && sy >= 0.0 && sx < sw && sy < sh {
                    self.selection = hit_test_min_area(&d.blocks, sx as u32, sy as u32)
                        .map(|block_idx| Selection {
                            display_idx: d.display_idx,
                            block_idx,
                        });
                } else {
                    self.selection = None;
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
                    if self.show_loupe {
                        draw_l1_loupe(ui, pos, &d.grid_l1, self.view.fill, sx as u32, sy as u32);
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
                        // QP/bpp/mode are already in the summary; MV-heat
                        // needs its |MV| (in pixels) prepended.
                        if self.view.fill == FillMode::MvHeat {
                            let mv = b
                                .mv
                                .map(|(x, y)| (x.abs() + y.abs()) as f32)
                                .unwrap_or(0.0);
                            text = format!("|MV| {:.1}px · {text}", mv / 4.0);
                        }
                        response.clone().on_hover_text(text);
                    }
                }
            }
        }

        // Legend — always visible, bottom-left (§5): min/max re-normalize per
        // frame, so hiding it would make colours incomparable.
        if !peek {
            if let Some(d) = self.derived.as_ref() {
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
                    });

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
            });
    }

    // -- filmstrip (§4) --------------------------------------------------------

    fn ui_filmstrip(&mut self, ctx: &egui::Context, snap: &Snapshot) {
        egui::TopBottomPanel::bottom("bs_filmstrip")
            .exact_height(30.0)
            .show(ctx, |ui| {
                let total = snap.viewer_total;
                if total == 0 {
                    ui.weak("filmstrip: no video loaded");
                    return;
                }
                let avail_w = ui.available_width();
                let cell_w = (avail_w / total as f32).max(3.0);
                let strip_w = cell_w * total as f32;
                let h = 24.0_f32;
                egui::ScrollArea::horizontal()
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        let (rect, resp) = ui.allocate_exact_size(
                            egui::vec2(strip_w.max(avail_w), h),
                            egui::Sense::click(),
                        );
                        let painter = ui.painter_at(rect);
                        for i in 0..total {
                            let x0 = rect.min.x + i as f32 * cell_w;
                            let cell = egui::Rect::from_min_size(
                                egui::pos2(x0, rect.min.y + 2.0),
                                egui::vec2((cell_w - 1.0).max(1.0), h - 4.0),
                            );
                            let summary = snap.file.as_ref().and_then(|f| {
                                viewer_to_catb_display(i, snap.offset)
                                    .and_then(|d| f.frame_summary(d))
                            });
                            let class = summary
                                .as_ref()
                                .map(|s| FrameTypeClass::from_label(&s.frame_type));
                            painter.rect_filled(cell, 1.0, filmstrip_color(class));
                            if i == snap.viewer_frame {
                                painter.rect_stroke(
                                    cell.expand(1.0),
                                    1.0,
                                    egui::Stroke::new(2.0, egui::Color32::WHITE),
                                    egui::StrokeKind::Outside,
                                );
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

    fn ui_transport_and_status(&mut self, ctx: &egui::Context, snap: &Snapshot) {
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
                // Transient hint (disabled-layer keys etc.), ~2.5 s.
                if let Some((text, t0)) = &self.hint {
                    if ctx.input(|i| i.time) - t0 < 2.5 {
                        ui.weak(text);
                        ctx.request_repaint_after(std::time::Duration::from_millis(300));
                    } else {
                        self.hint = None;
                    }
                }
            });
        });
    }

    // -- Correlation tab (M2, UX §6) --------------------------------------------

    /// Refresh the cached current-frame correlation derivation.
    fn refresh_corr_derived(&mut self, snap: &Snapshot) {
        let Some(file) = snap.file.as_ref() else {
            self.corr_derived = None;
            return;
        };
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
        let current = self.corr_derived.as_ref().is_some_and(|d| {
            d.frame_idx == snap.viewer_frame
                && d.display_idx == display_idx
                && d.g == self.corr_g
                && d.x == self.corr_x
                && d.y == self.corr_y
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
        let bs_g = aggregate_bitstream_to_g(&derived.grid_l1, sw, sh, self.corr_g);
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
        self.corr_derived = Some(CorrDerived {
            frame_idx: snap.viewer_frame,
            display_idx,
            g: self.corr_g,
            x: self.corr_x,
            y: self.corr_y,
            pair,
            bs_g,
            classes,
            r,
            rho,
        });
    }

    /// True when the stored scan was produced with different X/Y/G settings
    /// than the combos currently show — its numbers must not be presented
    /// next to the current labels.
    fn corr_scan_stale(&self, scan: &CorrScanResult) -> bool {
        scan.request.g != self.corr_g
            || scan.request.x != self.corr_x
            || scan.request.y != self.corr_y
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
                Some(scan) if self.corr_scan_stale(scan) => {
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
        self.ui_transport_and_status(ctx, snap);
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
                        // Preset pairs above the separator (02 §2).
                        for (label, x, y) in PRESET_PAIRS {
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

        // -- right dock: conditional class table (02 §2 view 3) --
        egui::SidePanel::right("bs_corr_classes")
            .default_width(300.0)
            .show(ctx, |ui| {
                ui.heading("Motion class × bitstream");
                ui.separator();
                let Some(d) = self.corr_derived.as_ref() else {
                    ui.label(if snap.file.is_none() {
                        "No .catb loaded."
                    } else {
                        "No data for this frame."
                    });
                    return;
                };
                match &d.classes {
                    Some((classes, valid, cols, rows)) => {
                        let table = class_table(classes, valid, *cols, *rows, &d.bs_g);
                        egui::Grid::new("bs_corr_class_grid")
                            .striped(true)
                            .spacing(egui::vec2(10.0, 4.0))
                            .show(ui, |ui| {
                                for head in ["class", "cells", "QP", "bpp", "|MV|px", "intra%"] {
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
                                        ui.monospace(format!("{:.0}%", row.intra_ratio * 100.0));
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
            let mut note: Option<String> = None;
            if self.corr_range_mode {
                match snap.corr_scan.as_ref() {
                    Some(scan) => {
                        if self.corr_scan_stale(scan) {
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
            Plot::new("bs_corr_scatter")
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
                });
        });
    }

    /// Write the CSV dump for the current mode. Returns a status line.
    fn save_corr_csv(&self, snap: &Snapshot) -> String {
        let csv = if self.corr_range_mode {
            match snap.corr_scan.as_ref() {
                Some(scan) => {
                    if self.corr_scan_stale(scan) {
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

    // -- Frame Graph tab -------------------------------------------------------

    fn ui_frame_graph_tab(&mut self, ctx: &egui::Context, snap: &Snapshot) {
        self.ui_transport_and_status(ctx, snap);
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

            Plot::new("bs_frame_graph")
                .legend(Legend::default())
                .allow_drag(true)
                .allow_zoom(true)
                .allow_scroll(true)
                .x_axis_label("display order")
                .show(ui, |plot_ui| {
                    if self.graph_show_bits {
                        plot_ui.line(
                            Line::new(PlotPoints::new(bits))
                                .name("frame bits")
                                .color(egui::Color32::from_rgb(255, 150, 20)),
                        );
                    }
                    if self.graph_show_qp {
                        plot_ui.line(
                            Line::new(PlotPoints::new(qps))
                                .name("avg QP")
                                .color(egui::Color32::from_rgb(225, 60, 60)),
                        );
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
        FillMode::None => {}
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
                let color = fill_color(
                    fill,
                    if fill == FillMode::Qp { v } else { 0.0 },
                    if fill == FillMode::Bpp { v } else { 0.0 },
                    ModeClass::Unknown,
                    if fill == FillMode::MvHeat { v * 4.0 } else { 0.0 },
                    stats,
                    1.0,
                )
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
                let color = fill_color(
                    // The loupe reads QP when fill is None — always useful.
                    if fill == FillMode::None { FillMode::Qp } else { fill },
                    grid.qp[i],
                    grid.bpp[i],
                    grid.mode[i],
                    grid.mv_mag[i],
                    &stats,
                    1.0,
                )
                .unwrap_or(empty_fill);
                painter.rect_filled(cell_rect.shrink(0.5), 0.0, color);
                let txt = fill_value_text(
                    if fill == FillMode::None { FillMode::Qp } else { fill },
                    grid.qp[i],
                    grid.bpp[i],
                    grid.mode[i],
                    grid.mv_mag[i],
                );
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

        let m = Preset::Motion.config();
        assert_eq!(m.fill, FillMode::MvHeat);
        assert!(m.mv && m.grid);

        let md = Preset::Mode.config();
        assert_eq!(md.fill, FillMode::Mode);
        assert!(md.part && md.grid);

        let c = Preset::Clean.config();
        assert_eq!(c.fill, FillMode::None);
        assert!(!c.mv && !c.part && !c.label && !c.grid && !c.sel);
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
        let s = cfg.to_settings(true, false);
        assert_eq!(s.fill, "MV-heat");
        assert!(s.layer_mv && s.layer_grid && s.show_loupe);
        let back = ViewConfig::from_settings(&s);
        assert_eq!(back, cfg);
    }

    #[test]
    fn fill_mode_label_roundtrip() {
        for f in [
            FillMode::None,
            FillMode::Qp,
            FillMode::Bpp,
            FillMode::Mode,
            FillMode::MvHeat,
        ] {
            assert_eq!(FillMode::from_label(f.label()), f);
        }
        // Unknown labels (e.g. future "Opportunity" read from an old file)
        // degrade to None rather than failing.
        assert_eq!(FillMode::from_label("Opportunity"), FillMode::None);
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
        };
        let half = fill_color(FillMode::Bpp, 0.0, 1.0, ModeClass::Unknown, 0.0, &stats, 1.0)
            .unwrap();
        let full = fill_color(FillMode::Bpp, 0.0, 2.0, ModeClass::Unknown, 0.0, &stats, 1.0)
            .unwrap();
        assert!(full.a() > half.a());
        // Opacity 0 → fully transparent fill (V8).
        let zero = fill_color(FillMode::Bpp, 0.0, 2.0, ModeClass::Unknown, 0.0, &stats, 0.0)
            .unwrap();
        assert_eq!(zero.a(), 0);
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
