//! M4 bitstream visualization layers (MV arrows / Partition outlines /
//! Intra directions) plus the main-canvas mirror overlay.
//!
//! Implemented from the CC0 .catb v4 format specification
//! (codec-analyzer docs/catb-v4-format.md); no GPL code consulted.
//!
//! The three layer renderers are free functions shared by the Bitstream
//! Analysis window canvas ([`crate::ui::bitstream_window`]) and the main
//! viewer canvas overlay ([`draw_bitstream_overlay`]) — one renderer, two
//! call sites, so both surfaces stay pixel-identical. The main-canvas
//! overlay mirrors the window's [`ViewConfig`] directly (single source of
//! truth, §D): there is no separate overlay configuration.

use std::sync::Arc;

use eframe::egui;

use crate::analysis::bitstream_stats::{
    aggregate_block_tx, extract_intra_modes, intra_grid_dim, intra_subblock_pos, lod_cell_size,
    rasterize_blocks_tx, use_lod, BitstreamFile, BitstreamGrid, BlockTxAgg, IntraDir,
};
use crate::core::catb::{BsBlock, BsRef, TxRow};
use crate::ui::bitstream_window::{
    fill_color, frame_fill_stats, FillMode, FillSample, FrameFillStats, ViewConfig,
};

// ---------------------------------------------------------------------------
// Layer geometry + constants
// ---------------------------------------------------------------------------

/// Stream-px → screen mapping shared by all layer renderers.
#[derive(Debug, Clone, Copy)]
pub struct LayerGeom {
    /// Screen position of stream pixel (0, 0).
    pub origin: egui::Pos2,
    /// Screen px per stream px, horizontal.
    pub zx: f32,
    /// Screen px per stream px, vertical.
    pub zy: f32,
    /// Culling rect (canvas / image area).
    pub clip: egui::Rect,
}

impl LayerGeom {
    pub fn rect(&self, x: f32, y: f32, w: f32, h: f32) -> egui::Rect {
        egui::Rect::from_min_size(
            egui::pos2(self.origin.x + x * self.zx, self.origin.y + y * self.zy),
            egui::vec2(w * self.zx, h * self.zy),
        )
    }

    pub fn pos(&self, x: f32, y: f32) -> egui::Pos2 {
        egui::pos2(self.origin.x + x * self.zx, self.origin.y + y * self.zy)
    }

    /// Effective zoom for LOD decisions (min of the two axes — anisotropy
    /// only occurs on stream↔viewer resolution mismatch).
    pub fn zoom(&self) -> f32 {
        self.zx.min(self.zy)
    }
}

/// MV / Intra layers are skipped below this zoom (same threshold as the
/// fill's LOD switch — sub-CU geometry is unreadable there).
pub const LAYER_MIN_ZOOM: f32 = 1.5;
/// MV arrows are skipped for blocks narrower than this on screen.
pub const MV_MIN_BLOCK_PX: f32 = 12.0;
/// Hard cap on arrows drawn per frame (painter stall guard).
pub const MV_ARROW_CAP: usize = 4000;

/// Which motion vector the MV layer draws (§C design decision: an explicit
/// combo next to the MV toggle instead of a hidden 3-state cycle — the
/// active source must be visible at all times because MV/MVP/MVD share one
/// arrow style and silently cycling them invites misreading).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MvSource {
    #[default]
    Mv,
    Mvp,
    Mvd,
}

pub const MV_SOURCES: [MvSource; 3] = [MvSource::Mv, MvSource::Mvp, MvSource::Mvd];

impl MvSource {
    pub fn label(&self) -> &'static str {
        match self {
            MvSource::Mv => "MV",
            MvSource::Mvp => "MVP",
            MvSource::Mvd => "MVD",
        }
    }

    pub fn from_label(s: &str) -> Self {
        match s {
            "MVP" => MvSource::Mvp,
            "MVD" => MvSource::Mvd,
            _ => MvSource::Mv,
        }
    }

    fn of_block(&self, b: &BsBlock) -> Option<(i32, i32)> {
        match self {
            MvSource::Mv => b.mv,
            MvSource::Mvp => b.mvp,
            MvSource::Mvd => b.mvd,
        }
    }

    fn of_ref(&self, r: &BsRef) -> Option<(i32, i32)> {
        match self {
            MvSource::Mv => r.mv,
            MvSource::Mvp => r.mvp,
            MvSource::Mvd => r.mvd,
        }
    }
}

// Layer colours (documented in the toolbar tooltips, §C).
const CU_OUTLINE: egui::Color32 = egui::Color32::from_rgba_premultiplied(200, 200, 200, 150);
const PU_OUTLINE: egui::Color32 = egui::Color32::from_rgba_premultiplied(210, 100, 210, 190);
const MV_L0: egui::Color32 = egui::Color32::from_rgb(255, 150, 20);
const MV_L1: egui::Color32 = egui::Color32::from_rgb(170, 90, 230);
const INTRA_LINE: egui::Color32 = egui::Color32::from_rgb(90, 230, 150);
/// TU outline base colour (M-B): yellow, brightness scaled by depth.
const TU_OUTLINE: egui::Color32 = egui::Color32::from_rgb(235, 210, 60);
/// transform_skip TU shading (M-B).
const TU_SKIP_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(60, 55, 15, 60);

// ---------------------------------------------------------------------------
// Part layer — CU outlines on every block + per-PU boundaries where REF
// records carry PU geometry. TU outlines are deliberately absent: TRANSFORM
// records are not parsed (M4 scope; catb.rs module docs).
// ---------------------------------------------------------------------------

/// `refs[i]` is parallel to `blocks[i]` (empty when the block has no REF
/// rows).
pub fn draw_part_layer(
    painter: &egui::Painter,
    geom: &LayerGeom,
    blocks: &[BsBlock],
    refs: &[Vec<BsRef>],
) {
    let cu_stroke = egui::Stroke::new(1.0_f32, CU_OUTLINE);
    let pu_stroke = egui::Stroke::new(1.0_f32, PU_OUTLINE);
    for (i, b) in blocks.iter().enumerate() {
        let rect = geom.rect(b.x as f32, b.y as f32, b.w as f32, b.h as f32);
        // Cull off-screen and sub-3px rects (outline mush at far zoom-out).
        if !rect.intersects(geom.clip) || rect.width() < 3.0 {
            continue;
        }
        painter.rect_stroke(rect, 0.0, cu_stroke, egui::StrokeKind::Inside);
        // PU boundaries: dedupe identical rects (an L0+L1 bi-pred PU yields
        // two REF rows over the same geometry). Skip PUs that span the whole
        // CU — they add no boundary beyond the CU outline.
        let Some(rows) = refs.get(i) else { continue };
        let mut seen: Vec<(i32, i32, i32, i32)> = Vec::with_capacity(rows.len());
        for r in rows {
            if r.pu_w <= 0 || r.pu_h <= 0 {
                continue;
            }
            let key = (r.pu_x, r.pu_y, r.pu_w, r.pu_h);
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            if r.pu_x as u32 == b.x
                && r.pu_y as u32 == b.y
                && r.pu_w as u32 == b.w
                && r.pu_h as u32 == b.h
            {
                continue; // 2Nx2N — CU outline already drawn
            }
            let pr = geom.rect(r.pu_x as f32, r.pu_y as f32, r.pu_w as f32, r.pu_h as f32);
            if pr.intersects(geom.clip) && pr.width() >= 3.0 {
                painter.rect_stroke(pr, 0.0, pu_stroke, egui::StrokeKind::Inside);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MV layer — arrows from PU centres (REF per-PU geometry when present,
// block-level fallback otherwise). Quarter-pel → px (/4) → screen (×zoom).
// ---------------------------------------------------------------------------

fn mv_color(list: Option<&str>) -> egui::Color32 {
    if list.map(|l| l.eq_ignore_ascii_case("l1")).unwrap_or(false) {
        MV_L1
    } else {
        MV_L0
    }
}

/// One arrow: `start` = PU centre (screen), `mv` in quarter-pel. Dashed when
/// `list_index > 0` (higher reference-list entries). Returns false when the
/// arrow was culled.
fn draw_arrow(
    painter: &egui::Painter,
    geom: &LayerGeom,
    start: egui::Pos2,
    mv: (i32, i32),
    color: egui::Color32,
    dashed: bool,
) -> bool {
    if !geom.clip.contains(start) {
        return false;
    }
    // Quarter-pel → luma px (/4, §14.5) → screen px (×zoom per axis).
    let d = egui::vec2(
        mv.0 as f32 / 4.0 * geom.zx,
        mv.1 as f32 / 4.0 * geom.zy,
    );
    let stroke = egui::Stroke::new(1.5_f32, color);
    if d.length() < 1.0 {
        // Zero / sub-pixel motion: a dot instead of an invisible arrow.
        painter.circle_filled(start, 1.5, color);
        return true;
    }
    let end = start + d;
    if dashed {
        painter.add(egui::Shape::dashed_line(&[start, end], stroke, 4.0, 3.0));
    } else {
        painter.line_segment([start, end], stroke);
    }
    // Arrowhead: two barbs at ±150° from the shaft direction.
    let dir = d.normalized();
    let head = (d.length() * 0.35).clamp(3.0, 7.0);
    let rot = |v: egui::Vec2, deg: f32| {
        let (s, c) = deg.to_radians().sin_cos();
        egui::vec2(v.x * c - v.y * s, v.x * s + v.y * c)
    };
    painter.line_segment([end, end + rot(-dir, -30.0) * head], stroke);
    painter.line_segment([end, end + rot(-dir, 30.0) * head], stroke);
    true
}

/// Draw the MV layer. LOD: skipped entirely below [`LAYER_MIN_ZOOM`];
/// per-block skip under [`MV_MIN_BLOCK_PX`] screen width; at most
/// [`MV_ARROW_CAP`] arrows per frame.
pub fn draw_mv_layer(
    painter: &egui::Painter,
    geom: &LayerGeom,
    blocks: &[BsBlock],
    refs: &[Vec<BsRef>],
    source: MvSource,
) {
    if geom.zoom() < LAYER_MIN_ZOOM {
        return;
    }
    let mut drawn = 0usize;
    for (i, b) in blocks.iter().enumerate() {
        if drawn >= MV_ARROW_CAP {
            break;
        }
        if (b.w as f32) * geom.zx < MV_MIN_BLOCK_PX {
            continue;
        }
        let rows = refs.get(i).map(|r| r.as_slice()).unwrap_or(&[]);
        let mut any_pu = false;
        for r in rows {
            let Some(mv) = source.of_ref(r) else { continue };
            if r.pu_w <= 0 || r.pu_h <= 0 {
                continue;
            }
            any_pu = true;
            let centre = geom.pos(
                r.pu_x as f32 + r.pu_w as f32 * 0.5,
                r.pu_y as f32 + r.pu_h as f32 * 0.5,
            );
            let color = mv_color(r.list.as_deref());
            if draw_arrow(painter, geom, centre, mv, color, r.list_index > 0) {
                drawn += 1;
                if drawn >= MV_ARROW_CAP {
                    break;
                }
            }
        }
        if !any_pu {
            // Block-level fallback (REF rows absent or without geometry).
            if let Some(mv) = source.of_block(b) {
                let centre = geom.pos(
                    b.x as f32 + b.w as f32 * 0.5,
                    b.y as f32 + b.h as f32 * 0.5,
                );
                let color = mv_color(b.reference_list.as_deref());
                let dashed = b.reference_list_index.map(|i| i > 0).unwrap_or(false);
                if draw_arrow(painter, geom, centre, mv, color, dashed) {
                    drawn += 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TU layer (M-B) — transform-unit rects: yellow outline (brightness by
// depth), dashed when cbf == 0, transform_skip shaded, Coeffs:n labels at
// zoom ≥ 2x. VQA Transform mode analogue.
// ---------------------------------------------------------------------------

/// TU outline stroke for a transform-tree depth: deeper TUs draw brighter
/// (smaller rects need more contrast to read).
pub fn tu_stroke(depth: i32) -> egui::Stroke {
    let t = (depth.clamp(0, 3) as f32) / 3.0;
    let scale = 0.55 + 0.45 * t;
    let c = egui::Color32::from_rgb(
        (TU_OUTLINE.r() as f32 * scale) as u8,
        (TU_OUTLINE.g() as f32 * scale) as u8,
        (TU_OUTLINE.b() as f32 * scale) as u8,
    );
    egui::Stroke::new(1.0_f32, c)
}

/// True when the resolved TX type label denotes a transform-skip TU.
pub fn is_transform_skip(tx_type: &str) -> bool {
    tx_type.to_ascii_lowercase().contains("skip")
}

/// `tx[i]` is parallel to `blocks[i]` (empty when the block has no TX
/// rows). Skipped entirely below [`LAYER_MIN_ZOOM`] (LOD rule).
pub fn draw_tu_layer(painter: &egui::Painter, geom: &LayerGeom, tx: &[Vec<TxRow>]) {
    if geom.zoom() < LAYER_MIN_ZOOM {
        return;
    }
    let labels = geom.zoom() >= 2.0;
    for rows in tx {
        for t in rows {
            if t.w <= 0 || t.h <= 0 {
                continue;
            }
            let rect = geom.rect(t.x as f32, t.y as f32, t.w as f32, t.h as f32);
            if !rect.intersects(geom.clip) || rect.width() < 3.0 {
                continue;
            }
            if is_transform_skip(&t.tx_type) {
                painter.rect_filled(rect, 0.0, TU_SKIP_FILL);
            }
            let stroke = tu_stroke(t.depth);
            if t.any_cbf() {
                painter.rect_stroke(rect, 0.0, stroke, egui::StrokeKind::Inside);
            } else {
                // cbf = 0 (no coded residual): dashed outline.
                let p = [
                    rect.left_top(),
                    rect.right_top(),
                    rect.right_bottom(),
                    rect.left_bottom(),
                    rect.left_top(),
                ];
                for w in p.windows(2) {
                    painter.add(egui::Shape::dashed_line(w, stroke, 3.0, 3.0));
                }
            }
            if labels && rect.width() >= 34.0 {
                if let Some(nz) = t.nonzero_coeff_count {
                    let size = (rect.height() * 0.25).clamp(7.0, 11.0);
                    painter.text(
                        rect.left_top() + egui::vec2(2.0, 2.0),
                        egui::Align2::LEFT_TOP,
                        format!("Coeffs:{nz}"),
                        egui::FontId::monospace(size),
                        stroke.color,
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LF layer (M-E) — deblocking edge segments from frame_aux.loop_filters,
// colour-coded by applied filter strength (VQA Loop Filter mode convention:
// Green = strong, Yellow = weak, Red = evaluated but not applied).
// ---------------------------------------------------------------------------

/// Strong-filter edge (green).
const LF_STRONG: egui::Color32 = egui::Color32::from_rgb(70, 220, 90);
/// Weak-filter edge (yellow).
const LF_WEAK: egui::Color32 = egui::Color32::from_rgb(235, 210, 60);
/// Evaluated but not applied (red); BS = 0 edges render dimmer so the
/// thresholded-out edges (the interesting "almost filtered" cases) stand
/// out over the dominant BS-0 population (fixtures: ~64% of rows).
const LF_NONE: egui::Color32 = egui::Color32::from_rgb(230, 70, 60);
const LF_NONE_BS0: egui::Color32 = egui::Color32::from_rgba_premultiplied(115, 35, 30, 128);

/// Draw one frame's deblocking-edge rows. Fixture rows are 1-px-thick edge
/// segments (`w`=1 vertical / `h`=1 horizontal); each renders as a line
/// along the block boundary it sits on. Skipped below [`LAYER_MIN_ZOOM`]
/// (edge segments are sub-CU geometry).
pub fn draw_lf_layer(
    painter: &egui::Painter,
    geom: &LayerGeom,
    rows: &[crate::core::catb::LoopFilterRow],
) {
    if geom.zoom() < LAYER_MIN_ZOOM {
        return;
    }
    for r in rows {
        let color = match r.filter_strength.as_str() {
            "strong" => LF_STRONG,
            "weak" => LF_WEAK,
            _ if r.boundary_strength == 0 => LF_NONE_BS0,
            _ => LF_NONE,
        };
        let (a, b) = if r.vertical {
            (
                geom.pos(r.x as f32, r.y as f32),
                geom.pos(r.x as f32, (r.y + r.h.max(1)) as f32),
            )
        } else {
            (
                geom.pos(r.x as f32, r.y as f32),
                geom.pos((r.x + r.w.max(1)) as f32, r.y as f32),
            )
        };
        if !egui::Rect::from_two_pos(a, b).intersects(geom.clip) {
            continue;
        }
        painter.line_segment([a, b], egui::Stroke::new(1.5_f32, color));
    }
}

// ---------------------------------------------------------------------------
// Intra layer — direction lines through (sub)block centres, P/DC badges.
// ---------------------------------------------------------------------------

/// `intra[i]` is parallel to `blocks[i]` (empty when the block has no intra
/// luma modes). Multi-mode blocks store their dirs in decode order — z-scan
/// for AVC 4x4 — so cell placement goes through `intra_subblock_pos` (see
/// `extract_intra_modes` for the fixture-backed ordering evidence).
/// Skipped below [`LAYER_MIN_ZOOM`]; badges render at zoom ≥ 2 only.
pub fn draw_intra_layer(
    painter: &egui::Painter,
    geom: &LayerGeom,
    blocks: &[BsBlock],
    intra: &[Vec<IntraDir>],
) {
    if geom.zoom() < LAYER_MIN_ZOOM {
        return;
    }
    let stroke = egui::Stroke::new(1.5_f32, INTRA_LINE);
    let badges = geom.zoom() >= 2.0;
    for (i, b) in blocks.iter().enumerate() {
        let Some(dirs) = intra.get(i).filter(|d| !d.is_empty()) else {
            continue;
        };
        let rect = geom.rect(b.x as f32, b.y as f32, b.w as f32, b.h as f32);
        if !rect.intersects(geom.clip) {
            continue;
        }
        let dim = intra_grid_dim(dirs.len());
        let sub_w = b.w as f32 / dim as f32;
        let sub_h = b.h as f32 / dim as f32;
        for (k, dir) in dirs.iter().enumerate() {
            let Some((col, row)) = intra_subblock_pos(k, dim) else {
                continue; // more dirs than the grid holds — draw what fits
            };
            let centre = geom.pos(
                b.x as f32 + (col as f32 + 0.5) * sub_w,
                b.y as f32 + (row as f32 + 0.5) * sub_h,
            );
            match dir.angle_deg {
                Some(deg) => {
                    // Math convention (y up) → screen (y down): negate y.
                    let (s, c) = deg.to_radians().sin_cos();
                    let v = egui::vec2(c, -s);
                    let half = (sub_w * geom.zx).min(sub_h * geom.zy) * 0.4;
                    if half >= 2.0 {
                        painter.line_segment([centre - v * half, centre + v * half], stroke);
                    }
                }
                None if badges && !dir.badge.is_empty() => {
                    let size = ((sub_h * geom.zy) * 0.35).clamp(7.0, 12.0);
                    painter.text(
                        centre,
                        egui::Align2::CENTER_CENTER,
                        dir.badge,
                        egui::FontId::monospace(size),
                        INTRA_LINE,
                    );
                }
                None => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-frame layer data (REF rows / intra dirs, parallel to the block list)
// ---------------------------------------------------------------------------

/// Parse every block's REF rows (empty vec on blocks without any / on parse
/// errors — a malformed sub-record range must not kill the whole layer).
pub fn build_refs(file: &BitstreamFile, blocks: &[BsBlock]) -> Vec<Vec<BsRef>> {
    blocks
        .iter()
        .map(|b| {
            if b.ref_n > 0 {
                file.catb.refs_for_block(b).unwrap_or_default()
            } else {
                Vec::new()
            }
        })
        .collect()
}

/// Extract every block's intra luma dirs (empty on non-intra blocks).
pub fn build_intra(file: &BitstreamFile, blocks: &[BsBlock]) -> Vec<Vec<IntraDir>> {
    blocks
        .iter()
        .map(|b| {
            if b.syntax_n > 0 {
                extract_intra_modes(&file.catb.syntax_for_block(b).unwrap_or_default())
            } else {
                Vec::new()
            }
        })
        .collect()
}

/// Parse every block's TX rows (M-B TU layer; empty vec on blocks without
/// any / on parse errors — a malformed sub-record range must not kill the
/// whole layer).
pub fn build_tx(file: &BitstreamFile, blocks: &[BsBlock]) -> Vec<Vec<TxRow>> {
    blocks
        .iter()
        .map(|b| {
            if b.tx_n > 0 {
                file.catb.tx_for_block(b).unwrap_or_default()
            } else {
                Vec::new()
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Main-canvas overlay (§D) — mirrors the window's ViewConfig
// ---------------------------------------------------------------------------

/// Cached per-frame overlay data for the main canvas, keyed on file
/// identity (`Arc::as_ptr`) and display index (optional layer data is
/// filled lazily). The pointer key self-invalidates on load-over-load (the
/// new Arc is allocated before the old one drops), but NOT on
/// unload-then-load — the allocator may reuse the freed address — so
/// `unload_bitstream` / `finalize_bitstream_load` must reset this cache
/// explicitly.
#[derive(Default)]
pub struct OverlayCache {
    key: Option<(usize, usize)>,
    blocks: Option<Arc<Vec<BsBlock>>>,
    /// Per-block TX aggregate (M-B fills), parallel to `blocks`.
    tx_agg: Vec<BlockTxAgg>,
    stats: FrameFillStats,
    grid_lod: Option<BitstreamGrid>,
    refs: Option<Vec<Vec<BsRef>>>,
    intra: Option<Vec<Vec<IntraDir>>>,
    /// Per-block TX rows (M-B TU layer), lazy like `refs` / `intra`.
    tx: Option<Vec<Vec<TxRow>>>,
}

/// Everything [`draw_bitstream_overlay`] needs from the app.
pub struct BitstreamOverlayParams<'a> {
    pub painter: &'a egui::Painter,
    /// Screen rect the viewer image occupies (stream px map onto it, R5).
    pub image_rect: egui::Rect,
    /// The Bitstream window's live view config (single source of truth).
    pub view: &'a ViewConfig,
    pub mv_source: MvSource,
    pub file: &'a Arc<BitstreamFile>,
    /// catb display-order frame index (offset already applied).
    pub display_idx: usize,
}

/// Render the window's current fill heatmap + Grid + MV/Part/Intra layers
/// on the main viewer canvas. The Opportunity fill is window-only (it needs
/// the Correlation tab's aligned pair) and renders as no fill here.
pub fn draw_bitstream_overlay(p: &BitstreamOverlayParams, cache: &mut OverlayCache) {
    let (sw, sh) = (p.file.width.max(1), p.file.height.max(1));
    let geom = LayerGeom {
        origin: p.image_rect.min,
        zx: p.image_rect.width() / sw as f32,
        zy: p.image_rect.height() / sh as f32,
        clip: p.image_rect,
    };

    // Refresh the cache for (file, frame).
    let key = (Arc::as_ptr(p.file) as usize, p.display_idx);
    if cache.key != Some(key) {
        *cache = OverlayCache::default();
        let Ok(blocks) = p.file.blocks_display(p.display_idx) else {
            return;
        };
        cache.tx_agg = aggregate_block_tx(p.file, &blocks);
        cache.stats = frame_fill_stats(&blocks, Some(&cache.tx_agg));
        cache.grid_lod = Some(rasterize_blocks_tx(
            &blocks,
            Some(&cache.tx_agg),
            sw,
            sh,
            lod_cell_size(&p.file.catb.meta.codec),
        ));
        cache.blocks = Some(blocks);
        cache.key = Some(key);
    }
    let Some(blocks) = cache.blocks.clone() else {
        return;
    };
    if (p.view.mv || p.view.part) && cache.refs.is_none() {
        cache.refs = Some(build_refs(p.file, &blocks));
    }
    if p.view.intra && cache.intra.is_none() {
        cache.intra = Some(build_intra(p.file, &blocks));
    }
    if p.view.tu && cache.tx.is_none() {
        cache.tx = Some(build_tx(p.file, &blocks));
    }

    // Fill layer — same LOD rule as the window canvas.
    let lod_mode = use_lod(geom.zoom());
    if !matches!(p.view.fill, FillMode::None | FillMode::Opportunity) {
        if let (true, Some(g)) = (lod_mode, cache.grid_lod.as_ref()) {
            let cell = g.cell as f32;
            for r in 0..g.rows {
                for c in 0..g.cols {
                    let i = (r * g.cols + c) as usize;
                    if g.coverage[i] <= 0.0 {
                        continue;
                    }
                    if let Some(color) = fill_color(
                        p.view.fill,
                        &FillSample::from_grid(g, i),
                        &cache.stats,
                        p.view.opacity,
                    ) {
                        let rect = geom.rect(c as f32 * cell, r as f32 * cell, cell, cell);
                        if rect.intersects(geom.clip) {
                            p.painter.rect_filled(rect, 0.0, color);
                        }
                    }
                }
            }
        } else if !lod_mode {
            for (bi, b) in blocks.iter().enumerate() {
                if let Some(color) = fill_color(
                    p.view.fill,
                    &FillSample::from_block(b, cache.tx_agg.get(bi)),
                    &cache.stats,
                    p.view.opacity,
                ) {
                    let rect = geom.rect(b.x as f32, b.y as f32, b.w as f32, b.h as f32);
                    if rect.intersects(geom.clip) {
                        p.painter.rect_filled(rect, 0.0, color);
                    }
                }
            }
        }
    }

    // Grid layer: CTU / 4-MB boundaries (window canvas rule).
    if p.view.grid {
        let step = lod_cell_size(&p.file.catb.meta.codec) as f32;
        let stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_rgba_unmultiplied(0, 220, 0, 130));
        let mut x = step;
        while x < sw as f32 {
            let sx = geom.origin.x + x * geom.zx;
            p.painter.line_segment(
                [
                    egui::pos2(sx, p.image_rect.min.y),
                    egui::pos2(sx, p.image_rect.max.y),
                ],
                stroke,
            );
            x += step;
        }
        let mut y = step;
        while y < sh as f32 {
            let sy = geom.origin.y + y * geom.zy;
            p.painter.line_segment(
                [
                    egui::pos2(p.image_rect.min.x, sy),
                    egui::pos2(p.image_rect.max.x, sy),
                ],
                stroke,
            );
            y += step;
        }
    }

    // M4/M-B layers — the exact renderers the window canvas uses.
    if p.view.part {
        if let Some(refs) = cache.refs.as_ref() {
            draw_part_layer(p.painter, &geom, &blocks, refs);
        }
    }
    if p.view.tu {
        if let Some(tx) = cache.tx.as_ref() {
            draw_tu_layer(p.painter, &geom, tx);
        }
    }
    if p.view.mv {
        if let Some(refs) = cache.refs.as_ref() {
            draw_mv_layer(p.painter, &geom, &blocks, refs, p.mv_source);
        }
    }
    if p.view.intra {
        if let Some(intra) = cache.intra.as_ref() {
            draw_intra_layer(p.painter, &geom, &blocks, intra);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — pure logic only.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mv_source_label_roundtrip() {
        for s in MV_SOURCES {
            assert_eq!(MvSource::from_label(s.label()), s);
        }
        // Unknown labels (old settings.toml) degrade to the default MV.
        assert_eq!(MvSource::from_label("bogus"), MvSource::Mv);
        assert_eq!(MvSource::default(), MvSource::Mv);
    }

    #[test]
    fn layer_geom_maps_stream_to_screen() {
        let g = LayerGeom {
            origin: egui::pos2(10.0, 20.0),
            zx: 2.0,
            zy: 3.0,
            clip: egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(500.0, 500.0)),
        };
        let r = g.rect(4.0, 8.0, 16.0, 16.0);
        assert_eq!(r.min, egui::pos2(18.0, 44.0));
        assert_eq!(r.size(), egui::vec2(32.0, 48.0));
        assert_eq!(g.zoom(), 2.0, "LOD zoom is the min axis scale");
    }
}
