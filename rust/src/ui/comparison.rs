use eframe::egui;

use crate::analysis::metrics::{MetricBlock, SpatialMetricKind, SpatialMetricMap};

const MIN_VIEW_ZOOM: f32 = 1.0;
const MAX_VIEW_ZOOM: f32 = 32.0;

/// User actions requested from the comparison panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonUiAction {
    OpenReference,
    /// Open a fresh "current" file via the regular File → Open dialog.
    /// Lets users replace the current slot from inside Video Diff without
    /// closing the comparison view.
    OpenCurrent,
    Refresh,
    Close,
    /// Swap the Reference and Current slots (file paths, readers, decoded
    /// frames, textures). Useful when the user loaded the wrong file as
    /// reference / current and wants to flip them.
    Swap,
}

/// Pane role inside the comparison view. Used to gate drop-target behaviour and
/// hover-overlay rendering on a per-pane basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaneRole {
    Reference,
    Current,
    Metric,
}

/// Three-pane video diff view: reference, current, and selected diff/metric map.
pub struct ComparisonView {
    /// Whether the comparison panel is visible.
    pub is_open: bool,
    /// Selected right-pane metric.
    pub metric_kind: SpatialMetricKind,
    /// Difference heatmap gain for signed luma diff mode.
    pub diff_gain: f32,
    /// Shared zoom/pan viewport for all comparison panes.
    pub viewport: ComparisonViewport,
    /// Reference image pixels (RGB ColorImage). Stored as ColorImage so it
    /// can travel across viewports — TextureHandle is bound to a single ctx.
    /// The deferred Video Diff viewport recreates a texture from this each
    /// time it renders.
    pub ref_image: Option<egui::ColorImage>,
    /// Current image pixels.
    pub current_image: Option<egui::ColorImage>,
    /// Right-pane diff/metric image.
    pub metric_image: Option<egui::ColorImage>,
    /// Last computed spatial metric values, used for labels and summary.
    pub metric_map: Option<SpatialMetricMap>,
    /// Last source image size.
    pub image_size: Option<(u32, u32)>,
    /// Last comparison status/error.
    pub message: Option<String>,
    /// Frame-wide diff statistics (avg/var of Y and MS) for the header readout.
    pub diff_stats: Option<crate::analysis::metrics::DiffStats>,
    /// Last frame's Reference pane rect, in screen coords. Updated each frame
    /// the comparison view renders. Consumed by the app's drop router.
    pub last_ref_pane_rect: Option<egui::Rect>,
    /// Last frame's Current pane rect, in screen coords.
    pub last_current_pane_rect: Option<egui::Rect>,
    /// Action posted by the deferred viewport for the root to drain next tick.
    pub pending_action: Option<ComparisonUiAction>,
    /// Set by the deferred viewport when the OS close button is clicked.
    pub close_requested: bool,
    /// Bumped whenever any cross-viewport-relevant data changes, so the
    /// deferred viewport's cached textures can be invalidated.
    pub generation: u64,
}

/// Normalized image viewport shared by all comparison panes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComparisonViewport {
    /// Center point in normalized image coordinates.
    pub center: egui::Vec2,
    /// Magnification relative to the full image view.
    pub zoom: f32,
}

impl Default for ComparisonViewport {
    fn default() -> Self {
        Self::new()
    }
}

impl ComparisonViewport {
    pub fn new() -> Self {
        Self {
            center: egui::vec2(0.5, 0.5),
            zoom: MIN_VIEW_ZOOM,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn uv_rect(&self) -> egui::Rect {
        let visible = self.visible_size();
        let center = self.clamped_center_for_visible(visible);
        egui::Rect::from_center_size(egui::pos2(center.x, center.y), visible)
    }

    pub fn zoom_at(&mut self, rel_norm: egui::Vec2, factor: f32) {
        if !rel_norm.is_finite() || !factor.is_finite() || factor <= 0.0 {
            return;
        }

        let rel_norm = clamp_vec2(rel_norm, 0.0, 1.0);
        let old_uv = self.uv_rect();
        let anchor = old_uv.min.to_vec2() + rel_norm * old_uv.size();

        self.zoom = (self.zoom * factor).clamp(MIN_VIEW_ZOOM, MAX_VIEW_ZOOM);
        let visible = self.visible_size();
        self.center = anchor - (rel_norm - egui::vec2(0.5, 0.5)) * visible;
        self.clamp_center();
    }

    pub fn pan_by_pixels(&mut self, delta_pixels: egui::Vec2, display_size: egui::Vec2) {
        if !delta_pixels.is_finite()
            || !display_size.is_finite()
            || display_size.x <= 0.0
            || display_size.y <= 0.0
        {
            return;
        }

        let visible = self.visible_size();
        self.center.x -= delta_pixels.x / display_size.x * visible.x;
        self.center.y -= delta_pixels.y / display_size.y * visible.y;
        self.clamp_center();
    }

    fn visible_size(&self) -> egui::Vec2 {
        let side = 1.0 / self.zoom.clamp(MIN_VIEW_ZOOM, MAX_VIEW_ZOOM);
        egui::vec2(side, side)
    }

    fn clamp_center(&mut self) {
        let visible = self.visible_size();
        self.center = self.clamped_center_for_visible(visible);
    }

    fn clamped_center_for_visible(&self, visible: egui::Vec2) -> egui::Vec2 {
        let half = visible * 0.5;
        egui::vec2(
            self.center.x.clamp(half.x, 1.0 - half.x),
            self.center.y.clamp(half.y, 1.0 - half.y),
        )
    }
}

impl Default for ComparisonView {
    fn default() -> Self {
        Self::new()
    }
}

impl ComparisonView {
    pub fn new() -> Self {
        Self {
            is_open: false,
            metric_kind: SpatialMetricKind::SignedDiff,
            diff_gain: 4.0,
            viewport: ComparisonViewport::new(),
            ref_image: None,
            current_image: None,
            metric_image: None,
            metric_map: None,
            image_size: None,
            message: None,
            diff_stats: None,
            last_ref_pane_rect: None,
            last_current_pane_rect: None,
            pending_action: None,
            close_requested: false,
            generation: 0,
        }
    }

    /// Store an RGB buffer as the reference image. The underlying ColorImage
    /// is created here; the deferred Video Diff viewport recreates a texture
    /// from it when rendering. `_ctx` is unused but kept for call-site stability.
    pub fn set_reference_image(&mut self, _ctx: &egui::Context, rgb: &[u8], w: u32, h: u32) {
        self.ref_image = Some(rgb_to_color_image(rgb, w, h));
        self.image_size = Some((w, h));
        self.generation = self.generation.wrapping_add(1);
    }

    /// Store an RGB buffer as the current image.
    pub fn set_current_image(&mut self, _ctx: &egui::Context, rgb: &[u8], w: u32, h: u32) {
        self.current_image = Some(rgb_to_color_image(rgb, w, h));
        self.image_size = Some((w, h));
        self.generation = self.generation.wrapping_add(1);
    }

    /// Recompute the selected right-pane map.
    pub fn compute_metric_map(
        &mut self,
        ctx: &egui::Context,
        reference_rgb: &[u8],
        current_rgb: &[u8],
        w: u32,
        h: u32,
        grid_size: u32,
    ) {
        let map = crate::analysis::metrics::calculate_spatial_metric_map(
            reference_rgb,
            current_rgb,
            w,
            h,
            grid_size,
            self.metric_kind,
        );
        let heatmap = match self.metric_kind {
            SpatialMetricKind::SignedDiff => {
                self.signed_diff_heatmap(reference_rgb, current_rgb, w, h)
            }
            _ => metric_tile_heatmap(&map),
        };

        let _ = ctx;
        self.metric_image = Some(rgb_to_color_image(&heatmap, w, h));
        self.metric_map = Some(map);
        self.image_size = Some((w, h));
        self.message = None;
        self.generation = self.generation.wrapping_add(1);
    }

    /// Clear loaded comparison frames.
    pub fn clear(&mut self) {
        self.viewport.reset();
        self.ref_image = None;
        self.current_image = None;
        self.metric_image = None;
        self.metric_map = None;
        self.image_size = None;
        self.message = None;
        self.generation = self.generation.wrapping_add(1);
    }

    /// Render the comparison view and return requested app actions.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        reference_path: Option<&str>,
        current_path: Option<&str>,
        grid_size: u32,
    ) -> Option<ComparisonUiAction> {
        if !self.is_open {
            return None;
        }

        let mut action = None;
        let prev_metric = self.metric_kind;
        let prev_gain = self.diff_gain;

        ui.horizontal_wrapped(|ui| {
            ui.heading("Video Diff");
            ui.separator();
            if ui.button("Open reference...").clicked() {
                action = Some(ComparisonUiAction::OpenReference);
            }
            if ui
                .button("Swap ⇄")
                .on_hover_text("Swap the Reference and Current slots")
                .clicked()
            {
                action = Some(ComparisonUiAction::Swap);
            }
            if ui.button("Refresh").clicked() {
                action = Some(ComparisonUiAction::Refresh);
            }
            if ui.button("Reset view").clicked() {
                self.viewport.reset();
            }
            if ui.button("Close").clicked() {
                action = Some(ComparisonUiAction::Close);
            }
            ui.separator();
            egui::ComboBox::from_id_salt("comparison_metric_kind")
                .selected_text(self.metric_kind.display_name())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.metric_kind,
                        SpatialMetricKind::SignedDiff,
                        "Diff ΔY",
                    );
                    ui.selectable_value(
                        &mut self.metric_kind,
                        SpatialMetricKind::MsPsnr,
                        "MS-PSNR",
                    );
                    ui.selectable_value(
                        &mut self.metric_kind,
                        SpatialMetricKind::MsSsim,
                        "MS-SSIM",
                    );
                    ui.selectable_value(
                        &mut self.metric_kind,
                        SpatialMetricKind::VmafNegProxy,
                        "VMAF-NEG proxy",
                    );
                });
            if self.metric_kind == SpatialMetricKind::SignedDiff {
                ui.add(
                    egui::Slider::new(&mut self.diff_gain, 1.0..=16.0)
                        .text("diff gain")
                        .clamping(egui::SliderClamping::Always),
                );
            }
        });

        if action.is_none()
            && (self.metric_kind != prev_metric
                || (self.diff_gain - prev_gain).abs() > f32::EPSILON)
        {
            action = Some(ComparisonUiAction::Refresh);
        }

        ui.horizontal_wrapped(|ui| {
            ui.label("Reference:");
            if ui
                .small_button("\u{1F4C2}")
                .on_hover_text("Open reference file…")
                .clicked()
            {
                action = Some(ComparisonUiAction::OpenReference);
            }
            ui.monospace(reference_path.unwrap_or("--"));
            ui.separator();
            ui.label("Current:");
            if ui
                .small_button("\u{1F4C2}")
                .on_hover_text("Open current file…")
                .clicked()
            {
                action = Some(ComparisonUiAction::OpenCurrent);
            }
            ui.monospace(current_path.unwrap_or("--"));
            ui.separator();
            let grid_text = if grid_size > 0 {
                format!("main grid: {} px", grid_size)
            } else {
                "main grid: off (64 px analysis tiles)".to_string()
            };
            ui.label(grid_text);
        });

        if let Some(ref message) = self.message {
            ui.colored_label(egui::Color32::YELLOW, message);
        }

        if let Some(ref map) = self.metric_map {
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("Overall {}:", map.kind.display_name()));
                ui.monospace(format_metric_value(map.kind, map.overall));
                if map.kind.higher_is_better() {
                    ui.weak("higher/green = closer to reference");
                } else {
                    ui.weak("signed ΔY: green +, red -");
                }
            });
        }
        if let Some(ds) = self.diff_stats {
            ui.horizontal_wrapped(|ui| {
                ui.label("Diff stats:");
                ui.monospace(format!(
                    "avg(Y) {:+.3}  var(Y) {:.3}  avg(MS) {:+.3}  var(MS) {:.3}",
                    ds.avg_y, ds.var_y, ds.avg_ms, ds.var_ms,
                ));
                ui.weak("MS = (6Y + Cb + Cr) / 8 (BT.709)");
            });
        }

        ui.separator();

        // Layout: render panes even when image_size is None so users can drop
        // files into Reference / Current panes before any image is loaded.
        // Without an image, fall back to a 16:9 placeholder aspect so the panes
        // still occupy useful space.
        let (w, h) = self.image_size.unwrap_or((16, 9));

        let available = ui.available_size();
        let gap = 8.0_f32;
        let label_h = 22.0_f32;
        let pane_w = ((available.x - gap * 2.0) / 3.0).max(80.0);
        let pane_h = (available.y - label_h).max(80.0);
        let aspect = w as f32 / h as f32;
        let mut image_w = pane_w;
        let mut image_h = image_w / aspect;
        if image_h > pane_h {
            image_h = pane_h;
            image_w = image_h * aspect;
        }
        let display_size = egui::vec2(image_w, image_h);

        // Recreate textures from the stored ColorImages every frame using
        // this viewport's `ctx`. Required because TextureHandle is bound to
        // the ctx that allocated it and Video Diff lives in a deferred
        // viewport with its own ctx.
        let ctx = ui.ctx().clone();
        let ref_texture_id = self
            .ref_image
            .as_ref()
            .map(|img| ctx.load_texture("comparison_ref", img.clone(), egui::TextureOptions::LINEAR).id());
        let current_texture_id = self
            .current_image
            .as_ref()
            .map(|img| ctx.load_texture("comparison_current", img.clone(), egui::TextureOptions::LINEAR).id());
        let metric_texture_id = self
            .metric_image
            .as_ref()
            .map(|img| ctx.load_texture("comparison_metric", img.clone(), egui::TextureOptions::LINEAR).id());
        let metric_label = self.metric_kind.display_name();

        // Reset cached rects each frame; show_pane fills them in for ref/current.
        self.last_ref_pane_rect = None;
        self.last_current_pane_rect = None;
        let drag_in_progress = ui.ctx().input(|i| !i.raw.hovered_files.is_empty());

        ui.horizontal_top(|ui| {
            self.show_pane(
                ui,
                "Reference",
                ref_texture_id,
                display_size,
                PaneRole::Reference,
                drag_in_progress,
            );
            ui.add_space(gap);
            self.show_pane(
                ui,
                "Current",
                current_texture_id,
                display_size,
                PaneRole::Current,
                drag_in_progress,
            );
            ui.add_space(gap);
            self.show_pane(
                ui,
                metric_label,
                metric_texture_id,
                display_size,
                PaneRole::Metric,
                drag_in_progress,
            );
        });

        ui.add_space(4.0);
        ui.weak("Drop a file on Reference or Current to load it there. Mouse wheel zooms every pane; drag pans the shared view.");

        action
    }

    fn show_pane(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        texture_id: Option<egui::TextureId>,
        display_size: egui::Vec2,
        role: PaneRole,
        drag_in_progress: bool,
    ) {
        ui.vertical(|ui| {
            ui.label(egui::RichText::new(label).strong());
            let (rect, response) =
                ui.allocate_exact_size(display_size, egui::Sense::click_and_drag());

            // Cache rect for ref / current so the app's drop handler can route
            // a dropped file to the correct slot.
            match role {
                PaneRole::Reference => self.last_ref_pane_rect = Some(rect),
                PaneRole::Current => self.last_current_pane_rect = Some(rect),
                PaneRole::Metric => {}
            }

            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 0.0, ui.visuals().extreme_bg_color);

            if response.hovered() {
                let scroll_delta = ui.input(|input| input.smooth_scroll_delta.y);
                if scroll_delta.abs() > f32::EPSILON {
                    if let Some(pointer_pos) = ui.input(|input| input.pointer.hover_pos()) {
                        if rect.contains(pointer_pos) {
                            let rel = pointer_pos - rect.min;
                            let rel_norm =
                                egui::vec2(rel.x / rect.width(), rel.y / rect.height());
                            let zoom_change = (scroll_delta * 0.001).clamp(-0.10, 0.10);
                            self.viewport.zoom_at(rel_norm, 1.0 + zoom_change);
                            ui.ctx().request_repaint();
                        }
                    }
                }
            }

            if response.dragged_by(egui::PointerButton::Primary)
                || response.dragged_by(egui::PointerButton::Middle)
            {
                let delta = ui.input(|input| input.pointer.delta());
                self.viewport.pan_by_pixels(delta, rect.size());
                ui.ctx().request_repaint();
            }

            let uv = self.viewport.uv_rect();
            if let Some(texture_id) = texture_id {
                painter.image(texture_id, rect, uv, egui::Color32::WHITE);
            } else {
                painter.rect_stroke(
                    rect,
                    0.0,
                    egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.fg_stroke.color),
                    egui::StrokeKind::Outside,
                );
                let placeholder = match role {
                    PaneRole::Reference => "drop a file here for reference",
                    PaneRole::Current => "drop a file here for current",
                    PaneRole::Metric => "not loaded",
                };
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    placeholder,
                    egui::FontId::proportional(13.0),
                    ui.visuals().weak_text_color(),
                );
            }

            if matches!(role, PaneRole::Metric) {
                if let (Some(map), Some((iw, ih))) = (&self.metric_map, self.image_size) {
                    draw_metric_overlay(&painter, rect, uv, map, iw, ih);
                }
            }

            // Drop-target hover overlay: only ref / current accept files.
            if drag_in_progress && !matches!(role, PaneRole::Metric) {
                let pointer = ui.input(|i| i.pointer.hover_pos());
                let active = pointer.map(|p| rect.contains(p)).unwrap_or(false);
                let tint = if active {
                    egui::Color32::from_rgba_unmultiplied(40, 110, 220, 110)
                } else {
                    egui::Color32::from_rgba_unmultiplied(40, 110, 220, 50)
                };
                painter.rect_filled(rect, 0.0, tint);
                painter.rect_stroke(
                    rect,
                    0.0,
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(60, 140, 240)),
                    egui::StrokeKind::Inside,
                );
                let hint = match role {
                    PaneRole::Reference => "Drop here for Reference",
                    PaneRole::Current => "Drop here for Current",
                    PaneRole::Metric => "",
                };
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    hint,
                    egui::FontId::proportional(16.0),
                    egui::Color32::WHITE,
                );
            }
        });
    }

    fn signed_diff_heatmap(
        &self,
        reference_rgb: &[u8],
        current_rgb: &[u8],
        w: u32,
        h: u32,
    ) -> Vec<u8> {
        let len = (w as usize) * (h as usize);
        let mut out = vec![0u8; len * 3];
        let gain = self.diff_gain as f64;

        for i in 0..len {
            let base = i * 3;
            if base + 2 >= reference_rgb.len() || base + 2 >= current_rgb.len() {
                continue;
            }
            let ref_y = luma_at(reference_rgb, base);
            let cur_y = luma_at(current_rgb, base);
            let delta = (cur_y - ref_y) * gain;
            let mag = delta.abs().min(255.0) as u8;
            if delta >= 0.0 {
                out[base] = 10;
                out[base + 1] = mag;
                out[base + 2] = 40;
            } else {
                out[base] = mag;
                out[base + 1] = 20;
                out[base + 2] = 10;
            }
        }

        out
    }
}

fn clamp_vec2(value: egui::Vec2, min: f32, max: f32) -> egui::Vec2 {
    egui::vec2(value.x.clamp(min, max), value.y.clamp(min, max))
}

fn rgb_to_color_image(rgb: &[u8], w: u32, h: u32) -> egui::ColorImage {
    egui::ColorImage::from_rgb([w as usize, h as usize], rgb)
}

fn luma_at(rgb: &[u8], base: usize) -> f64 {
    0.2126 * rgb[base] as f64 + 0.7152 * rgb[base + 1] as f64 + 0.0722 * rgb[base + 2] as f64
}

fn metric_tile_heatmap(map: &SpatialMetricMap) -> Vec<u8> {
    let width = map.width as usize;
    let height = map.height as usize;
    let mut out = vec![0u8; width * height * 3];

    for block in &map.blocks {
        let color = metric_color(map.kind, block.value);
        fill_block(&mut out, width, height, block, color);
    }

    out
}

fn fill_block(out: &mut [u8], full_w: usize, full_h: usize, block: &MetricBlock, color: [u8; 3]) {
    let x0 = block.x as usize;
    let y0 = block.y as usize;
    let x1 = (x0 + block.w as usize).min(full_w);
    let y1 = (y0 + block.h as usize).min(full_h);
    for y in y0..y1 {
        for x in x0..x1 {
            let idx = (y * full_w + x) * 3;
            out[idx] = color[0];
            out[idx + 1] = color[1];
            out[idx + 2] = color[2];
        }
    }
}

fn metric_color(kind: SpatialMetricKind, value: f64) -> [u8; 3] {
    match kind {
        SpatialMetricKind::SignedDiff => {
            let mag = value.abs().min(64.0) / 64.0;
            if value >= 0.0 {
                [20, (40.0 + 215.0 * mag) as u8, 40]
            } else {
                [(40.0 + 215.0 * mag) as u8, 35, 25]
            }
        }
        SpatialMetricKind::MsPsnr => {
            let t = if value.is_infinite() {
                1.0
            } else {
                ((value - 20.0) / 30.0).clamp(0.0, 1.0)
            };
            quality_color(t)
        }
        SpatialMetricKind::MsSsim => {
            let t = ((value - 0.85) / 0.15).clamp(0.0, 1.0);
            quality_color(t)
        }
        SpatialMetricKind::VmafNegProxy => {
            let t = (value / 100.0).clamp(0.0, 1.0);
            quality_color(t)
        }
    }
}

fn quality_color(t: f64) -> [u8; 3] {
    // Red -> amber -> green, with enough blue removed to keep labels legible.
    let r = (220.0 * (1.0 - t) + 30.0 * t) as u8;
    let g = (45.0 * (1.0 - t) + 210.0 * t) as u8;
    let b = (35.0 * (1.0 - t) + 70.0 * t) as u8;
    [r, g, b]
}

fn draw_metric_overlay(
    painter: &egui::Painter,
    rect: egui::Rect,
    uv: egui::Rect,
    map: &SpatialMetricMap,
    image_w: u32,
    image_h: u32,
) {
    if image_w == 0 || image_h == 0 || uv.width() <= 0.0 || uv.height() <= 0.0 {
        return;
    }

    let stroke = egui::Stroke::new(0.5, egui::Color32::from_black_alpha(140));

    for block in &map.blocks {
        let block_uv = egui::Rect::from_min_max(
            egui::pos2(
                block.x as f32 / image_w as f32,
                block.y as f32 / image_h as f32,
            ),
            egui::pos2(
                (block.x + block.w) as f32 / image_w as f32,
                (block.y + block.h) as f32 / image_h as f32,
            ),
        );
        let Some(visible_uv) = intersect_rect(block_uv, uv) else {
            continue;
        };

        let min = uv_to_screen(rect, uv, visible_uv.min);
        let max = uv_to_screen(rect, uv, visible_uv.max);
        let block_rect = egui::Rect::from_min_max(min, max);
        painter.rect_stroke(block_rect, 0.0, stroke, egui::StrokeKind::Inside);

        if block_rect.width() >= 42.0 && block_rect.height() >= 18.0 {
            painter.text(
                block_rect.center(),
                egui::Align2::CENTER_CENTER,
                format_metric_value(map.kind, block.value),
                egui::FontId::monospace(10.0),
                text_color_for_value(map.kind, block.value),
            );
        }
    }
}

fn intersect_rect(a: egui::Rect, b: egui::Rect) -> Option<egui::Rect> {
    let min = egui::pos2(a.left().max(b.left()), a.top().max(b.top()));
    let max = egui::pos2(a.right().min(b.right()), a.bottom().min(b.bottom()));
    if min.x >= max.x || min.y >= max.y {
        None
    } else {
        Some(egui::Rect::from_min_max(min, max))
    }
}

fn uv_to_screen(rect: egui::Rect, uv: egui::Rect, point: egui::Pos2) -> egui::Pos2 {
    egui::pos2(
        rect.left() + (point.x - uv.left()) / uv.width() * rect.width(),
        rect.top() + (point.y - uv.top()) / uv.height() * rect.height(),
    )
}

fn text_color_for_value(kind: SpatialMetricKind, value: f64) -> egui::Color32 {
    let bg = metric_color(kind, value);
    let luminance = 0.2126 * bg[0] as f64 + 0.7152 * bg[1] as f64 + 0.0722 * bg[2] as f64;
    if luminance > 120.0 {
        egui::Color32::BLACK
    } else {
        egui::Color32::WHITE
    }
}

pub fn format_metric_value(kind: SpatialMetricKind, value: f64) -> String {
    match kind {
        SpatialMetricKind::SignedDiff => format!("{:+.1}", value),
        SpatialMetricKind::MsPsnr => {
            if value.is_infinite() {
                "∞ dB".to_string()
            } else {
                format!("{:.1} dB", value)
            }
        }
        SpatialMetricKind::MsSsim => format!("{:.4}", value),
        SpatialMetricKind::VmafNegProxy => format!("{:.1}", value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewport_default_shows_full_image() {
        let viewport = ComparisonViewport::new();
        let uv = viewport.uv_rect();

        assert_close(uv.left(), 0.0);
        assert_close(uv.top(), 0.0);
        assert_close(uv.right(), 1.0);
        assert_close(uv.bottom(), 1.0);
        assert_close(viewport.zoom, 1.0);
    }

    #[test]
    fn viewport_zoom_at_center_keeps_centered_crop() {
        let mut viewport = ComparisonViewport::new();
        viewport.zoom_at(egui::vec2(0.5, 0.5), 2.0);

        let uv = viewport.uv_rect();
        assert_close(uv.left(), 0.25);
        assert_close(uv.top(), 0.25);
        assert_close(uv.right(), 0.75);
        assert_close(uv.bottom(), 0.75);
    }

    #[test]
    fn viewport_zoom_keeps_hovered_source_point_stable() {
        let mut viewport = ComparisonViewport::new();
        let rel = egui::vec2(0.25, 0.75);
        let before = uv_point(viewport.uv_rect(), rel);

        viewport.zoom_at(rel, 4.0);

        let after = uv_point(viewport.uv_rect(), rel);
        assert_close(before.x, after.x);
        assert_close(before.y, after.y);
    }

    #[test]
    fn viewport_pan_clamps_to_image_bounds() {
        let mut viewport = ComparisonViewport::new();
        viewport.zoom_at(egui::vec2(0.5, 0.5), 4.0);

        viewport.pan_by_pixels(egui::vec2(10_000.0, 10_000.0), egui::vec2(100.0, 100.0));
        let uv = viewport.uv_rect();

        assert!(uv.left() >= 0.0);
        assert!(uv.top() >= 0.0);
        assert!(uv.right() <= 1.0);
        assert!(uv.bottom() <= 1.0);
        assert_close(uv.left(), 0.0);
        assert_close(uv.top(), 0.0);
    }

    #[test]
    fn viewport_ignores_non_finite_zoom_anchor() {
        let mut viewport = ComparisonViewport::new();
        let before = viewport;

        viewport.zoom_at(egui::vec2(f32::NAN, 0.5), 2.0);

        assert_eq!(viewport, before);
    }

    #[test]
    fn viewport_ignores_non_finite_pan_delta() {
        let mut viewport = ComparisonViewport::new();
        viewport.zoom_at(egui::vec2(0.5, 0.5), 2.0);
        let before = viewport;

        viewport.pan_by_pixels(egui::vec2(f32::NAN, 10.0), egui::vec2(100.0, 100.0));

        assert_eq!(viewport, before);
    }

    #[test]
    fn intersect_rect_returns_visible_overlap_only() {
        let a = egui::Rect::from_min_max(egui::pos2(0.25, 0.25), egui::pos2(0.75, 0.75));
        let b = egui::Rect::from_min_max(egui::pos2(0.50, 0.00), egui::pos2(1.00, 0.50));

        let overlap = intersect_rect(a, b).expect("rectangles should overlap");

        assert_close(overlap.left(), 0.50);
        assert_close(overlap.top(), 0.25);
        assert_close(overlap.right(), 0.75);
        assert_close(overlap.bottom(), 0.50);
    }

    fn uv_point(rect: egui::Rect, rel: egui::Vec2) -> egui::Vec2 {
        rect.min.to_vec2() + rel * rect.size()
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }
}
