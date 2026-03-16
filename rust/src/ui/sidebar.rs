use std::sync::Arc;

use eframe::egui;
use parking_lot::Mutex;

use crate::core::pixel::PixelInfo;

/// Which analysis tab is active in the sidebar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisTab {
    Histogram,
    Waveform,
    Vectorscope,
    Metrics,
}

/// Shared analysis data passed to the separate viewport via Arc<Mutex<>>.
/// TextureHandle cannot cross viewports, so waveform is stored as ColorImage
/// and loaded as a texture inside the viewport callback.
pub struct AnalysisShared {
    pub active_tab: AnalysisTab,
    pub histogram_data: Option<std::collections::HashMap<String, Vec<f64>>>,
    pub vectorscope_data: Option<Vec<[f64; 2]>>,
    pub waveform_image: Option<egui::ColorImage>,
    pub psnr: Option<f64>,
    pub ssim: Option<f64>,
    pub frame_diff: Option<f64>,
    /// Set to true by the viewport callback when the user closes the window.
    pub close_requested: bool,
    /// Set to true when the user switches tabs, so the main loop recomputes.
    pub tab_changed: bool,
    /// Generation counter — bumped when any analysis data changes.
    pub generation: u64,
    /// Waveform data generation — bumped only when waveform_image is updated.
    pub waveform_data_gen: u64,
    /// Cached waveform texture handle (avoids re-uploading every frame).
    pub waveform_texture: Option<egui::TextureHandle>,
    /// Generation at which the waveform texture was last loaded.
    pub waveform_texture_gen: u64,
}

impl Default for AnalysisShared {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalysisShared {
    pub fn new() -> Self {
        Self {
            active_tab: AnalysisTab::Histogram,
            histogram_data: None,
            vectorscope_data: None,
            waveform_image: None,
            psnr: None,
            ssim: None,
            frame_diff: None,
            close_requested: false,
            tab_changed: false,
            generation: 0,
            waveform_data_gen: 0,
            waveform_texture: None,
            waveform_texture_gen: 0,
        }
    }
}

/// Sidebar panel showing pixel inspector and analysis toggle.
pub struct Sidebar {
    pub pixel_info: Option<PixelInfo>,
    /// Whether the mouse is currently over the image (values shown vs "--").
    pub pixel_active: bool,
    pub show_analysis: bool,

    /// Shared state for the analysis viewport.
    pub analysis: Arc<Mutex<AnalysisShared>>,

    /// Current grid overlay size (0 = off). Mirrors toolbar value.
    pub grid_size: u32,
    /// Current sub-grid overlay size (0 = off). Mirrors toolbar value.
    pub sub_grid_size: u32,
}

impl Default for Sidebar {
    fn default() -> Self {
        Self::new()
    }
}

impl Sidebar {
    pub fn new() -> Self {
        Self {
            pixel_info: None,
            pixel_active: false,
            show_analysis: false,
            analysis: Arc::new(Mutex::new(AnalysisShared::new())),
            grid_size: 0,
            sub_grid_size: 0,
        }
    }

    pub fn set_pixel_info(&mut self, info: Option<PixelInfo>) {
        self.pixel_info = info;
    }

    /// Render the sidebar contents (pixel inspector only).
    pub fn show(&mut self, ui: &mut egui::Ui) {
        ui.set_min_width(220.0);

        // ── Pixel Inspector ──────────────────────────────────────────
        ui.heading("Pixel Inspector");
        ui.separator();

        // Use a fixed-height frame so the content below doesn't jump
        // when pixel info appears/disappears.
        let min_inspector_height = 160.0;
        egui::Frame::NONE.show(ui, |ui| {
            ui.set_min_height(min_inspector_height);

            if let Some(ref info) = self.pixel_info {
                let active = self.pixel_active;

                // Position
                if active {
                    ui.monospace(format!("X: {}  Y: {}", info.x, info.y));
                } else {
                    ui.monospace("X: --  Y: --");
                }

                // Raw hex
                if active {
                    ui.monospace(format!("Raw: {}", info.raw_hex));
                } else {
                    ui.monospace("Raw: --");
                }

                // Components
                ui.add_space(4.0);
                ui.label("Components:");
                if active {
                    let mut keys: Vec<&String> = info.components.keys().collect();
                    keys.sort();
                    let comp_str: String = keys
                        .iter()
                        .map(|k| format!("{}: {}", k, info.components[*k]))
                        .collect::<Vec<_>>()
                        .join("  ");
                    ui.monospace(comp_str);
                } else {
                    ui.monospace("--");
                }

                // 8x8 Neighbourhood grid with highlight
                ui.add_space(8.0);
                ui.label("Neighborhood (8x8):");
                let grid_id = ui.id().with("pixel_neighborhood");
                let cursor_row = info.nb_cursor_row;
                let cursor_col = info.nb_cursor_col;
                // Current pixel: bright yellow bg + black text
                let cursor_bg = egui::Color32::from_rgb(220, 200, 60);
                // Crosshair: very subtle tint
                let cross_bg = egui::Color32::from_rgba_premultiplied(60, 60, 40, 15);
                let spacing_x = 2.0_f32;
                let spacing_y = 1.0_f32;
                let grid_resp = egui::Grid::new(grid_id)
                    .spacing(egui::vec2(spacing_x, spacing_y))
                    .show(ui, |ui| {
                        for (ri, row) in info.neighborhood.iter().enumerate() {
                            for (ci, cell) in row.iter().enumerate() {
                                let has_value = active && !cell.is_empty();
                                let display = if has_value { cell.as_str() } else { "--" };
                                let is_cursor = ri == cursor_row && ci == cursor_col;
                                let is_cross = ri == cursor_row || ci == cursor_col;

                                let text = if is_cursor && active {
                                    // Current pixel: black text on yellow
                                    egui::RichText::new(display).monospace().size(10.0)
                                        .strong().color(egui::Color32::BLACK)
                                } else if is_cross && active {
                                    // Crosshair: white text
                                    egui::RichText::new(display).monospace().size(10.0)
                                        .color(egui::Color32::WHITE)
                                } else if !has_value {
                                    egui::RichText::new(display).monospace().size(10.0).weak()
                                } else {
                                    egui::RichText::new(display).monospace().size(10.0)
                                };

                                let bg = if is_cursor && active {
                                    Some(cursor_bg)
                                } else if is_cross && active {
                                    Some(cross_bg)
                                } else {
                                    None
                                };

                                if let Some(color) = bg {
                                    egui::Frame::NONE
                                        .fill(color)
                                        .show(ui, |ui| { ui.label(text); });
                                } else {
                                    ui.label(text);
                                }
                            }
                            ui.end_row();
                        }
                    });

                // Draw grid/subgrid boundary lines over the neighborhood
                let gs = self.grid_size;
                let sgs = self.sub_grid_size;
                if (gs > 0 || sgs > 0) && active {
                    let rect = grid_resp.response.rect;
                    let painter = ui.painter();
                    let nb: usize = 8;
                    // Cell dimensions (uniform monospace content)
                    let cell_w = (rect.width() - (nb - 1) as f32 * spacing_x) / nb as f32;
                    let cell_h = (rect.height() - (nb - 1) as f32 * spacing_y) / nb as f32;

                    let half = 4_i64;
                    let px_x0 = info.x as i64 - half;
                    let px_y0 = info.y as i64 - half;

                    let grid_color = egui::Color32::from_rgb(0, 200, 0);
                    let sub_grid_color = egui::Color32::from_rgb(200, 200, 0);

                    // Vertical boundaries (between columns c and c+1)
                    for c in 0..(nb - 1) {
                        let pixel_next = px_x0 + c as i64 + 1;
                        if pixel_next <= 0 { continue; }
                        let is_grid = gs > 0 && pixel_next % gs as i64 == 0;
                        let is_sub = sgs > 0 && pixel_next % sgs as i64 == 0;
                        if is_grid || is_sub {
                            let color = if is_grid { grid_color } else { sub_grid_color };
                            let x = rect.min.x + (c as f32 + 1.0) * cell_w + (c as f32 + 0.5) * spacing_x;
                            painter.line_segment(
                                [egui::pos2(x, rect.min.y), egui::pos2(x, rect.max.y)],
                                egui::Stroke::new(2.0, color),
                            );
                        }
                    }

                    // Horizontal boundaries (between rows r and r+1)
                    for r in 0..(nb - 1) {
                        let pixel_next = px_y0 + r as i64 + 1;
                        if pixel_next <= 0 { continue; }
                        let is_grid = gs > 0 && pixel_next % gs as i64 == 0;
                        let is_sub = sgs > 0 && pixel_next % sgs as i64 == 0;
                        if is_grid || is_sub {
                            let color = if is_grid { grid_color } else { sub_grid_color };
                            let y = rect.min.y + (r as f32 + 1.0) * cell_h + (r as f32 + 0.5) * spacing_y;
                            painter.line_segment(
                                [egui::pos2(rect.min.x, y), egui::pos2(rect.max.x, y)],
                                egui::Stroke::new(2.0, color),
                            );
                        }
                    }
                }
            } else {
                ui.label("Hover over the image to inspect pixels.");
            }
        });

        ui.add_space(8.0);
        ui.separator();

        // ── Analysis toggle ──────────────────────────────────────────
        ui.checkbox(&mut self.show_analysis, "Show Analysis (separate window)");
    }

    /// Show analysis in a separate OS viewport window.
    /// Uses `show_viewport_deferred` to create a real OS window that works
    /// on both native platforms and WSLg (Wayland/XWayland).
    pub fn show_analysis_window(&mut self, ctx: &egui::Context) {
        // Check if the viewport requested close.
        {
            let mut shared = self.analysis.lock();
            if shared.close_requested {
                shared.close_requested = false;
                self.show_analysis = false;
            }
        }

        if !self.show_analysis {
            return;
        }

        let shared = Arc::clone(&self.analysis);

        ctx.show_viewport_deferred(
            egui::ViewportId::from_hash_of("analysis_viewport"),
            egui::ViewportBuilder::default()
                .with_title("Analysis")
                .with_inner_size([450.0, 400.0]),
            move |ctx, class| {
                // If the viewport is being closed by the OS, signal it.
                if matches!(class, egui::ViewportClass::Deferred) {
                    let close = ctx.input(|i| i.viewport().close_requested());
                    if close {
                        shared.lock().close_requested = true;
                        // Still show an empty CentralPanel so egui doesn't warn.
                        egui::CentralPanel::default().show(ctx, |_| {});
                        return;
                    }
                }

                egui::CentralPanel::default().show(ctx, |ui| {
                    // Snapshot shared state and release the lock before rendering.
                    // Only clone waveform_image when the waveform tab is active AND data changed.
                    let (active_tab, histogram, vectorscope, waveform, psnr, ssim, frame_diff) = {
                        let data = shared.lock();
                        let wf = if data.active_tab == AnalysisTab::Waveform
                            && data.waveform_data_gen != data.waveform_texture_gen
                        {
                            data.waveform_image.clone()
                        } else {
                            None
                        };
                        (
                            data.active_tab,
                            data.histogram_data.clone(),
                            data.vectorscope_data.clone(),
                            wf,
                            data.psnr,
                            data.ssim,
                            data.frame_diff,
                        )
                    };

                    // Tab bar + controls — writes back to shared state only on change.
                    let mut tab = active_tab;
                    let mut reset_view = false;
                    let mut zoom_in = false;
                    let mut zoom_out = false;
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut tab, AnalysisTab::Histogram, "Histogram");
                        ui.selectable_value(&mut tab, AnalysisTab::Waveform, "Waveform");
                        ui.selectable_value(&mut tab, AnalysisTab::Vectorscope, "Vectorscope");
                        ui.selectable_value(&mut tab, AnalysisTab::Metrics, "Metrics");
                        ui.separator();
                        if ui.small_button("+").on_hover_text("Zoom in").clicked() {
                            zoom_in = true;
                        }
                        if ui.small_button("-").on_hover_text("Zoom out").clicked() {
                            zoom_out = true;
                        }
                        if ui.small_button("Reset").on_hover_text("Reset view").clicked() {
                            reset_view = true;
                        }
                    });
                    if tab != active_tab {
                        let mut s = shared.lock();
                        s.active_tab = tab;
                        s.tab_changed = true;
                    }

                    ui.separator();

                    match tab {
                        AnalysisTab::Histogram => {
                            Self::render_histogram(ui, &histogram, reset_view, zoom_in, zoom_out);
                        }
                        AnalysisTab::Waveform => {
                            Self::render_waveform_from_image(ctx, ui, &shared, waveform);
                        }
                        AnalysisTab::Vectorscope => {
                            Self::render_vectorscope(ui, &vectorscope, reset_view, zoom_in, zoom_out);
                        }
                        AnalysisTab::Metrics => {
                            Self::render_metrics(ui, psnr, ssim, frame_diff);
                        }
                    }
                });
            },
        );
    }

    // ── Tab implementations (static to avoid borrow conflicts) ─────

    fn render_histogram(
        ui: &mut egui::Ui,
        histogram_data: &Option<std::collections::HashMap<String, Vec<f64>>>,
        reset_view: bool,
        zoom_in: bool,
        zoom_out: bool,
    ) {
        use egui_plot::{Bar, BarChart, Plot};

        // Apply button-driven zoom via plot memory
        if zoom_in || zoom_out {
            let plot_id = ui.id().with("histogram_plot");
            if let Some(mut mem) = egui_plot::PlotMemory::load(ui.ctx(), plot_id) {
                let factor = if zoom_in { 0.8_f32 } else { 1.25 };
                let mut tf = mem.transform();
                let center = tf.frame().center();
                tf.zoom(egui::vec2(factor, factor), center);
                mem.set_transform(tf);
                mem.store(ui.ctx(), plot_id);
            }
        }

        if let Some(ref hist) = histogram_data {
            let plot_height = (ui.available_height() - 40.0).max(120.0);
            let mut plot = Plot::new("histogram_plot")
                .height(plot_height)
                .allow_drag(true)
                .allow_zoom(true)
                .allow_scroll(true)
                .show_axes([true, true])
                .include_x(0.0)
                .include_x(255.0);
            if reset_view {
                plot = plot.reset();
            }

            let channel_colors = [
                ("Y", egui::Color32::WHITE),
                ("R", egui::Color32::RED),
                ("G", egui::Color32::GREEN),
                ("B", egui::Color32::BLUE),
            ];

            plot.show(ui, |plot_ui| {
                for (name, color) in &channel_colors {
                    if let Some(bins) = hist.get(*name) {
                        let bars: Vec<Bar> = bins
                            .iter()
                            .enumerate()
                            .map(|(i, &count)| Bar::new(i as f64, count).width(1.0))
                            .collect();
                        let chart = BarChart::new(bars)
                            .name(*name)
                            .color(*color);
                        plot_ui.bar_chart(chart);
                    }
                }
            });
            ui.add_space(4.0);
            ui.weak("R/G/B channel pixel value distribution (0-255). Peaks indicate dominant intensities.");
        } else {
            ui.label("No histogram data available.");
        }
    }

    /// Render waveform from a ColorImage (loaded as texture in the viewport).
    /// Only re-uploads the texture when `generation` has changed.
    fn render_waveform_from_image(
        ctx: &egui::Context,
        ui: &mut egui::Ui,
        shared: &Arc<Mutex<AnalysisShared>>,
        waveform_image: Option<egui::ColorImage>,
    ) {
        // Check if we need to reload the texture (compare against waveform-specific gen).
        let needs_reload = {
            let s = shared.lock();
            s.waveform_texture.is_none() || s.waveform_texture_gen != s.waveform_data_gen
        };
        if needs_reload {
            if let Some(img) = waveform_image {
                let tex = ctx.load_texture("waveform_viewport", img, egui::TextureOptions::LINEAR);
                let mut s = shared.lock();
                s.waveform_texture = Some(tex);
                s.waveform_texture_gen = s.waveform_data_gen;
            }
        }
        let s = shared.lock();
        if let Some(ref tex) = s.waveform_texture {
            let wf_height = (ui.available_height() - 40.0).max(120.0);
            let size = egui::vec2(ui.available_width(), wf_height);
            ui.image(egui::load::SizedTexture::new(tex.id(), size));
            ui.add_space(4.0);
            ui.weak("Luma intensity by column. Bright areas show where pixel values concentrate vertically.");
        } else {
            ui.label("No waveform data available.");
        }
    }

    fn render_vectorscope(ui: &mut egui::Ui, vectorscope_data: &Option<Vec<[f64; 2]>>, reset_view: bool, zoom_in: bool, zoom_out: bool) {
        use egui_plot::{Line, Plot, PlotPoints, Points};

        // Apply button-driven zoom via plot memory
        if zoom_in || zoom_out {
            let plot_id = ui.id().with("vectorscope_plot");
            if let Some(mut mem) = egui_plot::PlotMemory::load(ui.ctx(), plot_id) {
                let factor = if zoom_in { 0.8_f32 } else { 1.25 };
                let mut tf = mem.transform();
                let center = tf.frame().center();
                tf.zoom(egui::vec2(factor, factor), center);
                mem.set_transform(tf);
                mem.store(ui.ctx(), plot_id);
            }
        }

        if let Some(ref points) = vectorscope_data {
            let plot_height = (ui.available_height() - 40.0).max(120.0);
            let mut plot = Plot::new("vectorscope_plot")
                .height(plot_height)
                .data_aspect(1.0)
                .allow_drag(true)
                .allow_zoom(true)
                .allow_scroll(true)
                .include_x(-128.0)
                .include_x(128.0)
                .include_y(-128.0)
                .include_y(128.0);
            if reset_view {
                plot = plot.reset();
            }

            let pts: Vec<[f64; 2]> = points.clone();

            plot.show(ui, |plot_ui| {
                // ±128 boundary box (dashed lines)
                let b = 128.0_f64;
                let dash_color = egui::Color32::from_rgb(100, 100, 100);
                let boundary_lines = [
                    // Top: (-128,128) to (128,128)
                    vec![[-b, b], [b, b]],
                    // Bottom: (-128,-128) to (128,-128)
                    vec![[-b, -b], [b, -b]],
                    // Left: (-128,-128) to (-128,128)
                    vec![[-b, -b], [-b, b]],
                    // Right: (128,-128) to (128,128)
                    vec![[b, -b], [b, b]],
                ];
                for line_pts in &boundary_lines {
                    let line = Line::new(PlotPoints::new(line_pts.to_vec()))
                        .color(dash_color)
                        .width(1.0)
                        .style(egui_plot::LineStyle::dashed_dense());
                    plot_ui.line(line);
                }
                // Cross axes through center
                let axis_color = egui::Color32::from_rgb(60, 60, 60);
                plot_ui.line(Line::new(PlotPoints::new(vec![[-128.0, 0.0], [128.0, 0.0]]))
                    .color(axis_color).width(0.5));
                plot_ui.line(Line::new(PlotPoints::new(vec![[0.0, -128.0], [0.0, 128.0]]))
                    .color(axis_color).width(0.5));

                // Data points
                let scatter = Points::new(pts)
                    .radius(1.0)
                    .color(egui::Color32::LIGHT_GREEN);
                plot_ui.points(scatter);
            });
            ui.add_space(4.0);
            ui.weak("Cb vs Cr (BT.709, centered). Dashed box = ±128 range.");
        } else {
            ui.label("No vectorscope data available.");
        }
    }

    fn render_metrics(ui: &mut egui::Ui, psnr: Option<f64>, ssim: Option<f64>, frame_diff: Option<f64>) {
        egui::Grid::new("metrics_grid")
            .spacing(egui::vec2(12.0, 4.0))
            .show(ui, |ui| {
                ui.label("PSNR:");
                if let Some(v) = psnr {
                    ui.monospace(format!("{:.2} dB", v));
                } else {
                    ui.monospace("--");
                }
                ui.end_row();

                ui.label("SSIM:");
                if let Some(v) = ssim {
                    ui.monospace(format!("{:.4}", v));
                } else {
                    ui.monospace("--");
                }
                ui.end_row();

                ui.label("Frame diff:");
                if let Some(v) = frame_diff {
                    ui.monospace(format!("{:.2}", v));
                } else {
                    ui.monospace("--");
                }
                ui.end_row();
            });
        ui.add_space(4.0);
        ui.weak("PSNR: signal-to-noise ratio (higher = more similar). SSIM: structural similarity (1.0 = identical).");
    }
}
