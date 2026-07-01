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
    /// Per-block luma statistics (mean & variance side-by-side).
    Block,
    /// Per-block inter-frame motion classification (vs previous frame).
    Motion,
    IspSideband,
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
    /// Per-block luma stats for the Block tab.
    pub block_stats: Option<crate::analysis::block_stats::BlockStats>,
    /// User-selected block size for the Block & Motion tabs. Default 32.
    pub block_size: u32,
    /// Which scalar metric the Block & Motion tabs aggregate (Y or MS).
    pub block_metric: crate::analysis::block_stats::BlockMetric,
    /// Per-block motion stats (current vs previous frame) for the Motion tab.
    pub motion_stats: Option<crate::analysis::motion::MotionStats>,
    /// Scoring method for the Motion tab (pixel diff vs avg/std comparison).
    pub motion_method: crate::analysis::motion::MotionMethod,
    /// Adjustable class thresholds for the Motion tab.
    pub motion_thresholds: crate::analysis::motion::MotionThresholds,
    /// Mirrors the main window's playback state. While true, the analysis
    /// viewport keeps requesting its own repaint so it refreshes every frame
    /// during playback (a parent-driven `request_repaint_of` alone can be
    /// starved/coalesced while the root window animates continuously).
    pub is_playing: bool,
    /// Source frame index the currently-stored analysis data was computed from.
    /// Shown in the window so the user can see it advance during playback.
    pub source_frame: usize,
    /// Show the value loupe (zoomed grid of cells with numeric values) when
    /// hovering the Block / Motion heatmaps.
    pub show_loupe: bool,
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
            block_stats: None,
            block_size: 32,
            block_metric: crate::analysis::block_stats::BlockMetric::Y,
            motion_stats: None,
            motion_method: crate::analysis::motion::MotionMethod::PixelDiff,
            motion_thresholds: crate::analysis::motion::MotionThresholds::default(),
            is_playing: false,
            source_frame: 0,
            show_loupe: false,
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
    /// Whether the Pixel Inspector section is rendered. Toggled from
    /// View → "Show Pixel Inspector". When false (and no other side-panel
    /// content is visible), the right SidePanel itself is collapsed.
    pub show_pixel_inspector: bool,

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
            // Default off: a thin clickable strip on the right edge offers a
            // one-click expand affordance, plus View → "Show Pixel Inspector"
            // for keyboard-driven users.
            show_pixel_inspector: false,
            analysis: Arc::new(Mutex::new(AnalysisShared::new())),
            grid_size: 0,
            sub_grid_size: 0,
        }
    }

    pub fn set_pixel_info(&mut self, info: Option<PixelInfo>) {
        self.pixel_info = info;
    }

    /// Render the sidebar contents (pixel inspector only).
    /// Returns true if anything visible was rendered, false otherwise — caller
    /// can use this to elide the SidePanel altogether when it would be empty.
    pub fn show(&mut self, ui: &mut egui::Ui) -> bool {
        if !self.show_pixel_inspector {
            return false;
        }
        ui.set_min_width(220.0);

        // ── Pixel Inspector ──────────────────────────────────────────
        // Header is a frameless button so the user can click the title (or
        // the trailing › chevron) to collapse the section back into the
        // edge strip. Same affordance as the strip in reverse.
        let header = ui.add(
            egui::Button::new(
                egui::RichText::new("Pixel Inspector  \u{203A}").heading(),
            )
            .frame(false),
        );
        if header
            .on_hover_text("Click to collapse")
            .clicked()
        {
            self.show_pixel_inspector = false;
            return false;
        }
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
        // Analysis toggle was previously here; moved to the Analysis menu
        // ("Show frame analysis") so the sidebar holds inspection content only.
        true
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

        // Immediate viewport (not deferred): it renders synchronously inside the
        // root's update pass, every frame the root paints. A deferred viewport is
        // a separate OS window whose redraw is starved while the root animates
        // continuously during playback (confirmed: analysis data was recomputed
        // every frame but the deferred window only repainted once playback
        // stopped). Immediate keeps it in lockstep with the main window.
        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("analysis_viewport"),
            egui::ViewportBuilder::default()
                .with_title("Analysis")
                .with_inner_size([460.0, 440.0])
                .with_min_inner_size([420.0, 360.0]),
            move |ctx, class| {
                // If the viewport is being closed by the OS, signal it.
                if matches!(
                    class,
                    egui::ViewportClass::Deferred | egui::ViewportClass::Immediate
                ) {
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
                    let (
                        active_tab,
                        histogram,
                        vectorscope,
                        waveform,
                        psnr,
                        ssim,
                        frame_diff,
                        block_stats,
                        mut block_size,
                        mut block_metric,
                        motion_stats,
                        mut motion_method,
                        mut motion_thresholds,
                        is_playing,
                        source_frame,
                        mut show_loupe,
                    ) = {
                        let data = shared.lock();
                        let wf = if data.active_tab == AnalysisTab::Waveform
                            && data.waveform_data_gen != data.waveform_texture_gen
                        {
                            data.waveform_image.clone()
                        } else {
                            None
                        };
                        let bs = if data.active_tab == AnalysisTab::Block {
                            data.block_stats.clone()
                        } else {
                            None
                        };
                        let ms = if data.active_tab == AnalysisTab::Motion {
                            data.motion_stats.clone()
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
                            bs,
                            data.block_size,
                            data.block_metric,
                            ms,
                            data.motion_method,
                            data.motion_thresholds,
                            data.is_playing,
                            data.source_frame,
                            data.show_loupe,
                        )
                    };
                    let prev_block_size = block_size;
                    let prev_block_metric = block_metric;
                    let prev_motion_method = motion_method;
                    let prev_motion_thresholds = motion_thresholds;
                    let prev_show_loupe = show_loupe;

                    // Tab bar + controls — writes back to shared state only on change.
                    let mut tab = active_tab;
                    let mut reset_view = false;
                    let mut zoom_in = false;
                    let mut zoom_out = false;
                    // Wrapped so the now-seven tabs plus zoom controls flow onto a
                    // second row instead of overflowing the ~460px-wide window.
                    ui.horizontal_wrapped(|ui| {
                        ui.selectable_value(&mut tab, AnalysisTab::Histogram, "Histogram");
                        ui.selectable_value(&mut tab, AnalysisTab::Waveform, "Waveform");
                        ui.selectable_value(&mut tab, AnalysisTab::Vectorscope, "Vectorscope");
                        ui.selectable_value(&mut tab, AnalysisTab::Metrics, "Metrics");
                        ui.selectable_value(&mut tab, AnalysisTab::Block, "Block");
                        ui.selectable_value(&mut tab, AnalysisTab::Motion, "Motion");
                        ui.selectable_value(&mut tab, AnalysisTab::IspSideband, "ISP Sideband");
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
                        ui.separator();
                        // Source frame this analysis was computed from. Advances
                        // live during playback — a quick way to confirm the panel
                        // is tracking the main window.
                        ui.weak(format!("frame {}", source_frame))
                            .on_hover_text("Frame index the current analysis was computed from");
                    });
                    if tab != active_tab {
                        let mut s = shared.lock();
                        s.active_tab = tab;
                        s.tab_changed = true;
                        drop(s);
                        // Wake the root viewport so its update() loop picks up
                        // tab_changed and runs update_analysis. Without this
                        // the change only takes effect on the next time the
                        // user interacts with the main window.
                        ctx.request_repaint_of(egui::ViewportId::ROOT);
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
                        AnalysisTab::Block => {
                            Self::render_block(
                                ui,
                                &block_stats,
                                &mut block_size,
                                &mut block_metric,
                                &mut show_loupe,
                            );
                            if show_loupe != prev_show_loupe {
                                shared.lock().show_loupe = show_loupe;
                            }
                            if block_size != prev_block_size
                                || block_metric != prev_block_metric
                            {
                                let mut s = shared.lock();
                                s.block_size = block_size;
                                s.block_metric = block_metric;
                                // Same path as a tab change: tells the main
                                // loop to recompute block_stats next frame.
                                s.tab_changed = true;
                                drop(s);
                                ctx.request_repaint_of(egui::ViewportId::ROOT);
                            }
                        }
                        AnalysisTab::Motion => {
                            // The Motion tab packs the most controls of any tab
                            // (tab bar, two control rows, three sliders, heatmap,
                            // legend, summary); a scroll area keeps it usable when
                            // the window is short.
                            egui::ScrollArea::vertical()
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    Self::render_motion(
                                        ui,
                                        &motion_stats,
                                        &mut block_size,
                                        &mut block_metric,
                                        &mut motion_method,
                                        &mut motion_thresholds,
                                        &mut show_loupe,
                                    );
                                });
                            if show_loupe != prev_show_loupe {
                                shared.lock().show_loupe = show_loupe;
                            }
                            if block_size != prev_block_size
                                || block_metric != prev_block_metric
                                || motion_method != prev_motion_method
                                || motion_thresholds != prev_motion_thresholds
                            {
                                let mut s = shared.lock();
                                s.block_size = block_size;
                                s.block_metric = block_metric;
                                s.motion_method = motion_method;
                                s.motion_thresholds = motion_thresholds;
                                // Same path as a tab change: tells the main loop
                                // to recompute motion_stats next frame.
                                s.tab_changed = true;
                                drop(s);
                                ctx.request_repaint_of(egui::ViewportId::ROOT);
                            }
                        }
                        AnalysisTab::IspSideband => {
                            ui.label("ISP Sideband analysis is shown in the right sidebar panel.");
                            ui.label("Load a sideband.bin file from the sidebar to visualize CTU heatmaps.");
                        }
                    }

                    // While the main window is playing, keep this viewport
                    // repainting on its own. The root drives playback and pushes
                    // fresh data via `request_repaint_of`, but a cross-viewport
                    // repaint request can be coalesced/starved while the root
                    // animates continuously — so without this the analysis window
                    // appears frozen during playback. Self-requesting a repaint
                    // makes it read the latest shared data every frame.
                    if is_playing {
                        ctx.request_repaint();
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

    /// Render the Block tab: side-by-side heatmaps of per-block luma mean
    /// and variance. `block_size` and `block_metric` are mutable so the user
    /// can change them — caller propagates changes back into `AnalysisShared`.
    fn render_block(
        ui: &mut egui::Ui,
        block_stats: &Option<crate::analysis::block_stats::BlockStats>,
        block_size: &mut u32,
        block_metric: &mut crate::analysis::block_stats::BlockMetric,
        show_loupe: &mut bool,
    ) {
        use crate::analysis::block_stats::BlockMetric;
        ui.horizontal(|ui| {
            ui.label("Metric:");
            ui.selectable_value(block_metric, BlockMetric::Y, "Y");
            ui.selectable_value(block_metric, BlockMetric::Ms, "MS");
            ui.separator();
            ui.label("Block size:");
            for &size in &[8u32, 16, 32, 64, 128] {
                ui.selectable_value(block_size, size, format!("{}×{}", size, size));
            }
            ui.separator();
            ui.checkbox(show_loupe, "🔍 Loupe")
                .on_hover_text("Hover a heatmap to zoom the surrounding cells and read exact values");
        });
        ui.separator();

        let Some(stats) = block_stats.as_ref() else {
            ui.centered_and_justified(|ui| {
                ui.label("Computing block statistics…");
            });
            return;
        };
        if stats.cols == 0 || stats.rows == 0 {
            ui.label("No image loaded.");
            return;
        }

        let max_var = stats.max_var().max(1.0);
        // Cell colour maps — defined once and shared by both the heatmap texture
        // and the value loupe (Copy closures capturing only `max_var`).
        let mean_color = |v: f32| {
            let g = v.clamp(0.0, 255.0) as u8;
            egui::Color32::from_rgb(g, g, g)
        };
        let var_color = move |v: f32| {
            // Normalize to [0,1] then map to a viridis-like ramp:
            // 0 → dark blue, 0.5 → green, 1 → yellow.
            let t = (v / max_var).clamp(0.0, 1.0);
            let r = (t * 255.0) as u8;
            let g = ((1.0 - (t - 0.5).abs() * 2.0).max(0.0) * 220.0) as u8;
            let b = ((1.0 - t) * 200.0) as u8;
            egui::Color32::from_rgb(r, g, b)
        };
        let mean_image =
            block_grid_to_color_image(stats.cols, stats.rows, &stats.means, mean_color);
        let var_image = block_grid_to_color_image(stats.cols, stats.rows, &stats.vars, var_color);

        let aspect = stats.width as f32 / stats.height.max(1) as f32;
        let avail = ui.available_size();
        let gap = 8.0_f32;
        let pane_w = ((avail.x - gap) * 0.5 - 4.0).max(120.0);
        let pane_h_raw = (pane_w / aspect).max(80.0);
        let pane_h = pane_h_raw.min(avail.y - 100.0);
        let pane_w = (pane_h * aspect).min(pane_w);
        let display_size = egui::vec2(pane_w, pane_h);

        let mean_handle = ui.ctx().load_texture(
            "block_mean",
            mean_image,
            egui::TextureOptions::NEAREST,
        );
        let var_handle = ui.ctx().load_texture(
            "block_var",
            var_image,
            egui::TextureOptions::NEAREST,
        );

        let var_summary = aggregate_block_stats(&stats.vars);

        let label_metric = stats.metric.label();
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.label(egui::RichText::new(format!("Mean ({})", label_metric)).strong());
                Self::draw_block_pane(
                    ui,
                    mean_handle.id(),
                    display_size,
                    stats.cols,
                    stats.rows,
                    &stats.means,
                    "block_mean_pane",
                    |v| format!("{:.0}", v),
                    |v| {
                        // Pick a contrasting text colour against greyscale fill.
                        if v.clamp(0.0, 255.0) > 140.0 {
                            egui::Color32::BLACK
                        } else {
                            egui::Color32::WHITE
                        }
                    },
                    *show_loupe,
                    mean_color,
                );
                // Whole-frame luma stats: min, max, average, variance.
                // (Not the average-of-block-means; computed pixel-wise.)
                ui.label(egui::RichText::new(format!(
                    "frame  min {:.1}  max {:.1}  avg {:.1}  var {:.1}",
                    stats.frame_min, stats.frame_max,
                    stats.frame_mean, stats.frame_var
                )).monospace().small());
            });
            ui.add_space(gap);
            ui.vertical(|ui| {
                ui.label(egui::RichText::new(format!("Variance ({})", label_metric)).strong());
                Self::draw_block_pane(
                    ui,
                    var_handle.id(),
                    display_size,
                    stats.cols,
                    stats.rows,
                    &stats.vars,
                    "block_var_pane",
                    |v| {
                        // Variance can grow large; truncate to fit a small cell.
                        if v >= 9999.5 {
                            format!("{:.0}k", v / 1000.0)
                        } else {
                            format!("{:.0}", v)
                        }
                    },
                    |v| {
                        // Heatmap goes blue→green→yellow; black text reads on
                        // the brighter (green/yellow) end.
                        let t = (v / max_var).clamp(0.0, 1.0);
                        if t > 0.45 {
                            egui::Color32::BLACK
                        } else {
                            egui::Color32::WHITE
                        }
                    },
                    *show_loupe,
                    var_color,
                );
                ui.label(egui::RichText::new(format!(
                    "min {:.1}  max {:.1}  avg {:.1}",
                    var_summary.min, var_summary.max, var_summary.avg
                )).monospace().small());
            });
        });
        ui.add_space(4.0);
        ui.weak(format!(
            "{} × {} blocks of {}px each. Scroll to zoom at the cursor · drag to pan · double-click to reset. Hover a block for its value; enable 🔍 Loupe to read exact values on fine grids. Variance heatmap range: 0 → {:.0}.",
            stats.cols, stats.rows, stats.block_size, max_var,
        ));
    }

    /// Heatmap fill colour for a motion class. Uses blue→amber→red rather than
    /// green→amber→red so the ordering survives red-green colour-blindness
    /// (the most common form): blue is unambiguous against amber/red. The
    /// legend's text labels and per-block score numbers provide a further
    /// non-colour cue.
    fn motion_class_color(c: crate::analysis::motion::MotionClass) -> egui::Color32 {
        use crate::analysis::motion::MotionClass;
        match c {
            MotionClass::None => egui::Color32::from_rgb(38, 42, 52),
            MotionClass::Slight => egui::Color32::from_rgb(60, 120, 210),
            MotionClass::Much => egui::Color32::from_rgb(235, 160, 45),
            MotionClass::Full => egui::Color32::from_rgb(214, 56, 56),
        }
    }

    /// Text colour that reads on top of [`motion_class_color`] for the given
    /// class: black on the bright amber `Much` fill, white on the rest.
    fn motion_text_color(c: crate::analysis::motion::MotionClass) -> egui::Color32 {
        use crate::analysis::motion::MotionClass;
        match c {
            MotionClass::Much => egui::Color32::BLACK,
            _ => egui::Color32::WHITE,
        }
    }

    /// Render the Motion tab: a per-block motion-class heatmap comparing the
    /// current frame against the previous one, with selectable scoring method
    /// (pixel diff vs avg/std) and adjustable class thresholds.
    fn render_motion(
        ui: &mut egui::Ui,
        motion_stats: &Option<crate::analysis::motion::MotionStats>,
        block_size: &mut u32,
        block_metric: &mut crate::analysis::block_stats::BlockMetric,
        motion_method: &mut crate::analysis::motion::MotionMethod,
        thresholds: &mut crate::analysis::motion::MotionThresholds,
        show_loupe: &mut bool,
    ) {
        use crate::analysis::block_stats::BlockMetric;
        use crate::analysis::motion::{MotionClass, MotionMethod, MotionThresholds};

        ui.horizontal(|ui| {
            // Metric/Block size are shared with the Block tab; keep the labels
            // and "N×N" rendering identical so the shared setting is obvious.
            ui.label("Metric:");
            ui.selectable_value(block_metric, BlockMetric::Y, BlockMetric::Y.label());
            ui.selectable_value(block_metric, BlockMetric::Ms, BlockMetric::Ms.label());
            ui.separator();
            ui.label("Block size:");
            for &size in &[8u32, 16, 32, 64, 128] {
                ui.selectable_value(block_size, size, format!("{}×{}", size, size));
            }
            ui.separator();
            ui.checkbox(show_loupe, "🔍 Loupe")
                .on_hover_text("Hover the map to zoom the surrounding cells and read exact scores");
        });
        ui.horizontal(|ui| {
            ui.label("Method:");
            ui.selectable_value(motion_method, MotionMethod::PixelDiff, MotionMethod::PixelDiff.label())
                .on_hover_text(
                    "Mean absolute per-pixel difference (MAD) inside each block. \
                     Flags any pixel movement, even when the block's average and \
                     spread are unchanged.",
                );
            ui.selectable_value(motion_method, MotionMethod::StatsDiff, MotionMethod::StatsDiff.label())
                .on_hover_text(
                    "Compares each block's mean and std-dev between frames: \
                     |Δmean| + |Δstd|. Blind to spatial rearrangement that keeps \
                     the same average and spread.",
                );
            ui.separator();
            // Named "Reset thresholds" (not just "Reset") to avoid confusion
            // with the tab-bar's view "Reset" button.
            if ui.button("Reset thresholds").on_hover_text("Restore default thresholds").clicked() {
                *thresholds = MotionThresholds::default();
            }
        });
        // Ranges span both methods: PixelDiff (MAD) is bounded 0..255, while
        // StatsDiff (|Δmean| + |Δstd|) can reach ~382, so `full` goes to 400.
        ui.add(
            egui::Slider::new(&mut thresholds.slight, 0.0..=50.0)
                .text("slight ≥")
                .fixed_decimals(1),
        );
        ui.add(
            egui::Slider::new(&mut thresholds.much, 0.0..=200.0)
                .text("much ≥")
                .fixed_decimals(1),
        );
        ui.add(
            egui::Slider::new(&mut thresholds.full, 0.0..=400.0)
                .text("full ≥")
                .fixed_decimals(1),
        );
        // Keep the bands ordered so the legend stays meaningful.
        thresholds.much = thresholds.much.max(thresholds.slight);
        thresholds.full = thresholds.full.max(thresholds.much);

        ui.separator();

        let Some(stats) = motion_stats.as_ref() else {
            ui.centered_and_justified(|ui| {
                ui.label("Computing motion…");
            });
            return;
        };
        if stats.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label(
                    "Motion compares each frame with the previous one.\n\
                     It populates during playback or when you step forward (→).\n\
                     Scrubbing the slider or stepping back (←) clears it — \
                     step forward again to refresh.",
                );
            });
            return;
        }

        // Build the class-coloured heatmap (cols × rows, nearest-sampled).
        let w = stats.cols as usize;
        let h = stats.rows as usize;
        let pixels: Vec<egui::Color32> = stats
            .classes
            .iter()
            .map(|&c| Self::motion_class_color(c))
            .collect();
        let motion_image = egui::ColorImage {
            size: [w, h],
            pixels,
        };

        let aspect = stats.width as f32 / stats.height.max(1) as f32;
        let avail = ui.available_size();
        let pane_w_cap = (avail.x - 8.0).max(160.0);
        let pane_h_raw = (pane_w_cap / aspect).max(120.0);
        let pane_h = pane_h_raw.min((avail.y - 160.0).max(120.0));
        let pane_w = (pane_h * aspect).min(pane_w_cap);
        let display_size = egui::vec2(pane_w, pane_h);

        let handle = ui.ctx().load_texture(
            "motion_map",
            motion_image,
            egui::TextureOptions::NEAREST,
        );

        // Describe the heatmap with the thresholds the classes were ACTUALLY
        // computed with (stats.thresholds), not the live slider values — those
        // only take effect after the next recompute, so using them here would
        // momentarily disagree with the colours on screen.
        let t = stats.thresholds;
        Self::draw_block_pane(
            ui,
            handle.id(),
            display_size,
            stats.cols,
            stats.rows,
            &stats.scores,
            "motion_pane",
            |v| format!("{:.0}", v),
            move |v| Self::motion_text_color(MotionClass::from_score(v, t)),
            *show_loupe,
            move |v| Self::motion_class_color(MotionClass::from_score(v, t)),
        );

        ui.add_space(6.0);
        let total = (stats.cols * stats.rows).max(1);
        // Band range string per class, derived from the displayed thresholds so
        // colour, name, range, and count all read on a single legend row.
        let band_for = |c: MotionClass| -> String {
            match c {
                MotionClass::None => format!("< {:.0}", t.slight),
                MotionClass::Slight => format!("{:.0} – {:.0}", t.slight, t.much),
                MotionClass::Much => format!("{:.0} – {:.0}", t.much, t.full),
                MotionClass::Full => format!("≥ {:.0}", t.full),
            }
        };
        egui::Grid::new("motion_legend")
            .spacing(egui::vec2(10.0, 3.0))
            .show(ui, |ui| {
                for c in MotionClass::all() {
                    let count = stats.class_counts[c.index()];
                    let pct = 100.0 * count as f32 / total as f32;
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 2.0, Self::motion_class_color(c));
                    ui.label(c.label());
                    ui.monospace(format!("{:>9}", band_for(c)));
                    ui.monospace(format!("{:>5}  {:>5.1}%", count, pct));
                    ui.end_row();
                }
            });

        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(format!(
                "frame motion  avg {:.1}   peak {:.1}   [{}]",
                stats.frame_score,
                stats.max_score,
                stats.method.label()
            ))
            .monospace()
            .small(),
        );
        ui.add_space(2.0);
        ui.weak(format!(
            "{} × {} blocks of {}px vs previous frame. Colour = motion class; numbers show each block's score. Scroll to zoom · drag to pan · double-click to reset · 🔍 Loupe for exact values.",
            stats.cols, stats.rows, stats.block_size,
        ));
    }

    /// Render one of the two block heatmap panes: the texture, an optional
    /// per-block numeric overlay (when each block is wide enough), and a
    /// hover tooltip showing the value at the cursor.
    #[allow(clippy::too_many_arguments)]
    fn draw_block_pane<L, C, K>(
        ui: &mut egui::Ui,
        texture_id: egui::TextureId,
        display_size: egui::Vec2,
        cols: u32,
        rows: u32,
        values: &[f32],
        id_salt: &str,
        label_for: L,
        text_color_for: C,
        show_loupe: bool,
        cell_color: K,
    ) where
        L: Fn(f32) -> String,
        C: Fn(f32) -> egui::Color32,
        K: Fn(f32) -> egui::Color32,
    {
        let (rect, response) =
            ui.allocate_exact_size(display_size, egui::Sense::click_and_drag());

        // Per-pane zoom/pan, persisted in egui memory keyed by the pane id so
        // each heatmap (mean / variance / motion) zooms independently.
        let view_id = ui.id().with(("pane_view", id_salt));
        let mut view = ui
            .ctx()
            .data_mut(|d| d.get_temp::<PaneView>(view_id))
            .unwrap_or_default();

        // Double-click resets; drag pans; wheel zooms anchored at the cursor.
        if response.double_clicked() {
            view = PaneView::default();
        }
        if response.dragged() {
            view.pan += response.drag_delta();
        }
        if response.hovered() {
            // Consume the vertical scroll so an enclosing ScrollArea (Motion tab)
            // doesn't also scroll while the wheel is zooming the heatmap.
            let scroll = ui.input_mut(|i| {
                let d = i.smooth_scroll_delta.y;
                i.smooth_scroll_delta = egui::Vec2::ZERO;
                d
            });
            if scroll != 0.0 {
                if let Some(cursor) = response.hover_pos() {
                    let old = view.zoom;
                    let factor = 1.0 + (scroll * 0.002).clamp(-0.25, 0.25);
                    let new = (old * factor).clamp(1.0, 24.0);
                    if (new - old).abs() > f32::EPSILON {
                        view.pan = Self::zoom_anchor_pan(view.pan, old, new, cursor - rect.min);
                        view.zoom = new;
                    }
                }
            }
        }

        // Clamp zoom, and pan so the zoomed image always covers the pane.
        view.zoom = view.zoom.clamp(1.0, 24.0);
        let img_w = rect.width() * view.zoom;
        let img_h = rect.height() * view.zoom;
        view.pan.x = view.pan.x.clamp(rect.width() - img_w, 0.0);
        view.pan.y = view.pan.y.clamp(rect.height() - img_h, 0.0);
        ui.ctx().data_mut(|d| d.insert_temp(view_id, view));

        let image_rect = egui::Rect::from_min_size(rect.min + view.pan, egui::vec2(img_w, img_h));

        let painter = ui.painter_at(rect);
        // Solid background so out-of-image area shows the panel's neutral colour.
        painter.rect_filled(rect, 0.0, ui.visuals().extreme_bg_color);
        painter.image(
            texture_id,
            image_rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );

        if cols == 0 || rows == 0 {
            return;
        }
        let cell_w = image_rect.width() / cols as f32;
        let cell_h = image_rect.height() / rows as f32;
        // Numeric overlay for the visible cells, when each is big enough to read.
        let overlay_threshold = 22.0_f32;
        if cell_w >= overlay_threshold && cell_h >= overlay_threshold {
            let font_size = (cell_h.min(cell_w) * 0.45).clamp(8.0, 16.0);
            let font = egui::FontId::monospace(font_size);
            // Only iterate cells intersecting the visible pane rect (matters when
            // zoomed in: the full grid may be far larger than the viewport).
            let c0 = ((rect.min.x - image_rect.min.x) / cell_w).floor().clamp(0.0, cols as f32) as u32;
            let c1 = ((rect.max.x - image_rect.min.x) / cell_w).ceil().clamp(0.0, cols as f32) as u32;
            let r0 = ((rect.min.y - image_rect.min.y) / cell_h).floor().clamp(0.0, rows as f32) as u32;
            let r1 = ((rect.max.y - image_rect.min.y) / cell_h).ceil().clamp(0.0, rows as f32) as u32;
            for r in r0..r1 {
                for c in c0..c1 {
                    let idx = (r * cols + c) as usize;
                    let v = values.get(idx).copied().unwrap_or(0.0);
                    let cx = image_rect.min.x + (c as f32 + 0.5) * cell_w;
                    let cy = image_rect.min.y + (r as f32 + 0.5) * cell_h;
                    painter.text(
                        egui::pos2(cx, cy),
                        egui::Align2::CENTER_CENTER,
                        label_for(v),
                        font.clone(),
                        text_color_for(v),
                    );
                }
            }
        }

        // Hover: value loupe or single-cell tooltip. Map cursor → cell through
        // the (possibly zoomed/panned) image rect.
        if let Some(pos) = response.hover_pos() {
            let rx = ((pos.x - image_rect.min.x) / cell_w).floor() as i32;
            let ry = ((pos.y - image_rect.min.y) / cell_h).floor() as i32;
            if rx >= 0 && ry >= 0 && (rx as u32) < cols && (ry as u32) < rows {
                if show_loupe {
                    Self::draw_value_loupe(
                        ui, id_salt, pos, cols, rows, values, rx, ry,
                        &label_for, &text_color_for, &cell_color,
                    );
                } else {
                    let idx = (ry as u32 * cols + rx as u32) as usize;
                    if let Some(&v) = values.get(idx) {
                        response.on_hover_text(format!(
                            "block ({}, {})  value: {}",
                            rx, ry,
                            label_for(v),
                        ));
                    }
                }
            }
        }
    }

    /// New pan offset that keeps the point at `anchor` (relative to the pane's
    /// top-left) fixed while the zoom changes `old_zoom → new_zoom`. Derived so
    /// that `(anchor − pan) / zoom` — the image coordinate under the anchor — is
    /// invariant across the zoom change.
    fn zoom_anchor_pan(
        pan: egui::Vec2,
        old_zoom: f32,
        new_zoom: f32,
        anchor: egui::Vec2,
    ) -> egui::Vec2 {
        let ratio = new_zoom / old_zoom;
        anchor * (1.0 - ratio) + pan * ratio
    }

    /// Look up a value at grid `(col, row)`, returning `None` when out of range.
    fn loupe_value_at(values: &[f32], cols: u32, rows: u32, col: i32, row: i32) -> Option<f32> {
        if col < 0 || row < 0 || col as u32 >= cols || row as u32 >= rows {
            return None;
        }
        values.get((row as u32 * cols + col as u32) as usize).copied()
    }

    /// Floating value loupe: a zoomed `(2K+1)×(2K+1)` grid of the cells around
    /// `(cx, cy)`, each drawn with its heatmap colour and numeric value, so the
    /// user can read exact values even where the in-pane overlay is too small.
    #[allow(clippy::too_many_arguments)]
    fn draw_value_loupe<L, C, K>(
        ui: &egui::Ui,
        id_salt: &str,
        cursor: egui::Pos2,
        cols: u32,
        rows: u32,
        values: &[f32],
        center_col: i32,
        center_row: i32,
        label_for: &L,
        text_color_for: &C,
        cell_color: &K,
    ) where
        L: Fn(f32) -> String,
        C: Fn(f32) -> egui::Color32,
        K: Fn(f32) -> egui::Color32,
    {
        const HALF: i32 = 2; // 5×5 window
        const CELL: f32 = 30.0;
        let span = (2 * HALF + 1) as f32;
        let size = span * CELL;

        // Place to the upper-right of the cursor, clamped to the window.
        let screen = ui.ctx().screen_rect();
        let mut origin = egui::pos2(cursor.x + 24.0, cursor.y - size - 12.0);
        origin.x = origin.x.clamp(screen.left() + 4.0, screen.right() - size - 4.0);
        origin.y = origin.y.clamp(screen.top() + 4.0, screen.bottom() - size - 4.0);
        let loupe_rect = egui::Rect::from_min_size(origin, egui::vec2(size, size));

        // Draw on a foreground layer so it overlaps neighbouring panes.
        let painter = ui.ctx().layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            ui.id().with(("value_loupe", id_salt)),
        ));
        painter.rect_filled(loupe_rect.expand(2.0), 3.0, egui::Color32::from_black_alpha(230));
        painter.rect_stroke(
            loupe_rect.expand(2.0),
            3.0,
            egui::Stroke::new(1.0, egui::Color32::GRAY),
            egui::StrokeKind::Outside,
        );

        let font = egui::FontId::monospace((CELL * 0.34).clamp(8.0, 12.0));
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
                match Self::loupe_value_at(values, cols, rows, col, row) {
                    Some(v) => {
                        painter.rect_filled(cell_rect.shrink(0.5), 0.0, cell_color(v));
                        painter.text(
                            cell_rect.center(),
                            egui::Align2::CENTER_CENTER,
                            label_for(v),
                            font.clone(),
                            text_color_for(v),
                        );
                    }
                    None => {
                        painter.rect_filled(cell_rect.shrink(0.5), 0.0, empty_fill);
                    }
                }
            }
        }

        // Highlight the centre (hovered) cell.
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

        // Caption: centre cell coordinates + exact value.
        if let Some(v) = Self::loupe_value_at(values, cols, rows, center_col, center_row) {
            painter.text(
                egui::pos2(loupe_rect.left(), loupe_rect.bottom() + 3.0),
                egui::Align2::LEFT_TOP,
                format!("({}, {}) = {}", center_col, center_row, label_for(v)),
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgb(230, 230, 230),
            );
        }
    }
}

/// Per-pane zoom / pan state for a heatmap pane, stored in egui temp memory.
/// `zoom` is ≥ 1.0 (1.0 = fit); `pan` is a screen-space offset of the image's
/// top-left relative to the pane's top-left (≤ 0 on each axis when zoomed).
#[derive(Clone, Copy)]
struct PaneView {
    zoom: f32,
    pan: egui::Vec2,
}

impl Default for PaneView {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
        }
    }
}

/// Min / max / mean summary for a block-statistic vector.
struct BlockSummary {
    min: f32,
    max: f32,
    avg: f32,
}

fn aggregate_block_stats(values: &[f32]) -> BlockSummary {
    if values.is_empty() {
        return BlockSummary { min: 0.0, max: 0.0, avg: 0.0 };
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;
    for &v in values {
        if v < min { min = v; }
        if v > max { max = v; }
        sum += v as f64;
    }
    BlockSummary {
        min,
        max,
        avg: (sum / values.len() as f64) as f32,
    }
}

/// Build a `ColorImage` of size (cols, rows) from a row-major `values`
/// slice, using `map` to colourise each value. Returns a 1×1 transparent
/// image when the inputs are empty.
fn block_grid_to_color_image<F>(
    cols: u32,
    rows: u32,
    values: &[f32],
    map: F,
) -> egui::ColorImage
where
    F: Fn(f32) -> egui::Color32,
{
    let w = cols.max(1) as usize;
    let h = rows.max(1) as usize;
    let mut pixels = Vec::with_capacity(w * h);
    for i in 0..(w * h) {
        let v = values.get(i).copied().unwrap_or(0.0);
        pixels.push(map(v));
    }
    egui::ColorImage {
        size: [w, h],
        pixels,
    }
}

#[cfg(test)]
mod loupe_tests {
    use super::Sidebar;

    #[test]
    fn loupe_value_at_in_range_and_out_of_range() {
        // 3×2 grid, row-major: row0 = [0,1,2], row1 = [3,4,5]
        let v = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let (cols, rows) = (3u32, 2u32);
        // In range
        assert_eq!(Sidebar::loupe_value_at(&v, cols, rows, 0, 0), Some(0.0));
        assert_eq!(Sidebar::loupe_value_at(&v, cols, rows, 2, 0), Some(2.0));
        assert_eq!(Sidebar::loupe_value_at(&v, cols, rows, 1, 1), Some(4.0));
        assert_eq!(Sidebar::loupe_value_at(&v, cols, rows, 2, 1), Some(5.0));
        // Out of range → None (negatives and past the edges)
        assert_eq!(Sidebar::loupe_value_at(&v, cols, rows, -1, 0), None);
        assert_eq!(Sidebar::loupe_value_at(&v, cols, rows, 0, -1), None);
        assert_eq!(Sidebar::loupe_value_at(&v, cols, rows, 3, 0), None);
        assert_eq!(Sidebar::loupe_value_at(&v, cols, rows, 0, 2), None);
    }

    #[test]
    fn zoom_anchor_pan_keeps_the_anchor_point_fixed() {
        // The image coordinate under the anchor — (anchor − pan) / zoom — must be
        // unchanged after zooming, so the cursor stays over the same cell.
        let pan0 = egui::Vec2::new(-5.0, 3.0);
        let (z0, z1) = (1.5_f32, 4.0_f32);
        let anchor = egui::Vec2::new(20.0, 10.0);
        let pan1 = Sidebar::zoom_anchor_pan(pan0, z0, z1, anchor);
        let uv0 = (anchor - pan0) / z0;
        let uv1 = (anchor - pan1) / z1;
        assert!((uv0.x - uv1.x).abs() < 1e-4, "x {} vs {}", uv0.x, uv1.x);
        assert!((uv0.y - uv1.y).abs() < 1e-4, "y {} vs {}", uv0.y, uv1.y);
    }
}
