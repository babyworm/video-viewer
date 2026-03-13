use eframe::egui;

use crate::core::pixel::PixelInfo;

/// Which analysis tab is active in the sidebar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisTab {
    Histogram,
    Waveform,
    Vectorscope,
    Metrics,
}

/// Sidebar panel showing pixel inspector and analysis toggle.
pub struct Sidebar {
    pub pixel_info: Option<PixelInfo>,
    pub active_tab: AnalysisTab,
    pub show_analysis: bool,

    // Optional analysis data (populated externally).
    /// Per-channel histograms: key = channel name, value = 256 bin counts.
    pub histogram_data: Option<std::collections::HashMap<String, Vec<f64>>>,
    /// Vectorscope scatter points (U, V) in display range.
    pub vectorscope_data: Option<Vec<[f64; 2]>>,
    /// Waveform as an RGB texture.
    pub waveform_texture: Option<egui::TextureHandle>,
    /// Scalar metrics.
    pub psnr: Option<f64>,
    pub ssim: Option<f64>,
    pub frame_diff: Option<f64>,
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
            active_tab: AnalysisTab::Histogram,
            show_analysis: false,
            histogram_data: None,
            vectorscope_data: None,
            waveform_texture: None,
            psnr: None,
            ssim: None,
            frame_diff: None,
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
                // Position
                ui.monospace(format!("X: {}  Y: {}", info.x, info.y));

                // Raw hex
                ui.monospace(format!("Raw: {}", info.raw_hex));

                // Components
                ui.add_space(4.0);
                ui.label("Components:");
                let mut keys: Vec<&String> = info.components.keys().collect();
                keys.sort();
                let comp_str: String = keys
                    .iter()
                    .map(|k| format!("{}: {}", k, info.components[*k]))
                    .collect::<Vec<_>>()
                    .join("  ");
                ui.monospace(comp_str);

                // Neighbourhood grid
                ui.add_space(8.0);
                ui.label("Neighborhood:");
                let grid_id = ui.id().with("pixel_neighborhood");
                egui::Grid::new(grid_id)
                    .spacing(egui::vec2(4.0, 2.0))
                    .show(ui, |ui| {
                        for row in &info.neighborhood {
                            for cell in row {
                                ui.monospace(cell);
                            }
                            ui.end_row();
                        }
                    });
            } else {
                ui.label("Hover over the image to inspect pixels.");
            }
        });

        ui.add_space(8.0);
        ui.separator();

        // ── Analysis toggle ──────────────────────────────────────────
        ui.checkbox(&mut self.show_analysis, "Show Analysis (separate window)");
    }

    /// Render the analysis in a separate OS-level viewport window.
    /// Call this from the app's update() after other panels.
    pub fn show_analysis_window(&mut self, ctx: &egui::Context) {
        if !self.show_analysis {
            return;
        }

        let mut active = self.active_tab;

        // Collect references to data before the closure to avoid borrowing self.
        let histogram_data = &self.histogram_data;
        let vectorscope_data = &self.vectorscope_data;
        let waveform_texture = &self.waveform_texture;
        let psnr = self.psnr;
        let ssim = self.ssim;
        let frame_diff = self.frame_diff;

        let mut close_requested = false;

        let vp_id = egui::ViewportId::from_hash_of("analysis_viewport");
        ctx.show_viewport_immediate(
            vp_id,
            egui::ViewportBuilder::default()
                .with_title("Analysis")
                .with_inner_size([420.0, 380.0]),
            |ctx, _class| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    // Tab bar
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut active, AnalysisTab::Histogram, "Histogram");
                        ui.selectable_value(&mut active, AnalysisTab::Waveform, "Waveform");
                        ui.selectable_value(&mut active, AnalysisTab::Vectorscope, "Vectorscope");
                        ui.selectable_value(&mut active, AnalysisTab::Metrics, "Metrics");
                    });

                    ui.separator();

                    match active {
                        AnalysisTab::Histogram => {
                            Self::render_histogram(ui, histogram_data);
                        }
                        AnalysisTab::Waveform => {
                            Self::render_waveform(ui, waveform_texture);
                        }
                        AnalysisTab::Vectorscope => {
                            Self::render_vectorscope(ui, vectorscope_data);
                        }
                        AnalysisTab::Metrics => {
                            Self::render_metrics(ui, psnr, ssim, frame_diff);
                        }
                    }
                });

                if ctx.input(|i| i.viewport().close_requested()) {
                    close_requested = true;
                }
            },
        );

        self.active_tab = active;
        if close_requested {
            self.show_analysis = false;
        }
    }

    // ── Tab implementations (static to avoid borrow conflicts) ─────

    fn render_histogram(
        ui: &mut egui::Ui,
        histogram_data: &Option<std::collections::HashMap<String, Vec<f64>>>,
    ) {
        use egui_plot::{Bar, BarChart, Plot};

        if let Some(ref hist) = histogram_data {
            let plot = Plot::new("histogram_plot")
                .height(200.0)
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false)
                .show_axes([true, false]);

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
        } else {
            ui.label("No histogram data available.");
        }
    }

    fn render_waveform(ui: &mut egui::Ui, waveform_texture: &Option<egui::TextureHandle>) {
        if let Some(ref tex) = waveform_texture {
            let size = egui::vec2(ui.available_width(), 200.0);
            ui.image(egui::load::SizedTexture::new(tex.id(), size));
        } else {
            ui.label("No waveform data available.");
        }
    }

    fn render_vectorscope(ui: &mut egui::Ui, vectorscope_data: &Option<Vec<[f64; 2]>>) {
        use egui_plot::{Plot, Points};

        if let Some(ref points) = vectorscope_data {
            let plot = Plot::new("vectorscope_plot")
                .height(200.0)
                .data_aspect(1.0)
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false);

            let pts: Vec<[f64; 2]> = points.clone();

            plot.show(ui, |plot_ui| {
                let scatter = Points::new(pts)
                    .radius(1.0)
                    .color(egui::Color32::LIGHT_GREEN);
                plot_ui.points(scatter);
            });
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
    }
}
