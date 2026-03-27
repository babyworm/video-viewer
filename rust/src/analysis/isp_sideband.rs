use eframe::egui;
use crate::core::sideband::{SidebandFile, SidebandFrame, SidebandOverlayMode};

pub struct SidebandPanel;

impl SidebandPanel {
    pub fn new() -> Self {
        Self
    }

    /// Show the sideband control panel.
    /// Returns Some(action) if the user wants to load/unload a file.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        sideband: Option<&SidebandFile>,
        sideband_path: Option<&str>,
        current_frame: Option<&SidebandFrame>,
        mode: &mut SidebandOverlayMode,
        opacity: &mut f32,
        show_values: &mut bool,
        current_frame_idx: usize,
    ) -> Option<SidebandAction> {
        // File section
        ui.heading("ISP Sideband");
        ui.separator();

        if sideband.is_some() {
            if let Some(path) = sideband_path {
                ui.label(format!("File: {}", path.rsplit('/').next().unwrap_or(path)));
            }
            if ui.button("Unload").clicked() {
                return Some(SidebandAction::Unload);
            }
        } else {
            if ui.button("Load sideband.bin...").clicked() {
                return Some(SidebandAction::LoadRequested);
            }
            ui.label("No sideband file loaded");
            return None;
        }

        ui.separator();

        // Overlay mode selector
        ui.label("Overlay Mode:");
        egui::ComboBox::from_id_salt("sideband_mode")
            .selected_text(format!("{:?}", mode))
            .show_ui(ui, |ui| {
                ui.selectable_value(mode, SidebandOverlayMode::None, "None");
                ui.selectable_value(mode, SidebandOverlayMode::QpDelta, "QP Delta");
                ui.selectable_value(mode, SidebandOverlayMode::Activity, "Activity");
                ui.selectable_value(mode, SidebandOverlayMode::Flatness, "Flatness");
                ui.selectable_value(mode, SidebandOverlayMode::Saliency, "Saliency");
                ui.selectable_value(mode, SidebandOverlayMode::EdgeDensity, "Edge Density");
                ui.selectable_value(mode, SidebandOverlayMode::Noise, "Noise");
                ui.selectable_value(mode, SidebandOverlayMode::Confidence, "Confidence");
                ui.selectable_value(mode, SidebandOverlayMode::TemporalStability, "Temporal Stability");
            });

        // Opacity slider
        ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));

        // Show values toggle
        ui.checkbox(show_values, "Show values (zoom >= 2x)");

        ui.separator();

        // Frame info
        if let Some(frame) = current_frame {
            ui.heading(format!("Frame {} (id={})", current_frame_idx, frame.frame_id));

            egui::Grid::new("sideband_frame_info").show(ui, |ui| {
                ui.label("Scene class:");
                ui.label(format!("{}", frame.scene_class));
                ui.end_row();
                ui.label("Noise class:");
                ui.label(format!("{}", frame.noise_class));
                ui.end_row();
                ui.label("Motion class:");
                ui.label(format!("{}", frame.motion_class));
                ui.end_row();
                ui.label("Denoise:");
                ui.label(format!("{}", frame.denoise_strength));
                ui.end_row();
                ui.label("Sharpen:");
                ui.label(format!("{}", frame.sharpen_strength));
                ui.end_row();
                ui.label("Confidence:");
                ui.label(format!("{}", frame.global_confidence));
                ui.end_row();
                ui.label("QP bias:");
                ui.label(format!("{}", frame.frame_qp_bias));
                ui.end_row();
                ui.label("Lambda:");
                ui.label(format!("{:.3}", frame.lambda_scale()));
                ui.end_row();
            });

            ui.separator();

            // CTU statistics summary
            if !frame.ctus.is_empty() {
                let n = frame.ctus.len();
                let mean_qp: f64 =
                    frame.ctus.iter().map(|c| c.qp_delta as f64).sum::<f64>() / n as f64;
                let min_qp = frame.ctus.iter().map(|c| c.qp_delta).min().unwrap_or(0);
                let max_qp = frame.ctus.iter().map(|c| c.qp_delta).max().unwrap_or(0);
                let mean_act: f64 =
                    frame.ctus.iter().map(|c| c.activity as f64).sum::<f64>() / n as f64;

                ui.label(format!("{} CTUs", n));
                egui::Grid::new("sideband_ctu_stats").show(ui, |ui| {
                    ui.label("QP delta:");
                    ui.label(format!("mean={:.1} [{}, +{}]", mean_qp, min_qp, max_qp));
                    ui.end_row();
                    ui.label("Activity:");
                    ui.label(format!("mean={:.1}", mean_act));
                    ui.end_row();
                });
            }
        } else if sideband.is_some() {
            ui.label(format!("Frame {} -- no sideband data", current_frame_idx));
        }

        None
    }
}

pub enum SidebandAction {
    LoadRequested,
    Unload,
}
