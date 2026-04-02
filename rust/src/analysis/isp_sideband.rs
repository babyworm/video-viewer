use eframe::egui;
use crate::core::sideband::{SidebandFile, SidebandFrame, SidebandOverlayMode};

pub struct SidebandPanel;

impl Default for SidebandPanel {
    fn default() -> Self {
        Self
    }
}

/// Context for the sideband panel UI.
pub struct SidebandPanelContext<'a> {
    pub sideband: Option<&'a SidebandFile>,
    pub sideband_path: Option<&'a str>,
    pub current_frame: Option<&'a SidebandFrame>,
    pub mode: &'a mut SidebandOverlayMode,
    pub opacity: &'a mut f32,
    pub show_values: &'a mut bool,
    pub current_frame_idx: usize,
}

impl SidebandPanel {
    pub fn new() -> Self {
        Self
    }

    /// Show the sideband control panel.
    /// Returns Some(action) if the user wants to load/unload a file.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &mut SidebandPanelContext,
    ) -> Option<SidebandAction> {
        // File section
        ui.heading("ISP Sideband");
        ui.separator();

        if ctx.sideband.is_some() {
            if let Some(path) = ctx.sideband_path {
                ui.label(format!("File: {}", path.rsplit('/').next().unwrap_or(path)));
            }
            if ui.button("Unload").clicked() {
                return Some(SidebandAction::Unload);
            }
        } else {
            if ui.button("Load sideband file...").clicked() {
                return Some(SidebandAction::LoadRequested);
            }
            ui.label("No sideband file loaded");
            return None;
        }

        ui.separator();

        // Overlay mode selector
        ui.label("Overlay Mode:");
        egui::ComboBox::from_id_salt("sideband_mode")
            .selected_text(format!("{:?}", ctx.mode))
            .show_ui(ui, |ui| {
                ui.selectable_value(ctx.mode, SidebandOverlayMode::None, "None");
                ui.selectable_value(ctx.mode, SidebandOverlayMode::QpDelta, "QP Delta");
                ui.selectable_value(ctx.mode, SidebandOverlayMode::Activity, "Activity");
                ui.selectable_value(ctx.mode, SidebandOverlayMode::Flatness, "Flatness");
                ui.selectable_value(ctx.mode, SidebandOverlayMode::Saliency, "Saliency");
                ui.selectable_value(ctx.mode, SidebandOverlayMode::EdgeDensity, "Edge Density");
                ui.selectable_value(ctx.mode, SidebandOverlayMode::Noise, "Noise");
                ui.selectable_value(ctx.mode, SidebandOverlayMode::Confidence, "Confidence");
                ui.selectable_value(ctx.mode, SidebandOverlayMode::TemporalStability, "Temporal Stability");
                ui.separator();
                ui.label("── v1 fields ──");
                ui.selectable_value(ctx.mode, SidebandOverlayMode::NoiseSigma, "Noise Sigma");
                ui.selectable_value(ctx.mode, SidebandOverlayMode::NoiseConfidence, "Noise Confidence");
                ui.selectable_value(ctx.mode, SidebandOverlayMode::ClipRisk, "Clip Risk");
                ui.selectable_value(ctx.mode, SidebandOverlayMode::StructureClass, "Structure Class");
                ui.selectable_value(ctx.mode, SidebandOverlayMode::DofSharpness, "DoF Sharpness");
                ui.selectable_value(ctx.mode, SidebandOverlayMode::VignettingGain, "Vignetting Gain");
                ui.selectable_value(ctx.mode, SidebandOverlayMode::DenoiseConfidence, "Denoise Confidence");
            });

        // Opacity slider
        ui.add(egui::Slider::new(ctx.opacity, 0.0..=1.0).text("Opacity"));

        // Show values toggle
        ui.checkbox(ctx.show_values, "Show values (zoom >= 2x)");

        ui.separator();

        // Frame info
        if let Some(frame) = ctx.current_frame {
            ui.heading(format!("Frame {} (id={})", ctx.current_frame_idx, frame.frame_id));

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

                // v1 frame fields (show if any v1 field is non-zero)
                if frame.iso_class > 0 || frame.ae_state > 0 || frame.scene_change_score > 0
                    || frame.dynamic_range_q8 > 0 || frame.histogram_shape > 0
                    || frame.ca_severity_q8 > 0 || frame.distortion_k1_q8 != 0
                {
                    ui.label("──");
                    ui.label("v1 ──");
                    ui.end_row();
                    ui.label("ISO class:");
                    ui.label(match frame.iso_class {
                        0 => "Low".to_string(),
                        1 => "Med".to_string(),
                        2 => "High".to_string(),
                        v => format!("{}", v),
                    });
                    ui.end_row();
                    ui.label("AE state:");
                    ui.label(match frame.ae_state {
                        0 => "Stable".to_string(),
                        1 => "Converging".to_string(),
                        2 => "Hunting".to_string(),
                        v => format!("{}", v),
                    });
                    ui.end_row();
                    ui.label("Histogram:");
                    ui.label(match frame.histogram_shape {
                        0 => "Normal".to_string(),
                        1 => "LowKey".to_string(),
                        2 => "HighKey".to_string(),
                        3 => "Bimodal".to_string(),
                        v => format!("{}", v),
                    });
                    ui.end_row();
                    ui.label("Dyn range:");
                    ui.label(format!("{:.1} stops", frame.dynamic_range_q8 as f64 / 16.0));
                    ui.end_row();
                    ui.label("CA severity:");
                    ui.label(format!("{}", frame.ca_severity_q8));
                    ui.end_row();
                    ui.label("Distortion k1:");
                    ui.label(format!("{:.2}", frame.distortion_k1_q8 as f64 / 100.0));
                    ui.end_row();
                    ui.label("Scene change:");
                    ui.label(format!("{}", frame.scene_change_score));
                    ui.end_row();
                }
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
                    let max_str = if max_qp >= 0 { format!("+{}", max_qp) } else { format!("{}", max_qp) };
                    ui.label(format!("mean={:.1} [{}, {}]", mean_qp, min_qp, max_str));
                    ui.end_row();
                    ui.label("Activity:");
                    ui.label(format!("mean={:.1}", mean_act));
                    ui.end_row();
                });
            }
        } else if ctx.sideband.is_some() {
            ui.label(format!("Frame {} -- no sideband data", ctx.current_frame_idx));
        }

        None
    }
}

pub enum SidebandAction {
    LoadRequested,
    Unload,
}
