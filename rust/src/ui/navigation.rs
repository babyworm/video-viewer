use std::collections::HashSet;
use eframe::egui;

pub const FPS_OPTIONS: [u32; 8] = [1, 5, 10, 15, 24, 25, 30, 60];

#[derive(Debug, Clone)]
pub enum NavigationAction {
    Seek(usize),
    TogglePlay,
    NextFrame,
    PrevFrame,
    FirstFrame,
    LastFrame,
    SetFps(u32),
}

pub struct NavigationBar {
    pub selected_fps_idx: usize,
}

impl NavigationBar {
    pub fn new() -> Self {
        // Default to 30 fps (index 6 in FPS_OPTIONS)
        let default_idx = FPS_OPTIONS.iter().position(|&f| f == 30).unwrap_or(6);
        Self {
            selected_fps_idx: default_idx,
        }
    }

    pub fn fps(&self) -> u32 {
        FPS_OPTIONS[self.selected_fps_idx]
    }

    /// Render the navigation bar. Returns `Some(action)` when the user interacts.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        current_frame: usize,
        total_frames: usize,
        is_playing: bool,
        bookmarks: &HashSet<usize>,
        scene_changes: &[usize],
    ) -> Option<NavigationAction> {
        let mut action: Option<NavigationAction> = None;

        ui.horizontal(|ui| {
            // --- Frame slider ---
            let max_frame = if total_frames > 0 { total_frames - 1 } else { 0 };
            let mut slider_val = current_frame as u32;
            let slider = egui::Slider::new(&mut slider_val, 0..=max_frame as u32)
                .show_value(false);
            let slider_resp = ui.add_sized([300.0, 20.0], slider);
            if slider_resp.changed() {
                action = Some(NavigationAction::Seek(slider_val as usize));
            }
            let slider_rect = slider_resp.rect;

            // Draw scene change and bookmark markers on the slider
            if total_frames > 1 {
                let painter = ui.painter_at(slider_rect);
                for &frame_idx in scene_changes {
                    let x = slider_rect.left()
                        + (frame_idx as f32 / max_frame as f32) * slider_rect.width();
                    painter.line_segment(
                        [
                            egui::pos2(x, slider_rect.top()),
                            egui::pos2(x, slider_rect.bottom()),
                        ],
                        egui::Stroke::new(2.0, egui::Color32::RED),
                    );
                }
                for &frame_idx in bookmarks {
                    let x = slider_rect.left()
                        + (frame_idx as f32 / max_frame as f32) * slider_rect.width();
                    painter.line_segment(
                        [
                            egui::pos2(x, slider_rect.top()),
                            egui::pos2(x, slider_rect.bottom()),
                        ],
                        egui::Stroke::new(2.0, egui::Color32::from_rgb(0, 200, 200)),
                    );
                }
            }

            ui.separator();

            // --- Frame label ---
            ui.label(format!("Frame {}/{}", current_frame, max_frame));

            ui.separator();

            // --- Transport buttons ---
            if ui.small_button("|<").on_hover_text("First frame").clicked() {
                action = Some(NavigationAction::FirstFrame);
            }
            if ui.small_button("<").on_hover_text("Previous frame").clicked() {
                action = Some(NavigationAction::PrevFrame);
            }

            let play_label = if is_playing { "Pause" } else { "Play" };
            if ui.button(play_label).clicked() {
                action = Some(NavigationAction::TogglePlay);
            }

            if ui.small_button(">").on_hover_text("Next frame").clicked() {
                action = Some(NavigationAction::NextFrame);
            }
            if ui.small_button(">|").on_hover_text("Last frame").clicked() {
                action = Some(NavigationAction::LastFrame);
            }

            ui.separator();

            // --- FPS combo ---
            ui.label("FPS:");
            let current_fps = FPS_OPTIONS[self.selected_fps_idx];
            egui::ComboBox::from_id_salt("fps_combo")
                .selected_text(current_fps.to_string())
                .show_ui(ui, |ui| {
                    for (idx, &fps) in FPS_OPTIONS.iter().enumerate() {
                        if ui
                            .selectable_value(&mut self.selected_fps_idx, idx, fps.to_string())
                            .clicked()
                        {
                            action = Some(NavigationAction::SetFps(fps));
                        }
                    }
                });
        });

        action
    }
}

impl Default for NavigationBar {
    fn default() -> Self {
        Self::new()
    }
}
