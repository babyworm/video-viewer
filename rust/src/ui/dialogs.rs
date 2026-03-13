use eframe::egui;
use crate::core::formats::get_all_formats;

#[derive(Debug, Clone, PartialEq)]
pub enum DialogState {
    None,
    Parameters,
    Export,
    Convert,
    BatchConvert,
    PngExport,
    Settings,
    Shortcuts,
    Bookmarks,
    About,
}

impl Default for DialogState {
    fn default() -> Self {
        Self::None
    }
}

// ---------------------------------------------------------------------------
// Parameters Dialog
// ---------------------------------------------------------------------------

pub struct ParametersDialog {
    pub width: u32,
    pub height: u32,
    pub format_idx: usize,
    pub format_names: Vec<String>,
}

impl ParametersDialog {
    pub fn new(width: u32, height: u32, current_format: &str) -> Self {
        let format_names: Vec<String> = get_all_formats()
            .iter()
            .map(|f| f.name.clone())
            .collect();
        let format_idx = format_names
            .iter()
            .position(|n| n.eq_ignore_ascii_case(current_format))
            .unwrap_or(0);
        Self {
            width,
            height,
            format_idx,
            format_names,
        }
    }

    /// Returns Some((w, h, format_name)) on OK, None if still open or cancelled.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<Option<(u32, u32, String)>> {
        let mut result = None;
        let mut open = true;

        egui::Window::new("Video Parameters")
            .open(&mut open)
            .resizable(false)
            .collapsible(false)
            .show(ctx, |ui| {
                egui::Grid::new("params_grid").show(ui, |ui| {
                    ui.label("Width:");
                    ui.add(egui::DragValue::new(&mut self.width).range(1..=8192));
                    ui.end_row();

                    ui.label("Height:");
                    ui.add(egui::DragValue::new(&mut self.height).range(1..=8192));
                    ui.end_row();

                    ui.label("Format:");
                    let current = self
                        .format_names
                        .get(self.format_idx)
                        .cloned()
                        .unwrap_or_default();
                    egui::ComboBox::from_id_salt("format_combo")
                        .selected_text(&current)
                        .show_ui(ui, |ui| {
                            for (idx, name) in self.format_names.iter().enumerate() {
                                ui.selectable_value(&mut self.format_idx, idx, name);
                            }
                        });
                    ui.end_row();
                });

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("OK").clicked() {
                        let fmt = self
                            .format_names
                            .get(self.format_idx)
                            .cloned()
                            .unwrap_or_else(|| "I420".to_string());
                        result = Some(Some((self.width, self.height, fmt)));
                    }
                    if ui.button("Cancel").clicked() {
                        result = Some(None);
                    }
                });
            });

        if !open {
            return Some(None);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Export Dialog (raw clip)
// ---------------------------------------------------------------------------

pub struct ExportDialog {
    pub start_frame: usize,
    pub end_frame: usize,
    pub total_frames: usize,
    pub output_path: String,
}

impl ExportDialog {
    pub fn new(total_frames: usize) -> Self {
        Self {
            start_frame: 0,
            end_frame: total_frames.saturating_sub(1),
            total_frames,
            output_path: String::new(),
        }
    }

    /// Returns Some((start, end, path)) on OK, None if cancelled.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<Option<(usize, usize, String)>> {
        let mut result = None;
        let mut open = true;

        egui::Window::new("Export Clip")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                let max = self.total_frames.saturating_sub(1);
                egui::Grid::new("export_grid").show(ui, |ui| {
                    ui.label("Start frame:");
                    ui.add(egui::DragValue::new(&mut self.start_frame).range(0..=max));
                    ui.end_row();

                    ui.label("End frame:");
                    ui.add(egui::DragValue::new(&mut self.end_frame).range(0..=max));
                    ui.end_row();

                    ui.label("Output path:");
                    ui.text_edit_singleline(&mut self.output_path);
                    if ui.button("Browse...").clicked() {
                        if let Some(path) = rfd::FileDialog::new().save_file() {
                            self.output_path = path.display().to_string();
                        }
                    }
                    ui.end_row();
                });

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Export").clicked() && !self.output_path.is_empty() {
                        result = Some(Some((
                            self.start_frame,
                            self.end_frame,
                            self.output_path.clone(),
                        )));
                    }
                    if ui.button("Cancel").clicked() {
                        result = Some(None);
                    }
                });
            });

        if !open {
            return Some(None);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Convert Dialog
// ---------------------------------------------------------------------------

pub struct ConvertDialog {
    pub input_info: String,
    pub output_format_idx: usize,
    pub format_names: Vec<String>,
    pub output_path: String,
}

impl ConvertDialog {
    pub fn new(input_info: &str) -> Self {
        let format_names: Vec<String> = get_all_formats()
            .iter()
            .map(|f| f.name.clone())
            .collect();
        Self {
            input_info: input_info.to_string(),
            output_format_idx: 0,
            format_names,
            output_path: String::new(),
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> Option<Option<(String, String)>> {
        let mut result = None;
        let mut open = true;

        egui::Window::new("Convert")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(format!("Input: {}", self.input_info));
                ui.separator();

                egui::Grid::new("convert_grid").show(ui, |ui| {
                    ui.label("Output format:");
                    let current = self
                        .format_names
                        .get(self.output_format_idx)
                        .cloned()
                        .unwrap_or_default();
                    egui::ComboBox::from_id_salt("convert_format")
                        .selected_text(&current)
                        .show_ui(ui, |ui| {
                            for (idx, name) in self.format_names.iter().enumerate() {
                                ui.selectable_value(&mut self.output_format_idx, idx, name);
                            }
                        });
                    ui.end_row();

                    ui.label("Output path:");
                    ui.text_edit_singleline(&mut self.output_path);
                    if ui.button("Browse...").clicked() {
                        if let Some(path) = rfd::FileDialog::new().save_file() {
                            self.output_path = path.display().to_string();
                        }
                    }
                    ui.end_row();
                });

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Convert").clicked() && !self.output_path.is_empty() {
                        let fmt = self
                            .format_names
                            .get(self.output_format_idx)
                            .cloned()
                            .unwrap_or_default();
                        result = Some(Some((fmt, self.output_path.clone())));
                    }
                    if ui.button("Cancel").clicked() {
                        result = Some(None);
                    }
                });
            });

        if !open {
            return Some(None);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// PNG Export Dialog
// ---------------------------------------------------------------------------

pub struct PngExportDialog {
    pub start_frame: usize,
    pub end_frame: usize,
    pub total_frames: usize,
    pub output_dir: String,
    pub prefix: String,
}

impl PngExportDialog {
    pub fn new(total_frames: usize) -> Self {
        Self {
            start_frame: 0,
            end_frame: total_frames.saturating_sub(1),
            total_frames,
            output_dir: String::new(),
            prefix: "frame".to_string(),
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> Option<Option<(usize, usize, String, String)>> {
        let mut result = None;
        let mut open = true;

        egui::Window::new("Export PNG Sequence")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                let max = self.total_frames.saturating_sub(1);
                egui::Grid::new("png_export_grid").show(ui, |ui| {
                    ui.label("Start frame:");
                    ui.add(egui::DragValue::new(&mut self.start_frame).range(0..=max));
                    ui.end_row();

                    ui.label("End frame:");
                    ui.add(egui::DragValue::new(&mut self.end_frame).range(0..=max));
                    ui.end_row();

                    ui.label("Output dir:");
                    ui.text_edit_singleline(&mut self.output_dir);
                    if ui.button("Browse...").clicked() {
                        if let Some(path) = rfd::FileDialog::new().pick_folder() {
                            self.output_dir = path.display().to_string();
                        }
                    }
                    ui.end_row();

                    ui.label("Prefix:");
                    ui.text_edit_singleline(&mut self.prefix);
                    ui.end_row();
                });

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Export").clicked() && !self.output_dir.is_empty() {
                        result = Some(Some((
                            self.start_frame,
                            self.end_frame,
                            self.output_dir.clone(),
                            self.prefix.clone(),
                        )));
                    }
                    if ui.button("Cancel").clicked() {
                        result = Some(None);
                    }
                });
            });

        if !open {
            return Some(None);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Settings Dialog
// ---------------------------------------------------------------------------

pub struct SettingsDialog {
    pub max_memory_mb: u32,
    pub zoom_min: f32,
    pub zoom_max: f32,
    pub dark_theme: bool,
    pub default_fps: u32,
    pub default_width: u32,
    pub default_height: u32,
    pub color_matrix: String,
}

impl SettingsDialog {
    pub fn new(settings: &super::settings::Settings) -> Self {
        Self {
            max_memory_mb: settings.cache.max_memory_mb,
            zoom_min: settings.display.zoom_min,
            zoom_max: settings.display.zoom_max,
            dark_theme: settings.display.dark_theme,
            default_fps: settings.defaults.fps,
            default_width: settings.defaults.width,
            default_height: settings.defaults.height,
            color_matrix: settings.defaults.color_matrix.clone(),
        }
    }

    /// Returns true on OK (apply settings), None if still open, false on cancel.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<bool> {
        let mut result = None;
        let mut open = true;

        egui::Window::new("Settings")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.heading("Cache");
                ui.horizontal(|ui| {
                    ui.label("Max memory (MB):");
                    ui.add(egui::DragValue::new(&mut self.max_memory_mb).range(64..=4096));
                });

                ui.separator();
                ui.heading("Display");
                ui.checkbox(&mut self.dark_theme, "Dark theme");
                ui.horizontal(|ui| {
                    ui.label("Zoom min:");
                    ui.add(egui::DragValue::new(&mut self.zoom_min).speed(0.01).range(0.01..=1.0));
                    ui.label("Zoom max:");
                    ui.add(egui::DragValue::new(&mut self.zoom_max).speed(1.0).range(2.0..=100.0));
                });

                ui.separator();
                ui.heading("Defaults");
                egui::Grid::new("defaults_grid").show(ui, |ui| {
                    ui.label("FPS:");
                    ui.add(egui::DragValue::new(&mut self.default_fps).range(1..=120));
                    ui.end_row();

                    ui.label("Width:");
                    ui.add(egui::DragValue::new(&mut self.default_width).range(1..=8192));
                    ui.end_row();

                    ui.label("Height:");
                    ui.add(egui::DragValue::new(&mut self.default_height).range(1..=8192));
                    ui.end_row();

                    ui.label("Color matrix:");
                    egui::ComboBox::from_id_salt("color_matrix")
                        .selected_text(&self.color_matrix)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.color_matrix,
                                "BT.601".to_string(),
                                "BT.601",
                            );
                            ui.selectable_value(
                                &mut self.color_matrix,
                                "BT.709".to_string(),
                                "BT.709",
                            );
                        });
                    ui.end_row();
                });

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("OK").clicked() {
                        result = Some(true);
                    }
                    if ui.button("Cancel").clicked() {
                        result = Some(false);
                    }
                });
            });

        if !open {
            return Some(false);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Shortcuts Dialog
// ---------------------------------------------------------------------------

pub fn show_shortcuts_dialog(ctx: &egui::Context) -> bool {
    let mut open = true;
    egui::Window::new("Keyboard Shortcuts")
        .open(&mut open)
        .resizable(false)
        .show(ctx, |ui| {
            egui::Grid::new("shortcuts_grid")
                .striped(true)
                .show(ui, |ui| {
                    let shortcuts = [
                        ("Space", "Play / Pause"),
                        ("Left / Right", "Previous / Next frame"),
                        ("Home / End", "First / Last frame"),
                        ("0-4", "Component view (Full/Y/U/V/Split)"),
                        ("F", "Fit to view"),
                        ("G", "Cycle grid size"),
                        ("B", "Toggle bookmark"),
                        ("Ctrl+S", "Save frame as PNG"),
                        ("Ctrl+C", "Copy frame to clipboard"),
                        ("Ctrl+O", "Open file"),
                        ("Ctrl+Q", "Quit"),
                    ];
                    for (key, action) in shortcuts {
                        ui.monospace(key);
                        ui.label(action);
                        ui.end_row();
                    }
                });
        });
    open
}

// ---------------------------------------------------------------------------
// About Dialog
// ---------------------------------------------------------------------------

pub fn show_about_dialog(ctx: &egui::Context) -> bool {
    let mut open = true;
    egui::Window::new("About Video Viewer")
        .open(&mut open)
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            ui.heading("Video Viewer (Rust)");
            ui.label("Version 0.1.0");
            ui.separator();
            ui.label("YUV/Raw Video Viewer with egui");
            ui.label("Copyright (c) babyworm (Hyun-Gyu Kim)");
            ui.separator();
            ui.label("Built with egui, opencv-rust, rayon");
        });
    open
}
