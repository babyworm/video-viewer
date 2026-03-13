use eframe::egui;
use crate::core::formats::get_all_formats;

#[derive(Debug, Clone, PartialEq)]
pub enum DialogState {
    None,
    OpenFile,
    SaveFile,
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
// Open File Dialog (text-based path input, fallback for when rfd is unavailable)
// ---------------------------------------------------------------------------

pub struct OpenFileDialog {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub format_idx: usize,
    pub format_names: Vec<String>,
    pub is_y4m: bool,
    /// Track last path for which hints were applied, to avoid re-overriding user edits.
    last_hinted_path: String,
}

impl OpenFileDialog {
    pub fn new(default_width: u32, default_height: u32, default_format: &str) -> Self {
        let format_names: Vec<String> = get_all_formats()
            .iter()
            .map(|f| f.name.clone())
            .collect();
        let format_idx = format_names
            .iter()
            .position(|n| n.eq_ignore_ascii_case(default_format))
            .unwrap_or(0);
        Self {
            path: String::new(),
            width: default_width,
            height: default_height,
            format_idx,
            format_names,
            is_y4m: false,
            last_hinted_path: String::new(),
        }
    }

    /// Returns Some((path, w, h, format)) on Open, None on cancel, stays open otherwise.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<Option<(String, u32, u32, String)>> {
        let mut result = None;
        let mut open = true;

        egui::Window::new("Open File")
            .open(&mut open)
            .resizable(false)
            .collapsible(false)
            .min_width(400.0)
            .show(ctx, |ui| {
                egui::Grid::new("open_file_grid")
                    .num_columns(2)
                    .spacing(egui::vec2(8.0, 4.0))
                    .show(ui, |ui| {
                        ui.label("File path:");
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut self.path)
                                    .desired_width(300.0)
                                    .hint_text("/path/to/video.yuv"),
                            );
                            if ui.button("Browse...").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter(
                                        "Video",
                                        &["yuv", "y4m", "rgb", "raw", "nv12", "nv21"],
                                    )
                                    .pick_file()
                                {
                                    self.path = path.display().to_string();
                                }
                            }
                        });
                        ui.end_row();

                        // Auto-detect Y4M and apply hints when path changes
                        let ext = std::path::Path::new(&self.path)
                            .extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("")
                            .to_lowercase();
                        self.is_y4m = ext == "y4m";

                        // Auto-fill from filename hints when path changes
                        if !self.is_y4m && self.path != self.last_hinted_path && !self.path.is_empty() {
                            let hints = crate::core::hints::parse_filename_hints(&self.path);
                            if let Some(w) = hints.width {
                                self.width = w;
                            }
                            if let Some(h) = hints.height {
                                self.height = h;
                            }
                            if let Some(ref fmt) = hints.format {
                                if let Some(idx) = self.format_names.iter().position(|n| n.eq_ignore_ascii_case(fmt)) {
                                    self.format_idx = idx;
                                }
                            }
                            self.last_hinted_path = self.path.clone();
                        }

                        if !self.is_y4m {
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
                            egui::ComboBox::from_id_salt("open_format_combo")
                                .selected_text(&current)
                                .show_ui(ui, |ui| {
                                    for (idx, name) in self.format_names.iter().enumerate() {
                                        ui.selectable_value(&mut self.format_idx, idx, name);
                                    }
                                });
                            ui.end_row();
                        } else {
                            ui.label("Info:");
                            ui.label("Y4M: parameters auto-detected from header");
                            ui.end_row();
                        }
                    });

                if !self.path.is_empty() {
                    // Show hint-detected info
                    let hints = crate::core::hints::parse_filename_hints(&self.path);
                    if hints.width.is_some() || hints.format.is_some() {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(format!(
                                "Detected: {}{}{}",
                                hints
                                    .width
                                    .map(|w| format!("{}x{}", w, hints.height.unwrap_or(0)))
                                    .unwrap_or_default(),
                                if hints.width.is_some() && hints.format.is_some() {
                                    " "
                                } else {
                                    ""
                                },
                                hints.format.as_deref().unwrap_or(""),
                            ))
                            .color(egui::Color32::LIGHT_BLUE),
                        );
                    }
                }

                ui.separator();
                ui.horizontal(|ui| {
                    let can_open = !self.path.is_empty()
                        && std::path::Path::new(&self.path).exists();
                    if ui
                        .add_enabled(can_open, egui::Button::new("Open"))
                        .clicked()
                    {
                        let (w, h, fmt) = if self.is_y4m {
                            (0, 0, String::new())
                        } else {
                            (
                                self.width,
                                self.height,
                                self.format_names
                                    .get(self.format_idx)
                                    .cloned()
                                    .unwrap_or_else(|| "I420".to_string()),
                            )
                        };
                        result = Some(Some((self.path.clone(), w, h, fmt)));
                    }
                    if ui.button("Cancel").clicked() {
                        result = Some(None);
                    }
                    if !self.path.is_empty() && !std::path::Path::new(&self.path).exists() {
                        ui.colored_label(egui::Color32::RED, "File not found");
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
// Save File Dialog (text-based path input)
// ---------------------------------------------------------------------------

pub struct SaveFileDialog {
    pub path: String,
    pub title: String,
}

impl SaveFileDialog {
    pub fn new(title: &str, default_name: &str) -> Self {
        Self {
            path: default_name.to_string(),
            title: title.to_string(),
        }
    }

    /// Returns Some(path) on Save, None on cancel.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<Option<String>> {
        let mut result = None;
        let mut open = true;

        egui::Window::new(&self.title)
            .open(&mut open)
            .resizable(false)
            .collapsible(false)
            .min_width(400.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Save to:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.path)
                            .desired_width(300.0)
                            .hint_text("/path/to/output.png"),
                    );
                    if ui.button("Browse...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("PNG", &["png"])
                            .set_file_name(&self.path)
                            .save_file()
                        {
                            self.path = path.display().to_string();
                        }
                    }
                });

                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!self.path.is_empty(), egui::Button::new("Save"))
                        .clicked()
                    {
                        result = Some(Some(self.path.clone()));
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
