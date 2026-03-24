use eframe::egui;
use crate::core::formats::get_all_formats;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Default)]
pub enum DialogState {
    #[default]
    None,
    OpenFile,
    SaveFile,
    Parameters,
    Export,
    Convert,
    PngExport,
    Settings,
    BatchConvert,
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

/// Commonly used formats (by name prefix) shown at the top of the convert combo.
const COMMON_FORMAT_PREFIXES: &[&str] = &[
    "I420", "YV12", "NV12", "NV21", "YUYV", "UYVY",
    "YUV422P", "YUV444P",
    "RGB24", "BGR24", "Greyscale (8",
];

/// Entry in the convert format list — either a selectable format or a separator.
#[derive(Clone)]
enum ConvertFormatEntry {
    Format { name: String, idx: usize },
    Separator,
}

pub struct ConvertDialog {
    pub input_info: String,
    /// Original input file path (for auto-generating output name).
    input_path: String,
    /// Input file stem (cached).
    input_stem: String,
    /// Selected index into `selectable_formats`.
    pub output_format_idx: usize,
    /// Previous format index (to detect changes for auto-name update).
    prev_format_idx: usize,
    /// Ordered display entries (common formats, separator, rest).
    entries: Vec<ConvertFormatEntry>,
    /// Flat list of selectable format names (for result lookup).
    selectable_formats: Vec<String>,
    pub output_path: String,
    /// Whether user has manually typed in the output path text field.
    user_typed_path: bool,
    /// Output directory (updated by browser selection; defaults to input dir).
    output_dir: PathBuf,
    /// Inline directory browser.
    dir_browser: Option<FileBrowser>,
    /// Starting directory for browser.
    initial_dir: PathBuf,
    /// Conversion progress: None = idle, Some((current, total)) = running.
    pub progress: Option<(usize, usize)>,
}

impl ConvertDialog {
    pub fn new(input_info: &str, input_path: Option<&str>) -> Self {
        let all: Vec<String> = get_all_formats()
            .iter()
            .map(|f| f.name.clone())
            .collect();

        // Partition into common and rest
        let mut common = Vec::new();
        let mut rest = Vec::new();
        for name in &all {
            if COMMON_FORMAT_PREFIXES.iter().any(|p| name.starts_with(p)) {
                common.push(name.clone());
            } else {
                rest.push(name.clone());
            }
        }

        // Build flat selectable list: common first, then rest
        let mut selectable_formats = Vec::with_capacity(all.len());
        selectable_formats.extend(common.iter().cloned());
        selectable_formats.extend(rest.iter().cloned());

        // Build display entries with separator
        let mut entries = Vec::new();
        for (i, name) in common.iter().enumerate() {
            entries.push(ConvertFormatEntry::Format { name: name.clone(), idx: i });
        }
        if !common.is_empty() && !rest.is_empty() {
            entries.push(ConvertFormatEntry::Separator);
        }
        let offset = common.len();
        for (i, name) in rest.iter().enumerate() {
            entries.push(ConvertFormatEntry::Format { name: name.clone(), idx: offset + i });
        }

        let initial_dir = input_path
            .and_then(|p| Path::new(p).parent())
            .filter(|p| p.is_dir())
            .map(|p| p.to_path_buf())
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("/"));

        let input_stem = input_path
            .map(|p| Path::new(p).file_stem().and_then(|s| s.to_str()).unwrap_or("output").to_string())
            .unwrap_or_else(|| "output".to_string());

        // Auto-generate initial output path
        let default_output = Self::make_output_path(
            input_path.unwrap_or(""),
            selectable_formats.first().map(|s| s.as_str()).unwrap_or(""),
        );

        Self {
            input_info: input_info.to_string(),
            input_path: input_path.unwrap_or("").to_string(),
            input_stem,
            output_format_idx: 0,
            prev_format_idx: usize::MAX, // force initial update
            entries,
            selectable_formats,
            output_path: default_output,
            user_typed_path: false,
            output_dir: initial_dir.clone(),
            dir_browser: None,
            initial_dir,
            progress: None,
        }
    }

    /// Generate output filename tag and extension from format name.
    fn format_tag_ext(format_name: &str) -> (String, &'static str) {
        let tag = format_name
            .split_whitespace()
            .next()
            .unwrap_or("raw")
            .to_lowercase();
        let ext = if tag.contains("rgb") || tag.contains("bgr") || tag.contains("rgba") || tag.contains("argb") {
            "rgb"
        } else if tag.contains("grey") || tag.contains("gray") {
            "gray"
        } else if tag.contains("bayer") {
            "raw"
        } else {
            "yuv"
        };
        (tag, ext)
    }

    /// Generate output path: `<input_stem>_<format_tag>.<ext>`
    fn make_output_path(input_path: &str, format_name: &str) -> String {
        if input_path.is_empty() {
            return String::new();
        }
        let p = Path::new(input_path);
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        let dir = p.parent().unwrap_or(Path::new("."));
        let (tag, ext) = Self::format_tag_ext(format_name);
        dir.join(format!("{}_{}.{}", stem, tag, ext))
            .to_string_lossy()
            .to_string()
    }

    pub fn show(&mut self, ctx: &egui::Context) -> Option<Option<(String, String)>> {
        let mut result = None;
        let mut open = true;

        egui::Window::new("Convert")
            .open(&mut open)
            .resizable(true)
            .min_width(500.0)
            .show(ctx, |ui| {
                ui.label(format!("Input: {}", self.input_info));
                ui.separator();

                // --- Output format ---
                ui.horizontal(|ui| {
                    ui.label("Output format:");
                    let current = self
                        .selectable_formats
                        .get(self.output_format_idx)
                        .cloned()
                        .unwrap_or_default();
                    egui::ComboBox::from_id_salt("convert_format")
                        .selected_text(&current)
                        .show_ui(ui, |ui| {
                            for entry in &self.entries {
                                match entry {
                                    ConvertFormatEntry::Format { name, idx } => {
                                        ui.selectable_value(&mut self.output_format_idx, *idx, name.as_str());
                                    }
                                    ConvertFormatEntry::Separator => {
                                        ui.separator();
                                    }
                                }
                            }
                        });
                });

                // Auto-update output path when format changes (unless user typed manually)
                if self.output_format_idx != self.prev_format_idx {
                    self.prev_format_idx = self.output_format_idx;
                    if !self.user_typed_path {
                        let fmt_name = self.selectable_formats
                            .get(self.output_format_idx)
                            .cloned()
                            .unwrap_or_default();
                        let (tag, ext) = Self::format_tag_ext(&fmt_name);
                        self.output_path = self.output_dir
                            .join(format!("{}_{}.{}", self.input_stem, tag, ext))
                            .to_string_lossy()
                            .to_string();
                    }
                }

                // --- Output path with Browse ---
                ui.horizontal(|ui| {
                    ui.label("Output path:");
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut self.output_path)
                            .desired_width(350.0)
                            .hint_text("/path/to/output.yuv"),
                    );
                    if resp.changed() {
                        self.user_typed_path = true;
                    }
                    let browse_label = if self.dir_browser.is_some() { "Close" } else { "Browse..." };
                    if ui.button(browse_label).clicked() {
                        if self.dir_browser.is_some() {
                            self.dir_browser = None;
                        } else {
                            // Start from output path's dir, or initial_dir
                            let start = if !self.output_path.is_empty() {
                                Path::new(&self.output_path)
                                    .parent()
                                    .filter(|p| p.is_dir())
                                    .map(|p| p.to_path_buf())
                                    .unwrap_or_else(|| self.initial_dir.clone())
                            } else {
                                self.initial_dir.clone()
                            };
                            self.dir_browser = Some(FileBrowser::new(start));
                        }
                    }
                });

                // Directory browser with "Select this folder" button
                if self.dir_browser.is_some() {
                    ui.separator();

                    // Collect needed values before borrowing browser
                    let cur_dir_display = self.dir_browser.as_ref().unwrap()
                        .current_dir.display().to_string();
                    let cur_dir_clone = self.dir_browser.as_ref().unwrap()
                        .current_dir.clone();
                    let fmt_name = self.selectable_formats
                        .get(self.output_format_idx)
                        .cloned()
                        .unwrap_or_default();
                    let input_stem = Path::new(&self.input_path)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("output")
                        .to_string();
                    let (tag, ext) = Self::format_tag_ext(&fmt_name);

                    // "Select this folder" button
                    let mut select_folder = false;
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(format!("Dir: {}", cur_dir_display)).small());
                        if ui.button("Select this folder").clicked() {
                            select_folder = true;
                        }
                    });

                    if select_folder {
                        self.output_dir = cur_dir_clone.clone();
                        self.output_path = cur_dir_clone
                            .join(format!("{}_{}.{}", input_stem, tag, ext))
                            .to_string_lossy()
                            .to_string();
                        self.user_typed_path = false; // allow format changes to update name
                        self.dir_browser = None;
                    }

                    // File list (navigating dirs, or picking a file directly)
                    if let Some(ref mut browser) = self.dir_browser {
                        if let Some(selected) = browser.show(ui) {
                            let selected_path = Path::new(&selected);
                            if selected_path.is_dir() {
                                self.output_dir = selected_path.to_path_buf();
                                self.output_path = selected_path
                                    .join(format!("{}_{}.{}", input_stem, tag, ext))
                                    .to_string_lossy()
                                    .to_string();
                                self.user_typed_path = false;
                            } else {
                                // User picked a specific file — use it directly
                                self.output_path = selected;
                                self.user_typed_path = true;
                            }
                            self.dir_browser = None;
                        }
                    }
                }

                // --- Progress bar ---
                if let Some((current, total)) = self.progress {
                    ui.separator();
                    let frac = if total > 0 { current as f32 / total as f32 } else { 0.0 };
                    ui.add(egui::ProgressBar::new(frac).text(format!("{}/{} frames", current, total)));
                }

                ui.separator();
                ui.horizontal(|ui| {
                    let is_running = self.progress.is_some();
                    let can_convert = !self.output_path.is_empty() && !is_running;
                    if ui
                        .add_enabled(can_convert, egui::Button::new("Convert"))
                        .clicked()
                    {
                        let fmt = self
                            .selectable_formats
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
// Batch Convert Dialog
// ---------------------------------------------------------------------------

pub struct BatchConvertDialog {
    pub files: Vec<String>,
    pub output_dir: String,
    pub output_format_idx: usize,
    pub selectable_formats: Vec<String>,
    pub new_file_path: String,
    /// Width/height for raw files (ignored for Y4M).
    pub width: u32,
    pub height: u32,
}

impl BatchConvertDialog {
    pub fn new(default_width: u32, default_height: u32) -> Self {
        let common_formats = [
            "I420", "NV12", "NV21", "YV12", "YUYV", "UYVY", "YUV422P",
            "YUV444P", "RGB24", "BGR24",
        ];
        Self {
            files: Vec::new(),
            output_dir: String::new(),
            output_format_idx: 0,
            selectable_formats: common_formats.iter().map(|s| s.to_string()).collect(),
            new_file_path: String::new(),
            width: default_width,
            height: default_height,
        }
    }

    /// Returns Some(list of (input_path, output_format, output_path)) on Convert,
    /// Some(empty vec) on Cancel, None if still open.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<Vec<(String, String, String)>> {
        let mut result = None;
        let mut open = true;

        egui::Window::new("Batch Convert")
            .open(&mut open)
            .resizable(true)
            .default_width(500.0)
            .show(ctx, |ui| {
                // --- Add files ---
                ui.horizontal(|ui| {
                    ui.label("Add file:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.new_file_path)
                            .desired_width(300.0)
                            .hint_text("/path/to/video.yuv"),
                    );
                    if ui.button("+").clicked() && !self.new_file_path.is_empty() {
                        self.files.push(self.new_file_path.clone());
                        self.new_file_path.clear();
                    }
                });

                // --- File list ---
                ui.separator();
                ui.label(format!("Files ({})", self.files.len()));
                let mut remove_idx = None;
                egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                    for (i, path) in self.files.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.monospace(
                                std::path::Path::new(path)
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or(path),
                            );
                            if ui.small_button("x").clicked() {
                                remove_idx = Some(i);
                            }
                        });
                    }
                });
                if let Some(idx) = remove_idx {
                    self.files.remove(idx);
                }

                ui.separator();

                // --- Parameters ---
                egui::Grid::new("batch_params").show(ui, |ui| {
                    ui.label("Width:");
                    ui.add(egui::DragValue::new(&mut self.width).range(1..=8192));
                    ui.label("Height:");
                    ui.add(egui::DragValue::new(&mut self.height).range(1..=8192));
                    ui.end_row();

                    ui.label("Output format:");
                    let selected = self.selectable_formats
                        .get(self.output_format_idx)
                        .cloned()
                        .unwrap_or_default();
                    egui::ComboBox::from_id_salt("batch_out_fmt")
                        .selected_text(&selected)
                        .show_ui(ui, |ui| {
                            for (idx, name) in self.selectable_formats.iter().enumerate() {
                                ui.selectable_value(&mut self.output_format_idx, idx, name);
                            }
                        });
                    ui.end_row();

                    ui.label("Output dir:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.output_dir)
                            .desired_width(250.0)
                            .hint_text("/path/to/output/"),
                    );
                    ui.end_row();
                });

                ui.separator();
                ui.horizontal(|ui| {
                    let can_convert = !self.files.is_empty() && !self.output_dir.is_empty();
                    if ui
                        .add_enabled(can_convert, egui::Button::new("Convert All"))
                        .clicked()
                    {
                        let fmt = self.selectable_formats
                            .get(self.output_format_idx)
                            .cloned()
                            .unwrap_or_default();
                        let out_dir = std::path::Path::new(&self.output_dir);
                        let jobs: Vec<(String, String, String)> = self.files.iter().map(|input| {
                            let stem = std::path::Path::new(input)
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("output");
                            let out_path = out_dir
                                .join(format!("{}_{}.yuv", stem, fmt.to_lowercase()))
                                .to_string_lossy()
                                .to_string();
                            (input.clone(), fmt.clone(), out_path)
                        }).collect();
                        result = Some(jobs);
                    }
                    if ui.button("Cancel").clicked() {
                        result = Some(Vec::new());
                    }
                });
            });

        if !open {
            return Some(Vec::new());
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
// Inline File Browser (egui-based, no OS dialog dependency)
// ---------------------------------------------------------------------------

/// Video file extensions recognized by the browser filter.
const VIDEO_EXTENSIONS: &[&str] = &[
    "yuv", "y4m", "raw", "rgb", "bgr", "nv12", "nv21", "yuyv", "uyvy",
    "yvyu", "i420", "yv12", "422p", "444p", "grey", "gray", "bayer",
];

struct FileBrowser {
    pub current_dir: PathBuf,
    entries: Vec<DirEntry>,
    filter_video_only: bool,
    filter_text: String,
    error: Option<String>,
}

struct DirEntry {
    name: String,
    is_dir: bool,
    full_path: PathBuf,
}

impl FileBrowser {
    fn new(start_dir: PathBuf) -> Self {
        let mut fb = Self {
            current_dir: start_dir,
            entries: Vec::new(),
            filter_video_only: false,
            filter_text: String::new(),
            error: None,
        };
        fb.refresh();
        fb
    }

    fn navigate_to(&mut self, dir: PathBuf) {
        self.current_dir = dir;
        self.refresh();
    }

    fn refresh(&mut self) {
        self.entries.clear();
        self.error = None;
        let read = match std::fs::read_dir(&self.current_dir) {
            Ok(r) => r,
            Err(e) => {
                self.error = Some(format!("Cannot read directory: {e}"));
                return;
            }
        };
        let mut dirs = Vec::new();
        let mut files = Vec::new();
        for entry in read.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') {
                continue; // skip hidden files
            }
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if is_dir {
                dirs.push(DirEntry {
                    name,
                    is_dir: true,
                    full_path: entry.path(),
                });
            } else {
                if self.filter_video_only {
                    let ext = Path::new(&name)
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_lowercase();
                    if !VIDEO_EXTENSIONS.contains(&ext.as_str()) {
                        continue;
                    }
                }
                files.push(DirEntry {
                    name,
                    is_dir: false,
                    full_path: entry.path(),
                });
            }
        }
        dirs.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        files.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        self.entries.extend(dirs);
        self.entries.extend(files);
    }

    /// Show the browser UI. Returns Some(path) when a file is selected.
    fn show(&mut self, ui: &mut egui::Ui) -> Option<String> {
        let mut selected_path = None;

        // Current directory + parent navigation
        ui.horizontal(|ui| {
            if ui.button("⬆ Up").clicked() {
                if let Some(parent) = self.current_dir.parent() {
                    let parent = parent.to_path_buf();
                    self.navigate_to(parent);
                }
            }
            // Editable directory path
            let mut dir_str = self.current_dir.to_string_lossy().to_string();
            let resp = ui.add(
                egui::TextEdit::singleline(&mut dir_str)
                    .desired_width(ui.available_width() - 5.0),
            );
            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                let p = PathBuf::from(&dir_str);
                if p.is_dir() {
                    self.navigate_to(p);
                }
            }
        });

        ui.horizontal(|ui| {
            if ui.checkbox(&mut self.filter_video_only, "Video files only").changed() {
                self.refresh();
            }
            ui.separator();
            ui.label("Filter:");
            ui.add(
                egui::TextEdit::singleline(&mut self.filter_text)
                    .desired_width(150.0)
                    .hint_text("name filter"),
            );
        });

        if let Some(ref err) = self.error {
            ui.colored_label(egui::Color32::RED, err);
        }

        // File list with scroll
        let filter = self.filter_text.trim().to_lowercase();
        egui::ScrollArea::vertical()
            .max_height(250.0)
            .show(ui, |ui| {
                for entry in &self.entries {
                    // Directories always shown; files filtered by pattern
                    if !entry.is_dir && !filter.is_empty() {
                        let name_lower = entry.name.to_lowercase();
                        let matches = if filter.starts_with('*') && filter.ends_with('*') && filter.len() > 2 {
                            // *text* → contains
                            name_lower.contains(&filter[1..filter.len() - 1])
                        } else if let Some(suffix) = filter.strip_prefix('*') {
                            // *text → suffix (ends with)
                            name_lower.ends_with(suffix)
                        } else if filter.ends_with('*') {
                            // text* → prefix (starts with)
                            name_lower.starts_with(&filter[..filter.len() - 1])
                        } else {
                            // plain text → prefix match
                            name_lower.starts_with(&filter)
                        };
                        if !matches {
                            continue;
                        }
                    }

                    let label = if entry.is_dir {
                        format!("📁 {}", entry.name)
                    } else {
                        format!("  {}", entry.name)
                    };
                    let resp = ui.selectable_label(false, &label);
                    if resp.double_clicked() {
                        if entry.is_dir {
                            let path = entry.full_path.clone();
                            selected_path = Some(("__nav__".to_string(), path));
                        } else {
                            selected_path = Some((
                                entry.full_path.to_string_lossy().to_string(),
                                entry.full_path.clone(),
                            ));
                        }
                    } else if resp.clicked() && !entry.is_dir {
                        selected_path = Some((
                            entry.full_path.to_string_lossy().to_string(),
                            entry.full_path.clone(),
                        ));
                    }
                }
            });

        // Process navigation vs selection
        if let Some((action, path)) = selected_path {
            if action == "__nav__" {
                self.navigate_to(path);
                None
            } else {
                Some(action)
            }
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Open File Dialog (text path input + inline file browser)
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
    /// Inline file browser (created on demand when Browse is clicked).
    file_browser: Option<FileBrowser>,
    /// Starting directory for file browser.
    initial_dir: PathBuf,
}

impl OpenFileDialog {
    /// Create a new Open File dialog.
    /// `initial_dir` sets the starting directory for the file browser:
    /// pass the directory of the currently open file, or None for cwd.
    pub fn new(default_width: u32, default_height: u32, default_format: &str, initial_dir: Option<&str>) -> Self {
        let format_names: Vec<String> = get_all_formats()
            .iter()
            .map(|f| f.name.clone())
            .collect();
        let format_idx = format_names
            .iter()
            .position(|n| n.eq_ignore_ascii_case(default_format))
            .unwrap_or(0);
        let initial_dir = initial_dir
            .map(PathBuf::from)
            .filter(|p| p.is_dir())
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| {
                std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/"))
            });
        Self {
            path: String::new(),
            width: default_width,
            height: default_height,
            format_idx,
            format_names,
            is_y4m: false,
            last_hinted_path: String::new(),
            file_browser: None,
            initial_dir,
        }
    }

    /// Returns Some((path, w, h, format)) on Open, None on cancel, stays open otherwise.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<Option<(String, u32, u32, String)>> {
        let mut result = None;
        let mut open = true;

        egui::Window::new("Open File")
            .open(&mut open)
            .resizable(true)
            .collapsible(false)
            .min_width(500.0)
            .show(ctx, |ui| {
                // File path + Browse outside Grid so button isn't clipped
                ui.horizontal(|ui| {
                    ui.label("File path:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.path)
                            .desired_width(350.0)
                            .hint_text("/path/to/video.yuv"),
                    );
                    let browse_label = if self.file_browser.is_some() { "Close" } else { "Browse..." };
                    if ui.button(browse_label).clicked() {
                        if self.file_browser.is_some() {
                            self.file_browser = None;
                        } else {
                            let start = if !self.path.is_empty() {
                                Path::new(&self.path)
                                    .parent()
                                    .filter(|p| p.is_dir())
                                    .map(|p| p.to_path_buf())
                                    .unwrap_or_else(|| self.initial_dir.clone())
                            } else {
                                self.initial_dir.clone()
                            };
                            self.file_browser = Some(FileBrowser::new(start));
                        }
                    }
                });

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

                // Width/Height/Format in a Grid
                egui::Grid::new("open_file_grid")
                    .num_columns(2)
                    .spacing(egui::vec2(8.0, 4.0))
                    .show(ui, |ui| {
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

                // Inline file browser (shown when Browse is active)
                if let Some(ref mut browser) = self.file_browser {
                    ui.separator();
                    if let Some(selected) = browser.show(ui) {
                        self.path = selected;
                        self.file_browser = None; // auto-close on selection
                    }
                }

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
                            .desired_width(350.0)
                            .hint_text("/path/to/output.png"),
                    );
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
                        ("M", "Toggle magnifier"),
                        ("C", "Center image"),
                        ("Ctrl+B", "Next bookmark"),
                        ("Ctrl+Shift+B", "Previous bookmark"),
                        ("Ctrl+Left", "Previous scene change"),
                        ("Ctrl+Right", "Next scene change"),
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
            ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
            ui.separator();
            ui.label("YUV/Raw Video Viewer with egui");
            ui.label("Copyright (c) babyworm (Hyun-Gyu Kim)");
            ui.hyperlink_to("Github: /babyworm/video-viewer", "https://github.com/babyworm/video-viewer");
            ui.separator();
            ui.label("Built with egui, rayon");
        });
    open
}
