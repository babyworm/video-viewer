use std::collections::HashSet;
use std::time::Instant;
use eframe::egui;

use crate::core::reader::VideoReader;
use crate::ui::canvas::ImageCanvas;
use crate::ui::dialogs::{self, DialogState};
use crate::ui::navigation::{NavigationBar, NavigationAction};
use crate::ui::settings::Settings;
use crate::ui::sidebar::Sidebar;
use crate::ui::toolbar::{Toolbar, ToolbarAction, colorize_channel};

pub struct VideoViewerApp {
    pub current_file: Option<String>,
    pub reader: Option<VideoReader>,
    pub canvas: ImageCanvas,
    pub toolbar: Toolbar,
    pub nav: NavigationBar,
    pub sidebar: Sidebar,
    pub settings: Settings,
    pub current_frame_idx: usize,
    pub is_playing: bool,
    pub loop_playback: bool,
    pub last_frame_time: Option<Instant>,
    pub current_component: u8,
    pub bookmarks: HashSet<usize>,
    /// Current raw frame data (for channel extraction).
    pub current_raw: Option<Vec<u8>>,
    /// Current RGB frame data (for export).
    pub current_rgb: Option<Vec<u8>>,
    /// CLI-provided args for auto-open on startup.
    startup_input: Option<String>,
    startup_width: Option<u32>,
    startup_height: Option<u32>,
    startup_format: Option<String>,
    /// Error message to show in the status bar.
    status_error: Option<String>,
    /// Active dialog state.
    dialog_state: DialogState,
    /// Dialog instances (created on demand).
    params_dialog: Option<dialogs::ParametersDialog>,
    export_dialog: Option<dialogs::ExportDialog>,
    convert_dialog: Option<dialogs::ConvertDialog>,
    png_export_dialog: Option<dialogs::PngExportDialog>,
    settings_dialog: Option<dialogs::SettingsDialog>,
    show_shortcuts: bool,
    show_about: bool,
}

impl VideoViewerApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        input: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        format: Option<String>,
    ) -> Self {
        let settings = Settings::load();
        if settings.display.dark_theme {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
        } else {
            cc.egui_ctx.set_visuals(egui::Visuals::light());
        }

        Self {
            current_file: None,
            reader: None,
            canvas: ImageCanvas::new(),
            toolbar: Toolbar::new(),
            nav: NavigationBar::new(),
            sidebar: Sidebar::new(),
            settings,
            current_frame_idx: 0,
            is_playing: false,
            loop_playback: true,
            last_frame_time: None,
            current_component: 0,
            bookmarks: HashSet::new(),
            current_raw: None,
            current_rgb: None,
            startup_input: input,
            startup_width: width,
            startup_height: height,
            startup_format: format,
            status_error: None,
            dialog_state: DialogState::None,
            params_dialog: None,
            export_dialog: None,
            convert_dialog: None,
            png_export_dialog: None,
            settings_dialog: None,
            show_shortcuts: false,
            show_about: false,
        }
    }

    /// Open a file by path.
    fn open_file(
        &mut self,
        ctx: &egui::Context,
        path: String,
        width: u32,
        height: u32,
        format: &str,
    ) {
        let color_matrix = &self.settings.defaults.color_matrix;
        match VideoReader::open(&path, width, height, format, color_matrix) {
            Ok(mut reader) => {
                self.current_frame_idx = 0;
                self.bookmarks.clear();
                if let Some(fps) = reader.y4m_fps() {
                    let fps_u32 = fps.round() as u32;
                    use crate::ui::navigation::FPS_OPTIONS;
                    if let Some(idx) = FPS_OPTIONS.iter().position(|&f| f == fps_u32) {
                        self.nav.selected_fps_idx = idx;
                    }
                }
                match reader.seek_frame(0) {
                    Ok(raw) => match reader.convert_to_rgb(&raw) {
                        Ok(rgb) => {
                            self.canvas.set_image(ctx, &rgb, reader.width(), reader.height());
                            self.current_rgb = Some(rgb);
                            self.current_raw = Some(raw);
                            self.status_error = None;
                        }
                        Err(e) => {
                            self.status_error = Some(format!("Convert error: {e}"));
                        }
                    },
                    Err(e) => {
                        self.status_error = Some(format!("Seek error: {e}"));
                    }
                }
                self.settings.add_recent_file(&path);
                self.settings.save();
                self.current_file = Some(path);
                self.reader = Some(reader);
                self.is_playing = false;
                self.last_frame_time = None;
            }
            Err(e) => {
                self.status_error = Some(format!("Open error: {e}"));
            }
        }
    }

    /// Seek to `idx`, decode, and push to canvas.
    fn goto_frame(&mut self, ctx: &egui::Context, idx: usize) -> bool {
        let total = self.total_frames();
        if total == 0 {
            return false;
        }
        let idx = idx.min(total - 1);
        let reader = match self.reader.as_mut() {
            Some(r) => r,
            None => return false,
        };
        let raw = match reader.seek_frame(idx) {
            Ok(r) => r,
            Err(e) => {
                self.status_error = Some(format!("Seek error: {e}"));
                return false;
            }
        };
        let rgb = match reader.convert_to_rgb(&raw) {
            Ok(r) => r,
            Err(e) => {
                self.status_error = Some(format!("Convert error: {e}"));
                return false;
            }
        };
        let (w, h) = (reader.width(), reader.height());
        // Apply component view — use reader fields directly to avoid borrow conflict
        let display_rgb = Self::compute_component_view(
            self.current_component, reader, &raw, &rgb, w, h,
        );
        self.canvas.set_image(ctx, &display_rgb, w, h);
        self.current_rgb = Some(rgb);
        self.current_raw = Some(raw);
        self.current_frame_idx = idx;
        self.status_error = None;
        true
    }

    /// Apply component view (channel selection/colorization).
    fn compute_component_view(
        component: u8,
        reader: &VideoReader,
        raw: &[u8],
        rgb: &[u8],
        w: u32,
        h: u32,
    ) -> Vec<u8> {
        if component == 0 {
            return rgb.to_vec();
        }

        let channels = reader.get_channels(raw);
        let is_yuv = {
            let ft = reader.format().format_type;
            matches!(
                ft,
                crate::core::formats::FormatType::YuvPlanar
                    | crate::core::formats::FormatType::YuvSemiPlanar
                    | crate::core::formats::FormatType::YuvPacked
            )
        };
        let names: [&str; 3] = if is_yuv {
            ["Y", "U", "V"]
        } else {
            ["R", "G", "B"]
        };

        match component {
            1..=3 => {
                let ch_idx = (component - 1) as usize;
                let name = names[ch_idx];
                if let Some(ch) = channels.get(name) {
                    colorize_channel(ch, w, h, name)
                } else {
                    rgb.to_vec()
                }
            }
            4 => {
                // Split 2x2: top-left=full, top-right=ch1, bottom-left=ch2, bottom-right=ch3
                let half_w = w as usize / 2;
                let half_h = h as usize / 2;
                let mut out = vec![0u8; (w as usize) * (h as usize) * 3];

                let full = rgb;
                let ch0 = if let Some(ch) = channels.get(names[0]) {
                    colorize_channel(ch, w, h, names[0])
                } else {
                    rgb.to_vec()
                };
                let ch1 = if let Some(ch) = channels.get(names[1]) {
                    colorize_channel(ch, w, h, names[1])
                } else {
                    rgb.to_vec()
                };
                let ch2 = if let Some(ch) = channels.get(names[2]) {
                    colorize_channel(ch, w, h, names[2])
                } else {
                    rgb.to_vec()
                };

                let stride = w as usize * 3;
                for y in 0..half_h {
                    for x in 0..half_w {
                        let src_y = y * 2;
                        let src_x = x * 2;
                        let src_idx = (src_y * w as usize + src_x) * 3;
                        // Top-left: full
                        let dst_idx = (y * w as usize + x) * 3;
                        if src_idx + 2 < full.len() && dst_idx + 2 < out.len() {
                            out[dst_idx] = full[src_idx];
                            out[dst_idx + 1] = full[src_idx + 1];
                            out[dst_idx + 2] = full[src_idx + 2];
                        }
                        // Top-right: ch0
                        let dst_idx = (y * w as usize + x + half_w) * 3;
                        if src_idx + 2 < ch0.len() && dst_idx + 2 < out.len() {
                            out[dst_idx] = ch0[src_idx];
                            out[dst_idx + 1] = ch0[src_idx + 1];
                            out[dst_idx + 2] = ch0[src_idx + 2];
                        }
                        // Bottom-left: ch1
                        let dst_idx = ((y + half_h) * w as usize + x) * 3;
                        if src_idx + 2 < ch1.len() && dst_idx + 2 < out.len() {
                            out[dst_idx] = ch1[src_idx];
                            out[dst_idx + 1] = ch1[src_idx + 1];
                            out[dst_idx + 2] = ch1[src_idx + 2];
                        }
                        // Bottom-right: ch2
                        let dst_idx = ((y + half_h) * w as usize + x + half_w) * 3;
                        if src_idx + 2 < ch2.len() && dst_idx + 2 < out.len() {
                            out[dst_idx] = ch2[src_idx];
                            out[dst_idx + 1] = ch2[src_idx + 1];
                            out[dst_idx + 2] = ch2[src_idx + 2];
                        }
                    }
                }
                out
            }
            _ => rgb.to_vec(),
        }
    }

    fn total_frames(&self) -> usize {
        self.reader.as_ref().map(|r| r.total_frames()).unwrap_or(0)
    }

    fn save_frame_as_png(&self) {
        if let (Some(rgb), Some(reader)) = (&self.current_rgb, &self.reader) {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("PNG", &["png"])
                .set_file_name(&format!("frame_{:06}.png", self.current_frame_idx))
                .save_file()
            {
                let w = reader.width();
                let h = reader.height();
                if let Some(img) = image::RgbImage::from_raw(w, h, rgb.clone()) {
                    if let Err(e) = img.save(&path) {
                        log::error!("Failed to save PNG: {e}");
                    }
                }
            }
        }
    }

    fn copy_frame_to_clipboard(&self) {
        if let (Some(rgb), Some(reader)) = (&self.current_rgb, &self.reader) {
            let w = reader.width() as usize;
            let h = reader.height() as usize;
            // Convert RGB to RGBA for arboard
            let mut rgba = Vec::with_capacity(w * h * 4);
            for pixel in rgb.chunks_exact(3) {
                rgba.push(pixel[0]);
                rgba.push(pixel[1]);
                rgba.push(pixel[2]);
                rgba.push(255);
            }
            let img_data = arboard::ImageData {
                width: w,
                height: h,
                bytes: rgba.into(),
            };
            if let Ok(mut clipboard) = arboard::Clipboard::new() {
                let _ = clipboard.set_image(img_data);
            }
        }
    }
}

impl eframe::App for VideoViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- Auto-open from CLI args ---
        if let Some(path) = self.startup_input.take() {
            let w = self.startup_width.unwrap_or(self.settings.defaults.width);
            let h = self.startup_height.unwrap_or(self.settings.defaults.height);
            let fmt = self
                .startup_format
                .clone()
                .unwrap_or_else(|| self.settings.defaults.format.clone());
            self.open_file(ctx, path, w, h, &fmt);
        }

        // --- Drag & drop ---
        let dropped: Vec<_> = ctx.input(|i| {
            i.raw.dropped_files
                .iter()
                .filter_map(|f| f.path.as_ref().map(|p| p.display().to_string()))
                .collect()
        });
        if let Some(path) = dropped.into_iter().next() {
            let ext = std::path::Path::new(&path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            let (w, h, fmt) = if ext == "y4m" {
                (0, 0, String::new())
            } else {
                // Try filename hints
                let hints = crate::core::hints::parse_filename_hints(&path);
                (
                    hints.width.unwrap_or(self.settings.defaults.width),
                    hints.height.unwrap_or(self.settings.defaults.height),
                    hints.format.unwrap_or_else(|| self.settings.defaults.format.clone()),
                )
            };
            self.open_file(ctx, path, w, h, &fmt);
        }

        // --- Playback tick ---
        if self.is_playing {
            let fps = self.nav.fps();
            let frame_duration_ms = 1000 / fps.max(1);
            let elapsed = self
                .last_frame_time
                .map(|t| t.elapsed().as_millis() as u64)
                .unwrap_or(u64::MAX);

            if elapsed >= frame_duration_ms as u64 {
                let total = self.total_frames();
                let next = self.current_frame_idx + 1;
                if next >= total {
                    if self.loop_playback {
                        self.goto_frame(ctx, 0);
                    } else {
                        self.is_playing = false;
                    }
                } else {
                    self.goto_frame(ctx, next);
                }
                self.last_frame_time = Some(Instant::now());
            }
            ctx.request_repaint();
        }

        // --- Keyboard shortcuts ---
        let keys = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::Space),           // 0
                i.key_pressed(egui::Key::ArrowLeft),       // 1
                i.key_pressed(egui::Key::ArrowRight),      // 2
                i.key_pressed(egui::Key::Home),            // 3
                i.key_pressed(egui::Key::End),             // 4
                i.key_pressed(egui::Key::F),               // 5
                i.key_pressed(egui::Key::G),               // 6
                i.key_pressed(egui::Key::B),               // 7
                i.key_pressed(egui::Key::Num0),            // 8
                i.key_pressed(egui::Key::Num1),            // 9
                i.key_pressed(egui::Key::Num2),            // 10
                i.key_pressed(egui::Key::Num3),            // 11
                i.key_pressed(egui::Key::Num4),            // 12
                i.modifiers.ctrl && i.key_pressed(egui::Key::S), // 13
                i.modifiers.ctrl && i.key_pressed(egui::Key::C), // 14
                i.modifiers.ctrl && i.key_pressed(egui::Key::O), // 15
            )
        });

        if keys.0 {
            self.is_playing = !self.is_playing;
            if self.is_playing {
                self.last_frame_time = Some(Instant::now());
            }
        }
        if keys.1 {
            let cur = self.current_frame_idx;
            if cur > 0 { self.goto_frame(ctx, cur - 1); }
            self.is_playing = false;
        }
        if keys.2 {
            let total = self.total_frames();
            let next = self.current_frame_idx + 1;
            if next < total { self.goto_frame(ctx, next); }
            self.is_playing = false;
        }
        if keys.3 {
            self.goto_frame(ctx, 0);
            self.is_playing = false;
        }
        if keys.4 {
            let last = self.total_frames().saturating_sub(1);
            self.goto_frame(ctx, last);
            self.is_playing = false;
        }
        if keys.5 {
            let avail = ctx.available_rect().size();
            self.canvas.fit_to_view(avail);
        }
        if keys.6 {
            // G: cycle grid
            self.toolbar.grid_idx = (self.toolbar.grid_idx + 1) % 5;
            self.toolbar.grid_size = [0, 16, 32, 64, 128][self.toolbar.grid_idx];
            self.canvas.set_grid_size(self.toolbar.grid_size);
        }
        if keys.7 {
            // B: toggle bookmark
            let idx = self.current_frame_idx;
            if self.bookmarks.contains(&idx) {
                self.bookmarks.remove(&idx);
            } else {
                self.bookmarks.insert(idx);
            }
        }
        // Number keys 0-4: component selection
        for (i, &pressed) in [keys.8, keys.9, keys.10, keys.11, keys.12].iter().enumerate() {
            if pressed {
                self.current_component = i as u8;
                self.toolbar.current_component = i as u8;
                // Re-render current frame with new component
                if self.reader.is_some() {
                    self.goto_frame(ctx, self.current_frame_idx);
                }
            }
        }
        if keys.13 { self.save_frame_as_png(); }
        if keys.14 { self.copy_frame_to_clipboard(); }
        if keys.15 {
            // Ctrl+O: open file
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Video", &["yuv", "y4m", "rgb", "raw", "nv12", "nv21"])
                .pick_file()
            {
                let path_str = path.display().to_string();
                let hints = crate::core::hints::parse_filename_hints(&path_str);
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                let (w, h, fmt) = if ext == "y4m" {
                    (0, 0, String::new())
                } else {
                    (
                        hints.width.unwrap_or(self.settings.defaults.width),
                        hints.height.unwrap_or(self.settings.defaults.height),
                        hints.format.unwrap_or_else(|| self.settings.defaults.format.clone()),
                    )
                };
                self.open_file(ctx, path_str, w, h, &fmt);
            }
        }

        // --- Menu bar ---
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open... (Ctrl+O)").clicked() {
                        ui.close_menu();
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Video", &["yuv", "y4m", "rgb", "raw", "nv12", "nv21"])
                            .pick_file()
                        {
                            let path_str = path.display().to_string();
                            let hints = crate::core::hints::parse_filename_hints(&path_str);
                            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                            let (w, h, fmt) = if ext == "y4m" {
                                (0, 0, String::new())
                            } else {
                                (
                                    hints.width.unwrap_or(self.settings.defaults.width),
                                    hints.height.unwrap_or(self.settings.defaults.height),
                                    hints.format.unwrap_or_else(|| self.settings.defaults.format.clone()),
                                )
                            };
                            self.open_file(ctx, path_str, w, h, &fmt);
                        }
                    }
                    if ui.button("Save Frame (Ctrl+S)").clicked() {
                        ui.close_menu();
                        self.save_frame_as_png();
                    }
                    if ui.button("Export Clip...").clicked() {
                        ui.close_menu();
                        self.export_dialog = Some(dialogs::ExportDialog::new(self.total_frames()));
                        self.dialog_state = DialogState::Export;
                    }
                    if ui.button("Export PNG Sequence...").clicked() {
                        ui.close_menu();
                        self.png_export_dialog = Some(dialogs::PngExportDialog::new(self.total_frames()));
                        self.dialog_state = DialogState::PngExport;
                    }
                    ui.separator();
                    // Recent files
                    if !self.settings.recent_files.is_empty() {
                        ui.menu_button("Recent Files", |ui| {
                            let recent = self.settings.recent_files.clone();
                            for path in &recent {
                                if ui.button(path).clicked() {
                                    let hints = crate::core::hints::parse_filename_hints(path);
                                    let (w, h, fmt) = (
                                        hints.width.unwrap_or(self.settings.defaults.width),
                                        hints.height.unwrap_or(self.settings.defaults.height),
                                        hints.format.unwrap_or_else(|| self.settings.defaults.format.clone()),
                                    );
                                    self.open_file(ctx, path.clone(), w, h, &fmt);
                                    ui.close_menu();
                                }
                            }
                        });
                    }
                    ui.separator();
                    if ui.button("Settings...").clicked() {
                        ui.close_menu();
                        self.settings_dialog = Some(dialogs::SettingsDialog::new(&self.settings));
                        self.dialog_state = DialogState::Settings;
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui.button("Fit to View (F)").clicked() {
                        let avail = ctx.available_rect().size();
                        self.canvas.fit_to_view(avail);
                        ui.close_menu();
                    }
                    if ui.button("1:1 Zoom").clicked() {
                        self.canvas.zoom = 1.0;
                        ui.close_menu();
                    }
                    if ui.button("2:1 Zoom").clicked() {
                        self.canvas.zoom = 2.0;
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.checkbox(&mut self.loop_playback, "Loop Playback");
                    ui.checkbox(&mut self.sidebar.show_analysis, "Show Analysis");
                    ui.separator();
                    let mut dark = self.settings.display.dark_theme;
                    if ui.checkbox(&mut dark, "Dark Theme").changed() {
                        self.settings.display.dark_theme = dark;
                        if dark {
                            ctx.set_visuals(egui::Visuals::dark());
                        } else {
                            ctx.set_visuals(egui::Visuals::light());
                        }
                        self.settings.save();
                    }
                });
                ui.menu_button("Tools", |ui| {
                    if ui.button("Video Parameters...").clicked() {
                        ui.close_menu();
                        if let Some(ref r) = self.reader {
                            self.params_dialog = Some(dialogs::ParametersDialog::new(
                                r.width(),
                                r.height(),
                                r.format_name(),
                            ));
                        } else {
                            self.params_dialog = Some(dialogs::ParametersDialog::new(
                                self.settings.defaults.width,
                                self.settings.defaults.height,
                                &self.settings.defaults.format,
                            ));
                        }
                        self.dialog_state = DialogState::Parameters;
                    }
                    if ui.button("Convert...").clicked() {
                        ui.close_menu();
                        let info = if let (Some(ref path), Some(ref r)) = (&self.current_file, &self.reader) {
                            format!("{} ({}x{} {})", path, r.width(), r.height(), r.format_name())
                        } else {
                            "No file loaded".to_string()
                        };
                        self.convert_dialog = Some(dialogs::ConvertDialog::new(&info));
                        self.dialog_state = DialogState::Convert;
                    }
                    if ui.button("Copy Frame (Ctrl+C)").clicked() {
                        ui.close_menu();
                        self.copy_frame_to_clipboard();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("Keyboard Shortcuts").clicked() {
                        ui.close_menu();
                        self.show_shortcuts = true;
                    }
                    if ui.button("About").clicked() {
                        ui.close_menu();
                        self.show_about = true;
                    }
                });
            });
        });

        // --- Toolbar ---
        let is_yuv = self
            .reader
            .as_ref()
            .map(|r| {
                let ft = r.format().format_type;
                matches!(
                    ft,
                    crate::core::formats::FormatType::YuvPlanar
                        | crate::core::formats::FormatType::YuvSemiPlanar
                        | crate::core::formats::FormatType::YuvPacked
                )
            })
            .unwrap_or(true);
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            if let Some(act) = self.toolbar.show(ui, is_yuv) {
                match act {
                    ToolbarAction::SetComponent(c) => {
                        self.current_component = c;
                        if self.reader.is_some() {
                            self.goto_frame(ctx, self.current_frame_idx);
                        }
                    }
                    ToolbarAction::ToggleGrid => {
                        self.canvas.set_grid_size(self.toolbar.grid_size);
                    }
                    ToolbarAction::ToggleSubGrid => {
                        self.canvas.set_sub_grid_size(self.toolbar.sub_grid_size);
                    }
                    ToolbarAction::FitToView => {
                        let avail = ctx.available_rect().size();
                        self.canvas.fit_to_view(avail);
                    }
                    ToolbarAction::ZoomIn => {
                        self.canvas.zoom = (self.canvas.zoom * 1.25).min(50.0);
                    }
                    ToolbarAction::ZoomOut => {
                        self.canvas.zoom = (self.canvas.zoom / 1.25).max(0.1);
                    }
                    ToolbarAction::Zoom1to1 => {
                        self.canvas.zoom = 1.0;
                    }
                    ToolbarAction::Zoom2to1 => {
                        self.canvas.zoom = 2.0;
                    }
                }
            }
        });

        // --- Navigation bar ---
        let total = self.total_frames();
        let cur = self.current_frame_idx;
        let is_playing = self.is_playing;
        egui::TopBottomPanel::bottom("navigation").show(ctx, |ui| {
            let action = self.nav.show(ui, cur, total, is_playing);
            if let Some(act) = action {
                match act {
                    NavigationAction::Seek(idx) => {
                        self.goto_frame(ctx, idx);
                        self.is_playing = false;
                    }
                    NavigationAction::TogglePlay => {
                        self.is_playing = !self.is_playing;
                        if self.is_playing {
                            self.last_frame_time = Some(Instant::now());
                        }
                    }
                    NavigationAction::NextFrame => {
                        let next = self.current_frame_idx + 1;
                        if next < total { self.goto_frame(ctx, next); }
                        self.is_playing = false;
                    }
                    NavigationAction::PrevFrame => {
                        let cur = self.current_frame_idx;
                        if cur > 0 { self.goto_frame(ctx, cur - 1); }
                        self.is_playing = false;
                    }
                    NavigationAction::FirstFrame => {
                        self.goto_frame(ctx, 0);
                        self.is_playing = false;
                    }
                    NavigationAction::LastFrame => {
                        let last = total.saturating_sub(1);
                        self.goto_frame(ctx, last);
                        self.is_playing = false;
                    }
                    NavigationAction::SetFps(_) => {}
                }
            }
        });

        // --- Status bar ---
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(ref err) = self.status_error {
                    ui.colored_label(egui::Color32::RED, err);
                } else if let Some(ref path) = self.current_file {
                    if let Some(ref r) = self.reader {
                        ui.label(format!(
                            "{}  |  {}x{}  {}  |  {:.0}%  |  Frame {}/{}",
                            path,
                            r.width(),
                            r.height(),
                            r.format_name(),
                            self.canvas.zoom_level() * 100.0,
                            self.current_frame_idx,
                            r.total_frames().saturating_sub(1),
                        ));
                    }
                } else {
                    ui.label("No file loaded — drop a file or use File → Open");
                }
            });
        });

        // --- Sidebar (right panel) ---
        egui::SidePanel::right("sidebar")
            .default_width(250.0)
            .show(ctx, |ui| {
                self.sidebar.show(ui);
            });

        // --- Central panel (canvas) ---
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.reader.is_some() {
                let response = self.canvas.show(ui);
                // Update pixel info on hover
                if let Some(hover_pos) = response.hover_pos() {
                    let image_origin = response.rect.min;
                    let rel = egui::pos2(
                        hover_pos.x - image_origin.x,
                        hover_pos.y - image_origin.y,
                    );
                    if let Some((ix, iy)) = self.canvas.image_pos_from_screen(rel) {
                        if let (Some(ref raw), Some(ref reader)) = (&self.current_raw, &self.reader) {
                            let info = crate::core::pixel::get_pixel_info(
                                raw,
                                reader.width(),
                                reader.height(),
                                reader.format(),
                                ix,
                                iy,
                                self.toolbar.sub_grid_size,
                            );
                            self.sidebar.set_pixel_info(Some(info));
                        }
                    }
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Drop a file or use File \u{2192} Open");
                });
            }
        });

        // --- Dialogs ---
        self.show_dialogs(ctx);
    }
}

// Dialog rendering (separate impl block to keep update() manageable).
impl VideoViewerApp {
    fn show_dialogs(&mut self, ctx: &egui::Context) {
        // Parameters dialog
        if self.dialog_state == DialogState::Parameters {
            if let Some(ref mut dlg) = self.params_dialog {
                if let Some(result) = dlg.show(ctx) {
                    if let Some((w, h, fmt)) = result {
                        if let Some(ref path) = self.current_file.clone() {
                            self.open_file(ctx, path.clone(), w, h, &fmt);
                        }
                    }
                    self.dialog_state = DialogState::None;
                    self.params_dialog = None;
                }
            }
        }

        // Export dialog
        if self.dialog_state == DialogState::Export {
            if let Some(ref mut dlg) = self.export_dialog {
                if let Some(result) = dlg.show(ctx) {
                    if let Some((start, end, path)) = result {
                        self.export_clip(start, end, &path);
                    }
                    self.dialog_state = DialogState::None;
                    self.export_dialog = None;
                }
            }
        }

        // PNG export dialog
        if self.dialog_state == DialogState::PngExport {
            if let Some(ref mut dlg) = self.png_export_dialog {
                if let Some(result) = dlg.show(ctx) {
                    if let Some((start, end, dir, prefix)) = result {
                        self.export_png_sequence(ctx, start, end, &dir, &prefix);
                    }
                    self.dialog_state = DialogState::None;
                    self.png_export_dialog = None;
                }
            }
        }

        // Settings dialog
        if self.dialog_state == DialogState::Settings {
            if let Some(ref mut dlg) = self.settings_dialog {
                if let Some(ok) = dlg.show(ctx) {
                    if ok {
                        self.settings.cache.max_memory_mb = dlg.max_memory_mb;
                        self.settings.display.zoom_min = dlg.zoom_min;
                        self.settings.display.zoom_max = dlg.zoom_max;
                        self.settings.display.dark_theme = dlg.dark_theme;
                        self.settings.defaults.fps = dlg.default_fps;
                        self.settings.defaults.width = dlg.default_width;
                        self.settings.defaults.height = dlg.default_height;
                        self.settings.defaults.color_matrix = dlg.color_matrix.clone();
                        self.settings.save();
                        if dlg.dark_theme {
                            ctx.set_visuals(egui::Visuals::dark());
                        } else {
                            ctx.set_visuals(egui::Visuals::light());
                        }
                    }
                    self.dialog_state = DialogState::None;
                    self.settings_dialog = None;
                }
            }
        }

        // Shortcuts dialog
        if self.show_shortcuts {
            self.show_shortcuts = dialogs::show_shortcuts_dialog(ctx);
        }

        // About dialog
        if self.show_about {
            self.show_about = dialogs::show_about_dialog(ctx);
        }
    }

    fn export_clip(&mut self, start: usize, end: usize, path: &str) {
        let reader = match self.reader.as_mut() {
            Some(r) => r,
            None => return,
        };
        let mut file = match std::fs::File::create(path) {
            Ok(f) => f,
            Err(e) => {
                self.status_error = Some(format!("Export error: {e}"));
                return;
            }
        };
        use std::io::Write;
        for idx in start..=end {
            match reader.seek_frame(idx) {
                Ok(raw) => {
                    if let Err(e) = file.write_all(&raw) {
                        self.status_error = Some(format!("Write error at frame {idx}: {e}"));
                        return;
                    }
                }
                Err(e) => {
                    self.status_error = Some(format!("Seek error at frame {idx}: {e}"));
                    return;
                }
            }
        }
        self.status_error = None;
    }

    fn export_png_sequence(&mut self, _ctx: &egui::Context, start: usize, end: usize, dir: &str, prefix: &str) {
        let reader = match self.reader.as_mut() {
            Some(r) => r,
            None => return,
        };
        let w = reader.width();
        let h = reader.height();
        for idx in start..=end {
            match reader.seek_frame(idx) {
                Ok(raw) => match reader.convert_to_rgb(&raw) {
                    Ok(rgb) => {
                        let path = format!("{}/{}_{:06}.png", dir, prefix, idx);
                        if let Some(img) = image::RgbImage::from_raw(w, h, rgb) {
                            if let Err(e) = img.save(&path) {
                                self.status_error = Some(format!("PNG save error: {e}"));
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        self.status_error = Some(format!("Convert error at frame {idx}: {e}"));
                        return;
                    }
                },
                Err(e) => {
                    self.status_error = Some(format!("Seek error at frame {idx}: {e}"));
                    return;
                }
            }
        }
        self.status_error = None;
    }
}
