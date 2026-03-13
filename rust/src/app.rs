use std::time::Instant;
use eframe::egui;

use crate::core::reader::VideoReader;
use crate::ui::canvas::ImageCanvas;
use crate::ui::navigation::{NavigationBar, NavigationAction};
use crate::ui::toolbar::{Toolbar, ToolbarAction};

pub struct VideoViewerApp {
    pub current_file: Option<String>,
    pub reader: Option<VideoReader>,
    pub canvas: ImageCanvas,
    pub toolbar: Toolbar,
    pub nav: NavigationBar,
    pub current_frame_idx: usize,
    pub is_playing: bool,
    pub loop_playback: bool,
    pub last_frame_time: Option<Instant>,
    pub current_component: u8,
    /// CLI-provided args for auto-open on startup.
    startup_input: Option<String>,
    startup_width: Option<u32>,
    startup_height: Option<u32>,
    startup_format: Option<String>,
    /// Error message to show in the status bar.
    status_error: Option<String>,
}

impl VideoViewerApp {
    pub fn new(
        _cc: &eframe::CreationContext<'_>,
        input: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        format: Option<String>,
    ) -> Self {
        Self {
            current_file: None,
            reader: None,
            canvas: ImageCanvas::new(),
            toolbar: Toolbar::new(),
            nav: NavigationBar::new(),
            current_frame_idx: 0,
            is_playing: false,
            loop_playback: true,
            last_frame_time: None,
            current_component: 0,
            startup_input: input,
            startup_width: width,
            startup_height: height,
            startup_format: format,
            status_error: None,
        }
    }

    /// Open a file by path. Uses Y4M auto-detection; falls back to provided
    /// width/height/format or a 1920×1080 I420 default for raw files.
    fn open_file(
        &mut self,
        ctx: &egui::Context,
        path: String,
        width: u32,
        height: u32,
        format: &str,
    ) {
        let color_matrix = "BT.601";
        match VideoReader::open(&path, width, height, format, color_matrix) {
            Ok(mut reader) => {
                self.current_frame_idx = 0;
                // Apply Y4M fps to nav bar if available.
                if let Some(fps) = reader.y4m_fps() {
                    let fps_u32 = fps.round() as u32;
                    use crate::ui::navigation::FPS_OPTIONS;
                    if let Some(idx) = FPS_OPTIONS.iter().position(|&f| f == fps_u32) {
                        self.nav.selected_fps_idx = idx;
                    }
                }
                // Decode and display first frame.
                match reader.seek_frame(0) {
                    Ok(raw) => match reader.convert_to_rgb(&raw) {
                        Ok(rgb) => {
                            self.canvas.set_image(ctx, &rgb, reader.width(), reader.height());
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

    /// Seek to `idx`, decode, and push to canvas. Returns false on error.
    fn goto_frame(&mut self, ctx: &egui::Context, idx: usize) -> bool {
        let total = self
            .reader
            .as_ref()
            .map(|r| r.total_frames())
            .unwrap_or(0);
        if total == 0 {
            return false;
        }
        let idx = idx.min(total - 1);
        let reader = match self.reader.as_mut() {
            Some(r) => r,
            None => return false,
        };
        match reader.seek_frame(idx) {
            Ok(raw) => match reader.convert_to_rgb(&raw) {
                Ok(rgb) => {
                    let (w, h) = (reader.width(), reader.height());
                    self.canvas.set_image(ctx, &rgb, w, h);
                    self.current_frame_idx = idx;
                    self.status_error = None;
                    true
                }
                Err(e) => {
                    self.status_error = Some(format!("Convert error: {e}"));
                    false
                }
            },
            Err(e) => {
                self.status_error = Some(format!("Seek error: {e}"));
                false
            }
        }
    }

    fn total_frames(&self) -> usize {
        self.reader.as_ref().map(|r| r.total_frames()).unwrap_or(0)
    }
}

impl eframe::App for VideoViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- Auto-open from CLI args (first frame only) ---
        if let Some(path) = self.startup_input.take() {
            let w = self.startup_width.unwrap_or(1920);
            let h = self.startup_height.unwrap_or(1080);
            let fmt = self
                .startup_format
                .clone()
                .unwrap_or_else(|| "I420".to_string());
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
                i.key_pressed(egui::Key::Space),
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                i.key_pressed(egui::Key::Home),
                i.key_pressed(egui::Key::End),
                i.key_pressed(egui::Key::F),
            )
        });

        if keys.0 {
            // Space: toggle play
            self.is_playing = !self.is_playing;
            if self.is_playing {
                self.last_frame_time = Some(Instant::now());
            }
        }
        if keys.1 {
            // Left arrow: prev frame
            let cur = self.current_frame_idx;
            if cur > 0 {
                self.goto_frame(ctx, cur - 1);
            }
            self.is_playing = false;
        }
        if keys.2 {
            // Right arrow: next frame
            let total = self.total_frames();
            let next = self.current_frame_idx + 1;
            if next < total {
                self.goto_frame(ctx, next);
            }
            self.is_playing = false;
        }
        if keys.3 {
            // Home: first frame
            self.goto_frame(ctx, 0);
            self.is_playing = false;
        }
        if keys.4 {
            // End: last frame
            let last = self.total_frames().saturating_sub(1);
            self.goto_frame(ctx, last);
            self.is_playing = false;
        }
        if keys.5 {
            // F: fit to view
            let avail = ctx.available_rect().size();
            self.canvas.fit_to_view(avail);
        }

        // --- Menu bar ---
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Video", &["yuv", "y4m", "rgb", "raw", "nv12", "nv21"])
                            .pick_file()
                        {
                            let path_str = path.display().to_string();
                            let ext = path
                                .extension()
                                .and_then(|e| e.to_str())
                                .unwrap_or("")
                                .to_lowercase();
                            let (w, h, fmt) = if ext == "y4m" {
                                (0, 0, "".to_string())
                            } else {
                                (1920, 1080, "I420".to_string())
                            };
                            self.open_file(ctx, path_str, w, h, &fmt);
                        }
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui.button("Fit to View").clicked() {
                        let avail = ctx.available_rect().size();
                        self.canvas.fit_to_view(avail);
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.checkbox(&mut self.loop_playback, "Loop Playback");
                });
                ui.menu_button("Tools", |_ui| {});
                ui.menu_button("Help", |ui| {
                    if ui.button("About").clicked() {
                        ui.close_menu();
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

        // --- Navigation bar (above status bar) ---
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
                        if next < total {
                            self.goto_frame(ctx, next);
                        }
                        self.is_playing = false;
                    }
                    NavigationAction::PrevFrame => {
                        let cur = self.current_frame_idx;
                        if cur > 0 {
                            self.goto_frame(ctx, cur - 1);
                        }
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
                    NavigationAction::SetFps(_fps) => {
                        // selected_fps_idx already updated inside nav.show()
                    }
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
                            "{}  |  {}x{}  {}  |  {:.0}%",
                            path,
                            r.width(),
                            r.height(),
                            r.format_name(),
                            self.canvas.zoom_level() * 100.0,
                        ));
                    } else {
                        ui.label(format!("File: {}", path));
                    }
                } else {
                    ui.label("No file loaded");
                }
            });
        });

        // --- Central panel (canvas) ---
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.reader.is_some() {
                self.canvas.show(ui);
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Drop a file or use File \u{2192} Open");
                });
            }
        });
    }
}
