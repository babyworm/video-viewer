use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;
use eframe::egui;

use crate::core::reader::VideoReader;
use crate::ui::canvas::ImageCanvas;
use crate::ui::dialogs::{self, DialogState};
use crate::ui::navigation::{NavigationBar, NavigationAction};
use crate::ui::settings::Settings;
use crate::ui::sidebar::Sidebar;
use crate::ui::toolbar::{Toolbar, ToolbarAction, colorize_channel};

/// Tagged scene detection output: (job_id, Ok(changes) | Err(message)).
type SceneDetectOutput = Arc<std::sync::Mutex<Option<(usize, Result<Vec<usize>, String>)>>>;

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
    /// Previous frame RGB (for metrics: PSNR, SSIM, frame diff).
    prev_rgb: Option<Vec<u8>>,
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
    open_file_dialog: Option<dialogs::OpenFileDialog>,
    save_file_dialog: Option<dialogs::SaveFileDialog>,
    params_dialog: Option<dialogs::ParametersDialog>,
    export_dialog: Option<dialogs::ExportDialog>,
    convert_dialog: Option<dialogs::ConvertDialog>,
    png_export_dialog: Option<dialogs::PngExportDialog>,
    settings_dialog: Option<dialogs::SettingsDialog>,
    batch_convert_dialog: Option<dialogs::BatchConvertDialog>,
    show_shortcuts: bool,
    show_about: bool,
    /// Timestamps of actual YUV frame changes for playback FPS calculation.
    playback_frame_times: VecDeque<Instant>,
    playback_fps: f64,
    /// Auto-fit mode: automatically fit image when window resizes.
    auto_fit: bool,
    /// Last known available size for auto-fit detection.
    last_available_size: Option<(f32, f32)>,
    /// Background conversion progress tracking.
    convert_progress_current: Arc<AtomicUsize>,
    convert_progress_total: Arc<AtomicUsize>,
    convert_done: Arc<AtomicBool>,
    convert_error: Arc<std::sync::Mutex<Option<String>>>,
    convert_running: bool,
    /// Scene change indices detected by analysis.
    scene_changes: Vec<usize>,
    /// Whether scene detection is currently running in the background.
    scene_detect_running: bool,
    /// Tagged result slot: (job_id, Ok(changes) | Err(message)).
    scene_detect_output: SceneDetectOutput,
    /// Monotonic job ID counter — incremented for each detection run.
    scene_detect_job_id: usize,
    /// The job ID of the currently expected detection run (shared with workers).
    scene_detect_active_job: Arc<AtomicUsize>,
    /// Show the scene detection settings dialog.
    show_scene_detect_dialog: bool,
    scene_detect_algorithm: crate::analysis::scene::SceneAlgorithm,
    scene_detect_threshold: f64,
    /// Test video download state.
    test_download_status: Arc<std::sync::Mutex<Option<String>>>,
    test_download_path: Arc<std::sync::Mutex<Option<String>>>,
    test_downloading: bool,
    test_dl_current: Arc<AtomicUsize>,
    test_dl_total: Arc<AtomicUsize>,
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
            prev_rgb: None,
            startup_input: input,
            startup_width: width,
            startup_height: height,
            startup_format: format,
            status_error: None,
            dialog_state: DialogState::None,
            open_file_dialog: None,
            save_file_dialog: None,
            params_dialog: None,
            export_dialog: None,
            convert_dialog: None,
            png_export_dialog: None,
            settings_dialog: None,
            batch_convert_dialog: None,
            show_shortcuts: false,
            show_about: false,
            playback_frame_times: VecDeque::new(),
            playback_fps: 0.0,
            auto_fit: true,
            last_available_size: None,
            convert_progress_current: Arc::new(AtomicUsize::new(0)),
            convert_progress_total: Arc::new(AtomicUsize::new(0)),
            convert_done: Arc::new(AtomicBool::new(false)),
            convert_error: Arc::new(std::sync::Mutex::new(None)),
            convert_running: false,
            scene_changes: Vec::new(),
            scene_detect_running: false,
            scene_detect_output: Arc::new(std::sync::Mutex::new(None)),
            scene_detect_job_id: 0,
            scene_detect_active_job: Arc::new(AtomicUsize::new(0)),
            show_scene_detect_dialog: false,
            scene_detect_algorithm: crate::analysis::scene::SceneAlgorithm::Mad,
            scene_detect_threshold: 45.0,
            test_download_status: Arc::new(std::sync::Mutex::new(None)),
            test_download_path: Arc::new(std::sync::Mutex::new(None)),
            test_downloading: false,
            test_dl_current: Arc::new(AtomicUsize::new(0)),
            test_dl_total: Arc::new(AtomicUsize::new(0)),
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
            Ok(reader) => {
                self.current_frame_idx = 0;
                self.bookmarks.clear();
                if let Some(fps) = reader.y4m_fps() {
                    let fps_u32 = fps.round() as u32;
                    use crate::ui::navigation::FPS_OPTIONS;
                    if let Some(idx) = FPS_OPTIONS.iter().position(|&f| f == fps_u32) {
                        self.nav.selected_fps_idx = idx;
                    }
                }
                self.settings.add_recent_file(&path);
                self.settings.save();
                self.current_file = Some(path);
                self.reader = Some(reader);
                self.is_playing = false;
                self.last_frame_time = None;
                // Clear stale frame data to prevent cross-file metrics.
                self.prev_rgb = None;
                self.current_rgb = None;
                self.current_raw = None;
                // Invalidate scene changes from a previous file.
                // Any in-flight detection will be discarded by the job ID check.
                self.scene_changes.clear();
                self.scene_detect_active_job.store(0, Ordering::Relaxed);
                self.scene_detect_running = false;
                // Use goto_frame so component view is applied correctly.
                self.goto_frame(ctx, 0);
                // Apply auto-fit immediately so the image fills the canvas.
                if self.auto_fit {
                    let avail = ctx.available_rect().size();
                    self.canvas.fit_to_view(avail);
                    self.last_available_size = None; // force re-detect on next frame
                }
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
        // Only keep prev_rgb for sequential navigation (metrics compare adjacent frames).
        if idx == self.current_frame_idx + 1 {
            self.prev_rgb = self.current_rgb.take();
        } else {
            self.prev_rgb = None;
            drop(self.current_rgb.take());
        }
        self.current_rgb = Some(rgb);
        self.current_raw = Some(raw);
        self.current_frame_idx = idx;
        self.status_error = None;
        // Track playback FPS (actual frame change rate).
        let now = Instant::now();
        self.playback_frame_times.push_back(now);
        while self.playback_frame_times.len() > 30 {
            self.playback_frame_times.pop_front();
        }
        if self.playback_frame_times.len() >= 2 {
            let elapsed = self.playback_frame_times.back().unwrap()
                .duration_since(*self.playback_frame_times.front().unwrap());
            let secs = elapsed.as_secs_f64();
            if secs > 0.0 {
                self.playback_fps = (self.playback_frame_times.len() - 1) as f64 / secs;
            }
        }
        // Update analysis visualizations.
        self.update_analysis(ctx);
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

    /// Handle window edge resize (custom, since OS decorations are disabled).
    fn handle_window_resize(&self, ctx: &egui::Context) {
        use egui::viewport::ResizeDirection;

        let is_max = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
        if is_max {
            return; // No resize when maximized.
        }

        let screen = ctx.screen_rect();
        let border = 5.0_f32;

        if let Some(pos) = ctx.input(|i| i.pointer.hover_pos()) {
            let left = pos.x - screen.min.x < border;
            let right = screen.max.x - pos.x < border;
            let top = pos.y - screen.min.y < border;
            let bottom = screen.max.y - pos.y < border;

            let dir = match (left, right, top, bottom) {
                (true, _, true, _) => Some(ResizeDirection::NorthWest),
                (_, true, true, _) => Some(ResizeDirection::NorthEast),
                (true, _, _, true) => Some(ResizeDirection::SouthWest),
                (_, true, _, true) => Some(ResizeDirection::SouthEast),
                (true, _, _, _) => Some(ResizeDirection::West),
                (_, true, _, _) => Some(ResizeDirection::East),
                (_, _, true, _) => Some(ResizeDirection::North),
                (_, _, _, true) => Some(ResizeDirection::South),
                _ => None,
            };

            if let Some(direction) = dir {
                let cursor = match direction {
                    ResizeDirection::North | ResizeDirection::South => egui::CursorIcon::ResizeVertical,
                    ResizeDirection::East | ResizeDirection::West => egui::CursorIcon::ResizeHorizontal,
                    ResizeDirection::NorthWest | ResizeDirection::SouthEast => egui::CursorIcon::ResizeNwSe,
                    ResizeDirection::NorthEast | ResizeDirection::SouthWest => egui::CursorIcon::ResizeNeSw,
                };
                ctx.set_cursor_icon(cursor);

                if ctx.input(|i| i.pointer.any_pressed()) {
                    ctx.send_viewport_cmd(egui::ViewportCommand::BeginResize(direction));
                }
            }
        }
    }

    /// Compute analysis data from the current RGB frame and feed it to the
    /// shared analysis state (consumed by the separate analysis viewport).
    /// Only computes data for the currently active tab to avoid unnecessary work.
    fn update_analysis(&mut self, ctx: &egui::Context) {
        use crate::ui::sidebar::AnalysisTab;

        if !self.sidebar.show_analysis {
            return;
        }
        let rgb = match &self.current_rgb {
            Some(r) => r,
            None => return,
        };
        let (w, h) = match &self.reader {
            Some(r) => (r.width(), r.height()),
            None => return,
        };

        let active_tab = self.sidebar.analysis.lock().active_tab;

        match active_tab {
            AnalysisTab::Histogram => {
                let hist_u32 = crate::analysis::histogram::calculate_histogram(rgb, w, h, "RGB");
                let hist_f64: std::collections::HashMap<String, Vec<f64>> = hist_u32
                    .into_iter()
                    .map(|(k, v)| (k, v.into_iter().map(|c| c as f64).collect()))
                    .collect();
                let mut shared = self.sidebar.analysis.lock();
                shared.histogram_data = Some(hist_f64);
                shared.generation += 1;
            }
            AnalysisTab::Vectorscope => {
                let (cb, cr) = crate::analysis::vectorscope::calculate_vectorscope(rgb, w, h);
                let max_points = 50_000;
                let step = if cb.len() > max_points { cb.len() / max_points } else { 1 };
                let scatter: Vec<[f64; 2]> = cb
                    .into_iter()
                    .zip(cr)
                    .step_by(step)
                    .map(|(u, v)| [u as f64, v as f64])
                    .collect();
                let mut shared = self.sidebar.analysis.lock();
                shared.vectorscope_data = Some(scatter);
                shared.generation += 1;
            }
            AnalysisTab::Waveform => {
                let wf = crate::analysis::waveform::calculate_waveform(rgb, w, h, "luma");
                let wf_width = wf.first().map(|r| r.len()).unwrap_or(0);
                let wf_height = wf.len();
                if wf_width > 0 {
                    let max_count = wf.iter().flat_map(|row| row.iter()).copied().max().unwrap_or(1).max(1);
                    let mut pixels = vec![0u8; wf_width * wf_height * 3];
                    for level in 0..wf_height {
                        let src_row = wf_height - 1 - level;
                        for (col, &count) in wf[src_row].iter().enumerate().take(wf_width) {
                            let intensity = ((count as f64 / max_count as f64).sqrt() * 255.0) as u8;
                            let idx = (level * wf_width + col) * 3;
                            pixels[idx] = intensity / 3;
                            pixels[idx + 1] = intensity;
                            pixels[idx + 2] = intensity / 3;
                        }
                    }
                    let color_image = egui::ColorImage::from_rgb([wf_width, wf_height], &pixels);
                    let mut shared = self.sidebar.analysis.lock();
                    shared.waveform_image = Some(color_image);
                    shared.waveform_data_gen += 1;
                    shared.generation += 1;
                }
            }
            AnalysisTab::Metrics => {
                let mut shared = self.sidebar.analysis.lock();
                if let Some(ref prev) = self.prev_rgb {
                    if prev.len() == rgb.len() {
                        shared.psnr = Some(crate::analysis::metrics::calculate_psnr(prev, rgb, w, h));
                        shared.ssim = Some(crate::analysis::metrics::calculate_ssim(prev, rgb, w, h));
                        shared.frame_diff = Some(crate::analysis::metrics::calculate_frame_difference(prev, rgb));
                    } else {
                        shared.psnr = None;
                        shared.ssim = None;
                        shared.frame_diff = None;
                    }
                } else {
                    shared.psnr = None;
                    shared.ssim = None;
                    shared.frame_diff = None;
                }
                shared.generation += 1;
            }
        }

        // Request repaint of the analysis viewport so it picks up new data.
        ctx.request_repaint_of(egui::ViewportId::from_hash_of("analysis_viewport"));
    }

    fn total_frames(&self) -> usize {
        self.reader.as_ref().map(|r| r.total_frames()).unwrap_or(0)
    }

    fn save_frame_as_png_to(&mut self, path: &str) {
        if let (Some(rgb), Some(reader)) = (&self.current_rgb, &self.reader) {
            let w = reader.width();
            let h = reader.height();
            if let Some(img) = image::RgbImage::from_raw(w, h, rgb.clone()) {
                if let Err(e) = img.save(path) {
                    self.status_error = Some(format!("PNG save error: {e}"));
                } else {
                    self.status_error = None;
                }
            } else {
                self.status_error = Some(format!(
                    "PNG save error: RGB buffer size mismatch ({}x{}, {} bytes)",
                    w, h, rgb.len()
                ));
            }
        } else {
            self.status_error = Some("No frame loaded to save".to_string());
        }
    }

    fn run_batch_single(&mut self, input: &str, width: u32, height: u32, out_format: &str, out_path: &str) {
        use crate::core::reader::VideoReader;
        use crate::conversion::converter::VideoConverter;

        let color_matrix = &self.settings.defaults.color_matrix;
        let reader = match VideoReader::open(input, width, height, "I420", color_matrix) {
            Ok(r) => r,
            Err(e) => {
                self.status_error = Some(format!("Batch: failed to open {input}: {e}"));
                return;
            }
        };
        let src_format = reader.format_name().to_string();
        let converter = VideoConverter::new();
        match converter.convert(
            input,
            (reader.width(), reader.height()),
            &src_format,
            out_path,
            out_format,
            None,
        ) {
            Ok(_) => {}
            Err(e) => {
                self.status_error = Some(format!("Batch: conversion failed for {input}: {e}"));
            }
        }
    }

    fn run_scene_detection(&mut self, ctx: &egui::Context) {
        let reader = match self.reader.as_ref() {
            Some(r) => r,
            None => return,
        };
        if reader.total_frames() < 2 {
            return;
        }
        // Capture file info so the background thread can open its own reader.
        let path = match &self.current_file {
            Some(p) => p.clone(),
            None => return,
        };
        let w = reader.width();
        let h = reader.height();
        let fmt = reader.format_name().to_string();
        // Use the active reader's color matrix, not the current default setting.
        let color_matrix = reader.color_matrix.clone();
        let algo = self.scene_detect_algorithm;
        let threshold = self.scene_detect_threshold;
        let output = Arc::clone(&self.scene_detect_output);
        let active_job = Arc::clone(&self.scene_detect_active_job);
        // Assign a unique job ID so stale completions are ignored.
        // Store under output lock so any in-flight publisher sees the new active_job.
        self.scene_detect_job_id += 1;
        let job_id = self.scene_detect_job_id;
        {
            let _lock = output.lock().unwrap();
            active_job.store(job_id, Ordering::Release);
        }
        self.status_error = None; // Clear prior error banner
        let ctx2 = ctx.clone();
        // Helper: only publish if this job is still the active one.
        // The check is inside the lock to prevent TOCTOU races between workers.
        let publish = move |out: &SceneDetectOutput,
                            active: &Arc<AtomicUsize>,
                            id: usize,
                            val: Result<Vec<usize>, String>,
                            ctx: &egui::Context| {
            let mut slot = out.lock().unwrap();
            if active.load(Ordering::Acquire) == id {
                *slot = Some((id, val));
                drop(slot);
                ctx.request_repaint();
            }
        };
        std::thread::spawn(move || {
            // Open a separate reader in the background thread to avoid blocking UI.
            let mut bg_reader = match crate::core::reader::VideoReader::open(
                &path, w, h, &fmt, &color_matrix,
            ) {
                Ok(r) => r,
                Err(e) => {
                    publish(&output, &active_job, job_id, Err(format!("Scene detect: failed to open file: {e}")), &ctx2);
                    return;
                }
            };
            let total = bg_reader.total_frames();
            // Process frames pair-wise to avoid loading all into memory at once.
            let mut changes = Vec::new();
            let mut prev_rgb: Option<Vec<u8>> = None;
            for i in 0..total {
                // Early exit if this job has been superseded.
                if active_job.load(Ordering::Acquire) != job_id {
                    return;
                }
                let rgb = match bg_reader.seek_frame(i)
                    .and_then(|raw| bg_reader.convert_to_rgb(&raw))
                {
                    Ok(r) => r,
                    Err(e) => {
                        publish(&output, &active_job, job_id, Err(format!("Scene detect: failed to read frame {i}: {e}")), &ctx2);
                        return;
                    }
                };
                if let Some(ref prev) = prev_rgb {
                    let diff = match algo {
                        crate::analysis::scene::SceneAlgorithm::Mad =>
                            crate::analysis::scene::mean_abs_diff(prev, &rgb),
                        crate::analysis::scene::SceneAlgorithm::Histogram =>
                            crate::analysis::scene::histogram_diff(prev, &rgb),
                        crate::analysis::scene::SceneAlgorithm::Ssim =>
                            crate::analysis::scene::ssim_diff(prev, &rgb, w, h),
                    };
                    if diff > threshold {
                        changes.push(i);
                    }
                }
                prev_rgb = Some(rgb);
            }
            publish(&output, &active_job, job_id, Ok(changes), &ctx2);
        });
        self.scene_detect_running = true;
    }

    fn run_conversion(&mut self, out_format: &str, out_path: &str) {
        let reader = match self.reader.as_ref() {
            Some(r) => r,
            None => {
                self.status_error = Some("No file loaded".to_string());
                return;
            }
        };
        let in_path = match &self.current_file {
            Some(p) => p.clone(),
            None => return,
        };
        let src_fmt = reader.format_name().to_string();
        let w = reader.width();
        let h = reader.height();
        let out_format = out_format.to_string();
        let out_path = out_path.to_string();

        // Reset progress
        self.convert_progress_current.store(0, Ordering::Relaxed);
        self.convert_progress_total.store(0, Ordering::Relaxed);
        self.convert_done.store(false, Ordering::Relaxed);
        *self.convert_error.lock().unwrap() = None;
        self.convert_running = true;

        let progress_current = Arc::clone(&self.convert_progress_current);
        let progress_total = Arc::clone(&self.convert_progress_total);
        let done = Arc::clone(&self.convert_done);
        let error = Arc::clone(&self.convert_error);

        std::thread::spawn(move || {
            use crate::conversion::converter::VideoConverter;
            let converter = VideoConverter::new();
            let cb = |current: usize, total: usize| -> bool {
                progress_current.store(current, Ordering::Relaxed);
                progress_total.store(total, Ordering::Relaxed);
                true // never cancel (for now)
            };
            match converter.convert(&in_path, (w, h), &src_fmt, &out_path, &out_format, Some(&cb)) {
                Ok((count, cancelled)) => {
                    if cancelled {
                        *error.lock().unwrap() = Some("Conversion cancelled".to_string());
                    } else {
                        // Store total as final count
                        progress_current.store(count, Ordering::Relaxed);
                        progress_total.store(count, Ordering::Relaxed);
                        log::info!("Converted {} frames to {}", count, out_path);
                    }
                }
                Err(e) => {
                    *error.lock().unwrap() = Some(format!("Convert error: {e}"));
                }
            }
            done.store(true, Ordering::Relaxed);
        });
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

    fn start_test_download(&mut self, ctx: &egui::Context, filename: &str) {
        let url = format!("https://media.xiph.org/video/derf/y4m/{}", filename);
        let status = Arc::clone(&self.test_download_status);
        let path_out = Arc::clone(&self.test_download_path);
        let dl_current = Arc::clone(&self.test_dl_current);
        let dl_total = Arc::clone(&self.test_dl_total);
        let ctx2 = ctx.clone();
        *status.lock().unwrap() = Some(format!("Connecting ({filename})..."));
        dl_current.store(0, Ordering::Relaxed);
        dl_total.store(0, Ordering::Relaxed);
        self.test_downloading = true;
        let filename_owned = filename.to_string();
        std::thread::spawn(move || {
            let dir = std::env::temp_dir().join("video-viewer-test");
            let _ = std::fs::create_dir_all(&dir);
            let dest = dir.join(&filename_owned);
            let result = (|| -> Result<String, String> {
                let resp = ureq::get(&url).call().map_err(|e| format!("Download error: {e}"))?;
                let total = resp.headers().get("content-length")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                dl_total.store(total, Ordering::Relaxed);
                *status.lock().unwrap() = Some("Downloading...".to_string());
                use std::io::{Read, Write};
                let mut reader = resp.into_body().into_reader();
                let mut file = std::fs::File::create(&dest).map_err(|e| format!("File create error: {e}"))?;
                let mut buf = [0u8; 32768];
                let mut downloaded = 0usize;
                loop {
                    let n = reader.read(&mut buf).map_err(|e| format!("Read error: {e}"))?;
                    if n == 0 { break; }
                    file.write_all(&buf[..n]).map_err(|e| format!("Write error: {e}"))?;
                    downloaded += n;
                    dl_current.store(downloaded, Ordering::Relaxed);
                    ctx2.request_repaint();
                }
                Ok(dest.to_string_lossy().to_string())
            })();
            match result {
                Ok(p) => {
                    *status.lock().unwrap() = Some(format!("Saved: {}", p));
                    *path_out.lock().unwrap() = Some(p);
                }
                Err(e) => {
                    *status.lock().unwrap() = Some(e);
                }
            }
            ctx2.request_repaint();
        });
    }
}

impl eframe::App for VideoViewerApp {
    fn clear_color(&self, visuals: &egui::Visuals) -> [f32; 4] {
        let c = visuals.extreme_bg_color;
        [c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0, 1.0]
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Paint full viewport background to prevent maximize artifacts on WSLg.
        let bg_color = ctx.style().visuals.extreme_bg_color;
        ctx.layer_painter(egui::LayerId::background()).rect_filled(
            ctx.screen_rect(),
            0.0,
            bg_color,
        );

        // Custom window resize handles (no OS decorations).
        self.handle_window_resize(ctx);

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

        // --- Poll test video download ---
        if self.test_downloading {
            let dl_path = self.test_download_path.lock().unwrap().take();
            if let Some(path) = dl_path {
                self.test_downloading = false;
                // Y4M: width/height/format parsed from header automatically
                self.open_file(ctx, path, 0, 0, "");
            }
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
            let frame_duration_ms = (1000.0 / fps.max(1) as f64) as u64;
            let elapsed = self
                .last_frame_time
                .map(|t| t.elapsed().as_millis() as u64)
                .unwrap_or(u64::MAX);

            if elapsed >= frame_duration_ms {
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
        // Suppress shortcuts when a dialog is open (dialogs contain text fields, sliders, etc.).
        let dialog_open = self.dialog_state != DialogState::None
            || self.show_shortcuts
            || self.show_about
            || self.show_scene_detect_dialog;
        let keys = ctx.input(|i| {
            // Plain/navigation keys: only fire when no dialog is open.
            let plain = !dialog_open;
            (
                plain && i.key_pressed(egui::Key::Space),  // 0
                plain && !i.modifiers.ctrl && !i.modifiers.command && i.key_pressed(egui::Key::ArrowLeft),  // 1
                plain && !i.modifiers.ctrl && !i.modifiers.command && i.key_pressed(egui::Key::ArrowRight), // 2
                plain && i.key_pressed(egui::Key::Home),   // 3
                plain && i.key_pressed(egui::Key::End),    // 4
                plain && i.key_pressed(egui::Key::F),      // 5
                plain && i.key_pressed(egui::Key::G),      // 6
                plain && !i.modifiers.ctrl && !i.modifiers.command && i.key_pressed(egui::Key::B),  // 7
                plain && i.key_pressed(egui::Key::Num0),   // 8
                plain && i.key_pressed(egui::Key::Num1),   // 9
                plain && i.key_pressed(egui::Key::Num2),   // 10
                plain && i.key_pressed(egui::Key::Num3),   // 11
                plain && i.key_pressed(egui::Key::Num4),   // 12
                plain && i.modifiers.ctrl && i.key_pressed(egui::Key::S), // 13: Ctrl+S (save frame)
                plain && i.modifiers.ctrl && i.key_pressed(egui::Key::C), // 14: Ctrl+C (frame copy)
                plain && i.modifiers.ctrl && i.key_pressed(egui::Key::O), // 15: Ctrl+O (open file)
                i.modifiers.ctrl && i.key_pressed(egui::Key::Q), // 16: Ctrl+Q (always: quit)
                plain && !i.modifiers.ctrl && !i.modifiers.command && i.key_pressed(egui::Key::M),  // 17
                plain && i.modifiers.ctrl && !i.modifiers.shift && i.key_pressed(egui::Key::B), // 18
                plain && i.modifiers.ctrl && i.modifiers.shift && i.key_pressed(egui::Key::B), // 19
                plain && i.modifiers.ctrl && i.key_pressed(egui::Key::ArrowLeft),  // 20
                plain && i.modifiers.ctrl && i.key_pressed(egui::Key::ArrowRight), // 21
                plain && !i.modifiers.ctrl && !i.modifiers.command && i.key_pressed(egui::Key::C), // 22: center
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
        if keys.13 {
            // Ctrl+S: save frame
            let name = format!("frame_{:06}.png", self.current_frame_idx);
            self.save_file_dialog = Some(dialogs::SaveFileDialog::new("Save Frame", &name));
            self.dialog_state = DialogState::SaveFile;
        }
        if keys.14 { self.copy_frame_to_clipboard(); }
        // M: toggle magnifier
        if keys.17 {
            self.canvas.show_magnifier = !self.canvas.show_magnifier;
        }
        // C: center image
        if keys.22 {
            self.canvas.center_image();
        }
        // Ctrl+B: next bookmark
        if keys.18 {
            let cur = self.current_frame_idx;
            if let Some(&next) = self.bookmarks.iter()
                .filter(|&&b| b > cur)
                .min()
            {
                self.goto_frame(ctx, next);
                self.is_playing = false;
            }
        }
        // Ctrl+Shift+B: previous bookmark
        if keys.19 {
            let cur = self.current_frame_idx;
            if let Some(&prev) = self.bookmarks.iter()
                .filter(|&&b| b < cur)
                .max()
            {
                self.goto_frame(ctx, prev);
                self.is_playing = false;
            }
        }
        // Ctrl+Left: previous scene change
        if keys.20 {
            let cur = self.current_frame_idx;
            if let Some(&prev) = self.scene_changes.iter().rev()
                .find(|&&s| s < cur)
            {
                self.goto_frame(ctx, prev);
                self.is_playing = false;
            }
        }
        // Ctrl+Right: next scene change
        if keys.21 {
            let cur = self.current_frame_idx;
            if let Some(&next) = self.scene_changes.iter()
                .find(|&&s| s > cur)
            {
                self.goto_frame(ctx, next);
                self.is_playing = false;
            }
        }
        if keys.15 {
            // Ctrl+O: open file dialog
            self.open_file_dialog = Some(dialogs::OpenFileDialog::new(
                self.settings.defaults.width,
                self.settings.defaults.height,
                &self.settings.defaults.format,
                self.current_file.as_ref()
                    .and_then(|f| std::path::Path::new(f).parent())
                    .and_then(|p| p.to_str()),
            ));
            self.dialog_state = DialogState::OpenFile;
        }
        if keys.16 {
            // Ctrl+Q: quit
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }

        // --- Menu bar ---
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            // Make the menu bar draggable as a title bar (no OS decorations).
            let title_bar_response = ui.interact(
                ui.max_rect(),
                ui.id().with("title_bar_drag"),
                egui::Sense::click_and_drag(),
            );
            if title_bar_response.dragged() {
                ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
            }
            if title_bar_response.double_clicked() {
                let is_max = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
                ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!is_max));
            }

            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open... (Ctrl+O)").clicked() {
                        ui.close_menu();
                        self.open_file_dialog = Some(dialogs::OpenFileDialog::new(
                            self.settings.defaults.width,
                            self.settings.defaults.height,
                            &self.settings.defaults.format,
                            self.current_file.as_ref()
                                .and_then(|f| std::path::Path::new(f).parent())
                                .and_then(|p| p.to_str()),
                        ));
                        self.dialog_state = DialogState::OpenFile;
                    }
                    if ui.button("Save Frame (Ctrl+S)").clicked() {
                        ui.close_menu();
                        let name = format!("frame_{:06}.png", self.current_frame_idx);
                        self.save_file_dialog = Some(dialogs::SaveFileDialog::new("Save Frame", &name));
                        self.dialog_state = DialogState::SaveFile;
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
                                    let ext = std::path::Path::new(path.as_str())
                                        .extension()
                                        .and_then(|e| e.to_str())
                                        .unwrap_or("")
                                        .to_lowercase();
                                    let (w, h, fmt) = if ext == "y4m" {
                                        (0, 0, String::new())
                                    } else {
                                        let hints = crate::core::hints::parse_filename_hints(path);
                                        (
                                            hints.width.unwrap_or(self.settings.defaults.width),
                                            hints.height.unwrap_or(self.settings.defaults.height),
                                            hints.format.unwrap_or_else(|| self.settings.defaults.format.clone()),
                                        )
                                    };
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
                    if ui.button("Center Image (C)").clicked() {
                        self.canvas.center_image();
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
                    ui.checkbox(&mut self.canvas.show_magnifier, "Magnifier (M)");
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
                        self.convert_dialog = Some(dialogs::ConvertDialog::new(
                            &info,
                            self.current_file.as_deref(),
                        ));
                        self.dialog_state = DialogState::Convert;
                    }
                    if ui.button("Batch Convert...").clicked() {
                        ui.close_menu();
                        self.batch_convert_dialog = Some(dialogs::BatchConvertDialog::new(
                            self.settings.defaults.width,
                            self.settings.defaults.height,
                        ));
                        self.dialog_state = DialogState::BatchConvert;
                    }
                    ui.separator();
                    if ui.button("Copy Frame (Ctrl+C)").clicked() {
                        ui.close_menu();
                        self.copy_frame_to_clipboard();
                    }
                });
                ui.menu_button("Analysis", |ui| {
                    if ui.button("Detect Scene Changes...").clicked() {
                        ui.close_menu();
                        self.show_scene_detect_dialog = true;
                    }
                    if !self.scene_changes.is_empty() && ui.button("Clear Scene Changes").clicked() {
                        ui.close_menu();
                        self.scene_changes.clear();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("Keyboard Shortcuts").clicked() {
                        ui.close_menu();
                        self.show_shortcuts = true;
                    }
                    ui.separator();
                    let test_videos: &[(&str, &str)] = &[
                        ("akiyo (CIF)", "akiyo_cif.y4m"),
                        ("bus (CIF)", "bus_cif.y4m"),
                        ("carphone (QCIF)", "carphone_qcif.y4m"),
                        ("foreman (QCIF)", "foreman_qcif.y4m"),
                        ("garden (SIF)", "garden_sif.y4m"),
                        ("harbour (CIF)", "harbour_cif.y4m"),
                        ("mobile (CIF)", "mobile_cif.y4m"),
                        ("tennis (SIF)", "tennis_sif.y4m"),
                    ];
                    ui.menu_button("Download Test Video", |ui| {
                        for &(label, filename) in test_videos {
                            if ui.add_enabled(!self.test_downloading, egui::Button::new(label)).clicked() {
                                ui.close_menu();
                                self.start_test_download(ctx, filename);
                            }
                        }
                        ui.separator();
                        if self.test_downloading {
                            let current = self.test_dl_current.load(Ordering::Relaxed);
                            let total = self.test_dl_total.load(Ordering::Relaxed);
                            if total > 0 {
                                let frac = current as f32 / total as f32;
                                let mb_cur = current as f64 / (1024.0 * 1024.0);
                                let mb_tot = total as f64 / (1024.0 * 1024.0);
                                ui.add(egui::ProgressBar::new(frac)
                                    .text(format!("{:.1}/{:.1} MB", mb_cur, mb_tot)));
                            } else if let Some(ref status) = *self.test_download_status.lock().unwrap() {
                                ui.label(egui::RichText::new(status).small());
                            }
                        } else if let Some(ref status) = *self.test_download_status.lock().unwrap() {
                            ui.label(egui::RichText::new(status).small());
                        }
                        ui.separator();
                        if ui.button("Visit derf Collection...").clicked() {
                            ui.close_menu();
                            ctx.open_url(egui::OpenUrl::new_tab("https://media.xiph.org/video/derf/"));
                        }
                    });
                    ui.separator();
                    if ui.button("About").clicked() {
                        ui.close_menu();
                        self.show_about = true;
                    }
                });

                // --- Window control buttons (right-aligned) ---
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("x").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    let is_max = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
                    let max_label = if is_max { "🗗" } else { "🗖" };
                    if ui.button(max_label).clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!is_max));
                    }
                    if ui.button("🗕").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                    }
                    ui.separator();
                    ui.label("Video Viewer");
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
            if let Some(act) = self.toolbar.show(ui, is_yuv, self.auto_fit) {
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
                    ToolbarAction::ToggleAutoFit => {
                        self.auto_fit = !self.auto_fit;
                        if self.auto_fit {
                            let avail = ctx.available_rect().size();
                            self.canvas.fit_to_view(avail);
                        }
                    }
                    ToolbarAction::Refresh => {
                        // Re-decode and re-display the current frame
                        let idx = self.current_frame_idx;
                        self.goto_frame(ctx, idx);
                    }
                    ToolbarAction::ZoomIn => {
                        self.canvas.zoom = (self.canvas.zoom * 1.25).min(50.0);
                        self.auto_fit = false;
                    }
                    ToolbarAction::ZoomOut => {
                        self.canvas.zoom = (self.canvas.zoom / 1.25).max(0.1);
                        self.auto_fit = false;
                    }
                    ToolbarAction::Zoom1to1 => {
                        self.canvas.zoom = 1.0;
                        self.auto_fit = false;
                    }
                    ToolbarAction::Zoom2to1 => {
                        self.canvas.zoom = 2.0;
                        self.auto_fit = false;
                    }
                    ToolbarAction::CenterImage => {
                        self.canvas.center_image();
                    }
                }
            }
        });

        // --- Status bar (declared first → stacks above navigation) ---
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Download progress (shown inline in status bar)
                if self.test_downloading {
                    let current = self.test_dl_current.load(Ordering::Relaxed);
                    let total = self.test_dl_total.load(Ordering::Relaxed);
                    if total > 0 {
                        let frac = current as f32 / total as f32;
                        let mb_cur = current as f64 / (1024.0 * 1024.0);
                        let mb_tot = total as f64 / (1024.0 * 1024.0);
                        ui.add(egui::ProgressBar::new(frac)
                            .text(format!("Downloading: {:.1}/{:.1} MB", mb_cur, mb_tot))
                            .desired_width(250.0));
                        ui.separator();
                    } else {
                        ui.label("Connecting...");
                        ui.separator();
                    }
                }
                if let Some(ref err) = self.status_error {
                    ui.colored_label(egui::Color32::RED, err);
                } else if let Some(ref path) = self.current_file {
                    if let Some(ref r) = self.reader {
                        ui.label(format!(
                            "{}  |  {}x{}  {}  |  {:.0}%  |  Frame {}/{}  |  {:.1} fps",
                            path,
                            r.width(),
                            r.height(),
                            r.format_name(),
                            self.canvas.zoom_level() * 100.0,
                            self.current_frame_idx,
                            r.total_frames().saturating_sub(1),
                            self.playback_fps,
                        ));
                    }
                } else {
                    ui.label("No file loaded — drop a file or use File → Open");
                }
            });
        });

        // --- Navigation bar (declared second → sits at bottom edge) ---
        let total = self.total_frames();
        let cur = self.current_frame_idx;
        let is_playing = self.is_playing;
        egui::TopBottomPanel::bottom("navigation").show(ctx, |ui| {
            let action = self.nav.show(ui, cur, total, is_playing, &self.bookmarks, &self.scene_changes);
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
                    NavigationAction::SetFps(_) => {
                        self.last_frame_time = Some(Instant::now());
                    }
                }
            }
        });

        // --- Sidebar (right panel) ---
        let analysis_was_off = !self.sidebar.show_analysis;
        egui::SidePanel::right("sidebar")
            .default_width(250.0)
            .show(ctx, |ui| {
                self.sidebar.show(ui);
            });
        // If user just toggled analysis on, always recompute (data may be stale).
        if analysis_was_off && self.sidebar.show_analysis {
            self.update_analysis(ctx);
        }
        // If the user switched tabs in the viewport, recompute immediately.
        {
            let mut shared = self.sidebar.analysis.lock();
            if shared.tab_changed {
                shared.tab_changed = false;
                drop(shared);
                self.update_analysis(ctx);
            }
        }
        // Analysis as a separate floating window.
        self.sidebar.show_analysis_window(ctx);

        // --- Central panel (canvas) ---
        egui::CentralPanel::default().show(ctx, |ui| {
            // Detect window resize and force repaints (fixes WSLg compositor artifacts).
            {
                let avail = ui.available_size();
                let cur = (avail.x, avail.y);
                let changed = match self.last_available_size {
                    Some(prev) => (prev.0 - cur.0).abs() > 1.0 || (prev.1 - cur.1).abs() > 1.0,
                    None => true,
                };
                if changed {
                    if self.auto_fit && self.reader.is_some() {
                        self.canvas.fit_to_view(avail);
                    }
                    self.last_available_size = Some(cur);
                    // Force multiple repaints so the compositor fully redraws.
                    ctx.request_repaint();
                    ctx.request_repaint();
                }
            }

            if self.reader.is_some() {
                let response = self.canvas.show(ui);
                // Update pixel info on hover — pass absolute screen pos
                // Update pixel info on hover — keep last info when mouse leaves image
                if let Some(hover_pos) = response.hover_pos() {
                    if let Some((ix, iy)) = self.canvas.image_pos_from_screen(hover_pos) {
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
        // Open file dialog
        if self.dialog_state == DialogState::OpenFile {
            if let Some(ref mut dlg) = self.open_file_dialog {
                if let Some(result) = dlg.show(ctx) {
                    if let Some((path, w, h, fmt)) = result {
                        self.open_file(ctx, path, w, h, &fmt);
                    }
                    self.dialog_state = DialogState::None;
                    self.open_file_dialog = None;
                }
            }
        }

        // Save file dialog
        if self.dialog_state == DialogState::SaveFile {
            if let Some(ref mut dlg) = self.save_file_dialog {
                if let Some(result) = dlg.show(ctx) {
                    if let Some(path) = result {
                        self.save_frame_as_png_to(&path);
                    }
                    self.dialog_state = DialogState::None;
                    self.save_file_dialog = None;
                }
            }
        }

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

        // Convert dialog
        if self.dialog_state == DialogState::Convert {
            // Poll background conversion progress
            if self.convert_running {
                let current = self.convert_progress_current.load(Ordering::Relaxed);
                let total = self.convert_progress_total.load(Ordering::Relaxed);
                if let Some(ref mut dlg) = self.convert_dialog {
                    dlg.progress = Some((current, total));
                }
                ctx.request_repaint(); // keep UI updating during conversion

                if self.convert_done.load(Ordering::Relaxed) {
                    self.convert_running = false;
                    let err = self.convert_error.lock().unwrap().take();
                    if let Some(e) = err {
                        self.status_error = Some(e);
                    } else {
                        self.status_error = None;
                    }
                    if let Some(ref mut dlg) = self.convert_dialog {
                        dlg.progress = None;
                    }
                    self.dialog_state = DialogState::None;
                    self.convert_dialog = None;
                }
            }

            if let Some(ref mut dlg) = self.convert_dialog {
                if let Some(result) = dlg.show(ctx) {
                    if let Some((out_format, out_path)) = result {
                        self.run_conversion(&out_format, &out_path);
                        // Don't close dialog yet — it stays open showing progress
                    } else {
                        // Cancel
                        self.dialog_state = DialogState::None;
                        self.convert_dialog = None;
                    }
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

        // Batch convert dialog
        if self.dialog_state == DialogState::BatchConvert {
            if let Some(ref mut dlg) = self.batch_convert_dialog {
                if let Some(jobs) = dlg.show(ctx) {
                    if !jobs.is_empty() {
                        let w = dlg.width;
                        let h = dlg.height;
                        for (input, out_fmt, out_path) in &jobs {
                            self.run_batch_single(input, w, h, out_fmt, out_path);
                        }
                    }
                    self.dialog_state = DialogState::None;
                    self.batch_convert_dialog = None;
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

        // Scene detection dialog
        if self.show_scene_detect_dialog {
            let mut open = true;
            let mut start_detect = false;
            let mut algo = self.scene_detect_algorithm;
            let mut thresh = self.scene_detect_threshold;

            egui::Window::new("Detect Scene Changes")
                .open(&mut open)
                .resizable(false)
                .collapsible(false)
                .show(ctx, |ui| {
                    use crate::analysis::scene::SceneAlgorithm;

                    ui.horizontal(|ui| {
                        ui.label("Algorithm:");
                        egui::ComboBox::from_id_salt("scene_algo")
                            .selected_text(format!("{}", algo))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut algo, SceneAlgorithm::Mad, "MAD");
                                ui.selectable_value(&mut algo, SceneAlgorithm::Histogram, "Histogram");
                                ui.selectable_value(&mut algo, SceneAlgorithm::Ssim, "SSIM");
                            });
                    });
                    // Reset threshold to default when algorithm changes.
                    if algo != self.scene_detect_algorithm {
                        thresh = algo.default_threshold();
                    }
                    ui.horizontal(|ui| {
                        ui.label("Threshold:");
                        ui.add(egui::DragValue::new(&mut thresh).speed(0.1));
                        if ui.small_button("Reset").clicked() {
                            thresh = algo.default_threshold();
                        }
                    });
                    ui.add_space(4.0);
                    ui.weak(match algo {
                        SceneAlgorithm::Mad => "Mean Absolute Difference: higher threshold = fewer detections.",
                        SceneAlgorithm::Histogram => "Luma histogram correlation: higher threshold = fewer detections.",
                        SceneAlgorithm::Ssim => "Structural similarity: higher threshold = fewer detections.",
                    });
                    ui.add_space(8.0);
                    let has_file = self.reader.is_some();
                    ui.add_enabled_ui(has_file && !self.scene_detect_running, |ui| {
                        if ui.button("Detect").clicked() {
                            start_detect = true;
                        }
                    });
                    if self.scene_detect_running {
                        ui.spinner();
                        ui.label("Detecting...");
                    }
                    if !self.scene_changes.is_empty() {
                        ui.label(format!("{} scene changes found.", self.scene_changes.len()));
                    }
                });

            self.scene_detect_algorithm = algo;
            self.scene_detect_threshold = thresh;
            if !open {
                self.show_scene_detect_dialog = false;
            }

            if start_detect {
                self.run_scene_detection(ctx);
            }
        }

        // Poll scene detection background result using tagged job ID.
        if self.scene_detect_running {
            if let Some((job_id, result)) = self.scene_detect_output.lock().unwrap().take() {
                if job_id == self.scene_detect_active_job.load(Ordering::Acquire) {
                    self.scene_detect_running = false;
                    match result {
                        Ok(changes) => {
                            self.scene_changes = changes;
                            self.status_error = None;
                        }
                        Err(err) => {
                            self.status_error = Some(err);
                        }
                    }
                }
                // else: stale job output — already consumed and discarded by take()
            }
        }
    }

    fn export_clip(&mut self, start: usize, end: usize, path: &str) {
        if start > end {
            self.status_error = Some(format!("Export error: start frame ({start}) > end frame ({end})"));
            return;
        }
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
        if start > end {
            self.status_error = Some(format!("PNG export error: start frame ({start}) > end frame ({end})"));
            return;
        }
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
