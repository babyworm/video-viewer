pub mod core;
pub mod app;
pub mod ui;
pub mod analysis;
pub mod conversion;

pub fn run_gui(
    input: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    format: Option<String>,
) {
    // Force software rendering on WSL2 / headless environments where GPU
    // drivers are unavailable.  LIBGL_ALWAYS_SOFTWARE makes Mesa's llvmpipe
    // kick in, and EGL_PLATFORM=x11 avoids the broken Wayland path.
    if std::env::var("LIBGL_ALWAYS_SOFTWARE").is_err() {
        // Only set if the user hasn't already chosen a value.
        std::env::set_var("LIBGL_ALWAYS_SOFTWARE", "1");
    }

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("Video Viewer")
            .with_transparent(false)
            .with_decorations(false),
        ..Default::default()
    };

    if let Err(e) = eframe::run_native(
        "Video Viewer",
        options,
        Box::new(move |cc| {
            Ok(Box::new(app::VideoViewerApp::new(
                cc,
                input.clone(),
                width,
                height,
                format.clone(),
            )))
        }),
    ) {
        log::error!("eframe exited with error: {e}");
    }
}
