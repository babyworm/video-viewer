pub mod core;
pub mod app;
pub mod ui;

pub fn run_gui(
    input: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    format: Option<String>,
) {
    let _ = (input, width, height, format); // will be used once VideoReader is integrated

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("Video Viewer"),
        ..Default::default()
    };

    eframe::run_native(
        "Video Viewer",
        options,
        Box::new(|cc| Ok(Box::new(app::VideoViewerApp::new(cc)))),
    )
    .unwrap();
}
