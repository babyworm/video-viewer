use eframe::egui;

pub struct VideoViewerApp {
    pub current_file: Option<String>,
    // Placeholder fields for future
}

impl VideoViewerApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self { current_file: None }
    }
}

impl eframe::App for VideoViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Top menu bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Video", &["yuv", "y4m", "rgb", "raw", "nv12", "nv21"])
                            .pick_file()
                        {
                            self.current_file = Some(path.display().to_string());
                        }
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("View", |_ui| {});
                ui.menu_button("Tools", |_ui| {});
                ui.menu_button("Help", |ui| {
                    if ui.button("About").clicked() {
                        // TODO: about dialog
                        ui.close_menu();
                    }
                });
            });
        });

        // Bottom status bar
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(ref path) = self.current_file {
                    ui.label(format!("File: {}", path));
                } else {
                    ui.label("No file loaded");
                }
            });
        });

        // Central panel (canvas placeholder)
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.current_file.is_some() {
                ui.label("Canvas will be here");
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Drop a file or use File \u{2192} Open");
                });
            }
        });
    }
}
