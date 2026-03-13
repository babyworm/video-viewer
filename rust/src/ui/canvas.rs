use eframe::egui;

pub struct ImageCanvas {
    pub texture: Option<egui::TextureHandle>,
    pub zoom: f32,
    pub pan_offset: egui::Vec2,
    pub image_size: Option<(u32, u32)>,
    pub grid_size: u32,
    pub sub_grid_size: u32,
}

impl ImageCanvas {
    pub fn new() -> Self {
        Self {
            texture: None,
            zoom: 1.0,
            pan_offset: egui::Vec2::ZERO,
            image_size: None,
            grid_size: 0,
            sub_grid_size: 0,
        }
    }

    pub fn set_grid_size(&mut self, size: u32) {
        self.grid_size = size;
    }

    pub fn set_sub_grid_size(&mut self, size: u32) {
        self.sub_grid_size = size;
    }

    pub fn set_image(&mut self, ctx: &egui::Context, rgb: &[u8], width: u32, height: u32) {
        let color_image = egui::ColorImage::from_rgb(
            [width as usize, height as usize],
            rgb,
        );
        self.texture = Some(ctx.load_texture(
            "canvas_image",
            color_image,
            egui::TextureOptions::LINEAR,
        ));
        self.image_size = Some((width, height));
    }

    pub fn show(&mut self, ui: &mut egui::Ui) -> egui::Response {
        if let Some(ref texture) = self.texture {
            let (w, h) = self.image_size.unwrap_or((1, 1));
            let display_size = egui::vec2(w as f32 * self.zoom, h as f32 * self.zoom);

            let response = ui.add(
                egui::Image::new(texture)
                    .fit_to_exact_size(display_size)
            );

            // Grid overlay
            let image_origin = response.rect.min;
            let painter = ui.painter();

            if self.sub_grid_size > 0 {
                let step = self.sub_grid_size as f32 * self.zoom;
                let img_w = w as f32 * self.zoom;
                let img_h = h as f32 * self.zoom;
                let color = egui::Color32::from_rgb(200, 200, 0);
                let stroke = egui::Stroke::new(0.5, color);
                let mut x = step;
                while x < img_w {
                    painter.line_segment(
                        [
                            egui::pos2(image_origin.x + x, image_origin.y),
                            egui::pos2(image_origin.x + x, image_origin.y + img_h),
                        ],
                        stroke,
                    );
                    x += step;
                }
                let mut y = step;
                while y < img_h {
                    painter.line_segment(
                        [
                            egui::pos2(image_origin.x, image_origin.y + y),
                            egui::pos2(image_origin.x + img_w, image_origin.y + y),
                        ],
                        stroke,
                    );
                    y += step;
                }
            }

            if self.grid_size > 0 {
                let step = self.grid_size as f32 * self.zoom;
                let img_w = w as f32 * self.zoom;
                let img_h = h as f32 * self.zoom;
                let color = egui::Color32::from_rgb(0, 200, 0);
                let stroke = egui::Stroke::new(1.0, color);
                let mut x = step;
                while x < img_w {
                    painter.line_segment(
                        [
                            egui::pos2(image_origin.x + x, image_origin.y),
                            egui::pos2(image_origin.x + x, image_origin.y + img_h),
                        ],
                        stroke,
                    );
                    x += step;
                }
                let mut y = step;
                while y < img_h {
                    painter.line_segment(
                        [
                            egui::pos2(image_origin.x, image_origin.y + y),
                            egui::pos2(image_origin.x + img_w, image_origin.y + y),
                        ],
                        stroke,
                    );
                    y += step;
                }
            }

            // Mouse wheel zoom
            let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll_delta != 0.0 {
                let factor = if scroll_delta > 0.0 { 1.1_f32 } else { 1.0 / 1.1 };
                self.zoom = (self.zoom * factor).clamp(0.1, 50.0);
            }

            // Middle-click drag for panning
            if response.dragged_by(egui::PointerButton::Middle) {
                self.pan_offset += response.drag_delta();
            }

            response
        } else {
            ui.allocate_response(ui.available_size(), egui::Sense::hover())
        }
    }

    pub fn fit_to_view(&mut self, available: egui::Vec2) {
        if let Some((w, h)) = self.image_size {
            let zoom_w = available.x / w as f32;
            let zoom_h = available.y / h as f32;
            self.zoom = zoom_w.min(zoom_h);
            self.pan_offset = egui::Vec2::ZERO;
        }
    }

    pub fn zoom_level(&self) -> f32 {
        self.zoom
    }

    pub fn image_pos_from_screen(&self, screen_pos: egui::Pos2) -> Option<(u32, u32)> {
        let (w, h) = self.image_size?;
        let x = (screen_pos.x / self.zoom) as i32;
        let y = (screen_pos.y / self.zoom) as i32;
        if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
            Some((x as u32, y as u32))
        } else {
            None
        }
    }
}

impl Default for ImageCanvas {
    fn default() -> Self {
        Self::new()
    }
}
