use eframe::egui;

pub struct ImageCanvas {
    pub texture: Option<egui::TextureHandle>,
    pub zoom: f32,
    pub pan_offset: egui::Vec2,
    pub image_size: Option<(u32, u32)>,
    pub grid_size: u32,
    pub sub_grid_size: u32,
    /// Actual image rect in screen coordinates (set during show()).
    image_rect: Option<egui::Rect>,
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
            image_rect: None,
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
        // Allocate the full available space for interaction (zoom, pan, hover).
        let available = ui.available_size();
        let (response, painter) =
            ui.allocate_painter(available, egui::Sense::click_and_drag().union(egui::Sense::hover()));
        let panel_rect = response.rect;

        if let (Some(ref texture), Some((w, h))) = (&self.texture, self.image_size) {
            let img_w = w as f32 * self.zoom;
            let img_h = h as f32 * self.zoom;

            // Center image in panel, then apply pan offset.
            let center_offset = egui::vec2(
                (panel_rect.width() - img_w) / 2.0,
                (panel_rect.height() - img_h) / 2.0,
            );
            let origin = panel_rect.min + center_offset + self.pan_offset;
            let image_rect = egui::Rect::from_min_size(origin, egui::vec2(img_w, img_h));

            // Paint the image.
            painter.image(
                texture.id(),
                image_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );

            // Store for coordinate conversion.
            self.image_rect = Some(image_rect);

            // Grid overlays (drawn on top of image).
            self.draw_grid(&painter, image_rect, w, h);

            // Mouse wheel zoom (zoom towards cursor).
            let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll_delta != 0.0 {
                let factor = if scroll_delta > 0.0 { 1.1_f32 } else { 1.0 / 1.1 };
                let new_zoom = (self.zoom * factor).clamp(0.1, 50.0);
                // Zoom towards the mouse cursor position.
                if let Some(pointer) = ui.input(|i| i.pointer.hover_pos()) {
                    let rel = pointer - origin;
                    self.pan_offset += rel * (1.0 - new_zoom / self.zoom);
                }
                self.zoom = new_zoom;
            }

            // Middle-click drag for panning.
            if response.dragged_by(egui::PointerButton::Middle) {
                self.pan_offset += response.drag_delta();
            }
        } else {
            self.image_rect = None;
        }

        response
    }

    fn draw_grid(&self, painter: &egui::Painter, image_rect: egui::Rect, w: u32, h: u32) {
        let origin = image_rect.min;
        let img_w = w as f32 * self.zoom;
        let img_h = h as f32 * self.zoom;

        if self.sub_grid_size > 0 {
            let step = self.sub_grid_size as f32 * self.zoom;
            let color = egui::Color32::from_rgb(200, 200, 0);
            let stroke = egui::Stroke::new(0.5, color);
            let mut x = step;
            while x < img_w {
                painter.line_segment(
                    [egui::pos2(origin.x + x, origin.y),
                     egui::pos2(origin.x + x, origin.y + img_h)],
                    stroke,
                );
                x += step;
            }
            let mut y = step;
            while y < img_h {
                painter.line_segment(
                    [egui::pos2(origin.x, origin.y + y),
                     egui::pos2(origin.x + img_w, origin.y + y)],
                    stroke,
                );
                y += step;
            }
        }

        if self.grid_size > 0 {
            let step = self.grid_size as f32 * self.zoom;
            let color = egui::Color32::from_rgb(0, 200, 0);
            let stroke = egui::Stroke::new(1.0, color);
            let mut x = step;
            while x < img_w {
                painter.line_segment(
                    [egui::pos2(origin.x + x, origin.y),
                     egui::pos2(origin.x + x, origin.y + img_h)],
                    stroke,
                );
                x += step;
            }
            let mut y = step;
            while y < img_h {
                painter.line_segment(
                    [egui::pos2(origin.x, origin.y + y),
                     egui::pos2(origin.x + img_w, origin.y + y)],
                    stroke,
                );
                y += step;
            }
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

    /// Convert a screen position to image pixel coordinates.
    /// `screen_pos` is in absolute screen coordinates (e.g., from hover_pos()).
    pub fn image_pos_from_screen(&self, screen_pos: egui::Pos2) -> Option<(u32, u32)> {
        let (w, h) = self.image_size?;
        let image_rect = self.image_rect?;
        let rel = screen_pos - image_rect.min;
        let x = (rel.x / self.zoom) as i32;
        let y = (rel.y / self.zoom) as i32;
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
