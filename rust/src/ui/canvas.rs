use eframe::egui;

pub struct ImageCanvas {
    pub texture: Option<egui::TextureHandle>,
    pub zoom: f32,
    pub pan_offset: egui::Vec2,
    pub image_size: Option<(u32, u32)>,
    pub grid_size: u32,
    pub sub_grid_size: u32,
    /// Show a magnifier overlay when hovering the image.
    pub show_magnifier: bool,
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
            show_magnifier: false,
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

        // Clear background to prevent resize artifacts.
        painter.rect_filled(panel_rect, 0.0, ui.visuals().extreme_bg_color);

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

            // Magnifier overlay: shows a zoomed-in region around the cursor.
            if self.show_magnifier {
                if let Some(pointer) = ui.input(|i| i.pointer.hover_pos()) {
                    if image_rect.contains(pointer) {
                        let mag_factor = 8.0_f32;
                        let mag_size = 160.0_f32; // display size of the magnifier
                        // Source region in UV coordinates.
                        let src_half = (mag_size / mag_factor) / 2.0;
                        let rel = pointer - image_rect.min;
                        let uv_center_x = rel.x / image_rect.width();
                        let uv_center_y = rel.y / image_rect.height();
                        let uv_half_x = src_half / image_rect.width();
                        let uv_half_y = src_half / image_rect.height();
                        let uv_min = egui::pos2(
                            (uv_center_x - uv_half_x).clamp(0.0, 1.0),
                            (uv_center_y - uv_half_y).clamp(0.0, 1.0),
                        );
                        let uv_max = egui::pos2(
                            (uv_center_x + uv_half_x).clamp(0.0, 1.0),
                            (uv_center_y + uv_half_y).clamp(0.0, 1.0),
                        );
                        // Position magnifier at top-right of cursor with offset.
                        let mag_pos = egui::pos2(pointer.x + 20.0, pointer.y - mag_size - 10.0);
                        // Clamp to panel bounds.
                        let mag_pos = egui::pos2(
                            mag_pos.x.clamp(panel_rect.left(), panel_rect.right() - mag_size),
                            mag_pos.y.clamp(panel_rect.top(), panel_rect.bottom() - mag_size),
                        );
                        let mag_rect = egui::Rect::from_min_size(mag_pos, egui::vec2(mag_size, mag_size));
                        // Background + border.
                        painter.rect_filled(mag_rect, 2.0, egui::Color32::BLACK);
                        painter.image(
                            texture.id(),
                            mag_rect.shrink(1.0),
                            egui::Rect::from_min_max(uv_min, uv_max),
                            egui::Color32::WHITE,
                        );
                        painter.rect_stroke(mag_rect, 2.0, egui::Stroke::new(1.0, egui::Color32::GRAY), egui::StrokeKind::Outside);
                        // Crosshair in center.
                        let center = mag_rect.center();
                        let ch = 6.0;
                        let ch_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(255, 255, 0, 180));
                        painter.line_segment([egui::pos2(center.x - ch, center.y), egui::pos2(center.x + ch, center.y)], ch_stroke);
                        painter.line_segment([egui::pos2(center.x, center.y - ch), egui::pos2(center.x, center.y + ch)], ch_stroke);
                    }
                }
            }

            // Mouse wheel zoom — only when canvas panel is hovered.
            let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll_delta != 0.0 && response.hovered() {
                // Scale zoom proportionally to scroll amount, capped at ±10% per frame.
                let sensitivity = 0.001_f32;
                let zoom_change = (scroll_delta * sensitivity).clamp(-0.10, 0.10);
                let factor = 1.0 + zoom_change;
                let new_zoom = (self.zoom * factor).clamp(0.1, 50.0);
                // Zoom anchor: cursor if over the image, otherwise image center.
                let anchor = ui.input(|i| i.pointer.hover_pos())
                    .filter(|p| image_rect.contains(*p))
                    .unwrap_or(image_rect.center());
                let rel = anchor - origin;
                self.pan_offset += rel * (1.0 - new_zoom / self.zoom);
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
            self.zoom = zoom_w.min(zoom_h).clamp(0.1, 50.0);
            self.pan_offset = egui::Vec2::ZERO;
        }
    }

    /// Center the image in the canvas without changing zoom level.
    pub fn center_image(&mut self) {
        self.pan_offset = egui::Vec2::ZERO;
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
