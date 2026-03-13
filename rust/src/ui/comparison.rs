use eframe::egui;

/// Comparison display mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonMode {
    Split,
    Overlay,
    Diff,
}

/// A/B comparison view for two video frames.
pub struct ComparisonView {
    pub mode: ComparisonMode,
    /// Normalised split position (0.0 = full reference, 1.0 = full main).
    pub split_pos: f32,
    /// Overlay opacity for the reference image.
    pub overlay_opacity: f32,
    /// Reference image texture.
    pub ref_texture: Option<egui::TextureHandle>,
    /// Amplified-difference heatmap texture.
    pub diff_texture: Option<egui::TextureHandle>,
    /// Whether the comparison panel is visible.
    pub is_open: bool,
}

impl Default for ComparisonView {
    fn default() -> Self {
        Self::new()
    }
}

impl ComparisonView {
    pub fn new() -> Self {
        Self {
            mode: ComparisonMode::Split,
            split_pos: 0.5,
            overlay_opacity: 0.5,
            ref_texture: None,
            diff_texture: None,
            is_open: false,
        }
    }

    /// Upload an RGB buffer as the reference image.
    pub fn set_reference_image(
        &mut self,
        ctx: &egui::Context,
        rgb: &[u8],
        w: u32,
        h: u32,
    ) {
        let color_image =
            egui::ColorImage::from_rgb([w as usize, h as usize], rgb);
        self.ref_texture = Some(ctx.load_texture(
            "comparison_ref",
            color_image,
            egui::TextureOptions::LINEAR,
        ));
    }

    /// Compute an amplified (10x) absolute-difference heatmap between two RGB
    /// buffers and store the result as `diff_texture`.
    pub fn compute_diff(
        &mut self,
        ctx: &egui::Context,
        main_rgb: &[u8],
        ref_rgb: &[u8],
        w: u32,
        h: u32,
    ) {
        let len = (w as usize) * (h as usize) * 3;
        let mut diff_buf = vec![0u8; len];
        for i in 0..len.min(main_rgb.len()).min(ref_rgb.len()) {
            let d = (main_rgb[i] as i16 - ref_rgb[i] as i16).unsigned_abs();
            diff_buf[i] = (d.saturating_mul(10)).min(255) as u8;
        }
        let color_image =
            egui::ColorImage::from_rgb([w as usize, h as usize], &diff_buf);
        self.diff_texture = Some(ctx.load_texture(
            "comparison_diff",
            color_image,
            egui::TextureOptions::LINEAR,
        ));
    }

    /// Render the comparison view.
    ///
    /// `main_texture` is the currently displayed frame; `image_size` is its
    /// (width, height) in pixels.
    pub fn show(
        &mut self,
        _ctx: &egui::Context,
        ui: &mut egui::Ui,
        main_texture: Option<&egui::TextureHandle>,
        image_size: Option<(u32, u32)>,
    ) {
        if !self.is_open {
            return;
        }

        // Mode selector
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.mode, ComparisonMode::Split, "Split");
            ui.selectable_value(&mut self.mode, ComparisonMode::Overlay, "Overlay");
            ui.selectable_value(&mut self.mode, ComparisonMode::Diff, "Diff");
        });
        ui.separator();

        let (w, h) = match image_size {
            Some(s) => s,
            None => {
                ui.label("No image loaded.");
                return;
            }
        };

        let avail = ui.available_size();
        let aspect = w as f32 / h as f32;
        let display_w = avail.x.min(avail.y * aspect);
        let display_h = display_w / aspect;
        let display_size = egui::vec2(display_w, display_h);

        match self.mode {
            ComparisonMode::Split => self.show_split(ui, main_texture, display_size),
            ComparisonMode::Overlay => self.show_overlay(ui, main_texture, display_size),
            ComparisonMode::Diff => self.show_diff(ui, display_size),
        }
    }

    // ── Split mode ───────────────────────────────────────────────────

    fn show_split(
        &mut self,
        ui: &mut egui::Ui,
        main_texture: Option<&egui::TextureHandle>,
        display_size: egui::Vec2,
    ) {
        ui.label("Drag the divider to adjust split position.");
        ui.add(egui::Slider::new(&mut self.split_pos, 0.0..=1.0).text("Split"));

        let (rect, response) =
            ui.allocate_exact_size(display_size, egui::Sense::click_and_drag());

        // Drag to move the split divider
        if response.dragged() {
            if let Some(pos) = response.interact_pointer_pos() {
                self.split_pos =
                    ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
            }
        }

        let painter = ui.painter_at(rect);
        let split_x = rect.left() + rect.width() * self.split_pos;

        // Left half: main texture
        if let Some(tex) = main_texture {
            let left_rect = egui::Rect::from_min_max(rect.min, egui::pos2(split_x, rect.max.y));
            let uv_right = self.split_pos;
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(uv_right, 1.0));
            painter.image(tex.id(), left_rect, uv, egui::Color32::WHITE);
        }

        // Right half: reference texture
        if let Some(ref tex) = self.ref_texture {
            let right_rect =
                egui::Rect::from_min_max(egui::pos2(split_x, rect.min.y), rect.max);
            let uv =
                egui::Rect::from_min_max(egui::pos2(self.split_pos, 0.0), egui::pos2(1.0, 1.0));
            painter.image(tex.id(), right_rect, uv, egui::Color32::WHITE);
        }

        // Divider line
        painter.line_segment(
            [
                egui::pos2(split_x, rect.top()),
                egui::pos2(split_x, rect.bottom()),
            ],
            egui::Stroke::new(2.0, egui::Color32::YELLOW),
        );
    }

    // ── Overlay mode ─────────────────────────────────────────────────

    fn show_overlay(
        &mut self,
        ui: &mut egui::Ui,
        main_texture: Option<&egui::TextureHandle>,
        display_size: egui::Vec2,
    ) {
        ui.add(
            egui::Slider::new(&mut self.overlay_opacity, 0.0..=1.0).text("Ref opacity"),
        );

        let (rect, _response) =
            ui.allocate_exact_size(display_size, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));

        // Main texture at full opacity
        if let Some(tex) = main_texture {
            painter.image(tex.id(), rect, uv, egui::Color32::WHITE);
        }

        // Reference texture blended on top
        if let Some(ref tex) = self.ref_texture {
            let alpha = (self.overlay_opacity * 255.0) as u8;
            let tint = egui::Color32::from_rgba_unmultiplied(255, 255, 255, alpha);
            painter.image(tex.id(), rect, uv, tint);
        }
    }

    // ── Diff mode ────────────────────────────────────────────────────

    fn show_diff(&self, ui: &mut egui::Ui, display_size: egui::Vec2) {
        if let Some(ref tex) = self.diff_texture {
            ui.image(egui::load::SizedTexture::new(tex.id(), display_size));
        } else {
            ui.label("No diff computed. Load a reference image first.");
        }
    }
}
