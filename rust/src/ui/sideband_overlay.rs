use egui::{Color32, Painter, Pos2, Rect};
use crate::core::sideband::{SidebandCtu, SidebandFrame, SidebandOverlayMode};

/// Parameters for drawing sideband CTU overlay.
pub struct SidebandOverlayParams<'a> {
    pub painter: &'a Painter,
    pub image_rect: Rect,
    pub frame: &'a SidebandFrame,
    pub mode: SidebandOverlayMode,
    pub opacity: f32,
    pub show_values: bool,
    pub image_width: u32,
    pub image_height: u32,
    pub ctu_size: u32,
    pub zoom: f32,
}

/// Draw the sideband CTU overlay on top of the image.
pub fn draw_sideband_overlay(params: &SidebandOverlayParams) {
    if params.mode == SidebandOverlayMode::None
        || params.ctu_size == 0
        || params.image_width == 0
        || params.image_height == 0
    {
        return;
    }

    let ctu_cols = params.image_width.div_ceil(params.ctu_size);
    let ctu_rows = params.image_height.div_ceil(params.ctu_size);
    let max_ctus = (ctu_cols * ctu_rows) as usize;
    let alpha = (params.opacity * 180.0) as u8; // max ~70% alpha

    for (idx, ctu) in params.frame.ctus.iter().take(max_ctus).enumerate() {
        let col = idx as u32 % ctu_cols;
        let row = idx as u32 / ctu_cols;

        let x0 = params.image_rect.left() + col as f32 * params.ctu_size as f32 * params.zoom;
        let y0 = params.image_rect.top() + row as f32 * params.ctu_size as f32 * params.zoom;
        let x1 = x0 + params.ctu_size as f32 * params.zoom;
        let y1 = y0 + params.ctu_size as f32 * params.zoom;

        let ctu_rect = Rect::from_min_max(
            Pos2::new(x0, y0),
            Pos2::new(x1.min(params.image_rect.right()), y1.min(params.image_rect.bottom())),
        );

        // Clip to image area
        if !params.image_rect.intersects(ctu_rect) {
            continue;
        }

        let (value, color) = get_ctu_color(ctu, params.mode);
        let fill = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha);
        params.painter.rect_filled(ctu_rect, 0.0, fill);

        // Show value label when zoomed in enough
        if params.show_values && params.zoom >= 2.0 {
            let label = format_value(value, params.mode);
            let center = ctu_rect.center();
            params.painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(10.0 * params.zoom.min(4.0)),
                Color32::WHITE,
            );
        }
    }
}

fn get_ctu_color(ctu: &SidebandCtu, mode: SidebandOverlayMode) -> (f64, Color32) {
    match mode {
        SidebandOverlayMode::QpDelta => {
            let v = ctu.qp_delta as f64;
            // Diverging: blue(-3) -> white(0) -> red(+3)
            let t = (v + 3.0) / 6.0; // 0.0 to 1.0
            let color = diverging_colormap(t.clamp(0.0, 1.0));
            (v, color)
        }
        SidebandOverlayMode::Activity => {
            let v = ctu.activity as f64;
            (v, sequential_colormap(v / 63.0, [20, 20, 20], [255, 230, 0]))
        }
        SidebandOverlayMode::Flatness => {
            let v = ctu.flatness as f64;
            (v, sequential_colormap(v / 63.0, [20, 20, 20], [0, 200, 80]))
        }
        SidebandOverlayMode::Saliency => {
            let v = ctu.saliency as f64;
            (v, sequential_colormap(v / 63.0, [20, 20, 20], [180, 0, 255]))
        }
        SidebandOverlayMode::EdgeDensity => {
            let v = ctu.edge_density as f64;
            (v, sequential_colormap(v / 63.0, [20, 20, 20], [0, 220, 220]))
        }
        SidebandOverlayMode::Noise => {
            let v = ctu.noise as f64;
            (v, sequential_colormap(v / 15.0, [20, 20, 20], [255, 60, 60]))
        }
        SidebandOverlayMode::Confidence => {
            let v = ctu.confidence as f64;
            (v, sequential_colormap(v / 15.0, [255, 60, 60], [60, 200, 60]))
        }
        SidebandOverlayMode::TemporalStability => {
            let v = ctu.temporal_stability as f64;
            (v, sequential_colormap(v / 255.0, [255, 60, 60], [60, 60, 255]))
        }
        SidebandOverlayMode::None => (0.0, Color32::TRANSPARENT),
    }
}

/// Blue -> White -> Red diverging colormap
fn diverging_colormap(t: f64) -> Color32 {
    let t = t as f32;
    if t < 0.5 {
        // Blue to White (t: 0.0 -> 0.5)
        let s = t * 2.0;
        Color32::from_rgb((s * 255.0) as u8, (s * 255.0) as u8, 255)
    } else {
        // White to Red (t: 0.5 -> 1.0)
        let s = (t - 0.5) * 2.0;
        Color32::from_rgb(255, ((1.0 - s) * 255.0) as u8, ((1.0 - s) * 255.0) as u8)
    }
}

/// Linear interpolation sequential colormap
fn sequential_colormap(t: f64, from: [u8; 3], to: [u8; 3]) -> Color32 {
    let t = t.clamp(0.0, 1.0) as f32;
    Color32::from_rgb(
        (from[0] as f32 + (to[0] as f32 - from[0] as f32) * t) as u8,
        (from[1] as f32 + (to[1] as f32 - from[1] as f32) * t) as u8,
        (from[2] as f32 + (to[2] as f32 - from[2] as f32) * t) as u8,
    )
}

fn format_value(value: f64, mode: SidebandOverlayMode) -> String {
    match mode {
        SidebandOverlayMode::QpDelta => {
            if value >= 0.0 {
                format!("+{}", value as i8)
            } else {
                format!("{}", value as i8)
            }
        }
        _ => format!("{}", value.round() as i64),
    }
}
