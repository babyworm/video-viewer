use egui::{Color32, Painter, Pos2, Rect};
use crate::core::sideband::{SidebandCtu, SidebandFrame, SidebandOverlayMode};

/// Draw the sideband CTU overlay on top of the image.
pub fn draw_sideband_overlay(
    painter: &Painter,
    image_rect: Rect,
    frame: &SidebandFrame,
    mode: SidebandOverlayMode,
    opacity: f32,
    show_values: bool,
    image_width: u32,
    image_height: u32,
    ctu_size: u32,
    zoom: f32,
    offset: Pos2, // pan offset
) {
    if mode == SidebandOverlayMode::None {
        return;
    }

    let ctu_cols = (image_width + ctu_size - 1) / ctu_size;
    let _ctu_rows = (image_height + ctu_size - 1) / ctu_size;
    let alpha = (opacity * 180.0) as u8; // max ~70% alpha

    for (idx, ctu) in frame.ctus.iter().enumerate() {
        let col = idx as u32 % ctu_cols;
        let row = idx as u32 / ctu_cols;

        let x0 = image_rect.left() + col as f32 * ctu_size as f32 * zoom + offset.x;
        let y0 = image_rect.top() + row as f32 * ctu_size as f32 * zoom + offset.y;
        let x1 = x0 + ctu_size as f32 * zoom;
        let y1 = y0 + ctu_size as f32 * zoom;

        let ctu_rect = Rect::from_min_max(
            Pos2::new(x0, y0),
            Pos2::new(x1.min(image_rect.right()), y1.min(image_rect.bottom())),
        );

        // Clip to image area
        if !image_rect.intersects(ctu_rect) {
            continue;
        }

        let (value, color) = get_ctu_color(ctu, mode);
        let fill = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha);
        painter.rect_filled(ctu_rect, 0.0, fill);

        // Show value label when zoomed in enough
        if show_values && zoom >= 2.0 {
            let label = format_value(value, mode);
            let center = ctu_rect.center();
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(10.0 * zoom.min(4.0)),
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
        _ => format!("{}", value as i64),
    }
}
