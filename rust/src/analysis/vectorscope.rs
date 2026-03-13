/// Calculate vectorscope data from RGB image.
///
/// Converts RGB to YCbCr using BT.709 coefficients and returns
/// Cb and Cr values for scatter-plot visualization.
///
/// # Arguments
/// * `rgb` - Raw RGB24 pixel data
/// * `w` - Image width
/// * `h` - Image height
///
/// # Returns
/// Tuple of (Cb_values, Cr_values) as f32 vectors.
/// Downsamples by subsampling every Nth pixel if total pixels > 640*480.
pub fn calculate_vectorscope(rgb: &[u8], w: u32, h: u32) -> (Vec<f32>, Vec<f32>) {
    let pixel_count = (w * h) as usize;
    if rgb.len() < pixel_count * 3 {
        log::warn!("RGB buffer too small for vectorscope: expected {}, got {}", pixel_count * 3, rgb.len());
        return (Vec::new(), Vec::new());
    }

    let max_pixels = 640 * 480;
    let step = if pixel_count > max_pixels {
        pixel_count / max_pixels
    } else {
        1
    };

    let output_count = pixel_count.div_ceil(step);
    let mut cb_values = Vec::with_capacity(output_count);
    let mut cr_values = Vec::with_capacity(output_count);

    // BT.709 YCbCr conversion:
    // Y  =  0.2126*R + 0.7152*G + 0.0722*B
    // Cb = -0.1146*R - 0.3854*G + 0.5000*B + 128
    // Cr =  0.5000*R - 0.4542*G - 0.0458*B + 128
    for i in (0..pixel_count).step_by(step) {
        let base = i * 3;
        let r = rgb[base] as f32;
        let g = rgb[base + 1] as f32;
        let b = rgb[base + 2] as f32;

        let cb = -0.1146 * r - 0.3854 * g + 0.5000 * b + 128.0;
        let cr = 0.5000 * r - 0.4542 * g - 0.0458 * b + 128.0;

        cb_values.push(cb);
        cr_values.push(cr);
    }

    (cb_values, cr_values)
}
