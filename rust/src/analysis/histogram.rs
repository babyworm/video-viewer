use std::collections::HashMap;

/// Calculate histogram from RGB image data.
///
/// # Arguments
/// * `rgb` - Raw RGB24 pixel data (R, G, B, R, G, B, ...)
/// * `w` - Image width
/// * `h` - Image height
/// * `mode` - "RGB" for per-channel histograms, "Y" for BT.709 luma histogram
///
/// # Returns
/// HashMap with channel names as keys and 256-bin count vectors as values.
pub fn calculate_histogram(rgb: &[u8], w: u32, h: u32, mode: &str) -> HashMap<String, Vec<u32>> {
    let pixel_count = (w * h) as usize;
    assert!(
        rgb.len() >= pixel_count * 3,
        "RGB buffer too small: expected at least {}, got {}",
        pixel_count * 3,
        rgb.len()
    );

    let mut result = HashMap::new();

    match mode {
        "RGB" => {
            let mut r_hist = vec![0u32; 256];
            let mut g_hist = vec![0u32; 256];
            let mut b_hist = vec![0u32; 256];

            for i in 0..pixel_count {
                let base = i * 3;
                r_hist[rgb[base] as usize] += 1;
                g_hist[rgb[base + 1] as usize] += 1;
                b_hist[rgb[base + 2] as usize] += 1;
            }

            result.insert("R".to_string(), r_hist);
            result.insert("G".to_string(), g_hist);
            result.insert("B".to_string(), b_hist);
        }
        "Y" => {
            let mut y_hist = vec![0u32; 256];

            for i in 0..pixel_count {
                let base = i * 3;
                let r = rgb[base] as f64;
                let g = rgb[base + 1] as f64;
                let b = rgb[base + 2] as f64;
                // BT.709 luma
                let y = (0.2126 * r + 0.7152 * g + 0.0722 * b).round() as u8;
                y_hist[y as usize] += 1;
            }

            result.insert("Y".to_string(), y_hist);
        }
        _ => {
            panic!("Unknown histogram mode: {}. Use \"RGB\" or \"Y\".", mode);
        }
    }

    result
}
