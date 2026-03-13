/// Calculate waveform data from RGB image.
///
/// # Arguments
/// * `rgb` - Raw RGB24 pixel data
/// * `w` - Image width
/// * `h` - Image height
/// * `channel` - "luma", "r", "g", or "b"
///
/// # Returns
/// 2D vector [256][display_width] — intensity count at each (column, brightness level).
/// Width is downsampled to max 720 for performance.
pub fn calculate_waveform(rgb: &[u8], w: u32, h: u32, channel: &str) -> Vec<Vec<u32>> {
    let pixel_count = (w * h) as usize;
    if rgb.len() < pixel_count * 3 {
        log::warn!("RGB buffer too small for waveform: expected {}, got {}", pixel_count * 3, rgb.len());
        return vec![vec![0u32; 256]; 0];
    }

    let max_display_width: u32 = 720;
    let display_width = w.min(max_display_width) as usize;

    // waveform[intensity_level][column]
    let mut waveform = vec![vec![0u32; display_width]; 256];

    let w = w as usize;
    let h = h as usize;

    for row in 0..h {
        for col in 0..w {
            let pixel_idx = (row * w + col) * 3;
            let r = rgb[pixel_idx] as f64;
            let g = rgb[pixel_idx + 1] as f64;
            let b = rgb[pixel_idx + 2] as f64;

            let value = match channel {
                "luma" => (0.2126 * r + 0.7152 * g + 0.0722 * b).round() as u8,
                "r" => rgb[pixel_idx],
                "g" => rgb[pixel_idx + 1],
                "b" => rgb[pixel_idx + 2],
                _ => panic!("Unknown channel: {}. Use \"luma\", \"r\", \"g\", or \"b\".", channel),
            };

            // Map column to display width
            let display_col = col * display_width / w;
            waveform[value as usize][display_col] += 1;
        }
    }

    waveform
}
