use video_viewer::core::formats::get_format_by_name;

// --- RGB 16-bit format frame sizes ---

#[test]
fn test_rgb565_frame_size() {
    let fmt = get_format_by_name("RGB565").unwrap();
    // 16 bpp = 2 bytes per pixel
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 2);
}

#[test]
fn test_rgb555_frame_size() {
    let fmt = get_format_by_name("RGB555").unwrap();
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 2);
}

#[test]
fn test_rgb332_frame_size() {
    let fmt = get_format_by_name("RGB332").unwrap();
    // 8 bpp = 1 byte per pixel
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080);
}

#[test]
fn test_rgba32_frame_size() {
    let fmt = get_format_by_name("RGBA32").unwrap();
    // 32 bpp = 4 bytes per pixel
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 4);
}

#[test]
fn test_hsv24_frame_size() {
    let fmt = get_format_by_name("HSV24").unwrap();
    // 24 bpp = 3 bytes per pixel
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 3);
}

// --- Semi-planar frame sizes ---

#[test]
fn test_nv16_frame_size() {
    let fmt = get_format_by_name("NV16").unwrap();
    // 4:2:2 semi-planar: Y(w*h) + UV(w*h) = 2*w*h
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 2);
}

#[test]
fn test_nv24_frame_size() {
    let fmt = get_format_by_name("NV24").unwrap();
    // 4:4:4 semi-planar: Y(w*h) + UV(2*w*h) = 3*w*h
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 3);
}

// --- Planar extra ---

#[test]
fn test_yuv411p_frame_size() {
    let fmt = get_format_by_name("YUV411P").unwrap();
    // 4:1:1: Y(w*h) + U(w/4*h) + V(w/4*h)
    let w: usize = 1920;
    let h: usize = 1080;
    let expected = w * h + (w / 4) * h * 2;
    assert_eq!(fmt.frame_size(1920, 1080), expected);
}

// --- Packed YUV extra ---

#[test]
fn test_vuya_frame_size() {
    let fmt = get_format_by_name("VUYA").unwrap();
    // 4:4:4 packed with alpha = 4 bytes per pixel
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 4);
}
