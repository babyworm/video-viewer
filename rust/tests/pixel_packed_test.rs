use video_viewer::core::formats::get_format_by_fourcc;
use video_viewer::core::pixel::get_pixel_info;

// --- YUYV ---

#[test]
fn test_pixel_info_yuyv_even() {
    // YUYV layout: [Y0 U0 Y1 V0] per pixel pair
    // 4x2 image = 4 pixel pairs = 16 bytes
    let w = 4u32;
    let h = 2u32;
    let mut data = vec![0u8; (w * h * 2) as usize]; // 16 bytes
    // First pair at (0,0)-(1,0): Y0=200, U0=100, Y1=180, V0=150
    data[0] = 200; // Y0
    data[1] = 100; // U0
    data[2] = 180; // Y1
    data[3] = 150; // V0

    let fmt = get_format_by_fourcc("YUYV").expect("YUYV format not found");

    // Even pixel (0, 0)
    let info = get_pixel_info(&data, w, h, fmt, 0, 0, 0);
    assert_eq!(info.components.get("Y").copied(), Some(200));
    assert_eq!(info.components.get("U").copied(), Some(100));
    assert_eq!(info.components.get("V").copied(), Some(150));
}

#[test]
fn test_pixel_info_yuyv_odd() {
    let w = 4u32;
    let h = 2u32;
    let mut data = vec![0u8; (w * h * 2) as usize];
    data[0] = 200; // Y0
    data[1] = 100; // U0
    data[2] = 180; // Y1
    data[3] = 150; // V0

    let fmt = get_format_by_fourcc("YUYV").expect("YUYV format not found");

    // Odd pixel (1, 0) — should pick Y1 but same U/V pair
    let info = get_pixel_info(&data, w, h, fmt, 1, 0, 0);
    assert_eq!(info.components.get("Y").copied(), Some(180));
    assert_eq!(info.components.get("U").copied(), Some(100));
    assert_eq!(info.components.get("V").copied(), Some(150));
}

// --- UYVY ---

#[test]
fn test_pixel_info_uyvy_even() {
    // UYVY layout: [U0 Y0 V0 Y1]
    let w = 4u32;
    let h = 2u32;
    let mut data = vec![0u8; (w * h * 2) as usize];
    data[0] = 100; // U0
    data[1] = 200; // Y0
    data[2] = 150; // V0
    data[3] = 180; // Y1

    let fmt = get_format_by_fourcc("UYVY").expect("UYVY format not found");

    let info = get_pixel_info(&data, w, h, fmt, 0, 0, 0);
    assert_eq!(info.components.get("Y").copied(), Some(200));
    assert_eq!(info.components.get("U").copied(), Some(100));
    assert_eq!(info.components.get("V").copied(), Some(150));
}

#[test]
fn test_pixel_info_uyvy_odd() {
    let w = 4u32;
    let h = 2u32;
    let mut data = vec![0u8; (w * h * 2) as usize];
    data[0] = 100; // U0
    data[1] = 200; // Y0
    data[2] = 150; // V0
    data[3] = 180; // Y1

    let fmt = get_format_by_fourcc("UYVY").expect("UYVY format not found");

    let info = get_pixel_info(&data, w, h, fmt, 1, 0, 0);
    assert_eq!(info.components.get("Y").copied(), Some(180));
    assert_eq!(info.components.get("U").copied(), Some(100));
    assert_eq!(info.components.get("V").copied(), Some(150));
}

// --- NV16 (4:2:2 semi-planar) ---

#[test]
fn test_pixel_info_nv16() {
    // NV16: Y plane (w*h) + interleaved UV plane (w*h)
    let w = 4u32;
    let h = 4u32;
    let y_size = (w * h) as usize; // 16

    let mut data = vec![0u8; y_size * 2]; // 32 bytes
    data[0] = 220; // Y[0,0]
    // UV plane starts at y_size. NV16: UV pairs at half horizontal res
    // (0,0) → c_x=0, c_y=0, uv_idx=0
    data[y_size] = 90;     // U
    data[y_size + 1] = 170; // V

    let fmt = get_format_by_fourcc("NV16").expect("NV16 format not found");
    let info = get_pixel_info(&data, w, h, fmt, 0, 0, 0);

    assert_eq!(info.components.get("Y").copied(), Some(220));
    assert_eq!(info.components.get("U").copied(), Some(90));
    assert_eq!(info.components.get("V").copied(), Some(170));
}
