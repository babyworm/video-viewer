use video_viewer::core::formats::get_format_by_fourcc;
use video_viewer::core::pixel::get_pixel_info;

// --- I420 (YU12) ---

#[test]
fn test_pixel_info_i420() {
    // 4x4 I420 frame
    // Y plane: 16 bytes, U plane: 4 bytes, V plane: 4 bytes = 24 bytes total
    let w = 4u32;
    let h = 4u32;
    let y_size = (w * h) as usize; // 16
    let uv_size = (w / 2 * h / 2) as usize; // 4

    let mut data = vec![0u8; y_size + uv_size * 2];
    // Set Y[0,0] = 200
    data[0] = 200;
    // U plane starts at y_size; U[0,0] = 100 (covers top-left 2x2 block)
    data[y_size] = 100;
    // V plane starts at y_size + uv_size; V[0,0] = 150
    data[y_size + uv_size] = 150;

    let fmt = get_format_by_fourcc("YU12").expect("YU12 format not found");
    let info = get_pixel_info(&data, w, h, fmt, 0, 0, 0);

    assert_eq!(info.x, 0);
    assert_eq!(info.y, 0);
    assert_eq!(info.components.get("Y").copied(), Some(200));
    assert_eq!(info.components.get("U").copied(), Some(100));
    assert_eq!(info.components.get("V").copied(), Some(150));
}

#[test]
fn test_pixel_info_i420_interior() {
    // Verify a non-origin pixel in I420
    let w = 4u32;
    let h = 4u32;
    let y_size = (w * h) as usize;
    let uv_size = (w / 2 * h / 2) as usize;

    let mut data = vec![128u8; y_size + uv_size * 2];
    // Y at (2, 2) = index 2*4+2 = 10
    data[10] = 77;
    // U at (2,2): c_x=1, c_y=1, c_idx=1*2+1=3 → data[y_size + 3]
    data[y_size + 3] = 55;
    // V at (2,2): data[y_size + uv_size + 3]
    data[y_size + uv_size + 3] = 33;

    let fmt = get_format_by_fourcc("YU12").expect("YU12 format not found");
    let info = get_pixel_info(&data, w, h, fmt, 2, 2, 0);

    assert_eq!(info.components.get("Y").copied(), Some(77));
    assert_eq!(info.components.get("U").copied(), Some(55));
    assert_eq!(info.components.get("V").copied(), Some(33));
}

// --- YV12 ---

#[test]
fn test_pixel_info_yv12() {
    let w = 4u32;
    let h = 4u32;
    let y_size = (w * h) as usize;
    let uv_size = (w / 2 * h / 2) as usize;

    let mut data = vec![0u8; y_size + uv_size * 2];
    data[0] = 180; // Y[0,0]
    // YV12: V plane first, then U
    data[y_size] = 90;      // V[0,0]
    data[y_size + uv_size] = 60; // U[0,0]

    let fmt = get_format_by_fourcc("YV12").expect("YV12 format not found");
    let info = get_pixel_info(&data, w, h, fmt, 0, 0, 0);

    assert_eq!(info.components.get("Y").copied(), Some(180));
    assert_eq!(info.components.get("V").copied(), Some(90));
    assert_eq!(info.components.get("U").copied(), Some(60));
}

// --- NV12 ---

#[test]
fn test_pixel_info_nv12() {
    let w = 4u32;
    let h = 4u32;
    let y_size = (w * h) as usize; // 16

    // NV12: Y plane (16 bytes) + interleaved UV (8 bytes) = 24 bytes
    let mut data = vec![0u8; y_size + y_size / 2];
    data[0] = 210; // Y[0,0]
    // UV interleaved: NV12 has U then V
    // (0,0) → c_x=0, c_y=0, uv_idx=0 → data[y_size+0]=U, data[y_size+1]=V
    data[y_size] = 120;     // U
    data[y_size + 1] = 140; // V

    let fmt = get_format_by_fourcc("NV12").expect("NV12 format not found");
    let info = get_pixel_info(&data, w, h, fmt, 0, 0, 0);

    assert_eq!(info.components.get("Y").copied(), Some(210));
    assert_eq!(info.components.get("U").copied(), Some(120));
    assert_eq!(info.components.get("V").copied(), Some(140));
}

#[test]
fn test_pixel_info_nv21() {
    let w = 4u32;
    let h = 4u32;
    let y_size = (w * h) as usize;

    let mut data = vec![0u8; y_size + y_size / 2];
    data[0] = 50; // Y[0,0]
    // NV21: V then U
    data[y_size] = 70;      // V
    data[y_size + 1] = 80;  // U

    let fmt = get_format_by_fourcc("NV21").expect("NV21 format not found");
    let info = get_pixel_info(&data, w, h, fmt, 0, 0, 0);

    assert_eq!(info.components.get("Y").copied(), Some(50));
    assert_eq!(info.components.get("V").copied(), Some(70));
    assert_eq!(info.components.get("U").copied(), Some(80));
}

// --- RGB24 ---

#[test]
fn test_pixel_info_rgb24() {
    let w = 4u32;
    let h = 4u32;
    let mut data = vec![0u8; (w * h * 3) as usize];
    // Pixel (0,0): offset=0 → R=255, G=128, B=64
    data[0] = 255;
    data[1] = 128;
    data[2] = 64;

    let fmt = get_format_by_fourcc("RGB3").expect("RGB3 format not found");
    let info = get_pixel_info(&data, w, h, fmt, 0, 0, 0);

    assert_eq!(info.components.get("R").copied(), Some(255));
    assert_eq!(info.components.get("G").copied(), Some(128));
    assert_eq!(info.components.get("B").copied(), Some(64));
}

#[test]
fn test_pixel_info_bgr24() {
    let w = 4u32;
    let h = 4u32;
    let mut data = vec![0u8; (w * h * 3) as usize];
    // BGR3: B=10, G=20, R=30 at (0,0)
    data[0] = 10; // B
    data[1] = 20; // G
    data[2] = 30; // R

    let fmt = get_format_by_fourcc("BGR3").expect("BGR3 format not found");
    let info = get_pixel_info(&data, w, h, fmt, 0, 0, 0);

    assert_eq!(info.components.get("B").copied(), Some(10));
    assert_eq!(info.components.get("G").copied(), Some(20));
    assert_eq!(info.components.get("R").copied(), Some(30));
}

// --- Grey ---

#[test]
fn test_pixel_info_grey() {
    let w = 4u32;
    let h = 4u32;
    let mut data = vec![0u8; (w * h) as usize];
    data[0] = 123; // Y[0,0]
    data[5] = 99;  // Y[1,1] = index 1*4+1=5

    let fmt = get_format_by_fourcc("GREY").expect("GREY format not found");

    let info0 = get_pixel_info(&data, w, h, fmt, 0, 0, 0);
    assert_eq!(info0.components.get("Y").copied(), Some(123));

    let info1 = get_pixel_info(&data, w, h, fmt, 1, 1, 0);
    assert_eq!(info1.components.get("Y").copied(), Some(99));
}

// --- raw_hex ---

#[test]
fn test_pixel_info_hex_i420() {
    let w = 4u32;
    let h = 4u32;
    let y_size = (w * h) as usize;
    let uv_size = (w / 2 * h / 2) as usize;
    let mut data = vec![0u8; y_size + uv_size * 2];
    data[0] = 0xC8; // 200 decimal

    let fmt = get_format_by_fourcc("YU12").expect("YU12 format not found");
    let info = get_pixel_info(&data, w, h, fmt, 0, 0, 0);

    assert!(
        info.raw_hex.contains("C8"),
        "raw_hex '{}' should contain 'C8'",
        info.raw_hex
    );
}

#[test]
fn test_pixel_info_hex_rgb24() {
    let w = 2u32;
    let h = 2u32;
    let mut data = vec![0u8; (w * h * 3) as usize];
    data[0] = 0xFF;
    data[1] = 0x80;
    data[2] = 0x40;

    let fmt = get_format_by_fourcc("RGB3").expect("RGB3 format not found");
    let info = get_pixel_info(&data, w, h, fmt, 0, 0, 0);

    // raw_hex for RGB3 is "FF 80 40"
    assert_eq!(info.raw_hex, "FF 80 40");
}

// --- Neighborhood ---

#[test]
fn test_neighborhood_3x3_center() {
    // 3x3 I420 is awkward (odd dims); use 4x4 and check center pixel (1,1)
    let w = 4u32;
    let h = 4u32;
    let y_size = (w * h) as usize;
    let uv_size = (w / 2 * h / 2) as usize;
    let mut data = vec![0u8; y_size + uv_size * 2];
    // Fill Y plane with sequential values for easy checking
    for i in 0..y_size {
        data[i] = i as u8;
    }

    let fmt = get_format_by_fourcc("YU12").expect("YU12 format not found");
    let info = get_pixel_info(&data, w, h, fmt, 1, 1, 0);

    // 3x3 neighborhood centered at (1,1)
    assert_eq!(info.neighborhood.len(), 3);
    assert_eq!(info.neighborhood[0].len(), 3);

    // Center cell [1][1] should be Y[1,1] = index 1*4+1 = 5 → "05"
    assert_eq!(info.neighborhood[1][1], "05");
    // Top-left [0][0] is Y[0,0] = 0 → "00"
    assert_eq!(info.neighborhood[0][0], "00");
}

#[test]
fn test_neighborhood_3x3_edge() {
    let w = 4u32;
    let h = 4u32;
    let y_size = (w * h) as usize;
    let uv_size = (w / 2 * h / 2) as usize;
    let data = vec![42u8; y_size + uv_size * 2];

    let fmt = get_format_by_fourcc("YU12").expect("YU12 format not found");
    // Pixel at top-left corner (0,0) — neighbors to the left and above are OOB
    let info = get_pixel_info(&data, w, h, fmt, 0, 0, 0);

    assert_eq!(info.neighborhood.len(), 3);
    // Row 0 (dy=-1): all "--" since py=-1
    assert!(info.neighborhood[0].iter().all(|s| s == "--"));
    // Row 1 (dy=0): first cell "--" (dx=-1), then valid
    assert_eq!(info.neighborhood[1][0], "--");
    assert_eq!(info.neighborhood[1][1], "2A"); // 42 = 0x2A
}
