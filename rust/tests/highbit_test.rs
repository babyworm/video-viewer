//! Tests for 10-bit (and higher) video format support.

use std::io::Write;
use tempfile::NamedTempFile;
use video_viewer::core::colorspace::*;
use video_viewer::core::formats::{get_format_by_fourcc, get_format_by_name};
use video_viewer::core::pixel::get_pixel_info;
use video_viewer::core::reader::VideoReader;

// ============================================================
// Helper: build raw P010 frame (MSB-aligned, semi-planar 4:2:0)
// ============================================================

/// Build a 4x4 P010 frame with given Y (10-bit), U (10-bit), V (10-bit).
/// P010: Y plane = w*h u16 LE (MSB-aligned), UV plane = interleaved u16 LE pairs.
fn make_p010_frame(w: usize, h: usize, y_val: u16, u_val: u16, v_val: u16) -> Vec<u8> {
    let mut data = Vec::new();
    // Y plane: MSB-aligned → shift left by 6
    for _ in 0..(w * h) {
        data.extend_from_slice(&((y_val << 6) as u16).to_le_bytes());
    }
    // UV interleaved: (w/2)*(h/2) pairs
    for _ in 0..((w / 2) * (h / 2)) {
        data.extend_from_slice(&((u_val << 6) as u16).to_le_bytes());
        data.extend_from_slice(&((v_val << 6) as u16).to_le_bytes());
    }
    data
}

/// Build a 4x4 YUV420P10LE frame (LSB-aligned, fully planar 4:2:0).
fn make_yuv420p10le_frame(w: usize, h: usize, y_val: u16, u_val: u16, v_val: u16) -> Vec<u8> {
    let mut data = Vec::new();
    // Y plane
    for _ in 0..(w * h) {
        data.extend_from_slice(&y_val.to_le_bytes());
    }
    // U plane
    for _ in 0..((w / 2) * (h / 2)) {
        data.extend_from_slice(&u_val.to_le_bytes());
    }
    // V plane
    for _ in 0..((w / 2) * (h / 2)) {
        data.extend_from_slice(&v_val.to_le_bytes());
    }
    data
}

/// Build a 4x2 Y210 frame (MSB-aligned, packed 4:2:2).
/// Layout per pixel pair: [Y0:u16, Cb:u16, Y1:u16, Cr:u16]
fn make_y210_frame(w: usize, h: usize, y_val: u16, u_val: u16, v_val: u16) -> Vec<u8> {
    let mut data = Vec::new();
    for _ in 0..(w / 2 * h) {
        data.extend_from_slice(&((y_val << 6) as u16).to_le_bytes()); // Y0
        data.extend_from_slice(&((u_val << 6) as u16).to_le_bytes()); // Cb
        data.extend_from_slice(&((y_val << 6) as u16).to_le_bytes()); // Y1
        data.extend_from_slice(&((v_val << 6) as u16).to_le_bytes()); // Cr
    }
    data
}

/// Build a 4x4 Grey 10-bit frame (LSB-aligned in u16 LE).
fn make_grey10_frame(w: usize, h: usize, val: u16) -> Vec<u8> {
    let mut data = Vec::new();
    for _ in 0..(w * h) {
        data.extend_from_slice(&val.to_le_bytes());
    }
    data
}

// ============================================================
// Format definition tests
// ============================================================

#[test]
fn test_yuv420p10le_format() {
    let fmt = get_format_by_name("YUV420P10LE").expect("YUV420P10LE not found");
    assert_eq!(fmt.fourcc, "0T20");
    assert_eq!(fmt.bit_depth, 10);
    assert_eq!(fmt.subsampling, (2, 2));
    // frame_size: w*h*3 for 4x4
    assert_eq!(fmt.frame_size(4, 4), 48);
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 3);
}

#[test]
fn test_yuv422p10le_format() {
    let fmt = get_format_by_name("YUV422P10LE").expect("YUV422P10LE not found");
    assert_eq!(fmt.fourcc, "2T22");
    assert_eq!(fmt.bit_depth, 10);
    assert_eq!(fmt.subsampling, (2, 1));
    assert_eq!(fmt.frame_size(4, 4), 64); // w*h*4
}

#[test]
fn test_yuv444p10le_format() {
    let fmt = get_format_by_name("YUV444P10LE").expect("YUV444P10LE not found");
    assert_eq!(fmt.fourcc, "4T44");
    assert_eq!(fmt.bit_depth, 10);
    assert_eq!(fmt.subsampling, (1, 1));
    assert_eq!(fmt.frame_size(4, 4), 96); // w*h*6
}

#[test]
fn test_p210_format() {
    let fmt = get_format_by_name("P210").expect("P210 not found");
    assert_eq!(fmt.fourcc, "P210");
    assert_eq!(fmt.bit_depth, 10);
    assert_eq!(fmt.subsampling, (2, 1));
    assert_eq!(fmt.frame_size(4, 4), 64); // w*h*4
}

#[test]
fn test_p010_frame_size() {
    let fmt = get_format_by_name("P010").expect("P010 not found");
    assert_eq!(fmt.frame_size(4, 4), 48); // w*h*3
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 3);
}

// ============================================================
// Colorspace conversion tests
// ============================================================

#[test]
fn test_yuv420p10le_neutral_gray() {
    // Y=512, U=512, V=512 (10-bit mid-range) → neutral gray
    // 512 >> 2 = 128 → same as 8-bit neutral
    let w = 4;
    let h = 4;
    let raw = make_yuv420p10le_frame(w, h, 512, 512, 512);

    let rgb = yuv_to_rgb_planar_highbit(&raw, w, h, (2, 2), 10, false);
    assert_eq!(rgb.len(), w * h * 3);
    for i in 0..w * h {
        assert_eq!(rgb[i * 3], 128, "R at pixel {i}");
        assert_eq!(rgb[i * 3 + 1], 128, "G at pixel {i}");
        assert_eq!(rgb[i * 3 + 2], 128, "B at pixel {i}");
    }
}

#[test]
fn test_yuv420p10le_black() {
    // Y=0, U=512, V=512 → black (R≈0, G≈0, B≈0)
    let w = 4;
    let h = 4;
    let raw = make_yuv420p10le_frame(w, h, 0, 512, 512);

    let rgb = yuv_to_rgb_planar_highbit(&raw, w, h, (2, 2), 10, false);
    for i in 0..w * h {
        assert!(rgb[i * 3] < 5, "R={} should be near 0", rgb[i * 3]);
        assert!(rgb[i * 3 + 1] < 5, "G={} should be near 0", rgb[i * 3 + 1]);
        assert!(rgb[i * 3 + 2] < 5, "B={} should be near 0", rgb[i * 3 + 2]);
    }
}

#[test]
fn test_p010_neutral_gray() {
    // P010: MSB-aligned semi-planar, Y=512, U=512, V=512
    let w = 4;
    let h = 4;
    let raw = make_p010_frame(w, h, 512, 512, 512);

    let rgb = yuv_to_rgb_semi_planar_highbit(&raw, w, h, (2, 2), 10, false, false);
    assert_eq!(rgb.len(), w * h * 3);
    for i in 0..w * h {
        assert_eq!(rgb[i * 3], 128, "R at pixel {i}");
        assert_eq!(rgb[i * 3 + 1], 128, "G at pixel {i}");
        assert_eq!(rgb[i * 3 + 2], 128, "B at pixel {i}");
    }
}

#[test]
fn test_p010_bt709() {
    let w = 4;
    let h = 4;
    let raw = make_p010_frame(w, h, 512, 512, 512);

    let rgb = yuv_to_rgb_semi_planar_highbit(&raw, w, h, (2, 2), 10, false, true);
    for i in 0..w * h {
        assert_eq!(rgb[i * 3], 128, "R at pixel {i}");
        assert_eq!(rgb[i * 3 + 1], 128, "G at pixel {i}");
        assert_eq!(rgb[i * 3 + 2], 128, "B at pixel {i}");
    }
}

#[test]
fn test_y210_neutral_gray() {
    let w = 4;
    let h = 2;
    let raw = make_y210_frame(w, h, 512, 512, 512);

    let rgb = yuv_to_rgb_y21x(&raw, w, h, false);
    assert_eq!(rgb.len(), w * h * 3);
    for i in 0..w * h {
        assert_eq!(rgb[i * 3], 128, "R at pixel {i}");
        assert_eq!(rgb[i * 3 + 1], 128, "G at pixel {i}");
        assert_eq!(rgb[i * 3 + 2], 128, "B at pixel {i}");
    }
}

#[test]
fn test_grey10_to_rgb() {
    // 10-bit value 512 → 512 >> 2 = 128
    let w = 4;
    let h = 4;
    let raw = make_grey10_frame(w, h, 512);

    let rgb = grey_highbit_to_rgb(&raw, w, h, 10);
    assert_eq!(rgb.len(), w * h * 3);
    for i in 0..w * h {
        assert_eq!(rgb[i * 3], 128, "R at pixel {i}");
        assert_eq!(rgb[i * 3 + 1], 128, "G at pixel {i}");
        assert_eq!(rgb[i * 3 + 2], 128, "B at pixel {i}");
    }
}

#[test]
fn test_grey10_full_range() {
    let w = 2;
    let h = 1;
    // 10-bit 0 → 0, 10-bit 1023 → 255
    let mut raw = Vec::new();
    raw.extend_from_slice(&0u16.to_le_bytes());
    raw.extend_from_slice(&1023u16.to_le_bytes());

    let rgb = grey_highbit_to_rgb(&raw, w, h, 10);
    assert_eq!(rgb[0], 0, "10-bit 0 → 0");
    assert_eq!(rgb[3], 255, "10-bit 1023 → 255");
}

// ============================================================
// Reader integration tests
// ============================================================

#[test]
fn test_reader_p010_convert() {
    let w = 4u32;
    let h = 4u32;
    let frame = make_p010_frame(w as usize, h as usize, 512, 512, 512);

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&frame).unwrap();
    file.flush().unwrap();

    let path = file.path().to_str().unwrap();
    let mut reader = VideoReader::open(path, w, h, "P010", "BT.601")
        .expect("open P010 should succeed");

    assert_eq!(reader.total_frames(), 1);
    let raw = reader.seek_frame(0).expect("seek frame 0");
    let rgb = reader.convert_to_rgb(&raw).expect("P010 convert_to_rgb");
    assert_eq!(rgb.len(), (w * h * 3) as usize);

    // Neutral gray
    for i in 0..(w * h) as usize {
        assert_eq!(rgb[i * 3], 128, "R at pixel {i}");
    }
}

#[test]
fn test_reader_yuv420p10le_convert() {
    let w = 4u32;
    let h = 4u32;
    let frame = make_yuv420p10le_frame(w as usize, h as usize, 512, 512, 512);

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&frame).unwrap();
    file.flush().unwrap();

    let path = file.path().to_str().unwrap();
    let mut reader = VideoReader::open(path, w, h, "YUV420P10LE", "BT.601")
        .expect("open YUV420P10LE should succeed");

    assert_eq!(reader.total_frames(), 1);
    let raw = reader.seek_frame(0).expect("seek frame 0");
    let rgb = reader.convert_to_rgb(&raw).expect("YUV420P10LE convert_to_rgb");
    assert_eq!(rgb.len(), (w * h * 3) as usize);

    for i in 0..(w * h) as usize {
        assert_eq!(rgb[i * 3], 128, "R at pixel {i}");
    }
}

#[test]
fn test_reader_y210_convert() {
    let w = 4u32;
    let h = 2u32;
    let frame = make_y210_frame(w as usize, h as usize, 512, 512, 512);

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&frame).unwrap();
    file.flush().unwrap();

    let path = file.path().to_str().unwrap();
    let mut reader = VideoReader::open(path, w, h, "Y210", "BT.601")
        .expect("open Y210 should succeed");

    assert_eq!(reader.total_frames(), 1);
    let raw = reader.seek_frame(0).expect("seek frame 0");
    let rgb = reader.convert_to_rgb(&raw).expect("Y210 convert_to_rgb");
    assert_eq!(rgb.len(), (w * h * 3) as usize);

    for i in 0..(w * h) as usize {
        assert_eq!(rgb[i * 3], 128, "R at pixel {i}");
    }
}

#[test]
fn test_reader_grey10_convert() {
    let w = 4u32;
    let h = 4u32;
    let frame = make_grey10_frame(w as usize, h as usize, 512);

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&frame).unwrap();
    file.flush().unwrap();

    let path = file.path().to_str().unwrap();
    let mut reader = VideoReader::open(path, w, h, "Greyscale (10-bit)", "BT.601")
        .expect("open Grey 10-bit should succeed");

    assert_eq!(reader.total_frames(), 1);
    let raw = reader.seek_frame(0).expect("seek frame 0");
    let rgb = reader.convert_to_rgb(&raw).expect("Grey10 convert_to_rgb");
    assert_eq!(rgb.len(), (w * h * 3) as usize);

    for i in 0..(w * h) as usize {
        assert_eq!(rgb[i * 3], 128, "grey pixel {i}");
    }
}

// ============================================================
// Pixel inspection tests
// ============================================================

#[test]
fn test_pixel_info_p010() {
    let w = 4u32;
    let h = 4u32;
    // P010 with Y=700 (10-bit), U=300, V=600
    let data = make_p010_frame(w as usize, h as usize, 700, 300, 600);

    let fmt = get_format_by_fourcc("P010").expect("P010 format not found");
    let info = get_pixel_info(&data, w, h, fmt, 0, 0, 0);

    assert_eq!(info.x, 0);
    assert_eq!(info.y, 0);
    assert_eq!(info.components.get("Y").copied(), Some(700));
    assert_eq!(info.components.get("U").copied(), Some(300));
    assert_eq!(info.components.get("V").copied(), Some(600));
}

#[test]
fn test_pixel_info_yuv420p10le() {
    let w = 4u32;
    let h = 4u32;
    let data = make_yuv420p10le_frame(w as usize, h as usize, 800, 400, 500);

    let fmt = get_format_by_fourcc("0T20").expect("0T20 format not found");
    let info = get_pixel_info(&data, w, h, fmt, 0, 0, 0);

    assert_eq!(info.components.get("Y").copied(), Some(800));
    assert_eq!(info.components.get("U").copied(), Some(400));
    assert_eq!(info.components.get("V").copied(), Some(500));
}

#[test]
fn test_pixel_info_yuv420p10le_interior() {
    // Test non-origin pixel (2,2) in 4x4 frame
    let w = 4u32;
    let h = 4u32;
    let data = make_yuv420p10le_frame(w as usize, h as usize, 512, 256, 768);

    let fmt = get_format_by_fourcc("0T20").expect("0T20 format not found");
    let info = get_pixel_info(&data, w, h, fmt, 2, 2, 0);

    // Uniform frame, so all pixels have the same values
    assert_eq!(info.components.get("Y").copied(), Some(512));
    assert_eq!(info.components.get("U").copied(), Some(256));
    assert_eq!(info.components.get("V").copied(), Some(768));
}

#[test]
fn test_pixel_info_y210() {
    let w = 4u32;
    let h = 2u32;
    let data = make_y210_frame(w as usize, h as usize, 600, 400, 500);

    let fmt = get_format_by_fourcc("Y210").expect("Y210 format not found");
    // Even pixel (0,0)
    let info = get_pixel_info(&data, w, h, fmt, 0, 0, 0);
    assert_eq!(info.components.get("Y").copied(), Some(600));
    assert_eq!(info.components.get("U").copied(), Some(400));
    assert_eq!(info.components.get("V").copied(), Some(500));

    // Odd pixel (1,0) — same Y (uniform), shared U/V
    let info2 = get_pixel_info(&data, w, h, fmt, 1, 0, 0);
    assert_eq!(info2.components.get("Y").copied(), Some(600));
    assert_eq!(info2.components.get("U").copied(), Some(400));
    assert_eq!(info2.components.get("V").copied(), Some(500));
}

// ============================================================
// Channel extraction tests
// ============================================================

#[test]
fn test_get_channels_p010() {
    let w = 4u32;
    let h = 4u32;
    let frame = make_p010_frame(w as usize, h as usize, 512, 300, 700);

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&frame).unwrap();
    file.flush().unwrap();

    let path = file.path().to_str().unwrap();
    let mut reader = VideoReader::open(path, w, h, "P010", "BT.601")
        .expect("open P010");

    let raw = reader.seek_frame(0).unwrap();
    let channels = reader.get_channels(&raw);

    assert!(channels.contains_key("Y"));
    assert!(channels.contains_key("U"));
    assert!(channels.contains_key("V"));

    // Y: 512 MSB-aligned → (512 << 6) >> 8 = 128
    let y = &channels["Y"];
    assert_eq!(y.len(), 16);
    assert!(y.iter().all(|&v| v == 128), "Y should be 128, got {:?}", &y[..4]);
}

#[test]
fn test_get_channels_yuv420p10le() {
    let w = 4u32;
    let h = 4u32;
    let frame = make_yuv420p10le_frame(w as usize, h as usize, 512, 256, 768);

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&frame).unwrap();
    file.flush().unwrap();

    let path = file.path().to_str().unwrap();
    let mut reader = VideoReader::open(path, w, h, "YUV420P10LE", "BT.601")
        .expect("open YUV420P10LE");

    let raw = reader.seek_frame(0).unwrap();
    let channels = reader.get_channels(&raw);

    assert!(channels.contains_key("Y"));
    assert!(channels.contains_key("U"));
    assert!(channels.contains_key("V"));

    // Y: 512 LSB, (512 & 0x3FF) >> 2 = 128
    let y = &channels["Y"];
    assert_eq!(y.len(), 16);
    assert!(y.iter().all(|&v| v == 128), "Y should be 128");

    // U upsampled to full res: 256 >> 2 = 64
    let u = &channels["U"];
    assert_eq!(u.len(), 16);
    assert!(u.iter().all(|&v| v == 64), "U should be 64");

    // V upsampled to full res: 768 >> 2 = 192
    let v = &channels["V"];
    assert_eq!(v.len(), 16);
    assert!(v.iter().all(|&v| v == 192), "V should be 192");
}

// ============================================================
// Y4M 10-bit colorspace detection
// ============================================================

#[test]
fn test_y4m_420p10_detection() {
    use video_viewer::core::y4m::parse_y4m_header;

    let header = b"YUV4MPEG2 W1920 H1080 F24:1 C420p10\n";
    let parsed = parse_y4m_header(header).expect("parse Y4M 420p10");
    assert_eq!(parsed.to_format_name(), "YUV420P10LE");
}

#[test]
fn test_y4m_422p10_detection() {
    use video_viewer::core::y4m::parse_y4m_header;

    let header = b"YUV4MPEG2 W1920 H1080 F24:1 C422p10\n";
    let parsed = parse_y4m_header(header).expect("parse Y4M 422p10");
    assert_eq!(parsed.to_format_name(), "YUV422P10LE");
}

#[test]
fn test_y4m_444p10_detection() {
    use video_viewer::core::y4m::parse_y4m_header;

    let header = b"YUV4MPEG2 W1920 H1080 F24:1 C444p10\n";
    let parsed = parse_y4m_header(header).expect("parse Y4M 444p10");
    assert_eq!(parsed.to_format_name(), "YUV444P10LE");
}

#[test]
fn test_y4m_420_still_8bit() {
    use video_viewer::core::y4m::parse_y4m_header;

    let header = b"YUV4MPEG2 W1920 H1080 F24:1 C420\n";
    let parsed = parse_y4m_header(header).expect("parse Y4M 420");
    assert_eq!(parsed.to_format_name(), "I420");
}

// ============================================================
// Filename hints 10-bit aliases
// ============================================================

#[test]
fn test_hints_yuv420p10le() {
    use video_viewer::core::hints::parse_filename_hints;

    let hints = parse_filename_hints("video_1920x1080_yuv420p10le.raw");
    assert_eq!(hints.format.as_deref(), Some("YUV420P10LE"));
    assert_eq!(hints.width, Some(1920));
    assert_eq!(hints.height, Some(1080));
}

#[test]
fn test_hints_p010() {
    use video_viewer::core::hints::parse_filename_hints;

    let hints = parse_filename_hints("stream_1920x1080_p010.raw");
    assert_eq!(hints.format.as_deref(), Some("P010"));
}

#[test]
fn test_hints_y210() {
    use video_viewer::core::hints::parse_filename_hints;

    let hints = parse_filename_hints("capture_720p_y210.yuv");
    assert_eq!(hints.format.as_deref(), Some("Y210"));
}

// ============================================================
// P012 / Y212 / Y216 format tests
// ============================================================

#[test]
fn test_p012_format_and_convert() {
    let fmt = get_format_by_name("P012").expect("P012 not found");
    assert_eq!(fmt.fourcc, "P012");
    assert_eq!(fmt.bit_depth, 12);
    assert_eq!(fmt.frame_size(4, 4), 48); // same as P010

    // P012: MSB-aligned 12-bit → (val << 4) stored in u16 LE
    let w = 4u32;
    let h = 4u32;
    let mut data = Vec::new();
    for _ in 0..(w * h) as usize {
        data.extend_from_slice(&((2048u16 << 4) as u16).to_le_bytes()); // Y=2048 (12-bit mid)
    }
    for _ in 0..((w / 2) * (h / 2)) as usize {
        data.extend_from_slice(&((2048u16 << 4) as u16).to_le_bytes()); // U
        data.extend_from_slice(&((2048u16 << 4) as u16).to_le_bytes()); // V
    }

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&data).unwrap();
    file.flush().unwrap();

    let path = file.path().to_str().unwrap();
    let mut reader = VideoReader::open(path, w, h, "P012", "BT.601").expect("open P012");
    let raw = reader.seek_frame(0).unwrap();
    let rgb = reader.convert_to_rgb(&raw).expect("P012 convert");
    assert_eq!(rgb.len(), (w * h * 3) as usize);
    // 2048 << 4 = 32768, >> 8 = 128
    for i in 0..(w * h) as usize {
        assert_eq!(rgb[i * 3], 128, "R at pixel {i}");
    }
}

#[test]
fn test_y212_format() {
    let fmt = get_format_by_name("Y212").expect("Y212 not found");
    assert_eq!(fmt.fourcc, "Y212");
    assert_eq!(fmt.bit_depth, 12);
    assert_eq!(fmt.frame_size(4, 2), 32); // w*h*4
}

#[test]
fn test_y216_format() {
    let fmt = get_format_by_name("Y216").expect("Y216 not found");
    assert_eq!(fmt.fourcc, "Y216");
    assert_eq!(fmt.bit_depth, 16);
    assert_eq!(fmt.frame_size(4, 2), 32);
}

#[test]
fn test_y212_convert() {
    // Y212: same layout as Y210, 12-bit MSB-aligned
    let w = 4u32;
    let h = 2u32;
    // Build frame: [Y0:u16, Cb:u16, Y1:u16, Cr:u16] per pair, 12-bit MSB → shift left 4
    let mut data = Vec::new();
    for _ in 0..((w / 2) * h) as usize {
        data.extend_from_slice(&((2048u16 << 4) as u16).to_le_bytes()); // Y0
        data.extend_from_slice(&((2048u16 << 4) as u16).to_le_bytes()); // Cb
        data.extend_from_slice(&((2048u16 << 4) as u16).to_le_bytes()); // Y1
        data.extend_from_slice(&((2048u16 << 4) as u16).to_le_bytes()); // Cr
    }

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&data).unwrap();
    file.flush().unwrap();

    let path = file.path().to_str().unwrap();
    let mut reader = VideoReader::open(path, w, h, "Y212", "BT.601").expect("open Y212");
    let raw = reader.seek_frame(0).unwrap();
    let rgb = reader.convert_to_rgb(&raw).expect("Y212 convert");
    for i in 0..(w * h) as usize {
        assert_eq!(rgb[i * 3], 128, "R at pixel {i}");
    }
}

// ============================================================
// NV15 / NV20 packed 10-bit tests
// ============================================================

/// Build a packed 10-bit buffer from a slice of 10-bit values (LE bitstream).
fn pack_10bit_le(values: &[u16]) -> Vec<u8> {
    assert!(values.len() % 4 == 0, "values must be multiple of 4");
    let mut out = Vec::new();
    for chunk in values.chunks(4) {
        let s0 = chunk[0];
        let s1 = chunk[1];
        let s2 = chunk[2];
        let s3 = chunk[3];
        out.push((s0 & 0xFF) as u8);
        out.push((((s0 >> 8) & 0x03) | ((s1 & 0x3F) << 2)) as u8);
        out.push((((s1 >> 6) & 0x0F) | ((s2 & 0x0F) << 4)) as u8);
        out.push((((s2 >> 4) & 0x3F) | ((s3 & 0x03) << 6)) as u8);
        out.push(((s3 >> 2) & 0xFF) as u8);
    }
    out
}

#[test]
fn test_nv15_format() {
    let fmt = get_format_by_name("NV15").expect("NV15 not found");
    assert_eq!(fmt.fourcc, "NV15");
    assert_eq!(fmt.bit_depth, 10);
    assert_eq!(fmt.subsampling, (2, 2));
    // 4x4: y_size=16, frame = 16*15/8 = 30
    assert_eq!(fmt.frame_size(4, 4), 30);
}

#[test]
fn test_nv20_format() {
    let fmt = get_format_by_name("NV20").expect("NV20 not found");
    assert_eq!(fmt.fourcc, "NV20");
    assert_eq!(fmt.bit_depth, 10);
    assert_eq!(fmt.subsampling, (2, 1));
    // 4x4: y_size=16, frame = 16*5/2 = 40
    assert_eq!(fmt.frame_size(4, 4), 40);
}

#[test]
fn test_nv15_neutral_gray() {
    let w = 4usize;
    let h = 4usize;
    // Y plane: 16 samples of 512 (10-bit mid-range)
    let y_vals: Vec<u16> = vec![512; w * h];
    let y_packed = pack_10bit_le(&y_vals);
    // UV plane: (w/2)*(h/2)*2 = 8 samples of 512 (neutral)
    let uv_vals: Vec<u16> = vec![512; (w / 2) * (h / 2) * 2];
    let uv_packed = pack_10bit_le(&uv_vals);

    let mut raw = y_packed;
    raw.extend_from_slice(&uv_packed);

    let rgb = yuv_to_rgb_nv15_nv20(&raw, w, h, (2, 2), false);
    assert_eq!(rgb.len(), w * h * 3);
    for i in 0..w * h {
        assert_eq!(rgb[i * 3], 128, "R at pixel {i}");
        assert_eq!(rgb[i * 3 + 1], 128, "G at pixel {i}");
        assert_eq!(rgb[i * 3 + 2], 128, "B at pixel {i}");
    }
}

#[test]
fn test_nv15_reader() {
    let w = 4u32;
    let h = 4u32;
    let y_packed = pack_10bit_le(&vec![512u16; (w * h) as usize]);
    let uv_packed = pack_10bit_le(&vec![512u16; ((w / 2) * (h / 2) * 2) as usize]);
    let mut frame = y_packed;
    frame.extend_from_slice(&uv_packed);

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&frame).unwrap();
    file.flush().unwrap();

    let path = file.path().to_str().unwrap();
    let mut reader = VideoReader::open(path, w, h, "NV15", "BT.601").expect("open NV15");
    let raw = reader.seek_frame(0).unwrap();
    let rgb = reader.convert_to_rgb(&raw).expect("NV15 convert");
    for i in 0..(w * h) as usize {
        assert_eq!(rgb[i * 3], 128, "pixel {i}");
    }
}

#[test]
fn test_nv20_neutral_gray() {
    let w = 4usize;
    let h = 4usize;
    let y_packed = pack_10bit_le(&vec![512u16; w * h]);
    // NV20: 4:2:2 → UV samples = (w/2)*h*2
    let uv_packed = pack_10bit_le(&vec![512u16; (w / 2) * h * 2]);
    let mut raw = y_packed;
    raw.extend_from_slice(&uv_packed);

    let rgb = yuv_to_rgb_nv15_nv20(&raw, w, h, (2, 1), false);
    for i in 0..w * h {
        assert_eq!(rgb[i * 3], 128, "R at pixel {i}");
    }
}

// ============================================================
// Y10BPACK / Y10P grey packed tests
// ============================================================

#[test]
fn test_y10bpack_format() {
    let fmt = get_format_by_name("Y10B").expect("Y10B not found");
    assert_eq!(fmt.fourcc, "Y10B");
    assert_eq!(fmt.frame_size(4, 4), 20); // 16*5/4
}

#[test]
fn test_y10p_format() {
    let fmt = get_format_by_name("Y10P").expect("Y10P not found");
    assert_eq!(fmt.fourcc, "Y10P");
    assert_eq!(fmt.frame_size(4, 4), 20);
}

/// Build a BE-packed 10-bit grey buffer (Y10BPACK).
fn pack_10bit_be(values: &[u16]) -> Vec<u8> {
    assert!(values.len() % 4 == 0);
    let mut out = Vec::new();
    for chunk in values.chunks(4) {
        let s0 = chunk[0];
        let s1 = chunk[1];
        let s2 = chunk[2];
        let s3 = chunk[3];
        out.push((s0 >> 2) as u8);
        out.push((((s0 & 0x03) << 6) | ((s1 >> 4) & 0x3F)) as u8);
        out.push((((s1 & 0x0F) << 4) | ((s2 >> 6) & 0x0F)) as u8);
        out.push((((s2 & 0x3F) << 2) | ((s3 >> 8) & 0x03)) as u8);
        out.push((s3 & 0xFF) as u8);
    }
    out
}

#[test]
fn test_y10bpack_neutral_gray() {
    let w = 4;
    let h = 4;
    let packed = pack_10bit_be(&vec![512u16; w * h]);

    let rgb = grey_10bpack_to_rgb(&packed, w, h);
    assert_eq!(rgb.len(), w * h * 3);
    for i in 0..w * h {
        assert_eq!(rgb[i * 3], 128, "pixel {i}");
    }
}

#[test]
fn test_y10bpack_reader() {
    let w = 4u32;
    let h = 4u32;
    let packed = pack_10bit_be(&vec![512u16; (w * h) as usize]);

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&packed).unwrap();
    file.flush().unwrap();

    let path = file.path().to_str().unwrap();
    let mut reader = VideoReader::open(path, w, h, "Greyscale (10-bit BE packed)", "BT.601")
        .expect("open Y10BPACK");
    let raw = reader.seek_frame(0).unwrap();
    let rgb = reader.convert_to_rgb(&raw).expect("Y10BPACK convert");
    for i in 0..(w * h) as usize {
        assert_eq!(rgb[i * 3], 128, "pixel {i}");
    }
}

/// Build a MIPI RAW10 packed grey buffer (Y10P).
fn pack_y10p(values: &[u16]) -> Vec<u8> {
    assert!(values.len() % 4 == 0);
    let mut out = Vec::new();
    for chunk in values.chunks(4) {
        // 4 MSB bytes
        for &v in chunk {
            out.push((v >> 2) as u8);
        }
        // 1 LSB byte
        let lsb = ((chunk[0] & 0x03))
            | ((chunk[1] & 0x03) << 2)
            | ((chunk[2] & 0x03) << 4)
            | ((chunk[3] & 0x03) << 6);
        out.push(lsb as u8);
    }
    out
}

#[test]
fn test_y10p_neutral_gray() {
    let w = 4;
    let h = 4;
    let packed = pack_y10p(&vec![512u16; w * h]);

    let rgb = grey_y10p_to_rgb(&packed, w, h);
    assert_eq!(rgb.len(), w * h * 3);
    for i in 0..w * h {
        assert_eq!(rgb[i * 3], 128, "pixel {i}");
    }
}

#[test]
fn test_y10p_reader() {
    let w = 4u32;
    let h = 4u32;
    let packed = pack_y10p(&vec![512u16; (w * h) as usize]);

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&packed).unwrap();
    file.flush().unwrap();

    let path = file.path().to_str().unwrap();
    let mut reader = VideoReader::open(path, w, h, "Greyscale (10-bit MIPI)", "BT.601")
        .expect("open Y10P");
    let raw = reader.seek_frame(0).unwrap();
    let rgb = reader.convert_to_rgb(&raw).expect("Y10P convert");
    for i in 0..(w * h) as usize {
        assert_eq!(rgb[i * 3], 128, "pixel {i}");
    }
}

// ============================================================
// T010 (tiled P010) tests
// ============================================================

#[test]
fn test_t010_format() {
    let fmt = get_format_by_name("T010").expect("T010 not found");
    assert_eq!(fmt.fourcc, "T010");
    assert_eq!(fmt.bit_depth, 10);
    assert_eq!(fmt.frame_size(4, 4), 48); // same as P010
}

#[test]
fn test_t010_convert() {
    // 8x8 frame: Y = 2x2 tiles of 4x4, UV plane (8x4) = 2x1 tiles of 4x4.
    // With uniform data, tiled == linear, so we can reuse make_p010_frame.
    let w = 8u32;
    let h = 8u32;
    let frame = make_p010_frame(w as usize, h as usize, 512, 512, 512);

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&frame).unwrap();
    file.flush().unwrap();

    let path = file.path().to_str().unwrap();
    let mut reader = VideoReader::open(path, w, h, "T010", "BT.601").expect("open T010");
    let raw = reader.seek_frame(0).unwrap();
    let rgb = reader.convert_to_rgb(&raw).expect("T010 convert");
    for i in 0..(w * h) as usize {
        assert_eq!(rgb[i * 3], 128, "pixel {i}");
    }
}

// ============================================================
// Additional hints tests
// ============================================================

#[test]
fn test_hints_p012() {
    use video_viewer::core::hints::parse_filename_hints;
    let hints = parse_filename_hints("video_1080p_p012.raw");
    assert_eq!(hints.format.as_deref(), Some("P012"));
}

#[test]
fn test_hints_nv15() {
    use video_viewer::core::hints::parse_filename_hints;
    let hints = parse_filename_hints("capture_720p_nv15.raw");
    assert_eq!(hints.format.as_deref(), Some("NV15"));
}

#[test]
fn test_hints_y10bpack() {
    use video_viewer::core::hints::parse_filename_hints;
    let hints = parse_filename_hints("sensor_vga_y10bpack.raw");
    assert_eq!(hints.format.as_deref(), Some("Greyscale (10-bit BE packed)"));
}
