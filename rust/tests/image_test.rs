use std::io::Write;
use tempfile::tempdir;
use video_viewer::core::reader::{is_auto_detect_ext, is_image_ext, VideoReader};

// ===========================================================================
// is_image_ext
// ===========================================================================

#[test]
fn test_is_image_ext_png() {
    assert!(is_image_ext("png"));
}

#[test]
fn test_is_image_ext_bmp() {
    assert!(is_image_ext("bmp"));
}

#[test]
fn test_is_image_ext_jpg() {
    assert!(is_image_ext("jpg"));
    assert!(is_image_ext("jpeg"));
}

#[test]
fn test_is_image_ext_tiff() {
    assert!(is_image_ext("tif"));
    assert!(is_image_ext("tiff"));
}

#[test]
fn test_is_image_ext_negative() {
    assert!(!is_image_ext("yuv"));
    assert!(!is_image_ext("y4m"));
    assert!(!is_image_ext("ppm"));
    assert!(!is_image_ext("raw"));
    assert!(!is_image_ext(""));
}

// ===========================================================================
// is_auto_detect_ext
// ===========================================================================

#[test]
fn test_is_auto_detect_ext_positive() {
    assert!(is_auto_detect_ext("y4m"));
    assert!(is_auto_detect_ext("ppm"));
    assert!(is_auto_detect_ext("png"));
    assert!(is_auto_detect_ext("bmp"));
    assert!(is_auto_detect_ext("jpg"));
    assert!(is_auto_detect_ext("tiff"));
}

#[test]
fn test_is_auto_detect_ext_negative() {
    assert!(!is_auto_detect_ext("yuv"));
    assert!(!is_auto_detect_ext("raw"));
    assert!(!is_auto_detect_ext("rgb"));
    assert!(!is_auto_detect_ext(""));
}

// ===========================================================================
// Error handling: corrupt / empty images
// ===========================================================================

#[test]
fn test_reader_png_zero_byte() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.png");
    std::fs::File::create(&path).unwrap(); // 0-byte file
    assert!(VideoReader::open(path.to_str().unwrap(), 0, 0, "", "BT.601").is_err());
}

#[test]
fn test_reader_png_corrupt() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad.png");
    std::fs::write(&path, b"not a real png file").unwrap();
    assert!(VideoReader::open(path.to_str().unwrap(), 0, 0, "", "BT.601").is_err());
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Create a minimal 4×4 PNG file in the given directory.
/// Returns the file path.
fn write_test_png(dir: &std::path::Path, name: &str, r: u8, g: u8, b: u8) -> String {
    let path = dir.join(name);
    let img = image::RgbImage::from_fn(4, 4, |_x, _y| image::Rgb([r, g, b]));
    img.save(&path).expect("failed to save test PNG");
    path.to_str().unwrap().to_string()
}

/// Create a minimal 4×4 BMP file in the given directory.
fn write_test_bmp(dir: &std::path::Path, name: &str, r: u8, g: u8, b: u8) -> String {
    let path = dir.join(name);
    let img = image::RgbImage::from_fn(4, 4, |_x, _y| image::Rgb([r, g, b]));
    img.save(&path).expect("failed to save test BMP");
    path.to_str().unwrap().to_string()
}

// ===========================================================================
// Single PNG reading
// ===========================================================================

#[test]
fn test_reader_open_single_png() {
    let dir = tempdir().unwrap();
    let path = write_test_png(dir.path(), "single.png", 255, 0, 0);

    let reader = VideoReader::open(&path, 0, 0, "", "BT.601")
        .expect("should open single PNG");
    assert_eq!(reader.width(), 4);
    assert_eq!(reader.height(), 4);
    assert_eq!(reader.total_frames(), 1);
    assert!(reader.format_name().starts_with("RGB24"));
}

#[test]
fn test_reader_png_rgb_data() {
    let dir = tempdir().unwrap();
    let path = write_test_png(dir.path(), "red.png", 200, 100, 50);

    let mut reader = VideoReader::open(&path, 0, 0, "", "BT.601").unwrap();
    let frame = reader.seek_frame(0).unwrap();

    // Each pixel should be RGB(200, 100, 50), 4x4 = 16 pixels
    assert_eq!(frame.len(), 4 * 4 * 3);
    assert_eq!(frame[0], 200); // R
    assert_eq!(frame[1], 100); // G
    assert_eq!(frame[2], 50);  // B
}

#[test]
fn test_reader_png_convert_to_rgb_identity() {
    let dir = tempdir().unwrap();
    let path = write_test_png(dir.path(), "green.png", 0, 255, 0);

    let mut reader = VideoReader::open(&path, 0, 0, "", "BT.601").unwrap();
    let raw = reader.seek_frame(0).unwrap();
    let rgb = reader.convert_to_rgb(&raw).unwrap();

    // RGB24 convert_to_rgb is identity — should be the same data
    assert_eq!(raw, rgb);
}

// ===========================================================================
// PNG sequence detection and reading
// ===========================================================================

#[test]
fn test_reader_png_sequence_detection() {
    let dir = tempdir().unwrap();

    // Create a numbered sequence: frame_000.png through frame_004.png
    for i in 0..5 {
        let val = (i as u8) * 50;
        write_test_png(dir.path(), &format!("frame_{:03}.png", i), val, val, val);
    }

    let first = dir.path().join("frame_000.png");
    let reader = VideoReader::open(first.to_str().unwrap(), 0, 0, "", "BT.601")
        .expect("should open PNG sequence");

    assert_eq!(reader.width(), 4);
    assert_eq!(reader.height(), 4);
    assert_eq!(reader.total_frames(), 5);
    assert!(reader.format_name().starts_with("RGB24"));
}

#[test]
fn test_reader_png_sequence_seek() {
    let dir = tempdir().unwrap();

    // Create 3 frames with distinct colors
    write_test_png(dir.path(), "img_00.png", 10, 0, 0);
    write_test_png(dir.path(), "img_01.png", 20, 0, 0);
    write_test_png(dir.path(), "img_02.png", 30, 0, 0);

    let first = dir.path().join("img_00.png");
    let mut reader = VideoReader::open(first.to_str().unwrap(), 0, 0, "", "BT.601").unwrap();

    // Seek to each frame and check first pixel R value
    let f0 = reader.seek_frame(0).unwrap();
    assert_eq!(f0[0], 10);

    let f1 = reader.seek_frame(1).unwrap();
    assert_eq!(f1[0], 20);

    let f2 = reader.seek_frame(2).unwrap();
    assert_eq!(f2[0], 30);
}

#[test]
fn test_reader_png_sequence_out_of_bounds() {
    let dir = tempdir().unwrap();
    write_test_png(dir.path(), "x_0.png", 0, 0, 0);
    write_test_png(dir.path(), "x_1.png", 0, 0, 0);

    let first = dir.path().join("x_0.png");
    let mut reader = VideoReader::open(first.to_str().unwrap(), 0, 0, "", "BT.601").unwrap();

    assert_eq!(reader.total_frames(), 2);
    assert!(reader.seek_frame(2).is_err());
}

#[test]
fn test_reader_png_sequence_non_contiguous_numbers() {
    let dir = tempdir().unwrap();

    // Sequence with gaps: 0, 5, 10
    write_test_png(dir.path(), "shot_00.png", 100, 0, 0);
    write_test_png(dir.path(), "shot_05.png", 150, 0, 0);
    write_test_png(dir.path(), "shot_10.png", 200, 0, 0);

    let first = dir.path().join("shot_00.png");
    let mut reader = VideoReader::open(first.to_str().unwrap(), 0, 0, "", "BT.601").unwrap();

    // All 3 files should be found, sorted by number
    assert_eq!(reader.total_frames(), 3);
    let f0 = reader.seek_frame(0).unwrap();
    assert_eq!(f0[0], 100);
    let f2 = reader.seek_frame(2).unwrap();
    assert_eq!(f2[0], 200);
}

#[test]
fn test_reader_png_no_sequence_without_digits() {
    let dir = tempdir().unwrap();
    write_test_png(dir.path(), "photo.png", 128, 128, 128);

    let path = dir.path().join("photo.png");
    let reader = VideoReader::open(path.to_str().unwrap(), 0, 0, "", "BT.601").unwrap();

    // No trailing digits → single frame, not a sequence
    assert_eq!(reader.total_frames(), 1);
}

// ===========================================================================
// Edge case: all-digit filename should NOT trigger sequence detection
// ===========================================================================

#[test]
fn test_reader_png_all_digit_filename_no_sequence() {
    let dir = tempdir().unwrap();
    // Files named "0001.png", "0002.png" — stem is entirely digits
    write_test_png(dir.path(), "0001.png", 10, 0, 0);
    write_test_png(dir.path(), "0002.png", 20, 0, 0);

    let path = dir.path().join("0001.png");
    let reader = VideoReader::open(path.to_str().unwrap(), 0, 0, "", "BT.601").unwrap();

    // All-digit stems should NOT be detected as a sequence
    assert_eq!(reader.total_frames(), 1);
}

// ===========================================================================
// Edge case: mismatched dimensions in sequence
// ===========================================================================

#[test]
fn test_reader_png_sequence_dimension_mismatch() {
    let dir = tempdir().unwrap();

    // Frame 0: 4x4
    let img0 = image::RgbImage::from_fn(4, 4, |_, _| image::Rgb([100, 0, 0]));
    img0.save(dir.path().join("mix_00.png")).unwrap();

    // Frame 1: 8x8 (different dimensions!)
    let img1 = image::RgbImage::from_fn(8, 8, |_, _| image::Rgb([200, 0, 0]));
    img1.save(dir.path().join("mix_01.png")).unwrap();

    let path = dir.path().join("mix_00.png");
    let mut reader = VideoReader::open(path.to_str().unwrap(), 0, 0, "", "BT.601").unwrap();

    // Frame 0 should work fine
    assert!(reader.seek_frame(0).is_ok());

    // Frame 1 should fail with dimension mismatch error
    let err = reader.seek_frame(1).unwrap_err();
    assert!(err.contains("dimensions"), "Expected dimension mismatch error, got: {}", err);
}

// ===========================================================================
// BMP reading (other image formats via image crate)
// ===========================================================================

#[test]
fn test_reader_open_bmp() {
    let dir = tempdir().unwrap();
    let path = write_test_bmp(dir.path(), "test.bmp", 0, 0, 255);

    let reader = VideoReader::open(&path, 0, 0, "", "BT.601")
        .expect("should open BMP file");
    assert_eq!(reader.width(), 4);
    assert_eq!(reader.height(), 4);
    assert_eq!(reader.total_frames(), 1);
    assert!(reader.format_name().starts_with("RGB24"));
}

// ===========================================================================
// PPM reading via reader (regression test)
// ===========================================================================

#[test]
fn test_reader_open_ppm_still_works() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.ppm");

    // Write a minimal P6 PPM: 4x4, maxval 255
    let mut file = std::fs::File::create(&path).unwrap();
    write!(file, "P6\n4 4\n255\n").unwrap();
    // 4*4*3 = 48 bytes of pixel data (all red)
    let pixels: Vec<u8> = (0..48)
        .map(|i| if i % 3 == 0 { 255 } else { 0 })
        .collect();
    file.write_all(&pixels).unwrap();
    file.flush().unwrap();

    let reader = VideoReader::open(path.to_str().unwrap(), 0, 0, "", "BT.601")
        .expect("should open PPM file");
    assert_eq!(reader.width(), 4);
    assert_eq!(reader.height(), 4);
    assert_eq!(reader.total_frames(), 1);
    assert!(reader.format_name().starts_with("RGB24"));
}

// ===========================================================================
// Interlace getter
// ===========================================================================

#[test]
fn test_reader_interlace_y4m_progressive() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.y4m");

    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(b"YUV4MPEG2 W4 H4 F25:1 Ip C420\n").unwrap();
    file.write_all(b"FRAME\n").unwrap();
    file.write_all(&[128u8; 24]).unwrap(); // 4x4 I420
    file.flush().unwrap();

    let reader = VideoReader::open(path.to_str().unwrap(), 0, 0, "", "BT.601").unwrap();
    assert_eq!(reader.interlace(), "progressive");
}

#[test]
fn test_reader_interlace_y4m_tff() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("tff.y4m");

    let mut file = std::fs::File::create(&path).unwrap();
    // It = top-field-first
    file.write_all(b"YUV4MPEG2 W4 H4 F25:1 It C420\n").unwrap();
    file.write_all(b"FRAME\n").unwrap();
    file.write_all(&[128u8; 24]).unwrap();
    file.flush().unwrap();

    let reader = VideoReader::open(path.to_str().unwrap(), 0, 0, "", "BT.601").unwrap();
    assert_eq!(reader.interlace(), "tff");
}

#[test]
fn test_reader_interlace_y4m_bff() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bff.y4m");

    let mut file = std::fs::File::create(&path).unwrap();
    // Ib = bottom-field-first
    file.write_all(b"YUV4MPEG2 W4 H4 F25:1 Ib C420\n").unwrap();
    file.write_all(b"FRAME\n").unwrap();
    file.write_all(&[128u8; 24]).unwrap();
    file.flush().unwrap();

    let reader = VideoReader::open(path.to_str().unwrap(), 0, 0, "", "BT.601").unwrap();
    assert_eq!(reader.interlace(), "bff");
}

#[test]
fn test_reader_interlace_y4m_mixed() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mixed.y4m");

    let mut file = std::fs::File::create(&path).unwrap();
    // Im = mixed interlace
    file.write_all(b"YUV4MPEG2 W4 H4 F25:1 Im C420\n").unwrap();
    file.write_all(b"FRAME\n").unwrap();
    file.write_all(&[128u8; 24]).unwrap();
    file.flush().unwrap();

    let reader = VideoReader::open(path.to_str().unwrap(), 0, 0, "", "BT.601").unwrap();
    assert_eq!(reader.interlace(), "mixed");
}

#[test]
fn test_reader_interlace_raw_empty() {
    // Raw files have no interlace info
    let dir = tempdir().unwrap();
    let path = dir.path().join("raw.yuv");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&[128u8; 24]).unwrap(); // 4x4 I420
    file.flush().unwrap();

    let reader = VideoReader::open(path.to_str().unwrap(), 4, 4, "I420", "BT.601").unwrap();
    assert_eq!(reader.interlace(), "");
}

#[test]
fn test_reader_interlace_png_empty() {
    let dir = tempdir().unwrap();
    let path = write_test_png(dir.path(), "test.png", 0, 0, 0);

    let reader = VideoReader::open(&path, 0, 0, "", "BT.601").unwrap();
    assert_eq!(reader.interlace(), "");
}
