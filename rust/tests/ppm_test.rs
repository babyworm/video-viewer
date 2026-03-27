use std::io::Write;
use video_viewer::core::ppm::{parse_ppm_header, write_ppm};

// ============================================================
// PPM Header Parsing
// ============================================================

#[test]
fn test_parse_ppm_p6_header() {
    let data = b"P6\n4 4\n255\n";
    let header = parse_ppm_header(data).expect("should parse valid P6 header");
    assert_eq!(header.width, 4);
    assert_eq!(header.height, 4);
    assert_eq!(header.max_val, 255);
    assert_eq!(header.data_offset, data.len());
}

#[test]
fn test_parse_ppm_header_with_comments() {
    let data = b"P6\n# this is a comment\n4 4\n255\n";
    let header = parse_ppm_header(data).expect("should parse header with comments");
    assert_eq!(header.width, 4);
    assert_eq!(header.height, 4);
    assert_eq!(header.max_val, 255);
}

#[test]
fn test_parse_ppm_header_multiple_spaces() {
    // Some tools use \r\n or extra whitespace
    let data = b"P6\n  4   4  \n255\n";
    let header = parse_ppm_header(data).expect("should parse header with extra whitespace");
    assert_eq!(header.width, 4);
    assert_eq!(header.height, 4);
}

#[test]
fn test_parse_ppm_header_invalid_magic() {
    let data = b"P5\n4 4\n255\n";
    assert!(parse_ppm_header(data).is_err());
}

#[test]
fn test_parse_ppm_header_empty() {
    let data = b"";
    assert!(parse_ppm_header(data).is_err());
}

// ============================================================
// PPM Writing
// ============================================================

#[test]
fn test_write_ppm_basic() {
    // 2x2 red image
    let rgb = vec![
        255, 0, 0, 255, 0, 0,
        255, 0, 0, 255, 0, 0,
    ];
    let mut buf: Vec<u8> = Vec::new();
    write_ppm(&mut buf, 2, 2, &rgb).expect("should write PPM");

    // Parse back
    let header = parse_ppm_header(&buf).expect("should parse written PPM");
    assert_eq!(header.width, 2);
    assert_eq!(header.height, 2);
    assert_eq!(header.max_val, 255);

    // Pixel data after header should match input
    let pixel_data = &buf[header.data_offset..];
    assert_eq!(pixel_data, &rgb[..]);
}

#[test]
fn test_write_ppm_roundtrip() {
    // 4x4 gradient
    let mut rgb = Vec::with_capacity(4 * 4 * 3);
    for i in 0..16 {
        rgb.push((i * 16) as u8); // R
        rgb.push((i * 8) as u8);  // G
        rgb.push((i * 4) as u8);  // B
    }

    let mut buf: Vec<u8> = Vec::new();
    write_ppm(&mut buf, 4, 4, &rgb).unwrap();

    let header = parse_ppm_header(&buf).unwrap();
    let pixel_data = &buf[header.data_offset..];
    assert_eq!(pixel_data, &rgb[..]);
}

// ============================================================
// PPM Read via VideoReader
// ============================================================

#[test]
fn test_reader_opens_ppm_file() {
    use video_viewer::core::reader::VideoReader;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.ppm");

    // 4x4 solid green PPM
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(b"P6\n4 4\n255\n").unwrap();
    let pixel = [0u8, 255, 0]; // green
    for _ in 0..16 {
        f.write_all(&pixel).unwrap();
    }
    drop(f);

    let mut reader = VideoReader::open(path.to_str().unwrap(), 0, 0, "", "BT.601")
        .expect("should open PPM file");
    assert_eq!(reader.width(), 4);
    assert_eq!(reader.height(), 4);
    assert_eq!(reader.total_frames(), 1);

    let raw = reader.seek_frame(0).expect("should seek frame");
    let frame = reader.convert_to_rgb(&raw).expect("should convert frame");
    // All pixels should be green (0, 255, 0)
    for pixel_idx in 0..16 {
        let off = pixel_idx * 3;
        assert_eq!(frame[off], 0, "R should be 0");
        assert_eq!(frame[off + 1], 255, "G should be 255");
        assert_eq!(frame[off + 2], 0, "B should be 0");
    }
}

// ============================================================
// PPM Export via Converter
// ============================================================

#[test]
fn test_convert_i420_to_ppm() {
    use video_viewer::conversion::converter::VideoConverter;

    let dir = tempfile::tempdir().unwrap();

    // Create a 4x4 I420 frame (Y=128, U=128, V=128 → neutral gray)
    let mut frame = Vec::with_capacity(24);
    frame.extend(std::iter::repeat_n(128u8, 16)); // Y
    frame.extend(std::iter::repeat_n(128u8, 4));  // U
    frame.extend(std::iter::repeat_n(128u8, 4));  // V

    let input_path = dir.path().join("input.yuv");
    std::fs::write(&input_path, &frame).unwrap();

    let output_path = dir.path().join("output.ppm");

    let converter = VideoConverter::new();
    let (n, cancelled) = converter
        .convert(
            input_path.to_str().unwrap(),
            (4, 4),
            "I420",
            output_path.to_str().unwrap(),
            "PPM",
            None,
        )
        .expect("conversion should succeed");

    assert_eq!(n, 1);
    assert!(!cancelled);

    // Verify output is a valid PPM
    let output_data = std::fs::read(&output_path).unwrap();
    let header = parse_ppm_header(&output_data).expect("output should be valid PPM");
    assert_eq!(header.width, 4);
    assert_eq!(header.height, 4);

    // Pixel data should be 4*4*3 = 48 bytes of RGB
    let pixel_data = &output_data[header.data_offset..];
    assert_eq!(pixel_data.len(), 48);
}

#[test]
fn test_convert_rgb24_to_ppm() {
    use video_viewer::conversion::converter::VideoConverter;

    let dir = tempfile::tempdir().unwrap();

    // Create a 2x2 RGB24 frame: red, green, blue, white
    let frame = vec![
        255, 0, 0,     // red
        0, 255, 0,     // green
        0, 0, 255,     // blue
        255, 255, 255, // white
    ];

    let input_path = dir.path().join("input.rgb");
    std::fs::write(&input_path, &frame).unwrap();

    let output_path = dir.path().join("output.ppm");

    let converter = VideoConverter::new();
    let (n, cancelled) = converter
        .convert(
            input_path.to_str().unwrap(),
            (2, 2),
            "RGB24",
            output_path.to_str().unwrap(),
            "PPM",
            None,
        )
        .expect("conversion should succeed");

    assert_eq!(n, 1);
    assert!(!cancelled);

    let output_data = std::fs::read(&output_path).unwrap();
    let header = parse_ppm_header(&output_data).unwrap();
    let pixel_data = &output_data[header.data_offset..];

    // RGB24 → PPM should preserve pixel values exactly
    assert_eq!(pixel_data, &frame[..]);
}
