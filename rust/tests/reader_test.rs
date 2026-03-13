use std::io::Write;
use tempfile::NamedTempFile;
use video_viewer::core::reader::VideoReader;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write `frames` frames of 4×4 I420 data to a temp file.
///
/// Each frame is 24 bytes: 16 bytes of Y followed by 4 U and 4 V.
/// The Y byte value for frame `i` is `(i as u8 * 10 + 100)` so frames can be
/// distinguished.
fn make_raw_i420(frames: usize) -> (NamedTempFile, Vec<Vec<u8>>) {
    let mut file = NamedTempFile::new().unwrap();
    let mut frame_data = Vec::new();
    for i in 0..frames {
        let y_val = 100u8.wrapping_add(i as u8 * 10);
        // 16 Y bytes + 4 U bytes + 4 V bytes = 24 bytes (4x4 I420)
        let mut frame = vec![y_val; 16]; // Y plane
        frame.extend_from_slice(&[128u8; 4]); // U plane (2x2)
        frame.extend_from_slice(&[128u8; 4]); // V plane (2x2)
        file.write_all(&frame).unwrap();
        frame_data.push(frame);
    }
    file.flush().unwrap();
    (file, frame_data)
}

/// Write a minimal Y4M file with a single 4×4 I420 frame.
fn make_y4m_file() -> NamedTempFile {
    // Use suffix .y4m so VideoReader auto-detects it.
    let mut file = tempfile::Builder::new()
        .suffix(".y4m")
        .tempfile()
        .unwrap();

    // Header line: YUV4MPEG2 W4 H4 F25:1 Ip C420\n
    file.write_all(b"YUV4MPEG2 W4 H4 F25:1 Ip C420\n").unwrap();

    // One frame: frame header + 24 bytes of pixel data
    file.write_all(b"FRAME\n").unwrap();
    file.write_all(&[128u8; 16]).unwrap(); // Y plane
    file.write_all(&[128u8; 4]).unwrap();  // U plane
    file.write_all(&[128u8; 4]).unwrap();  // V plane

    file.flush().unwrap();
    file
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_reader_open_raw() {
    let (file, _) = make_raw_i420(3);
    let path = file.path().to_str().unwrap();

    let reader = VideoReader::open(path, 4, 4, "I420", "BT.601")
        .expect("open should succeed");

    assert_eq!(reader.width(), 4);
    assert_eq!(reader.height(), 4);
    assert_eq!(reader.total_frames(), 3);
    assert!(!reader.is_y4m());
}

#[test]
fn test_reader_seek_frame() {
    let (file, frame_data) = make_raw_i420(3);
    let path = file.path().to_str().unwrap();

    let mut reader = VideoReader::open(path, 4, 4, "I420", "BT.601")
        .expect("open should succeed");

    let f0 = reader.seek_frame(0).expect("seek frame 0");
    assert_eq!(f0, frame_data[0], "frame 0 data mismatch");

    let f1 = reader.seek_frame(1).expect("seek frame 1");
    assert_eq!(f1, frame_data[1], "frame 1 data mismatch");

    // Seeking the same frame again (cache hit) should return identical data.
    let f0_again = reader.seek_frame(0).expect("seek frame 0 again");
    assert_eq!(f0_again, frame_data[0], "cached frame 0 data mismatch");
}

#[test]
fn test_reader_out_of_bounds() {
    let (file, _) = make_raw_i420(3);
    let path = file.path().to_str().unwrap();

    let mut reader = VideoReader::open(path, 4, 4, "I420", "BT.601")
        .expect("open should succeed");

    let result = reader.seek_frame(3); // index == total_frames → out of bounds
    assert!(result.is_err(), "expected Err for out-of-bounds frame");

    let result2 = reader.seek_frame(100);
    assert!(result2.is_err(), "expected Err for far out-of-bounds frame");
}

#[test]
fn test_reader_y4m() {
    let file = make_y4m_file();
    let path = file.path().to_str().unwrap();

    // Pass 0/0 for width/height — should be ignored for Y4M.
    let reader = VideoReader::open(path, 0, 0, "", "BT.601")
        .expect("Y4M open should succeed");

    assert!(reader.is_y4m(), "should be detected as Y4M");
    assert_eq!(reader.width(), 4);
    assert_eq!(reader.height(), 4);
    assert_eq!(reader.total_frames(), 1);
    // FPS from header: 25/1 = 25.0
    assert_eq!(reader.y4m_fps(), Some(25.0));
}

#[test]
fn test_convert_i420_to_rgb() {
    // Y=128, U=128, V=128 → near-gray
    let (file, _) = make_raw_i420(1); // Y=100 for frame 0
    let _path = file.path().to_str().unwrap();

    // Build a custom 4x4 I420 frame with Y=128, U=128, V=128.
    let mut gray_file = tempfile::Builder::new()
        .suffix(".yuv")
        .tempfile()
        .unwrap();
    let frame: Vec<u8> = std::iter::repeat(128u8)
        .take(16 + 4 + 4) // Y(16) + U(4) + V(4)
        .collect();
    gray_file.write_all(&frame).unwrap();
    gray_file.flush().unwrap();

    let gray_path = gray_file.path().to_str().unwrap();
    let mut reader = VideoReader::open(gray_path, 4, 4, "I420", "BT.601")
        .expect("open should succeed");

    let raw = reader.seek_frame(0).expect("seek frame 0");
    let rgb = reader.convert_to_rgb(&raw).expect("convert_to_rgb should succeed");

    // RGB output should be H*W*3 = 48 bytes.
    assert_eq!(rgb.len(), 4 * 4 * 3);

    // Y=128, U=128 (neutral), V=128 (neutral) → approximately gray.
    // Each channel should be in the range [100, 156] (near 128).
    for (i, &v) in rgb.iter().enumerate() {
        assert!(
            v >= 100 && v <= 156,
            "channel[{i}] = {v} out of expected near-gray range [100,156]"
        );
    }
}

#[test]
fn test_get_channels_yuv() {
    let mut gray_file = tempfile::Builder::new()
        .suffix(".yuv")
        .tempfile()
        .unwrap();
    // 4x4 I420: Y=200, U=100, V=150
    let mut frame = vec![200u8; 16]; // Y
    frame.extend_from_slice(&[100u8; 4]); // U (2x2)
    frame.extend_from_slice(&[150u8; 4]); // V (2x2)
    gray_file.write_all(&frame).unwrap();
    gray_file.flush().unwrap();

    let path = gray_file.path().to_str().unwrap();
    let mut reader = VideoReader::open(path, 4, 4, "I420", "BT.601")
        .expect("open should succeed");

    let raw = reader.seek_frame(0).expect("seek frame 0");
    let channels = reader.get_channels(&raw);

    assert!(channels.contains_key("Y"), "missing Y channel");
    assert!(channels.contains_key("U"), "missing U channel");
    assert!(channels.contains_key("V"), "missing V channel");

    // Y channel: 16 pixels, all 200
    let y = &channels["Y"];
    assert_eq!(y.len(), 4 * 4, "Y channel wrong size");
    assert!(y.iter().all(|&v| v == 200), "Y channel values incorrect");

    // U upsampled to full res: 16 pixels, all 100
    let u = &channels["U"];
    assert_eq!(u.len(), 4 * 4, "U channel wrong size (should be upsampled)");
    assert!(u.iter().all(|&v| v == 100), "U channel values incorrect");

    // V upsampled to full res: 16 pixels, all 150
    let v = &channels["V"];
    assert_eq!(v.len(), 4 * 4, "V channel wrong size (should be upsampled)");
    assert!(v.iter().all(|&v| v == 150), "V channel values incorrect");
}
