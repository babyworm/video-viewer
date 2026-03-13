use video_viewer::core::formats::get_format_by_name;
use video_viewer::conversion::converter::{
    VideoConverter, extract_yuv_planes, pack_yuv,
};
use std::io::Write;

/// Create a minimal I420 frame (4x4):
///   Y plane: 16 bytes, U plane: 4 bytes, V plane: 4 bytes = 24 bytes total
fn make_i420_frame(y_val: u8, u_val: u8, v_val: u8) -> Vec<u8> {
    let mut frame = Vec::with_capacity(24);
    // Y plane: 4x4
    frame.extend(std::iter::repeat(y_val).take(16));
    // U plane: 2x2
    frame.extend(std::iter::repeat(u_val).take(4));
    // V plane: 2x2
    frame.extend(std::iter::repeat(v_val).take(4));
    frame
}

fn write_temp_file(dir: &std::path::Path, name: &str, data: &[u8]) -> String {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(data).unwrap();
    path.to_str().unwrap().to_string()
}

#[test]
fn test_convert_i420_to_nv12() {
    let dir = tempfile::tempdir().unwrap();
    let frame = make_i420_frame(0x80, 0x40, 0xC0);
    let input_path = write_temp_file(dir.path(), "input.yuv", &frame);
    let output_path = dir.path().join("output.yuv");
    let output_str = output_path.to_str().unwrap();

    let converter = VideoConverter::new();
    let (n, cancelled) = converter
        .convert(&input_path, 4, 4, "I420", output_str, "NV12", None)
        .expect("conversion failed");

    assert_eq!(n, 1);
    assert!(!cancelled);

    let output_data = std::fs::read(&output_path).unwrap();
    let nv12_fmt = get_format_by_name("NV12").unwrap();
    let expected_size = nv12_fmt.frame_size(4, 4);
    assert_eq!(output_data.len(), expected_size); // 24 bytes

    // Y plane should be preserved identically
    assert_eq!(&output_data[..16], &[0x80u8; 16]);

    // UV interleaved: NV12 stores U,V pairs
    // 4 samples: U0,V0, U1,V1, U2,V2, U3,V3
    let uv = &output_data[16..];
    assert_eq!(uv.len(), 8);
    for i in 0..4 {
        assert_eq!(uv[i * 2], 0x40, "U sample {} mismatch", i);
        assert_eq!(uv[i * 2 + 1], 0xC0, "V sample {} mismatch", i);
    }
}

#[test]
fn test_convert_same_format() {
    let dir = tempfile::tempdir().unwrap();
    let frame = make_i420_frame(0xAA, 0x55, 0xBB);
    let input_path = write_temp_file(dir.path(), "input.yuv", &frame);
    let output_path = dir.path().join("output.yuv");
    let output_str = output_path.to_str().unwrap();

    let converter = VideoConverter::new();
    let (n, cancelled) = converter
        .convert(&input_path, 4, 4, "I420", output_str, "I420", None)
        .expect("conversion failed");

    assert_eq!(n, 1);
    assert!(!cancelled);

    let output_data = std::fs::read(&output_path).unwrap();
    // Identity copy: output should be byte-identical to input
    assert_eq!(output_data, frame);
}

#[test]
fn test_convert_frame_count() {
    let dir = tempfile::tempdir().unwrap();
    // Two frames of 4x4 I420 = 48 bytes
    let mut data = Vec::with_capacity(48);
    data.extend(make_i420_frame(0x10, 0x20, 0x30));
    data.extend(make_i420_frame(0x50, 0x60, 0x70));
    let input_path = write_temp_file(dir.path(), "input.yuv", &data);
    let output_path = dir.path().join("output.yuv");
    let output_str = output_path.to_str().unwrap();

    let converter = VideoConverter::new();
    let (n, cancelled) = converter
        .convert(&input_path, 4, 4, "I420", output_str, "NV12", None)
        .expect("conversion failed");

    assert_eq!(n, 2);
    assert!(!cancelled);

    let output_data = std::fs::read(&output_path).unwrap();
    let nv12_size = get_format_by_name("NV12").unwrap().frame_size(4, 4);
    assert_eq!(output_data.len(), nv12_size * 2);

    // Verify first frame Y
    assert_eq!(&output_data[..16], &[0x10u8; 16]);
    // Verify second frame Y
    assert_eq!(&output_data[nv12_size..nv12_size + 16], &[0x50u8; 16]);
}

#[test]
fn test_extract_and_pack_roundtrip() {
    let frame = make_i420_frame(0x80, 0x40, 0xC0);
    let fmt = get_format_by_name("I420").unwrap();
    let (y, u, v) = extract_yuv_planes(&frame, 4, 4, fmt);

    assert_eq!(y.len(), 16);
    assert_eq!(u.len(), 4);
    assert_eq!(v.len(), 4);
    assert!(y.iter().all(|&b| b == 0x80));
    assert!(u.iter().all(|&b| b == 0x40));
    assert!(v.iter().all(|&b| b == 0xC0));

    let repacked = pack_yuv(&y, &u, &v, 4, 4, fmt);
    assert_eq!(repacked, frame);
}

#[test]
fn test_progress_cancellation() {
    let dir = tempfile::tempdir().unwrap();
    let mut data = Vec::new();
    for _ in 0..5 {
        data.extend(make_i420_frame(0x80, 0x80, 0x80));
    }
    let input_path = write_temp_file(dir.path(), "input.yuv", &data);
    let output_path = dir.path().join("output.yuv");
    let output_str = output_path.to_str().unwrap();

    let converter = VideoConverter::new();
    // Cancel after 2 frames
    let (n, cancelled) = converter
        .convert(
            &input_path, 4, 4, "I420", output_str, "I420",
            Some(&|current, _total| current < 2),
        )
        .expect("conversion failed");

    assert!(cancelled);
    assert_eq!(n, 2);
}
