use video_viewer::conversion::converter::resample_chroma;
use video_viewer::conversion::converter::{VideoConverter, extract_yuv_planes, pack_yuv};
use video_viewer::core::formats::get_format_by_name;
use std::io::Write;

fn make_i420_frame(y_val: u8, u_val: u8, v_val: u8) -> Vec<u8> {
    let mut frame = Vec::with_capacity(24);
    frame.extend(std::iter::repeat_n(y_val, 16)); // Y: 4x4
    frame.extend(std::iter::repeat_n(u_val, 4));  // U: 2x2
    frame.extend(std::iter::repeat_n(v_val, 4));  // V: 2x2
    frame
}

fn write_temp_file(dir: &std::path::Path, name: &str, data: &[u8]) -> String {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(data).unwrap();
    path.to_str().unwrap().to_string()
}

// --- resample_chroma ---

#[test]
fn test_resample_chroma_identity() {
    let plane = vec![100u8, 110, 120, 130];
    let result = resample_chroma(&plane, 2, 2, 2, 2);
    assert_eq!(result, plane);
}

#[test]
fn test_resample_chroma_upsample_2x() {
    // 2x2 → 4x4 (420 → 444 chroma upsample) using bilinear interpolation
    // Source:
    //   10  20
    //   30  40
    let plane = vec![
        10, 20,
        30, 40,
    ];
    let result = resample_chroma(&plane, 2, 2, 4, 4);
    assert_eq!(result.len(), 16); // 4x4

    // Bilinear: src_x_f = dx * 2/4 = dx * 0.5
    // (0,0): sx=0, fx=0.0, sy=0, fy=0.0 → 10
    assert_eq!(result[0], 10);
    // (1,0): sx=0, fx=0.5, sy=0, fy=0.0 → 10*0.5 + 20*0.5 = 15
    assert_eq!(result[1], 15);
    // (2,0): sx=1, fx=0.0, sy=0, fy=0.0 → 20
    assert_eq!(result[2], 20);

    // (0,1): sx=0, fx=0.0, sy=0, fy=0.5 → 10*0.5 + 30*0.5 = 20
    assert_eq!(result[4], 20);
    // (1,1): bilinear of all 4 corners → (10+20+30+40)/4 = 25
    assert_eq!(result[5], 25);
}

#[test]
fn test_resample_chroma_downsample_2x() {
    // 4x4 → 2x2 (444 → 420 chroma downsample)
    let plane = vec![
        10, 20, 30, 40,
        50, 60, 70, 80,
        90, 100, 110, 120,
        130, 140, 150, 160,
    ];
    let result = resample_chroma(&plane, 4, 4, 2, 2);
    assert_eq!(result.len(), 4); // 2x2
}

// --- Cross-format conversion: I420 → YV12 ---

#[test]
fn test_convert_i420_to_yv12() {
    let dir = tempfile::tempdir().unwrap();
    let frame = make_i420_frame(0x80, 0x40, 0xC0);
    let input_path = write_temp_file(dir.path(), "input.yuv", &frame);
    let output_path = dir.path().join("output.yuv");
    let output_str = output_path.to_str().unwrap();

    let converter = VideoConverter::new();
    let (n, cancelled) = converter
        .convert(&input_path, (4, 4), "I420", output_str, "YV12", None)
        .expect("conversion failed");

    assert_eq!(n, 1);
    assert!(!cancelled);

    let output_data = std::fs::read(&output_path).unwrap();
    assert_eq!(output_data.len(), 24); // same size as I420

    // Y plane: same
    assert_eq!(&output_data[..16], &[0x80u8; 16]);
    // YV12: V comes first, then U (opposite of I420)
    assert_eq!(&output_data[16..20], &[0xC0u8; 4]); // V
    assert_eq!(&output_data[20..24], &[0x40u8; 4]); // U
}

// --- Cross-format conversion: I420 → 422P (chroma upsample) ---

#[test]
fn test_convert_i420_to_422p() {
    let dir = tempfile::tempdir().unwrap();
    let frame = make_i420_frame(0x80, 0x40, 0xC0);
    let input_path = write_temp_file(dir.path(), "input.yuv", &frame);
    let output_path = dir.path().join("output.yuv");
    let output_str = output_path.to_str().unwrap();

    let converter = VideoConverter::new();
    let (n, cancelled) = converter
        .convert(&input_path, (4, 4), "I420", output_str, "YUV422P", None)
        .expect("conversion failed");

    assert_eq!(n, 1);
    assert!(!cancelled);

    let output_data = std::fs::read(&output_path).unwrap();
    let fmt_422p = get_format_by_name("YUV422P").unwrap();
    let expected_size = fmt_422p.frame_size(4, 4); // 4*4*2 = 32
    assert_eq!(output_data.len(), expected_size);

    // Y plane should be preserved
    assert_eq!(&output_data[..16], &[0x80u8; 16]);
}

// --- Extract/pack roundtrip for NV12 ---

#[test]
fn test_extract_pack_nv12_roundtrip() {
    // NV12: Y(16) + UV interleaved(8) = 24 bytes for 4x4
    let mut frame = vec![0x80u8; 16]; // Y
    // UV interleaved: U,V pairs
    frame.extend_from_slice(&[0x40, 0xC0, 0x40, 0xC0, 0x40, 0xC0, 0x40, 0xC0]);

    let fmt = get_format_by_name("NV12").unwrap();
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
