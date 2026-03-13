use video_viewer::core::y4m::{parse_y4m_header, build_frame_offsets};

#[test]
fn test_basic_y4m_header() {
    let data = b"YUV4MPEG2 W720 H576 F25:1 Ip C420\n";
    let hdr = parse_y4m_header(data).unwrap();
    assert_eq!(hdr.width, 720);
    assert_eq!(hdr.height, 576);
    assert_eq!(hdr.fps_num, 25);
    assert_eq!(hdr.fps_den, 1);
    assert_eq!(hdr.colorspace, "420");
    assert_eq!(hdr.interlace, "progressive");
}

#[test]
fn test_y4m_30000_1001() {
    let data = b"YUV4MPEG2 W1920 H1080 F30000:1001 Ip C420\n";
    let hdr = parse_y4m_header(data).unwrap();
    assert_eq!(hdr.fps_num, 30000);
    assert_eq!(hdr.fps_den, 1001);
    let fps = hdr.fps();
    assert!((fps - 29.97).abs() < 0.01, "fps was {fps}");
}

#[test]
fn test_y4m_422() {
    let data = b"YUV4MPEG2 W320 H240 F30:1 Ip C422\n";
    let hdr = parse_y4m_header(data).unwrap();
    assert_eq!(hdr.colorspace, "422");
}

#[test]
fn test_y4m_mono() {
    let data = b"YUV4MPEG2 W320 H240 F30:1 Ip Cmono\n";
    let hdr = parse_y4m_header(data).unwrap();
    assert_eq!(hdr.colorspace, "mono");
}

#[test]
fn test_y4m_colorspace_to_format() {
    let cases: &[(&[u8], &str)] = &[
        (b"YUV4MPEG2 W4 H4 C420\n",  "I420"),
        (b"YUV4MPEG2 W4 H4 C422\n",  "YUV422P"),
        (b"YUV4MPEG2 W4 H4 C444\n",  "YUV444P"),
        (b"YUV4MPEG2 W4 H4 Cmono\n", "Greyscale (8-bit)"),
    ];
    for (data, expected) in cases {
        let hdr = parse_y4m_header(data).unwrap();
        assert_eq!(hdr.to_format_name(), *expected, "colorspace={}", hdr.colorspace);
    }
}

#[test]
fn test_y4m_invalid() {
    let data = b"NOT_Y4M W720 H576\n";
    assert!(parse_y4m_header(data).is_err());
}

#[test]
fn test_y4m_frame_offsets() {
    // Build a minimal Y4M file with 2 frames of 4x4 I420 (24 bytes each).
    let file_header = b"YUV4MPEG2 W4 H4 F25:1 Ip C420\n";
    let frame_header = b"FRAME\n";
    let frame_data = vec![0x80u8; 24]; // 4*4*3/2 = 24

    let mut data = Vec::new();
    data.extend_from_slice(file_header);
    let frame0_offset = data.len() + frame_header.len();
    data.extend_from_slice(frame_header);
    data.extend_from_slice(&frame_data);
    let frame1_offset = data.len() + frame_header.len();
    data.extend_from_slice(frame_header);
    data.extend_from_slice(&frame_data);

    let offsets = build_frame_offsets(&data, 24);
    assert_eq!(offsets.len(), 2);
    assert_eq!(offsets[0], frame0_offset);
    assert_eq!(offsets[1], frame1_offset);
}

#[test]
fn test_y4m_defaults() {
    // No F, C, or I tags — should use defaults.
    let data = b"YUV4MPEG2 W4 H4\n";
    let hdr = parse_y4m_header(data).unwrap();
    assert_eq!(hdr.fps_num, 30);
    assert_eq!(hdr.fps_den, 1);
    assert_eq!(hdr.colorspace, "420");
    assert_eq!(hdr.interlace, "progressive");
}
