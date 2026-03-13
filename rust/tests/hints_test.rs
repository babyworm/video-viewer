use video_viewer::core::hints::parse_filename_hints;

#[test]
fn test_resolution_wxh() {
    let hints = parse_filename_hints("test_1920x1080_nv12.yuv");
    assert_eq!(hints.width, Some(1920));
    assert_eq!(hints.height, Some(1080));
}

#[test]
fn test_named_resolution_720p() {
    let hints = parse_filename_hints("video_720p.yuv");
    assert_eq!(hints.width, Some(1280));
    assert_eq!(hints.height, Some(720));
}

#[test]
fn test_format_alias_nv12() {
    let hints = parse_filename_hints("test_1920x1080_nv12.yuv");
    assert_eq!(hints.format, Some("NV12".to_string()));
}

#[test]
fn test_format_alias_i420() {
    let hints = parse_filename_hints("test_i420_cif.yuv");
    assert_eq!(hints.format, Some("I420".to_string()));
}

#[test]
fn test_fps_suffix() {
    let hints = parse_filename_hints("video_30fps.yuv");
    assert_eq!(hints.fps, Some(30.0));
}

#[test]
fn test_bit_depth() {
    let hints = parse_filename_hints("video_10bit.yuv");
    assert_eq!(hints.bit_depth, Some(10));
}

#[test]
fn test_named_resolution_cif() {
    let hints = parse_filename_hints("foreman_cif.yuv");
    assert_eq!(hints.width, Some(352));
    assert_eq!(hints.height, Some(288));
}

#[test]
fn test_named_resolution_4k() {
    let hints = parse_filename_hints("video_4k_nv12.yuv");
    assert_eq!(hints.width, Some(3840));
    assert_eq!(hints.height, Some(2160));
}

#[test]
fn test_path_fallback_resolution() {
    let hints = parse_filename_hints("/data/1920x1080/video.yuv");
    assert_eq!(hints.width, Some(1920));
    assert_eq!(hints.height, Some(1080));
}

#[test]
fn test_no_hints() {
    let hints = parse_filename_hints("video.yuv");
    assert_eq!(hints.width, None);
    assert_eq!(hints.height, None);
    assert_eq!(hints.format, None);
    assert_eq!(hints.fps, None);
    assert_eq!(hints.bit_depth, None);
}
