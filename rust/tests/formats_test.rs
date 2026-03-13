use video_viewer::core::formats::{FormatType, get_all_formats, get_format_by_name, get_format_by_fourcc, get_formats_by_type, get_formats_by_category};

#[test]
fn test_format_type_variants() {
    let types = [
        FormatType::YuvPlanar,
        FormatType::YuvSemiPlanar,
        FormatType::YuvPacked,
        FormatType::Bayer,
        FormatType::Rgb,
        FormatType::Grey,
        FormatType::Compressed,
    ];
    assert_eq!(types.len(), 7);
}

#[test]
fn test_i420_format() {
    let fmt = get_format_by_name("I420").unwrap();
    assert_eq!(fmt.fourcc, "YU12");
    assert_eq!(fmt.format_type, FormatType::YuvPlanar);
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 3 / 2);
}

#[test]
fn test_nv12_format() {
    let fmt = get_format_by_name("NV12").unwrap();
    assert_eq!(fmt.format_type, FormatType::YuvSemiPlanar);
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 3 / 2);
}

#[test]
fn test_yuyv_format() {
    let fmt = get_format_by_name("YUYV").unwrap();
    assert_eq!(fmt.format_type, FormatType::YuvPacked);
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 2);
}

#[test]
fn test_rgb24_format() {
    let fmt = get_format_by_name("RGB24").unwrap();
    assert_eq!(fmt.format_type, FormatType::Rgb);
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 3);
}

#[test]
fn test_bayer_rggb8_format() {
    let fmt = get_format_by_name("Bayer RGGB (8-bit)").unwrap();
    assert_eq!(fmt.format_type, FormatType::Bayer);
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080);
}

#[test]
fn test_bayer_10bit_packed() {
    let fmt = get_format_by_name("Bayer RGGB (10-bit packed)").unwrap();
    assert_eq!(fmt.frame_size(1920, 1080), (1920 * 1080 * 5) / 4);
}

#[test]
fn test_bayer_10bit_unpacked() {
    let fmt = get_format_by_name("Bayer RGGB (10-bit)").unwrap();
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 2);
}

#[test]
fn test_grey_format() {
    let fmt = get_format_by_name("Greyscale (8-bit)").unwrap();
    assert_eq!(fmt.format_type, FormatType::Grey);
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080);
}

#[test]
fn test_grey_16bit() {
    let fmt = get_format_by_name("Greyscale (16-bit)").unwrap();
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 2);
}

#[test]
fn test_p010_format() {
    let fmt = get_format_by_name("P010").unwrap();
    assert_eq!(fmt.format_type, FormatType::YuvSemiPlanar);
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 3);
}

#[test]
fn test_ayuv_format() {
    let fmt = get_format_by_name("AYUV").unwrap();
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 4);
}

#[test]
fn test_y210_format() {
    let fmt = get_format_by_name("Y210").unwrap();
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 4);
}

#[test]
fn test_y41p_format() {
    let fmt = get_format_by_name("Y41P").unwrap();
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 3 / 2);
}

#[test]
fn test_all_formats_count() {
    let all = get_all_formats();
    assert!(all.len() >= 70, "Expected at least 70 formats, got {}", all.len());
}

#[test]
fn test_format_lookup_by_fourcc() {
    let fmt = get_format_by_fourcc("YU12").unwrap();
    assert!(fmt.name.contains("I420"));
}

#[test]
fn test_format_lookup_by_display_name() {
    let fmt = get_format_by_name("I420 (4:2:0) [YU12]").unwrap();
    assert_eq!(fmt.fourcc, "YU12");
}

#[test]
fn test_format_lookup_not_found() {
    assert!(get_format_by_name("NONEXISTENT").is_none());
}

#[test]
fn test_formats_by_type() {
    let yuv_planar = get_formats_by_type(FormatType::YuvPlanar);
    assert_eq!(yuv_planar.len(), 9);

    let rgb = get_formats_by_type(FormatType::Rgb);
    assert_eq!(rgb.len(), 24);

    let bayer = get_formats_by_type(FormatType::Bayer);
    assert_eq!(bayer.len(), 20);

    let grey = get_formats_by_type(FormatType::Grey);
    assert_eq!(grey.len(), 4);
}

#[test]
fn test_formats_by_category() {
    let cats = get_formats_by_category();
    assert!(cats.contains_key("YUV Planar"));
    assert!(cats.contains_key("RGB"));
    assert!(cats.contains_key("Bayer"));
    assert!(cats.contains_key("Grey"));
    // Compressed should not appear
    assert!(!cats.contains_key("Compressed"));
}

#[test]
fn test_yuv410_frame_size() {
    let fmt = get_format_by_name("YUV410").unwrap();
    // YUV410: Y + (W/4)*(H/4)*2
    let w: usize = 1920;
    let h: usize = 1080;
    let expected = w * h + (w / 4) * (h / 4) * 2;
    assert_eq!(fmt.frame_size(1920, 1080), expected);
}
