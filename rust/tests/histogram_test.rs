use video_viewer::analysis::histogram::calculate_histogram;

#[test]
fn test_histogram_uniform() {
    let w = 16u32;
    let h = 16u32;
    let pixel_count = (w * h) as usize;
    // All pixels are RGB(128, 128, 128)
    let rgb: Vec<u8> = vec![128; pixel_count * 3];

    let hist = calculate_histogram(&rgb, w, h, "RGB");

    assert_eq!(hist["R"][128], pixel_count as u32);
    assert_eq!(hist["G"][128], pixel_count as u32);
    assert_eq!(hist["B"][128], pixel_count as u32);
    assert_eq!(hist["R"][0], 0);
    assert_eq!(hist["R"][255], 0);
}

#[test]
fn test_histogram_y_mode() {
    let w = 8u32;
    let h = 8u32;
    let pixel_count = (w * h) as usize;
    // All pixels white: RGB(255, 255, 255) → Y ≈ 255
    let rgb: Vec<u8> = vec![255; pixel_count * 3];

    let hist = calculate_histogram(&rgb, w, h, "Y");

    assert!(hist.contains_key("Y"));
    assert_eq!(hist["Y"].len(), 256);

    // All pixels should map to Y=255 (0.2126*255 + 0.7152*255 + 0.0722*255 = 255)
    let total: u32 = hist["Y"].iter().sum();
    assert_eq!(total, pixel_count as u32);
    assert_eq!(hist["Y"][255], pixel_count as u32);
}
