use video_viewer::analysis::waveform::calculate_waveform;

#[test]
fn test_waveform_luma_uniform() {
    // 8x8 image, all pixels RGB(128, 128, 128) → luma = 128
    let w = 8u32;
    let h = 8u32;
    let rgb = vec![128u8; (w * h * 3) as usize];

    let wf = calculate_waveform(&rgb, w, h, "luma");

    assert_eq!(wf.len(), 256);
    // All hits should be at intensity level 128
    let total_at_128: u32 = wf[128].iter().sum();
    assert_eq!(total_at_128, w * h);

    // No hits at other levels
    let total_at_0: u32 = wf[0].iter().sum();
    assert_eq!(total_at_0, 0);
    let total_at_255: u32 = wf[255].iter().sum();
    assert_eq!(total_at_255, 0);
}

#[test]
fn test_waveform_red_channel() {
    let w = 4u32;
    let h = 4u32;
    let pixel_count = (w * h) as usize;
    // All pixels: R=200, G=100, B=50
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for _ in 0..pixel_count {
        rgb.extend_from_slice(&[200, 100, 50]);
    }

    let wf = calculate_waveform(&rgb, w, h, "r");

    let total_at_200: u32 = wf[200].iter().sum();
    assert_eq!(total_at_200, w * h);
}

#[test]
fn test_waveform_green_channel() {
    let w = 4u32;
    let h = 4u32;
    let pixel_count = (w * h) as usize;
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for _ in 0..pixel_count {
        rgb.extend_from_slice(&[200, 100, 50]);
    }

    let wf = calculate_waveform(&rgb, w, h, "g");

    let total_at_100: u32 = wf[100].iter().sum();
    assert_eq!(total_at_100, w * h);
}

#[test]
fn test_waveform_blue_channel() {
    let w = 4u32;
    let h = 4u32;
    let pixel_count = (w * h) as usize;
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for _ in 0..pixel_count {
        rgb.extend_from_slice(&[200, 100, 50]);
    }

    let wf = calculate_waveform(&rgb, w, h, "b");

    let total_at_50: u32 = wf[50].iter().sum();
    assert_eq!(total_at_50, w * h);
}

#[test]
fn test_waveform_empty_on_short_buffer() {
    let wf = calculate_waveform(&[0u8; 10], 8, 8, "luma");
    // Should return empty (0 columns) due to buffer too small
    assert_eq!(wf.len(), 0);
}

#[test]
fn test_waveform_display_width_cap() {
    // Image wider than 720 → display columns capped at 720
    let w = 1920u32;
    let h = 2u32;
    let rgb = vec![128u8; (w * h * 3) as usize];

    let wf = calculate_waveform(&rgb, w, h, "luma");

    assert_eq!(wf.len(), 256);
    // Column count should be capped at 720
    assert_eq!(wf[128].len(), 720);
}
