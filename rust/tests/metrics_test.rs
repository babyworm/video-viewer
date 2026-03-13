use video_viewer::analysis::metrics::{calculate_frame_difference, calculate_psnr, calculate_ssim};

#[test]
fn test_psnr_identical() {
    let w = 16u32;
    let h = 16u32;
    let n = (w * h * 3) as usize;
    let img: Vec<u8> = vec![100; n];

    let psnr = calculate_psnr(&img, &img, w, h);
    assert!(psnr.is_infinite() || psnr > 100.0);
}

#[test]
fn test_psnr_different() {
    let w = 16u32;
    let h = 16u32;
    let n = (w * h * 3) as usize;
    let img1: Vec<u8> = vec![100; n];
    let img2: Vec<u8> = vec![200; n];

    let psnr = calculate_psnr(&img1, &img2, w, h);
    assert!(psnr.is_finite());
    assert!(psnr > 0.0);
    // Expected: 10 * log10(255^2 / 100^2) = 10 * log10(6.5025) ≈ 8.13
    assert!(psnr < 50.0);
}

#[test]
fn test_ssim_identical() {
    let w = 32u32;
    let h = 32u32;
    let n = (w * h * 3) as usize;
    let img: Vec<u8> = vec![128; n];

    let ssim = calculate_ssim(&img, &img, w, h);
    assert!(
        (ssim - 1.0).abs() < 0.001,
        "SSIM of identical images should be ~1.0, got {}",
        ssim
    );
}

#[test]
fn test_frame_difference() {
    let n = 64 * 3;
    let img1: Vec<u8> = vec![100; n];
    let img2: Vec<u8> = vec![150; n];

    let diff = calculate_frame_difference(&img1, &img2);
    assert!(
        (diff - 50.0).abs() < 0.001,
        "Expected mean abs diff of 50.0, got {}",
        diff
    );
}
