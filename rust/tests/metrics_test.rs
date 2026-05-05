use video_viewer::analysis::metrics::{
    calculate_frame_difference, calculate_ms_psnr, calculate_ms_ssim, calculate_psnr,
    calculate_spatial_metric_map, calculate_ssim, calculate_vmaf_neg_proxy, SpatialMetricKind,
};

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
fn test_psnr_empty_image_returns_zero() {
    let psnr = calculate_psnr(&[], &[], 0, 0);
    assert_eq!(psnr, 0.0);
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
fn test_ssim_empty_image_returns_zero() {
    let ssim = calculate_ssim(&[], &[], 0, 0);
    assert_eq!(ssim, 0.0);
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

#[test]
fn test_ms_metrics_identical() {
    let w = 32u32;
    let h = 32u32;
    let n = (w * h * 3) as usize;
    let img: Vec<u8> = vec![128; n];

    let ms_psnr = calculate_ms_psnr(&img, &img, w, h);
    let ms_ssim = calculate_ms_ssim(&img, &img, w, h);
    let vmaf_proxy = calculate_vmaf_neg_proxy(&img, &img, w, h);

    assert!(ms_psnr.is_infinite() || ms_psnr > 90.0);
    assert!((ms_ssim - 1.0).abs() < 0.001);
    assert!(vmaf_proxy > 99.0);
}

#[test]
fn test_ms_metrics_degrade_when_different() {
    let w = 32u32;
    let h = 32u32;
    let n = (w * h * 3) as usize;
    let img1: Vec<u8> = vec![96; n];
    let img2: Vec<u8> = vec![160; n];

    let ms_psnr = calculate_ms_psnr(&img1, &img2, w, h);
    let ms_ssim = calculate_ms_ssim(&img1, &img2, w, h);
    let vmaf_proxy = calculate_vmaf_neg_proxy(&img1, &img2, w, h);

    assert!(ms_psnr.is_finite());
    assert!(ms_psnr < 30.0);
    assert!(ms_ssim < 1.0);
    assert!(vmaf_proxy < 100.0);
}

#[test]
fn test_spatial_metric_map_uses_grid_tiles() {
    let w = 16u32;
    let h = 16u32;
    let n = (w * h * 3) as usize;
    let reference: Vec<u8> = vec![100; n];
    let current: Vec<u8> = vec![110; n];

    let map =
        calculate_spatial_metric_map(&reference, &current, w, h, 8, SpatialMetricKind::SignedDiff);

    assert_eq!(map.tile_size, 8);
    assert_eq!(map.blocks.len(), 4);
    assert!(map.blocks.iter().all(|b| (b.value - 10.0).abs() < 0.001));
}

#[test]
fn test_spatial_metric_map_uses_default_tile_when_grid_off() {
    let w = 128u32;
    let h = 64u32;
    let n = (w * h * 3) as usize;
    let reference: Vec<u8> = vec![80; n];
    let current: Vec<u8> = vec![90; n];

    let map = calculate_spatial_metric_map(
        &reference,
        &current,
        w,
        h,
        0,
        SpatialMetricKind::SignedDiff,
    );

    assert_eq!(map.tile_size, 64);
    assert_eq!(map.blocks.len(), 2);
    assert_eq!((map.blocks[0].w, map.blocks[0].h), (64, 64));
}

#[test]
fn test_spatial_metric_map_reports_default_tile_when_grid_off_without_frames() {
    let map = calculate_spatial_metric_map(&[], &[], 16, 16, 0, SpatialMetricKind::SignedDiff);

    assert_eq!(map.tile_size, 64);
    assert!(map.blocks.is_empty());
}

#[test]
fn test_spatial_metric_map_keeps_partial_edge_tiles() {
    let w = 20u32;
    let h = 18u32;
    let n = (w * h * 3) as usize;
    let reference: Vec<u8> = vec![100; n];
    let current: Vec<u8> = vec![105; n];

    let map = calculate_spatial_metric_map(
        &reference,
        &current,
        w,
        h,
        16,
        SpatialMetricKind::SignedDiff,
    );

    assert_eq!(map.blocks.len(), 4);
    assert!(map.blocks.iter().any(|b| b.x == 16 && b.w == 4));
    assert!(map.blocks.iter().any(|b| b.y == 16 && b.h == 2));
    assert!(map.blocks.iter().all(|b| (b.value - 5.0).abs() < 0.001));
}
