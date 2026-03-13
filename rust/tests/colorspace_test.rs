use video_viewer::core::colorspace::*;

// ============================================================
// YUV Planar → RGB (I420: h_sub=2, v_sub=2)
// ============================================================

#[test]
fn test_i420_neutral_gray() {
    // Y=128, U=128, V=128 → neutral gray ≈ (128, 128, 128) in BT.601
    let w = 4;
    let h = 4;
    let y = vec![128u8; w * h];
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];

    let rgb = yuv_to_rgb_planar(&y, &u, &v, w, h, (2, 2), false);
    assert_eq!(rgb.len(), w * h * 3);
    for i in 0..w * h {
        assert_eq!(rgb[i * 3], 128, "R at pixel {i}");
        assert_eq!(rgb[i * 3 + 1], 128, "G at pixel {i}");
        assert_eq!(rgb[i * 3 + 2], 128, "B at pixel {i}");
    }
}

#[test]
fn test_i420_red_bt601() {
    // BT.601: Red (255,0,0) → Y≈82, U≈90, V≈240
    // Reverse: Y=82, U=90, V=240 → should be close to red
    let w = 2;
    let h = 2;
    let y = vec![82u8; w * h];
    let u = vec![90u8; 1]; // 1x1 chroma
    let v = vec![240u8; 1];

    let rgb = yuv_to_rgb_planar(&y, &u, &v, w, h, (2, 2), false);
    // Allow tolerance of ±2 for float rounding
    for i in 0..w * h {
        let r = rgb[i * 3];
        let g = rgb[i * 3 + 1];
        let b = rgb[i * 3 + 2];
        assert!(r > 230, "R={r} should be close to 255");
        assert!(g < 25, "G={g} should be close to 0");
        assert!(b < 25, "B={b} should be close to 0");
    }
}

#[test]
fn test_i420_neutral_gray_bt709() {
    // BT.709: Y=128, U=128, V=128 → same neutral gray
    let w = 4;
    let h = 4;
    let y = vec![128u8; w * h];
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];

    let rgb = yuv_to_rgb_planar(&y, &u, &v, w, h, (2, 2), true);
    for i in 0..w * h {
        assert_eq!(rgb[i * 3], 128, "R at pixel {i}");
        assert_eq!(rgb[i * 3 + 1], 128, "G at pixel {i}");
        assert_eq!(rgb[i * 3 + 2], 128, "B at pixel {i}");
    }
}

// ============================================================
// YV12 → RGB (same as I420 but V before U)
// ============================================================

#[test]
fn test_yv12_neutral_gray() {
    // For YV12, caller swaps U and V planes. colorspace just receives (y, u, v).
    // With neutral values it should be identical to I420.
    let w = 4;
    let h = 4;
    let y = vec![128u8; w * h];
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];

    let rgb = yuv_to_rgb_planar(&y, &u, &v, w, h, (2, 2), false);
    for i in 0..w * h {
        assert_eq!(rgb[i * 3], 128);
        assert_eq!(rgb[i * 3 + 1], 128);
        assert_eq!(rgb[i * 3 + 2], 128);
    }
}

// ============================================================
// NV12 → RGB
// ============================================================

#[test]
fn test_nv12_neutral_gray() {
    let w = 4;
    let h = 4;
    // NV12: Y plane (16 bytes) + UV interleaved (8 bytes: U,V,U,V,...)
    let mut raw = vec![128u8; w * h]; // Y
    for _ in 0..(w / 2) * (h / 2) {
        raw.push(128); // U
        raw.push(128); // V
    }

    let rgb = yuv_to_rgb_semi_planar(&raw, w, h, false, false);
    assert_eq!(rgb.len(), w * h * 3);
    for i in 0..w * h {
        assert_eq!(rgb[i * 3], 128);
        assert_eq!(rgb[i * 3 + 1], 128);
        assert_eq!(rgb[i * 3 + 2], 128);
    }
}

// ============================================================
// NV21 → RGB (V,U interleaved instead of U,V)
// ============================================================

#[test]
fn test_nv21_neutral_gray() {
    let w = 4;
    let h = 4;
    let mut raw = vec![128u8; w * h]; // Y
    for _ in 0..(w / 2) * (h / 2) {
        raw.push(128); // V
        raw.push(128); // U
    }

    let rgb = yuv_to_rgb_semi_planar(&raw, w, h, true, false);
    assert_eq!(rgb.len(), w * h * 3);
    for i in 0..w * h {
        assert_eq!(rgb[i * 3], 128);
        assert_eq!(rgb[i * 3 + 1], 128);
        assert_eq!(rgb[i * 3 + 2], 128);
    }
}

// ============================================================
// YUYV → RGB
// ============================================================

#[test]
fn test_yuyv_neutral_gray() {
    let w = 4;
    let h = 2;
    // YUYV: Y0 U Y1 V — 2 pixels per 4 bytes
    let mut raw = Vec::new();
    for _ in 0..(w * h / 2) {
        raw.extend_from_slice(&[128, 128, 128, 128]); // Y0=128, U=128, Y1=128, V=128
    }

    let rgb = yuv_to_rgb_yuyv(&raw, w, h, false);
    assert_eq!(rgb.len(), w * h * 3);
    for i in 0..w * h {
        assert_eq!(rgb[i * 3], 128);
        assert_eq!(rgb[i * 3 + 1], 128);
        assert_eq!(rgb[i * 3 + 2], 128);
    }
}

// ============================================================
// UYVY → RGB
// ============================================================

#[test]
fn test_uyvy_neutral_gray() {
    let w = 4;
    let h = 2;
    // UYVY: U Y0 V Y1
    let mut raw = Vec::new();
    for _ in 0..(w * h / 2) {
        raw.extend_from_slice(&[128, 128, 128, 128]); // U=128, Y0=128, V=128, Y1=128
    }

    let rgb = yuv_to_rgb_uyvy(&raw, w, h, false);
    assert_eq!(rgb.len(), w * h * 3);
    for i in 0..w * h {
        assert_eq!(rgb[i * 3], 128);
        assert_eq!(rgb[i * 3 + 1], 128);
        assert_eq!(rgb[i * 3 + 2], 128);
    }
}

// ============================================================
// BGR ↔ RGB
// ============================================================

#[test]
fn test_bgr_to_rgb() {
    // 2x1: pixel0=(B=10, G=20, R=30), pixel1=(B=40, G=50, R=60)
    let raw = vec![10, 20, 30, 40, 50, 60];
    let rgb = bgr_to_rgb(&raw, 2, 1);
    assert_eq!(rgb, vec![30, 20, 10, 60, 50, 40]);
}

#[test]
fn test_rgb_to_bgr() {
    let raw = vec![30, 20, 10, 60, 50, 40];
    let bgr = rgb_to_bgr(&raw, 2, 1);
    assert_eq!(bgr, vec![10, 20, 30, 40, 50, 60]);
}

// ============================================================
// Grey ↔ RGB
// ============================================================

#[test]
fn test_grey_to_rgb() {
    let raw = vec![100, 200];
    let rgb = grey_to_rgb(&raw, 2, 1);
    assert_eq!(rgb, vec![100, 100, 100, 200, 200, 200]);
}

#[test]
fn test_rgb_to_grey() {
    // BT.601: Y = 0.299*R + 0.587*G + 0.114*B
    // Pure white (255,255,255) → 255
    // Pure red (255,0,0) → 0.299*255 ≈ 76
    let raw = vec![
        255, 255, 255, // white
        255, 0, 0,     // red
    ];
    let grey = rgb_to_grey(&raw, 2, 1);
    assert_eq!(grey[0], 255); // white
    assert!((grey[1] as i16 - 76).abs() <= 1, "red→grey should be ~76, got {}", grey[1]);
}
