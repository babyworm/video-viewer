use video_viewer::analysis::vectorscope::calculate_vectorscope;

#[test]
fn test_vectorscope_neutral_gray() {
    // RGB(128, 128, 128) → Cb ≈ 0, Cr ≈ 0 (neutral, centered)
    let w = 4u32;
    let h = 4u32;
    let rgb = vec![128u8; (w * h * 3) as usize];

    let (cb, cr) = calculate_vectorscope(&rgb, w, h);

    assert_eq!(cb.len(), (w * h) as usize);
    assert_eq!(cr.len(), (w * h) as usize);

    // All values should be near 0 (neutral center)
    for &v in &cb {
        assert!(v.abs() < 1.0, "Cb = {} expected ~0", v);
    }
    for &v in &cr {
        assert!(v.abs() < 1.0, "Cr = {} expected ~0", v);
    }
}

#[test]
fn test_vectorscope_pure_red() {
    // RGB(255, 0, 0) → high Cr, negative Cb
    let w = 4u32;
    let h = 4u32;
    let pixel_count = (w * h) as usize;
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for _ in 0..pixel_count {
        rgb.extend_from_slice(&[255, 0, 0]);
    }

    let (cb, cr) = calculate_vectorscope(&rgb, w, h);

    // Cr = 0.5*255 = 127.5
    for &v in &cr {
        assert!(v > 100.0, "Cr = {} expected > 100 for pure red", v);
    }
    // Cb = -0.1146*255 = -29.2
    for &v in &cb {
        assert!(v < 0.0, "Cb = {} expected < 0 for pure red", v);
    }
}

#[test]
fn test_vectorscope_pure_blue() {
    // RGB(0, 0, 255) → high Cb, negative Cr
    let w = 4u32;
    let h = 4u32;
    let pixel_count = (w * h) as usize;
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for _ in 0..pixel_count {
        rgb.extend_from_slice(&[0, 0, 255]);
    }

    let (cb, cr) = calculate_vectorscope(&rgb, w, h);

    // Cb = 0.5*255 = 127.5
    for &v in &cb {
        assert!(v > 100.0, "Cb = {} expected > 100 for pure blue", v);
    }
    // Cr = -0.0458*255 ≈ -11.7
    for &v in &cr {
        assert!(v < 0.0, "Cr = {} expected < 0 for pure blue", v);
    }
}

#[test]
fn test_vectorscope_empty_on_short_buffer() {
    let (cb, cr) = calculate_vectorscope(&[0u8; 10], 8, 8);
    assert!(cb.is_empty());
    assert!(cr.is_empty());
}

#[test]
fn test_vectorscope_subsampling() {
    // Large image should be subsampled (pixel_count > 640*480)
    let w = 1280u32;
    let h = 720u32;
    let pixel_count = (w * h) as usize;
    let rgb = vec![128u8; pixel_count * 3];

    let (cb, cr) = calculate_vectorscope(&rgb, w, h);

    // Should be subsampled: output < total pixels
    assert!(cb.len() < pixel_count, "Expected subsampling, got {} == {}", cb.len(), pixel_count);
    assert!(!cb.is_empty());
    assert_eq!(cb.len(), cr.len());
}
