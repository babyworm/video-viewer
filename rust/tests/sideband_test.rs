use video_viewer::core::sideband::SidebandFile;

// ---------------------------------------------------------------------------
// Test helpers: replicate the writer's encoding logic
// ---------------------------------------------------------------------------

/// Build a sideband binary header for a given CTU count.
fn write_header(buf: &mut Vec<u8>, num_ctus: usize) {
    buf.push(b'I');
    buf.push(b'P');
    buf.push(0x00); // version
    if num_ctus <= 255 {
        buf.push(num_ctus as u8);
    } else {
        buf.push(0xFF);
        buf.push((num_ctus >> 8) as u8);
        buf.push((num_ctus & 0xFF) as u8);
    }
}

/// Build a 20-byte frame param block.
#[allow(clippy::too_many_arguments)]
fn write_frame_params(
    buf: &mut Vec<u8>,
    frame_id: i32,
    scene_class: u8,
    noise_class: u8,
    motion_class: u8,
    denoise_strength: u8,
    sharpen_strength: u8,
    scene_flags: u8,
    frame_qp_bias: i8,
    frame_chroma_cb_bias: i8,
    frame_chroma_cr_bias: i8,
    frame_lambda_scale_q8: u16,
    global_confidence: u8,
) {
    buf.push((frame_id >> 24) as u8);
    buf.push((frame_id >> 16) as u8);
    buf.push((frame_id >> 8) as u8);
    buf.push(frame_id as u8);
    buf.push(0xFF); // valid_fields_mask hi
    buf.push(0xFF); // valid_fields_mask lo
    buf.push(scene_class << 4);
    buf.push(noise_class << 5);
    buf.push(motion_class << 5);
    buf.push(denoise_strength << 3);
    buf.push(sharpen_strength << 3);
    buf.push(scene_flags);
    buf.push(frame_qp_bias as u8);
    buf.push(frame_chroma_cb_bias as u8);
    buf.push(frame_chroma_cr_bias as u8);
    buf.push((frame_lambda_scale_q8 >> 8) as u8);
    buf.push((frame_lambda_scale_q8 & 0xFF) as u8);
    buf.push(global_confidence);
    buf.push(0); // reserved
    buf.push(0); // reserved
}

/// Build a 16-byte CTU param block.
#[allow(clippy::too_many_arguments)]
fn write_ctu_params(
    buf: &mut Vec<u8>,
    activity: u8,
    flatness: u8,
    edge_density: u8,
    noise: u8,
    saliency: u8,
    chroma_importance: u8,
    confidence: u8,
    ctu_flags: u8,
    qp_delta: i8,
    chroma_cb_delta: i8,
    chroma_cr_delta: i8,
    lambda_scale_q8: u16,
    rc_importance_weight: u8,
    sao_prior: i8,
    temporal_stability: u8,
) {
    buf.push(activity << 2);
    buf.push(flatness << 2);
    buf.push(edge_density << 2);
    buf.push(noise << 4);
    buf.push(saliency << 2);
    buf.push(chroma_importance << 3);
    buf.push(confidence << 4);
    buf.push(ctu_flags << 5);
    buf.push(qp_delta as u8);
    buf.push(chroma_cb_delta as u8);
    buf.push(chroma_cr_delta as u8);
    buf.push((lambda_scale_q8 >> 8) as u8);
    buf.push((lambda_scale_q8 & 0xFF) as u8);
    buf.push(rc_importance_weight);
    buf.push(sao_prior as u8);
    buf.push(temporal_stability);
}

/// Build a neutral frame binary (mirrors isp_emulator's write_neutral_frame).
fn build_neutral_frame(frame_id: i32, num_ctus: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    write_header(&mut buf, num_ctus);
    write_frame_params(
        &mut buf,
        frame_id,
        0,   // scene_class
        0,   // noise_class
        0,   // motion_class
        0,   // denoise_strength
        0,   // sharpen_strength
        0,   // scene_flags
        0,   // frame_qp_bias
        0,   // frame_chroma_cb_bias
        0,   // frame_chroma_cr_bias
        256, // lambda = 1.0
        200, // global_confidence
    );
    for _ in 0..num_ctus {
        write_ctu_params(
            &mut buf,
            0,   // activity
            0,   // flatness
            0,   // edge_density
            0,   // noise
            0,   // saliency
            0,   // chroma_importance
            15,  // confidence
            0,   // ctu_flags
            0,   // qp_delta
            0,   // chroma_cb_delta
            0,   // chroma_cr_delta
            256, // lambda = 1.0
            128, // rc_importance_weight = neutral
            0,   // sao_prior
            0,   // temporal_stability
        );
    }
    buf
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_parse_empty_returns_empty() {
    let sb = SidebandFile::from_bytes(&[]).unwrap();
    assert_eq!(sb.num_frames(), 0);
    assert!(sb.frames.is_empty());
}

#[test]
fn test_parse_single_neutral_frame() {
    let bin = build_neutral_frame(0, 4);
    let sb = SidebandFile::from_bytes(&bin).unwrap();

    assert_eq!(sb.num_frames(), 1);
    let f = sb.frame(0).unwrap();
    assert_eq!(f.frame_id, 0);
    assert_eq!(f.valid_fields_mask, 0xFFFF);
    assert_eq!(f.scene_class, 0);
    assert_eq!(f.noise_class, 0);
    assert_eq!(f.motion_class, 0);
    assert_eq!(f.denoise_strength, 0);
    assert_eq!(f.sharpen_strength, 0);
    assert_eq!(f.scene_flags, 0);
    assert_eq!(f.frame_qp_bias, 0);
    assert_eq!(f.frame_chroma_cb_bias, 0);
    assert_eq!(f.frame_chroma_cr_bias, 0);
    assert_eq!(f.frame_lambda_scale_q8, 256);
    assert!((f.lambda_scale() - 1.0).abs() < 1e-6);
    assert_eq!(f.global_confidence, 200);
    assert_eq!(f.num_ctus(), 4);

    for ctu in &f.ctus {
        assert_eq!(ctu.qp_delta, 0);
        assert_eq!(ctu.chroma_cb_delta, 0);
        assert_eq!(ctu.chroma_cr_delta, 0);
        assert_eq!(ctu.lambda_scale_q8, 256);
        assert!((ctu.lambda_scale() - 1.0).abs() < 1e-6);
        assert_eq!(ctu.rc_importance_weight, 128);
        assert_eq!(ctu.confidence, 15);
    }
}

#[test]
fn test_parse_known_values() {
    let mut buf = Vec::new();
    write_header(&mut buf, 1);
    write_frame_params(
        &mut buf, 42, 3, 2, 1, 8, 25, 0x21, 0, 0, 0, 256, 180,
    );
    write_ctu_params(
        &mut buf, 15, 48, 3, 2, 40, 12, 10, 0, -1, 0, 0, 251, 160, 2, 230,
    );

    let sb = SidebandFile::from_bytes(&buf).unwrap();
    assert_eq!(sb.num_frames(), 1);
    let f = sb.frame(0).unwrap();

    assert_eq!(f.frame_id, 42);
    assert_eq!(f.scene_class, 3);
    assert_eq!(f.noise_class, 2);
    assert_eq!(f.motion_class, 1);
    assert_eq!(f.denoise_strength, 8);
    assert_eq!(f.sharpen_strength, 25);
    assert_eq!(f.scene_flags, 0x21);
    assert_eq!(f.global_confidence, 180);

    let ctu = &f.ctus[0];
    assert_eq!(ctu.activity, 15);
    assert_eq!(ctu.flatness, 48);
    assert_eq!(ctu.edge_density, 3);
    assert_eq!(ctu.noise, 2);
    assert_eq!(ctu.saliency, 40);
    assert_eq!(ctu.chroma_importance, 12);
    assert_eq!(ctu.confidence, 10);
    assert_eq!(ctu.temporal_stability, 230);
    assert_eq!(ctu.qp_delta, -1);
    assert_eq!(ctu.chroma_cb_delta, 0);
    assert_eq!(ctu.chroma_cr_delta, 0);
    assert_eq!(ctu.lambda_scale_q8, 251);
    assert!((ctu.lambda_scale() - 0.98046875).abs() < 1e-6);
    assert_eq!(ctu.rc_importance_weight, 160);
    assert_eq!(ctu.sao_prior, 2);
}

#[test]
fn test_parse_extended_header() {
    let num_ctus = 300;
    let bin = build_neutral_frame(0, num_ctus);
    let sb = SidebandFile::from_bytes(&bin).unwrap();

    assert_eq!(sb.num_frames(), 1);
    assert_eq!(sb.frame(0).unwrap().num_ctus(), 300);
}

#[test]
fn test_parse_multiple_frames() {
    let mut buf = Vec::new();
    for i in 0..3 {
        buf.extend_from_slice(&build_neutral_frame(i, 10));
    }

    let sb = SidebandFile::from_bytes(&buf).unwrap();
    assert_eq!(sb.num_frames(), 3);
    for (i, f) in sb.frames.iter().enumerate() {
        assert_eq!(f.frame_id, i as i32);
        assert_eq!(f.num_ctus(), 10);
    }
}

#[test]
fn test_invalid_magic() {
    let mut full = vec![b'X', b'Y', 0x00, 0x01];
    full.extend_from_slice(&[0u8; 36]); // 20 frame + 16 CTU
    let err = SidebandFile::from_bytes(&full).unwrap_err();
    assert!(err.contains("invalid magic"), "got: {}", err);
}

#[test]
fn test_open_nonexistent_file() {
    let err = SidebandFile::open("/tmp/nonexistent_sideband_12345.bin").unwrap_err();
    assert!(err.contains("failed to read"), "got: {}", err);
}

#[test]
fn test_frame_display() {
    let bin = build_neutral_frame(7, 2);
    let sb = SidebandFile::from_bytes(&bin).unwrap();
    let display = format!("{}", sb.frame(0).unwrap());
    assert!(display.contains("id=7"), "got: {}", display);
    assert!(display.contains("ctus=2"), "got: {}", display);
}

#[test]
fn test_ctu_display() {
    let mut buf = Vec::new();
    write_header(&mut buf, 1);
    write_frame_params(&mut buf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 256, 200);
    write_ctu_params(&mut buf, 30, 10, 5, 0, 0, 0, 15, 0, -2, 0, 0, 256, 128, 0, 0);

    let sb = SidebandFile::from_bytes(&buf).unwrap();
    let display = format!("{}", &sb.frame(0).unwrap().ctus[0]);
    assert!(display.contains("act=30"), "got: {}", display);
    assert!(display.contains("qp_delta=-2"), "got: {}", display);
}

#[test]
fn test_frame_out_of_range_returns_none() {
    let bin = build_neutral_frame(0, 1);
    let sb = SidebandFile::from_bytes(&bin).unwrap();
    assert!(sb.frame(0).is_some());
    assert!(sb.frame(1).is_none());
    assert!(sb.frame(999).is_none());
}

#[test]
fn test_zero_ctu_frame() {
    let bin = build_neutral_frame(0, 0);
    let sb = SidebandFile::from_bytes(&bin).unwrap();
    assert_eq!(sb.num_frames(), 1);
    assert_eq!(sb.frame(0).unwrap().num_ctus(), 0);
}

#[test]
fn test_signed_fields() {
    let mut buf = Vec::new();
    write_header(&mut buf, 1);
    write_frame_params(&mut buf, 0, 0, 0, 0, 0, 0, 0, -3, 2, -1, 256, 200);
    write_ctu_params(&mut buf, 0, 0, 0, 0, 0, 0, 15, 0, -3, 2, -1, 230, 128, -2, 0);

    let sb = SidebandFile::from_bytes(&buf).unwrap();
    let f = sb.frame(0).unwrap();
    assert_eq!(f.frame_qp_bias, -3);
    assert_eq!(f.frame_chroma_cb_bias, 2);
    assert_eq!(f.frame_chroma_cr_bias, -1);

    let ctu = &f.ctus[0];
    assert_eq!(ctu.qp_delta, -3);
    assert_eq!(ctu.chroma_cb_delta, 2);
    assert_eq!(ctu.chroma_cr_delta, -1);
    assert_eq!(ctu.sao_prior, -2);
    assert_eq!(ctu.lambda_scale_q8, 230);
}

#[test]
fn test_1080p_ctu_count() {
    // 1080p with 64x64 CTUs: ceil(1920/64) * ceil(1080/64) = 30 * 17 = 510
    let num_ctus = 510;
    let bin = build_neutral_frame(0, num_ctus);
    let sb = SidebandFile::from_bytes(&bin).unwrap();
    assert_eq!(sb.frame(0).unwrap().num_ctus(), 510);
}
