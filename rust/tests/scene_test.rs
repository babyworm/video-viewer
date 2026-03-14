use video_viewer::analysis::scene::{
    detect_scene_changes, detect_scene_changes_with_algorithm,
    histogram_diff, mean_abs_diff, ssim_diff,
    save_scene_list, load_scene_list, SceneAlgorithm,
};

#[test]
fn test_scene_change_identical() {
    let frame = vec![128u8; 10 * 10 * 3];
    let frames = vec![frame.clone(), frame.clone(), frame.clone()];
    let changes = detect_scene_changes(&frames, 10, 10, 45.0);
    assert!(changes.is_empty());
}

#[test]
fn test_scene_change_detected() {
    let frame_a = vec![50u8; 10 * 10 * 3];
    let frame_b = vec![200u8; 10 * 10 * 3];
    let frames = vec![frame_a.clone(), frame_a.clone(), frame_b.clone()];
    let changes = detect_scene_changes(&frames, 10, 10, 45.0);
    assert!(changes.contains(&2));
}

#[test]
fn test_scene_change_threshold() {
    // Difference = 10, threshold = 20 → no change
    let frame_a = vec![100u8; 10 * 10 * 3];
    let frame_b = vec![110u8; 10 * 10 * 3];
    let frames = vec![frame_a, frame_b];
    let changes = detect_scene_changes(&frames, 10, 10, 20.0);
    assert!(changes.is_empty());
}

// --- mean_abs_diff tests ---

#[test]
fn test_mean_abs_diff_identical() {
    let frame = vec![128u8; 10 * 10 * 3];
    assert_eq!(mean_abs_diff(&frame, &frame), 0.0);
}

#[test]
fn test_mean_abs_diff_different() {
    let a = vec![100u8; 10 * 10 * 3];
    let b = vec![200u8; 10 * 10 * 3];
    let diff = mean_abs_diff(&a, &b);
    assert!((diff - 100.0).abs() < 0.01);
}

// --- histogram_diff tests ---

#[test]
fn test_histogram_diff_identical() {
    let frame = vec![128u8; 10 * 10 * 3];
    let diff = histogram_diff(&frame, &frame);
    assert!(diff.abs() < 1e-6, "identical frames should have diff ~0, got {diff}");
}

#[test]
fn test_histogram_diff_different() {
    let a = vec![50u8; 10 * 10 * 3];
    let b = vec![200u8; 10 * 10 * 3];
    let diff = histogram_diff(&a, &b);
    assert!(diff > 0.3, "very different frames should exceed default threshold 0.3, got {diff}");
}

// --- ssim_diff tests ---

#[test]
fn test_ssim_diff_identical() {
    // Need at least 8x8 for one SSIM block
    let frame = vec![128u8; 16 * 16 * 3];
    let diff = ssim_diff(&frame, &frame, 16, 16);
    assert!(diff.abs() < 1e-6, "identical frames should have diff ~0, got {diff}");
}

#[test]
fn test_ssim_diff_different() {
    let a = vec![50u8; 16 * 16 * 3];
    let b = vec![200u8; 16 * 16 * 3];
    let diff = ssim_diff(&a, &b, 16, 16);
    assert!(diff > 40.0, "very different frames should exceed default threshold 40, got {diff}");
}

// --- detect_scene_changes_with_algorithm tests ---

#[test]
fn test_scene_detect_histogram_algorithm() {
    let frame_a = vec![50u8; 10 * 10 * 3];
    let frame_b = vec![200u8; 10 * 10 * 3];
    let frames = vec![frame_a.clone(), frame_a, frame_b];
    let changes = detect_scene_changes_with_algorithm(&frames, 10, 10, 0.3, SceneAlgorithm::Histogram);
    assert!(changes.contains(&2));
}

#[test]
fn test_scene_detect_ssim_algorithm() {
    let frame_a = vec![50u8; 16 * 16 * 3];
    let frame_b = vec![200u8; 16 * 16 * 3];
    let frames = vec![frame_a.clone(), frame_a, frame_b];
    let changes = detect_scene_changes_with_algorithm(&frames, 16, 16, 40.0, SceneAlgorithm::Ssim);
    assert!(changes.contains(&2));
}

#[test]
fn test_scene_list_save_load() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scenes.txt");
    let path_str = path.to_str().unwrap();

    let indices = vec![5, 42, 100];
    save_scene_list(path_str, &indices).unwrap();
    let loaded = load_scene_list(path_str).unwrap();
    assert_eq!(loaded, indices);
}
