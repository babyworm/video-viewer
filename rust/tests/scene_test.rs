use video_viewer::analysis::scene::{detect_scene_changes, save_scene_list, load_scene_list};

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
