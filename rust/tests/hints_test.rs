use video_viewer::core::hints::{
    guess_resolution_from_size, parse_filename_hints, resolve_raw_params, NAMED_RESOLUTIONS,
};

#[test]
fn test_resolution_wxh() {
    let hints = parse_filename_hints("test_1920x1080_nv12.yuv");
    assert_eq!(hints.width, Some(1920));
    assert_eq!(hints.height, Some(1080));
}

#[test]
fn test_named_resolution_720p() {
    let hints = parse_filename_hints("video_720p.yuv");
    assert_eq!(hints.width, Some(1280));
    assert_eq!(hints.height, Some(720));
}

#[test]
fn test_format_alias_nv12() {
    let hints = parse_filename_hints("test_1920x1080_nv12.yuv");
    assert_eq!(hints.format, Some("NV12".to_string()));
}

#[test]
fn test_format_alias_i420() {
    let hints = parse_filename_hints("test_i420_cif.yuv");
    assert_eq!(hints.format, Some("I420".to_string()));
}

#[test]
fn test_fps_suffix() {
    let hints = parse_filename_hints("video_30fps.yuv");
    assert_eq!(hints.fps, Some(30.0));
}

#[test]
fn test_bit_depth() {
    let hints = parse_filename_hints("video_10bit.yuv");
    assert_eq!(hints.bit_depth, Some(10));
}

#[test]
fn test_named_resolution_cif() {
    let hints = parse_filename_hints("foreman_cif.yuv");
    assert_eq!(hints.width, Some(352));
    assert_eq!(hints.height, Some(288));
}

#[test]
fn test_named_resolution_4k() {
    let hints = parse_filename_hints("video_4k_nv12.yuv");
    assert_eq!(hints.width, Some(3840));
    assert_eq!(hints.height, Some(2160));
}

#[test]
fn test_path_fallback_resolution() {
    let hints = parse_filename_hints("/data/1920x1080/video.yuv");
    assert_eq!(hints.width, Some(1920));
    assert_eq!(hints.height, Some(1080));
}

#[test]
fn test_no_hints() {
    let hints = parse_filename_hints("video.yuv");
    assert_eq!(hints.width, None);
    assert_eq!(hints.height, None);
    assert_eq!(hints.format, None);
    assert_eq!(hints.fps, None);
    assert_eq!(hints.bit_depth, None);
}

// --- NAMED_RESOLUTIONS table sanity ---

#[test]
fn test_named_resolutions_aliases_unique() {
    // Every alias should appear exactly once across the whole table.
    // A duplicate would silently overwrite in the HashMap build and cause
    // subtle lookup bugs.
    let mut seen: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
    for r in NAMED_RESOLUTIONS {
        for &alias in r.aliases {
            assert!(
                seen.insert(alias),
                "duplicate alias {alias:?} in NAMED_RESOLUTIONS"
            );
        }
    }
}

#[test]
fn test_named_resolutions_sif_roundtrip() {
    // Ensure the SOT table still feeds filename hint lookup.
    let hints = parse_filename_hints("clip_sif.yuv");
    assert_eq!(hints.width, Some(352));
    assert_eq!(hints.height, Some(240));
}

#[test]
fn test_named_resolutions_menu_preset_count() {
    // Menu size should match the 16 original hardcoded VIDEO_SIZE_PRESETS.
    // If a reviewer changes this on purpose, update the expectation.
    let count = NAMED_RESOLUTIONS.iter().filter(|r| r.show_in_menu).count();
    assert_eq!(count, 16, "Video Size menu entry count drifted");
}

// --- File-size guess (recovered from Python v0.1 main_window._guess_resolution) ---

#[test]
fn test_guess_1080p_i420_single_frame() {
    // 1920 × 1080 × 1.5 = 3,110,400 bytes
    let g = guess_resolution_from_size(3_110_400).expect("must match 1080p I420");
    assert_eq!((g.width, g.height, g.format, g.num_frames), (1920, 1080, "I420", 1));
}

#[test]
fn test_guess_vga_i420_ten_frames() {
    let size = 640u64 * 480 * 3 / 2 * 10;
    let g = guess_resolution_from_size(size).expect("must match VGA I420 10f");
    assert_eq!((g.width, g.height, g.format, g.num_frames), (640, 480, "I420", 10));
}

#[test]
fn test_guess_empty_file_is_none() {
    assert!(guess_resolution_from_size(0).is_none());
}

#[test]
fn test_guess_odd_size_is_none() {
    // Prime-ish number unlikely to divide any frame size.
    assert!(guess_resolution_from_size(12_345).is_none());
}

#[test]
fn test_guess_prefers_larger_resolution() {
    // 3,110,400 bytes = 1 frame of 1920×1080 I420 = N frames of smaller sizes too.
    // Larger-first ordering must pick 1920×1080 rather than a 1-frame smaller match.
    let g = guess_resolution_from_size(3_110_400).unwrap();
    assert_eq!(g.width, 1920);
    assert_eq!(g.height, 1080);
}

// --- resolve_raw_params priority ---

#[test]
fn test_resolve_filename_wins_over_guess() {
    // Filename carries WxH + format, so guess must be bypassed (info = None).
    let size = 1920u64 * 1080 * 3 / 2;
    let r = resolve_raw_params("clip_1920x1080_nv12.yuv", Some(size), 100, 100, "I420");
    assert_eq!(r.width, 1920);
    assert_eq!(r.height, 1080);
    assert_eq!(r.format, "NV12");
    assert!(r.info.is_none(), "filename hits shouldn't emit guess info");
}

#[test]
fn test_resolve_guess_fills_missing_filename_hints() {
    // Filename has no resolution tokens; file size matches 640×480 I420 × 10.
    let size = 640u64 * 480 * 3 / 2 * 10;
    let r = resolve_raw_params("unknown_raw.yuv", Some(size), 1920, 1080, "I420");
    assert_eq!(r.width, 640);
    assert_eq!(r.height, 480);
    assert_eq!(r.format, "I420");
    assert!(r.info.is_some(), "a successful guess must surface status info");
}

#[test]
fn test_resolve_defaults_when_no_hints_no_guess() {
    // Nonsense size can't divide any candidate frame size.
    let r = resolve_raw_params("whatever.yuv", Some(12_345), 1920, 1080, "I420");
    assert_eq!(r.width, 1920);
    assert_eq!(r.height, 1080);
    assert_eq!(r.format, "I420");
    assert!(r.info.is_none());
}

#[test]
fn test_resolve_format_hint_survives_guess() {
    // Filename declares NV12 but no WxH; file size matches a candidate.
    // The filename format hint must take precedence over the guess's I420 default.
    let size = 640u64 * 480 * 3 / 2; // matches VGA for both I420 and NV12 (same frame size)
    let r = resolve_raw_params("raw_nv12.yuv", Some(size), 100, 100, "I420");
    assert_eq!(r.format, "NV12", "filename format hint must override guess");
}
