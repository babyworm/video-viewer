use video_viewer::core::hints::{
    guess_all_resolutions_no_hint, guess_resolution_from_size,
    guess_resolutions_with_frame_count, parse_filename_hints, resolve_raw_params,
    resolve_raw_params_for_dimensions, NAMED_RESOLUTIONS,
};
use video_viewer::ui::dialogs::{OpenFileDialog, ParametersDialog};
use video_viewer::ui::navigation::FPS_OPTIONS;

#[test]
fn test_resolution_wxh() {
    let hints = parse_filename_hints("test_1920x1080_nv12.yuv");
    assert_eq!(hints.width, Some(1920));
    assert_eq!(hints.height, Some(1080));
}

#[test]
fn test_jv_ctc_filename_hints() {
    let cases = [
        ("B_BasketballDrive_1920x1080_50.yuv", 1920, 1080, 50),
        ("B_BQTerrace_1920x1080_60.yuv", 1920, 1080, 60),
        ("B_Cactus_1920x1080_50.yuv", 1920, 1080, 50),
        ("B_Kimono_1920x1080_24.yuv", 1920, 1080, 24),
        ("B_ParkScene_1920x1080_24.yuv", 1920, 1080, 24),
        ("C_BasketballDrill_832x480_50.yuv", 832, 480, 50),
        ("C_BQMall_832x480_60.yuv", 832, 480, 60),
        ("C_PartyScene_832x480_50.yuv", 832, 480, 50),
        ("C_RaceHorses_832x480_30.yuv", 832, 480, 30),
        ("D_BasketballPass_416x240_50.yuv", 416, 240, 50),
        ("D_BlowingBubbles_416x240_50.yuv", 416, 240, 50),
        ("D_BQSquare_416x240_60.yuv", 416, 240, 60),
        ("D_RaceHorses_416x240_30.yuv", 416, 240, 30),
        ("E_FourPeople_1280x720_60.yuv", 1280, 720, 60),
        ("E_Johnny_1280x720_60.yuv", 1280, 720, 60),
        ("E_KristenAndSara_1280x720_60.yuv", 1280, 720, 60),
    ];

    for (filename, width, height, fps) in cases {
        let hints = parse_filename_hints(filename);
        assert_eq!(
            (hints.width, hints.height),
            (Some(width), Some(height)),
            "{filename}"
        );
        assert_eq!(hints.fps, Some(fps as f64), "{filename}");
    }
}

#[test]
fn test_playback_fps_options_cover_jv_ctc() {
    for fps in [24, 30, 50, 60] {
        assert!(FPS_OPTIONS.contains(&fps), "missing {fps} FPS option");
    }
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
    // Originally 16 (matching the Python VIDEO_SIZE_PRESETS); +1 after HEVC CTC
    // Class C (832×480) was added. Update the expectation only with intent.
    let count = NAMED_RESOLUTIONS.iter().filter(|r| r.show_in_menu).count();
    assert_eq!(count, 17, "Video Size menu entry count drifted");
}

#[test]
fn test_named_resolution_hevc_class_c_present() {
    // HEVC CTC Class C (BasketballDrill / BQMall / PartyScene / RaceHorses).
    let entry = NAMED_RESOLUTIONS
        .iter()
        .find(|r| r.width == 832 && r.height == 480)
        .expect("HEVC class C resolution missing from NAMED_RESOLUTIONS");
    assert!(entry.show_in_menu, "HEVC C must appear in Video Size menu");
    assert!(entry.aliases.contains(&"classc"));
}

// --- Guess-with-frame-count (View → Video Size → Guess with hint…) ---

#[test]
fn test_guess_with_frames_returns_matching_candidates() {
    // 1920×1080 I420 = 3,110,400 bytes/frame; pick 5 frames.
    let size = 1920u64 * 1080 * 3 / 2 * 5;
    let cands = guess_resolutions_with_frame_count(size, 5);
    assert!(
        cands.iter().any(|c| c.width == 1920 && c.height == 1080 && c.format == "I420"),
        "expected 1080p I420 in candidates"
    );
    assert!(cands.iter().all(|c| c.num_frames == 5));
}

#[test]
fn test_guess_with_frames_zero_frames_returns_empty() {
    assert!(guess_resolutions_with_frame_count(1024, 0).is_empty());
}

#[test]
fn test_guess_with_frames_indivisible_returns_empty() {
    // file_size = 100 bytes, 7 frames → not evenly divisible.
    assert!(guess_resolutions_with_frame_count(100, 7).is_empty());
}

#[test]
fn test_guess_with_frames_hevc_classc_yuv420_disambiguates() {
    // 832×480 I420 = 599,040 bytes/frame.
    let size = 832u64 * 480 * 3 / 2 * 100;
    let cands = guess_resolutions_with_frame_count(size, 100);
    assert!(
        cands.iter().any(|c| c.width == 832 && c.height == 480 && c.format == "I420"),
        "expected 832×480 I420 in candidates: {:?}",
        cands
    );
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

// --- No-hint multi-candidate enumeration (View → Video Size → Guess again…) ---

#[test]
fn test_guess_no_hint_includes_truth_for_1080p_i420() {
    // 1080p I420 × 5 frames = 15,552,000 bytes.
    let size = 1920u64 * 1080 * 3 / 2 * 5;
    let cands = guess_all_resolutions_no_hint(size);
    assert!(
        cands
            .iter()
            .any(|c| c.width == 1920 && c.height == 1080 && c.format == "I420" && c.num_frames == 5),
        "1080p I420 5f truth missing: {:?}",
        cands
    );
}

#[test]
fn test_guess_no_hint_finds_multiple_candidates() {
    // 1080p I420 × 4 frames also matches 540p I420 × 16 (4× ratio).
    let size = 1920u64 * 1080 * 3 / 2 * 4;
    let cands = guess_all_resolutions_no_hint(size);
    let has_1080 = cands.iter().any(|c| c.width == 1920 && c.height == 1080 && c.format == "I420");
    let has_540 = cands.iter().any(|c| c.width == 960 && c.height == 540 && c.format == "I420");
    // 540p is not show_in_menu in the canonical table, so this only checks the
    // weaker invariant: at least the 1080p truth is present and the list is multi-format.
    assert!(has_1080, "expected 1080p I420 in candidates: {:?}", cands);
    let _ = has_540; // 540p is intentionally excluded from menu set; not asserted.
    let formats: std::collections::HashSet<&str> =
        cands.iter().map(|c| c.format).collect();
    assert!(formats.len() > 1, "expected multiple formats in candidates: {:?}", cands);
}

#[test]
fn test_guess_no_hint_empty_file_is_empty() {
    assert!(guess_all_resolutions_no_hint(0).is_empty());
}

#[test]
fn test_guess_no_hint_odd_size_yields_no_yuv_candidates() {
    // 7 bytes — no YUV format frame size divides this; result should be empty
    // because all HINT_FORMATS produce frame sizes well above 7.
    let cands = guess_all_resolutions_no_hint(7);
    assert!(cands.is_empty(), "expected empty for prime-sized tiny file: {:?}", cands);
}

// --- Raw V4L2 FourCC support in filename hints (new in this change) ---

#[test]
fn test_4cc_yv12_in_filename() {
    let hints = parse_filename_hints("video_YV12_1920x1080.yuv");
    assert_eq!(hints.width, Some(1920));
    assert_eq!(hints.height, Some(1080));
    assert_eq!(hints.format, Some("YV12".to_string()));
}

#[test]
fn test_4cc_nv12_uppercase() {
    let hints = parse_filename_hints("clip_NV12_1080p.raw");
    assert_eq!(hints.format, Some("NV12".to_string()));
}

#[test]
fn test_4cc_yuyv_mixed_with_resolution() {
    let hints = parse_filename_hints("foreman_720p_YUYV.yuv");
    assert_eq!(hints.width, Some(1280));
    assert_eq!(hints.height, Some(720));
    assert_eq!(hints.format, Some("YUYV".to_string()));
}

#[test]
fn test_4cc_p010() {
    let hints = parse_filename_hints("test_P010_1920x1080_10bit.yuv");
    assert_eq!(hints.format, Some("P010".to_string()));
    assert_eq!(hints.bit_depth, Some(10));
}

#[test]
fn test_4cc_bayer_rggb10_packed() {
    let hints = parse_filename_hints("sensor_pRAA_1920x1080_10bit.raw");
    assert_eq!(hints.format, Some("pRAA".to_string()));
}

#[test]
fn test_4cc_bayer_rg12_with_uppercase_x_resolution() {
    let hints = parse_filename_hints("Test_raw_3840X2160_RG12_000.raw");
    assert_eq!(hints.width, Some(3840));
    assert_eq!(hints.height, Some(2160));
    assert_eq!(hints.format, Some("RG12".to_string()));
}

#[test]
fn test_4cc_grey_y10bpack() {
    let hints = parse_filename_hints("thermal_Y10B_640x512.yuv");
    // "Y10B" is both a valid fourcc and has a friendly alias pointing to the long name.
    // Either is acceptable downstream.
    let fmt = hints.format.as_deref().unwrap();
    assert!(
        fmt == "Y10B" || fmt == "Greyscale (10-bit BE packed)",
        "unexpected format for Y10B 4CC: {fmt}"
    );
}

#[test]
fn test_4cc_case_insensitive_mixed() {
    // Even if someone writes nv12 in lowercase, the friendly alias still catches it.
    // Uppercase 4CC should also work.
    let hints1 = parse_filename_hints("file_nv12.yuv");
    let hints2 = parse_filename_hints("file_NV12.yuv");
    assert_eq!(hints1.format, Some("NV12".to_string()));
    assert_eq!(hints2.format, Some("NV12".to_string()));
}

#[test]
fn test_4cc_unknown_fourcc_is_ignored() {
    // Random 4-letter uppercase string that is not a registered format
    let hints = parse_filename_hints("mystery_ABCD_1920x1080.yuv");
    assert!(hints.format.is_none(), "unknown 4CC-like token must not set format");
}

// --- resolve_raw_params: with FourCC vs without (guessing) ---

#[test]
fn test_resolve_with_fourcc_prefers_fourcc_over_default() {
    // Filename carries a 4CC (YUYV). Even if default is I420 and file size matches something else,
    // the 4CC in the name must win for the format.
    let size = 1920u64 * 1080 * 2; // matches 1080p YUYV (2 bytes per pixel)
    let r = resolve_raw_params("capture_YUYV_1920x1080.yuv", Some(size), 640, 480, "I420");
    assert_eq!(r.width, 1920);
    assert_eq!(r.height, 1080);
    assert_eq!(r.format, "YUYV"); // 4CC wins
    assert!(r.info.is_none());
}

#[test]
fn test_resolve_without_fourcc_or_alias_uses_famous_format_first_i420() {
    // No format/4CC/alias in filename, no WxH.
    // File size exactly matches one 1080p I420 frame.
    // Should guess the famous format first (I420 is first in GUESS_FORMATS).
    let size = 1920u64 * 1080 * 3 / 2; // exactly 1 frame of 1080p I420
    let r = resolve_raw_params("unknown_raw_video.yuv", Some(size), 100, 100, "I420");
    assert_eq!(r.width, 1920);
    assert_eq!(r.height, 1080);
    assert_eq!(r.format, "I420"); // famous format first
    assert!(r.info.is_some()); // file-size guess message
}

#[test]
fn test_resolve_without_fourcc_size_guess_from_disk_prefers_i420() {
    // No name hints at all (no WxH, no format token).
    // File size matches 10 frames of VGA I420.
    // resolve must use file-size guessing and pick I420 (famous) first.
    let size = 640u64 * 480 * 3 / 2 * 10;
    let r = resolve_raw_params("raw_capture.yuv", Some(size), 1920, 1080, "NV12");
    assert_eq!(r.width, 640);
    assert_eq!(r.height, 480);
    assert_eq!(r.format, "I420"); // I420 is first in the guess list, even if caller default was NV12
    assert!(r.info.is_some());
}

#[test]
fn test_resolve_with_fourcc_and_file_size_guess() {
    // 4CC present (NV12), no WxH in name, but file size allows resolution guessing.
    // 4CC must be used for format; size comes from disk guess.
    let size = 1920u64 * 1080 * 3 / 2 * 5; // 5 frames of 1080p NV12/I420 (same size)
    let r = resolve_raw_params("security_cam_NV12.yuv", Some(size), 100, 100, "I420");
    assert_eq!(r.width, 1920);
    assert_eq!(r.height, 1080);
    assert_eq!(r.format, "NV12"); // 4CC from filename wins
    assert!(r.info.is_some());
}

fn write_yuv420_probe(prefix: &str, nv12: bool, neutral: bool) -> tempfile::NamedTempFile {
    use std::io::Write;

    const WIDTH: usize = 16;
    const HEIGHT: usize = 16;
    const CHROMA_WIDTH: usize = WIDTH / 2;
    const CHROMA_HEIGHT: usize = HEIGHT / 2;

    let y = vec![80_u8; WIDTH * HEIGHT];
    let (u, v) = if neutral {
        (
            vec![128_u8; CHROMA_WIDTH * CHROMA_HEIGHT],
            vec![128_u8; CHROMA_WIDTH * CHROMA_HEIGHT],
        )
    } else {
        let u = (0..CHROMA_HEIGHT)
            .flat_map(|row| (0..CHROMA_WIDTH).map(move |col| 40 + col as u8 + 2 * row as u8))
            .collect();
        let v = (0..CHROMA_HEIGHT)
            .flat_map(|row| (0..CHROMA_WIDTH).map(move |col| 210 - 2 * col as u8 - row as u8))
            .collect();
        (u, v)
    };

    let mut raw = y;
    if nv12 {
        for (&u_sample, &v_sample) in u.iter().zip(&v) {
            raw.push(u_sample);
            raw.push(v_sample);
        }
    } else {
        raw.extend_from_slice(&u);
        raw.extend_from_slice(&v);
    }

    let mut file = tempfile::Builder::new()
        .prefix(prefix)
        .suffix(".yuv")
        .tempfile()
        .unwrap();
    file.write_all(&raw).unwrap();
    file
}

#[test]
fn test_resolve_content_distinguishes_i420_from_nv12() {
    let i420 = write_yuv420_probe("sample_16x16_", false, false);
    let nv12 = write_yuv420_probe("sample_16x16_", true, false);

    let i420_result = resolve_raw_params(
        i420.path().to_str().unwrap(),
        Some(i420.as_file().metadata().unwrap().len()),
        640,
        480,
        "NV12",
    );
    let nv12_result = resolve_raw_params(
        nv12.path().to_str().unwrap(),
        Some(nv12.as_file().metadata().unwrap().len()),
        640,
        480,
        "I420",
    );

    assert_eq!(i420_result.format, "I420");
    assert_eq!(nv12_result.format, "NV12");
}

#[test]
fn test_explicit_dimensions_drive_content_format_guess() {
    let nv12 = write_yuv420_probe("sample_", true, false);

    let result = resolve_raw_params_for_dimensions(nv12.path().to_str().unwrap(), 16, 16);

    assert_eq!((result.width, result.height), (16, 16));
    assert_eq!(result.format, "NV12");
}

#[test]
fn test_resolve_ambiguous_content_falls_back_to_i420() {
    let neutral = write_yuv420_probe("sample_16x16_", false, true);
    let result = resolve_raw_params(
        neutral.path().to_str().unwrap(),
        Some(neutral.as_file().metadata().unwrap().len()),
        640,
        480,
        "NV12",
    );

    assert_eq!(result.format, "I420");
}

#[test]
fn test_resolve_fourcc_wins_over_content_guess() {
    let i420 = write_yuv420_probe("sample_NV12_16x16_", false, false);
    let result = resolve_raw_params(
        i420.path().to_str().unwrap(),
        Some(i420.as_file().metadata().unwrap().len()),
        640,
        480,
        "I420",
    );

    assert_eq!(result.format, "NV12");
}

#[test]
fn test_dialogs_resolve_short_i420_name_to_the_i420_entry() {
    let parameters = ParametersDialog::new(16, 16, "I420");
    let open_file = OpenFileDialog::new(16, 16, "I420", None);

    assert_eq!(
        parameters.format_names[parameters.format_idx],
        "I420 (4:2:0)"
    );
    assert_eq!(open_file.format_names[open_file.format_idx], "I420 (4:2:0)");
}

#[test]
fn test_unrenderable_fourcc_is_not_auto_applied_or_offered_for_viewing() {
    let hints = parse_filename_hints("sample_RGBP_16x16.yuv");
    let rgb565 = write_yuv420_probe("sample_RGBP_16x16_", true, false);
    let mixed = write_yuv420_probe("sample_RGBP_NV12_16x16_", false, false);
    let resolved = resolve_raw_params(
        rgb565.path().to_str().unwrap(),
        Some(rgb565.as_file().metadata().unwrap().len()),
        640,
        480,
        "NV12",
    );
    let mixed_resolved = resolve_raw_params(
        mixed.path().to_str().unwrap(),
        Some(mixed.as_file().metadata().unwrap().len()),
        640,
        480,
        "I420",
    );
    let parameters = ParametersDialog::new(16, 16, "I420");

    assert_eq!(hints.format, None);
    assert_eq!(resolved.format, "I420");
    assert!(resolved.info.unwrap().contains("RGBP"));
    assert_eq!(mixed_resolved.format, "NV12");
    assert_eq!(mixed_resolved.info, None);
    assert!(!parameters
        .format_names
        .iter()
        .any(|name| name.starts_with("RGB565")));
    assert_eq!(
        parameters.format_names[parameters.format_idx],
        "I420 (4:2:0)"
    );
}
