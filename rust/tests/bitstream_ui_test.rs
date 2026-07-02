//! M1 Bitstream Analysis window — pure-logic integration tests:
//! settings backward compatibility, preset application, offset mapping,
//! rasterization via the public API, and filmstrip classification.

use video_viewer::analysis::bitstream_stats::{
    hit_test_min_area, lod_cell_size, rasterize_blocks, use_lod, viewer_to_catb_display,
    FrameTypeClass, ModeClass,
};
use video_viewer::ui::bitstream_overlay::MvSource;
use video_viewer::ui::bitstream_window::{
    filmstrip_color, matching_preset, next_preset, FillMode, Preset, ViewConfig, PRESETS,
};
use video_viewer::ui::settings::{BitstreamViewSettings, Settings};

// ---------------------------------------------------------------------------
// Settings backward compatibility (old settings.toml without [bitstream])
// ---------------------------------------------------------------------------

const OLD_SETTINGS_TOML: &str = r#"
recent_files = ["/tmp/a.yuv"]

[cache]
max_memory_mb = 512

[display]
zoom_min = 0.1
zoom_max = 50.0
dark_theme = true

[defaults]
fps = 30
color_matrix = "BT.601"
width = 1920
height = 1080
format = "I420"
"#;

#[test]
fn test_settings_old_file_without_bitstream_section_parses() {
    let s: Settings = toml::from_str(OLD_SETTINGS_TOML).expect("old settings must still parse");
    // The new section falls back to its defaults.
    assert_eq!(s.bitstream, BitstreamViewSettings::default());
    assert_eq!(s.bitstream.fill, "QP");
    assert!((s.bitstream.opacity - 0.6).abs() < 1e-6);
    assert!(s.bitstream.layer_grid);
    // The default tuple must match the QP Map preset exactly, so a fresh
    // install shows "QP Map" (not "Custom") in the preset combo.
    assert_eq!(
        matching_preset(&ViewConfig::from_settings(&s.bitstream)),
        Some(Preset::QpMap)
    );
    // Pre-existing fields are untouched.
    assert_eq!(s.cache.max_memory_mb, 512);
    assert_eq!(s.recent_files, vec!["/tmp/a.yuv".to_string()]);
}

#[test]
fn test_settings_bitstream_section_roundtrip() {
    let mut s = Settings::default();
    s.bitstream.fill = "MV-heat".to_string();
    s.bitstream.opacity = 0.35;
    s.bitstream.layer_label = true;
    let toml_str = toml::to_string_pretty(&s).unwrap();
    let back: Settings = toml::from_str(&toml_str).unwrap();
    assert_eq!(back.bitstream, s.bitstream);
}

#[test]
fn test_settings_view_config_conversion() {
    let s = BitstreamViewSettings {
        fill: "bpp".to_string(),
        layer_mv: false,
        layer_part: false,
        layer_intra: false,
        layer_label: false,
        layer_grid: true,
        layer_sel: true,
        opacity: 0.6,
        show_loupe: false,
        inspector_collapsed: true,
        mv_source: "MV".to_string(),
    };
    let cfg = ViewConfig::from_settings(&s);
    assert_eq!(cfg.fill, FillMode::Bpp);
    // This particular tuple is exactly the Rate preset.
    assert_eq!(matching_preset(&cfg), Some(Preset::Rate));
    let back = cfg.to_settings(false, true, MvSource::Mv);
    assert_eq!(back, s);
}

#[test]
fn test_settings_pre_013_bitstream_section_parses() {
    // A 0.12.x settings file: [bitstream] exists but has no layer_intra /
    // mv_source — serde defaults must fill them (M4 backward compat).
    let toml_str = format!(
        r#"{OLD_SETTINGS_TOML}
[bitstream]
fill = "QP"
layer_mv = false
layer_part = false
layer_label = true
layer_grid = true
layer_sel = true
opacity = 0.6
show_loupe = false
inspector_collapsed = false
"#
    );
    let s: Settings = toml::from_str(&toml_str).expect("0.12 settings must parse");
    assert!(!s.bitstream.layer_intra);
    assert_eq!(s.bitstream.mv_source, "MV");
    assert_eq!(MvSource::from_label(&s.bitstream.mv_source), MvSource::Mv);
}

// ---------------------------------------------------------------------------
// Preset application (§3 table) through the public API
// ---------------------------------------------------------------------------

#[test]
fn test_preset_apply_table() {
    let cases: [(Preset, FillMode, bool, bool, bool, bool, bool); 6] = [
        // (preset, fill, grid, label, mv, part, intra) — M4 layer columns.
        (Preset::Rate, FillMode::Bpp, true, false, false, false, false),
        (Preset::QpMap, FillMode::Qp, true, true, false, false, false),
        (Preset::Motion, FillMode::MvHeat, true, false, true, false, false),
        (Preset::Mode, FillMode::Mode, true, false, false, true, true),
        (
            Preset::Opportunity,
            FillMode::Opportunity,
            true,
            false,
            false,
            false,
            false,
        ),
        (Preset::Clean, FillMode::None, false, false, false, false, false),
    ];
    for (preset, fill, grid, label, mv, part, intra) in cases {
        let c = preset.config();
        assert_eq!(c.fill, fill, "{preset:?} fill");
        assert_eq!(c.grid, grid, "{preset:?} grid");
        assert_eq!(c.label, label, "{preset:?} label");
        assert_eq!(c.mv, mv, "{preset:?} mv");
        assert_eq!(c.part, part, "{preset:?} part");
        assert_eq!(c.intra, intra, "{preset:?} intra");
        assert!((c.opacity - 0.6).abs() < 1e-6, "{preset:?} opacity");
        assert_eq!(matching_preset(&c), Some(preset));
    }
}

#[test]
fn test_preset_custom_detection_and_cycle() {
    // V10: manual layer edit → Custom; re-selecting the preset restores it.
    let mut cfg = Preset::Rate.config();
    cfg.label = true;
    assert_eq!(matching_preset(&cfg), None);
    assert_eq!(next_preset(&cfg), Preset::Rate); // Custom → Rate
    // Tab cycles through all presets and wraps.
    let mut p = Preset::Rate;
    for _ in 0..PRESETS.len() {
        p = next_preset(&p.config());
    }
    assert_eq!(p, Preset::Rate);
}

// ---------------------------------------------------------------------------
// Offset-applied frame mapping (V5)
// ---------------------------------------------------------------------------

#[test]
fn test_offset_frame_mapping() {
    assert_eq!(viewer_to_catb_display(0, 0), Some(0));
    assert_eq!(viewer_to_catb_display(0, 2), Some(2));
    assert_eq!(viewer_to_catb_display(7, -2), Some(5));
    assert_eq!(viewer_to_catb_display(1, -2), None);
}

// ---------------------------------------------------------------------------
// Rasterization / LOD / hit test via the public API
// ---------------------------------------------------------------------------

fn make_block(
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    qp: i32,
    bits: i64,
    mode: &str,
) -> video_viewer::core::catb::BsBlock {
    video_viewer::core::catb::BsBlock {
        x,
        y,
        w,
        h,
        ctu_address: 0,
        qp,
        partition: String::new(),
        prediction_mode: mode.to_string(),
        bits,
        bit_offset: 0,
        exactness_flags: 0,
        syntax_off: 0,
        syntax_n: 0,
        cabac_off: 0,
        cabac_n: 0,
        tx_off: 0,
        tx_n: 0,
        ref_off: 0,
        ref_n: 0,
        mv_flags: 0,
        mvp: None,
        mvd: None,
        mv: None,
        reference: None,
        reference_list: None,
        reference_label: None,
        reference_list_index: None,
        reference_poc: None,
        reference_frame: None,
        reference_long_term: None,
    }
}

#[test]
fn test_rasterize_bpp_and_partial_coverage() {
    // 16×16 frame, 8px cells. One 16×8 block (256 bits → 2 bpp) on top,
    // one 8×4 block on the bottom-left (partial cell coverage 0.5).
    let blocks = vec![
        make_block(0, 0, 16, 8, 30, 256, "Inter"),
        make_block(0, 8, 8, 4, 40, 64, "Intra"),
    ];
    let g = rasterize_blocks(&blocks, 16, 16, 8);
    assert_eq!((g.cols, g.rows), (2, 2));
    assert!((g.bpp[0] - 2.0).abs() < 1e-5);
    assert!((g.bpp[1] - 2.0).abs() < 1e-5);
    assert!((g.coverage[2] - 0.5).abs() < 1e-5); // bottom-left partial
    assert!((g.bpp[2] - 2.0).abs() < 1e-5); // 64 bits / 32 covered px
    assert_eq!(g.coverage[3], 0.0); // bottom-right untouched
    assert_eq!(g.mode[0], ModeClass::Inter);
    assert_eq!(g.mode[2], ModeClass::Intra);
}

#[test]
fn test_lod_threshold_and_codec_cell() {
    assert!(use_lod(0.5));
    assert!(!use_lod(2.0));
    assert_eq!(lod_cell_size("hevc"), 64);
    assert_eq!(lod_cell_size("avc"), 32);
}

#[test]
fn test_hit_test_min_area_nested() {
    let blocks = vec![
        make_block(0, 0, 64, 64, 30, 0, "Inter"),
        make_block(16, 16, 8, 8, 40, 0, "Intra"),
        make_block(16, 16, 4, 4, 45, 0, "Intra"),
    ];
    // Deepest (smallest) block wins.
    assert_eq!(hit_test_min_area(&blocks, 17, 17), Some(2));
    assert_eq!(hit_test_min_area(&blocks, 22, 22), Some(1));
    assert_eq!(hit_test_min_area(&blocks, 0, 0), Some(0));
    assert_eq!(hit_test_min_area(&blocks, 100, 100), None);
}

// ---------------------------------------------------------------------------
// Filmstrip classification (§4)
// ---------------------------------------------------------------------------

#[test]
fn test_filmstrip_frame_type_to_color_class() {
    // frame_type → class → colour dominance.
    let i = filmstrip_color(Some(FrameTypeClass::from_label("IDR")));
    let p = filmstrip_color(Some(FrameTypeClass::from_label("P")));
    let b = filmstrip_color(Some(FrameTypeClass::from_label("B")));
    assert!(i.r() > i.g() && i.r() > i.b(), "I frames are red");
    assert!(p.b() > p.r() && p.b() > p.g(), "P frames are blue");
    assert!(b.g() > b.r() && b.g() > b.b(), "B frames are green");
    // Frames outside the telemetry (offset gap) are dark grey.
    let gap = filmstrip_color(None);
    assert!(gap.r() < 80 && gap.g() < 80 && gap.b() < 80);
}
