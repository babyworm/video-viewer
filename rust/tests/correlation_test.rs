//! M2 verification tests for the correlation engine (UX spec 04 §12 V19/V20
//! plus a real-fixture end-to-end smoke).
//!
//! V19: synthetic flat/noise frame + synthetic catb with matching bits →
//! the full pipeline (BlockStats → BitstreamGrid → AlignedPair) must report
//! r > 0.8. V20: CSV row count, valid=false flags, and r re-computed from
//! the CSV must match the readout.

use std::path::PathBuf;

use video_viewer::analysis::bitstream_stats::{rasterize_blocks, BitstreamFile};
use video_viewer::analysis::correlation::{
    aggregate_bitstream_to_g, align, class_table, classes_at_g, compute_analysis_grids, csv_dump,
    pearson_r, spearman_rho, x_grid, AlignedPair, XMetric, YMetric, CSV_HEADER,
};

mod common;
use common::{block_record, build_catb, encode_strings, frame_record, write_temp, BlockSpec};

// ===========================================================================
// Synthetic fixture builders (fx_flat style)
// ===========================================================================

/// Grey RGB frame: left half flat luma 128, right half deterministic noise.
fn flat_noise_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    let mut seed = 0x0bad_cafe_u32;
    for _y in 0..h {
        for x in 0..w {
            let v = if x < w / 2 {
                128u8
            } else {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 24) as u8
            };
            rgb.extend_from_slice(&[v, v, v]);
        }
    }
    rgb
}

/// One-frame catb whose 8×8 blocks tile ceil(w/8) × ceil(h/8): blocks in the
/// left half get `low_bits`, right half `high_bits` (fx_flat's injected
/// correlation). QP is uniform 32, mode Intra.
fn flat_noise_catb(w: u32, h: u32, low_bits: i64, high_bits: i64) -> Vec<u8> {
    let strings = encode_strings(&["", "8x8", "Intra", "IDR"]);
    let meta = br#"{"schema_version":1,"decoder":{"codec":"hevc","name":"test","contract":"0.8.3","capture_level":""},"parameter_sets":[],"frames_meta":[]}"#;
    let cols = w.div_ceil(8);
    let rows = h.div_ceil(8);
    let mut blocks = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let x = (c * 8) as i32;
            blocks.extend(block_record(&BlockSpec {
                x,
                y: (r * 8) as i32,
                w: 8,
                h: 8,
                qp: 32,
                partition_id: 1,
                prediction_mode_id: 2,
                bits: if (x as u32) < w / 2 { low_bits } else { high_bits },
                ..Default::default()
            }));
        }
    }
    let n = (cols * rows) as i32;
    let frame = frame_record(0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, n, 0, 0);
    build_catb(1, &strings, meta, &frame, &blocks, &[], &[])
}

/// Full M2 pipeline for one frame: analysis grids + catb blocks →
/// AlignedPair at `g`.
#[allow(clippy::too_many_arguments)]
fn pipeline_pair(
    rgb: &[u8],
    prev: Option<&[u8]>,
    w: u32,
    h: u32,
    file: &BitstreamFile,
    x: XMetric,
    y: YMetric,
    g: u32,
) -> AlignedPair {
    let grids = compute_analysis_grids(rgb, prev, w, h);
    let xg = x_grid(&grids, x, g).expect("x grid");
    let blocks = file.blocks_display(0).expect("blocks");
    let (sw, sh) = (file.width, file.height);
    let l1 = rasterize_blocks(&blocks, sw, sh, 8);
    let bs = aggregate_bitstream_to_g(&l1, sw, sh, g);
    align(&xg, &bs, y)
}

// ===========================================================================
// V19: correlation readout on the flat/noise fixture
// ===========================================================================

#[test]
fn test_v19_variance_bpp_correlation_exceeds_0_8() {
    let (w, h) = (64u32, 64u32);
    let rgb = flat_noise_rgb(w, h);
    let catb = write_temp(&flat_noise_catb(w, h, 16, 512));
    let file = BitstreamFile::open(catb.path()).expect("open catb");
    assert_eq!((file.width, file.height), (64, 64), "block-extent resolution");

    let pair = pipeline_pair(&rgb, None, w, h, &file, XMetric::Variance, YMetric::Bpp, 16);
    assert_eq!((pair.cols, pair.rows), (4, 4));
    // 64 divides evenly into 16px cells with full coverage → all valid.
    assert_eq!(pair.n_valid(), 16, "N must match the grid");
    assert!((pair.valid_fraction() - 1.0).abs() < 1e-6);

    let r = pearson_r(&pair.a, &pair.b, &pair.valid).expect("r");
    assert!(r > 0.8, "V19: injected variance↔bpp correlation, got r={r}");
    let rho = spearman_rho(&pair.a, &pair.b, &pair.valid).expect("rho");
    assert!(rho > 0.8, "rho={rho}");
}

#[test]
fn test_v19_partial_edge_blocks_are_excluded() {
    // 60×44: the right column (x=56..60) and bottom row (y=40..44) of 8px
    // cells are partial → masked on both sides.
    let (w, h) = (60u32, 44u32);
    let rgb = flat_noise_rgb(w, h);
    let catb = write_temp(&flat_noise_catb(w, h, 16, 512));
    let file = BitstreamFile::open(catb.path()).expect("open catb");

    let pair = pipeline_pair(&rgb, None, w, h, &file, XMetric::Variance, YMetric::Bpp, 8);
    assert_eq!((pair.cols, pair.rows), (8, 6));
    // Valid cells: 7 full columns × 5 full rows.
    assert_eq!(pair.n_valid(), 35, "right/bottom partial cells excluded");
    for r in 0..6usize {
        assert!(!pair.valid[r * 8 + 7], "partial right column row {r}");
    }
    for c in 0..8usize {
        assert!(!pair.valid[5 * 8 + c], "partial bottom row col {c}");
    }
}

#[test]
fn test_v19_motion_score_correlates_with_bits() {
    // Frame pair: static left half, moving (re-seeded noise) right half.
    let (w, h) = (64u32, 64u32);
    let prev = flat_noise_rgb(w, h);
    let mut cur = prev.clone();
    // Perturb the right half only.
    let mut seed = 0x5555_aaaa_u32;
    for y in 0..h as usize {
        for x in (w as usize / 2)..w as usize {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = (seed >> 24) as u8;
            let p = (y * w as usize + x) * 3;
            cur[p] = v;
            cur[p + 1] = v;
            cur[p + 2] = v;
        }
    }
    let catb = write_temp(&flat_noise_catb(w, h, 16, 512));
    let file = BitstreamFile::open(catb.path()).expect("open catb");
    let pair = pipeline_pair(
        &cur,
        Some(&prev),
        w,
        h,
        &file,
        XMetric::MotionScore,
        YMetric::Bpp,
        16,
    );
    let r = pearson_r(&pair.a, &pair.b, &pair.valid).expect("r");
    assert!(r > 0.8, "motion↔bits on split-motion fixture, r={r}");

    // Class table: every valid cell lands in a class, and the moving half
    // must not be classified None.
    let grids = compute_analysis_grids(&cur, Some(&prev), w, h);
    let (classes, grid) = classes_at_g(&grids, 16).expect("classes");
    let blocks = file.blocks_display(0).expect("blocks");
    let l1 = rasterize_blocks(&blocks, 64, 64, 8);
    let bs = aggregate_bitstream_to_g(&l1, 64, 64, 16);
    let table = class_table(&classes, &grid.valid, grid.cols, grid.rows, &bs);
    let total_cells: usize = table.iter().map(|r| r.cells).sum();
    assert_eq!(total_cells, 16, "every valid cell is classified");
    // None-class cells (static half) burn fewer bits than the top class.
    let none_row = table[0];
    let hot = table.iter().rev().find(|r| r.cells > 0).unwrap();
    assert!(
        none_row.mean_bpp < hot.mean_bpp,
        "static cells {} bpp should undercut moving cells {} bpp",
        none_row.mean_bpp,
        hot.mean_bpp
    );
}

// ===========================================================================
// V20: CSV export
// ===========================================================================

#[test]
fn test_v20_csv_rows_flags_and_recomputed_r() {
    let (w, h) = (60u32, 44u32);
    let rgb = flat_noise_rgb(w, h);
    let catb = write_temp(&flat_noise_catb(w, h, 16, 512));
    let file = BitstreamFile::open(catb.path()).expect("open catb");
    let pair = pipeline_pair(&rgb, None, w, h, &file, XMetric::Variance, YMetric::Bpp, 8);

    let csv = csv_dump(&[(0, &pair)]);
    let mut lines = csv.lines();
    assert_eq!(lines.next(), Some(CSV_HEADER));
    let rows: Vec<&str> = lines.collect();
    assert_eq!(
        rows.len(),
        (pair.cols * pair.rows) as usize,
        "V20: row count = cols × rows"
    );

    // Re-parse and recompute r from the CSV alone (pandas-style path).
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut valid = Vec::new();
    let mut invalid_rows = 0;
    for row in rows {
        let f: Vec<&str> = row.split(',').collect();
        assert_eq!(f.len(), 6);
        assert_eq!(f[0], "0");
        a.push(f[3].parse::<f32>().expect("a"));
        b.push(f[4].parse::<f32>().expect("b"));
        let v = match f[5] {
            "1" => true,
            "0" => false,
            other => panic!("valid flag must be 1/0, got {other}"),
        };
        if !v {
            invalid_rows += 1;
        }
        valid.push(v);
    }
    assert!(invalid_rows > 0, "V20: valid=false rows must exist");

    let r_csv = pearson_r(&a, &b, &valid).expect("r from CSV");
    let r_readout = pearson_r(&pair.a, &pair.b, &pair.valid).expect("r readout");
    assert!(
        (r_csv - r_readout).abs() < 1e-9,
        "CSV r {r_csv} vs readout {r_readout}"
    );
}

// ===========================================================================
// End-to-end smoke on the real hevc_bslice fixture
// ===========================================================================

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root")
        .join("test_data/bitstream/hevc_bslice")
}

#[test]
fn test_real_fixture_end_to_end_smoke() {
    let dir = fixture_dir();
    let file = BitstreamFile::open(dir.join("hevc_bslice.catb")).expect("open fixture catb");
    let mut reader = video_viewer::core::reader::VideoReader::open(
        dir.join("hevc_bslice_64x64_i420.yuv").to_str().unwrap(),
        64,
        64,
        "I420",
        "BT.709",
    )
    .expect("open fixture yuv");
    assert!(reader.total_frames() > 0);

    let raw = reader.seek_frame(0).expect("seek");
    let rgb = reader.convert_to_rgb(&raw).expect("rgb");
    let grids = compute_analysis_grids(&rgb, None, 64, 64);
    for (x, y) in [
        (XMetric::Variance, YMetric::Qp),
        (XMetric::Mean, YMetric::Bpp),
        (XMetric::Orientation, YMetric::IntraRatio),
    ] {
        let xg = x_grid(&grids, x, 16).expect("x grid");
        let blocks = file.blocks_display(0).expect("blocks");
        let l1 = rasterize_blocks(&blocks, file.width, file.height, 8);
        let bs = aggregate_bitstream_to_g(&l1, file.width, file.height, 16);
        let pair = align(&xg, &bs, y);
        assert!(pair.n_valid() > 0, "{x:?}/{y:?}: N must be > 0");
        assert_eq!(pair.a.len(), (pair.cols * pair.rows) as usize);
    }
}
