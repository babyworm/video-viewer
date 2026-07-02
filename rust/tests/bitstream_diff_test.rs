//! M-D verification tests: dual-`.catb` Δ comparison.
//!
//! Synthetic pairs check the sign convention (Δ = A − B), antisymmetry,
//! mode agreement, B's independent display-map/offset, the ΔbppG
//! aggregation, and the load-time resolution gate. The real-fixture test
//! exploits that `hevc_multi_idr.h265` is `hevc_inter.h265` concatenated
//! twice — its first CVS decodes to the identical three frames, so the Δ
//! against `hevc_inter` must be exactly zero.

use std::path::PathBuf;

use video_viewer::analysis::bitstream_diff::{diff_grids, validate_b_resolution, DiffMetric};
use video_viewer::analysis::bitstream_stats::{
    rasterize_blocks, viewer_to_catb_display, BitstreamFile, ModeClass,
};
use video_viewer::analysis::correlation::{
    aggregate_bitstream_to_g, align, subtract_bitstream_g, GGrid, YMetric,
};

mod common;
use common::{block_record, build_catb, encode_strings, frame_record, write_temp, BlockSpec};

const META: &[u8] = br#"{"schema_version":1,"decoder":{"codec":"hevc","name":"test","contract":"0.8.3","capture_level":""},"parameter_sets":[],"frames_meta":[]}"#;

/// One-frame 16×16 catb "A": four 8×8 blocks, QP/bits/mode per block.
/// String table: 1 = partition, 2 = "Intra", 3 = "Inter", 4 = "IDR".
fn catb_a() -> Vec<u8> {
    let strings = encode_strings(&["", "8x8", "Intra", "Inter", "IDR"]);
    let specs = [
        (0, 0, 30, 64i64, 2),  // Intra
        (8, 0, 32, 128, 3),    // Inter
        (0, 8, 34, 192, 2),    // Intra
        (8, 8, 36, 256, 3),    // Inter
    ];
    let mut blocks = Vec::new();
    for (x, y, qp, bits, mode) in specs {
        blocks.extend(block_record(&BlockSpec {
            x,
            y,
            w: 8,
            h: 8,
            qp,
            partition_id: 1,
            prediction_mode_id: mode,
            bits,
            ..Default::default()
        }));
    }
    let frame = frame_record(0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0);
    build_catb(1, &strings, META, &frame, &blocks, &[], &[])
}

/// One-frame 16×16 catb "B": a single 16×16 CU (different partitioning —
/// the reason the Δ must live on the fixed 8px grid), QP 28, 320 bits,
/// Inter everywhere.
fn catb_b() -> Vec<u8> {
    let strings = encode_strings(&["", "16x16", "Intra", "Inter", "IDR"]);
    let blocks = block_record(&BlockSpec {
        x: 0,
        y: 0,
        w: 16,
        h: 16,
        qp: 28,
        partition_id: 1,
        prediction_mode_id: 3,
        bits: 320,
        ..Default::default()
    });
    let frame = frame_record(0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0);
    build_catb(1, &strings, META, &frame, &blocks, &[], &[])
}

/// Two-frame 16×16 catb: frame 0 uniform QP `qp0`, frame 1 uniform `qp1`
/// (single 16×16 block each) — offset-independence fixture.
fn catb_two_frames(qp0: i32, qp1: i32) -> Vec<u8> {
    let strings = encode_strings(&["", "16x16", "Inter", "IDR", "P"]);
    let mut blocks = Vec::new();
    for qp in [qp0, qp1] {
        blocks.extend(block_record(&BlockSpec {
            x: 0,
            y: 0,
            w: 16,
            h: 16,
            qp,
            partition_id: 1,
            prediction_mode_id: 2,
            bits: 100,
            ..Default::default()
        }));
    }
    let mut frames = frame_record(0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0);
    frames.extend(frame_record(1, 1, 4, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0));
    build_catb(2, &strings, META, &frames, &blocks, &[], &[])
}

fn l1_of(file: &BitstreamFile, display_idx: usize) -> video_viewer::analysis::bitstream_stats::BitstreamGrid {
    let blocks = file.blocks_display(display_idx).expect("blocks");
    rasterize_blocks(&blocks, file.width, file.height, 8)
}

// ===========================================================================
// Synthetic Δ grid: signs, antisymmetry, mode agreement
// ===========================================================================

#[test]
fn test_diff_grid_signed_values_and_antisymmetry() {
    let a = BitstreamFile::open(write_temp(&catb_a()).path()).expect("A");
    let b = BitstreamFile::open(write_temp(&catb_b()).path()).expect("B");
    assert_eq!((a.width, a.height), (16, 16));
    assert_eq!((b.width, b.height), (16, 16));
    let (ga, gb) = (l1_of(&a, 0), l1_of(&b, 0));

    let d = diff_grids(&ga, &gb).expect("same geometry");
    assert_eq!((d.cols, d.rows), (2, 2));
    assert!(d.valid.iter().all(|v| *v), "full coverage on both sides");
    // Δqp = A − B: [30,32,34,36] − 28.
    let want_qp = [2.0f32, 4.0, 6.0, 8.0];
    for (i, w) in want_qp.iter().enumerate() {
        assert!((d.d_qp[i] - w).abs() < 1e-4, "cell {i}: {} != {w}", d.d_qp[i]);
    }
    // Δbpp per 8px cell: A [1,2,3,4] bpp − B 320/256 = 1.25 bpp everywhere.
    let want_bpp = [-0.25f32, 0.75, 1.75, 2.75];
    for (i, w) in want_bpp.iter().enumerate() {
        assert!((d.d_bpp[i] - w).abs() < 1e-4, "cell {i}: {} != {w}", d.d_bpp[i]);
    }
    // Symmetric legend scale = max |Δ|.
    assert!((d.max_abs(DiffMetric::Qp) - 8.0).abs() < 1e-4);
    assert!((d.max_abs(DiffMetric::Bpp) - 2.75).abs() < 1e-4);

    // Antisymmetry: diff(B, A) is the exact negation.
    let d_rev = diff_grids(&gb, &ga).expect("same geometry");
    for i in 0..4 {
        assert!((d.d_qp[i] + d_rev.d_qp[i]).abs() < 1e-5);
        assert!((d.d_bpp[i] + d_rev.d_bpp[i]).abs() < 1e-5);
    }
}

#[test]
fn test_diff_grid_mode_agreement() {
    let a = BitstreamFile::open(write_temp(&catb_a()).path()).expect("A");
    let b = BitstreamFile::open(write_temp(&catb_b()).path()).expect("B");
    let d = diff_grids(&l1_of(&a, 0), &l1_of(&b, 0)).expect("diff");
    // A: Intra/Inter/Intra/Inter vs B: Inter everywhere → cells 1 and 3
    // agree, 0 and 2 differ (and keep A's mode for the colour).
    assert!(!d.mode_same(0) && d.mode_same(1) && !d.mode_same(2) && d.mode_same(3));
    assert_eq!(d.mode_a[0], ModeClass::Intra);
    assert_eq!(d.mode_b[0], ModeClass::Inter);
}

// ===========================================================================
// B display-map / offset independence
// ===========================================================================

#[test]
fn test_diff_uses_b_own_display_map_and_offset() {
    let a = BitstreamFile::open(write_temp(&catb_a()).path()).expect("A");
    let b = BitstreamFile::open(write_temp(&catb_two_frames(28, 40)).path()).expect("B");
    assert_eq!(b.frame_count(), 2);
    let ga = l1_of(&a, 0);

    // viewer#0 with B offset 0 → B display 0 (QP 28): Δqp cell 0 = +2.
    let db0 = viewer_to_catb_display(0, 0).expect("maps");
    let d0 = diff_grids(&ga, &l1_of(&b, db0)).expect("diff");
    assert!((d0.d_qp[0] - 2.0).abs() < 1e-4);

    // Same viewer frame with B offset +1 → B display 1 (QP 40): Δqp = −10.
    let db1 = viewer_to_catb_display(0, 1).expect("maps");
    assert_eq!(db1, 1);
    let d1 = diff_grids(&ga, &l1_of(&b, db1)).expect("diff");
    assert!((d1.d_qp[0] + 10.0).abs() < 1e-4);

    // Negative offset before the telemetry start → no B frame at all.
    assert_eq!(viewer_to_catb_display(0, -1), None);
}

// ===========================================================================
// Δbpp / ΔQP G-aggregation (Correlation Y metrics)
// ===========================================================================

#[test]
fn test_delta_bpp_g_aggregation_and_align() {
    let a = BitstreamFile::open(write_temp(&catb_a()).path()).expect("A");
    let b = BitstreamFile::open(write_temp(&catb_b()).path()).expect("B");
    let mut bs_a = aggregate_bitstream_to_g(&l1_of(&a, 0), 16, 16, 16);
    let bs_b = aggregate_bitstream_to_g(&l1_of(&b, 0), 16, 16, 16);
    subtract_bitstream_g(&mut bs_a, &bs_b);
    assert_eq!((bs_a.cols, bs_a.rows), (1, 1));
    assert!(bs_a.valid[0]);
    // A: 640 bits / 256 px = 2.5 bpp; B: 320/256 = 1.25 → Δ 1.25.
    assert!((bs_a.bpp[0] - 1.25).abs() < 1e-4, "{}", bs_a.bpp[0]);
    // A mean QP (equal areas) = 33; B = 28 → Δ 5.
    assert!((bs_a.qp[0] - 5.0).abs() < 1e-4, "{}", bs_a.qp[0]);

    // align() reads the subtracted slots through the delta metrics.
    let x = GGrid {
        g: 16,
        cols: 1,
        rows: 1,
        values: vec![7.0],
        valid: vec![true],
    };
    let p = align(&x, &bs_a, YMetric::DeltaBpp);
    assert_eq!(p.b, vec![1.25]);
    assert!(p.valid[0]);
    let p = align(&x, &bs_a, YMetric::DeltaQp);
    assert_eq!(p.b, vec![5.0]);
    assert!(YMetric::DeltaBpp.is_delta() && YMetric::DeltaQp.is_delta());
    assert!(!YMetric::Bpp.is_delta());
}

#[test]
fn test_delta_with_missing_b_frame_invalidates_all_cells() {
    // An empty B aggregate (viewer frame B does not cover) must AND every
    // cell invalid — never a fake A−0 difference.
    let a = BitstreamFile::open(write_temp(&catb_a()).path()).expect("A");
    let mut bs_a = aggregate_bitstream_to_g(&l1_of(&a, 0), 16, 16, 16);
    let empty = aggregate_bitstream_to_g(&rasterize_blocks(&[], 16, 16, 8), 16, 16, 16);
    subtract_bitstream_g(&mut bs_a, &empty);
    assert!(bs_a.valid.iter().all(|v| !v));
    assert!(bs_a.bpp.iter().all(|v| *v == 0.0), "no stale A value leaks");
}

// ===========================================================================
// Resolution gate (load-time rejection)
// ===========================================================================

#[test]
fn test_b_resolution_gate_rejects_mismatch() {
    let a = BitstreamFile::open(write_temp(&catb_a()).path()).expect("A");
    // 32-px-wide "B": two 16×16 blocks side by side (block-extent 32×16).
    let strings = encode_strings(&["", "16x16", "Inter", "IDR"]);
    let mut blocks = Vec::new();
    for x in [0, 16] {
        blocks.extend(block_record(&BlockSpec {
            x,
            y: 0,
            w: 16,
            h: 16,
            qp: 30,
            partition_id: 1,
            prediction_mode_id: 2,
            bits: 64,
            ..Default::default()
        }));
    }
    let frame = frame_record(0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0);
    let wide = build_catb(1, &strings, META, &frame, &blocks, &[], &[]);
    let b = BitstreamFile::open(write_temp(&wide).path()).expect("B");
    assert_eq!((b.width, b.height), (32, 16));

    let err = validate_b_resolution((a.width, a.height), (b.width, b.height))
        .expect_err("mismatch must be rejected");
    assert!(err.contains("resolution mismatch"), "{err}");
    assert!(validate_b_resolution((a.width, a.height), (16, 16)).is_ok());
}

// ===========================================================================
// Real fixtures: hevc_inter vs hevc_multi_idr (identical leading CVS)
// ===========================================================================

fn fixture(dir: &str, name: &str) -> Option<BitstreamFile> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root")
        .join("test_data/bitstream")
        .join(dir)
        .join(format!("{name}.catb"));
    path.exists().then(|| BitstreamFile::open(&path).expect("open fixture"))
}

#[test]
fn test_real_fixture_identical_streams_diff_is_zero() {
    // hevc_multi_idr.h265 is hevc_inter.h265 concatenated twice: its first
    // 3 display frames decode identically, so every Δ must be exactly 0
    // and every cell's mode must agree.
    let (Some(a), Some(b)) = (
        fixture("hevc_inter", "hevc_inter"),
        fixture("hevc_multi_idr", "hevc_multi_idr"),
    ) else {
        eprintln!("fixtures missing — skipping");
        return;
    };
    assert_eq!((a.width, a.height), (b.width, b.height), "same source");
    assert!(b.frame_count() >= a.frame_count() * 2 - 1);
    for f in 0..a.frame_count() {
        let d = diff_grids(&l1_of(&a, f), &l1_of(&b, f)).expect("geometry");
        assert!(d.valid.iter().any(|v| *v), "frame {f}: no valid cells?");
        for m in [
            DiffMetric::Qp,
            DiffMetric::Bpp,
            DiffMetric::MvMag,
            DiffMetric::CoeffEnergy,
            DiffMetric::NzDensity,
        ] {
            let max = d.max_abs(m);
            assert!(
                max.abs() < 1e-4,
                "frame {f}: {m:?} Δ must be 0, got max |Δ| = {max}"
            );
        }
        for i in 0..d.valid.len() {
            if d.valid[i] {
                assert!(d.mode_same(i), "frame {f} cell {i}: mode must agree");
            }
        }
    }

    // Cross-check the ΔbppG path on the same identical pair: all-zero, and
    // Pearson r over a constant Y is undefined (None), not a fake signal.
    let mut bs_a = aggregate_bitstream_to_g(&l1_of(&a, 0), a.width, a.height, 16);
    let bs_b = aggregate_bitstream_to_g(&l1_of(&b, 0), b.width, b.height, 16);
    subtract_bitstream_g(&mut bs_a, &bs_b);
    for (i, v) in bs_a.bpp.iter().enumerate() {
        if bs_a.valid[i] {
            assert!(v.abs() < 1e-5, "cell {i}: Δbpp = {v}");
        }
    }
}
