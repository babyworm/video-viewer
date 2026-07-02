//! M-E: `frame_aux` section tests — lazy per-frame loop-filter / SAO JSON
//! parsing (spec §12), lenient handling of truncated/mistyped blobs, the
//! `BitstreamFile` 1-frame LRU, and real-fixture structure checks.
//!
//! Implemented from the CC0 .catb v4 format specification; no GPL code
//! consulted. Fixture structure (key set, row counts) was measured directly
//! from the repo fixtures.

use std::path::{Path, PathBuf};

use video_viewer::analysis::bitstream_stats::BitstreamFile;
use video_viewer::core::catb::CatbFile;

mod common;
use common::{block_record, build_catb, encode_strings, frame_record, write_temp, BlockSpec};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../test_data/bitstream")
        .join(name)
}

/// Minimal 1-frame catb whose frame_aux section is `aux`, with the FRAME
/// record's `(aux_off, aux_n)` given explicitly.
fn catb_with_aux(aux: &[u8], aux_off: u64, aux_n: i64) -> Vec<u8> {
    let strings = encode_strings(&["", "64x64", "Intra", "IDR"]);
    let meta = br#"{"schema_version":1,"decoder":{"codec":"hevc","contract":"0.8.3","capture_level":""}}"#;
    let frame = frame_record(0, 0, 3, 1, 0, 0, 0, 100, 20, 80, 0, 1, aux_off, aux_n);
    let block = block_record(&BlockSpec {
        w: 64,
        h: 64,
        qp: 30,
        partition_id: 1,
        prediction_mode_id: 2,
        bits: 50,
        ..Default::default()
    });
    build_catb(1, &strings, meta, &frame, &block, &[], aux)
}

#[test]
fn test_frame_aux_synthetic_rows() {
    let aux = br#"{"loop_filters":[
        {"x":8,"y":0,"w":1,"h":4,"orientation":"vertical","segment":0,
         "boundary_strength":2,"filter_strength":"strong","reason":"","qp":22},
        {"x":0,"y":8,"w":4,"h":1,"orientation":"horizontal","segment":1,
         "boundary_strength":1,"filter_strength":"weak","reason":"","qp":23},
        {"x":16,"y":0,"w":1,"h":4,"orientation":"vertical",
         "boundary_strength":0,"filter_strength":"none",
         "reason":"boundary_strength_zero","qp":22}
    ],"sao":[
        {"x":0,"y":0,"w":64,"h":64,"type_y":"edge","type_cb":"not_applied",
         "type_cr":"not_applied","merge_left":true,"merge_up":false}
    ]}"#;
    let bytes = catb_with_aux(aux, 0, aux.len() as i64);
    let f = write_temp(&bytes);
    let catb = CatbFile::open(f.path()).unwrap();
    let a = catb.frame_aux_for_frame(0).unwrap();
    assert_eq!(a.loop_filters.len(), 3);
    let r0 = &a.loop_filters[0];
    assert!(r0.vertical);
    assert_eq!((r0.x, r0.y, r0.w, r0.h), (8, 0, 1, 4));
    assert_eq!(r0.boundary_strength, 2);
    assert_eq!(r0.filter_strength, "strong");
    assert_eq!(r0.qp, 22);
    let r1 = &a.loop_filters[1];
    assert!(!r1.vertical);
    assert_eq!(r1.filter_strength, "weak");
    let r2 = &a.loop_filters[2];
    assert_eq!(r2.reason, "boundary_strength_zero");
    assert_eq!(a.sao.len(), 1);
    assert!(a.sao[0].applied(), "type_y=edge counts as applied");
    assert!(a.sao[0].merge_left);
}

#[test]
fn test_frame_aux_empty_blob_and_zero_len() {
    // The reference writer's 28-byte empty blob.
    let aux = br#"{"loop_filters":[],"sao":[]}"#;
    assert_eq!(aux.len(), 28);
    let bytes = catb_with_aux(aux, 0, 28);
    let f = write_temp(&bytes);
    let catb = CatbFile::open(f.path()).unwrap();
    let a = catb.frame_aux_for_frame(0).unwrap();
    assert!(a.loop_filters.is_empty());
    assert!(a.sao.is_empty());

    // aux_n == 0 → "no aux" per spec: empty result, no error.
    let bytes = catb_with_aux(aux, 0, 0);
    let f = write_temp(&bytes);
    let catb = CatbFile::open(f.path()).unwrap();
    let a = catb.frame_aux_for_frame(0).unwrap();
    assert!(a.loop_filters.is_empty() && a.sao.is_empty());
}

#[test]
fn test_frame_aux_truncated_json_is_error_not_panic() {
    // aux_n cuts into the JSON → invalid blob → Err (never a panic).
    let aux = br#"{"loop_filters":[{"x":8,"y":0}],"sao":[]}"#;
    let bytes = catb_with_aux(aux, 0, 20);
    let f = write_temp(&bytes);
    let catb = CatbFile::open(f.path()).unwrap();
    assert!(catb.frame_aux_for_frame(0).is_err());
}

#[test]
fn test_frame_aux_out_of_range_and_mistyped_rows() {
    // (aux_off, aux_n) beyond the section → Err.
    let aux = br#"{"loop_filters":[],"sao":[]}"#;
    let bytes = catb_with_aux(aux, 10, 28);
    let f = write_temp(&bytes);
    let catb = CatbFile::open(f.path()).unwrap();
    assert!(catb.frame_aux_for_frame(0).is_err());

    // Mistyped members decay leniently: non-object rows skipped, wrong
    // scalar types default.
    let aux = br#"{"loop_filters":[7,"x",{"x":"oops","orientation":3}],"sao":"nope"}"#;
    let bytes = catb_with_aux(aux, 0, aux.len() as i64);
    let f = write_temp(&bytes);
    let catb = CatbFile::open(f.path()).unwrap();
    let a = catb.frame_aux_for_frame(0).unwrap();
    assert_eq!(a.loop_filters.len(), 1, "only the object row survives");
    assert_eq!(a.loop_filters[0].x, 0);
    assert!(!a.loop_filters[0].vertical);
    assert!(a.sao.is_empty());

    // Frame index out of range → Err, no panic.
    assert!(catb.frame_aux_for_frame(7).is_err());
}

#[test]
fn test_frame_aux_lru_identity() {
    // The BitstreamFile layer caches exactly one frame's aux (Arc identity
    // stable across repeat fetches of the same decode index).
    let file = BitstreamFile::open(fixture("hevc_bslice/hevc_bslice.catb")).unwrap();
    let a1 = file.frame_aux_decode(0).unwrap();
    let a2 = file.frame_aux_decode(0).unwrap();
    assert!(std::sync::Arc::ptr_eq(&a1, &a2), "cache hit shares the Arc");
    let b = file.frame_aux_decode(1).unwrap();
    assert!(!std::sync::Arc::ptr_eq(&a1, &b));
    // Going back re-parses (1-frame LRU) but yields equal content.
    let a3 = file.frame_aux_decode(0).unwrap();
    assert_eq!(a1.loop_filters, a3.loop_filters);
}

// ===========================================================================
// Real fixtures (measured structure)
// ===========================================================================

#[test]
fn test_frame_aux_fixture_hevc_bslice() {
    // Measured: every frame carries 224 loop-filter rows (112 vertical +
    // 112 horizontal, 1×4 / 4×1 segments) and 4 SAO rows (all not_applied).
    let catb = CatbFile::open(fixture("hevc_bslice/hevc_bslice.catb")).unwrap();
    assert_eq!(catb.frame_count, 4);
    let a = catb.frame_aux_for_frame(0).unwrap();
    assert_eq!(a.loop_filters.len(), 224);
    let vertical = a.loop_filters.iter().filter(|r| r.vertical).count();
    assert_eq!(vertical, 112);
    assert!(a
        .loop_filters
        .iter()
        .all(|r| (r.w == 1 && r.h == 4) == r.vertical));
    // Measured strength distribution for frame 0: 172 none / 52 weak.
    let weak = a
        .loop_filters
        .iter()
        .filter(|r| r.filter_strength == "weak")
        .count();
    assert_eq!(weak, 52);
    assert_eq!(a.sao.len(), 4);
    assert!(a.sao.iter().all(|s| !s.applied()), "fixture SAO all not_applied");
}

#[test]
fn test_frame_aux_fixture_strengths_and_avc_empty() {
    // hevc_inter frame 0 (measured): strong edges present (BS 2).
    let catb = CatbFile::open(fixture("hevc_inter/hevc_inter.catb")).unwrap();
    let a = catb.frame_aux_for_frame(0).unwrap();
    assert_eq!(a.loop_filters.len(), 224);
    assert!(a
        .loop_filters
        .iter()
        .any(|r| r.filter_strength == "strong" && r.boundary_strength == 2));

    // AVC fixture (measured): 28-byte empty blobs on both frames.
    let catb = CatbFile::open(fixture("avc_mb/avc_cavlc.catb")).unwrap();
    for i in 0..catb.frame_count as usize {
        let a = catb.frame_aux_for_frame(i).unwrap();
        assert!(a.loop_filters.is_empty());
        assert!(a.sao.is_empty());
    }
}
