//! Integration tests for the `.catb` v4 reader and the bitstream analysis
//! layer (display_map, resolution, differential tests against oracle JSON).
//!
//! Implemented from the CC0 .catb v4 format specification
//! (codec-analyzer docs/catb-v4-format.md); no GPL code consulted.

use std::path::PathBuf;

use video_viewer::analysis::bitstream_stats::{
    extract_intra_modes, intra_grid_dim, intra_subblock_pos, BitstreamFile, ResolutionSource,
};
use video_viewer::core::catb::{
    CatbFile, MV_HAS_LIST_INDEX, MV_HAS_LONG_TERM, MV_HAS_MV, MV_HAS_MVD, MV_HAS_MVP,
    MV_HAS_POC, MV_LONG_TERM_VALUE, REF_HAS_MV, REF_HAS_MVD, REF_HAS_PU_PART,
};

// Shared synthetic fixture writer (spec §2–§13) — tests/common/mod.rs.
mod common;
use common::{
    block_record, build_catb, build_catb_with_syntax, encode_strings, frame_record, ref_record,
    syntax_record, write_temp, BlockSpec, RefSpec,
};

/// The Appendix A minimal file: 1 IDR frame, 1 intra 64×64 block. 713 bytes.
fn appendix_a_bytes() -> Vec<u8> {
    let strings = encode_strings(&["", "64x64", "Intra", "IDR"]);
    let meta = br#"{"schema_version":1,"decoder":{"codec":"hevc","name":"ffmpeg-hevc","contract":"0.8.3","capture_level":""},"parameter_sets":[],"frames_meta":[{"slice_headers":[],"dpb":[],"stage_images":{},"exactness_missing":[""],"block_dropped_rows":[0]}]}"#;
    let frame = frame_record(0, 0, 3, 1, 0, 0, 0, 100, 20, 80, 0, 1, 0, 28);
    let block = block_record(&BlockSpec {
        w: 64,
        h: 64,
        qp: 32,
        partition_id: 1,
        prediction_mode_id: 2,
        bits: 50,
        bit_offset: 20,
        ..Default::default()
    });
    let aux = br#"{"loop_filters":[],"sao":[]}"#;
    build_catb(1, &strings, meta, &frame, &block, &[], aux)
}

// ===========================================================================
// Synthetic round-trip tests
// ===========================================================================

/// Byte-compare our hand-assembled Appendix A file against the hex dump in
/// the spec (header + directory region, string table, and total size).
#[test]
fn test_catb_appendix_a_byte_exact() {
    let bytes = appendix_a_bytes();
    assert_eq!(bytes.len(), 713, "Appendix A total file size");

    // Header + directory (0x00–0xAF) transcribed from the spec hex dump.
    #[rustfmt::skip]
    let expected_header: Vec<u8> = vec![
        0x43, 0x41, 0x54, 0x42, 0x30, 0x30, 0x30, 0x31, 0x04, 0, 0, 0, 0x01, 0, 0, 0,
        0xb0, 0, 0, 0, 0, 0, 0, 0, 0x21, 0, 0, 0, 0, 0, 0, 0, // strings    off=176 size=33
        0xd1, 0, 0, 0, 0, 0, 0, 0, 0xf0, 0, 0, 0, 0, 0, 0, 0, // meta       off=209 size=240
        0xc1, 0x01, 0, 0, 0, 0, 0, 0, 0x50, 0, 0, 0, 0, 0, 0, 0, // frames  off=449 size=80
        0x11, 0x02, 0, 0, 0, 0, 0, 0, 0x9c, 0, 0, 0, 0, 0, 0, 0, // blocks  off=529 size=156
        0xad, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    // syntax  off=685 size=0
        0xad, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    // cabac
        0xad, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    // transforms
        0xad, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    // refs
        0xad, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    // coeffs
        0xad, 0x02, 0, 0, 0, 0, 0, 0, 0x1c, 0, 0, 0, 0, 0, 0, 0, // frame_aux off=685 size=28
    ];
    assert_eq!(&bytes[..176], expected_header.as_slice());

    // String table bytes at 176 (count=4, "", "64x64", "Intra", "IDR").
    #[rustfmt::skip]
    let expected_strings: Vec<u8> = vec![
        0x04, 0, 0, 0,
        0x00, 0, 0, 0,
        0x05, 0, 0, 0, b'6', b'4', b'x', b'6', b'4',
        0x05, 0, 0, 0, b'I', b'n', b't', b'r', b'a',
        0x03, 0, 0, 0, b'I', b'D', b'R',
    ];
    assert_eq!(&bytes[176..209], expected_strings.as_slice());
}

#[test]
fn test_catb_appendix_a_roundtrip() {
    let f = write_temp(&appendix_a_bytes());
    let catb = CatbFile::open(f.path()).expect("open appendix A file");

    assert_eq!(catb.frame_count, 1);
    assert_eq!(
        catb.strings,
        vec!["".to_string(), "64x64".into(), "Intra".into(), "IDR".into()]
    );
    assert_eq!(catb.meta.schema_version, Some(1));
    assert_eq!(catb.meta.codec, "hevc");
    assert_eq!(catb.meta.contract, "0.8.3");
    assert_eq!(catb.meta.capture_level, "");
    assert!(catb.meta.has_frames_meta);
    assert!(catb.meta.parameter_sets.as_array().unwrap().is_empty());

    let fr = &catb.frames[0];
    assert_eq!(fr.index, 0);
    assert_eq!(fr.poc, 0);
    assert_eq!(fr.frame_type, "IDR");
    assert!(fr.output);
    assert_eq!(fr.vps_id, Some(0));
    assert_eq!(fr.sps_id, Some(0));
    assert_eq!(fr.pps_id, Some(0));
    assert_eq!(fr.slice_bits, Some(100));
    assert_eq!(fr.header_bits, Some(20));
    assert_eq!(fr.data_bits, Some(80));
    assert_eq!((fr.block_off, fr.block_n), (0, 1));
    assert_eq!((fr.aux_off, fr.aux_n), (0, 28));

    let blocks = catb.blocks_for_frame(0).expect("blocks");
    assert_eq!(blocks.len(), 1);
    let b = &blocks[0];
    assert_eq!((b.x, b.y, b.w, b.h), (0, 0, 64, 64));
    assert_eq!(b.ctu_address, 0);
    assert_eq!(b.qp, 32);
    assert_eq!(b.partition, "64x64");
    assert_eq!(b.prediction_mode, "Intra");
    assert_eq!(b.bits, 50);
    assert_eq!(b.bit_offset, 20);
    // mv_flags == 0: all optional fields absent (spec §6.1).
    assert_eq!(b.mv_flags, 0);
    assert!(b.mvp.is_none() && b.mvd.is_none() && b.mv.is_none());
    assert!(b.reference.is_none());
    assert!(b.reference_list.is_none());
    assert!(b.reference_label.is_none());
    assert!(b.reference_list_index.is_none());
    assert!(b.reference_poc.is_none());
    assert!(b.reference_frame.is_none());
    assert!(b.reference_long_term.is_none());
    assert!(catb.refs_for_block(b).expect("refs").is_empty());
}

/// Presence-bit variants: inter block with mvp/mvd/mv, list index, POC,
/// long-term bits, reference strings, and a REF record chain.
#[test]
fn test_catb_presence_bits_roundtrip() {
    let strings = encode_strings(&["", "2Nx2N", "Inter", "P", "L0", "L0[0]", "L0[0] POC 0"]);
    let meta = br#"{"schema_version":1,"decoder":{"codec":"hevc"},"parameter_sets":[],"frames_meta":[]}"#;
    // Frame 0: 2 blocks; sentinel −1 on vps/header/data bits.
    let frame = frame_record(0, 0, 3, 1, -1, 0, 0, 500, -1, -1, 0, 2, 0, 0);
    let inter = BlockSpec {
        x: 16,
        y: 32,
        w: 16,
        h: 8,
        ctu_address: 1,
        qp: 27,
        partition_id: 1,
        prediction_mode_id: 2,
        bits: 64,
        bit_offset: 100,
        ref_off: 0,
        ref_n: 1,
        mv_flags: MV_HAS_MVP
            | MV_HAS_MVD
            | MV_HAS_MV
            | MV_HAS_LIST_INDEX
            | MV_HAS_POC
            | MV_HAS_LONG_TERM
            | MV_LONG_TERM_VALUE,
        mvp: (0, -4),
        mvd: (112, 4),
        mv: (112, 0),
        reference_id: 5,
        reference_list_id: 4,
        reference_label_id: 6,
        reference_list_index: 0,
        reference_poc: 0,
        reference_frame: 99, // MV_HAS_FRAME clear → must be ignored
        ..Default::default()
    };
    // Value fields written non-zero but presence bits clear → absent.
    let absent = BlockSpec {
        x: 32,
        y: 0,
        w: 8,
        h: 8,
        qp: 30,
        mv_flags: 0,
        mvp: (7, 7),
        mvd: (7, 7),
        mv: (7, 7),
        reference_list_index: 3,
        reference_poc: 3,
        reference_frame: 3,
        ..Default::default()
    };
    let mut blocks = block_record(&inter);
    blocks.extend(block_record(&absent));
    let refs = ref_record(&RefSpec {
        list_id: 4,
        list_index: 0,
        reference_poc: 0,
        label_id: 6,
        pu: (16, 32, 16, 8),
        ref_flags: REF_HAS_MVD | REF_HAS_MV | REF_HAS_PU_PART,
        mvp: (9, 9), // REF_HAS_MVP clear → absent
        mvd: (112, 4),
        mv: (112, 0),
        reference_frame: 42, // REF_HAS_FRAME clear → absent
        pu_part_index: 1,
    });
    let bytes = build_catb(1, &strings, meta, &frame, &blocks, &refs, &[]);
    let f = write_temp(&bytes);
    let catb = CatbFile::open(f.path()).expect("open");

    let fr = &catb.frames[0];
    assert_eq!(fr.vps_id, None, "-1 sentinel → None");
    assert_eq!(fr.sps_id, Some(0));
    assert_eq!(fr.slice_bits, Some(500));
    assert_eq!(fr.header_bits, None);
    assert_eq!(fr.data_bits, None);

    let blocks = catb.blocks_for_frame(0).expect("blocks");
    assert_eq!(blocks.len(), 2);
    let b = &blocks[0];
    assert_eq!(b.mvp, Some((0, -4)));
    assert_eq!(b.mvd, Some((112, 4)));
    assert_eq!(b.mv, Some((112, 0)));
    assert_eq!(b.reference.as_deref(), Some("L0[0]"));
    assert_eq!(b.reference_list.as_deref(), Some("L0"));
    assert_eq!(b.reference_label.as_deref(), Some("L0[0] POC 0"));
    assert_eq!(b.reference_list_index, Some(0));
    assert_eq!(b.reference_poc, Some(0));
    assert_eq!(b.reference_frame, None, "HAS_FRAME clear → ignored");
    assert_eq!(b.reference_long_term, Some(true));

    let a = &blocks[1];
    assert!(a.mvp.is_none() && a.mvd.is_none() && a.mv.is_none());
    assert!(a.reference_list_index.is_none());
    assert!(a.reference_poc.is_none());
    assert!(a.reference_frame.is_none());
    assert!(a.reference_long_term.is_none());

    let refs = catb.refs_for_block(b).expect("refs");
    assert_eq!(refs.len(), 1);
    let r = &refs[0];
    assert_eq!(r.list.as_deref(), Some("L0"));
    assert_eq!(r.list_index, 0);
    assert_eq!(r.reference_poc, 0);
    assert_eq!(r.label.as_deref(), Some("L0[0] POC 0"));
    assert_eq!((r.pu_x, r.pu_y, r.pu_w, r.pu_h), (16, 32, 16, 8));
    assert_eq!(r.mvp, None, "REF_HAS_MVP clear → absent");
    assert_eq!(r.mvd, Some((112, 4)));
    assert_eq!(r.mv, Some((112, 0)));
    assert_eq!(r.reference_frame, None);
    assert_eq!(r.long_term, None);
    assert_eq!(r.pu_part_index, Some(1));
}

/// SYNTAX section round-trip (M4): a block with two syntax rows resolves
/// name/value strings and bits; the clamp guard rejects hostile counts.
#[test]
fn test_catb_syntax_for_block_roundtrip_and_guard() {
    let strings = encode_strings(&[
        "",
        "64x64",
        "Intra",
        "IDR",
        "intra_luma_pred_mode",
        "26",
        "cabac",
        "split_cu_flag",
        "0",
    ]);
    let meta = br#"{"schema_version":1,"decoder":{"codec":"hevc"},"parameter_sets":[]}"#;
    let frame = frame_record(0, 0, 3, 1, 0, 0, 0, 100, 20, 80, 0, 1, 0, 0);
    let block = block_record(&BlockSpec {
        w: 64,
        h: 64,
        qp: 32,
        partition_id: 1,
        prediction_mode_id: 2,
        syntax_off: 0,
        syntax_n: 2,
        ..Default::default()
    });
    let mut syntax = syntax_record(7, 8, 6, 0, 1); // split_cu_flag = 0
    syntax.extend(syntax_record(4, 5, 6, 1, 5)); // intra_luma_pred_mode = 26
    let bytes = build_catb_with_syntax(1, &strings, meta, &frame, &block, &syntax, &[], &[]);
    let f = write_temp(&bytes);
    let catb = CatbFile::open(f.path()).expect("open");
    let b = &catb.blocks_for_frame(0).expect("blocks")[0];
    let rows = catb.syntax_for_block(b).expect("syntax rows");
    assert_eq!(rows.len(), 2);
    assert_eq!((rows[0].name.as_str(), rows[0].value.as_str(), rows[0].bits), ("split_cu_flag", "0", 1));
    assert_eq!(
        (rows[1].name.as_str(), rows[1].value.as_str(), rows[1].bits),
        ("intra_luma_pred_mode", "26", 5)
    );
    // The extracted intra dir: HEVC mode 26 = vertical (90°).
    let dirs = extract_intra_modes(&rows);
    assert_eq!(dirs.len(), 1);
    assert_eq!(dirs[0].mode, 26);
    assert_eq!(dirs[0].angle_deg, Some(90.0));

    // OOM guard (0.12.1 F1 pattern): a hostile syntax_n larger than the
    // SYNTAX section must error before allocating.
    let mut hostile = b.clone();
    hostile.syntax_n = i32::MAX;
    let err = catb.syntax_for_block(&hostile).expect_err("must reject syntax_n");
    assert!(err.contains("SYNTAX section"), "unexpected error: {err}");
    hostile.syntax_n = 0;
    assert!(catb.syntax_for_block(&hostile).expect("zero rows ok").is_empty());
}

// ===========================================================================
// Error cases
// ===========================================================================

/// §7 fixed wording (UX spec 04) — string equality is the contract.
const UNSUPPORTED_MSG: &str =
    "Unsupported .catb (need v4) — regenerate with codec-analyzer ≥0.8.3";

#[test]
fn test_catb_bad_magic() {
    let mut bytes = appendix_a_bytes();
    bytes[0..8].copy_from_slice(b"NOTCATB!");
    let f = write_temp(&bytes);
    let err = CatbFile::open(f.path()).unwrap_err();
    assert_eq!(err, UNSUPPORTED_MSG);
}

#[test]
fn test_catb_bad_version() {
    let mut bytes = appendix_a_bytes();
    bytes[8..12].copy_from_slice(&3u32.to_le_bytes());
    let f = write_temp(&bytes);
    let err = CatbFile::open(f.path()).unwrap_err();
    assert_eq!(err, UNSUPPORTED_MSG);
}

#[test]
fn test_catb_truncated_file() {
    let bytes = appendix_a_bytes();
    // Cut past the header/directory but before the end of the blocks
    // section: rejected by the directory bounds check.
    let f = write_temp(&bytes[..500]);
    let err = CatbFile::open(f.path()).unwrap_err();
    assert!(err.contains("out of bounds"), "got: {err}");
    // Cut inside the directory itself (entry 0 offset field is incomplete):
    // the directory read hits EOF.
    let f2 = write_temp(&bytes[..20]);
    let err2 = CatbFile::open(f2.path()).unwrap_err();
    assert!(err2.contains("truncated file"), "got: {err2}");
}

#[test]
fn test_catb_directory_out_of_bounds() {
    let mut bytes = appendix_a_bytes();
    // Corrupt the blocks section offset (directory entry 3) to point past EOF.
    let base = 16 + 3 * 16;
    bytes[base..base + 8].copy_from_slice(&1_000_000u64.to_le_bytes());
    let f = write_temp(&bytes);
    let err = CatbFile::open(f.path()).unwrap_err();
    assert!(err.contains("out of bounds"), "got: {err}");
}

// ===========================================================================
// Real fixtures: smoke + differential vs oracle JSON
// ===========================================================================

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root")
        .join("test_data/bitstream")
}

const FIXTURES: &[(&str, &str, u32)] = &[
    ("hevc_intra", "hevc_intra", 2),
    ("hevc_inter", "hevc_inter", 3),
    ("hevc_bslice", "hevc_bslice", 4),
    ("hevc_multi_idr", "hevc_multi_idr", 6),
    ("avc_mb", "avc_cavlc", 2),
];

fn open_fixture(dir: &str, name: &str) -> CatbFile {
    let path = fixture_dir().join(dir).join(format!("{name}.catb"));
    CatbFile::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
}

fn load_oracle(dir: &str, name: &str, kind: &str) -> serde_json::Value {
    let path = fixture_dir().join(dir).join(format!("{name}_oracle_{kind}.json"));
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&data).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// Parse an oracle MV string like `"112,0"` or `"(112, 0)"`; `""` → None.
fn parse_oracle_mv(v: &serde_json::Value) -> Option<(i32, i32)> {
    let s = v.as_str()?;
    let s = s.trim().trim_start_matches('(').trim_end_matches(')');
    if s.is_empty() {
        return None;
    }
    let mut it = s.split(',');
    let x = it.next()?.trim().parse().ok()?;
    let y = it.next()?.trim().parse().ok()?;
    Some((x, y))
}

#[test]
fn test_catb_fixture_smoke_all() {
    for &(dir, name, expected_frames) in FIXTURES {
        let catb = open_fixture(dir, name);
        assert_eq!(catb.frame_count, expected_frames, "{dir} frame_count");
        assert_eq!(catb.frames.len(), expected_frames as usize);
        // Every frame's blocks must parse without error.
        for i in 0..catb.frames.len() {
            let blocks = catb.blocks_for_frame(i).unwrap_or_else(|e| panic!("{dir} f{i}: {e}"));
            assert_eq!(blocks.len(), catb.frames[i].block_n as usize);
            for b in &blocks {
                catb.refs_for_block(b).unwrap_or_else(|e| panic!("{dir} f{i} refs: {e}"));
            }
        }
    }
}

/// Differential test — frames: poc / frame_type / output / bits / block count
/// must match the codec-analyzer CLI oracle dump.
#[test]
fn test_catb_differential_frames() {
    for &(dir, name, _) in FIXTURES {
        let catb = open_fixture(dir, name);
        let oracle = load_oracle(dir, name, "frames");
        let oracle = oracle.as_array().expect("oracle frames array");
        assert_eq!(oracle.len(), catb.frames.len(), "{dir}: oracle frame count");
        for of in oracle {
            let decode_order = of["decode_order"].as_u64().expect("decode_order") as usize;
            let fr = &catb.frames[decode_order];
            let ctx = format!("{dir} decode#{decode_order}");
            assert_eq!(fr.index as u64, of["frame_index"].as_u64().unwrap(), "{ctx} index");
            assert_eq!(fr.poc as i64, of["poc"].as_i64().unwrap(), "{ctx} poc");
            assert_eq!(fr.frame_type, of["frame_type"].as_str().unwrap(), "{ctx} frame_type");
            assert_eq!(fr.output, of["output"].as_bool().unwrap(), "{ctx} output");
            // Bits fields: oracle emits numbers when present.
            if let Some(bits) = of["slice_bits"].as_i64() {
                assert_eq!(fr.slice_bits, Some(bits), "{ctx} slice_bits");
            }
            if let Some(bits) = of["header_bits"].as_i64() {
                assert_eq!(fr.header_bits, Some(bits), "{ctx} header_bits");
            }
            if let Some(bits) = of["data_bits"].as_i64() {
                assert_eq!(fr.data_bits, Some(bits), "{ctx} data_bits");
            }
            if let Some(n) = of["block_count"].as_u64() {
                assert_eq!(fr.block_n as u64, n, "{ctx} block_count");
            }
        }
    }
}

/// Differential test — blocks: per-frame block count and each block's
/// geometry, qp, bits, prediction_mode, partition, MVs, and reference label.
#[test]
fn test_catb_differential_blocks() {
    for &(dir, name, _) in FIXTURES {
        let catb = open_fixture(dir, name);
        let oracle = load_oracle(dir, name, "blocks");
        let oracle = oracle.as_array().expect("oracle blocks array");

        // Group oracle rows by decode-order frame index ("frame" key).
        let mut per_frame: Vec<Vec<&serde_json::Value>> = vec![Vec::new(); catb.frames.len()];
        for ob in oracle {
            let fi = ob["frame"].as_u64().expect("block frame idx") as usize;
            per_frame[fi].push(ob);
        }

        for (fi, oracle_blocks) in per_frame.iter().enumerate() {
            let blocks = catb.blocks_for_frame(fi).expect("blocks");
            assert_eq!(blocks.len(), oracle_blocks.len(), "{dir} f{fi} block count");
            for (bi, (b, ob)) in blocks.iter().zip(oracle_blocks.iter()).enumerate() {
                let ctx = format!("{dir} f{fi} b{bi}");
                assert_eq!(b.x as u64, ob["x"].as_u64().unwrap(), "{ctx} x");
                assert_eq!(b.y as u64, ob["y"].as_u64().unwrap(), "{ctx} y");
                assert_eq!(b.w as u64, ob["w"].as_u64().unwrap(), "{ctx} w");
                assert_eq!(b.h as u64, ob["h"].as_u64().unwrap(), "{ctx} h");
                assert_eq!(b.qp as i64, ob["qp"].as_i64().unwrap(), "{ctx} qp");
                assert_eq!(b.bits, ob["bits"].as_i64().unwrap(), "{ctx} bits");
                assert_eq!(
                    b.prediction_mode,
                    ob["prediction_mode"].as_str().unwrap(),
                    "{ctx} prediction_mode"
                );
                assert_eq!(b.partition, ob["partition"].as_str().unwrap(), "{ctx} partition");
                // MV presence AND value must match ("" in oracle = absent).
                assert_eq!(b.mv, parse_oracle_mv(&ob["mv"]), "{ctx} mv");
                assert_eq!(b.mvp, parse_oracle_mv(&ob["mvp"]), "{ctx} mvp");
                assert_eq!(b.mvd, parse_oracle_mv(&ob["mvd"]), "{ctx} mvd");
                // Reference summary label ("" = absent).
                let oref = ob["reference"].as_str().unwrap_or("");
                assert_eq!(b.reference.as_deref().unwrap_or(""), oref, "{ctx} reference");
            }
        }
    }
}

/// Differential test — intra modes (M4): the modes extracted from each
/// block's SYNTAX rows must agree with the oracle's
/// `intra_prediction_mode` family:
/// - `intra_prediction_mode_count` == number of extracted luma modes,
/// - non-`mixed` blocks: every extracted mode equals the oracle value,
/// - `mixed` blocks: the oracle's representative value is among ours.
///
/// Covers HEVC (`intra_luma_pred_mode`) and AVC
/// (`intra4x4/8x8/16x16_pred_mode`), the two intra-carrying fixture
/// families the task requires (hevc_intra / avc_cavlc) plus hevc_bslice's
/// angular modes.
#[test]
fn test_catb_differential_intra_modes() {
    let mut checked = 0usize;
    for (dir, name) in [
        ("hevc_intra", "hevc_intra"),
        ("hevc_bslice", "hevc_bslice"),
        ("avc_mb", "avc_cavlc"),
    ] {
        let catb = open_fixture(dir, name);
        let oracle = load_oracle(dir, name, "blocks");
        let oracle = oracle.as_array().expect("oracle blocks array");
        let mut per_frame: Vec<Vec<&serde_json::Value>> = vec![Vec::new(); catb.frames.len()];
        for ob in oracle {
            let fi = ob["frame"].as_u64().expect("block frame idx") as usize;
            per_frame[fi].push(ob);
        }
        for (fi, oracle_blocks) in per_frame.iter().enumerate() {
            let blocks = catb.blocks_for_frame(fi).expect("blocks");
            for (bi, (b, ob)) in blocks.iter().zip(oracle_blocks.iter()).enumerate() {
                let source = ob["intra_prediction_mode_source"].as_str().unwrap_or("");
                let rows = catb.syntax_for_block(b).expect("syntax rows");
                let dirs = extract_intra_modes(&rows);
                let ctx = format!("{dir} f{fi} b{bi}");
                if source.is_empty() {
                    // Oracle says no intra mode → we must extract none.
                    assert!(dirs.is_empty(), "{ctx}: spurious intra modes {dirs:?}");
                    continue;
                }
                let count = ob["intra_prediction_mode_count"].as_u64().unwrap() as usize;
                let value = ob["intra_prediction_mode"].as_i64().unwrap() as i32;
                let mixed = ob["intra_prediction_mode_mixed"].as_bool().unwrap_or(false);
                assert_eq!(dirs.len(), count, "{ctx}: mode count");
                let modes: Vec<i32> = dirs.iter().map(|d| d.mode).collect();
                if mixed {
                    assert!(
                        modes.contains(&value),
                        "{ctx}: representative {value} not in {modes:?}"
                    );
                } else {
                    assert!(
                        modes.iter().all(|&m| m == value),
                        "{ctx}: expected all {value}, got {modes:?}"
                    );
                }
                checked += 1;
            }
        }
    }
    assert!(checked >= 20, "expected ≥20 intra blocks verified, got {checked}");
}

/// Regression — AVC intra4x4 sub-block ordering is H.264 §6.4.3 z-scan,
/// not raster. Evidence block: `avc_cavlc` frame 0, block at (48, 0)
/// (16x16, mixed intra4x4). It sits on the frame's top row, so FFmpeg
/// substitutes LEFT_DC (mode 9) exactly where the top samples are
/// unavailable: the MB's top 4×4 row. In z-scan order those are list
/// positions {0,1,4,5}; a raster reading would demand {0,1,2,3} *and*
/// would put Vertical (mode 0, needs top samples) on the frame edge —
/// internally contradictory. `intra_subblock_pos` must therefore map every
/// mode-9 entry (and only those) of this block to grid row 0.
#[test]
fn test_catb_avc_intra4x4_zscan_placement() {
    let catb = open_fixture("avc_mb", "avc_cavlc");
    let blocks = catb.blocks_for_frame(0).expect("frame 0 blocks");
    let b = blocks
        .iter()
        .find(|b| b.x == 48 && b.y == 0 && b.w == 16 && b.h == 16)
        .expect("16x16 block at (48,0)");
    let dirs = extract_intra_modes(&catb.syntax_for_block(b).expect("syntax rows"));
    assert_eq!(dirs.len(), 16, "mixed intra4x4 MB has 16 luma modes");
    // Stored (z-scan) order as observed in the fixture's SYNTAX rows.
    let modes: Vec<i32> = dirs.iter().map(|d| d.mode).collect();
    assert_eq!(&modes[..8], &[9, 9, 0, 0, 9, 9, 0, 0], "z-scan prefix");
    // Placement: LEFT_DC (top unavailable) ⇔ grid row 0, nothing else.
    let dim = intra_grid_dim(dirs.len());
    assert_eq!(dim, 4);
    for (k, d) in dirs.iter().enumerate() {
        let (col, row) = intra_subblock_pos(k, dim).expect("in grid");
        assert_eq!(
            row == 0,
            d.mode == 9,
            "k={k} mode={} placed at ({col},{row}) — top row iff LEFT_DC",
            d.mode
        );
    }
}

// ===========================================================================
// BitstreamFile: display_map, resolution, summaries, cache
// ===========================================================================

fn open_bitstream(dir: &str, name: &str) -> BitstreamFile {
    let path = fixture_dir().join(dir).join(format!("{name}.catb"));
    BitstreamFile::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
}

/// B-slice stream decodes as POC 0,3,1,2 — display order must be POC 0,1,2,3.
#[test]
fn test_catb_display_map_bslice_reorder() {
    let bs = open_bitstream("hevc_bslice", "hevc_bslice");
    assert_eq!(bs.display_map, vec![0, 2, 3, 1]);
    let display_pocs: Vec<i32> = (0..bs.frame_count())
        .map(|i| bs.frame_summary(i).unwrap().poc)
        .collect();
    assert_eq!(display_pocs, vec![0, 1, 2, 3]);
    // Types in display order: IDR, B, B, P.
    let types: Vec<String> = (0..bs.frame_count())
        .map(|i| bs.frame_summary(i).unwrap().frame_type.clone())
        .collect();
    assert_eq!(types, vec!["IDR", "B", "B", "P"]);
}

/// Multi-IDR stream: POC resets at decode order 3. A naive global POC sort
/// would interleave the two GOPs; the CVS-aware map must keep the segments
/// in decode order: [0,1,2] then [3,4,5].
#[test]
fn test_catb_display_map_multi_idr_segments() {
    let bs = open_bitstream("hevc_multi_idr", "hevc_multi_idr");
    assert_eq!(bs.frame_count(), 6, "all 6 frames mapped");
    assert_eq!(bs.display_map, vec![0, 1, 2, 3, 4, 5]);

    // Derive the expectation from the oracle: segment boundaries at IDR,
    // POC-sorted within each segment.
    let oracle = load_oracle("hevc_multi_idr", "hevc_multi_idr", "frames");
    let oracle = oracle.as_array().unwrap();
    let mut expected: Vec<usize> = Vec::new();
    let mut segment: Vec<(i64, usize)> = Vec::new();
    for of in oracle {
        let dec = of["decode_order"].as_u64().unwrap() as usize;
        if of["frame_type"].as_str().unwrap() == "IDR" && !segment.is_empty() {
            segment.sort();
            expected.extend(segment.drain(..).map(|(_, d)| d));
        }
        if of["output"].as_bool().unwrap() {
            segment.push((of["poc"].as_i64().unwrap(), dec));
        }
    }
    segment.sort();
    expected.extend(segment.drain(..).map(|(_, d)| d));
    assert_eq!(bs.display_map, expected, "oracle-derived display map");

    // Segment boundary respected: first GOP's frames all precede second GOP's.
    let boundary_pos = bs.display_map.iter().position(|&d| d == 3).unwrap();
    assert!(bs.display_map[..boundary_pos].iter().all(|&d| d < 3));
    assert!(bs.display_map[boundary_pos..].iter().all(|&d| d >= 3));
}

#[test]
fn test_catb_display_map_identity_for_intra_and_avc() {
    for (dir, name, n) in [("hevc_intra", "hevc_intra", 2), ("hevc_inter", "hevc_inter", 3), ("avc_mb", "avc_cavlc", 2)] {
        let bs = open_bitstream(dir, name);
        assert_eq!(bs.display_map, (0..n).collect::<Vec<usize>>(), "{dir}");
    }
}

/// Resolution: all five fixtures carry SPS width/height in
/// meta.parameter_sets; values must match the oracle's frame_width/height.
#[test]
fn test_catb_resolution_from_parameter_sets() {
    for &(dir, name, _) in FIXTURES {
        let bs = open_bitstream(dir, name);
        let oracle = load_oracle(dir, name, "blocks");
        let ob = &oracle.as_array().unwrap()[0];
        let ow = ob["frame_width"].as_u64().unwrap() as u32;
        let oh = ob["frame_height"].as_u64().unwrap() as u32;
        assert_eq!((bs.width, bs.height), (ow, oh), "{dir} resolution");
        assert_eq!(
            bs.resolution_source,
            ResolutionSource::ParameterSets,
            "{dir} resolution source"
        );
    }
}

/// Resolution fallback: a synthetic file without parameter_sets must fall
/// back to the block bounding extent.
#[test]
fn test_catb_resolution_block_extent_fallback() {
    let strings = encode_strings(&["", "64x64", "Intra", "IDR"]);
    let meta = br#"{"schema_version":1,"decoder":{"codec":"hevc"},"parameter_sets":[],"frames_meta":[]}"#;
    let frame = frame_record(0, 0, 3, 1, 0, 0, 0, 100, 20, 80, 0, 2, 0, 0);
    let mut blocks = block_record(&BlockSpec { w: 64, h: 64, partition_id: 1, prediction_mode_id: 2, ..Default::default() });
    blocks.extend(block_record(&BlockSpec { x: 64, y: 0, w: 32, h: 48, partition_id: 1, prediction_mode_id: 2, ..Default::default() }));
    let bytes = build_catb(1, &strings, meta, &frame, &blocks, &[], &[]);
    let f = write_temp(&bytes);
    let bs = BitstreamFile::open(f.path()).expect("open");
    assert_eq!((bs.width, bs.height), (96, 64));
    assert_eq!(bs.resolution_source, ResolutionSource::BlockExtent);
}

#[test]
fn test_catb_frame_summary_and_block_cache() {
    let bs = open_bitstream("hevc_bslice", "hevc_bslice");
    // Display frame 1 is decode frame 2 (POC 1, B).
    let s = bs.frame_summary(1).expect("summary");
    assert_eq!(s.decode_idx, 2);
    assert_eq!(s.poc, 1);
    assert_eq!(s.frame_type, "B");
    assert_eq!(s.slice_bits, Some(2984));
    assert_eq!(s.block_count, 16);
    assert!(bs.frame_summary(99).is_none());

    // Cache: two lookups return the same Arc.
    let a = bs.blocks_display(1).expect("blocks");
    let b = bs.blocks_display(1).expect("blocks");
    assert!(std::sync::Arc::ptr_eq(&a, &b));
    assert_eq!(a.len(), 16);
    assert!(bs.blocks_display(99).is_err());
}

/// A malformed/hostile `block_n` (e.g. i32::MAX) must fail with an error —
/// never reach `Vec::with_capacity` and abort on OOM.
#[test]
fn test_catb_block_count_exceeding_section_errors() {
    let strings = encode_strings(&["", "IDR"]);
    let meta = br#"{"schema_version":1,"decoder":{"codec":"hevc"},"parameter_sets":[]}"#;
    // One frame claiming i32::MAX blocks; the BLOCK section holds exactly one.
    let frames = frame_record(0, 0, 1, 1, 0, 0, 0, 100, 20, 80, 0, i32::MAX, 0, 0);
    let blocks = block_record(&BlockSpec {
        w: 64,
        h: 64,
        qp: 32,
        ..Default::default()
    });
    let bytes = build_catb(1, &strings, meta, &frames, &blocks, &[], &[]);
    let f = write_temp(&bytes);
    let catb = CatbFile::open(f.path()).expect("header/frames still parse");
    let err = catb.blocks_for_frame(0).expect_err("must reject block_n");
    assert!(err.contains("BLOCK section"), "unexpected error: {err}");
    // Same guard on the REF side: a block claiming refs an empty REF
    // section cannot back.
    let mut b = video_viewer::core::catb::BsBlock {
        x: 0,
        y: 0,
        w: 64,
        h: 64,
        ctu_address: 0,
        qp: 32,
        partition: String::new(),
        prediction_mode: String::new(),
        bits: 0,
        bit_offset: 0,
        exactness_flags: 0,
        syntax_off: 0,
        syntax_n: 0,
        cabac_off: 0,
        cabac_n: 0,
        tx_off: 0,
        tx_n: 0,
        ref_off: 0,
        ref_n: i32::MAX,
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
    };
    let err = catb.refs_for_block(&b).expect_err("must reject ref_n");
    assert!(err.contains("REF section"), "unexpected error: {err}");
    b.ref_n = 0;
    assert!(catb.refs_for_block(&b).expect("zero refs ok").is_empty());
}

/// Malformed block coordinates must not derive an absurd resolution that
/// makes every downstream grid allocation explode: `BitstreamFile::open`
/// rejects extents beyond the sanity ceiling.
#[test]
fn test_catb_insane_block_extent_rejected() {
    let strings = encode_strings(&["", "IDR"]);
    // No parameter_sets → resolution falls back to the block extent.
    let meta = br#"{"schema_version":1,"decoder":{"codec":"hevc"},"parameter_sets":[]}"#;
    let frames = frame_record(0, 0, 1, 1, 0, 0, 0, 100, 20, 80, 0, 1, 0, 0);
    let blocks = block_record(&BlockSpec {
        x: 1 << 30,
        y: 1 << 30,
        w: 64,
        h: 64,
        qp: 32,
        ..Default::default()
    });
    let bytes = build_catb(1, &strings, meta, &frames, &blocks, &[], &[]);
    let f = write_temp(&bytes);
    let err = BitstreamFile::open(f.path()).expect_err("must reject extent");
    assert!(err.contains("exceeds"), "unexpected error: {err}");
}
