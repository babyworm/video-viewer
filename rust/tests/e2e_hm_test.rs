//! E2E differential test against a real HM-encoded random-access (B-pyramid)
//! HEVC stream analyzed by codec-analyzer.
//!
//! Gated on environment variables so CI (which has no HM stream) skips it:
//!   E2E_CATB   = path to the codec-analyzer .catb telemetry file
//!   E2E_ORACLE = path to `codec-analyzer frames --format json` output
//!
//! Run:
//!   E2E_CATB=/path/run/codec-analyzer-telemetry.catb \
//!   E2E_ORACLE=/path/oracle_frames.json \
//!   cargo test --test e2e_hm_test -- --nocapture

use video_viewer::analysis::bitstream_stats::{rasterize_blocks, BitstreamFile};

/// One oracle row (subset of the `frames` subcommand JSON schema).
#[derive(Debug)]
struct OracleFrame {
    decode_order: usize,
    poc: i32,
    frame_type: String,
    output: bool,
    decoder_block_count: usize,
}

fn load_oracle(path: &str) -> Vec<OracleFrame> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read E2E_ORACLE {path}: {e}"));
    let rows: serde_json::Value =
        serde_json::from_str(&text).expect("E2E_ORACLE is not valid JSON");
    let rows = rows.as_array().expect("E2E_ORACLE JSON must be an array");
    let mut frames: Vec<OracleFrame> = rows
        .iter()
        .map(|r| OracleFrame {
            decode_order: r["decode_order"].as_u64().expect("decode_order") as usize,
            poc: r["poc"].as_i64().expect("poc") as i32,
            frame_type: r["frame_type"].as_str().unwrap_or("").to_string(),
            output: r["output"].as_bool().unwrap_or(false),
            decoder_block_count: r["decoder_block_count"].as_u64().unwrap_or(0) as usize,
        })
        .collect();
    frames.sort_by_key(|f| f.decode_order);
    frames
}

/// Expected display order derived from the oracle alone: split the
/// decode-order sequence into CVS segments at IDR/BLA/KEY boundaries, sort
/// each segment's output frames by POC (stable), concatenate.
fn expected_display_pocs(oracle: &[OracleFrame]) -> Vec<i32> {
    let mut segments: Vec<Vec<&OracleFrame>> = Vec::new();
    let mut current: Vec<&OracleFrame> = Vec::new();
    for f in oracle {
        let ft = f.frame_type.to_ascii_uppercase();
        let boundary = matches!(ft.as_str(), "IDR" | "BLA" | "KEY");
        if boundary && !current.is_empty() {
            segments.push(std::mem::take(&mut current));
        }
        current.push(f);
    }
    if !current.is_empty() {
        segments.push(current);
    }
    let mut pocs = Vec::new();
    for segment in segments {
        let mut out: Vec<&OracleFrame> = segment.into_iter().filter(|f| f.output).collect();
        out.sort_by_key(|f| f.poc);
        pocs.extend(out.iter().map(|f| f.poc));
    }
    pocs
}

#[test]
fn test_e2e_hm_random_access_stream() {
    let (catb_path, oracle_path) = match (std::env::var("E2E_CATB"), std::env::var("E2E_ORACLE")) {
        (Ok(c), Ok(o)) => (c, o),
        _ => {
            eprintln!("skipping e2e_hm_test: E2E_CATB / E2E_ORACLE not set");
            return;
        }
    };

    let oracle = load_oracle(&oracle_path);
    assert!(!oracle.is_empty(), "oracle has no frames");

    let bs = BitstreamFile::open(&catb_path)
        .unwrap_or_else(|e| panic!("BitstreamFile::open({catb_path}) failed: {e}"));

    // --- Frame counts ---------------------------------------------------
    assert_eq!(
        bs.catb.frames.len(),
        oracle.len(),
        "catb decode-order frame count != oracle frame count"
    );
    let oracle_output_count = oracle.iter().filter(|f| f.output).count();
    assert_eq!(
        bs.display_map.len(),
        oracle_output_count,
        "display_map length != oracle output frame count"
    );
    assert_eq!(bs.frame_count(), oracle_output_count);

    // --- Decode-order POC / type / output agreement ---------------------
    for (i, of) in oracle.iter().enumerate() {
        let cf = &bs.catb.frames[i];
        assert_eq!(cf.poc, of.poc, "decode idx {i}: POC mismatch");
        assert_eq!(
            cf.frame_type.to_ascii_uppercase(),
            of.frame_type.to_ascii_uppercase(),
            "decode idx {i}: frame type mismatch"
        );
        assert_eq!(cf.output, of.output, "decode idx {i}: output flag mismatch");
    }

    // --- Display-order POC sequence vs oracle-derived expectation -------
    let expected = expected_display_pocs(&oracle);
    let actual: Vec<i32> = (0..bs.frame_count())
        .map(|d| bs.frame_summary(d).expect("frame_summary").poc)
        .collect();
    assert_eq!(
        actual, expected,
        "display-order POC sequence mismatch\n actual:   {actual:?}\n expected: {expected:?}"
    );

    // --- Blocks: parse every displayed frame, compare per-frame counts ---
    let mut total_blocks = 0usize;
    for display_idx in 0..bs.frame_count() {
        let decode_idx = bs.decode_idx(display_idx).expect("decode_idx");
        let blocks = bs
            .blocks_display(display_idx)
            .unwrap_or_else(|e| panic!("blocks_display({display_idx}) failed: {e}"));
        assert_eq!(
            blocks.len(),
            oracle[decode_idx].decoder_block_count,
            "decode idx {decode_idx} (POC {}): block count mismatch",
            oracle[decode_idx].poc
        );
        assert!(
            !blocks.is_empty(),
            "decode idx {decode_idx}: no blocks parsed"
        );
        total_blocks += blocks.len();

        // --- Rasterization must succeed and cover the frame -------------
        let grid = rasterize_blocks(&blocks, bs.width, bs.height, 8);
        assert!(
            !grid.is_empty(),
            "display idx {display_idx}: rasterized grid is empty"
        );
        assert!(
            grid.coverage.iter().any(|&c| c > 0.0),
            "display idx {display_idx}: rasterized grid has zero coverage"
        );
    }
    let oracle_total: usize = oracle.iter().map(|f| f.decoder_block_count).sum();
    assert_eq!(total_blocks, oracle_total, "total block count mismatch");

    eprintln!(
        "e2e_hm_test OK: {} frames ({} output), {} blocks, {}x{} ({:?})",
        oracle.len(),
        oracle_output_count,
        total_blocks,
        bs.width,
        bs.height,
        bs.resolution_source
    );
}
