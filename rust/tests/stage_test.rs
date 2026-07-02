//! M-C stage-image tests: `frames_meta[].stage_images` parsing from the
//! `.catb` meta JSON, the sidecar-resolving BMP loader, and an env-gated
//! test against a real `codec-analyzer decoder-run` workdir.
//!
//! Real-run gate:
//!   E2E_STAGE_RUN = directory containing a *.catb and its stage BMPs
//!   (`frameNNNNNN-{residual,prediction,recon_unfiltered,final_recon}.bmp`)
//!
//! Run:
//!   E2E_STAGE_RUN=/path/run cargo test --test stage_test -- --nocapture

use std::path::Path;
use std::sync::Arc;

use video_viewer::analysis::bitstream_stats::BitstreamFile;
use video_viewer::analysis::stage::{stage_available, StageCache, StageKind};
use video_viewer::core::catb::CatbFile;

mod common;
use common::{block_record, build_catb, encode_strings, frame_record, BlockSpec};

/// One-frame catb whose meta records `stage_images` and an SPS resolution.
fn stage_catb_bytes(stage_images_json: &str, w: u32, h: u32) -> Vec<u8> {
    let strings = encode_strings(&["", "64x64", "Intra", "IDR"]);
    let meta = format!(
        r#"{{"schema_version":1,"decoder":{{"codec":"hevc","name":"ffmpeg-hevc","contract":"0.8.3","capture_level":""}},"parameter_sets":[{{"kind":"SPS","id":0,"fields":{{"width":{w},"height":{h}}}}}],"frames_meta":[{{"slice_headers":[],"dpb":[],"stage_images":{stage_images_json},"exactness_missing":[""],"block_dropped_rows":[0]}}]}}"#
    );
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
    build_catb(1, &strings, meta.as_bytes(), &frame, &block, &[], b"{}")
}

/// Minimal 24-bit bottom-up BMP with a solid colour.
fn write_bmp(path: &Path, w: u32, h: u32, rgb: [u8; 3]) {
    let stride = ((w as usize * 3) + 3) & !3;
    let size = 54 + stride * h as usize;
    let mut b = Vec::with_capacity(size);
    b.extend_from_slice(b"BM");
    b.extend_from_slice(&(size as u32).to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&54u32.to_le_bytes());
    b.extend_from_slice(&40u32.to_le_bytes());
    b.extend_from_slice(&(w as i32).to_le_bytes());
    b.extend_from_slice(&(h as i32).to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&24u16.to_le_bytes());
    b.extend_from_slice(&[0u8; 24]); // BI_RGB + remaining header fields
    assert_eq!(b.len(), 54);
    for _ in 0..h {
        let mut row = Vec::with_capacity(stride);
        for _ in 0..w {
            row.extend_from_slice(&[rgb[2], rgb[1], rgb[0]]);
        }
        row.resize(stride, 0);
        b.extend_from_slice(&row);
    }
    std::fs::write(path, b).expect("write bmp");
}

#[test]
fn test_stage_images_parsed_from_catb_meta() {
    let bytes = stage_catb_bytes(
        r#"{"residual":"frame000000-residual.bmp","final_recon":"frame000000-final_recon.bmp"}"#,
        2,
        2,
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.catb");
    std::fs::write(&path, &bytes).unwrap();
    let catb = CatbFile::open(&path).expect("open catb");
    let meta = &catb.meta.frames_meta[0];
    // scalar_fields sorts key-alphabetically.
    assert_eq!(
        meta.stage_images,
        vec![
            (
                "final_recon".to_string(),
                "frame000000-final_recon.bmp".to_string()
            ),
            (
                "residual".to_string(),
                "frame000000-residual.bmp".to_string()
            ),
        ]
    );
}

#[test]
fn test_stage_images_absent_is_empty() {
    let bytes = stage_catb_bytes("{}", 2, 2);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.catb");
    std::fs::write(&path, &bytes).unwrap();
    let catb = CatbFile::open(&path).expect("open catb");
    assert!(catb.meta.frames_meta[0].stage_images.is_empty());
}

#[test]
fn test_stage_cache_loads_relative_and_sidecar() {
    // final_recon recorded relative (observed layout); prediction recorded
    // under a moved absolute path whose basename still sits next to the
    // .catb (sidecar); residual recorded but missing entirely.
    let bytes = stage_catb_bytes(
        r#"{"final_recon":"frame000000-final_recon.bmp","prediction":"/moved/away/frame000000-prediction.bmp","residual":"frame000000-residual.bmp"}"#,
        2,
        2,
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.catb");
    std::fs::write(&path, &bytes).unwrap();
    write_bmp(
        &dir.path().join("frame000000-final_recon.bmp"),
        2,
        2,
        [10, 20, 30],
    );
    write_bmp(
        &dir.path().join("frame000000-prediction.bmp"),
        2,
        2,
        [40, 50, 60],
    );

    let file = Arc::new(BitstreamFile::open(&path).expect("open bitstream"));
    assert_eq!((file.width, file.height), (2, 2));

    assert!(stage_available(&file, 0, StageKind::FinalRecon));
    assert!(stage_available(&file, 0, StageKind::Prediction));
    assert!(!stage_available(&file, 0, StageKind::Residual));
    // recon_unfiltered was never recorded.
    assert!(!stage_available(&file, 0, StageKind::ReconUnfiltered));

    let mut cache = StageCache::new();
    let recon = cache.get(&file, 0, StageKind::FinalRecon).expect("recon");
    assert_eq!((recon.width, recon.height), (2, 2));
    assert_eq!(&recon.rgb[0..3], &[10, 20, 30]);
    let pred = cache.get(&file, 0, StageKind::Prediction).expect("sidecar");
    assert_eq!(&pred.rgb[0..3], &[40, 50, 60]);
    assert!(cache.get(&file, 0, StageKind::Residual).is_none());
    // Second hit comes from the LRU (same Arc).
    let recon2 = cache.get(&file, 0, StageKind::FinalRecon).expect("cached");
    assert!(Arc::ptr_eq(&recon, &recon2));
    // Out-of-range decode index must not panic.
    assert!(cache.get(&file, 99, StageKind::FinalRecon).is_none());
}

/// Real decoder-run workdir (env-gated): every recorded stage image of the
/// first frames must load and match the stream resolution.
#[test]
fn test_e2e_stage_run() {
    let Ok(dir) = std::env::var("E2E_STAGE_RUN") else {
        eprintln!("E2E_STAGE_RUN not set — skipping");
        return;
    };
    let catb_path = std::fs::read_dir(&dir)
        .expect("read E2E_STAGE_RUN dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().map(|x| x == "catb").unwrap_or(false))
        .expect("no .catb in E2E_STAGE_RUN");
    let file = Arc::new(BitstreamFile::open(&catb_path).expect("open catb"));
    assert!(file.width > 0 && file.height > 0, "stream resolution");

    let mut cache = StageCache::new();
    let mut loaded = 0usize;
    let frames = file.catb.frames.len().min(4);
    for decode_idx in 0..frames {
        for kind in StageKind::ALL {
            if !stage_available(&file, decode_idx, kind) {
                continue;
            }
            let img = cache
                .get(&file, decode_idx, kind)
                .unwrap_or_else(|| panic!("frame {decode_idx} {kind:?} failed to load"));
            assert_eq!(
                (img.width, img.height),
                (file.width, file.height),
                "frame {decode_idx} {kind:?} resolution"
            );
            assert_eq!(
                img.rgb.len(),
                (file.width as usize) * (file.height as usize) * 3
            );
            loaded += 1;
        }
    }
    assert!(loaded > 0, "run has no loadable stage images");
    println!("E2E_STAGE_RUN: loaded {loaded} stage images from {frames} frames");
}
