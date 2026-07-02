//! Dropped-file classification against the real `test_data/bitstream`
//! fixtures (skipped when a fixture is absent, integration_test.rs
//! convention).

use std::path::PathBuf;

use video_viewer::core::dropped::{classify_dropped_file, DroppedKind};

fn fixture(rel: &str) -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../test_data/bitstream")
        .join(rel);
    p.is_file().then_some(p)
}

#[test]
fn test_dropped_fixture_catb_is_telemetry() {
    for rel in [
        "hevc_bslice/hevc_bslice.catb",
        "hevc_intra/hevc_intra.catb",
        "hevc_inter/hevc_inter.catb",
        "hevc_multi_idr/hevc_multi_idr.catb",
        "avc_mb/avc_cavlc.catb",
    ] {
        let Some(p) = fixture(rel) else { continue };
        assert_eq!(classify_dropped_file(&p), DroppedKind::Telemetry, "{rel}");
    }
}

#[test]
fn test_dropped_fixture_h265_h264_are_bitstream() {
    for rel in [
        "hevc_bslice/hevc_bslice.h265",
        "hevc_intra/hevc_intra.h265",
        "hevc_inter/hevc_inter.h265",
        "hevc_multi_idr/hevc_multi_idr.h265",
        "avc_mb/avc_cavlc.h264",
    ] {
        let Some(p) = fixture(rel) else { continue };
        assert_eq!(classify_dropped_file(&p), DroppedKind::Bitstream, "{rel}");
    }
}

#[test]
fn test_dropped_fixture_yuv_and_json_are_other() {
    for rel in [
        "hevc_bslice/hevc_bslice_64x64_i420.yuv",
        "hevc_bslice/hevc_bslice_oracle_frames.json",
        "avc_mb/avc_cavlc_oracle_blocks.json",
    ] {
        let Some(p) = fixture(rel) else { continue };
        assert_eq!(classify_dropped_file(&p), DroppedKind::Other, "{rel}");
    }
}

#[test]
fn test_dropped_fixture_bitstream_bytes_sniffed_without_extension() {
    // Copy a real .h265 to an extensionless name and to .bin: the Annex-B
    // start-code sniff must still classify it as a bitstream.
    let Some(src) = fixture("hevc_bslice/hevc_bslice.h265") else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    for name in ["renamed_stream", "renamed_stream.bin"] {
        let dst = dir.path().join(name);
        std::fs::copy(&src, &dst).unwrap();
        assert_eq!(classify_dropped_file(&dst), DroppedKind::Bitstream, "{name}");
    }
    // Same bytes under a known video extension: never sniffed → Other.
    let dst = dir.path().join("renamed_stream.yuv");
    std::fs::copy(&src, &dst).unwrap();
    assert_eq!(classify_dropped_file(&dst), DroppedKind::Other);
}

#[test]
fn test_dropped_fixture_catb_magic_beats_extension() {
    // A real .catb renamed to .yuv must still classify as telemetry
    // (magic has priority over the extension).
    let Some(src) = fixture("hevc_bslice/hevc_bslice.catb") else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let dst = dir.path().join("renamed.yuv");
    std::fs::copy(&src, &dst).unwrap();
    assert_eq!(classify_dropped_file(&dst), DroppedKind::Telemetry);
}
