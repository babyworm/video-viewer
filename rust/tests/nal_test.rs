//! M-E: Annex-B NAL scanner tests — synthetic streams, real fixture
//! spot-checks (offsets/types independently verified with hexdump), codec
//! detection, source inference, and the BLOCK `bit_offset` semantics
//! verdict (frame-local, not stream-absolute).
//!
//! Implemented from public H.264/H.265 standards knowledge and the CC0
//! .catb v4 spec; no GPL code consulted.

use std::path::{Path, PathBuf};

use video_viewer::core::catb::CatbFile;
use video_viewer::core::nal::{
    codec_hint_from_ext, detect_codec, infer_bitstream_source, nal_type_name, scan_annexb,
    NalCodec, NalScan,
};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../test_data/bitstream")
        .join(name)
}

/// One HEVC unit: start code + 2-byte header (+ payload byte with the
/// first_slice flag in bit 7 when given).
fn hevc_unit(four_byte: bool, nal_type: u8, tid_plus1: u8, payload: &[u8]) -> Vec<u8> {
    let mut v: Vec<u8> = if four_byte {
        vec![0, 0, 0, 1]
    } else {
        vec![0, 0, 1]
    };
    v.push(nal_type << 1);
    v.push(tid_plus1);
    v.extend_from_slice(payload);
    v
}

// ===========================================================================
// Synthetic streams
// ===========================================================================

#[test]
fn test_nal_synthetic_hevc_mixed_start_codes() {
    // VPS (4-byte code), SPS (4-byte), IDR slice (3-byte, fsf=1),
    // TRAIL_R (4-byte, fsf=1).
    let mut data = Vec::new();
    data.extend(hevc_unit(true, 32, 1, &[0xAA, 0xBB]));
    data.extend(hevc_unit(true, 33, 1, &[0xCC]));
    data.extend(hevc_unit(false, 20, 1, &[0x80, 0x11, 0x22]));
    data.extend(hevc_unit(true, 1, 1, &[0x80, 0x33]));
    let (units, truncated) = scan_annexb(&data, NalCodec::Hevc);
    assert!(!truncated);
    assert_eq!(units.len(), 4);

    assert_eq!(units[0].offset, 0);
    assert_eq!(units[0].start_code_len, 4);
    assert_eq!(units[0].nal_type, 32);
    assert_eq!(units[0].size, 4); // header(2) + payload(2)
    assert!(!units[0].is_vcl);
    assert_eq!(nal_type_name(NalCodec::Hevc, 32), "VPS_NUT");

    assert_eq!(units[1].offset, 8);
    assert_eq!(units[1].nal_type, 33);

    // 3-byte start code unit.
    assert_eq!(units[2].start_code_len, 3);
    assert_eq!(units[2].nal_type, 20);
    assert!(units[2].is_vcl);
    assert!(units[2].au_start, "fsf bit set → AU start");
    assert_eq!(units[2].au_id, 0);
    assert_eq!(nal_type_name(NalCodec::Hevc, 20), "IDR_N_LP");

    assert_eq!(units[3].nal_type, 1);
    assert!(units[3].au_start);
    assert_eq!(units[3].au_id, 1, "second slice with fsf=1 starts AU 1");
    // Parameter sets before the first VCL belong to AU 0.
    assert_eq!(units[0].au_id, 0);
    assert_eq!(units[1].au_id, 0);
}

#[test]
fn test_nal_synthetic_consecutive_units_no_gap() {
    // Back-to-back 3-byte codes with 2-byte payloads (header only).
    let mut data = Vec::new();
    data.extend(hevc_unit(false, 32, 1, &[]));
    data.extend(hevc_unit(false, 34, 1, &[]));
    let (units, _) = scan_annexb(&data, NalCodec::Hevc);
    assert_eq!(units.len(), 2);
    assert_eq!(units[0].size, 2);
    assert_eq!(units[1].offset, 5);
}

#[test]
fn test_nal_synthetic_avc_header_fields() {
    // SPS (ref_idc 3), IDR slice (ref_idc 3, first_mb_in_slice=0 → leading
    // '1' bit in the slice header byte).
    let data = [
        0u8, 0, 0, 1, 0x67, 0x64, 0x00, 0x0A, // SPS
        0, 0, 1, 0x65, 0x88, 0x84, // IDR, first byte 0x88 → bit7 = 1
        0, 0, 1, 0x41, 0x9A, // non-IDR, ref_idc 2
    ];
    let (units, _) = scan_annexb(&data, NalCodec::Avc);
    assert_eq!(units.len(), 3);
    assert_eq!(units[0].nal_type, 7);
    assert_eq!(units[0].nal_ref_idc, 3);
    assert!(!units[0].is_vcl);
    assert_eq!(nal_type_name(NalCodec::Avc, 7), "SPS");
    assert_eq!(units[1].nal_type, 5);
    assert!(units[1].is_vcl);
    assert!(units[1].au_start);
    assert_eq!(units[1].au_id, 0);
    assert_eq!(units[2].nal_type, 1);
    assert_eq!(units[2].nal_ref_idc, 2);
    assert!(units[2].au_start);
    assert_eq!(units[2].au_id, 1);
}

#[test]
fn test_nal_garbage_and_truncated_no_panic() {
    // Pure garbage: no start codes at all.
    let garbage: Vec<u8> = (0u16..4096).map(|i| (i % 251) as u8 | 0x02).collect();
    let (units, _) = scan_annexb(&garbage, NalCodec::Hevc);
    assert!(units.is_empty());

    // All zeros: no 01 terminator anywhere.
    let zeros = vec![0u8; 1024];
    let (units, _) = scan_annexb(&zeros, NalCodec::Avc);
    assert!(units.is_empty());

    // Start code at EOF (no header bytes) — unit with zero-size header.
    let tail = [0u8, 0, 1];
    let (units, _) = scan_annexb(&tail, NalCodec::Hevc);
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].size, 0);
    assert_eq!(units[0].nal_type, 0);
    assert!(!units[0].au_start);

    // 1-byte HEVC payload (truncated 2-byte header) — no panic.
    let short = [0u8, 0, 1, 0x40];
    let (units, _) = scan_annexb(&short, NalCodec::Hevc);
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].size, 1);

    // Empty input.
    let (units, _) = scan_annexb(&[], NalCodec::Hevc);
    assert!(units.is_empty());
}

#[test]
fn test_nal_long_zero_run_start_code() {
    // 00 00 00 00 01: extra leading zeros — the code is the last 4 bytes.
    let mut data = vec![0u8, 0, 0, 0, 1];
    data.extend_from_slice(&[0x40, 0x01, 0xFF]); // VPS
    let (units, _) = scan_annexb(&data, NalCodec::Hevc);
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].offset, 1);
    assert_eq!(units[0].start_code_len, 4);
    assert_eq!(units[0].nal_type, 32);
}

// ===========================================================================
// Codec detection
// ===========================================================================

#[test]
fn test_nal_detect_codec_by_headers() {
    let hevc = std::fs::read(fixture("hevc_bslice/hevc_bslice.h265")).unwrap();
    let avc = std::fs::read(fixture("avc_mb/avc_cavlc.h264")).unwrap();
    // No extension hint: header scoring alone must decide.
    assert_eq!(detect_codec(&hevc, None), NalCodec::Hevc);
    assert_eq!(detect_codec(&avc, None), NalCodec::Avc);
    // A wrong hint must not override a clear score.
    assert_eq!(detect_codec(&hevc, Some(NalCodec::Avc)), NalCodec::Hevc);
    assert_eq!(detect_codec(&avc, Some(NalCodec::Hevc)), NalCodec::Avc);
}

#[test]
fn test_nal_codec_hint_from_ext() {
    assert_eq!(
        codec_hint_from_ext(Path::new("a/b.h265")),
        Some(NalCodec::Hevc)
    );
    assert_eq!(
        codec_hint_from_ext(Path::new("b.HEVC")),
        Some(NalCodec::Hevc)
    );
    assert_eq!(codec_hint_from_ext(Path::new("c.264")), Some(NalCodec::Avc));
    assert_eq!(codec_hint_from_ext(Path::new("d.bin")), None);
    assert_eq!(codec_hint_from_ext(Path::new("noext")), None);
}

// ===========================================================================
// Real fixtures (counts/types/offsets independently verified via hexdump)
// ===========================================================================

#[test]
fn test_nal_fixture_hevc_inter() {
    let scan = NalScan::open(fixture("hevc_inter/hevc_inter.h265")).unwrap();
    assert_eq!(scan.codec, NalCodec::Hevc);
    let types: Vec<u8> = scan.units.iter().map(|u| u.nal_type).collect();
    assert_eq!(types, [32, 33, 34, 20, 1, 1], "VPS SPS PPS IDR TRAIL TRAIL");
    let offsets: Vec<u64> = scan.units.iter().map(|u| u.offset).collect();
    assert_eq!(offsets, [0x0, 0x1c, 0x47, 0x52, 0x77, 0x91]);
    assert_eq!(scan.vcl_count(), 3);
    // Every VCL unit carries fsf=1 (single-slice frames) → AUs 0, 1, 2.
    let vcl_aus: Vec<u32> = scan
        .units
        .iter()
        .filter(|u| u.is_vcl)
        .map(|u| u.au_id)
        .collect();
    assert_eq!(vcl_aus, [0, 1, 2]);
    assert!(scan.units.iter().all(|u| !u.forbidden));
    assert!(scan.units.iter().all(|u| u.temporal_id == 0));
}

#[test]
fn test_nal_fixture_hevc_intra_repeated_headers() {
    let scan = NalScan::open(fixture("hevc_intra/hevc_intra.h265")).unwrap();
    let types: Vec<u8> = scan.units.iter().map(|u| u.nal_type).collect();
    // VPS SPS PPS SEI IDR ×2 (parameter sets repeated per IDR).
    assert_eq!(types, [32, 33, 34, 39, 20, 32, 33, 34, 39, 20]);
    assert_eq!(scan.vcl_count(), 2);
    // The repeated non-VCL run is attributed to the *next* AU.
    assert_eq!(scan.units[5].au_id, 1, "second VPS belongs to AU 1");
    assert_eq!(scan.units[9].au_id, 1);
}

#[test]
fn test_nal_fixture_hevc_bslice_and_multi_idr() {
    let bslice = NalScan::open(fixture("hevc_bslice/hevc_bslice.h265")).unwrap();
    assert_eq!(bslice.units.len(), 7);
    assert_eq!(bslice.vcl_count(), 4);
    assert_eq!(bslice.units[3].nal_type, 20);
    assert_eq!(bslice.units[3].offset, 0x50);
    assert_eq!(bslice.units[3].start_code_len, 3);
    assert_eq!(bslice.units[3].size, 875);

    let multi = NalScan::open(fixture("hevc_multi_idr/hevc_multi_idr.h265")).unwrap();
    assert_eq!(multi.units.len(), 12);
    assert_eq!(multi.vcl_count(), 6);
    // Second GOP starts at the second VPS (offset 0xa4, hexdump-verified).
    assert_eq!(multi.units[6].nal_type, 32);
    assert_eq!(multi.units[6].offset, 0xa4);
}

#[test]
fn test_nal_fixture_avc_cavlc() {
    let scan = NalScan::open(fixture("avc_mb/avc_cavlc.h264")).unwrap();
    assert_eq!(scan.codec, NalCodec::Avc);
    let types: Vec<u8> = scan.units.iter().map(|u| u.nal_type).collect();
    assert_eq!(types, [7, 8, 6, 5, 7, 8, 5], "SPS PPS SEI IDR SPS PPS IDR");
    assert_eq!(scan.vcl_count(), 2);
    assert_eq!(scan.units[3].offset, 0x256);
    assert_eq!(scan.units[3].nal_ref_idc, 3);
    // unit_bytes includes the start code and stops at the next unit.
    let bytes = scan.unit_bytes(0);
    assert_eq!(bytes.len(), 4 + 20);
    assert_eq!(&bytes[..5], &[0, 0, 0, 1, 0x67]);
}

#[test]
fn test_nal_open_rejects_non_annexb() {
    let f = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(f.path(), b"not a bitstream at all").unwrap();
    assert!(NalScan::open(f.path()).is_err());
}

// ===========================================================================
// Source inference (catb → original bitstream)
// ===========================================================================

#[test]
fn test_nal_infer_source_from_workdir() {
    // decoder_run convention: <dir>/foo.h265 + <dir>/foo.h265.catb-run/x.catb
    let dir = tempfile::tempdir().unwrap();
    let stream = dir.path().join("foo.h265");
    std::fs::write(&stream, [0u8, 0, 0, 1, 0x40, 0x01]).unwrap();
    let workdir = dir.path().join("foo.h265.catb-run");
    std::fs::create_dir(&workdir).unwrap();
    let catb = workdir.join("telemetry.catb");
    std::fs::write(&catb, b"x").unwrap();
    assert_eq!(infer_bitstream_source(&catb), Some(stream));
}

#[test]
fn test_nal_infer_source_sibling_and_none() {
    // Same-stem sibling: bar.catb next to bar.h264.
    let dir = tempfile::tempdir().unwrap();
    let catb = dir.path().join("bar.catb");
    std::fs::write(&catb, b"x").unwrap();
    assert_eq!(infer_bitstream_source(&catb), None, "no sibling yet");
    let stream = dir.path().join("bar.h264");
    std::fs::write(&stream, b"y").unwrap();
    assert_eq!(infer_bitstream_source(&catb), Some(stream));
}

#[test]
fn test_nal_infer_source_real_fixture_layout() {
    // The repo fixtures sit as <name>/<name>.catb + <name>/<name>.h265.
    let catb = fixture("hevc_inter/hevc_inter.catb");
    assert_eq!(
        infer_bitstream_source(&catb),
        Some(fixture("hevc_inter/hevc_inter.h265"))
    );
}

// ===========================================================================
// bit_offset semantics (M-E §C verdict)
// ===========================================================================

/// BLOCK `bit_offset` is a **frame-local decode-domain** position, not a
/// stream-absolute one: oracle `bit_range` shows frame 1's first block at
/// bit 31 while the frame's slice data begins at byte 958 (bit 7664) of the
/// stream — hence the HEX view draws no bit_offset-derived highlight.
#[test]
fn test_catb_bit_offset_is_frame_local() {
    let catb = CatbFile::open(fixture("hevc_bslice/hevc_bslice.catb")).unwrap();
    // Oracle (hevc_bslice_oracle_blocks.json): frame 0 block 0 bit_range
    // "28-124"; frame 1 block 0 bit_range "31-95" with
    // frame_source_byte_range "958-1392".
    let f0 = catb.blocks_for_frame(0).unwrap();
    assert_eq!(f0[0].bit_offset, 28);
    assert_eq!(f0[0].bits, 96);
    let f1 = catb.blocks_for_frame(1).unwrap();
    assert_eq!(f1[0].bit_offset, 31);
    // Frame 1 starts at stream byte 958; a stream-absolute offset would be
    // ≥ 958·8 = 7664 bits. 31 ≪ 7664 ⇒ frame-local.
    assert!(f1[0].bit_offset < 958 * 8);
    // Monotone within the frame (decode order).
    for w in f1.windows(2) {
        assert!(w[1].bit_offset >= w[0].bit_offset);
    }
}
