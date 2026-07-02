//! Shared synthetic `.catb` v4 fixture writer (spec §2–§13, byte-for-byte
//! hand assembly) — used by `catb_test.rs` and `correlation_test.rs`.
//!
//! Implemented from the CC0 .catb v4 format specification
//! (codec-analyzer docs/catb-v4-format.md); no GPL code consulted.
#![allow(dead_code)] // each test binary uses a different helper subset

use std::io::Write;

use video_viewer::core::catb::{BLOCK_RECORD_SIZE, FRAME_RECORD_SIZE, REF_RECORD_SIZE};

/// Encode the string table (§3): u32 count + count × (u32 len, utf8 bytes).
pub fn encode_strings(strings: &[&str]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(strings.len() as u32).to_le_bytes());
    for s in strings {
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    out
}

/// Encode one 80-byte FRAME record (§5).
#[allow(clippy::too_many_arguments)]
pub fn frame_record(
    index: i32,
    poc: i32,
    frame_type_id: i32,
    output: i32,
    vps_id: i32,
    sps_id: i32,
    pps_id: i32,
    slice_bits: i64,
    header_bits: i64,
    data_bits: i64,
    block_off: u64,
    block_n: i32,
    aux_off: u64,
    aux_n: i64,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(FRAME_RECORD_SIZE);
    for v in [index, poc, frame_type_id, output, vps_id, sps_id, pps_id] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in [slice_bits, header_bits, data_bits] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&block_off.to_le_bytes());
    out.extend_from_slice(&block_n.to_le_bytes());
    out.extend_from_slice(&aux_off.to_le_bytes());
    out.extend_from_slice(&aux_n.to_le_bytes());
    assert_eq!(out.len(), FRAME_RECORD_SIZE);
    out
}

/// Named parameters for a synthetic BLOCK record (§6). Defaults are all-zero.
#[derive(Default, Clone)]
pub struct BlockSpec {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub ctu_address: i64,
    pub qp: i32,
    pub partition_id: i32,
    pub prediction_mode_id: i32,
    pub bits: i64,
    pub bit_offset: i64,
    pub exactness_flags: i32,
    pub syntax_off: u64,
    pub syntax_n: i32,
    pub cabac_off: u64,
    pub cabac_n: i32,
    pub tx_off: u64,
    pub tx_n: i32,
    pub ref_off: u64,
    pub ref_n: i32,
    pub mv_flags: u32,
    pub mvp: (i32, i32),
    pub mvd: (i32, i32),
    pub mv: (i32, i32),
    pub reference_id: i32,
    pub reference_list_id: i32,
    pub reference_label_id: i32,
    pub reference_list_index: i32,
    pub reference_poc: i32,
    pub reference_frame: i32,
}

/// Encode one 156-byte BLOCK record (§6).
pub fn block_record(b: &BlockSpec) -> Vec<u8> {
    let mut out = Vec::with_capacity(BLOCK_RECORD_SIZE);
    for v in [b.x, b.y, b.w, b.h] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&b.ctu_address.to_le_bytes());
    for v in [b.qp, b.partition_id, b.prediction_mode_id] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&b.bits.to_le_bytes());
    out.extend_from_slice(&b.bit_offset.to_le_bytes());
    out.extend_from_slice(&b.exactness_flags.to_le_bytes());
    out.extend_from_slice(&b.syntax_off.to_le_bytes());
    out.extend_from_slice(&b.syntax_n.to_le_bytes());
    out.extend_from_slice(&b.cabac_off.to_le_bytes());
    out.extend_from_slice(&b.cabac_n.to_le_bytes());
    out.extend_from_slice(&b.tx_off.to_le_bytes());
    out.extend_from_slice(&b.tx_n.to_le_bytes());
    out.extend_from_slice(&b.ref_off.to_le_bytes());
    out.extend_from_slice(&b.ref_n.to_le_bytes());
    out.extend_from_slice(&b.mv_flags.to_le_bytes());
    for v in [b.mvp.0, b.mvp.1, b.mvd.0, b.mvd.1, b.mv.0, b.mv.1] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in [
        b.reference_id,
        b.reference_list_id,
        b.reference_label_id,
        b.reference_list_index,
        b.reference_poc,
        b.reference_frame,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(out.len(), BLOCK_RECORD_SIZE);
    out
}

/// Named parameters for a synthetic REF record (§10).
#[derive(Default, Clone)]
pub struct RefSpec {
    pub list_id: i32,
    pub list_index: i32,
    pub reference_poc: i32,
    pub label_id: i32,
    pub pu: (i32, i32, i32, i32),
    pub ref_flags: u32,
    pub mvp: (i32, i32),
    pub mvd: (i32, i32),
    pub mv: (i32, i32),
    pub reference_frame: i32,
    pub pu_part_index: i32,
}

/// Encode one 68-byte REF record (§10).
pub fn ref_record(r: &RefSpec) -> Vec<u8> {
    let mut out = Vec::with_capacity(REF_RECORD_SIZE);
    for v in [r.list_id, r.list_index, r.reference_poc, r.label_id] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in [r.pu.0, r.pu.1, r.pu.2, r.pu.3] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&r.ref_flags.to_le_bytes());
    for v in [r.mvp.0, r.mvp.1, r.mvd.0, r.mvd.1, r.mv.0, r.mv.1] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&r.reference_frame.to_le_bytes());
    out.extend_from_slice(&r.pu_part_index.to_le_bytes());
    assert_eq!(out.len(), REF_RECORD_SIZE);
    out
}

/// Assemble a full `.catb` v4 file: 176-byte header + contiguous sections in
/// directory order starting at offset 176 (§2). Empty sections get the
/// running offset with size 0, like the reference writer.
pub fn build_catb(
    frame_count: u32,
    strings: &[u8],
    meta: &[u8],
    frames: &[u8],
    blocks: &[u8],
    refs: &[u8],
    frame_aux: &[u8],
) -> Vec<u8> {
    let sections: [&[u8]; 10] = [
        strings,
        meta,
        frames,
        blocks,
        &[], // syntax
        &[], // cabac
        &[], // transforms
        refs,
        &[], // coeffs
        frame_aux,
    ];
    let mut out = Vec::new();
    out.extend_from_slice(b"CATB0001");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&frame_count.to_le_bytes());
    let mut offset = 176u64;
    for sec in &sections {
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&(sec.len() as u64).to_le_bytes());
        offset += sec.len() as u64;
    }
    assert_eq!(out.len(), 176);
    for sec in &sections {
        out.extend_from_slice(sec);
    }
    out
}

pub fn write_temp(bytes: &[u8]) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().expect("temp file");
    f.write_all(bytes).expect("write catb bytes");
    f.flush().expect("flush");
    f
}
