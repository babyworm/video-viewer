//! M-A "Stream Structure" tests: structured meta parsing (parameter sets,
//! slice headers, DPB, exactness) differentially validated against the
//! codec-analyzer CLI oracle dumps (`*_oracle_frames.json`).
//!
//! Implemented from the CC0 .catb v4 format specification
//! (codec-analyzer docs/catb-v4-format.md); field names observed in the
//! fixture data — no GPL code consulted.

use std::path::PathBuf;

use video_viewer::core::catb::CatbFile;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root")
        .join("test_data/bitstream")
}

fn open_fixture(dir: &str, name: &str) -> CatbFile {
    let path = fixture_dir().join(dir).join(format!("{name}.catb"));
    CatbFile::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
}

fn load_oracle_frames(dir: &str, name: &str) -> Vec<serde_json::Value> {
    let path = fixture_dir()
        .join(dir)
        .join(format!("{name}_oracle_frames.json"));
    let data =
        std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let v: serde_json::Value = serde_json::from_slice(&data)
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    v.as_array().expect("oracle frames array").clone()
}

/// Fixtures the M-A differential runs against (task spec: inter + bslice).
const FIXTURES: &[(&str, &str)] = &[("hevc_inter", "hevc_inter"), ("hevc_bslice", "hevc_bslice")];

// ===========================================================================
// Parameter sets
// ===========================================================================

#[test]
fn test_structure_parameter_sets_present() {
    for &(dir, name) in FIXTURES {
        let catb = open_fixture(dir, name);
        let sets = &catb.meta.parameter_set_infos;
        assert!(!sets.is_empty(), "{dir}: parameter_set_infos empty");
        // An SPS must exist and carry the coded dimensions as sorted fields.
        let sps = sets
            .iter()
            .find(|s| s.kind.eq_ignore_ascii_case("sps"))
            .unwrap_or_else(|| panic!("{dir}: no SPS in parameter sets"));
        assert!(!sps.fields.is_empty(), "{dir}: SPS fields empty");
        let field = |n: &str| {
            sps.fields
                .iter()
                .find(|(k, _)| k == n)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(field("width").as_deref(), Some("64"), "{dir}");
        assert_eq!(field("height").as_deref(), Some("64"), "{dir}");
        // Fields are name-sorted (serde_json BTreeMap iteration).
        let names: Vec<&String> = sps.fields.iter().map(|(n, _)| n).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "{dir}: SPS fields not sorted");
        // VPS and PPS are present in the HEVC fixtures too.
        for kind in ["vps", "pps"] {
            assert!(
                sets.iter().any(|s| s.kind.eq_ignore_ascii_case(kind)),
                "{dir}: no {kind}"
            );
        }
    }
}

// ===========================================================================
// DPB + slice headers, differential vs oracle_frames.json
// ===========================================================================

#[test]
fn test_structure_dpb_differential() {
    for &(dir, name) in FIXTURES {
        let catb = open_fixture(dir, name);
        let oracle = load_oracle_frames(dir, name);
        assert_eq!(catb.meta.frames_meta.len(), catb.frames.len(), "{dir}");
        assert_eq!(oracle.len(), catb.frames.len(), "{dir}: oracle frame count");
        for (i, o) in oracle.iter().enumerate() {
            let m = &catb.meta.frames_meta[i];
            let ctx = format!("{dir} decode#{i}");
            // Row count and per-flag counts.
            assert_eq!(
                m.dpb.len() as u64,
                o["dpb_rows"].as_u64().unwrap(),
                "{ctx}: dpb_rows"
            );
            assert_eq!(
                m.dpb.iter().filter(|r| r.used_for_reference).count() as u64,
                o["dpb_reference_count"].as_u64().unwrap(),
                "{ctx}: dpb_reference_count"
            );
            assert_eq!(
                m.dpb.iter().filter(|r| !r.used_for_reference).count() as u64,
                o["dpb_hold_count"].as_u64().unwrap(),
                "{ctx}: dpb_hold_count"
            );
            assert_eq!(
                m.dpb.iter().filter(|r| r.long_term).count() as u64,
                o["dpb_long_term_count"].as_u64().unwrap(),
                "{ctx}: dpb_long_term_count"
            );
            assert_eq!(
                m.dpb.iter().filter(|r| r.output_mark).count() as u64,
                o["dpb_output_mark_count"].as_u64().unwrap(),
                "{ctx}: dpb_output_mark_count"
            );
            // POC list of the reference rows, oracle formatting ("a, b, c").
            let dpb_pocs = m
                .dpb
                .iter()
                .filter(|r| r.used_for_reference)
                .map(|r| r.poc.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            assert_eq!(
                dpb_pocs,
                o["dpb_reference_pocs"].as_str().unwrap(),
                "{ctx}: dpb_reference_pocs"
            );
            // reference_labels = DPB row labels then slice ref labels.
            let mut labels: Vec<String> =
                m.dpb.iter().map(|r| r.label.clone()).collect();
            for sh in &m.slice_headers {
                for e in sh.ref_list0.iter().chain(sh.ref_list1.iter()) {
                    labels.push(e.label.clone());
                }
            }
            assert_eq!(
                labels.join(", "),
                o["reference_labels"].as_str().unwrap(),
                "{ctx}: reference_labels"
            );
        }
    }
}

#[test]
fn test_structure_slice_headers_differential() {
    for &(dir, name) in FIXTURES {
        let catb = open_fixture(dir, name);
        let oracle = load_oracle_frames(dir, name);
        for (i, o) in oracle.iter().enumerate() {
            let m = &catb.meta.frames_meta[i];
            let ctx = format!("{dir} decode#{i}");
            assert!(!m.slice_headers.is_empty(), "{ctx}: no slice headers");
            let l0: usize = m.slice_headers.iter().map(|s| s.ref_list0.len()).sum();
            let l1: usize = m.slice_headers.iter().map(|s| s.ref_list1.len()).sum();
            assert_eq!(
                l0 as u64,
                o["slice_l0_count"].as_u64().unwrap(),
                "{ctx}: slice_l0_count"
            );
            assert_eq!(
                l1 as u64,
                o["slice_l1_count"].as_u64().unwrap(),
                "{ctx}: slice_l1_count"
            );
            assert_eq!(
                (l0 + l1) as u64,
                o["slice_reference_rows"].as_u64().unwrap(),
                "{ctx}: slice_reference_rows"
            );
            // Slice ref POCs in list order (L0 then L1 per slice).
            let pocs = m
                .slice_headers
                .iter()
                .flat_map(|s| s.ref_list0.iter().chain(s.ref_list1.iter()))
                .map(|e| e.poc.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            assert_eq!(
                pocs,
                o["slice_reference_pocs"].as_str().unwrap(),
                "{ctx}: slice_reference_pocs"
            );
            // Every slice carries type/name/fields (observed fixture shape).
            for sh in &m.slice_headers {
                assert!(!sh.slice_type_label.is_empty(), "{ctx}: slice_type_label");
                assert!(!sh.nal_unit_name.is_empty(), "{ctx}: nal_unit_name");
                assert!(!sh.fields.is_empty(), "{ctx}: fields empty");
                assert!(
                    sh.fields.iter().any(|(n, _)| n == "slice_type"),
                    "{ctx}: slice_type field missing"
                );
                assert!(!sh.syntax.is_empty(), "{ctx}: decoded syntax empty");
            }
        }
    }
}

// ===========================================================================
// Exactness arrays (parallel to the frame's BLOCK records)
// ===========================================================================

#[test]
fn test_structure_exactness_parallel_to_blocks() {
    for &(dir, name) in FIXTURES {
        let catb = open_fixture(dir, name);
        for (i, frame) in catb.frames.iter().enumerate() {
            let m = &catb.meta.frames_meta[i];
            assert_eq!(
                m.exactness_missing.len(),
                frame.block_n.max(0) as usize,
                "{dir} decode#{i}: exactness_missing not parallel to blocks"
            );
            assert_eq!(
                m.block_dropped_rows.len(),
                frame.block_n.max(0) as usize,
                "{dir} decode#{i}: block_dropped_rows not parallel to blocks"
            );
        }
    }
}

// ===========================================================================
// Decoder identity exposure (empty values allowed)
// ===========================================================================

#[test]
fn test_structure_capture_level_and_contract_tolerate_absence() {
    // The fixtures record neither capture_level nor contract — both must
    // surface as empty strings (rendered "n/a"), never an error.
    for &(dir, name) in FIXTURES {
        let catb = open_fixture(dir, name);
        assert_eq!(catb.meta.capture_level, "", "{dir}");
        assert_eq!(catb.meta.contract, "", "{dir}");
        assert_eq!(catb.meta.codec, "HEVC", "{dir}");
    }
}
