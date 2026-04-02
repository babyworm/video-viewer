//! Sideband binary parser for ISP parameter overlay.
//!
//! Schema-driven: binary layout is defined in `sideband_schema.toml`.
//! Shared across video-viewer, isp_emulator, and isp_hevc.

use std::fmt;

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Public data structs (API — unchanged)
// ---------------------------------------------------------------------------

/// A complete sideband file containing multiple frames.
#[derive(Debug, Clone)]
pub struct SidebandFile {
    pub frames: Vec<SidebandFrame>,
}

/// One frame's sideband data.
#[derive(Debug, Clone)]
pub struct SidebandFrame {
    pub frame_id: i32,
    pub valid_fields_mask: u16,
    pub scene_class: u8,
    pub noise_class: u8,
    pub motion_class: u8,
    pub denoise_strength: u8,
    pub sharpen_strength: u8,
    pub scene_flags: u8,
    pub frame_qp_bias: i8,
    pub frame_chroma_cb_bias: i8,
    pub frame_chroma_cr_bias: i8,
    pub frame_lambda_scale_q8: u16,
    pub global_confidence: u8,
    // v1 fields
    pub iso_class: u8,
    pub ae_state: u8,
    pub histogram_shape: u8,
    pub dynamic_range_q8: u8,
    pub ca_severity_q8: u8,
    pub distortion_k1_q8: i8,
    pub scene_change_score: u8,
    pub ctus: Vec<SidebandCtu>,
}

/// One CTU's sideband parameters.
#[derive(Debug, Clone, Default)]
pub struct SidebandCtu {
    pub activity: u8,
    pub flatness: u8,
    pub edge_density: u8,
    pub noise: u8,
    pub saliency: u8,
    pub chroma_importance: u8,
    pub confidence: u8,
    pub ctu_flags: u8,
    pub qp_delta: i8,
    pub chroma_cb_delta: i8,
    pub chroma_cr_delta: i8,
    pub lambda_scale_q8: u16,
    pub rc_importance_weight: u8,
    pub sao_prior: i8,
    pub temporal_stability: u8,
    // v1 fields
    pub noise_sigma_q8: u8,
    pub noise_confidence: u8,
    pub clip_risk: u8,
    pub structure_class: u8,
    pub dof_sharpness: u8,
    pub vignetting_gain_q8: u8,
    pub denoise_confidence: u8,
}

/// Which CTU field to visualize as a heatmap overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidebandOverlayMode {
    None,
    QpDelta,
    Activity,
    Flatness,
    Saliency,
    EdgeDensity,
    Noise,
    Confidence,
    TemporalStability,
    // v1 modes
    NoiseSigma,
    NoiseConfidence,
    ClipRisk,
    StructureClass,
    DofSharpness,
    VignettingGain,
    DenoiseConfidence,
}

// ---------------------------------------------------------------------------
// Schema types (deserialized from TOML)
// ---------------------------------------------------------------------------

/// Embedded default schema. Override at runtime via `SidebandFile::from_bytes_with_schema_str`.
const DEFAULT_SCHEMA: &str = include_str!("../../sideband_schema.toml");

#[derive(Debug, Deserialize)]
struct SchemaRoot {
    format: SchemaFormat,
    header: SchemaHeader,
    frame_params: SchemaSection,
    ctu_params: SchemaSection,
}

#[derive(Debug, Deserialize)]
struct SchemaFormat {
    magic: String,
    version: u8,
    #[serde(default)]
    endian: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SchemaHeader {
    short_size: usize,
    extended_size: usize,
    extended_marker: u8,
}

#[derive(Debug, Deserialize)]
struct SchemaSection {
    size: usize,
    fields: Vec<SchemaField>,
}

#[derive(Debug, Deserialize)]
struct SchemaField {
    name: String,
    offset: usize,
    #[serde(rename = "type")]
    field_type: String,
    #[serde(default)]
    shift: u8,
    #[serde(default)]
    #[allow(dead_code)]
    scale: u16,
    #[serde(default)]
    min_version: u8,
}

// ---------------------------------------------------------------------------
// Schema loading
// ---------------------------------------------------------------------------

fn validate_schema(schema: &SchemaRoot) -> Result<(), String> {
    if let Some(ref endian) = schema.format.endian {
        if endian != "big" {
            return Err(format!("unsupported endian '{}': only \"big\" is supported", endian));
        }
    }
    Ok(())
}

fn load_default_schema() -> Result<SchemaRoot, String> {
    let schema: SchemaRoot = toml::from_str(DEFAULT_SCHEMA)
        .map_err(|e| format!("failed to parse embedded sideband schema: {}", e))?;
    validate_schema(&schema)?;
    Ok(schema)
}

fn load_schema_from_str(toml_str: &str) -> Result<SchemaRoot, String> {
    let schema: SchemaRoot = toml::from_str(toml_str)
        .map_err(|e| format!("failed to parse sideband schema: {}", e))?;
    validate_schema(&schema)?;
    Ok(schema)
}

// ---------------------------------------------------------------------------
// Generic field extraction from binary buffer
// ---------------------------------------------------------------------------

/// Compute the effective byte size of a section for a given binary version.
///
/// For the highest schema version, returns the declared `section.size`.
/// For older versions, returns the start offset of the first field belonging
/// to a newer version (which is where the older version's data ends, including
/// any reserved/padding bytes between the last real field and the extension area).
fn effective_section_size(section: &SchemaSection, version: u8) -> usize {
    // Check if any field exceeds this version
    let has_newer = section.fields.iter().any(|f| f.min_version > version);
    if !has_newer {
        // This is the newest (or only) version — use declared size
        return section.size;
    }
    // Effective size = start offset of first field from a newer version
    section
        .fields
        .iter()
        .filter(|f| f.min_version > version)
        .map(|f| f.offset)
        .min()
        .unwrap_or(section.size)
}

/// Extract a single field value as i64 from a binary buffer according to schema.
fn extract_field(buf: &[u8], field: &SchemaField) -> i64 {
    match field.field_type.as_str() {
        "u8" => (buf[field.offset] >> field.shift) as i64,
        "i8" => ((buf[field.offset] >> field.shift) as i8) as i64,
        "u16" => {
            let v = ((buf[field.offset] as u16) << 8) | (buf[field.offset + 1] as u16);
            (v >> field.shift) as i64
        }
        "i32" => {
            let v = ((buf[field.offset] as i32) << 24)
                | ((buf[field.offset + 1] as i32) << 16)
                | ((buf[field.offset + 2] as i32) << 8)
                | (buf[field.offset + 3] as i32);
            v as i64
        }
        other => panic!("unknown field type '{}' in schema", other),
    }
}

/// Extract all fields from a section into a name→value map.
/// Only extracts fields whose `min_version` <= the binary version
/// and whose offset fits within the actual buffer.
fn extract_section<'a>(buf: &[u8], section: &'a SchemaSection, version: u8) -> Vec<(&'a str, i64)> {
    section
        .fields
        .iter()
        .filter(|f| {
            f.min_version <= version && f.offset < buf.len()
        })
        .map(|f| (f.name.as_str(), extract_field(buf, f)))
        .collect()
}

// ---------------------------------------------------------------------------
// Struct mapping (schema field name → Rust struct field)
// ---------------------------------------------------------------------------

fn map_frame_fields(values: &[(&str, i64)]) -> SidebandFrame {
    let mut frame = SidebandFrame {
        frame_id: 0,
        valid_fields_mask: 0,
        scene_class: 0,
        noise_class: 0,
        motion_class: 0,
        denoise_strength: 0,
        sharpen_strength: 0,
        scene_flags: 0,
        frame_qp_bias: 0,
        frame_chroma_cb_bias: 0,
        frame_chroma_cr_bias: 0,
        frame_lambda_scale_q8: 0,
        global_confidence: 0,
        // v1 fields default to 0/neutral
        iso_class: 0,
        ae_state: 0,
        histogram_shape: 0,
        dynamic_range_q8: 0,
        ca_severity_q8: 0,
        distortion_k1_q8: 0,
        scene_change_score: 0,
        ctus: Vec::new(),
    };
    for &(name, val) in values {
        match name {
            "frame_id" => frame.frame_id = val as i32,
            "valid_fields_mask" => frame.valid_fields_mask = val as u16,
            "scene_class" => frame.scene_class = val as u8,
            "noise_class" => frame.noise_class = val as u8,
            "motion_class" => frame.motion_class = val as u8,
            "denoise_strength" => frame.denoise_strength = val as u8,
            "sharpen_strength" => frame.sharpen_strength = val as u8,
            "scene_flags" => frame.scene_flags = val as u8,
            "frame_qp_bias" => frame.frame_qp_bias = val as i8,
            "frame_chroma_cb_bias" => frame.frame_chroma_cb_bias = val as i8,
            "frame_chroma_cr_bias" => frame.frame_chroma_cr_bias = val as i8,
            "frame_lambda_scale_q8" => frame.frame_lambda_scale_q8 = val as u16,
            "global_confidence" => frame.global_confidence = val as u8,
            // v1 fields
            "iso_class" => frame.iso_class = val as u8,
            "ae_state" => frame.ae_state = val as u8,
            "histogram_shape" => frame.histogram_shape = val as u8,
            "dynamic_range_q8" => frame.dynamic_range_q8 = val as u8,
            "ca_severity_q8" => frame.ca_severity_q8 = val as u8,
            "distortion_k1_q8" => frame.distortion_k1_q8 = val as i8,
            "scene_change_score" => frame.scene_change_score = val as u8,
            _ => {} // ignore unknown/reserved fields (forward compatibility)
        }
    }
    frame
}

fn map_ctu_fields(values: &[(&str, i64)]) -> SidebandCtu {
    let mut ctu = SidebandCtu::default();
    for &(name, val) in values {
        match name {
            "activity" => ctu.activity = val as u8,
            "flatness" => ctu.flatness = val as u8,
            "edge_density" => ctu.edge_density = val as u8,
            "noise" => ctu.noise = val as u8,
            "saliency" => ctu.saliency = val as u8,
            "chroma_importance" => ctu.chroma_importance = val as u8,
            "confidence" => ctu.confidence = val as u8,
            "ctu_flags" => ctu.ctu_flags = val as u8,
            "qp_delta" => ctu.qp_delta = val as i8,
            "chroma_cb_delta" => ctu.chroma_cb_delta = val as i8,
            "chroma_cr_delta" => ctu.chroma_cr_delta = val as i8,
            "lambda_scale_q8" => ctu.lambda_scale_q8 = val as u16,
            "rc_importance_weight" => ctu.rc_importance_weight = val as u8,
            "sao_prior" => ctu.sao_prior = val as i8,
            "temporal_stability" => ctu.temporal_stability = val as u8,
            // v1 fields
            "noise_sigma_q8" => ctu.noise_sigma_q8 = val as u8,
            "noise_confidence" => ctu.noise_confidence = val as u8,
            "clip_risk" => ctu.clip_risk = val as u8,
            "structure_class" => ctu.structure_class = val as u8,
            "dof_sharpness" => ctu.dof_sharpness = val as u8,
            "vignetting_gain_q8" => ctu.vignetting_gain_q8 = val as u8,
            "denoise_confidence" => ctu.denoise_confidence = val as u8,
            _ => {}
        }
    }
    ctu
}

// ---------------------------------------------------------------------------
// Binary parsing (schema-driven)
// ---------------------------------------------------------------------------

/// Maximum sideband file size (256 MB).
const MAX_SIDEBAND_FILE_SIZE: u64 = 256 * 1024 * 1024;

fn parse_one_frame(
    data: &[u8],
    offset: usize,
    schema: &SchemaRoot,
) -> Result<(SidebandFrame, usize), String> {
    let remaining = data.len() - offset;
    let magic = schema.format.magic.as_bytes();
    let hdr = &schema.header;

    if remaining < hdr.short_size {
        return Err(format!(
            "truncated header at offset {}: need {} bytes, have {}",
            offset, hdr.short_size, remaining
        ));
    }

    // Magic check
    if data[offset..offset + magic.len()] != *magic {
        return Err(format!(
            "invalid magic at offset {}: expected '{}', got [{:#04x}, {:#04x}]",
            offset, schema.format.magic, data[offset], data[offset + 1]
        ));
    }

    // Version check
    let version = data[offset + 2];
    if version > schema.format.version {
        return Err(format!(
            "unsupported sideband version {} at offset {} (max {})",
            version, offset, schema.format.version
        ));
    }

    // CTU count
    let count_byte = data[offset + 3];
    let (num_ctus, header_size) = if count_byte == hdr.extended_marker {
        if remaining < hdr.extended_size {
            return Err(format!("truncated extended header at offset {}", offset));
        }
        let ext = ((data[offset + 4] as usize) << 8) | (data[offset + 5] as usize);
        (ext, hdr.extended_size)
    } else {
        (count_byte as usize, hdr.short_size)
    };

    // Use version-aware sizes: a v0 binary uses smaller frame/CTU buffers
    let fp_size = effective_section_size(&schema.frame_params, version);
    let ctu_size = effective_section_size(&schema.ctu_params, version);
    let needed = header_size + fp_size + ctu_size * num_ctus;
    if remaining < needed {
        return Err(format!(
            "truncated frame at offset {}: need {} bytes ({} CTUs, v{}), have {}",
            offset, needed, num_ctus, version, remaining
        ));
    }

    // Extract frame params
    let fp_start = offset + header_size;
    let fp_buf = &data[fp_start..fp_start + fp_size];
    let fp_values = extract_section(fp_buf, &schema.frame_params, version);
    let mut frame = map_frame_fields(&fp_values);

    // Extract CTU params
    let ctu_base = fp_start + fp_size;
    frame.ctus.reserve(num_ctus);
    for i in 0..num_ctus {
        let co = ctu_base + i * ctu_size;
        let ctu_buf = &data[co..co + ctu_size];
        let ctu_values = extract_section(ctu_buf, &schema.ctu_params, version);
        frame.ctus.push(map_ctu_fields(&ctu_values));
    }

    Ok((frame, needed))
}

// ---------------------------------------------------------------------------
// SidebandFile
// ---------------------------------------------------------------------------

/// Schema file name to look for next to the sideband binary.
const SCHEMA_FILENAME: &str = "sideband_schema.toml";

/// Resolve schema: try loading from same directory as the sideband file,
/// fall back to the embedded default.
fn resolve_schema(sideband_path: &str) -> Result<SchemaRoot, String> {
    if let Some(parent) = std::path::Path::new(sideband_path).parent() {
        let candidate = parent.join(SCHEMA_FILENAME);
        if candidate.is_file() {
            let toml_str = std::fs::read_to_string(&candidate)
                .map_err(|e| format!("failed to read schema '{}': {}", candidate.display(), e))?;
            return load_schema_from_str(&toml_str);
        }
    }
    load_default_schema()
}

impl SidebandFile {
    /// Read a sideband binary file. Looks for `sideband_schema.toml` next to
    /// the file; falls back to the embedded default schema.
    pub fn open(path: &str) -> Result<Self, String> {
        let schema = resolve_schema(path)?;
        Self::open_with_schema(path, &schema)
    }

    /// Read a sideband binary file with a custom schema (parsed SchemaRoot).
    fn open_with_schema(path: &str, schema: &SchemaRoot) -> Result<Self, String> {
        let meta = std::fs::metadata(path)
            .map_err(|e| format!("failed to read sideband file '{}': {}", path, e))?;
        if meta.len() > MAX_SIDEBAND_FILE_SIZE {
            return Err(format!(
                "sideband file too large ({:.1} MB, max {} MB)",
                meta.len() as f64 / (1024.0 * 1024.0),
                MAX_SIDEBAND_FILE_SIZE / (1024 * 1024),
            ));
        }
        let data = std::fs::read(path)
            .map_err(|e| format!("failed to read sideband file '{}': {}", path, e))?;
        Self::from_bytes_with_schema(&data, schema)
    }

    /// Parse sideband data from bytes using the embedded default schema.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let schema = load_default_schema()?;
        Self::from_bytes_with_schema(data, &schema)
    }

    /// Parse sideband data from bytes using a custom schema TOML string.
    pub fn from_bytes_with_schema_str(data: &[u8], schema_toml: &str) -> Result<Self, String> {
        let schema = load_schema_from_str(schema_toml)?;
        Self::from_bytes_with_schema(data, &schema)
    }

    fn from_bytes_with_schema(data: &[u8], schema: &SchemaRoot) -> Result<Self, String> {
        if data.is_empty() {
            return Ok(SidebandFile { frames: Vec::new() });
        }
        let mut frames = Vec::new();
        let mut offset = 0;
        while offset < data.len() {
            let (frame, consumed) = parse_one_frame(data, offset, schema)?;
            frames.push(frame);
            offset += consumed;
        }
        Ok(SidebandFile { frames })
    }

    /// Number of frames in the file.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Get frame by index, returns None if out of range.
    pub fn frame(&self, idx: usize) -> Option<&SidebandFrame> {
        self.frames.get(idx)
    }
}

// ---------------------------------------------------------------------------
// SidebandFrame / SidebandCtu impl
// ---------------------------------------------------------------------------

impl SidebandFrame {
    pub fn num_ctus(&self) -> usize {
        self.ctus.len()
    }

    pub fn lambda_scale(&self) -> f64 {
        self.frame_lambda_scale_q8 as f64 / 256.0
    }
}

impl fmt::Display for SidebandFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Frame(id={}, scene={}, noise={}, motion={}, qp_bias={}, iso={}, ae={}, ctus={})",
            self.frame_id, self.scene_class, self.noise_class,
            self.motion_class, self.frame_qp_bias,
            self.iso_class, self.ae_state, self.ctus.len(),
        )
    }
}

impl SidebandCtu {
    pub fn lambda_scale(&self) -> f64 {
        self.lambda_scale_q8 as f64 / 256.0
    }
}

impl fmt::Display for SidebandCtu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CTU(act={}, flat={}, edge={}, qp_delta={}, lambda={:.3})",
            self.activity, self.flatness, self.edge_density,
            self.qp_delta, self.lambda_scale(),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_loads_successfully() {
        let schema = load_default_schema().unwrap();
        assert_eq!(schema.format.magic, "IP");
        assert_eq!(schema.format.version, 1);
        assert_eq!(schema.frame_params.size, 28);
        assert_eq!(schema.ctu_params.size, 24);
        // v0: 13 frame + v1: 8 (7 real + 1 reserved) = 21 total
        assert_eq!(schema.frame_params.fields.len(), 21);
        // v0: 15 ctu + v1: 8 (7 real + 1 reserved) = 23 total
        assert_eq!(schema.ctu_params.fields.len(), 23);
    }

    #[test]
    fn effective_sizes_match_versions() {
        let schema = load_default_schema().unwrap();
        // v0 effective sizes
        assert_eq!(effective_section_size(&schema.frame_params, 0), 20);
        assert_eq!(effective_section_size(&schema.ctu_params, 0), 16);
        // v1 effective sizes
        assert_eq!(effective_section_size(&schema.frame_params, 1), 28);
        assert_eq!(effective_section_size(&schema.ctu_params, 1), 24);
    }

    #[test]
    fn parse_empty_returns_empty() {
        let sb = SidebandFile::from_bytes(&[]).unwrap();
        assert_eq!(sb.num_frames(), 0);
    }

    #[test]
    fn invalid_magic_returns_error() {
        let data = [b'X', b'Y', 0x00, 0x00];
        let err = SidebandFile::from_bytes(&data).unwrap_err();
        assert!(err.contains("invalid magic"), "got: {}", err);
    }

    #[test]
    fn truncated_data_returns_error() {
        let data = [b'I', b'P', 0x00, 0x01];
        let err = SidebandFile::from_bytes(&data).unwrap_err();
        assert!(err.contains("truncated"), "got: {}", err);
    }
}
