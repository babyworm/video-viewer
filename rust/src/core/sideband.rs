//! Sideband binary parser for ISP parameter overlay.
//!
//! Reads the binary format produced by `isp_emulator` (IspParam.h compatible).
//! Self-contained: no external dependencies beyond `std`.

use std::fmt;

// ---------------------------------------------------------------------------
// Structs
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
    pub ctus: Vec<SidebandCtu>,
}

/// One CTU's sideband parameters.
#[derive(Debug, Clone, Default)]
pub struct SidebandCtu {
    pub activity: u8,           // 0-63
    pub flatness: u8,           // 0-63
    pub edge_density: u8,       // 0-63
    pub noise: u8,              // 0-15
    pub saliency: u8,           // 0-63
    pub chroma_importance: u8,  // 0-31
    pub confidence: u8,         // 0-15
    pub ctu_flags: u8,
    pub qp_delta: i8,
    pub chroma_cb_delta: i8,
    pub chroma_cr_delta: i8,
    pub lambda_scale_q8: u16,   // Q8.8, 256 = 1.0
    pub rc_importance_weight: u8, // 128 = neutral
    pub sao_prior: i8,
    pub temporal_stability: u8, // 0-255
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAGIC: [u8; 2] = [b'I', b'P'];
const FRAME_PARAM_SIZE: usize = 20;
const CTU_PARAM_SIZE: usize = 16;
const SHORT_HEADER_SIZE: usize = 4;
const EXTENDED_HEADER_SIZE: usize = 6;

// ---------------------------------------------------------------------------
// SidebandFile
// ---------------------------------------------------------------------------

impl SidebandFile {
    /// Read a sideband binary file from disk.
    pub fn open(path: &str) -> Result<Self, String> {
        let data = std::fs::read(path)
            .map_err(|e| format!("failed to read sideband file '{}': {}", path, e))?;
        Self::from_bytes(&data)
    }

    /// Parse sideband data from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.is_empty() {
            return Ok(SidebandFile { frames: Vec::new() });
        }

        let mut frames = Vec::new();
        let mut offset = 0;

        while offset < data.len() {
            let (frame, consumed) = parse_one_frame(data, offset)?;
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
// SidebandFrame
// ---------------------------------------------------------------------------

impl SidebandFrame {
    /// Number of CTUs in this frame.
    pub fn num_ctus(&self) -> usize {
        self.ctus.len()
    }

    /// Lambda scale as float (Q8.8 -> f64).
    pub fn lambda_scale(&self) -> f64 {
        self.frame_lambda_scale_q8 as f64 / 256.0
    }
}

impl fmt::Display for SidebandFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Frame(id={}, scene={}, noise={}, motion={}, qp_bias={}, ctus={})",
            self.frame_id,
            self.scene_class,
            self.noise_class,
            self.motion_class,
            self.frame_qp_bias,
            self.ctus.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// SidebandCtu
// ---------------------------------------------------------------------------

impl SidebandCtu {
    /// Lambda scale as float (Q8.8 -> f64).
    pub fn lambda_scale(&self) -> f64 {
        self.lambda_scale_q8 as f64 / 256.0
    }
}

impl fmt::Display for SidebandCtu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CTU(act={}, flat={}, edge={}, qp_delta={}, lambda={:.3})",
            self.activity,
            self.flatness,
            self.edge_density,
            self.qp_delta,
            self.lambda_scale(),
        )
    }
}

// ---------------------------------------------------------------------------
// Binary parsing
// ---------------------------------------------------------------------------

/// Parse one frame starting at `offset`, return the frame and bytes consumed.
fn parse_one_frame(data: &[u8], offset: usize) -> Result<(SidebandFrame, usize), String> {
    let remaining = data.len() - offset;

    if remaining < SHORT_HEADER_SIZE {
        return Err(format!(
            "truncated header at offset {}: need {} bytes, have {}",
            offset, SHORT_HEADER_SIZE, remaining
        ));
    }

    // Magic check
    if data[offset] != MAGIC[0] || data[offset + 1] != MAGIC[1] {
        return Err(format!(
            "invalid magic at offset {}: expected 'IP', got [{:#04x}, {:#04x}]",
            offset, data[offset], data[offset + 1]
        ));
    }

    let version = data[offset + 2];
    if version > 0 {
        return Err(format!(
            "unsupported sideband version {} at offset {}",
            version, offset
        ));
    }

    // CTU count: 0xFF byte signals extended header (2-byte big-endian count follows).
    // This matches the reference reader in isp_emulator exactly.
    // Note: the writer uses short form for num_ctus <= 255, but 255 CTUs
    // produces byte[3] = 0xFF which is also the extended escape. The reference
    // reader always interprets 0xFF as extended, so we do the same for
    // byte-for-byte compatibility.
    let header_byte = data[offset + 3];
    let (num_ctus, header_size) = if header_byte == 0xFF {
        if remaining < EXTENDED_HEADER_SIZE {
            return Err(format!(
                "truncated extended header at offset {}",
                offset
            ));
        }
        let ext_count = ((data[offset + 4] as usize) << 8) | (data[offset + 5] as usize);
        (ext_count, EXTENDED_HEADER_SIZE)
    } else {
        (header_byte as usize, SHORT_HEADER_SIZE)
    };

    let frame_start = offset + header_size;
    let needed = header_size + FRAME_PARAM_SIZE + CTU_PARAM_SIZE * num_ctus;
    if remaining < needed {
        return Err(format!(
            "truncated frame at offset {}: need {} bytes ({} CTUs), have {}",
            offset, needed, num_ctus, remaining
        ));
    }

    // Frame params (20 bytes)
    let buf = &data[frame_start..frame_start + FRAME_PARAM_SIZE];
    let frame_id = ((buf[0] as i32) << 24)
        | ((buf[1] as i32) << 16)
        | ((buf[2] as i32) << 8)
        | (buf[3] as i32);
    let valid_fields_mask = ((buf[4] as u16) << 8) | (buf[5] as u16);
    let scene_class = buf[6] >> 4;
    let noise_class = buf[7] >> 5;
    let motion_class = buf[8] >> 5;
    let denoise_strength = buf[9] >> 3;
    let sharpen_strength = buf[10] >> 3;
    let scene_flags = buf[11];
    let frame_qp_bias = buf[12] as i8;
    let frame_chroma_cb_bias = buf[13] as i8;
    let frame_chroma_cr_bias = buf[14] as i8;
    let frame_lambda_scale_q8 = ((buf[15] as u16) << 8) | (buf[16] as u16);
    let global_confidence = buf[17];

    // CTU params (16 bytes each)
    let mut ctus = Vec::with_capacity(num_ctus);
    let ctu_base = frame_start + FRAME_PARAM_SIZE;
    for i in 0..num_ctus {
        let co = ctu_base + i * CTU_PARAM_SIZE;
        let cb = &data[co..co + CTU_PARAM_SIZE];
        ctus.push(SidebandCtu {
            activity: cb[0] >> 2,
            flatness: cb[1] >> 2,
            edge_density: cb[2] >> 2,
            noise: cb[3] >> 4,
            saliency: cb[4] >> 2,
            chroma_importance: cb[5] >> 3,
            confidence: cb[6] >> 4,
            ctu_flags: cb[7] >> 5,
            qp_delta: cb[8] as i8,
            chroma_cb_delta: cb[9] as i8,
            chroma_cr_delta: cb[10] as i8,
            lambda_scale_q8: ((cb[11] as u16) << 8) | (cb[12] as u16),
            rc_importance_weight: cb[13],
            sao_prior: cb[14] as i8,
            temporal_stability: cb[15],
        });
    }

    let frame = SidebandFrame {
        frame_id,
        valid_fields_mask,
        scene_class,
        noise_class,
        motion_class,
        denoise_strength,
        sharpen_strength,
        scene_flags,
        frame_qp_bias,
        frame_chroma_cb_bias,
        frame_chroma_cr_bias,
        frame_lambda_scale_q8,
        global_confidence,
        ctus,
    };

    Ok((frame, needed))
}

// ---------------------------------------------------------------------------
// Overlay mode enum
// ---------------------------------------------------------------------------

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
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
        // Valid magic but not enough bytes for frame params
        let data = [b'I', b'P', 0x00, 0x01];
        let err = SidebandFile::from_bytes(&data).unwrap_err();
        assert!(err.contains("truncated"), "got: {}", err);
    }
}
