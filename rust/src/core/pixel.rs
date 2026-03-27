use std::collections::HashMap;

use super::formats::{FormatType, VideoFormat};

/// Unpack one 10-bit sample from LE bitstream (NV15/NV20: 4 samples in 5 bytes).
fn unpack_10bit_le_sample(data: &[u8], group_off: usize, idx: usize) -> u16 {
    let b0 = data[group_off] as u16;
    let b1 = data[group_off + 1] as u16;
    let b2 = data[group_off + 2] as u16;
    let b3 = data[group_off + 3] as u16;
    let b4 = data[group_off + 4] as u16;
    match idx {
        0 => b0 | ((b1 & 0x03) << 8),
        1 => (b1 >> 2) | ((b2 & 0x0F) << 6),
        2 => (b2 >> 4) | ((b3 & 0x3F) << 4),
        _ => (b3 >> 6) | (b4 << 2),
    }
}

/// Unpack one 10-bit sample from BE bitstream (Y10BPACK: 4 samples in 5 bytes, MSB-first).
fn unpack_10bit_be_sample(data: &[u8], group_off: usize, idx: usize) -> u16 {
    let b0 = data[group_off] as u16;
    let b1 = data[group_off + 1] as u16;
    let b2 = data[group_off + 2] as u16;
    let b3 = data[group_off + 3] as u16;
    let b4 = data[group_off + 4] as u16;
    match idx {
        0 => (b0 << 2) | (b1 >> 6),
        1 => ((b1 & 0x3F) << 4) | (b2 >> 4),
        2 => ((b2 & 0x0F) << 6) | (b3 >> 2),
        _ => ((b3 & 0x03) << 8) | b4,
    }
}

/// Unpack one 10-bit sample from MIPI RAW10 (Y10P: 4 MSB bytes + 1 LSB byte).
fn unpack_y10p_sample(data: &[u8], group_off: usize, idx: usize) -> u16 {
    let msb = data[group_off + idx] as u16;
    let lsb = ((data[group_off + 4] >> (idx * 2)) & 0x03) as u16;
    (msb << 2) | lsb
}

/// Read a packed 10-bit sample at pixel index from a Grey packed format.
fn grey_packed_sample(data: &[u8], fourcc: &str, pixel_idx: usize) -> Option<u16> {
    let group = pixel_idx / 4;
    let si = pixel_idx % 4;
    let off = group * 5;
    if off + 4 >= data.len() {
        return None;
    }
    Some(match fourcc {
        "Y10B" => unpack_10bit_be_sample(data, off, si),
        "Y10P" => unpack_y10p_sample(data, off, si),
        _ => 0,
    })
}

/// Compute Y sample byte offset for T010 (tiled P010, 4x4 tiles).
fn t010_y_offset(px: usize, py: usize, w: usize) -> usize {
    let tiles_x = w / 4;
    let tile_x = px / 4;
    let tile_y = py / 4;
    let in_x = px % 4;
    let in_y = py % 4;
    (tile_y * tiles_x + tile_x) * 32 + (in_y * 4 + in_x) * 2
}

/// Pixel information at a specific coordinate.
#[derive(Debug, Clone)]
pub struct PixelInfo {
    pub x: u32,
    pub y: u32,
    /// Hex representation of raw bytes at the pixel position (e.g., "C8 64 96").
    pub raw_hex: String,
    /// Component values keyed by name ("Y","U","V" or "R","G","B").
    pub components: HashMap<String, u16>,
    /// 8x8 neighborhood grid of hex strings, row-major. Out-of-bounds cells are empty.
    pub neighborhood: Vec<Vec<String>>,
    /// Current pixel's column index within the neighborhood grid.
    pub nb_cursor_col: usize,
    /// Current pixel's row index within the neighborhood grid.
    pub nb_cursor_row: usize,
}

/// Extract the first-byte (luminance or single-channel) hex at a given pixel position.
/// Returns "--" for out-of-bounds coordinates.
fn pixel_first_byte_hex(
    data: &[u8],
    width: u32,
    height: u32,
    format: &VideoFormat,
    px: u32,
    py: u32,
) -> String {
    if px >= width || py >= height {
        return "--".to_string();
    }
    let idx = (py as usize) * (width as usize) + (px as usize);
    match format.format_type {
        FormatType::YuvPlanar => {
            if format.bit_depth > 8 {
                let offset = idx * 2;
                if offset + 1 < data.len() {
                    let val = u16::from_le_bytes([data[offset], data[offset + 1]]);
                    let mask = (1u16 << format.bit_depth) - 1;
                    let shifted = ((val & mask) >> (format.bit_depth - 8)) as u8;
                    format!("{:02X}", shifted)
                } else {
                    "--".to_string()
                }
            } else if idx < data.len() {
                format!("{:02X}", data[idx])
            } else {
                "--".to_string()
            }
        }
        FormatType::YuvSemiPlanar => {
            let fc = format.fourcc.as_str();
            if matches!(fc, "NV15" | "NV20") {
                let group = idx / 4;
                let si = idx % 4;
                let off = group * 5;
                if off + 4 < data.len() {
                    format!("{:02X}", (unpack_10bit_le_sample(data, off, si) >> 2) as u8)
                } else {
                    "--".to_string()
                }
            } else if fc == "T010" {
                let off = t010_y_offset(px as usize, py as usize, width as usize);
                if off + 1 < data.len() {
                    let val = u16::from_le_bytes([data[off], data[off + 1]]);
                    format!("{:02X}", (val >> 8) as u8)
                } else {
                    "--".to_string()
                }
            } else if format.bit_depth > 8 {
                let offset = idx * 2;
                if offset + 1 < data.len() {
                    let val = u16::from_le_bytes([data[offset], data[offset + 1]]);
                    format!("{:02X}", (val >> 8) as u8)
                } else {
                    "--".to_string()
                }
            } else if idx < data.len() {
                format!("{:02X}", data[idx])
            } else {
                "--".to_string()
            }
        }
        FormatType::Grey => {
            let fc = format.fourcc.as_str();
            if matches!(fc, "Y10B" | "Y10P") {
                if let Some(val) = grey_packed_sample(data, fc, idx) {
                    format!("{:02X}", (val >> 2) as u8)
                } else {
                    "--".to_string()
                }
            } else if format.bit_depth > 8 {
                let offset = idx * 2;
                if offset + 1 < data.len() {
                    let val = u16::from_le_bytes([data[offset], data[offset + 1]]);
                    let shifted = (val >> (format.bit_depth - 8)) as u8;
                    format!("{:02X}", shifted)
                } else {
                    "--".to_string()
                }
            } else if idx < data.len() {
                format!("{:02X}", data[idx])
            } else {
                "--".to_string()
            }
        }
        FormatType::YuvPacked => {
            if format.bit_depth > 8 {
                // Y210: 8 bytes per pixel pair, Y samples at u16 offsets 0 and 4
                let pair_idx = px as usize / 2;
                let is_odd = (px % 2) == 1;
                let base = (py as usize * (width as usize) / 2 + pair_idx) * 8;
                let y_off = if is_odd { base + 4 } else { base };
                if y_off + 1 < data.len() {
                    let val = u16::from_le_bytes([data[y_off], data[y_off + 1]]);
                    format!("{:02X}", (val >> 8) as u8)
                } else {
                    "--".to_string()
                }
            } else {
                // 2 bytes per pixel; first byte is Y (or U for UYVY)
                let offset = idx * 2;
                if offset < data.len() {
                    format!("{:02X}", data[offset])
                } else {
                    "--".to_string()
                }
            }
        }
        FormatType::Rgb => {
            let bpp_bytes = (format.bpp as usize).max(8) / 8;
            let offset = idx * bpp_bytes;
            if offset < data.len() {
                format!("{:02X}", data[offset])
            } else {
                "--".to_string()
            }
        }
        FormatType::Bayer => {
            if format.bit_depth <= 8 {
                if idx < data.len() {
                    format!("{:02X}", data[idx])
                } else {
                    "--".to_string()
                }
            } else {
                let offset = idx * 2;
                if offset < data.len() {
                    format!("{:02X}", data[offset])
                } else {
                    "--".to_string()
                }
            }
        }
        FormatType::Compressed => "--".to_string(),
    }
}

/// Build raw hex string for a pixel (1–4 bytes depending on format).
fn raw_hex_at(data: &[u8], width: u32, height: u32, format: &VideoFormat, x: u32, y: u32) -> String {
    if x >= width || y >= height {
        return String::new();
    }
    let idx = (y as usize) * (width as usize) + (x as usize);

    let bytes: Vec<u8> = match format.format_type {
        FormatType::YuvPlanar => {
            if format.bit_depth > 8 {
                // 10-bit planar: Y sample is u16 LE
                let offset = idx * 2;
                if offset + 1 < data.len() {
                    vec![data[offset], data[offset + 1]]
                } else {
                    vec![]
                }
            } else if idx < data.len() {
                vec![data[idx]]
            } else {
                vec![]
            }
        }
        FormatType::YuvSemiPlanar => {
            let fc = format.fourcc.as_str();
            if matches!(fc, "NV15" | "NV20") {
                // Show the 5-byte group containing this pixel's Y sample
                let group = idx / 4;
                let off = group * 5;
                (0..5).filter_map(|i| data.get(off + i).copied()).collect()
            } else if fc == "T010" {
                let off = t010_y_offset(x as usize, y as usize, width as usize);
                if off + 1 < data.len() { vec![data[off], data[off + 1]] } else { vec![] }
            } else if format.bit_depth > 8 {
                let offset = idx * 2;
                if offset + 1 < data.len() {
                    vec![data[offset], data[offset + 1]]
                } else {
                    vec![]
                }
            } else if idx < data.len() {
                vec![data[idx]]
            } else {
                vec![]
            }
        }
        FormatType::YuvPacked => {
            if format.bit_depth > 8 {
                // Y210: 8 bytes per pixel pair
                let pair_idx = (x as usize) / 2;
                let base = ((y as usize) * (width as usize) / 2 + pair_idx) * 8;
                if base + 7 < data.len() {
                    // Show all 4 u16 values for the pair
                    let is_odd = (x % 2) == 1;
                    let y_off = if is_odd { base + 4 } else { base };
                    vec![data[y_off], data[y_off + 1], data[base + 2], data[base + 3], data[base + 6], data[base + 7]]
                } else {
                    vec![]
                }
            } else {
                let offset = idx * 2;
                if offset + 1 < data.len() {
                    vec![data[offset], data[offset + 1]]
                } else if offset < data.len() {
                    vec![data[offset]]
                } else {
                    vec![]
                }
            }
        }
        FormatType::Rgb => {
            let bpp_bytes = (format.bpp as usize).max(8) / 8;
            let offset = idx * bpp_bytes;
            (0..bpp_bytes)
                .filter_map(|i| data.get(offset + i).copied())
                .collect()
        }
        FormatType::Grey if matches!(format.fourcc.as_str(), "Y10B" | "Y10P") => {
            let group = idx / 4;
            let off = group * 5;
            (0..5).filter_map(|i| data.get(off + i).copied()).collect()
        }
        FormatType::Bayer | FormatType::Grey => {
            if format.bit_depth <= 8 {
                if idx < data.len() {
                    vec![data[idx]]
                } else {
                    vec![]
                }
            } else {
                let offset = idx * 2;
                if offset + 1 < data.len() {
                    vec![data[offset], data[offset + 1]]
                } else if offset < data.len() {
                    vec![data[offset]]
                } else {
                    vec![]
                }
            }
        }
        FormatType::Compressed => vec![],
    };

    bytes.iter().map(|b| format!("{:02X}", b)).collect::<Vec<_>>().join(" ")
}

/// Compute component values (Y/U/V or R/G/B) for a pixel.
fn extract_components(
    data: &[u8],
    width: u32,
    height: u32,
    format: &VideoFormat,
    x: u32,
    y: u32,
) -> HashMap<String, u16> {
    let mut components = HashMap::new();
    let w = width as usize;
    let h = height as usize;
    let px = x as usize;
    let py = y as usize;
    let y_size = w * h;

    match format.format_type {
        FormatType::YuvPlanar => {
            if format.bit_depth > 8 {
                // High-bit planar: each sample is u16 LE, LSB-aligned
                let mask = (1u16 << format.bit_depth) - 1;
                let (sx, sy) = (format.subsampling.0 as usize, format.subsampling.1 as usize);
                let y_samples = w * h;
                let uv_w = w / sx;
                let uv_samples = uv_w * (h / sy);

                let y_byte = (py * w + px) * 2;
                if y_byte + 1 < data.len() {
                    let val = u16::from_le_bytes([data[y_byte], data[y_byte + 1]]);
                    components.insert("Y".to_string(), val & mask);
                }

                let c_x = px / sx;
                let c_y = py / sy;
                let c_idx = c_y * uv_w + c_x;
                let u_byte = y_samples * 2 + c_idx * 2;
                let v_byte = y_samples * 2 + uv_samples * 2 + c_idx * 2;
                if u_byte + 1 < data.len() {
                    let val = u16::from_le_bytes([data[u_byte], data[u_byte + 1]]);
                    components.insert("U".to_string(), val & mask);
                }
                if v_byte + 1 < data.len() {
                    let val = u16::from_le_bytes([data[v_byte], data[v_byte + 1]]);
                    components.insert("V".to_string(), val & mask);
                }
            } else {
                let y_idx = py * w + px;
                if y_idx >= data.len() {
                    return components;
                }
                components.insert("Y".to_string(), data[y_idx] as u16);

                let (sx, sy) = (format.subsampling.0 as usize, format.subsampling.1 as usize);
                let c_w = w / sx;
                let c_x = px / sx;
                let c_y = py / sy;
                let c_idx = c_y * c_w + c_x;
                let uv_size = c_w * (h / sy);

                match format.fourcc.as_str() {
                    "YU12" | "422P" => {
                        let u_idx = y_size + c_idx;
                        let v_idx = y_size + uv_size + c_idx;
                        if u_idx < data.len() {
                            components.insert("U".to_string(), data[u_idx] as u16);
                        }
                        if v_idx < data.len() {
                            components.insert("V".to_string(), data[v_idx] as u16);
                        }
                    }
                    "YV12" => {
                        let v_idx = y_size + c_idx;
                        let u_idx = y_size + uv_size + c_idx;
                        if v_idx < data.len() {
                            components.insert("V".to_string(), data[v_idx] as u16);
                        }
                        if u_idx < data.len() {
                            components.insert("U".to_string(), data[u_idx] as u16);
                        }
                    }
                    "444P" | "Y444" => {
                        let u_idx = y_size + py * w + px;
                        let v_idx = y_size * 2 + py * w + px;
                        if u_idx < data.len() {
                            components.insert("U".to_string(), data[u_idx] as u16);
                        }
                        if v_idx < data.len() {
                            components.insert("V".to_string(), data[v_idx] as u16);
                        }
                    }
                    _ => {
                        // Generic planar: U then V
                        let u_idx = y_size + c_idx;
                        let v_idx = y_size + uv_size + c_idx;
                        if u_idx < data.len() {
                            components.insert("U".to_string(), data[u_idx] as u16);
                        }
                        if v_idx < data.len() {
                            components.insert("V".to_string(), data[v_idx] as u16);
                        }
                    }
                }
            }
        }

        FormatType::YuvSemiPlanar => {
            let fc = format.fourcc.as_str();

            if matches!(fc, "NV15" | "NV20") {
                // Packed 10-bit semi-planar: unpack Y and UV from LE bitstream
                let (sx, sy) = (format.subsampling.0 as usize, format.subsampling.1 as usize);
                let y_idx = py * w + px;
                let y_group = y_idx / 4;
                let y_si = y_idx % 4;
                let y_off = y_group * 5;
                if y_off + 4 < data.len() {
                    components.insert("Y".to_string(), unpack_10bit_le_sample(data, y_off, y_si));
                }
                let y_plane_bytes = (w * h * 5) / 4;
                let uv_w = w / sx;
                let c_x = px / sx;
                let c_y = py / sy;
                let uv_linear = c_y * uv_w * 2 + c_x * 2; // interleaved U,V
                let uv_group = uv_linear / 4;
                let uv_si = uv_linear % 4;
                let uv_off = y_plane_bytes + uv_group * 5;
                if uv_off + 4 < data.len() {
                    let u_val = unpack_10bit_le_sample(data, uv_off, uv_si);
                    components.insert("U".to_string(), u_val);
                    // V is the next sample
                    let v_linear = uv_linear + 1;
                    let v_group = v_linear / 4;
                    let v_si = v_linear % 4;
                    let v_off = y_plane_bytes + v_group * 5;
                    if v_off + 4 < data.len() {
                        components.insert("V".to_string(), unpack_10bit_le_sample(data, v_off, v_si));
                    }
                }
            } else if fc == "T010" {
                // Tiled P010: compute tiled addresses, then read as MSB-aligned u16
                let shift = 16u32 - format.bit_depth;
                let off = t010_y_offset(px, py, w);
                if off + 1 < data.len() {
                    let val = u16::from_le_bytes([data[off], data[off + 1]]);
                    components.insert("Y".to_string(), val >> shift);
                }
                // UV plane: tiles at offset y_plane_tiled_bytes, half height
                let y_plane_bytes = w * h * 2;
                let (sx, sy) = (format.subsampling.0 as usize, format.subsampling.1 as usize);
                let uv_w = w;
                let c_x = px / sx;
                let c_y = py / sy;
                // UV tile addressing: tile contains 4x4 u16 samples (interleaved U,V pairs)
                let uv_tiles_x = uv_w / 4;
                let tile_x = (c_x * 2) / 4; // c_x*2 because interleaved U,V
                let tile_y = c_y / 4;
                let in_x = (c_x * 2) % 4;
                let in_y = c_y % 4;
                let uv_tile_off = y_plane_bytes + (tile_y * uv_tiles_x + tile_x) * 32 + (in_y * 4 + in_x) * 2;
                if uv_tile_off + 3 < data.len() {
                    let u_val = u16::from_le_bytes([data[uv_tile_off], data[uv_tile_off + 1]]);
                    let v_val = u16::from_le_bytes([data[uv_tile_off + 2], data[uv_tile_off + 3]]);
                    components.insert("U".to_string(), u_val >> shift);
                    components.insert("V".to_string(), v_val >> shift);
                }
            } else if format.bit_depth > 8 {
                // P010/P012/P016/P210: Y is u16 LE MSB-aligned, UV interleaved u16 LE pairs
                let y_byte = (py * w + px) * 2;
                if y_byte + 1 < data.len() {
                    let val = u16::from_le_bytes([data[y_byte], data[y_byte + 1]]);
                    // Report the full N-bit value (MSB-aligned: shift right by 16-N)
                    let shift = 16 - format.bit_depth;
                    components.insert("Y".to_string(), val >> shift);
                }

                let y_plane_bytes = w * h * 2;
                let (sx, sy) = (format.subsampling.0 as usize, format.subsampling.1 as usize);
                let uv_w = w / sx;
                let c_x = px / sx;
                let c_y = py / sy;
                let uv_base = y_plane_bytes + (c_y * uv_w + c_x) * 4;
                let shift = 16 - format.bit_depth;
                if uv_base + 3 < data.len() {
                    let u_val = u16::from_le_bytes([data[uv_base], data[uv_base + 1]]);
                    let v_val = u16::from_le_bytes([data[uv_base + 2], data[uv_base + 3]]);
                    components.insert("U".to_string(), u_val >> shift);
                    components.insert("V".to_string(), v_val >> shift);
                }
            } else {
                let y_idx = py * w + px;
                if y_idx >= data.len() {
                    return components;
                }
                components.insert("Y".to_string(), data[y_idx] as u16);

                let (sx, sy) = (format.subsampling.0 as usize, format.subsampling.1 as usize);
                let c_w = w / sx;
                let c_x = px / sx;
                let c_y = py / sy;
                let uv_idx = (c_y * c_w + c_x) * 2;

                match fc {
                    "NV12" | "NV16" | "NM12" => {
                        let u_pos = y_size + uv_idx;
                        let v_pos = y_size + uv_idx + 1;
                        if u_pos < data.len() {
                            components.insert("U".to_string(), data[u_pos] as u16);
                        }
                        if v_pos < data.len() {
                            components.insert("V".to_string(), data[v_pos] as u16);
                        }
                    }
                    _ => {
                        // NV21, NV61, NM21 — V then U
                        let v_pos = y_size + uv_idx;
                        let u_pos = y_size + uv_idx + 1;
                        if v_pos < data.len() {
                            components.insert("V".to_string(), data[v_pos] as u16);
                        }
                        if u_pos < data.len() {
                            components.insert("U".to_string(), data[u_pos] as u16);
                        }
                    }
                }
            }
        }

        FormatType::YuvPacked => {
            if format.bit_depth > 8
                && matches!(format.fourcc.as_str(), "Y210" | "Y212" | "Y216")
            {
                // Y210/Y212/Y216: [Y0:u16, Cb:u16, Y1:u16, Cr:u16] per pixel pair, MSB-aligned
                let shift = 16 - format.bit_depth;
                let pair_idx = px / 2;
                let base = (py * w / 2 + pair_idx) * 8;
                let is_odd = (px % 2) == 1;
                let y_off = if is_odd { base + 4 } else { base };
                if y_off + 1 < data.len() {
                    let val = u16::from_le_bytes([data[y_off], data[y_off + 1]]);
                    components.insert("Y".to_string(), val >> shift);
                }
                if base + 3 < data.len() {
                    let val = u16::from_le_bytes([data[base + 2], data[base + 3]]);
                    components.insert("U".to_string(), val >> shift);
                }
                if base + 7 < data.len() {
                    let val = u16::from_le_bytes([data[base + 6], data[base + 7]]);
                    components.insert("V".to_string(), val >> shift);
                }
                return components;
            }

            // YUYV layout: [Y0 U0 Y1 V0] per pair of pixels
            let pair_idx = px / 2;
            let offset = (py * w + pair_idx * 2) * 2; // 4 bytes per pixel pair
            let is_odd = (px % 2) == 1;

            match format.fourcc.as_str() {
                "YUYV" => {
                    let y_off = if is_odd { offset + 2 } else { offset };
                    if y_off < data.len() {
                        components.insert("Y".to_string(), data[y_off] as u16);
                    }
                    if offset + 1 < data.len() {
                        components.insert("U".to_string(), data[offset + 1] as u16);
                    }
                    if offset + 3 < data.len() {
                        components.insert("V".to_string(), data[offset + 3] as u16);
                    }
                }
                "UYVY" => {
                    let y_off = if is_odd { offset + 3 } else { offset + 1 };
                    if y_off < data.len() {
                        components.insert("Y".to_string(), data[y_off] as u16);
                    }
                    if offset < data.len() {
                        components.insert("U".to_string(), data[offset] as u16);
                    }
                    if offset + 2 < data.len() {
                        components.insert("V".to_string(), data[offset + 2] as u16);
                    }
                }
                "YVYU" => {
                    let y_off = if is_odd { offset + 2 } else { offset };
                    if y_off < data.len() {
                        components.insert("Y".to_string(), data[y_off] as u16);
                    }
                    if offset + 1 < data.len() {
                        components.insert("V".to_string(), data[offset + 1] as u16);
                    }
                    if offset + 3 < data.len() {
                        components.insert("U".to_string(), data[offset + 3] as u16);
                    }
                }
                _ => {}
            }
        }

        FormatType::Rgb => {
            let bpp = format.bpp as usize;
            let bpp_bytes = bpp.max(8) / 8;
            let offset = (py * w + px) * bpp_bytes;

            match format.fourcc.as_str() {
                "RGB3" => {
                    if offset + 2 < data.len() {
                        components.insert("R".to_string(), data[offset] as u16);
                        components.insert("G".to_string(), data[offset + 1] as u16);
                        components.insert("B".to_string(), data[offset + 2] as u16);
                    }
                }
                "BGR3" => {
                    if offset + 2 < data.len() {
                        components.insert("B".to_string(), data[offset] as u16);
                        components.insert("G".to_string(), data[offset + 1] as u16);
                        components.insert("R".to_string(), data[offset + 2] as u16);
                    }
                }
                "RGB4" | "RGBX" => {
                    if offset + 3 < data.len() {
                        components.insert("R".to_string(), data[offset] as u16);
                        components.insert("G".to_string(), data[offset + 1] as u16);
                        components.insert("B".to_string(), data[offset + 2] as u16);
                    }
                }
                "BGR4" => {
                    if offset + 3 < data.len() {
                        components.insert("B".to_string(), data[offset] as u16);
                        components.insert("G".to_string(), data[offset + 1] as u16);
                        components.insert("R".to_string(), data[offset + 2] as u16);
                    }
                }
                "RGB1" => {
                    // RGB332: R[7:5] G[4:2] B[1:0]
                    if offset < data.len() {
                        let val = data[offset];
                        components.insert("R".to_string(), (val & 0xE0) as u16);
                        components.insert("G".to_string(), ((val & 0x1C) << 3) as u16);
                        components.insert("B".to_string(), ((val & 0x03) << 6) as u16);
                    }
                }
                _ => {
                    // Generic: read R/G/B from first 3 bytes if available
                    if bpp_bytes >= 3 && offset + 2 < data.len() {
                        components.insert("R".to_string(), data[offset] as u16);
                        components.insert("G".to_string(), data[offset + 1] as u16);
                        components.insert("B".to_string(), data[offset + 2] as u16);
                    }
                }
            }
        }

        FormatType::Grey => {
            let idx = py * w + px;
            let fc = format.fourcc.as_str();
            if matches!(fc, "Y10B" | "Y10P") {
                if let Some(val) = grey_packed_sample(data, fc, idx) {
                    components.insert("Y".to_string(), val);
                }
            } else if format.bit_depth <= 8 {
                if idx < data.len() {
                    components.insert("Y".to_string(), data[idx] as u16);
                }
            } else {
                let offset = idx * 2;
                if offset + 1 < data.len() {
                    let val = u16::from_le_bytes([data[offset], data[offset + 1]]);
                    components.insert("Y".to_string(), val);
                }
            }
        }

        FormatType::Bayer => {
            let idx = py * w + px;
            if format.bit_depth <= 8 {
                if idx < data.len() {
                    components.insert("Bayer".to_string(), data[idx] as u16);
                }
            } else {
                let offset = idx * 2;
                if offset + 1 < data.len() {
                    let val = u16::from_le_bytes([data[offset], data[offset + 1]]);
                    components.insert("Bayer".to_string(), val);
                }
            }
        }

        FormatType::Compressed => {}
    }

    components
}

/// Get pixel information at (x, y) in the given raw frame data.
///
/// - `sub_grid_size`: if > 0, neighborhood is built around the sub-grid cell containing (x,y)
///   with 1-pixel padding; if 0, a standard 3×3 grid centered at (x,y) is used.
pub fn get_pixel_info(
    data: &[u8],
    width: u32,
    height: u32,
    format: &VideoFormat,
    x: u32,
    y: u32,
    _sub_grid_size: u32,
) -> PixelInfo {
    let raw_hex = raw_hex_at(data, width, height, format, x, y);
    let components = extract_components(data, width, height, format, x, y);

    // Build 8x8 neighborhood grid centered on the current pixel.
    const NB_SIZE: i64 = 8;
    let half = NB_SIZE / 2; // 4
    let x_start = x as i64 - half;
    let y_start = y as i64 - half;
    let nb_cursor_col = half as usize;
    let nb_cursor_row = half as usize;
    let mut neighborhood = Vec::with_capacity(NB_SIZE as usize);
    for row_idx in 0..NB_SIZE {
        let py = y_start + row_idx;
        let mut row = Vec::with_capacity(NB_SIZE as usize);
        for col_idx in 0..NB_SIZE {
            let px = x_start + col_idx;
            if px < 0 || py < 0 || px >= width as i64 || py >= height as i64 {
                row.push(String::new());
            } else {
                row.push(pixel_first_byte_hex(
                    data, width, height, format, px as u32, py as u32,
                ));
            }
        }
        neighborhood.push(row);
    }

    PixelInfo {
        x,
        y,
        raw_hex,
        components,
        neighborhood,
        nb_cursor_col,
        nb_cursor_row,
    }
}
