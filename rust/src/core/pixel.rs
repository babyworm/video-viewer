use std::collections::HashMap;

use super::formats::{FormatType, VideoFormat};

/// Pixel information at a specific coordinate.
#[derive(Debug, Clone)]
pub struct PixelInfo {
    pub x: u32,
    pub y: u32,
    /// Hex representation of raw bytes at the pixel position (e.g., "C8 64 96").
    pub raw_hex: String,
    /// Component values keyed by name ("Y","U","V" or "R","G","B").
    pub components: HashMap<String, u16>,
    /// 3x3 neighborhood grid of hex strings, row-major. Out-of-bounds cells are "--".
    pub neighborhood: Vec<Vec<String>>,
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
        FormatType::YuvPlanar | FormatType::YuvSemiPlanar | FormatType::Grey => {
            if idx < data.len() {
                format!("{:02X}", data[idx])
            } else {
                "--".to_string()
            }
        }
        FormatType::YuvPacked => {
            // 2 bytes per pixel; first byte is Y (or U for UYVY)
            let offset = idx * 2;
            if offset < data.len() {
                format!("{:02X}", data[offset])
            } else {
                "--".to_string()
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
            // Return Y byte only (planar chroma lives at different offsets)
            if idx < data.len() {
                vec![data[idx]]
            } else {
                vec![]
            }
        }
        FormatType::YuvSemiPlanar => {
            if idx < data.len() {
                vec![data[idx]]
            } else {
                vec![]
            }
        }
        FormatType::YuvPacked => {
            let offset = idx * 2;
            if offset + 1 < data.len() {
                vec![data[offset], data[offset + 1]]
            } else if offset < data.len() {
                vec![data[offset]]
            } else {
                vec![]
            }
        }
        FormatType::Rgb => {
            let bpp_bytes = (format.bpp as usize).max(8) / 8;
            let offset = idx * bpp_bytes;
            (0..bpp_bytes)
                .filter_map(|i| data.get(offset + i).copied())
                .collect()
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

        FormatType::YuvSemiPlanar => {
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

            match format.fourcc.as_str() {
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

        FormatType::YuvPacked => {
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
            if format.bit_depth <= 8 {
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
    sub_grid_size: u32,
) -> PixelInfo {
    let raw_hex = raw_hex_at(data, width, height, format, x, y);
    let components = extract_components(data, width, height, format, x, y);

    // Build neighborhood grid
    let neighborhood = if sub_grid_size > 0 {
        let sg = sub_grid_size as i64;
        let cell_x = (x as i64 / sg) * sg;
        let cell_y = (y as i64 / sg) * sg;
        let x_start = cell_x - 1;
        let y_start = cell_y - 1;
        let x_end = cell_x + sg;
        let y_end = cell_y + sg;
        let mut nb = Vec::new();
        for py in y_start..=y_end {
            let mut row = Vec::new();
            for px in x_start..=x_end {
                if px < 0 || py < 0 || px >= width as i64 || py >= height as i64 {
                    row.push("--".to_string());
                } else {
                    row.push(pixel_first_byte_hex(
                        data,
                        width,
                        height,
                        format,
                        px as u32,
                        py as u32,
                    ));
                }
            }
            nb.push(row);
        }
        nb
    } else {
        // Standard 3×3
        let mut nb = Vec::new();
        for dy in [-1i64, 0, 1] {
            let mut row = Vec::new();
            for dx in [-1i64, 0, 1] {
                let px = x as i64 + dx;
                let py = y as i64 + dy;
                if px < 0 || py < 0 || px >= width as i64 || py >= height as i64 {
                    row.push("--".to_string());
                } else {
                    row.push(pixel_first_byte_hex(
                        data,
                        width,
                        height,
                        format,
                        px as u32,
                        py as u32,
                    ));
                }
            }
            nb.push(row);
        }
        nb
    };

    PixelInfo {
        x,
        y,
        raw_hex,
        components,
        neighborhood,
    }
}
