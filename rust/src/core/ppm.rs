use std::io::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PpmEncoding {
    Ascii,
    Binary,
}

/// Parsed PPM header information.
#[derive(Debug, Clone)]
pub struct PpmHeader {
    pub encoding: PpmEncoding,
    pub width: u32,
    pub height: u32,
    pub max_val: u16,
    /// Byte offset where pixel data begins.
    pub data_offset: usize,
}

/// Parse a PPM P3/P6 header from raw bytes.
///
/// Supports comments (lines starting with `#`) and flexible whitespace.
pub fn parse_ppm_header(data: &[u8]) -> Result<PpmHeader, String> {
    if data.len() < 2 {
        return Err("PPM data too short".to_string());
    }

    let encoding = match &data[..2] {
        b"P3" => PpmEncoding::Ascii,
        b"P6" => PpmEncoding::Binary,
        _ => return Err("Not a PPM P3/P6 file (bad magic)".to_string()),
    };
    let mut pos = 2;

    let width = read_ascii_int(data, &mut pos)?;
    let height = read_ascii_int(data, &mut pos)?;
    let max_val = read_ascii_int(data, &mut pos)?;

    if width == 0 || height == 0 {
        return Err("PPM width/height must be > 0".to_string());
    }
    if max_val == 0 || max_val > 65535 {
        return Err(format!("PPM max_val {max_val} out of range"));
    }

    match encoding {
        PpmEncoding::Ascii => skip_ws_comments(data, &mut pos),
        PpmEncoding::Binary => {
            if pos >= data.len() || !data[pos].is_ascii_whitespace() {
                return Err("Expected whitespace before PPM pixel data".to_string());
            }
            pos += 1;
        }
    }

    Ok(PpmHeader {
        encoding,
        width,
        height,
        max_val: max_val as u16,
        data_offset: pos,
    })
}

/// Decode PPM P3/P6 data to RGB24 pixels.
pub fn decode_ppm_to_rgb(data: &[u8]) -> Result<(PpmHeader, Vec<u8>), String> {
    let header = parse_ppm_header(data)?;
    let samples = sample_count(header.width, header.height)?;
    let max_val = header.max_val as u32;
    let mut rgb = Vec::with_capacity(samples);

    match header.encoding {
        PpmEncoding::Ascii => {
            let mut pos = header.data_offset;
            for _ in 0..samples {
                let value = read_ascii_int(data, &mut pos)?;
                if value > max_val {
                    return Err(format!(
                        "PPM sample {value} exceeds max_val {}",
                        header.max_val
                    ));
                }
                rgb.push(scale_sample(value, header.max_val));
            }
        }
        PpmEncoding::Binary if header.max_val < 256 => {
            let end = header
                .data_offset
                .checked_add(samples)
                .ok_or_else(|| "PPM pixel data range is too large".to_string())?;
            let bytes = data
                .get(header.data_offset..end)
                .ok_or_else(|| "PPM pixel data too short".to_string())?;
            if header.max_val == 255 {
                rgb.extend_from_slice(bytes);
            } else {
                rgb.extend(
                    bytes
                        .iter()
                        .map(|&value| scale_sample(value as u32, header.max_val)),
                );
            }
        }
        PpmEncoding::Binary => {
            let bytes = samples
                .checked_mul(2)
                .and_then(|len| header.data_offset.checked_add(len))
                .and_then(|end| data.get(header.data_offset..end))
                .ok_or_else(|| "PPM pixel data too short".to_string())?;
            for sample in bytes.chunks_exact(2) {
                let value = u16::from_be_bytes([sample[0], sample[1]]) as u32;
                if value > max_val {
                    return Err(format!(
                        "PPM sample {value} exceeds max_val {}",
                        header.max_val
                    ));
                }
                rgb.push(scale_sample(value, header.max_val));
            }
        }
    }

    Ok((header, rgb))
}

fn skip_ws_comments(data: &[u8], pos: &mut usize) {
    loop {
        while *pos < data.len() && data[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        if *pos < data.len() && data[*pos] == b'#' {
            while *pos < data.len() && data[*pos] != b'\n' {
                *pos += 1;
            }
        } else {
            break;
        }
    }
}

fn read_ascii_int(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    skip_ws_comments(data, pos);
    let start = *pos;
    while *pos < data.len() && data[*pos].is_ascii_digit() {
        *pos += 1;
    }
    if start == *pos {
        return Err("Expected integer in PPM data".to_string());
    }
    let s = std::str::from_utf8(&data[start..*pos])
        .map_err(|_| "Invalid UTF-8 in PPM data".to_string())?;
    s.parse::<u32>()
        .map_err(|e| format!("PPM int parse error: {e}"))
}

fn sample_count(width: u32, height: u32) -> Result<usize, String> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| "PPM dimensions are too large".to_string())
}

fn scale_sample(value: u32, max_val: u16) -> u8 {
    let max_val = max_val as u32;
    ((value * 255 + max_val / 2) / max_val) as u8
}

/// Write RGB24 pixel data as a PPM P6 file.
pub fn write_ppm<W: Write>(writer: &mut W, width: u32, height: u32, rgb: &[u8]) -> Result<(), String> {
    let expected = (width as usize) * (height as usize) * 3;
    if rgb.len() != expected {
        return Err(format!(
            "PPM write: RGB buffer size {} != expected {} ({}x{}x3)",
            rgb.len(), expected, width, height
        ));
    }

    let header = format!("P6\n{} {}\n255\n", width, height);
    writer.write_all(header.as_bytes())
        .map_err(|e| format!("PPM write header error: {e}"))?;
    writer.write_all(rgb)
        .map_err(|e| format!("PPM write pixel data error: {e}"))?;
    Ok(())
}
