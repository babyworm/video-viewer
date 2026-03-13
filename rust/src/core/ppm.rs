use std::io::Write;

/// Parsed PPM (P6) header information.
#[derive(Debug, Clone)]
pub struct PpmHeader {
    pub width: u32,
    pub height: u32,
    pub max_val: u16,
    /// Byte offset where pixel data begins.
    pub data_offset: usize,
}

/// Parse a PPM P6 (binary) header from raw bytes.
///
/// Supports comments (lines starting with `#`) and flexible whitespace.
pub fn parse_ppm_header(data: &[u8]) -> Result<PpmHeader, String> {
    if data.len() < 7 {
        return Err("PPM data too short".to_string());
    }

    // Parse magic "P6"
    if data.len() < 2 || data[0] != b'P' || data[1] != b'6' {
        return Err("Not a PPM P6 file (bad magic)".to_string());
    }
    let mut pos = 2;

    // Helper: skip whitespace and comments
    let skip_ws_comments = |pos: &mut usize| {
        loop {
            while *pos < data.len() && (data[*pos] == b' ' || data[*pos] == b'\t'
                || data[*pos] == b'\r' || data[*pos] == b'\n') {
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
    };

    // Helper: read next integer token
    let read_int = |pos: &mut usize| -> Result<u32, String> {
        skip_ws_comments(pos);
        let start = *pos;
        while *pos < data.len() && data[*pos].is_ascii_digit() {
            *pos += 1;
        }
        if start == *pos {
            return Err("Expected integer in PPM header".to_string());
        }
        let s = std::str::from_utf8(&data[start..*pos])
            .map_err(|_| "Invalid UTF-8 in PPM header".to_string())?;
        s.parse::<u32>().map_err(|e| format!("PPM header int parse error: {e}"))
    };

    let width = read_int(&mut pos)?;
    let height = read_int(&mut pos)?;
    let max_val = read_int(&mut pos)?;

    if width == 0 || height == 0 {
        return Err("PPM width/height must be > 0".to_string());
    }
    if max_val == 0 || max_val > 65535 {
        return Err(format!("PPM max_val {max_val} out of range"));
    }

    // Exactly one whitespace character after max_val before pixel data
    if pos < data.len() && (data[pos] == b'\n' || data[pos] == b' '
        || data[pos] == b'\t' || data[pos] == b'\r') {
        pos += 1;
    }

    Ok(PpmHeader {
        width,
        height,
        max_val: max_val as u16,
        data_offset: pos,
    })
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
