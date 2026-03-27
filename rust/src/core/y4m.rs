/// Y4M (YUV4MPEG2) file header parser.

#[derive(Debug, Clone, PartialEq)]
pub struct Y4mHeader {
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub colorspace: String,
    pub interlace: String,
    pub pixel_aspect: (u32, u32),
}

impl Y4mHeader {
    /// Return the frame rate as a floating-point value.
    pub fn fps(&self) -> f64 {
        self.fps_num as f64 / self.fps_den as f64
    }

    /// Map Y4M colorspace token to a display format name.
    ///
    /// Y4M colorspace tokens: "420", "420p10", "420p12", "422", "422p10", "444", "444p10", etc.
    pub fn to_format_name(&self) -> &str {
        let cs = self.colorspace.as_str();
        match cs {
            // 10-bit variants (e.g. "420p10", "420mpeg2p10")
            _ if cs.contains("420") && cs.contains("p10") => "YUV420P10LE",
            _ if cs.contains("422") && cs.contains("p10") => "YUV422P10LE",
            _ if cs.contains("444") && cs.contains("p10") => "YUV444P10LE",
            // 8-bit variants
            _ if cs.starts_with("420") => "I420",
            _ if cs.starts_with("422") => "YUV422P",
            _ if cs.starts_with("444") => "YUV444P",
            _ if cs.starts_with("mono") => "Greyscale (8-bit)",
            _ => "I420",
        }
    }
}

/// Parse a Y4M file header from the leading bytes of a file.
///
/// The header is the first line of the file (terminated by `\n`).
/// Returns `Err` if the magic "YUV4MPEG2" is not found or the data
/// contains no newline.
pub fn parse_y4m_header(data: &[u8]) -> Result<Y4mHeader, String> {
    // Find the header line (terminated by '\n')
    let newline_pos = data
        .iter()
        .position(|&b| b == b'\n')
        .ok_or_else(|| "No newline found in Y4M header".to_string())?;

    let header_str = std::str::from_utf8(&data[..newline_pos])
        .map_err(|e| format!("Y4M header is not valid UTF-8: {e}"))?;

    let parts: Vec<&str> = header_str.split(' ').collect();
    if parts.is_empty() || parts[0] != "YUV4MPEG2" {
        return Err(format!(
            "Invalid Y4M magic: expected 'YUV4MPEG2', got '{}'",
            parts.first().copied().unwrap_or("")
        ));
    }

    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;
    let mut fps_num: u32 = 30;
    let mut fps_den: u32 = 1;
    let mut colorspace = "420".to_string();
    let mut interlace = "progressive".to_string();
    let mut pixel_aspect: (u32, u32) = (1, 1);

    for part in &parts[1..] {
        if part.is_empty() {
            continue;
        }
        let (tag, value) = part.split_at(1);
        match tag {
            "W" => {
                width = value
                    .parse::<u32>()
                    .map_err(|e| format!("Invalid Y4M width '{value}': {e}"))
                    .ok();
            }
            "H" => {
                height = value
                    .parse::<u32>()
                    .map_err(|e| format!("Invalid Y4M height '{value}': {e}"))
                    .ok();
            }
            "F" => {
                if let Some((num_str, den_str)) = value.split_once(':') {
                    if let (Ok(n), Ok(d)) = (num_str.parse::<u32>(), den_str.parse::<u32>()) {
                        if d != 0 {
                            fps_num = n;
                            fps_den = d;
                        }
                    }
                }
            }
            "C" => {
                colorspace = value.to_string();
            }
            "I" => {
                interlace = match value.chars().next() {
                    Some('p') => "progressive".to_string(),
                    Some('t') => "tff".to_string(),
                    Some('b') => "bff".to_string(),
                    Some('m') => "mixed".to_string(),
                    _ => "progressive".to_string(),
                };
            }
            "A" => {
                if let Some((num_str, den_str)) = value.split_once(':') {
                    if let (Ok(n), Ok(d)) = (num_str.parse::<u32>(), den_str.parse::<u32>()) {
                        pixel_aspect = (n, d);
                    }
                }
            }
            _ => {
                // Unknown / extension tag — ignore
            }
        }
    }

    let width = width.ok_or_else(|| "Y4M header missing width (W tag)".to_string())?;
    let height = height.ok_or_else(|| "Y4M header missing height (H tag)".to_string())?;

    Ok(Y4mHeader {
        width,
        height,
        fps_num,
        fps_den,
        colorspace,
        interlace,
        pixel_aspect,
    })
}

/// Build a list of byte offsets pointing to the start of each frame's
/// pixel data within `data`.
///
/// Each Y4M frame is preceded by a line starting with "FRAME" and
/// terminated by `\n`.  The offset recorded is the byte immediately
/// after that `\n`.
///
/// `frame_size` is the expected number of raw pixel bytes per frame.
/// The scan starts at the position of the first "FRAME" marker (i.e.
/// the byte immediately after the file header newline).
pub fn build_frame_offsets(data: &[u8], frame_size: usize) -> Vec<usize> {
    let mut offsets = Vec::new();

    // Skip the file header (first line).
    let header_end = match data.iter().position(|&b| b == b'\n') {
        Some(pos) => pos + 1,
        None => return offsets,
    };

    let mut offset = header_end;
    while offset < data.len() {
        // Each frame starts with "FRAME..." terminated by '\n'.
        if !data[offset..].starts_with(b"FRAME") {
            break;
        }
        let newline_pos = match data[offset..].iter().position(|&b| b == b'\n') {
            Some(pos) => offset + pos,
            None => break,
        };
        let data_offset = newline_pos + 1;
        offsets.push(data_offset);
        offset = data_offset + frame_size;
    }

    offsets
}
