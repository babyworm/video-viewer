use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

/// Metadata extracted from a filename.
#[derive(Debug, Default, PartialEq)]
pub struct FilenameHints {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub format: Option<String>,
    pub fps: Option<f64>,
    pub bit_depth: Option<u32>,
}

fn named_resolutions() -> &'static HashMap<&'static str, (u32, u32)> {
    static MAP: OnceLock<HashMap<&'static str, (u32, u32)>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("qcif",  (176,  144));
        m.insert("cif",   (352,  288));
        m.insert("qvga",  (320,  240));
        m.insert("vga",   (640,  480));
        m.insert("wvga",  (800,  480));
        m.insert("svga",  (800,  600));
        m.insert("xga",   (1024, 768));
        m.insert("hd",    (1280, 720));
        m.insert("720p",  (1280, 720));
        m.insert("1080p", (1920, 1080));
        m.insert("2k",    (2560, 1440));
        m.insert("4k",    (3840, 2160));
        m.insert("sd",    (720,  576));
        m.insert("pal",   (720,  576));
        m.insert("ntsc",  (720,  480));
        m
    })
}

fn format_aliases() -> &'static HashMap<&'static str, &'static str> {
    static MAP: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("i420",    "I420");
        m.insert("yuv420p", "I420");
        m.insert("yuv420",  "I420");
        m.insert("yv12",    "YV12");
        m.insert("nv12",    "NV12");
        m.insert("nv21",    "NV21");
        m.insert("nv16",    "NV16");
        m.insert("nv61",    "NV61");
        m.insert("yuv422p", "YUV422P");
        m.insert("yuv422",  "YUV422P");
        m.insert("yuv444p", "YUV444P");
        m.insert("yuv444",  "YUV444P");
        m.insert("yuyv",    "YUYV");
        m.insert("yuy2",    "YUYV");
        m.insert("uyvy",    "UYVY");
        m.insert("yvyu",    "YVYU");
        m.insert("rgb24",   "RGB24");
        m.insert("rgb",     "RGB24");
        m.insert("bgr24",   "BGR24");
        m.insert("bgr",     "BGR24");
        m.insert("grey",    "Greyscale (8-bit)");
        m.insert("gray",    "Greyscale (8-bit)");
        // 10-bit formats
        m.insert("yuv420p10le", "YUV420P10LE");
        m.insert("yuv420p10",   "YUV420P10LE");
        m.insert("yuv422p10le", "YUV422P10LE");
        m.insert("yuv422p10",   "YUV422P10LE");
        m.insert("yuv444p10le", "YUV444P10LE");
        m.insert("yuv444p10",   "YUV444P10LE");
        m.insert("p010",        "P010");
        m.insert("p016",        "P016");
        m.insert("p210",        "P210");
        m.insert("y210",        "Y210");
        m
    })
}

/// Split a string on `_`, `.`, `-`, `/`, `\`.
fn split_tokens(s: &str) -> Vec<&str> {
    s.split(['_', '.', '-', '/', '\\'])
        .filter(|t| !t.is_empty())
        .collect()
}

/// Try to parse an explicit `WxH` pattern from `s`.
/// Returns `Some((w, h))` if found and both values are in [16, 8192].
fn parse_wxh(s: &str) -> Option<(u32, u32)> {
    // Scan for 'x' or 'X' separating two digit runs with boundary conditions.
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        // Find next 'x' or 'X'
        if bytes[i] != b'x' && bytes[i] != b'X' {
            i += 1;
            continue;
        }
        let x_pos = i;
        // Look backwards for digit run
        if x_pos == 0 || !bytes[x_pos - 1].is_ascii_digit() {
            i += 1;
            continue;
        }
        let w_end = x_pos;
        let mut w_start = w_end;
        while w_start > 0 && bytes[w_start - 1].is_ascii_digit() {
            w_start -= 1;
        }
        // Boundary before width: start of string or separator
        let boundary_before = w_start == 0
            || matches!(bytes[w_start - 1], b'_' | b'.' | b'-' | b'/' | b'\\');
        if !boundary_before {
            i += 1;
            continue;
        }
        // Look forwards for digit run
        let h_start = x_pos + 1;
        if h_start >= len || !bytes[h_start].is_ascii_digit() {
            i += 1;
            continue;
        }
        let mut h_end = h_start;
        while h_end < len && bytes[h_end].is_ascii_digit() {
            h_end += 1;
        }
        // Boundary after height: end of string or separator
        let boundary_after = h_end == len
            || matches!(bytes[h_end], b'_' | b'.' | b'-' | b'/' | b'\\');
        if !boundary_after {
            i += 1;
            continue;
        }
        let w_str = &s[w_start..w_end];
        let h_str = &s[h_start..h_end];
        if let (Ok(w), Ok(h)) = (w_str.parse::<u32>(), h_str.parse::<u32>()) {
            if (16..=8192).contains(&w) && (16..=8192).contains(&h) {
                return Some((w, h));
            }
        }
        i += 1;
    }
    None
}

/// Parse metadata hints from a filename (may be a full path).
pub fn parse_filename_hints(filename: &str) -> FilenameHints {
    let mut hints = FilenameHints::default();

    // Basename (case-insensitive matching)
    let basename = Path::new(filename)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(filename);
    let basename_lower = basename.to_lowercase();
    // Strip extension
    let name_no_ext = match basename_lower.rfind('.') {
        Some(pos) => &basename_lower[..pos],
        None => &basename_lower,
    };

    // --- Resolution ---
    // 1. Explicit WxH in basename (use original-case basename for x/X matching)
    let basename_no_ext = match basename.rfind('.') {
        Some(pos) => &basename[..pos],
        None => basename,
    };
    if let Some((w, h)) = parse_wxh(basename_no_ext) {
        hints.width = Some(w);
        hints.height = Some(h);
    }

    // 1b. Fallback: WxH anywhere in the full path
    if hints.width.is_none() {
        if let Some((w, h)) = parse_wxh(filename) {
            hints.width = Some(w);
            hints.height = Some(h);
        }
    }

    // 2. Named resolution (only if no explicit WxH)
    if hints.width.is_none() {
        let tokens = split_tokens(name_no_ext);
        let resolutions = named_resolutions();
        for token in &tokens {
            if let Some(&(w, h)) = resolutions.get(*token) {
                hints.width = Some(w);
                hints.height = Some(h);
                break;
            }
        }
    }

    // --- Format ---
    let tokens = split_tokens(name_no_ext);
    let aliases = format_aliases();
    for token in &tokens {
        if let Some(&fmt) = aliases.get(*token) {
            hints.format = Some(fmt.to_string());
            break;
        }
    }

    // --- FPS ---
    // 1. Explicit "Nfps" pattern (case-insensitive)
    let fps_lower = name_no_ext.to_lowercase();
    if let Some(fps) = parse_fps_suffix(&fps_lower) {
        hints.fps = Some(fps);
    } else {
        // 2. Bare number between separators matching common fps values
        if let Some(fps) = parse_fps_bare(name_no_ext) {
            hints.fps = Some(fps);
        }
    }

    // --- Bit depth ---
    if let Some(bd) = parse_bit_depth(name_no_ext) {
        hints.bit_depth = Some(bd);
    }

    hints
}

/// Parse "Nfps" or "N.Nfps" from the string. Returns fps if 0.1 <= fps <= 240.
fn parse_fps_suffix(s: &str) -> Option<f64> {
    // Find "fps" (already lowercased) and look backwards for digits/dot
    let mut search = s;
    while let Some(pos) = search.find("fps") {
        let before = &search[..pos];
        // Extract trailing number from `before`
        let num_start = before
            .rfind(|c: char| !c.is_ascii_digit() && c != '.')
            .map(|p| p + 1)
            .unwrap_or(0);
        let num_str = &before[num_start..];
        if !num_str.is_empty() {
            if let Ok(val) = num_str.parse::<f64>() {
                if (0.1..=240.0).contains(&val) {
                    return Some(val);
                }
            }
        }
        // Advance past this "fps"
        search = &search[pos + 3..];
    }
    None
}

/// Parse bare fps values between separators matching common fps set.
fn parse_fps_bare(s: &str) -> Option<f64> {
    const COMMON: &[f64] = &[1.0, 5.0, 10.0, 15.0, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0, 120.0];
    let tokens = split_tokens(s);
    for token in tokens {
        if let Ok(val) = token.parse::<f64>() {
            if COMMON.iter().any(|&c| (c - val).abs() < 1e-9) {
                return Some(val);
            }
        }
    }
    None
}

/// Parse "Nbit" or "N.Nbit" from the string. Returns bit depth if in {8,10,12,16}.
fn parse_bit_depth(s: &str) -> Option<u32> {
    let lower = s.to_lowercase();
    let mut search = lower.as_str();
    while let Some(pos) = search.find("bit") {
        // Boundary after "bit": end or separator
        let after_pos = pos + 3;
        let boundary_after = after_pos == search.len()
            || matches!(search.as_bytes()[after_pos], b'_' | b'.' | b'-' | b'/' | b'\\');
        if boundary_after {
            let before = &search[..pos];
            // Boundary before digits
            let num_start = before
                .rfind(|c: char| !c.is_ascii_digit())
                .map(|p| p + 1)
                .unwrap_or(0);
            // Check that char before digits is a separator or start
            let boundary_before = num_start == 0
                || matches!(before.as_bytes()[num_start - 1], b'_' | b'.' | b'-' | b'/' | b'\\');
            if boundary_before {
                let num_str = &before[num_start..];
                if !num_str.is_empty() {
                    if let Ok(bd) = num_str.parse::<u32>() {
                        if matches!(bd, 8 | 10 | 12 | 16) {
                            return Some(bd);
                        }
                    }
                }
            }
        }
        search = &search[pos + 3..];
    }
    None
}
