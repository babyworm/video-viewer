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

/// One named resolution entry. Single source of truth used by:
///   - filename alias lookup (parse_filename_hints)
///   - Video Size menu (show_in_menu=true)
///   - file-size-based resolution guess (show_in_menu=true used as candidate set)
pub struct NamedResolution {
    /// Display label for the Video Size menu (e.g. "1080p", "4K UHD").
    pub label: &'static str,
    /// Lowercase tokens recognised in filenames.
    pub aliases: &'static [&'static str],
    pub width: u32,
    pub height: u32,
    /// If true: shown in View → Video Size menu AND used as file-size guess candidate.
    pub show_in_menu: bool,
}

/// Canonical resolution table. Order = menu display order (small → large within group).
pub const NAMED_RESOLUTIONS: &[NamedResolution] = &[
    // CIF family
    NamedResolution { label: "SQCIF",  aliases: &["sqcif"],                    width: 128,  height: 96,   show_in_menu: true  },
    NamedResolution { label: "QCIF",   aliases: &["qcif"],                     width: 176,  height: 144,  show_in_menu: true  },
    NamedResolution { label: "SIF",    aliases: &["sif"],                      width: 352,  height: 240,  show_in_menu: true  },
    NamedResolution { label: "CIF",    aliases: &["cif"],                      width: 352,  height: 288,  show_in_menu: true  },
    NamedResolution { label: "2CIF",   aliases: &["2cif"],                     width: 704,  height: 288,  show_in_menu: true  },
    NamedResolution { label: "4CIF",   aliases: &["4cif"],                     width: 704,  height: 576,  show_in_menu: true  },
    NamedResolution { label: "16CIF",  aliases: &["16cif"],                    width: 1408, height: 1152, show_in_menu: false },
    // SD broadcast
    NamedResolution { label: "D1 NTSC",aliases: &["d1", "ntsc"],               width: 720,  height: 480,  show_in_menu: true  },
    NamedResolution { label: "D1 PAL", aliases: &["sd", "pal", "576p"],        width: 720,  height: 576,  show_in_menu: true  },
    // PC / display
    NamedResolution { label: "QQVGA",  aliases: &["qqvga"],                    width: 160,  height: 120,  show_in_menu: false },
    NamedResolution { label: "QVGA",   aliases: &["qvga", "240p"],             width: 320,  height: 240,  show_in_menu: true  },
    NamedResolution { label: "HVGA",   aliases: &["hvga"],                     width: 480,  height: 320,  show_in_menu: false },
    NamedResolution { label: "360p",   aliases: &["360p"],                     width: 640,  height: 360,  show_in_menu: false },
    NamedResolution { label: "VGA",    aliases: &["vga", "480p"],              width: 640,  height: 480,  show_in_menu: true  },
    NamedResolution { label: "WVGA",   aliases: &["wvga"],                     width: 800,  height: 480,  show_in_menu: false },
    NamedResolution { label: "SVGA",   aliases: &["svga"],                     width: 800,  height: 600,  show_in_menu: true  },
    NamedResolution { label: "XGA",    aliases: &["xga"],                      width: 1024, height: 768,  show_in_menu: true  },
    NamedResolution { label: "WXGA",   aliases: &["wxga"],                     width: 1280, height: 800,  show_in_menu: false },
    NamedResolution { label: "SXGA",   aliases: &["sxga"],                     width: 1280, height: 1024, show_in_menu: false },
    NamedResolution { label: "WSXGA+", aliases: &["wsxga"],                    width: 1680, height: 1050, show_in_menu: false },
    NamedResolution { label: "UXGA",   aliases: &["uxga"],                     width: 1600, height: 1200, show_in_menu: false },
    NamedResolution { label: "WUXGA",  aliases: &["wuxga"],                    width: 1920, height: 1200, show_in_menu: false },
    NamedResolution { label: "QXGA",   aliases: &["qxga"],                     width: 2048, height: 1536, show_in_menu: false },
    NamedResolution { label: "WQXGA",  aliases: &["wqxga"],                    width: 2560, height: 1600, show_in_menu: false },
    // HD / UHD
    NamedResolution { label: "qHD",    aliases: &["qhd"],                      width: 960,  height: 540,  show_in_menu: false },
    NamedResolution { label: "720p",   aliases: &["hd", "720p"],               width: 1280, height: 720,  show_in_menu: true  },
    NamedResolution { label: "1080p",  aliases: &["fhd", "fullhd", "1080p"],   width: 1920, height: 1080, show_in_menu: true  },
    NamedResolution { label: "1440p",  aliases: &["wqhd", "1440p", "2k"],      width: 2560, height: 1440, show_in_menu: true  },
    NamedResolution { label: "4K UHD", aliases: &["uhd", "4kuhd", "4k", "2160p"], width: 3840, height: 2160, show_in_menu: true },
    NamedResolution { label: "8K UHD", aliases: &["8k", "4320p"],              width: 7680, height: 4320, show_in_menu: false },
];

/// Build the alias → (width, height) lookup from NAMED_RESOLUTIONS.
fn named_resolutions() -> &'static HashMap<&'static str, (u32, u32)> {
    static MAP: OnceLock<HashMap<&'static str, (u32, u32)>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m = HashMap::new();
        for r in NAMED_RESOLUTIONS {
            for &alias in r.aliases {
                // Duplicate aliases indicate a table bug; last writer wins.
                m.insert(alias, (r.width, r.height));
            }
        }
        m
    })
}

/// Candidate set for file-size-based resolution guessing, sorted largest-first.
/// Larger candidates are tried first so a 4K raw file doesn't get misinterpreted
/// as a huge number of QCIF frames when frame-sizes coincidentally align.
fn size_guess_candidates() -> &'static [&'static NamedResolution] {
    static CANDIDATES: OnceLock<Vec<&'static NamedResolution>> = OnceLock::new();
    CANDIDATES
        .get_or_init(|| {
            let mut v: Vec<&'static NamedResolution> = NAMED_RESOLUTIONS
                .iter()
                .filter(|r| r.show_in_menu)
                .collect();
            v.sort_by_key(|r| std::cmp::Reverse((r.width as u64) * (r.height as u64)));
            v
        })
        .as_slice()
}

/// Result of a successful file-size-based guess.
#[derive(Debug, Clone, PartialEq)]
pub struct SizeGuess {
    pub width: u32,
    pub height: u32,
    pub format: &'static str, // e.g. "I420"
    pub num_frames: u64,
}

/// Try to recover (width, height, format) from a raw file size.
///
/// Returns the first candidate (w, h, fmt) whose per-frame size divides
/// `file_size` evenly, where candidates come from NAMED_RESOLUTIONS (menu set,
/// largest-first) crossed with the common raw formats `I420, NV12, YUYV, RGB24`.
/// This mirrors the original Python `MainWindow._guess_resolution` behaviour
/// that was lost during the v0.2.0 Rust rewrite.
pub fn guess_resolution_from_size(file_size: u64) -> Option<SizeGuess> {
    if file_size == 0 {
        return None;
    }
    const GUESS_FORMATS: &[&str] = &["I420", "NV12", "YUYV", "RGB24"];
    for cand in size_guess_candidates() {
        for &fmt_name in GUESS_FORMATS {
            let Some(fmt) = crate::core::formats::get_format_by_name(fmt_name) else {
                continue;
            };
            let fs = fmt.frame_size(cand.width, cand.height) as u64;
            if fs == 0 {
                continue;
            }
            if file_size.is_multiple_of(fs) {
                let num_frames = file_size / fs;
                if num_frames >= 1 {
                    return Some(SizeGuess {
                        width: cand.width,
                        height: cand.height,
                        format: fmt_name,
                        num_frames,
                    });
                }
            }
        }
    }
    None
}

/// Resolved open parameters for a raw file.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedParams {
    pub width: u32,
    pub height: u32,
    pub format: String,
    /// Informational message to surface to the user when a guess was applied.
    /// None when filename hints fully specified the resolution, or when falling
    /// back to defaults without a successful guess.
    pub info: Option<String>,
}

/// Decide the (width, height, format) to open a raw file with, using this priority:
///   1. Filename hints supply both width and height → use them (format from hint or default).
///   2. File-size guess succeeds → use it, attach an info message.
///   3. Defaults.
pub fn resolve_raw_params(
    path: &str,
    file_size: Option<u64>,
    default_width: u32,
    default_height: u32,
    default_format: &str,
) -> ResolvedParams {
    let hints = parse_filename_hints(path);

    // 1. Filename hints win when they specify the resolution.
    if let (Some(w), Some(h)) = (hints.width, hints.height) {
        return ResolvedParams {
            width: w,
            height: h,
            format: hints.format.unwrap_or_else(|| default_format.to_string()),
            info: None,
        };
    }

    // 2. Try file-size guess.
    if let Some(sz) = file_size {
        if let Some(g) = guess_resolution_from_size(sz) {
            // Preserve filename format hint if present; otherwise use the guess's format.
            let format = hints
                .format
                .clone()
                .unwrap_or_else(|| g.format.to_string());
            let info = Some(format!(
                "File-size guess: {}×{} {} ({} frames) — verify via Tools → Video Parameters",
                g.width, g.height, g.format, g.num_frames
            ));
            return ResolvedParams {
                width: g.width,
                height: g.height,
                format,
                info,
            };
        }
    }

    // 3. Defaults (format may still come from filename hint).
    ResolvedParams {
        width: default_width,
        height: default_height,
        format: hints.format.unwrap_or_else(|| default_format.to_string()),
        info: None,
    }
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
        m.insert("y212",        "Y212");
        m.insert("y216",        "Y216");
        m.insert("p012",        "P012");
        m.insert("nv15",        "NV15");
        m.insert("nv20",        "NV20");
        m.insert("t010",        "T010");
        m.insert("y10b",        "Greyscale (10-bit BE packed)");
        m.insert("y10bpack",    "Greyscale (10-bit BE packed)");
        m.insert("y10p",        "Greyscale (10-bit MIPI)");
        m.insert("yuv420p12le", "YUV420P12LE");
        m.insert("yuv420p12",   "YUV420P12LE");
        m.insert("yuv422p12le", "YUV422P12LE");
        m.insert("yuv422p12",   "YUV422P12LE");
        m.insert("yuv444p12le", "YUV444P12LE");
        m.insert("yuv444p12",   "YUV444P12LE");
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
