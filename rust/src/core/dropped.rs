//! Dropped-file classification (0.15.0).
//!
//! Routes files dragged onto a viewer window: `.catb` decoder telemetry
//! loads the bitstream-analysis layer, raw Annex-B bitstreams go to the
//! external decoder-run launcher, and everything else follows the normal
//! video-open path unchanged. The pure decision logic lives in
//! [`classify_from_parts`] so unit tests need no file I/O.

use std::io::Read;
use std::path::Path;

use super::catb::CATB_MAGIC;

/// How many leading bytes are examined (magic = 8, Annex-B sniff = 32).
const HEAD_LEN: usize = 32;

/// Routing decision for a file dropped onto any viewer window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DroppedKind {
    /// `.catb` decoder telemetry → `load_bitstream` + analysis window.
    Telemetry,
    /// Raw Annex-B bitstream → external decoder-run launcher.
    Bitstream,
    /// Anything else → the existing video-open path.
    Other,
}

/// Classify a dropped file by content magic, extension, and (for
/// extensionless / `.bin` files only) an Annex-B start-code sniff.
///
/// Open/read failures return [`DroppedKind::Other`] so the existing
/// video-open path surfaces the error to the user.
pub fn classify_dropped_file(path: &Path) -> DroppedKind {
    let mut head = [0u8; HEAD_LEN];
    let mut len = 0usize;
    match std::fs::File::open(path) {
        Ok(mut f) => loop {
            match f.read(&mut head[len..]) {
                Ok(0) => break,
                Ok(n) => {
                    len += n;
                    if len == HEAD_LEN {
                        break;
                    }
                }
                Err(_) => return DroppedKind::Other,
            }
        },
        Err(_) => return DroppedKind::Other,
    }
    classify_from_parts(path, &head[..len])
}

/// Pure classification from the path and the file's first bytes
/// (up to [`HEAD_LEN`]).
///
/// Priority order:
/// 1. `CATB0001` magic → Telemetry (regardless of extension)
/// 2. `.catb` extension → Telemetry
/// 3. `.h265` / `.hevc` / `.h264` / `.264` / `.265` → Bitstream
/// 4. Extensionless or `.bin`: Annex-B start code (`00 00 00 01` or
///    `00 00 01` — the 3-byte window matches both) anywhere in the first
///    32 bytes → Bitstream. Known video extensions (`.yuv`, `.raw`, …)
///    are never sniffed: a raw frame can legitimately start with zeros
///    and those files already have a well-known route.
/// 5. Everything else → Other (existing video-open path).
pub fn classify_from_parts(path: &Path, head: &[u8]) -> DroppedKind {
    if head.len() >= CATB_MAGIC.len() && &head[..CATB_MAGIC.len()] == CATB_MAGIC {
        return DroppedKind::Telemetry;
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    match ext.as_deref() {
        Some("catb") => DroppedKind::Telemetry,
        Some("h265" | "hevc" | "h264" | "264" | "265") => DroppedKind::Bitstream,
        None | Some("bin") => {
            if head.windows(3).any(|w| w == [0x00, 0x00, 0x01]) {
                DroppedKind::Bitstream
            } else {
                DroppedKind::Other
            }
        }
        _ => DroppedKind::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str, bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        (dir, path)
    }

    #[test]
    fn test_dropped_magic_beats_wrong_extension() {
        // CATB magic wins even with a video extension.
        let (_d, p) = tmp("frames.yuv", b"CATB0001rest-of-header");
        assert_eq!(classify_dropped_file(&p), DroppedKind::Telemetry);
    }

    #[test]
    fn test_dropped_catb_extension_without_magic() {
        let (_d, p) = tmp("telemetry.catb", b"not-the-magic");
        assert_eq!(classify_dropped_file(&p), DroppedKind::Telemetry);
    }

    #[test]
    fn test_dropped_bitstream_extensions() {
        for name in [
            "a.h265", "a.hevc", "a.h264", "a.264", "a.265", "A.H265", "A.HEVC",
        ] {
            let (_d, p) = tmp(name, &[0x00, 0x00, 0x00, 0x01, 0x40]);
            assert_eq!(classify_dropped_file(&p), DroppedKind::Bitstream, "{name}");
        }
        // Extension alone is enough — no content requirement.
        let (_d, p) = tmp("noheader.h265", b"xxxx");
        assert_eq!(classify_dropped_file(&p), DroppedKind::Bitstream);
    }

    #[test]
    fn test_dropped_bin_sniffs_annexb() {
        // 4-byte start code at offset 0.
        let (_d, p) = tmp("stream.bin", &[0x00, 0x00, 0x00, 0x01, 0x40, 0x01]);
        assert_eq!(classify_dropped_file(&p), DroppedKind::Bitstream);
        // 3-byte start code.
        let (_d, p) = tmp("stream.bin", &[0x00, 0x00, 0x01, 0x40]);
        assert_eq!(classify_dropped_file(&p), DroppedKind::Bitstream);
        // Start code after leading_zero_8bits padding.
        let mut padded = vec![0u8; 8];
        padded.extend_from_slice(&[0x01, 0x40, 0x01]);
        let (_d, p) = tmp("stream.bin", &padded);
        assert_eq!(classify_dropped_file(&p), DroppedKind::Bitstream);
        // No start code → Other.
        let (_d, p) = tmp("data.bin", &[0x10; 32]);
        assert_eq!(classify_dropped_file(&p), DroppedKind::Other);
        // All zeros never form 00 00 01.
        let (_d, p) = tmp("black.bin", &[0x00; 32]);
        assert_eq!(classify_dropped_file(&p), DroppedKind::Other);
    }

    #[test]
    fn test_dropped_extensionless_sniffs_annexb() {
        let (_d, p) = tmp("bitstream", &[0x00, 0x00, 0x00, 0x01, 0x42]);
        assert_eq!(classify_dropped_file(&p), DroppedKind::Bitstream);
        let (_d, p) = tmp("notes", b"hello world");
        assert_eq!(classify_dropped_file(&p), DroppedKind::Other);
    }

    #[test]
    fn test_dropped_known_video_extension_never_sniffed() {
        // A .yuv starting with a start-code-like pattern stays a video.
        let (_d, p) = tmp("clip.yuv", &[0x00, 0x00, 0x00, 0x01, 0x10, 0x10]);
        assert_eq!(classify_dropped_file(&p), DroppedKind::Other);
        let (_d, p) = tmp("clip.raw", &[0x00, 0x00, 0x01, 0x10]);
        assert_eq!(classify_dropped_file(&p), DroppedKind::Other);
    }

    #[test]
    fn test_dropped_short_and_empty_files() {
        let (_d, p) = tmp("empty.bin", b"");
        assert_eq!(classify_dropped_file(&p), DroppedKind::Other);
        let (_d, p) = tmp("tiny", &[0x00, 0x00]);
        assert_eq!(classify_dropped_file(&p), DroppedKind::Other);
        // Shorter than the magic but .catb extension still routes.
        let (_d, p) = tmp("short.catb", b"CA");
        assert_eq!(classify_dropped_file(&p), DroppedKind::Telemetry);
    }

    #[test]
    fn test_dropped_unreadable_is_other() {
        // Nonexistent path → open fails → Other.
        let p = Path::new("/nonexistent/definitely/missing.h265");
        assert_eq!(classify_dropped_file(p), DroppedKind::Other);
    }

    #[test]
    fn test_classify_from_parts_pure() {
        assert_eq!(
            classify_from_parts(Path::new("x.mkv"), b"CATB0001"),
            DroppedKind::Telemetry
        );
        assert_eq!(
            classify_from_parts(Path::new("x.catb"), b""),
            DroppedKind::Telemetry
        );
        assert_eq!(
            classify_from_parts(Path::new("x.264"), b""),
            DroppedKind::Bitstream
        );
        assert_eq!(
            classify_from_parts(Path::new("x.bin"), &[0, 0, 1]),
            DroppedKind::Bitstream
        );
        assert_eq!(
            classify_from_parts(Path::new("x.yuv"), &[0, 0, 1]),
            DroppedKind::Other
        );
        assert_eq!(
            classify_from_parts(Path::new("x.txt"), b"plain"),
            DroppedKind::Other
        );
    }
}
