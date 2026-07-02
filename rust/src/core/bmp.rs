//! Minimal BMP reader for codec-analyzer stage images (M-C).
//!
//! Self-contained parser written against the public Windows BMP format
//! (BITMAPFILEHEADER + BITMAPINFOHEADER family), no external decoding
//! dependency. Scope is deliberately narrow — exactly what decoder-run
//! stage dumps need (observed fixtures: 24-bit BI_RGB bottom-up,
//! BITMAPINFOHEADER 40):
//!
//! - Uncompressed (`BI_RGB` = 0) only.
//! - 24-bit BGR and 32-bit BGRA/BGRX pixels (alpha ignored).
//! - Bottom-up (positive height) and top-down (negative height) row order.
//! - Rows padded to 4-byte boundaries.
//! - `BITMAPINFOHEADER` (40) and the larger V4 (108) / V5 (124) headers
//!   (identical leading layout; the extra color-space fields are ignored).
//!
//! Every read is bounds-checked; malformed input returns `Err`, never
//! panics.

use std::path::Path;

/// A decoded BMP as a tightly-packed top-down RGB8 buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BmpImage {
    pub width: u32,
    pub height: u32,
    /// `width * height * 3` bytes, row-major, top-down.
    pub rgb: Vec<u8>,
}

/// Guard against absurd dimensions (a 1 GB RGB buffer) coming from a
/// corrupt header rather than a real stage dump.
const MAX_DIM: u32 = 32768;

fn read_u16(b: &[u8], off: usize) -> Result<u16, String> {
    b.get(off..off + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| format!("bmp: truncated file (u16 at {off})"))
}

fn read_u32(b: &[u8], off: usize) -> Result<u32, String> {
    b.get(off..off + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| format!("bmp: truncated file (u32 at {off})"))
}

fn read_i32(b: &[u8], off: usize) -> Result<i32, String> {
    Ok(read_u32(b, off)? as i32)
}

/// Parse a BMP from raw bytes. See the module docs for the supported subset.
pub fn parse_bmp(bytes: &[u8]) -> Result<BmpImage, String> {
    if bytes.len() < 2 || &bytes[0..2] != b"BM" {
        return Err("bmp: missing 'BM' signature".to_string());
    }
    // BITMAPFILEHEADER: sig(2) size(4) reserved(4) dataOffset(4) = 14 bytes.
    let data_off = read_u32(bytes, 10)? as usize;
    // DIB header size selects the header family.
    let hdr_size = read_u32(bytes, 14)?;
    if !matches!(hdr_size, 40 | 52 | 56 | 108 | 124) {
        return Err(format!(
            "bmp: unsupported DIB header size {hdr_size} (need BITMAPINFOHEADER family)"
        ));
    }
    let width_raw = read_i32(bytes, 18)?;
    let height_raw = read_i32(bytes, 22)?;
    let planes = read_u16(bytes, 26)?;
    let bpp = read_u16(bytes, 28)?;
    let compression = read_u32(bytes, 30)?;

    if planes != 1 {
        return Err(format!("bmp: planes must be 1, got {planes}"));
    }
    if compression != 0 {
        return Err(format!(
            "bmp: only uncompressed BI_RGB supported, got compression {compression}"
        ));
    }
    if bpp != 24 && bpp != 32 {
        return Err(format!("bmp: only 24/32-bit supported, got {bpp}-bit"));
    }
    if width_raw <= 0 || width_raw as u32 > MAX_DIM {
        return Err(format!("bmp: invalid width {width_raw}"));
    }
    // Negative height = top-down row order.
    let top_down = height_raw < 0;
    let height_abs = height_raw.unsigned_abs();
    if height_abs == 0 || height_abs > MAX_DIM {
        return Err(format!("bmp: invalid height {height_raw}"));
    }
    let (w, h) = (width_raw as u32, height_abs);

    let bytes_pp = (bpp / 8) as usize;
    // Rows are padded to 4-byte boundaries.
    let stride = ((w as usize * bytes_pp) + 3) & !3;
    let need = data_off
        .checked_add(stride.checked_mul(h as usize).ok_or("bmp: size overflow")?)
        .ok_or("bmp: size overflow")?;
    if bytes.len() < need {
        return Err(format!(
            "bmp: pixel data truncated (need {need} bytes, file has {})",
            bytes.len()
        ));
    }

    let mut rgb = vec![0u8; w as usize * h as usize * 3];
    for row in 0..h as usize {
        // Bottom-up files store the last image row first.
        let src_row = if top_down { row } else { h as usize - 1 - row };
        let src = data_off + src_row * stride;
        let dst = row * w as usize * 3;
        for x in 0..w as usize {
            let p = src + x * bytes_pp;
            // BMP stores BGR(A).
            rgb[dst + x * 3] = bytes[p + 2];
            rgb[dst + x * 3 + 1] = bytes[p + 1];
            rgb[dst + x * 3 + 2] = bytes[p];
        }
    }
    Ok(BmpImage { width: w, height: h, rgb })
}

/// Load and parse a BMP file.
pub fn load_bmp<P: AsRef<Path>>(path: P) -> Result<BmpImage, String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .map_err(|e| format!("bmp: cannot read {}: {e}", path.display()))?;
    parse_bmp(&bytes).map_err(|e| format!("{e} ({})", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic BMP: `rows` is top-down RGB rows; storage follows
    /// `top_down` (negative height when true).
    fn make_bmp(rows: &[Vec<[u8; 3]>], bpp: u16, top_down: bool) -> Vec<u8> {
        let h = rows.len() as u32;
        let w = rows[0].len() as u32;
        let bytes_pp = (bpp / 8) as usize;
        let stride = ((w as usize * bytes_pp) + 3) & !3;
        let data_off = 54u32;
        let size = data_off as usize + stride * h as usize;
        let mut b = Vec::with_capacity(size);
        b.extend_from_slice(b"BM");
        b.extend_from_slice(&(size as u32).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&data_off.to_le_bytes());
        b.extend_from_slice(&40u32.to_le_bytes()); // BITMAPINFOHEADER
        b.extend_from_slice(&(w as i32).to_le_bytes());
        let h_field = if top_down { -(h as i32) } else { h as i32 };
        b.extend_from_slice(&h_field.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes()); // planes
        b.extend_from_slice(&bpp.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB
        b.extend_from_slice(&[0u8; 20]); // sizeImage..clrImportant
        assert_eq!(b.len(), 54);
        // Storage order: bottom-up unless top_down.
        let order: Vec<usize> = if top_down {
            (0..h as usize).collect()
        } else {
            (0..h as usize).rev().collect()
        };
        for r in order {
            let mut row = Vec::with_capacity(stride);
            for px in &rows[r] {
                row.push(px[2]); // B
                row.push(px[1]); // G
                row.push(px[0]); // R
                if bytes_pp == 4 {
                    row.push(0xEE); // padding/alpha byte, must be ignored
                }
            }
            while row.len() < stride {
                row.push(0xAB); // row padding, must be ignored
            }
            b.extend_from_slice(&row);
        }
        b
    }

    #[test]
    fn parse_24bit_bottom_up_with_padding() {
        // 2x2, stride = 6 → padded to 8: exercises both row order and pad.
        let rows = vec![
            vec![[255, 0, 0], [0, 255, 0]],
            vec![[0, 0, 255], [10, 20, 30]],
        ];
        let img = parse_bmp(&make_bmp(&rows, 24, false)).unwrap();
        assert_eq!((img.width, img.height), (2, 2));
        assert_eq!(&img.rgb[0..6], &[255, 0, 0, 0, 255, 0]);
        assert_eq!(&img.rgb[6..12], &[0, 0, 255, 10, 20, 30]);
    }

    #[test]
    fn parse_24bit_top_down() {
        let rows = vec![
            vec![[1, 2, 3], [4, 5, 6]],
            vec![[7, 8, 9], [10, 11, 12]],
        ];
        let img = parse_bmp(&make_bmp(&rows, 24, true)).unwrap();
        assert_eq!(&img.rgb[0..6], &[1, 2, 3, 4, 5, 6]);
        assert_eq!(&img.rgb[6..12], &[7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn parse_32bit() {
        let rows = vec![vec![[200, 100, 50], [1, 2, 3], [4, 5, 6]]];
        let img = parse_bmp(&make_bmp(&rows, 32, false)).unwrap();
        assert_eq!((img.width, img.height), (3, 1));
        assert_eq!(&img.rgb, &[200, 100, 50, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn truncated_pixel_data_errors() {
        let rows = vec![vec![[1, 2, 3], [4, 5, 6]]];
        let mut b = make_bmp(&rows, 24, false);
        b.truncate(b.len() - 3);
        let err = parse_bmp(&b).unwrap_err();
        assert!(err.contains("truncated"), "{err}");
    }

    #[test]
    fn bad_signature_and_short_header_error() {
        assert!(parse_bmp(b"PNG").is_err());
        assert!(parse_bmp(b"BM\x00\x00").is_err()); // header cut off
        assert!(parse_bmp(&[]).is_err());
    }

    #[test]
    fn unsupported_variants_error() {
        let rows = vec![vec![[1, 2, 3]]];
        // 8-bit palette: not produced by decoder-run stage dumps.
        let mut b = make_bmp(&rows, 24, false);
        b[28] = 8;
        assert!(parse_bmp(&b).unwrap_err().contains("24/32-bit"));
        // RLE compression.
        let mut b = make_bmp(&rows, 24, false);
        b[30] = 1;
        assert!(parse_bmp(&b).unwrap_err().contains("uncompressed"));
        // Zero width.
        let mut b = make_bmp(&rows, 24, false);
        b[18..22].copy_from_slice(&0i32.to_le_bytes());
        assert!(parse_bmp(&b).is_err());
    }
}
