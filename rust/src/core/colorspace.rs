//! Pure Rust color space conversions.

/// YUV to RGB conversion using BT.601 or BT.709 coefficients.
#[inline]
fn yuv_to_rgb_pixel(y: u8, u: u8, v: u8, bt709: bool) -> (u8, u8, u8) {
    let y_val = y as f32;
    let u_val = u as f32 - 128.0;
    let v_val = v as f32 - 128.0;

    let (r, g, b) = if bt709 {
        (
            y_val + 1.5748 * v_val,
            y_val - 0.1873 * u_val - 0.4681 * v_val,
            y_val + 1.8556 * u_val,
        )
    } else {
        // BT.601
        (
            y_val + 1.402 * v_val,
            y_val - 0.344136 * u_val - 0.714136 * v_val,
            y_val + 1.772 * u_val,
        )
    };

    (
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    )
}

/// Convert YUV planar data to RGB24.
///
/// Handles I420 (h_sub=2, v_sub=2), YV12 (caller swaps u/v), 422P (h_sub=2, v_sub=1),
/// 444P (h_sub=1, v_sub=1), etc.
pub fn yuv_to_rgb_planar(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    w: usize,
    h: usize,
    subsampling: (usize, usize),
    bt709: bool,
) -> Vec<u8> {
    let (h_sub, v_sub) = subsampling;
    let uv_w = w / h_sub;
    let mut rgb = vec![0u8; w * h * 3];

    for py in 0..h {
        for px in 0..w {
            let y_val = y_plane[py * w + px];
            let cx = px / h_sub;
            let cy = py / v_sub;
            let uv_idx = cy * uv_w + cx;
            let (r, g, b) = yuv_to_rgb_pixel(y_val, u_plane[uv_idx], v_plane[uv_idx], bt709);

            let out = (py * w + px) * 3;
            rgb[out] = r;
            rgb[out + 1] = g;
            rgb[out + 2] = b;
        }
    }

    rgb
}

/// Convert YUV 4:2:0 semi-planar (NV12/NV21) to RGB24.
///
/// Layout: Y plane (w*h) followed by interleaved UV or VU pairs ((w/2)*(h/2)*2 bytes).
/// `uv_swapped`: false = NV12 (U,V order), true = NV21 (V,U order).
pub fn yuv_to_rgb_semi_planar(
    raw: &[u8],
    w: usize,
    h: usize,
    uv_swapped: bool,
    bt709: bool,
) -> Vec<u8> {
    let y_size = w * h;
    let uv_w = w / 2;
    let mut rgb = vec![0u8; w * h * 3];

    for py in 0..h {
        for px in 0..w {
            let y_val = raw[py * w + px];
            let cx = px / 2;
            let cy = py / 2;
            let uv_base = y_size + (cy * uv_w + cx) * 2;
            let (u_val, v_val) = if uv_swapped {
                (raw[uv_base + 1], raw[uv_base])
            } else {
                (raw[uv_base], raw[uv_base + 1])
            };
            let (r, g, b) = yuv_to_rgb_pixel(y_val, u_val, v_val, bt709);

            let out = (py * w + px) * 3;
            rgb[out] = r;
            rgb[out + 1] = g;
            rgb[out + 2] = b;
        }
    }

    rgb
}

/// Convert YUYV (4:2:2 packed) to RGB24.
///
/// Layout: [Y0, U, Y1, V] per 2 pixels.
pub fn yuv_to_rgb_yuyv(raw: &[u8], w: usize, h: usize, bt709: bool) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];

    for py in 0..h {
        for px_pair in 0..(w / 2) {
            let base = (py * w + px_pair * 2) * 2;
            let y0 = raw[base];
            let u = raw[base + 1];
            let y1 = raw[base + 2];
            let v = raw[base + 3];

            let (r0, g0, b0) = yuv_to_rgb_pixel(y0, u, v, bt709);
            let (r1, g1, b1) = yuv_to_rgb_pixel(y1, u, v, bt709);

            let out0 = (py * w + px_pair * 2) * 3;
            rgb[out0] = r0;
            rgb[out0 + 1] = g0;
            rgb[out0 + 2] = b0;

            let out1 = out0 + 3;
            rgb[out1] = r1;
            rgb[out1 + 1] = g1;
            rgb[out1 + 2] = b1;
        }
    }

    rgb
}

/// Convert UYVY (4:2:2 packed) to RGB24.
///
/// Layout: [U, Y0, V, Y1] per 2 pixels.
pub fn yuv_to_rgb_uyvy(raw: &[u8], w: usize, h: usize, bt709: bool) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];

    for py in 0..h {
        for px_pair in 0..(w / 2) {
            let base = (py * w + px_pair * 2) * 2;
            let u = raw[base];
            let y0 = raw[base + 1];
            let v = raw[base + 2];
            let y1 = raw[base + 3];

            let (r0, g0, b0) = yuv_to_rgb_pixel(y0, u, v, bt709);
            let (r1, g1, b1) = yuv_to_rgb_pixel(y1, u, v, bt709);

            let out0 = (py * w + px_pair * 2) * 3;
            rgb[out0] = r0;
            rgb[out0 + 1] = g0;
            rgb[out0 + 2] = b0;

            let out1 = out0 + 3;
            rgb[out1] = r1;
            rgb[out1 + 1] = g1;
            rgb[out1 + 2] = b1;
        }
    }

    rgb
}

/// Convert BGR24 to RGB24 (swap R and B channels).
pub fn bgr_to_rgb(raw: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for i in 0..(w * h) {
        let s = i * 3;
        rgb[s] = raw[s + 2];     // R = B_src
        rgb[s + 1] = raw[s + 1]; // G
        rgb[s + 2] = raw[s];     // B = R_src
    }
    rgb
}

/// Convert RGB24 to BGR24 (swap R and B channels).
pub fn rgb_to_bgr(raw: &[u8], w: usize, h: usize) -> Vec<u8> {
    bgr_to_rgb(raw, w, h) // same operation — swap is symmetric
}

/// Convert greyscale 8-bit to RGB24 (replicate to all channels).
pub fn grey_to_rgb(raw: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for (i, &v) in raw.iter().enumerate().take(w * h) {
        let o = i * 3;
        rgb[o] = v;
        rgb[o + 1] = v;
        rgb[o + 2] = v;
    }
    rgb
}

/// Read a 16-bit LE sample (MSB-aligned) and convert to 8-bit.
/// For MSB-aligned formats (P010, P210, Y210): data sits in the high bits.
/// Combined shift: (16 - bit_depth) + (bit_depth - 8) = 8.
#[inline]
fn read_u16_msb_to_u8(data: &[u8], offset: usize, _bit_depth: u32) -> u8 {
    let val = u16::from_le_bytes([data[offset], data[offset + 1]]);
    (val >> 8) as u8
}

/// Read a 16-bit LE sample (LSB-aligned: data in low bits) and convert to 8-bit.
#[inline]
fn read_u16_lsb_to_u8(data: &[u8], offset: usize, bit_depth: u32) -> u8 {
    let val = u16::from_le_bytes([data[offset], data[offset + 1]]);
    let mask = (1u16 << bit_depth) - 1;
    ((val & mask) >> (bit_depth - 8)) as u8
}

/// Convert high-bit-depth planar YUV (LSB-aligned, e.g. yuv420p10le) to RGB24.
///
/// Each Y/U/V sample is stored as a u16 LE word with data in the low bits.
pub fn yuv_to_rgb_planar_highbit(
    raw: &[u8],
    w: usize,
    h: usize,
    subsampling: (usize, usize),
    bit_depth: u32,
    bt709: bool,
) -> Vec<u8> {
    let (h_sub, v_sub) = subsampling;
    let y_plane_samples = w * h;
    let uv_w = w / h_sub;
    let uv_h = h / v_sub;
    let uv_plane_samples = uv_w * uv_h;

    // Byte offsets for each plane (2 bytes per sample)
    let y_off = 0usize;
    let u_off = y_plane_samples * 2;
    let v_off = u_off + uv_plane_samples * 2;

    let mut rgb = vec![0u8; w * h * 3];

    for py in 0..h {
        for px in 0..w {
            let y_val = read_u16_lsb_to_u8(raw, y_off + (py * w + px) * 2, bit_depth);
            let cx = px / h_sub;
            let cy = py / v_sub;
            let uv_idx = cy * uv_w + cx;
            let u_val = read_u16_lsb_to_u8(raw, u_off + uv_idx * 2, bit_depth);
            let v_val = read_u16_lsb_to_u8(raw, v_off + uv_idx * 2, bit_depth);
            let (r, g, b) = yuv_to_rgb_pixel(y_val, u_val, v_val, bt709);

            let out = (py * w + px) * 3;
            rgb[out] = r;
            rgb[out + 1] = g;
            rgb[out + 2] = b;
        }
    }
    rgb
}

/// Convert high-bit-depth semi-planar YUV (MSB-aligned, e.g. P010/P016/P210) to RGB24.
///
/// Layout: Y plane (w*h u16 LE) + interleaved UV plane (u16 LE pairs).
/// `uv_swapped`: false = U,V order, true = V,U order.
pub fn yuv_to_rgb_semi_planar_highbit(
    raw: &[u8],
    w: usize,
    h: usize,
    subsampling: (usize, usize),
    bit_depth: u32,
    uv_swapped: bool,
    bt709: bool,
) -> Vec<u8> {
    let (h_sub, v_sub) = subsampling;
    let y_plane_bytes = w * h * 2;
    let uv_w = w / h_sub;
    let mut rgb = vec![0u8; w * h * 3];

    for py in 0..h {
        for px in 0..w {
            let y_val = read_u16_msb_to_u8(raw, (py * w + px) * 2, bit_depth);
            let cx = px / h_sub;
            let cy = py / v_sub;
            let uv_base = y_plane_bytes + (cy * uv_w + cx) * 4; // 2 x u16 per pair
            let (u_val, v_val) = if uv_swapped {
                (
                    read_u16_msb_to_u8(raw, uv_base + 2, bit_depth),
                    read_u16_msb_to_u8(raw, uv_base, bit_depth),
                )
            } else {
                (
                    read_u16_msb_to_u8(raw, uv_base, bit_depth),
                    read_u16_msb_to_u8(raw, uv_base + 2, bit_depth),
                )
            };
            let (r, g, b) = yuv_to_rgb_pixel(y_val, u_val, v_val, bt709);

            let out = (py * w + px) * 3;
            rgb[out] = r;
            rgb[out + 1] = g;
            rgb[out + 2] = b;
        }
    }
    rgb
}

/// Convert Y210 (10-bit packed 4:2:2, MSB-aligned) to RGB24.
///
/// Layout per pixel pair: [Y0:u16, Cb:u16, Y1:u16, Cr:u16] — 8 bytes per 2 pixels.
pub fn yuv_to_rgb_y210(raw: &[u8], w: usize, h: usize, bt709: bool) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];

    for py in 0..h {
        for px_pair in 0..(w / 2) {
            let base = (py * w / 2 + px_pair) * 8;
            let y0 = read_u16_msb_to_u8(raw, base, 10);
            let cb = read_u16_msb_to_u8(raw, base + 2, 10);
            let y1 = read_u16_msb_to_u8(raw, base + 4, 10);
            let cr = read_u16_msb_to_u8(raw, base + 6, 10);

            let (r0, g0, b0) = yuv_to_rgb_pixel(y0, cb, cr, bt709);
            let (r1, g1, b1) = yuv_to_rgb_pixel(y1, cb, cr, bt709);

            let out0 = (py * w + px_pair * 2) * 3;
            rgb[out0] = r0;
            rgb[out0 + 1] = g0;
            rgb[out0 + 2] = b0;

            let out1 = out0 + 3;
            rgb[out1] = r1;
            rgb[out1 + 1] = g1;
            rgb[out1 + 2] = b1;
        }
    }
    rgb
}

/// Convert high-bit-depth greyscale (10/12/16-bit, LSB-aligned in u16 LE) to RGB24.
pub fn grey_highbit_to_rgb(raw: &[u8], w: usize, h: usize, bit_depth: u32) -> Vec<u8> {
    let shift = bit_depth - 8;
    let mut rgb = vec![0u8; w * h * 3];
    for i in 0..(w * h) {
        let offset = i * 2;
        if offset + 1 >= raw.len() {
            break;
        }
        let val = u16::from_le_bytes([raw[offset], raw[offset + 1]]);
        let g = (val >> shift) as u8;
        let o = i * 3;
        rgb[o] = g;
        rgb[o + 1] = g;
        rgb[o + 2] = g;
    }
    rgb
}

/// Convert RGB24 to greyscale 8-bit using BT.601 luma weights.
pub fn rgb_to_grey(raw: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut grey = vec![0u8; w * h];
    for (i, chunk) in raw.chunks_exact(3).enumerate().take(w * h) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        grey[i] = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0) as u8;
    }
    grey
}
