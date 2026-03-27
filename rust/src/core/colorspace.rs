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

/// Convert Y210/Y212/Y216 (MSB-aligned packed 4:2:2) to RGB24.
///
/// Layout per pixel pair: [Y0:u16, Cb:u16, Y1:u16, Cr:u16] — 8 bytes per 2 pixels.
/// All three formats use MSB-alignment in 16-bit words; the combined shift to 8-bit
/// is always `>> 8` regardless of bit_depth, so a single code path handles all.
pub fn yuv_to_rgb_y21x(raw: &[u8], w: usize, h: usize, bt709: bool) -> Vec<u8> {
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

/// Unpack 4 tightly-packed 10-bit samples from 5 bytes (LE bitstream, Rockchip NV15/NV20).
///
/// Layout:
///   byte0 = s0[7:0]
///   byte1 = s1[5:0]<<2 | s0[9:8]
///   byte2 = s2[3:0]<<4 | s1[9:6]
///   byte3 = s3[1:0]<<6 | s2[9:4]
///   byte4 = s3[9:2]
#[inline]
fn unpack_10bit_le_4samples(data: &[u8], offset: usize) -> [u16; 4] {
    let b0 = data[offset] as u16;
    let b1 = data[offset + 1] as u16;
    let b2 = data[offset + 2] as u16;
    let b3 = data[offset + 3] as u16;
    let b4 = data[offset + 4] as u16;
    [
        b0 | ((b1 & 0x03) << 8),
        (b1 >> 2) | ((b2 & 0x0F) << 6),
        (b2 >> 4) | ((b3 & 0x3F) << 4),
        (b3 >> 6) | (b4 << 2),
    ]
}

/// Convert NV15/NV20 (tightly packed 10-bit semi-planar) to RGB24.
///
/// Both Y and UV planes use LE bitstream packing (4 samples → 5 bytes).
/// NV15 = 4:2:0, NV20 = 4:2:2.
pub fn yuv_to_rgb_nv15_nv20(
    raw: &[u8],
    w: usize,
    h: usize,
    subsampling: (usize, usize),
    bt709: bool,
) -> Vec<u8> {
    let (h_sub, v_sub) = subsampling;
    let y_plane_bytes = (w * h * 10).div_ceil(8); // = w*h*5/4
    let uv_w = w / h_sub;
    let uv_h = h / v_sub;
    // UV plane: interleaved U,V → total uv_w*2 samples per row (treated as linear)
    let uv_samples_per_row = uv_w * 2;

    // Decode Y plane into a flat array of 8-bit values
    let y_count = w * h;
    let mut y_buf = vec![0u8; y_count];
    let groups = y_count / 4;
    for g in 0..groups {
        let samples = unpack_10bit_le_4samples(raw, g * 5);
        for k in 0..4 {
            y_buf[g * 4 + k] = (samples[k] >> 2) as u8;
        }
    }

    // Decode UV plane into flat arrays
    let uv_total_samples = uv_samples_per_row * uv_h;
    let uv_groups = uv_total_samples / 4;
    let mut uv_buf = vec![0u8; uv_total_samples];
    for g in 0..uv_groups {
        let samples = unpack_10bit_le_4samples(raw, y_plane_bytes + g * 5);
        for k in 0..4 {
            uv_buf[g * 4 + k] = (samples[k] >> 2) as u8;
        }
    }

    // Convert to RGB
    let mut rgb = vec![0u8; w * h * 3];
    for py in 0..h {
        for px in 0..w {
            let y_val = y_buf[py * w + px];
            let cx = px / h_sub;
            let cy = py / v_sub;
            let uv_base = cy * uv_samples_per_row + cx * 2;
            let u_val = uv_buf[uv_base];
            let v_val = uv_buf[uv_base + 1];
            let (r, g, b) = yuv_to_rgb_pixel(y_val, u_val, v_val, bt709);
            let out = (py * w + px) * 3;
            rgb[out] = r;
            rgb[out + 1] = g;
            rgb[out + 2] = b;
        }
    }
    rgb
}

/// Unpack 4 tightly-packed 10-bit samples from 5 bytes (BE bitstream, Y10BPACK).
///
/// Layout (MSB first):
///   s0 = byte0[7:0]<<2 | byte1[7:6]
///   s1 = byte1[5:0]<<4 | byte2[7:4]
///   s2 = byte2[3:0]<<6 | byte3[7:2]
///   s3 = byte3[1:0]<<8 | byte4[7:0]
#[inline]
fn unpack_10bit_be_4samples(data: &[u8], offset: usize) -> [u16; 4] {
    let b0 = data[offset] as u16;
    let b1 = data[offset + 1] as u16;
    let b2 = data[offset + 2] as u16;
    let b3 = data[offset + 3] as u16;
    let b4 = data[offset + 4] as u16;
    [
        (b0 << 2) | (b1 >> 6),
        ((b1 & 0x3F) << 4) | (b2 >> 4),
        ((b2 & 0x0F) << 6) | (b3 >> 2),
        ((b3 & 0x03) << 8) | b4,
    ]
}

/// Convert Y10BPACK (10-bit BE packed greyscale) to RGB24.
/// 4 pixels in 5 bytes, MSB-first.
pub fn grey_10bpack_to_rgb(raw: &[u8], w: usize, h: usize) -> Vec<u8> {
    let count = w * h;
    let groups = count / 4;
    let mut rgb = vec![0u8; count * 3];
    for g in 0..groups {
        let samples = unpack_10bit_be_4samples(raw, g * 5);
        for (k, &s) in samples.iter().enumerate() {
            let v = (s >> 2) as u8;
            let o = (g * 4 + k) * 3;
            rgb[o] = v;
            rgb[o + 1] = v;
            rgb[o + 2] = v;
        }
    }
    rgb
}

/// Convert Y10P (MIPI RAW10 packed greyscale) to RGB24.
/// Layout: [MSB0, MSB1, MSB2, MSB3, LSBs] per 4 pixels.
/// LSBs byte: s0[1:0] | s1[1:0]<<2 | s2[1:0]<<4 | s3[1:0]<<6
pub fn grey_y10p_to_rgb(raw: &[u8], w: usize, h: usize) -> Vec<u8> {
    let count = w * h;
    let groups = count / 4;
    let mut rgb = vec![0u8; count * 3];
    for g in 0..groups {
        let base = g * 5;
        let lsbs = raw[base + 4];
        for k in 0..4usize {
            let msb = raw[base + k] as u16;
            let lsb = ((lsbs >> (k * 2)) & 0x03) as u16;
            let val10 = (msb << 2) | lsb;
            let v = (val10 >> 2) as u8;
            let o = (g * 4 + k) * 3;
            rgb[o] = v;
            rgb[o + 1] = v;
            rgb[o + 2] = v;
        }
    }
    rgb
}

/// Detile P010_4L4 (T010) tiled data into linear P010 layout.
///
/// Tiles are 4x4 pixels. Each sample is u16 LE (same as P010).
/// Y plane: tiles in raster order, UV plane: tiles in raster order.
pub fn detile_p010_4l4(raw: &[u8], w: usize, h: usize) -> Vec<u8> {
    let y_plane_bytes = w * h * 2;
    let uv_h = h / 2;
    let uv_plane_bytes = w * uv_h * 2; // interleaved U,V at half height
    let mut out = vec![0u8; y_plane_bytes + uv_plane_bytes];

    let tile_w = 4;
    let tile_h = 4;
    let bytes_per_sample = 2;

    // Detile Y plane
    let tiles_x = w / tile_w;
    let tiles_y = h / tile_h;
    let tile_bytes = tile_w * tile_h * bytes_per_sample; // 32 bytes per tile
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let tile_idx = ty * tiles_x + tx;
            let src_base = tile_idx * tile_bytes;
            for row in 0..tile_h {
                let dst_y = ty * tile_h + row;
                let dst_x = tx * tile_w;
                let dst_off = (dst_y * w + dst_x) * bytes_per_sample;
                let src_off = src_base + row * tile_w * bytes_per_sample;
                let len = tile_w * bytes_per_sample;
                out[dst_off..dst_off + len].copy_from_slice(&raw[src_off..src_off + len]);
            }
        }
    }

    // Detile UV plane (4x4 tiles of interleaved UV, effectively 4x4 sample pairs at half height)
    // UV plane: each tile covers 4x4 UV samples. UV dimensions: w x (h/2).
    let uv_tiles_y = uv_h / tile_h;
    let uv_tile_bytes = tile_w * tile_h * bytes_per_sample;
    for ty in 0..uv_tiles_y {
        for tx in 0..tiles_x {
            let tile_idx = ty * tiles_x + tx;
            let src_base = y_plane_bytes + tile_idx * uv_tile_bytes;
            // Handle case where there might not be enough UV tiles
            if src_base + uv_tile_bytes > raw.len() {
                break;
            }
            for row in 0..tile_h {
                let dst_y = ty * tile_h + row;
                if dst_y >= uv_h {
                    break;
                }
                let dst_x = tx * tile_w;
                let dst_off = y_plane_bytes + (dst_y * w + dst_x) * bytes_per_sample;
                let src_off = src_base + row * tile_w * bytes_per_sample;
                let len = tile_w * bytes_per_sample;
                if dst_off + len <= out.len() && src_off + len <= raw.len() {
                    out[dst_off..dst_off + len].copy_from_slice(&raw[src_off..src_off + len]);
                }
            }
        }
    }

    out
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
