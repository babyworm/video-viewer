use std::fs::File;
use std::io::Write;

use memmap2::Mmap;

use crate::core::formats::{get_format_by_name, FormatType, VideoFormat};

/// Stateless video format converter.
pub struct VideoConverter;

impl VideoConverter {
    pub fn new() -> Self {
        VideoConverter
    }

    /// Convert a raw video file from one pixel format to another.
    ///
    /// Returns `(frames_converted, was_cancelled)`.
    ///
    /// The optional `progress_cb` receives `(current_frame, total_frames)` and
    /// should return `false` to cancel the conversion.
    pub fn convert(
        &self,
        input_path: &str,
        w: u32,
        h: u32,
        input_fmt_name: &str,
        output_path: &str,
        output_fmt_name: &str,
        progress_cb: Option<&dyn Fn(usize, usize) -> bool>,
    ) -> Result<(usize, bool), String> {
        let input_fmt = get_format_by_name(input_fmt_name)
            .ok_or_else(|| format!("Unknown input format '{input_fmt_name}'"))?;
        let output_fmt = get_format_by_name(output_fmt_name)
            .ok_or_else(|| format!("Unknown output format '{output_fmt_name}'"))?;

        let in_frame_size = input_fmt.frame_size(w, h);
        let out_frame_size = output_fmt.frame_size(w, h);

        if in_frame_size == 0 {
            return Err(format!(
                "Input frame size is 0 for format '{}'",
                input_fmt.name
            ));
        }
        if out_frame_size == 0 {
            return Err(format!(
                "Output frame size is 0 for format '{}'",
                output_fmt.name
            ));
        }

        // Memory-map input file
        let file = File::open(input_path)
            .map_err(|e| format!("Cannot open '{}': {e}", input_path))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("Cannot mmap '{}': {e}", input_path))?;
        let raw = mmap.as_ref();

        let total_frames = raw.len() / in_frame_size;
        if total_frames == 0 {
            return Err("Input file contains no complete frames".to_string());
        }

        // Open output
        let mut out_file = File::create(output_path)
            .map_err(|e| format!("Cannot create '{}': {e}", output_path))?;

        let same_format = input_fmt.fourcc == output_fmt.fourcc;
        let both_yuv = is_yuv(input_fmt) && is_yuv(output_fmt);

        for frame_idx in 0..total_frames {
            // Progress callback
            if let Some(cb) = progress_cb {
                if !cb(frame_idx, total_frames) {
                    return Ok((frame_idx, true));
                }
            }

            let offset = frame_idx * in_frame_size;
            let end = offset + in_frame_size;
            if end > raw.len() {
                break;
            }
            let frame_data = &raw[offset..end];

            let out_data = if same_format {
                // Identity copy
                frame_data.to_vec()
            } else if both_yuv {
                // Direct YUV plane manipulation
                convert_yuv_to_yuv(frame_data, w, h, input_fmt, output_fmt)?
            } else {
                // Go through RGB intermediate using opencv
                convert_via_rgb(frame_data, w, h, input_fmt, output_fmt)?
            };

            out_file
                .write_all(&out_data)
                .map_err(|e| format!("Write error: {e}"))?;
        }

        Ok((total_frames, false))
    }
}

impl Default for VideoConverter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// YUV <-> YUV direct conversion
// ---------------------------------------------------------------------------

fn is_yuv(fmt: &VideoFormat) -> bool {
    matches!(
        fmt.format_type,
        FormatType::YuvPlanar | FormatType::YuvSemiPlanar | FormatType::YuvPacked
    )
}

/// Extract Y, U, V planes from a raw frame in any YUV format.
pub fn extract_yuv_planes(
    raw: &[u8],
    w: u32,
    h: u32,
    fmt: &VideoFormat,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = w as usize;
    let h = h as usize;
    let y_size = w * h;
    let (h_sub, v_sub) = (fmt.subsampling.0 as usize, fmt.subsampling.1 as usize);
    let uv_w = w / h_sub;
    let uv_h = h / v_sub;

    match fmt.format_type {
        FormatType::YuvPlanar => {
            let y = raw[..y_size].to_vec();
            let uv_size = uv_w * uv_h;
            let fc = fmt.fourcc.as_str();
            let (u, v) = if fc == "YV12" || fc == "YM21" {
                // V before U
                let v = raw[y_size..y_size + uv_size].to_vec();
                let u = raw[y_size + uv_size..y_size + uv_size * 2].to_vec();
                (u, v)
            } else {
                let u = raw[y_size..y_size + uv_size].to_vec();
                let v = raw[y_size + uv_size..y_size + uv_size * 2].to_vec();
                (u, v)
            };
            (y, u, v)
        }
        FormatType::YuvSemiPlanar => {
            let y = raw[..y_size].to_vec();
            let uv_size = uv_w * uv_h;
            let uv_data = &raw[y_size..];
            let mut u = vec![0u8; uv_size];
            let mut v = vec![0u8; uv_size];
            let fc = fmt.fourcc.as_str();
            let u_first = matches!(fc, "NV12" | "NM12" | "NV16" | "NV24" | "P010" | "P016");
            for i in 0..uv_size {
                if u_first {
                    u[i] = uv_data[i * 2];
                    v[i] = uv_data[i * 2 + 1];
                } else {
                    v[i] = uv_data[i * 2];
                    u[i] = uv_data[i * 2 + 1];
                }
            }
            (y, u, v)
        }
        FormatType::YuvPacked => {
            let mut y = Vec::with_capacity(y_size);
            let mut u = Vec::with_capacity(uv_w * uv_h);
            let mut v = Vec::with_capacity(uv_w * uv_h);
            let fc = fmt.fourcc.as_str();
            match fc {
                "YUYV" => {
                    // Y0 U0 Y1 V0
                    for i in (0..raw.len()).step_by(4) {
                        y.push(raw[i]);
                        y.push(raw[i + 2]);
                        u.push(raw[i + 1]);
                        v.push(raw[i + 3]);
                    }
                }
                "UYVY" => {
                    // U0 Y0 V0 Y1
                    for i in (0..raw.len()).step_by(4) {
                        u.push(raw[i]);
                        y.push(raw[i + 1]);
                        v.push(raw[i + 2]);
                        y.push(raw[i + 3]);
                    }
                }
                "YVYU" => {
                    // Y0 V0 Y1 U0
                    for i in (0..raw.len()).step_by(4) {
                        y.push(raw[i]);
                        v.push(raw[i + 1]);
                        y.push(raw[i + 2]);
                        u.push(raw[i + 3]);
                    }
                }
                _ => {
                    // Fallback: treat as YUYV
                    for i in (0..raw.len()).step_by(4) {
                        if i + 3 < raw.len() {
                            y.push(raw[i]);
                            y.push(raw[i + 2]);
                            u.push(raw[i + 1]);
                            v.push(raw[i + 3]);
                        }
                    }
                }
            }
            (y, u, v)
        }
        _ => {
            // Non-YUV: return empty planes
            (vec![0u8; y_size], vec![128u8; uv_w * uv_h], vec![128u8; uv_w * uv_h])
        }
    }
}

/// Pack Y, U, V planes into the target format's byte layout.
pub fn pack_yuv(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    w: u32,
    h: u32,
    fmt: &VideoFormat,
) -> Vec<u8> {
    let w = w as usize;
    let h = h as usize;
    let (h_sub, v_sub) = (fmt.subsampling.0 as usize, fmt.subsampling.1 as usize);
    let uv_w = w / h_sub;
    let uv_h = h / v_sub;
    let uv_size = uv_w * uv_h;

    match fmt.format_type {
        FormatType::YuvPlanar => {
            let fc = fmt.fourcc.as_str();
            let mut out = Vec::with_capacity(y.len() + uv_size * 2);
            out.extend_from_slice(y);
            if fc == "YV12" || fc == "YM21" {
                out.extend_from_slice(v);
                out.extend_from_slice(u);
            } else {
                out.extend_from_slice(u);
                out.extend_from_slice(v);
            }
            out
        }
        FormatType::YuvSemiPlanar => {
            let fc = fmt.fourcc.as_str();
            let u_first = matches!(fc, "NV12" | "NM12" | "NV16" | "NV24" | "P010" | "P016");
            let mut out = Vec::with_capacity(y.len() + uv_size * 2);
            out.extend_from_slice(y);
            for i in 0..uv_size {
                if u_first {
                    out.push(u[i]);
                    out.push(v[i]);
                } else {
                    out.push(v[i]);
                    out.push(u[i]);
                }
            }
            out
        }
        FormatType::YuvPacked => {
            let fc = fmt.fourcc.as_str();
            let y_size = w * h;
            let mut out = vec![0u8; y_size * 2]; // 4:2:2 packed = 2 bytes/pixel
            match fc {
                "YUYV" => {
                    for row in 0..h {
                        for col in (0..w).step_by(2) {
                            let yi = row * w + col;
                            let ui = (row / v_sub) * uv_w + col / h_sub;
                            let oi = (row * w + col) * 2;
                            out[oi] = y[yi];
                            out[oi + 1] = u[ui];
                            out[oi + 2] = y[yi + 1];
                            out[oi + 3] = v[ui];
                        }
                    }
                }
                "UYVY" => {
                    for row in 0..h {
                        for col in (0..w).step_by(2) {
                            let yi = row * w + col;
                            let ui = (row / v_sub) * uv_w + col / h_sub;
                            let oi = (row * w + col) * 2;
                            out[oi] = u[ui];
                            out[oi + 1] = y[yi];
                            out[oi + 2] = v[ui];
                            out[oi + 3] = y[yi + 1];
                        }
                    }
                }
                "YVYU" => {
                    for row in 0..h {
                        for col in (0..w).step_by(2) {
                            let yi = row * w + col;
                            let ui = (row / v_sub) * uv_w + col / h_sub;
                            let oi = (row * w + col) * 2;
                            out[oi] = y[yi];
                            out[oi + 1] = v[ui];
                            out[oi + 2] = y[yi + 1];
                            out[oi + 3] = u[ui];
                        }
                    }
                }
                _ => {
                    // Default YUYV
                    for row in 0..h {
                        for col in (0..w).step_by(2) {
                            let yi = row * w + col;
                            let ui = (row / v_sub) * uv_w + col / h_sub;
                            let oi = (row * w + col) * 2;
                            out[oi] = y[yi];
                            out[oi + 1] = u[ui];
                            out[oi + 2] = y[yi + 1];
                            out[oi + 3] = v[ui];
                        }
                    }
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

/// Bilinear resize of a chroma plane.
pub fn resample_chroma(
    plane: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    let sw = src_w as usize;
    let sh = src_h as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;

    if sw == dw && sh == dh {
        return plane.to_vec();
    }

    let mut out = vec![0u8; dw * dh];

    for dy in 0..dh {
        let src_y_f = (dy as f64) * (sh as f64) / (dh as f64);
        let sy0 = (src_y_f as usize).min(sh.saturating_sub(1));
        let sy1 = (sy0 + 1).min(sh.saturating_sub(1));
        let fy = src_y_f - sy0 as f64;

        for dx in 0..dw {
            let src_x_f = (dx as f64) * (sw as f64) / (dw as f64);
            let sx0 = (src_x_f as usize).min(sw.saturating_sub(1));
            let sx1 = (sx0 + 1).min(sw.saturating_sub(1));
            let fx = src_x_f - sx0 as f64;

            let p00 = plane[sy0 * sw + sx0] as f64;
            let p10 = plane[sy0 * sw + sx1] as f64;
            let p01 = plane[sy1 * sw + sx0] as f64;
            let p11 = plane[sy1 * sw + sx1] as f64;

            let val = p00 * (1.0 - fx) * (1.0 - fy)
                + p10 * fx * (1.0 - fy)
                + p01 * (1.0 - fx) * fy
                + p11 * fx * fy;

            out[dy * dw + dx] = val.round().clamp(0.0, 255.0) as u8;
        }
    }

    out
}

/// Convert YUV frame to another YUV format via plane extraction + chroma resample + repack.
fn convert_yuv_to_yuv(
    frame: &[u8],
    w: u32,
    h: u32,
    src_fmt: &VideoFormat,
    dst_fmt: &VideoFormat,
) -> Result<Vec<u8>, String> {
    let (y, u, v) = extract_yuv_planes(frame, w, h, src_fmt);

    let (src_h_sub, src_v_sub) = (src_fmt.subsampling.0, src_fmt.subsampling.1);
    let (dst_h_sub, dst_v_sub) = (dst_fmt.subsampling.0, dst_fmt.subsampling.1);

    let src_uv_w = w / src_h_sub;
    let src_uv_h = h / src_v_sub;
    let dst_uv_w = w / dst_h_sub;
    let dst_uv_h = h / dst_v_sub;

    let u_out = resample_chroma(&u, src_uv_w, src_uv_h, dst_uv_w, dst_uv_h);
    let v_out = resample_chroma(&v, src_uv_w, src_uv_h, dst_uv_w, dst_uv_h);

    Ok(pack_yuv(&y, &u_out, &v_out, w, h, dst_fmt))
}

/// Convert a frame through an RGB intermediate using opencv.
fn convert_via_rgb(
    frame: &[u8],
    w: u32,
    h: u32,
    src_fmt: &VideoFormat,
    dst_fmt: &VideoFormat,
) -> Result<Vec<u8>, String> {
    #[allow(unused_imports)]
    use opencv::prelude::*;
    use opencv::core::{Mat, CV_8UC1, CV_8UC2, CV_8UC3};
    use opencv::imgproc;

    let iw = w as i32;
    let ih = h as i32;

    // --- Step 1: src format -> RGB ---
    let rgb = match src_fmt.format_type {
        FormatType::YuvPlanar if matches!(src_fmt.fourcc.as_str(), "YU12" | "YV12") => {
            let yuv_mat = unsafe {
                Mat::new_rows_cols_with_data_unsafe(
                    ih * 3 / 2, iw, CV_8UC1,
                    frame.as_ptr() as *mut std::ffi::c_void,
                    opencv::core::Mat_AUTO_STEP,
                ).map_err(|e| e.to_string())?
            };
            let code = if src_fmt.fourcc == "YU12" {
                imgproc::COLOR_YUV2RGB_I420
            } else {
                imgproc::COLOR_YUV2RGB_YV12
            };
            let mut rgb_mat = Mat::default();
            imgproc::cvt_color(&yuv_mat, &mut rgb_mat, code, 0).map_err(|e| e.to_string())?;
            mat_to_rgb_vec(&rgb_mat)?
        }
        FormatType::YuvSemiPlanar if matches!(src_fmt.fourcc.as_str(), "NV12" | "NV21") => {
            let yuv_mat = unsafe {
                Mat::new_rows_cols_with_data_unsafe(
                    ih * 3 / 2, iw, CV_8UC1,
                    frame.as_ptr() as *mut std::ffi::c_void,
                    opencv::core::Mat_AUTO_STEP,
                ).map_err(|e| e.to_string())?
            };
            let code = if src_fmt.fourcc == "NV12" {
                imgproc::COLOR_YUV2RGB_NV12
            } else {
                imgproc::COLOR_YUV2RGB_NV21
            };
            let mut rgb_mat = Mat::default();
            imgproc::cvt_color(&yuv_mat, &mut rgb_mat, code, 0).map_err(|e| e.to_string())?;
            mat_to_rgb_vec(&rgb_mat)?
        }
        FormatType::YuvPacked if matches!(src_fmt.fourcc.as_str(), "YUYV" | "UYVY") => {
            let yuv_mat = unsafe {
                Mat::new_rows_cols_with_data_unsafe(
                    ih, iw, CV_8UC2,
                    frame.as_ptr() as *mut std::ffi::c_void,
                    opencv::core::Mat_AUTO_STEP,
                ).map_err(|e| e.to_string())?
            };
            let code = if src_fmt.fourcc == "YUYV" {
                imgproc::COLOR_YUV2RGB_YUYV
            } else {
                imgproc::COLOR_YUV2RGB_UYVY
            };
            let mut rgb_mat = Mat::default();
            imgproc::cvt_color(&yuv_mat, &mut rgb_mat, code, 0).map_err(|e| e.to_string())?;
            mat_to_rgb_vec(&rgb_mat)?
        }
        FormatType::Rgb if src_fmt.fourcc == "RGB3" => {
            frame[..(w as usize * h as usize * 3)].to_vec()
        }
        FormatType::Rgb if src_fmt.fourcc == "BGR3" => {
            let bgr_mat = unsafe {
                Mat::new_rows_cols_with_data_unsafe(
                    ih, iw, CV_8UC3,
                    frame.as_ptr() as *mut std::ffi::c_void,
                    opencv::core::Mat_AUTO_STEP,
                ).map_err(|e| e.to_string())?
            };
            let mut rgb_mat = Mat::default();
            imgproc::cvt_color(&bgr_mat, &mut rgb_mat, imgproc::COLOR_BGR2RGB, 0)
                .map_err(|e| e.to_string())?;
            mat_to_rgb_vec(&rgb_mat)?
        }
        FormatType::Rgb => {
            // 32-bit and 16-bit RGB variants → RGB24
            rgb_variant_to_rgb24(frame, w, h, src_fmt)?
        }
        FormatType::Bayer if src_fmt.bit_depth == 8 => {
            bayer_to_rgb24(frame, w, h, src_fmt)?
        }
        FormatType::Bayer => {
            // 10/12/16-bit Bayer: truncate to 8-bit, then demosaic
            bayer_highbit_to_rgb24(frame, w, h, src_fmt)?
        }
        FormatType::Grey if src_fmt.bit_depth == 8 => {
            let grey_mat = unsafe {
                Mat::new_rows_cols_with_data_unsafe(
                    ih, iw, CV_8UC1,
                    frame.as_ptr() as *mut std::ffi::c_void,
                    opencv::core::Mat_AUTO_STEP,
                ).map_err(|e| e.to_string())?
            };
            let mut rgb_mat = Mat::default();
            imgproc::cvt_color(&grey_mat, &mut rgb_mat, imgproc::COLOR_GRAY2RGB, 0)
                .map_err(|e| e.to_string())?;
            mat_to_rgb_vec(&rgb_mat)?
        }
        FormatType::Grey => {
            // 10/12/16-bit greyscale: 2 bytes per pixel LE, shift to 8-bit
            grey_highbit_to_rgb24(frame, w, h, src_fmt)?
        }
        _ if is_yuv(src_fmt) => {
            // For other YUV variants, extract planes, upsample to 444, convert manually
            let (y, u, v) = extract_yuv_planes(frame, w, h, src_fmt);
            let (s_h, s_v) = (src_fmt.subsampling.0, src_fmt.subsampling.1);
            let uv_w = w / s_h;
            let uv_h = h / s_v;
            let u_full = resample_chroma(&u, uv_w, uv_h, w, h);
            let v_full = resample_chroma(&v, uv_w, uv_h, w, h);
            let pixels = (w * h) as usize;
            let mut rgb = vec![0u8; pixels * 3];
            for i in 0..pixels {
                let yv = y[i] as f32;
                let uv = u_full[i] as f32 - 128.0;
                let vv = v_full[i] as f32 - 128.0;
                rgb[i * 3] = (yv + 1.402 * vv).clamp(0.0, 255.0) as u8;
                rgb[i * 3 + 1] = (yv - 0.344136 * uv - 0.714136 * vv).clamp(0.0, 255.0) as u8;
                rgb[i * 3 + 2] = (yv + 1.772 * uv).clamp(0.0, 255.0) as u8;
            }
            rgb
        }
        _ => {
            return Err(format!(
                "Conversion from format '{}' ({}) not supported as source",
                src_fmt.name, src_fmt.fourcc
            ));
        }
    };

    // --- Step 2: RGB -> dst format ---
    match dst_fmt.format_type {
        FormatType::Rgb if dst_fmt.fourcc == "RGB3" => Ok(rgb),
        FormatType::Rgb if dst_fmt.fourcc == "BGR3" => {
            let rgb_mat = unsafe {
                Mat::new_rows_cols_with_data_unsafe(
                    ih, iw, CV_8UC3,
                    rgb.as_ptr() as *mut std::ffi::c_void,
                    opencv::core::Mat_AUTO_STEP,
                ).map_err(|e| e.to_string())?
            };
            let mut bgr_mat = Mat::default();
            imgproc::cvt_color(&rgb_mat, &mut bgr_mat, imgproc::COLOR_RGB2BGR, 0)
                .map_err(|e| e.to_string())?;
            mat_to_rgb_vec(&bgr_mat)
        }
        FormatType::Rgb => {
            // RGB24 → 32-bit/16-bit RGB variants
            rgb24_to_rgb_variant(&rgb, w, h, dst_fmt)
        }
        FormatType::Grey if dst_fmt.bit_depth == 8 => {
            let rgb_mat = unsafe {
                Mat::new_rows_cols_with_data_unsafe(
                    ih, iw, CV_8UC3,
                    rgb.as_ptr() as *mut std::ffi::c_void,
                    opencv::core::Mat_AUTO_STEP,
                ).map_err(|e| e.to_string())?
            };
            let mut grey_mat = Mat::default();
            imgproc::cvt_color(&rgb_mat, &mut grey_mat, imgproc::COLOR_RGB2GRAY, 0)
                .map_err(|e| e.to_string())?;
            mat_to_rgb_vec(&grey_mat)
        }
        FormatType::Grey => {
            // 10/12/16-bit greyscale destination: convert to 8-bit grey, then upshift
            rgb24_to_grey_highbit(&rgb, w, h, dst_fmt)
        }
        _ if is_yuv(dst_fmt) => {
            // RGB -> YUV via manual BT.601 then pack
            let pixels = (w * h) as usize;
            let mut y = vec![0u8; pixels];
            let mut u_full = vec![0u8; pixels];
            let mut v_full = vec![0u8; pixels];
            for i in 0..pixels {
                let r = rgb[i * 3] as f32;
                let g = rgb[i * 3 + 1] as f32;
                let b = rgb[i * 3 + 2] as f32;
                y[i] = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0) as u8;
                u_full[i] = (128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b).clamp(0.0, 255.0) as u8;
                v_full[i] = (128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b).clamp(0.0, 255.0) as u8;
            }
            let (d_h, d_v) = (dst_fmt.subsampling.0, dst_fmt.subsampling.1);
            let dst_uv_w = w / d_h;
            let dst_uv_h = h / d_v;
            let u_out = resample_chroma(&u_full, w, h, dst_uv_w, dst_uv_h);
            let v_out = resample_chroma(&v_full, w, h, dst_uv_w, dst_uv_h);
            Ok(pack_yuv(&y, &u_out, &v_out, w, h, dst_fmt))
        }
        _ => Err(format!(
            "Conversion to format '{}' not supported",
            dst_fmt.name
        )),
    }
}

/// Convert 32-bit/16-bit RGB variant → RGB24.
fn rgb_variant_to_rgb24(frame: &[u8], w: u32, h: u32, fmt: &VideoFormat) -> Result<Vec<u8>, String> {
    let pixels = (w * h) as usize;
    let mut rgb = vec![0u8; pixels * 3];
    let fc = fmt.fourcc.as_str();

    match fc {
        // 32-bit: 4 bytes per pixel
        "RGB4" => { // RGB32: R G B X
            for i in 0..pixels { rgb[i*3]=frame[i*4]; rgb[i*3+1]=frame[i*4+1]; rgb[i*3+2]=frame[i*4+2]; }
        }
        "BGR4" => { // BGR32: B G R X
            for i in 0..pixels { rgb[i*3]=frame[i*4+2]; rgb[i*3+1]=frame[i*4+1]; rgb[i*3+2]=frame[i*4]; }
        }
        "BA24" => { // ARGB32: A R G B
            for i in 0..pixels { rgb[i*3]=frame[i*4+1]; rgb[i*3+1]=frame[i*4+2]; rgb[i*3+2]=frame[i*4+3]; }
        }
        "AR24" => { // ABGR32: A B G R
            for i in 0..pixels { rgb[i*3]=frame[i*4+3]; rgb[i*3+1]=frame[i*4+2]; rgb[i*3+2]=frame[i*4+1]; }
        }
        "AB24" => { // RGBA32: R G B A
            for i in 0..pixels { rgb[i*3]=frame[i*4]; rgb[i*3+1]=frame[i*4+1]; rgb[i*3+2]=frame[i*4+2]; }
        }
        "RA24" => { // BGRA32: B G R A
            for i in 0..pixels { rgb[i*3]=frame[i*4+2]; rgb[i*3+1]=frame[i*4+1]; rgb[i*3+2]=frame[i*4]; }
        }
        "XB24" | "XR24" | "BX24" | "RX24" => {
            // XRGB/XBGR/RGBX/BGRX — same layout as ARGB/ABGR/RGBA/BGRA, alpha ignored
            let mapped = match fc {
                "XB24" => "BA24", // XRGB → treat as ARGB
                "XR24" => "AR24", // XBGR → treat as ABGR
                "BX24" => "AB24", // RGBX → treat as RGBA
                "RX24" => "RA24", // BGRX → treat as BGRA
                _ => unreachable!(),
            };
            let mut fake_fmt = fmt.clone();
            fake_fmt.fourcc = mapped.to_string();
            return rgb_variant_to_rgb24(frame, w, h, &fake_fmt);
        }
        // 16-bit: 2 bytes per pixel
        "RGBP" => { // RGB565: RRRRRGGG GGGBBBBB
            for i in 0..pixels {
                let v = u16::from_le_bytes([frame[i*2], frame[i*2+1]]);
                rgb[i*3]   = ((v >> 11) as u8) << 3;
                rgb[i*3+1] = (((v >> 5) & 0x3F) as u8) << 2;
                rgb[i*3+2] = ((v & 0x1F) as u8) << 3;
            }
        }
        "RGBO" => { // RGB555: 0RRRRRGGGGGBBBBB
            for i in 0..pixels {
                let v = u16::from_le_bytes([frame[i*2], frame[i*2+1]]);
                rgb[i*3]   = (((v >> 10) & 0x1F) as u8) << 3;
                rgb[i*3+1] = (((v >> 5) & 0x1F) as u8) << 3;
                rgb[i*3+2] = ((v & 0x1F) as u8) << 3;
            }
        }
        "R444" => { // RGB444: 0000RRRRGGGGBBBB
            for i in 0..pixels {
                let v = u16::from_le_bytes([frame[i*2], frame[i*2+1]]);
                rgb[i*3]   = (((v >> 8) & 0xF) as u8) << 4;
                rgb[i*3+1] = (((v >> 4) & 0xF) as u8) << 4;
                rgb[i*3+2] = ((v & 0xF) as u8) << 4;
            }
        }
        "RGB1" => { // RGB332: RRRGGGBB
            for i in 0..pixels {
                let v = frame[i];
                rgb[i*3]   = (v >> 5) << 5;
                rgb[i*3+1] = ((v >> 2) & 0x07) << 5;
                rgb[i*3+2] = (v & 0x03) << 6;
            }
        }
        _ => {
            return Err(format!("RGB variant '{}' ({}) not supported as source", fmt.name, fc));
        }
    }
    Ok(rgb)
}

/// Convert RGB24 → 32-bit/16-bit RGB variant.
fn rgb24_to_rgb_variant(rgb: &[u8], w: u32, h: u32, fmt: &VideoFormat) -> Result<Vec<u8>, String> {
    let pixels = (w * h) as usize;
    let fc = fmt.fourcc.as_str();

    match fc {
        // 32-bit
        "RGB4" => {
            let mut out = vec![0u8; pixels * 4];
            for i in 0..pixels { out[i*4]=rgb[i*3]; out[i*4+1]=rgb[i*3+1]; out[i*4+2]=rgb[i*3+2]; out[i*4+3]=0xFF; }
            Ok(out)
        }
        "BGR4" => {
            let mut out = vec![0u8; pixels * 4];
            for i in 0..pixels { out[i*4]=rgb[i*3+2]; out[i*4+1]=rgb[i*3+1]; out[i*4+2]=rgb[i*3]; out[i*4+3]=0xFF; }
            Ok(out)
        }
        "BA24" => { // ARGB
            let mut out = vec![0u8; pixels * 4];
            for i in 0..pixels { out[i*4]=0xFF; out[i*4+1]=rgb[i*3]; out[i*4+2]=rgb[i*3+1]; out[i*4+3]=rgb[i*3+2]; }
            Ok(out)
        }
        "AR24" => { // ABGR
            let mut out = vec![0u8; pixels * 4];
            for i in 0..pixels { out[i*4]=0xFF; out[i*4+1]=rgb[i*3+2]; out[i*4+2]=rgb[i*3+1]; out[i*4+3]=rgb[i*3]; }
            Ok(out)
        }
        "AB24" => { // RGBA
            let mut out = vec![0u8; pixels * 4];
            for i in 0..pixels { out[i*4]=rgb[i*3]; out[i*4+1]=rgb[i*3+1]; out[i*4+2]=rgb[i*3+2]; out[i*4+3]=0xFF; }
            Ok(out)
        }
        "RA24" => { // BGRA
            let mut out = vec![0u8; pixels * 4];
            for i in 0..pixels { out[i*4]=rgb[i*3+2]; out[i*4+1]=rgb[i*3+1]; out[i*4+2]=rgb[i*3]; out[i*4+3]=0xFF; }
            Ok(out)
        }
        "XB24" | "XR24" | "BX24" | "RX24" => {
            let mapped = match fc {
                "XB24" => "BA24", "XR24" => "AR24", "BX24" => "AB24", "RX24" => "RA24", _ => unreachable!(),
            };
            let mut fake_fmt = fmt.clone();
            fake_fmt.fourcc = mapped.to_string();
            rgb24_to_rgb_variant(rgb, w, h, &fake_fmt)
        }
        "AR12" | "AR15" | "XR12" | "XR15" => {
            // 16-bit ARGB variants → same as non-alpha versions
            let mapped = match fc {
                "AR12" | "XR12" => "R444", "AR15" | "XR15" => "RGBO", _ => unreachable!(),
            };
            let mut fake_fmt = fmt.clone();
            fake_fmt.fourcc = mapped.to_string();
            rgb24_to_rgb_variant(rgb, w, h, &fake_fmt)
        }
        // 16-bit
        "RGBP" => { // RGB565
            let mut out = vec![0u8; pixels * 2];
            for i in 0..pixels {
                let r = (rgb[i*3] >> 3) as u16;
                let g = (rgb[i*3+1] >> 2) as u16;
                let b = (rgb[i*3+2] >> 3) as u16;
                let v = (r << 11) | (g << 5) | b;
                let bytes = v.to_le_bytes();
                out[i*2] = bytes[0]; out[i*2+1] = bytes[1];
            }
            Ok(out)
        }
        "RGBO" | "RGBQ" => { // RGB555/RGB555X
            let mut out = vec![0u8; pixels * 2];
            for i in 0..pixels {
                let r = (rgb[i*3] >> 3) as u16;
                let g = (rgb[i*3+1] >> 3) as u16;
                let b = (rgb[i*3+2] >> 3) as u16;
                let v = (r << 10) | (g << 5) | b;
                let bytes = if fc == "RGBQ" { v.to_be_bytes() } else { v.to_le_bytes() };
                out[i*2] = bytes[0]; out[i*2+1] = bytes[1];
            }
            Ok(out)
        }
        "RGBR" => { // RGB565X (big-endian)
            let mut out = vec![0u8; pixels * 2];
            for i in 0..pixels {
                let r = (rgb[i*3] >> 3) as u16;
                let g = (rgb[i*3+1] >> 2) as u16;
                let b = (rgb[i*3+2] >> 3) as u16;
                let v = (r << 11) | (g << 5) | b;
                let bytes = v.to_be_bytes();
                out[i*2] = bytes[0]; out[i*2+1] = bytes[1];
            }
            Ok(out)
        }
        "R444" => { // RGB444
            let mut out = vec![0u8; pixels * 2];
            for i in 0..pixels {
                let r = (rgb[i*3] >> 4) as u16;
                let g = (rgb[i*3+1] >> 4) as u16;
                let b = (rgb[i*3+2] >> 4) as u16;
                let v = (r << 8) | (g << 4) | b;
                let bytes = v.to_le_bytes();
                out[i*2] = bytes[0]; out[i*2+1] = bytes[1];
            }
            Ok(out)
        }
        "RGB1" => { // RGB332
            let mut out = vec![0u8; pixels];
            for i in 0..pixels {
                out[i] = (rgb[i*3] & 0xE0) | ((rgb[i*3+1] >> 3) & 0x1C) | (rgb[i*3+2] >> 6);
            }
            Ok(out)
        }
        _ => Err(format!("Conversion to RGB variant '{}' ({}) not supported", fmt.name, fc)),
    }
}

/// Bayer 8-bit → RGB24 via simple bilinear demosaic.
fn bayer_to_rgb24(frame: &[u8], w: u32, h: u32, fmt: &VideoFormat) -> Result<Vec<u8>, String> {
    let w = w as usize;
    let h = h as usize;
    let mut rgb = vec![0u8; w * h * 3];
    let fc = fmt.fourcc.as_str();

    // Determine pattern: which color is at (0,0)?
    // RGGB: R at (0,0), G at (0,1)/(1,0), B at (1,1)
    // BGGR: B at (0,0), G at (0,1)/(1,0), R at (1,1)
    // GBRG: G at (0,0), B at (0,1), R at (1,0), G at (1,1)
    // GRBG: G at (0,0), R at (0,1), B at (1,0), G at (1,1)
    let (r_row, r_col, b_row, b_col) = match fc {
        "RGGB" => (0usize, 0usize, 1usize, 1usize),
        "BGGR" => (1, 1, 0, 0),
        "GBRG" => (1, 0, 0, 1),
        "GRBG" => (0, 1, 1, 0),
        _ => return Err(format!("Unsupported Bayer pattern: {fc}")),
    };

    // Simple nearest-neighbor demosaic (good enough for conversion)
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let yr = y % 2;
            let xr = x % 2;

            let (r, g, b);
            if yr == r_row && xr == r_col {
                // Red pixel
                r = frame[idx];
                g = avg_neighbors_cross(frame, w, h, x, y);
                b = avg_neighbors_diag(frame, w, h, x, y);
            } else if yr == b_row && xr == b_col {
                // Blue pixel
                b = frame[idx];
                g = avg_neighbors_cross(frame, w, h, x, y);
                r = avg_neighbors_diag(frame, w, h, x, y);
            } else {
                // Green pixel
                g = frame[idx];
                if yr == r_row {
                    r = avg_neighbors_h(frame, w, x, y);
                    b = avg_neighbors_v(frame, w, h, x, y);
                } else {
                    b = avg_neighbors_h(frame, w, x, y);
                    r = avg_neighbors_v(frame, w, h, x, y);
                }
            }

            rgb[idx * 3] = r;
            rgb[idx * 3 + 1] = g;
            rgb[idx * 3 + 2] = b;
        }
    }
    Ok(rgb)
}

fn avg_neighbors_cross(data: &[u8], w: usize, h: usize, x: usize, y: usize) -> u8 {
    let mut sum = 0u32;
    let mut count = 0u32;
    if x > 0 { sum += data[y * w + x - 1] as u32; count += 1; }
    if x + 1 < w { sum += data[y * w + x + 1] as u32; count += 1; }
    if y > 0 { sum += data[(y - 1) * w + x] as u32; count += 1; }
    if y + 1 < h { sum += data[(y + 1) * w + x] as u32; count += 1; }
    if count == 0 { data[y * w + x] } else { (sum / count) as u8 }
}

fn avg_neighbors_diag(data: &[u8], w: usize, h: usize, x: usize, y: usize) -> u8 {
    let mut sum = 0u32;
    let mut count = 0u32;
    if x > 0 && y > 0 { sum += data[(y-1) * w + x-1] as u32; count += 1; }
    if x+1 < w && y > 0 { sum += data[(y-1) * w + x+1] as u32; count += 1; }
    if x > 0 && y+1 < h { sum += data[(y+1) * w + x-1] as u32; count += 1; }
    if x+1 < w && y+1 < h { sum += data[(y+1) * w + x+1] as u32; count += 1; }
    if count == 0 { data[y * w + x] } else { (sum / count) as u8 }
}

fn avg_neighbors_h(data: &[u8], w: usize, x: usize, y: usize) -> u8 {
    let mut sum = 0u32;
    let mut count = 0u32;
    if x > 0 { sum += data[y * w + x - 1] as u32; count += 1; }
    if x + 1 < w { sum += data[y * w + x + 1] as u32; count += 1; }
    if count == 0 { data[y * w + x] } else { (sum / count) as u8 }
}

fn avg_neighbors_v(data: &[u8], w: usize, h: usize, x: usize, y: usize) -> u8 {
    let mut sum = 0u32;
    let mut count = 0u32;
    if y > 0 { sum += data[(y - 1) * w + x] as u32; count += 1; }
    if y + 1 < h { sum += data[(y + 1) * w + x] as u32; count += 1; }
    if count == 0 { data[y * w + x] } else { (sum / count) as u8 }
}

/// High bit-depth Bayer (10/12/16-bit, 2 bytes LE per pixel) → RGB24.
/// Truncates to 8-bit, then runs nearest-neighbor demosaic.
fn bayer_highbit_to_rgb24(frame: &[u8], w: u32, h: u32, fmt: &VideoFormat) -> Result<Vec<u8>, String> {
    let pixels = (w as usize) * (h as usize);
    let shift = fmt.bit_depth.saturating_sub(8);
    // Check for MIPI CSI-2 packed (10-bit packed = 5 bytes per 4 pixels)
    let is_packed = matches!(fmt.fourcc.as_str(), "pRAA" | "pBAA" | "pGAA" | "pGBA");

    let data_8bit: Vec<u8> = if is_packed {
        // MIPI CSI-2 10-bit packed: 4 pixels in 5 bytes
        let mut out = vec![0u8; pixels];
        let mut si = 0;
        let mut di = 0;
        while di + 3 < pixels && si + 4 < frame.len() {
            // Top 8 bits of each 10-bit sample
            out[di]     = frame[si];
            out[di + 1] = frame[si + 1];
            out[di + 2] = frame[si + 2];
            out[di + 3] = frame[si + 3];
            si += 5;
            di += 4;
        }
        out
    } else {
        // 16-bit LE container
        let mut out = vec![0u8; pixels];
        for i in 0..pixels {
            let lo = frame[i * 2] as u16;
            let hi = frame[i * 2 + 1] as u16;
            let val = lo | (hi << 8);
            out[i] = (val >> shift) as u8;
        }
        out
    };

    // Map fourcc to base 8-bit pattern
    let base_pattern = match fmt.fourcc.as_str() {
        "RG10" | "RG12" | "RG16" | "pRAA" => "RGGB",
        "BG10" | "BG12" | "BG16" | "pBAA" => "BGGR",
        "GB10" | "GB12" | "GB16" | "pGAA" => "GBRG",
        "BA10" | "BA12" | "GR16" | "pGBA" => "GRBG",
        _ => return Err(format!("Unknown high-bit Bayer pattern: {}", fmt.fourcc)),
    };

    let mut fake_fmt = fmt.clone();
    fake_fmt.fourcc = base_pattern.to_string();
    fake_fmt.bit_depth = 8;
    bayer_to_rgb24(&data_8bit, w, h, &fake_fmt)
}

/// High bit-depth greyscale (10/12/16-bit, 2 bytes LE) → RGB24.
fn grey_highbit_to_rgb24(frame: &[u8], w: u32, h: u32, fmt: &VideoFormat) -> Result<Vec<u8>, String> {
    let pixels = (w as usize) * (h as usize);
    let shift = fmt.bit_depth.saturating_sub(8);
    let mut rgb = vec![0u8; pixels * 3];
    for i in 0..pixels {
        let lo = frame[i * 2] as u16;
        let hi = frame[i * 2 + 1] as u16;
        let val = ((lo | (hi << 8)) >> shift) as u8;
        rgb[i * 3] = val;
        rgb[i * 3 + 1] = val;
        rgb[i * 3 + 2] = val;
    }
    Ok(rgb)
}

/// RGB24 → high bit-depth greyscale (10/12/16-bit, 2 bytes LE).
fn rgb24_to_grey_highbit(rgb: &[u8], w: u32, h: u32, fmt: &VideoFormat) -> Result<Vec<u8>, String> {
    let pixels = (w as usize) * (h as usize);
    let shift = fmt.bit_depth.saturating_sub(8);
    let mut out = vec![0u8; pixels * 2];
    for i in 0..pixels {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        let grey = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0) as u16;
        let val = grey << shift;
        out[i * 2] = val as u8;
        out[i * 2 + 1] = (val >> 8) as u8;
    }
    Ok(out)
}

fn mat_to_rgb_vec(mat: &opencv::core::Mat) -> Result<Vec<u8>, String> {
    #[allow(unused_imports)]
    use opencv::prelude::*;
    let total = mat.total();
    let channels = mat.channels();
    let n = total * channels as usize;
    let mut out = vec![0u8; n];
    unsafe {
        let ptr = mat.data();
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n);
    }
    Ok(out)
}
