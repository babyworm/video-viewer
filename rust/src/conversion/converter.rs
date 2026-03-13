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
        _ => {
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
