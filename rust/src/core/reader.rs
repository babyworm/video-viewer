use std::collections::HashMap;
use std::path::Path;
use memmap2::Mmap;
use std::fs::File;

use crate::core::cache::FrameCache;
use crate::core::formats::{VideoFormat, FormatType, get_format_by_name};
use crate::core::hints::parse_filename_hints;
use crate::core::y4m::{parse_y4m_header, build_frame_offsets};

/// Default cache budget: 512 MiB.
const DEFAULT_CACHE_BYTES: usize = 512 * 1024 * 1024;

/// Backing store for file data — either a memory-mapped view or a heap buffer.
enum FileData {
    Mmap(Mmap),
    Heap(Vec<u8>),
}

impl FileData {
    fn as_slice(&self) -> &[u8] {
        match self {
            FileData::Mmap(m) => m.as_ref(),
            FileData::Heap(v) => v.as_slice(),
        }
    }
}

pub struct VideoReader {
    data: FileData,
    width: u32,
    height: u32,
    format: &'static VideoFormat,
    frame_size: usize,
    total_frames: usize,
    cache: FrameCache,
    /// Byte offsets to each frame's pixel data (Y4M only).
    frame_offsets: Vec<usize>,
    /// Frame rate from Y4M header (fps_num / fps_den).
    y4m_fps: Option<f64>,
    pub color_matrix: String,
    is_y4m: bool,
}

impl VideoReader {
    /// Open a raw YUV or Y4M file.
    ///
    /// For Y4M files the header is parsed automatically; `width`, `height`, and
    /// `format_name` are ignored (pass 0 / "" if unknown).
    ///
    /// For raw files, if `width == 0` the function tries to infer resolution and
    /// format from the filename via [`parse_filename_hints`].
    pub fn open(
        path: &str,
        mut width: u32,
        mut height: u32,
        mut format_name: &str,
        color_matrix: &str,
    ) -> Result<Self, String> {
        // ------------------------------------------------------------------
        // 1. Read file into memory (mmap preferred, Vec<u8> as fallback).
        // ------------------------------------------------------------------
        let file = File::open(path)
            .map_err(|e| format!("Cannot open '{}': {e}", path))?;

        let data = match unsafe { Mmap::map(&file) } {
            Ok(m) => FileData::Mmap(m),
            Err(_) => {
                use std::io::Read;
                let mut buf = Vec::new();
                let mut f = File::open(path)
                    .map_err(|e| format!("Cannot re-open '{}': {e}", path))?;
                f.read_to_end(&mut buf)
                    .map_err(|e| format!("Cannot read '{}': {e}", path))?;
                FileData::Heap(buf)
            }
        };

        let raw = data.as_slice();

        // ------------------------------------------------------------------
        // 2. Detect Y4M vs raw.
        // ------------------------------------------------------------------
        let is_y4m = {
            let ext = Path::new(path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            ext == "y4m" || raw.starts_with(b"YUV4MPEG2")
        };

        let mut frame_offsets: Vec<usize> = Vec::new();
        let mut y4m_fps: Option<f64> = None;

        let format: &'static VideoFormat;
        let frame_size: usize;
        let total_frames: usize;

        if is_y4m {
            // --------------------------------------------------------------
            // 3a. Y4M path — parse header, build offset table.
            // --------------------------------------------------------------
            let header = parse_y4m_header(raw)?;
            width = header.width;
            height = header.height;
            if header.fps_den != 0 {
                y4m_fps = Some(header.fps());
            }
            let fmt_name = header.to_format_name();
            format = get_format_by_name(fmt_name)
                .ok_or_else(|| format!("Unknown Y4M format name '{fmt_name}'"))?;

            frame_size = format.frame_size(width, height);
            frame_offsets = build_frame_offsets(raw, frame_size);
            total_frames = frame_offsets.len();
        } else {
            // --------------------------------------------------------------
            // 3b. Raw path — apply filename hints if width not provided.
            // --------------------------------------------------------------
            if width == 0 {
                let hints = parse_filename_hints(path);
                if let (Some(w), Some(h)) = (hints.width, hints.height) {
                    width = w;
                    height = h;
                }
                if let Some(ref fmt) = hints.format {
                    if format_name.is_empty() {
                        format_name = Box::leak(fmt.clone().into_boxed_str());
                    }
                }
            }

            if width == 0 || height == 0 {
                return Err("Width and height must be provided for raw files".to_string());
            }

            let fmt_key = if format_name.is_empty() { "I420" } else { format_name };
            format = get_format_by_name(fmt_key)
                .ok_or_else(|| format!("Unknown format '{fmt_key}'"))?;

            frame_size = format.frame_size(width, height);
            if frame_size == 0 {
                return Err(format!("frame_size is 0 for format '{}'", format.name));
            }
            total_frames = raw.len() / frame_size;
        }

        Ok(VideoReader {
            data,
            width,
            height,
            format,
            frame_size,
            total_frames,
            cache: FrameCache::new(DEFAULT_CACHE_BYTES),
            frame_offsets,
            y4m_fps,
            color_matrix: color_matrix.to_string(),
            is_y4m,
        })
    }

    // ------------------------------------------------------------------
    // Getters
    // ------------------------------------------------------------------
    pub fn width(&self) -> u32 { self.width }
    pub fn height(&self) -> u32 { self.height }
    pub fn total_frames(&self) -> usize { self.total_frames }
    pub fn is_y4m(&self) -> bool { self.is_y4m }
    pub fn format_name(&self) -> &str { &self.format.name }
    pub fn format(&self) -> &crate::core::formats::VideoFormat { &self.format }
    pub fn y4m_fps(&self) -> Option<f64> { self.y4m_fps }

    // ------------------------------------------------------------------
    // Frame access
    // ------------------------------------------------------------------

    /// Seek to frame `idx` and return its raw pixel bytes.
    pub fn seek_frame(&mut self, idx: usize) -> Result<Vec<u8>, String> {
        if idx >= self.total_frames {
            return Err(format!(
                "Frame index {idx} out of bounds (total_frames={})",
                self.total_frames
            ));
        }

        // Cache hit?
        if let Some(cached) = self.cache.get(idx) {
            return Ok(cached.clone());
        }

        // Determine byte offset.
        let offset = if self.is_y4m {
            *self.frame_offsets.get(idx).ok_or_else(|| {
                format!("Y4M frame offset missing for index {idx}")
            })?
        } else {
            idx * self.frame_size
        };

        let raw = self.data.as_slice();
        let end = offset + self.frame_size;
        if end > raw.len() {
            return Err(format!(
                "Frame {idx}: data range {offset}..{end} exceeds file length {}",
                raw.len()
            ));
        }
        let frame_data = raw[offset..end].to_vec();
        self.cache.put(idx, frame_data.clone());
        Ok(frame_data)
    }

    // ------------------------------------------------------------------
    // Color conversion
    // ------------------------------------------------------------------

    /// Convert raw frame bytes to packed RGB (H × W × 3).
    pub fn convert_to_rgb(&self, raw: &[u8]) -> Result<Vec<u8>, String> {
        #[allow(unused_imports)]
        use opencv::prelude::*;
        use opencv::core::{Mat, CV_8UC1, CV_8UC2, CV_8UC3};
        use opencv::imgproc;

        let w = self.width as i32;
        let h = self.height as i32;

        match self.format.format_type {
            // ----------------------------------------------------------
            // I420 / YV12  (4:2:0 planar)
            // ----------------------------------------------------------
            FormatType::YuvPlanar if matches!(self.format.fourcc.as_str(), "YU12" | "YV12") => {
                let expected = self.frame_size;
                if raw.len() < expected {
                    return Err(format!("Raw data too short: {} < {}", raw.len(), expected));
                }
                // OpenCV YUV2RGB_I420 expects the entire YUV420 image as a (h*3/2 x w) Mat.
                let yuv_mat = unsafe {
                    Mat::new_rows_cols_with_data_unsafe(
                        h * 3 / 2,
                        w,
                        CV_8UC1,
                        raw.as_ptr() as *mut std::ffi::c_void,
                        opencv::core::Mat_AUTO_STEP,
                    )
                    .map_err(|e| e.to_string())?
                };
                let code = if self.format.fourcc == "YU12" {
                    imgproc::COLOR_YUV2RGB_I420
                } else {
                    imgproc::COLOR_YUV2RGB_YV12
                };
                let mut rgb_mat = Mat::default();
                imgproc::cvt_color(&yuv_mat, &mut rgb_mat, code, 0)
                    .map_err(|e| e.to_string())?;
                mat_to_vec(&rgb_mat)
            }

            // ----------------------------------------------------------
            // NV12 / NV21  (4:2:0 semi-planar)
            // ----------------------------------------------------------
            FormatType::YuvSemiPlanar
                if matches!(self.format.fourcc.as_str(), "NV12" | "NV21") =>
            {
                let yuv_mat = unsafe {
                    Mat::new_rows_cols_with_data_unsafe(
                        h * 3 / 2,
                        w,
                        CV_8UC1,
                        raw.as_ptr() as *mut std::ffi::c_void,
                        opencv::core::Mat_AUTO_STEP,
                    )
                    .map_err(|e| e.to_string())?
                };
                let code = if self.format.fourcc == "NV12" {
                    imgproc::COLOR_YUV2RGB_NV12
                } else {
                    imgproc::COLOR_YUV2RGB_NV21
                };
                let mut rgb_mat = Mat::default();
                imgproc::cvt_color(&yuv_mat, &mut rgb_mat, code, 0)
                    .map_err(|e| e.to_string())?;
                mat_to_vec(&rgb_mat)
            }

            // ----------------------------------------------------------
            // YUYV / UYVY  (4:2:2 packed)
            // ----------------------------------------------------------
            FormatType::YuvPacked
                if matches!(self.format.fourcc.as_str(), "YUYV" | "UYVY") =>
            {
                // OpenCV expects (h x w) Mat with 2 bytes per pixel (CV_8UC2).
                let yuv_mat = unsafe {
                    Mat::new_rows_cols_with_data_unsafe(
                        h,
                        w,
                        CV_8UC2,
                        raw.as_ptr() as *mut std::ffi::c_void,
                        opencv::core::Mat_AUTO_STEP,
                    )
                    .map_err(|e| e.to_string())?
                };
                let code = if self.format.fourcc == "YUYV" {
                    imgproc::COLOR_YUV2RGB_YUYV
                } else {
                    imgproc::COLOR_YUV2RGB_UYVY
                };
                let mut rgb_mat = Mat::default();
                imgproc::cvt_color(&yuv_mat, &mut rgb_mat, code, 0)
                    .map_err(|e| e.to_string())?;
                mat_to_vec(&rgb_mat)
            }

            // ----------------------------------------------------------
            // RGB24 — already RGB, just copy.
            // ----------------------------------------------------------
            FormatType::Rgb if self.format.fourcc == "RGB3" => {
                let expected = (w * h * 3) as usize;
                if raw.len() < expected {
                    return Err(format!("Raw data too short: {} < {}", raw.len(), expected));
                }
                Ok(raw[..expected].to_vec())
            }

            // ----------------------------------------------------------
            // BGR24
            // ----------------------------------------------------------
            FormatType::Rgb if self.format.fourcc == "BGR3" => {
                let bgr_mat = unsafe {
                    Mat::new_rows_cols_with_data_unsafe(
                        h,
                        w,
                        CV_8UC3,
                        raw.as_ptr() as *mut std::ffi::c_void,
                        opencv::core::Mat_AUTO_STEP,
                    )
                    .map_err(|e| e.to_string())?
                };
                let mut rgb_mat = Mat::default();
                imgproc::cvt_color(&bgr_mat, &mut rgb_mat, imgproc::COLOR_BGR2RGB, 0)
                    .map_err(|e| e.to_string())?;
                mat_to_vec(&rgb_mat)
            }

            // ----------------------------------------------------------
            // Greyscale 8-bit
            // ----------------------------------------------------------
            FormatType::Grey if self.format.bit_depth == 8 => {
                let grey_mat = unsafe {
                    Mat::new_rows_cols_with_data_unsafe(
                        h,
                        w,
                        CV_8UC1,
                        raw.as_ptr() as *mut std::ffi::c_void,
                        opencv::core::Mat_AUTO_STEP,
                    )
                    .map_err(|e| e.to_string())?
                };
                let mut rgb_mat = Mat::default();
                imgproc::cvt_color(&grey_mat, &mut rgb_mat, imgproc::COLOR_GRAY2RGB, 0)
                    .map_err(|e| e.to_string())?;
                mat_to_vec(&rgb_mat)
            }

            // ----------------------------------------------------------
            // 422P (planar 4:2:2)
            // ----------------------------------------------------------
            FormatType::YuvPlanar if self.format.fourcc == "422P" => {
                self.convert_yuv_planar_manual(raw, w as usize, h as usize, 2, 1)
            }

            // ----------------------------------------------------------
            // 444P (planar 4:4:4)
            // ----------------------------------------------------------
            FormatType::YuvPlanar if self.format.fourcc == "444P" => {
                self.convert_yuv_planar_manual(raw, w as usize, h as usize, 1, 1)
            }

            _ => Err(format!(
                "convert_to_rgb: unsupported format '{}' ({})",
                self.format.name, self.format.fourcc
            )),
        }
    }

    /// Manual YUV planar → RGB for 422P and 444P.
    ///
    /// `h_sub` and `v_sub` are the horizontal and vertical chroma subsampling
    /// factors (e.g. 2,1 for 4:2:2; 1,1 for 4:4:4).
    fn convert_yuv_planar_manual(
        &self,
        raw: &[u8],
        w: usize,
        h: usize,
        h_sub: usize,
        v_sub: usize,
    ) -> Result<Vec<u8>, String> {
        let y_size = w * h;
        let uv_w = w / h_sub;
        let uv_h = h / v_sub;
        let uv_size = uv_w * uv_h;

        if raw.len() < y_size + uv_size * 2 {
            return Err(format!(
                "Raw data too short for planar YUV: {} < {}",
                raw.len(),
                y_size + uv_size * 2
            ));
        }

        let y_plane = &raw[..y_size];
        let u_plane = &raw[y_size..y_size + uv_size];
        let v_plane = &raw[y_size + uv_size..y_size + uv_size * 2];

        let mut rgb = vec![0u8; w * h * 3];

        if self.color_matrix == "BT.709" {
            for py in 0..h {
                for px in 0..w {
                    let y_val = y_plane[py * w + px] as f32;
                    let cx = px / h_sub;
                    let cy = py / v_sub;
                    let uv_idx = cy * uv_w + cx;
                    let u_val = u_plane[uv_idx] as f32 - 128.0;
                    let v_val = v_plane[uv_idx] as f32 - 128.0;

                    let r = (y_val + 1.5748 * v_val).clamp(0.0, 255.0) as u8;
                    let g = (y_val - 0.1873 * u_val - 0.4681 * v_val).clamp(0.0, 255.0) as u8;
                    let b = (y_val + 1.8556 * u_val).clamp(0.0, 255.0) as u8;

                    let out_idx = (py * w + px) * 3;
                    rgb[out_idx] = r;
                    rgb[out_idx + 1] = g;
                    rgb[out_idx + 2] = b;
                }
            }
        } else {
            // BT.601 coefficients
            for py in 0..h {
                for px in 0..w {
                    let y_val = y_plane[py * w + px] as f32;
                    let cx = px / h_sub;
                    let cy = py / v_sub;
                    let uv_idx = cy * uv_w + cx;
                    let u_val = u_plane[uv_idx] as f32 - 128.0;
                    let v_val = v_plane[uv_idx] as f32 - 128.0;

                    let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
                    let g = (y_val - 0.344136 * u_val - 0.714136 * v_val)
                        .clamp(0.0, 255.0) as u8;
                    let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;

                    let out_idx = (py * w + px) * 3;
                    rgb[out_idx] = r;
                    rgb[out_idx + 1] = g;
                    rgb[out_idx + 2] = b;
                }
            }
        }

        Ok(rgb)
    }

    // ------------------------------------------------------------------
    // Channel extraction
    // ------------------------------------------------------------------

    /// Extract per-channel images as full-resolution `Vec<u8>`.
    ///
    /// YUV formats return keys "Y", "U", "V" (U and V upsampled to full res).
    /// RGB / Bayer / Grey formats return keys "R", "G", "B".
    pub fn get_channels(&self, raw: &[u8]) -> HashMap<String, Vec<u8>> {
        let mut channels = HashMap::new();
        let w = self.width as usize;
        let h = self.height as usize;

        match self.format.format_type {
            FormatType::YuvPlanar => {
                let y_size = w * h;
                let fc = self.format.fourcc.as_str();

                let y = raw[..y_size].to_vec();
                channels.insert("Y".to_string(), y);

                let (u_plane, v_plane, uv_w, uv_h) = match fc {
                    "YU12" => {
                        let uv_size = y_size / 4;
                        (
                            raw[y_size..y_size + uv_size].to_vec(),
                            raw[y_size + uv_size..y_size + uv_size * 2].to_vec(),
                            w / 2,
                            h / 2,
                        )
                    }
                    "YV12" => {
                        let uv_size = y_size / 4;
                        // YV12: V before U
                        (
                            raw[y_size + uv_size..y_size + uv_size * 2].to_vec(),
                            raw[y_size..y_size + uv_size].to_vec(),
                            w / 2,
                            h / 2,
                        )
                    }
                    "422P" => {
                        let uv_size = y_size / 2;
                        (
                            raw[y_size..y_size + uv_size].to_vec(),
                            raw[y_size + uv_size..y_size + uv_size * 2].to_vec(),
                            w / 2,
                            h,
                        )
                    }
                    "444P" => {
                        (
                            raw[y_size..y_size * 2].to_vec(),
                            raw[y_size * 2..y_size * 3].to_vec(),
                            w,
                            h,
                        )
                    }
                    _ => return channels,
                };

                channels.insert("U".to_string(), nearest_upsample(&u_plane, uv_w, uv_h, w, h));
                channels.insert("V".to_string(), nearest_upsample(&v_plane, uv_w, uv_h, w, h));
            }

            FormatType::YuvSemiPlanar => {
                let y_size = w * h;
                let y = raw[..y_size].to_vec();
                channels.insert("Y".to_string(), y);

                let fc = self.format.fourcc.as_str();
                // NV12/NV21: interleaved UV after Y plane
                if matches!(fc, "NV12" | "NV21") {
                    let uv_len = y_size / 2;
                    let uv_data = &raw[y_size..y_size + uv_len];
                    let uv_w = w / 2;
                    let uv_h = h / 2;
                    let mut u_small = vec![0u8; uv_w * uv_h];
                    let mut v_small = vec![0u8; uv_w * uv_h];
                    for i in 0..(uv_w * uv_h) {
                        if fc == "NV12" {
                            u_small[i] = uv_data[i * 2];
                            v_small[i] = uv_data[i * 2 + 1];
                        } else {
                            v_small[i] = uv_data[i * 2];
                            u_small[i] = uv_data[i * 2 + 1];
                        }
                    }
                    channels.insert("U".to_string(), nearest_upsample(&u_small, uv_w, uv_h, w, h));
                    channels.insert("V".to_string(), nearest_upsample(&v_small, uv_w, uv_h, w, h));
                } else {
                    // Fallback: convert to RGB then split as YUV
                    if let Ok(rgb) = self.convert_to_rgb(raw) {
                        if let Some(yuv) = rgb_to_ycbcr(&rgb, w, h) {
                            channels.insert("U".to_string(), yuv.1);
                            channels.insert("V".to_string(), yuv.2);
                        }
                    }
                }
            }

            FormatType::YuvPacked
            | FormatType::Rgb
            | FormatType::Bayer
            | FormatType::Grey
            | FormatType::Compressed => {
                if let Ok(rgb) = self.convert_to_rgb(raw) {
                    let pixels = w * h;
                    let mut r = vec![0u8; pixels];
                    let mut g = vec![0u8; pixels];
                    let mut b = vec![0u8; pixels];
                    for i in 0..pixels {
                        r[i] = rgb[i * 3];
                        g[i] = rgb[i * 3 + 1];
                        b[i] = rgb[i * 3 + 2];
                    }
                    channels.insert("R".to_string(), r);
                    channels.insert("G".to_string(), g);
                    channels.insert("B".to_string(), b);
                }
            }
        }

        channels
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Copy the data bytes from an OpenCV Mat into a Vec<u8>.
fn mat_to_vec(mat: &opencv::core::Mat) -> Result<Vec<u8>, String> {
    #[allow(unused_imports)]
    use opencv::prelude::*;
    let total = mat.total();
    let channels = mat.channels();
    let n = total * channels as usize;
    let mut out = vec![0u8; n];
    // SAFETY: Mat data pointer is valid for `n` bytes.
    unsafe {
        let ptr = mat.data();
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n);
    }
    Ok(out)
}

/// Nearest-neighbour upsample a planar greyscale image.
fn nearest_upsample(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    if src_w == dst_w && src_h == dst_h {
        return src.to_vec();
    }
    let mut dst = vec![0u8; dst_w * dst_h];
    for dy in 0..dst_h {
        let sy = (dy * src_h) / dst_h;
        for dx in 0..dst_w {
            let sx = (dx * src_w) / dst_w;
            dst[dy * dst_w + dx] = src[sy * src_w + sx];
        }
    }
    dst
}

/// Convert packed RGB to YCbCr (BT.601) and return (Y, Cb, Cr) full-res planes.
fn rgb_to_ycbcr(rgb: &[u8], w: usize, h: usize) -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let pixels = w * h;
    if rgb.len() < pixels * 3 {
        return None;
    }
    let mut y = vec![0u8; pixels];
    let mut cb = vec![0u8; pixels];
    let mut cr = vec![0u8; pixels];
    for i in 0..pixels {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        y[i] = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0) as u8;
        cb[i] = (128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b).clamp(0.0, 255.0) as u8;
        cr[i] = (128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b).clamp(0.0, 255.0) as u8;
    }
    Some((y, cb, cr))
}
