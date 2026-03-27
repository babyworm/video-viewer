use std::collections::HashMap;
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatType {
    YuvPlanar,
    YuvSemiPlanar,
    YuvPacked,
    Bayer,
    Rgb,
    Grey,
    Compressed,
}

#[derive(Debug, Clone)]
pub struct VideoFormat {
    pub name: String,
    pub display_name: String,
    pub fourcc: String,
    pub format_type: FormatType,
    pub bit_depth: u32,
    pub subsampling: (u32, u32), // (h_sub, v_sub)
    pub bpp: u32,                // bits per pixel (average)
    pub planes: u32,
    pub v4l2_name: String,
}

impl VideoFormat {
    pub fn frame_size(&self, width: u32, height: u32) -> usize {
        let w = width as usize;
        let h = height as usize;
        let y_size = w * h;

        match self.format_type {
            FormatType::YuvPlanar => {
                let fc = self.fourcc.as_str();
                match fc {
                    "YU12" | "YV12" | "YM12" | "YM21" => y_size * 3 / 2,
                    "422P" | "YM16" | "YM61" => y_size * 2,
                    "Y444" | "444P" => y_size * 3,
                    // 10-bit planar: each sample in 16-bit LE word
                    "0T20" => y_size * 3,       // Y*2 + U*0.5 + V*0.5
                    "2T22" => y_size * 4,       // Y*2 + U*1 + V*1
                    "4T44" => y_size * 6,       // Y*2 + U*2 + V*2
                    "YUV9" | "YVU9" => {
                        y_size + (w / 4) * (h / 4) * 2
                    }
                    "411P" => {
                        let u_size = (w / self.subsampling.0 as usize)
                            * (h / self.subsampling.1 as usize);
                        y_size + u_size * 2
                    }
                    _ => {
                        let u_size = (w / self.subsampling.0 as usize)
                            * (h / self.subsampling.1 as usize);
                        y_size + u_size * 2
                    }
                }
            }
            FormatType::YuvSemiPlanar => {
                let fc = self.fourcc.as_str();
                match fc {
                    "NV12" | "NV21" | "NM12" | "NM21" => y_size * 3 / 2,
                    "NV16" | "NV61" => y_size * 2,
                    "NV24" | "NV42" => y_size * 3,
                    "P010" | "P016" => y_size * 3, // 16-bit per sample, 4:2:0
                    "P210" => y_size * 4,          // 16-bit per sample, 4:2:2
                    _ => y_size * 3 / 2,
                }
            }
            FormatType::YuvPacked => {
                let fc = self.fourcc.as_str();
                match fc {
                    "YUYV" | "UYVY" | "YVYU" | "VYUY" => y_size * 2,
                    "AYUV" | "VUYA" => y_size * 4,
                    "Y41P" => y_size * 3 / 2,
                    "Y210" => y_size * 4,
                    _ => y_size * 2,
                }
            }
            FormatType::Bayer => {
                if self.bit_depth == 8 {
                    y_size
                } else if ["pRAA", "pBAA", "pGAA", "pGBA"]
                    .contains(&self.fourcc.as_str())
                {
                    (y_size * 5) / 4 // MIPI CSI-2 packed
                } else {
                    y_size * 2 // 16-bit container
                }
            }
            FormatType::Rgb => {
                if self.bpp > 0 {
                    y_size * self.bpp as usize / 8
                } else {
                    0
                }
            }
            FormatType::Grey => {
                if self.bit_depth == 8 {
                    y_size
                } else {
                    y_size * 2
                }
            }
            FormatType::Compressed => 0,
        }
    }
}

struct FormatEntry {
    name: &'static str,
    format_type: FormatType,
    fourcc: &'static str,
    bit_depth: u32,
    subsampling: (u32, u32),
    bpp: u32,
    planes: u32,
    v4l2_name: &'static str,
}

const FORMAT_DEFS: &[FormatEntry] = &[
    // --- Packed YUV ---
    FormatEntry { name: "YUYV (4:2:2)", format_type: FormatType::YuvPacked, fourcc: "YUYV", bit_depth: 8, subsampling: (2, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_YUYV" },
    FormatEntry { name: "UYVY (4:2:2)", format_type: FormatType::YuvPacked, fourcc: "UYVY", bit_depth: 8, subsampling: (2, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_UYVY" },
    FormatEntry { name: "YVYU (4:2:2)", format_type: FormatType::YuvPacked, fourcc: "YVYU", bit_depth: 8, subsampling: (2, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_YVYU" },
    FormatEntry { name: "VYUY (4:2:2)", format_type: FormatType::YuvPacked, fourcc: "VYUY", bit_depth: 8, subsampling: (2, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_VYUY" },
    FormatEntry { name: "Y41P (4:1:1)", format_type: FormatType::YuvPacked, fourcc: "Y41P", bit_depth: 8, subsampling: (4, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_Y41P" },
    FormatEntry { name: "AYUV (4:4:4)", format_type: FormatType::YuvPacked, fourcc: "AYUV", bit_depth: 8, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_AYUV32" },
    FormatEntry { name: "VUYA (4:4:4)", format_type: FormatType::YuvPacked, fourcc: "VUYA", bit_depth: 8, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_VUYA32" },
    FormatEntry { name: "Y210 (10-bit 4:2:2)", format_type: FormatType::YuvPacked, fourcc: "Y210", bit_depth: 10, subsampling: (2, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_Y210" },

    // --- Planar YUV (10-bit, 16-bit LE samples, LSB-aligned) ---
    FormatEntry { name: "YUV420P10LE (10-bit 4:2:0)", format_type: FormatType::YuvPlanar, fourcc: "0T20", bit_depth: 10, subsampling: (2, 2), bpp: 0, planes: 1, v4l2_name: "" },
    FormatEntry { name: "YUV422P10LE (10-bit 4:2:2)", format_type: FormatType::YuvPlanar, fourcc: "2T22", bit_depth: 10, subsampling: (2, 1), bpp: 0, planes: 1, v4l2_name: "" },
    FormatEntry { name: "YUV444P10LE (10-bit 4:4:4)", format_type: FormatType::YuvPlanar, fourcc: "4T44", bit_depth: 10, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "" },

    // --- Planar YUV ---
    FormatEntry { name: "I420 (4:2:0)", format_type: FormatType::YuvPlanar, fourcc: "YU12", bit_depth: 8, subsampling: (2, 2), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_YUV420" },
    FormatEntry { name: "YV12 (4:2:0)", format_type: FormatType::YuvPlanar, fourcc: "YV12", bit_depth: 8, subsampling: (2, 2), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_YVU420" },
    FormatEntry { name: "YUV422P (4:2:2)", format_type: FormatType::YuvPlanar, fourcc: "422P", bit_depth: 8, subsampling: (2, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_YUV422P" },
    FormatEntry { name: "YUV411P (4:1:1)", format_type: FormatType::YuvPlanar, fourcc: "411P", bit_depth: 8, subsampling: (4, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_YUV411P" },
    FormatEntry { name: "YUV444P (4:4:4)", format_type: FormatType::YuvPlanar, fourcc: "444P", bit_depth: 8, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_YUV444" },
    FormatEntry { name: "YUV410 (4:1:0)", format_type: FormatType::YuvPlanar, fourcc: "YUV9", bit_depth: 8, subsampling: (4, 4), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_YUV410" },
    FormatEntry { name: "YVU410 (4:1:0)", format_type: FormatType::YuvPlanar, fourcc: "YVU9", bit_depth: 8, subsampling: (4, 4), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_YVU410" },
    FormatEntry { name: "YUV420M (4:2:0 Multi)", format_type: FormatType::YuvPlanar, fourcc: "YM12", bit_depth: 8, subsampling: (2, 2), bpp: 0, planes: 3, v4l2_name: "V4L2_PIX_FMT_YUV420M" },
    FormatEntry { name: "YUV422M (4:2:2 Multi)", format_type: FormatType::YuvPlanar, fourcc: "YM16", bit_depth: 8, subsampling: (2, 1), bpp: 0, planes: 3, v4l2_name: "V4L2_PIX_FMT_YUV422M" },

    // --- Semi-Planar YUV ---
    FormatEntry { name: "NV12 (4:2:0)", format_type: FormatType::YuvSemiPlanar, fourcc: "NV12", bit_depth: 8, subsampling: (2, 2), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_NV12" },
    FormatEntry { name: "NV21 (4:2:0)", format_type: FormatType::YuvSemiPlanar, fourcc: "NV21", bit_depth: 8, subsampling: (2, 2), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_NV21" },
    FormatEntry { name: "NV16 (4:2:2)", format_type: FormatType::YuvSemiPlanar, fourcc: "NV16", bit_depth: 8, subsampling: (2, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_NV16" },
    FormatEntry { name: "NV61 (4:2:2)", format_type: FormatType::YuvSemiPlanar, fourcc: "NV61", bit_depth: 8, subsampling: (2, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_NV61" },
    FormatEntry { name: "NV24 (4:4:4)", format_type: FormatType::YuvSemiPlanar, fourcc: "NV24", bit_depth: 8, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_NV24" },
    FormatEntry { name: "NV42 (4:4:4)", format_type: FormatType::YuvSemiPlanar, fourcc: "NV42", bit_depth: 8, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_NV42" },
    FormatEntry { name: "P010 (10-bit NV12)", format_type: FormatType::YuvSemiPlanar, fourcc: "P010", bit_depth: 10, subsampling: (2, 2), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_P010" },
    FormatEntry { name: "P016 (16-bit NV12)", format_type: FormatType::YuvSemiPlanar, fourcc: "P016", bit_depth: 16, subsampling: (2, 2), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_P016" },
    FormatEntry { name: "P210 (10-bit NV16)", format_type: FormatType::YuvSemiPlanar, fourcc: "P210", bit_depth: 10, subsampling: (2, 1), bpp: 0, planes: 1, v4l2_name: "" },
    FormatEntry { name: "NV12M (4:2:0 Multi)", format_type: FormatType::YuvSemiPlanar, fourcc: "NM12", bit_depth: 8, subsampling: (2, 2), bpp: 0, planes: 2, v4l2_name: "V4L2_PIX_FMT_NV12M" },
    FormatEntry { name: "NV21M (4:2:0 Multi)", format_type: FormatType::YuvSemiPlanar, fourcc: "NM21", bit_depth: 8, subsampling: (2, 2), bpp: 0, planes: 2, v4l2_name: "V4L2_PIX_FMT_NV21M" },

    // --- RGB ---
    FormatEntry { name: "RGB332 (8-bit)", format_type: FormatType::Rgb, fourcc: "RGB1", bit_depth: 8, subsampling: (1, 1), bpp: 8, planes: 1, v4l2_name: "V4L2_PIX_FMT_RGB332" },
    FormatEntry { name: "RGB444 (16-bit)", format_type: FormatType::Rgb, fourcc: "R444", bit_depth: 8, subsampling: (1, 1), bpp: 16, planes: 1, v4l2_name: "V4L2_PIX_FMT_RGB444" },
    FormatEntry { name: "ARGB444 (16-bit)", format_type: FormatType::Rgb, fourcc: "AR12", bit_depth: 8, subsampling: (1, 1), bpp: 16, planes: 1, v4l2_name: "V4L2_PIX_FMT_ARGB444" },
    FormatEntry { name: "XRGB444 (16-bit)", format_type: FormatType::Rgb, fourcc: "XR12", bit_depth: 8, subsampling: (1, 1), bpp: 16, planes: 1, v4l2_name: "V4L2_PIX_FMT_XRGB444" },
    FormatEntry { name: "RGB555 (16-bit)", format_type: FormatType::Rgb, fourcc: "RGBO", bit_depth: 8, subsampling: (1, 1), bpp: 16, planes: 1, v4l2_name: "V4L2_PIX_FMT_RGB555" },
    FormatEntry { name: "ARGB555 (16-bit)", format_type: FormatType::Rgb, fourcc: "AR15", bit_depth: 8, subsampling: (1, 1), bpp: 16, planes: 1, v4l2_name: "V4L2_PIX_FMT_ARGB555" },
    FormatEntry { name: "XRGB555 (16-bit)", format_type: FormatType::Rgb, fourcc: "XR15", bit_depth: 8, subsampling: (1, 1), bpp: 16, planes: 1, v4l2_name: "V4L2_PIX_FMT_XRGB555" },
    FormatEntry { name: "RGB565 (16-bit)", format_type: FormatType::Rgb, fourcc: "RGBP", bit_depth: 8, subsampling: (1, 1), bpp: 16, planes: 1, v4l2_name: "V4L2_PIX_FMT_RGB565" },
    FormatEntry { name: "RGB555X (16-bit)", format_type: FormatType::Rgb, fourcc: "RGBQ", bit_depth: 8, subsampling: (1, 1), bpp: 16, planes: 1, v4l2_name: "V4L2_PIX_FMT_RGB555X" },
    FormatEntry { name: "RGB565X (16-bit)", format_type: FormatType::Rgb, fourcc: "RGBR", bit_depth: 8, subsampling: (1, 1), bpp: 16, planes: 1, v4l2_name: "V4L2_PIX_FMT_RGB565X" },
    FormatEntry { name: "BGR24 (24-bit)", format_type: FormatType::Rgb, fourcc: "BGR3", bit_depth: 8, subsampling: (1, 1), bpp: 24, planes: 1, v4l2_name: "V4L2_PIX_FMT_BGR24" },
    FormatEntry { name: "RGB24 (24-bit)", format_type: FormatType::Rgb, fourcc: "RGB3", bit_depth: 8, subsampling: (1, 1), bpp: 24, planes: 1, v4l2_name: "V4L2_PIX_FMT_RGB24" },
    FormatEntry { name: "BGR32 (32-bit)", format_type: FormatType::Rgb, fourcc: "BGR4", bit_depth: 8, subsampling: (1, 1), bpp: 32, planes: 1, v4l2_name: "V4L2_PIX_FMT_BGR32" },
    FormatEntry { name: "RGB32 (32-bit)", format_type: FormatType::Rgb, fourcc: "RGB4", bit_depth: 8, subsampling: (1, 1), bpp: 32, planes: 1, v4l2_name: "V4L2_PIX_FMT_RGB32" },
    FormatEntry { name: "ABGR32 (32-bit)", format_type: FormatType::Rgb, fourcc: "AR24", bit_depth: 8, subsampling: (1, 1), bpp: 32, planes: 1, v4l2_name: "V4L2_PIX_FMT_ABGR32" },
    FormatEntry { name: "ARGB32 (32-bit)", format_type: FormatType::Rgb, fourcc: "BA24", bit_depth: 8, subsampling: (1, 1), bpp: 32, planes: 1, v4l2_name: "V4L2_PIX_FMT_ARGB32" },
    FormatEntry { name: "BGRA32 (32-bit)", format_type: FormatType::Rgb, fourcc: "RA24", bit_depth: 8, subsampling: (1, 1), bpp: 32, planes: 1, v4l2_name: "V4L2_PIX_FMT_BGRA32" },
    FormatEntry { name: "RGBA32 (32-bit)", format_type: FormatType::Rgb, fourcc: "AB24", bit_depth: 8, subsampling: (1, 1), bpp: 32, planes: 1, v4l2_name: "V4L2_PIX_FMT_RGBA32" },
    FormatEntry { name: "XBGR32 (32-bit)", format_type: FormatType::Rgb, fourcc: "XR24", bit_depth: 8, subsampling: (1, 1), bpp: 32, planes: 1, v4l2_name: "V4L2_PIX_FMT_XBGR32" },
    FormatEntry { name: "XRGB32 (32-bit)", format_type: FormatType::Rgb, fourcc: "XB24", bit_depth: 8, subsampling: (1, 1), bpp: 32, planes: 1, v4l2_name: "V4L2_PIX_FMT_XRGB32" },
    FormatEntry { name: "BGRX32 (32-bit)", format_type: FormatType::Rgb, fourcc: "RX24", bit_depth: 8, subsampling: (1, 1), bpp: 32, planes: 1, v4l2_name: "V4L2_PIX_FMT_BGRX32" },
    FormatEntry { name: "RGBX32 (32-bit)", format_type: FormatType::Rgb, fourcc: "BX24", bit_depth: 8, subsampling: (1, 1), bpp: 32, planes: 1, v4l2_name: "V4L2_PIX_FMT_RGBX32" },
    FormatEntry { name: "HSV24 (24-bit)", format_type: FormatType::Rgb, fourcc: "HSV3", bit_depth: 8, subsampling: (1, 1), bpp: 24, planes: 1, v4l2_name: "V4L2_PIX_FMT_HSV24" },
    FormatEntry { name: "HSV32 (32-bit)", format_type: FormatType::Rgb, fourcc: "HSV4", bit_depth: 8, subsampling: (1, 1), bpp: 32, planes: 1, v4l2_name: "V4L2_PIX_FMT_HSV32" },

    // --- Bayer 8-bit ---
    FormatEntry { name: "Bayer RGGB (8-bit)", format_type: FormatType::Bayer, fourcc: "RGGB", bit_depth: 8, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SRGGB8" },
    FormatEntry { name: "Bayer BGGR (8-bit)", format_type: FormatType::Bayer, fourcc: "BGGR", bit_depth: 8, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SBGGR8" },
    FormatEntry { name: "Bayer GBRG (8-bit)", format_type: FormatType::Bayer, fourcc: "GBRG", bit_depth: 8, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SGBRG8" },
    FormatEntry { name: "Bayer GRBG (8-bit)", format_type: FormatType::Bayer, fourcc: "GRBG", bit_depth: 8, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SGRBG8" },

    // --- Bayer 10-bit ---
    FormatEntry { name: "Bayer RGGB (10-bit)", format_type: FormatType::Bayer, fourcc: "RG10", bit_depth: 10, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SRGGB10" },
    FormatEntry { name: "Bayer BGGR (10-bit)", format_type: FormatType::Bayer, fourcc: "BG10", bit_depth: 10, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SBGGR10" },
    FormatEntry { name: "Bayer GBRG (10-bit)", format_type: FormatType::Bayer, fourcc: "GB10", bit_depth: 10, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SGBRG10" },
    FormatEntry { name: "Bayer GRBG (10-bit)", format_type: FormatType::Bayer, fourcc: "BA10", bit_depth: 10, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SGRBG10" },

    // --- Bayer 10-bit packed (MIPI CSI-2) ---
    FormatEntry { name: "Bayer RGGB (10-bit packed)", format_type: FormatType::Bayer, fourcc: "pRAA", bit_depth: 10, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SRGGB10P" },
    FormatEntry { name: "Bayer BGGR (10-bit packed)", format_type: FormatType::Bayer, fourcc: "pBAA", bit_depth: 10, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SBGGR10P" },
    FormatEntry { name: "Bayer GBRG (10-bit packed)", format_type: FormatType::Bayer, fourcc: "pGAA", bit_depth: 10, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SGBRG10P" },
    FormatEntry { name: "Bayer GRBG (10-bit packed)", format_type: FormatType::Bayer, fourcc: "pGBA", bit_depth: 10, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SGRBG10P" },

    // --- Bayer 12-bit ---
    FormatEntry { name: "Bayer RGGB (12-bit)", format_type: FormatType::Bayer, fourcc: "RG12", bit_depth: 12, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SRGGB12" },
    FormatEntry { name: "Bayer BGGR (12-bit)", format_type: FormatType::Bayer, fourcc: "BG12", bit_depth: 12, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SBGGR12" },
    FormatEntry { name: "Bayer GBRG (12-bit)", format_type: FormatType::Bayer, fourcc: "GB12", bit_depth: 12, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SGBRG12" },
    FormatEntry { name: "Bayer GRBG (12-bit)", format_type: FormatType::Bayer, fourcc: "BA12", bit_depth: 12, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SGRBG12" },

    // --- Bayer 16-bit ---
    FormatEntry { name: "Bayer RGGB (16-bit)", format_type: FormatType::Bayer, fourcc: "RG16", bit_depth: 16, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SRGGB16" },
    FormatEntry { name: "Bayer BGGR (16-bit)", format_type: FormatType::Bayer, fourcc: "BG16", bit_depth: 16, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SBGGR16" },
    FormatEntry { name: "Bayer GBRG (16-bit)", format_type: FormatType::Bayer, fourcc: "GB16", bit_depth: 16, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SGBRG16" },
    FormatEntry { name: "Bayer GRBG (16-bit)", format_type: FormatType::Bayer, fourcc: "GR16", bit_depth: 16, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_SGRBG16" },

    // --- Grey ---
    FormatEntry { name: "Greyscale (8-bit)", format_type: FormatType::Grey, fourcc: "GREY", bit_depth: 8, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_GREY" },
    FormatEntry { name: "Greyscale (10-bit)", format_type: FormatType::Grey, fourcc: "Y10 ", bit_depth: 10, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_Y10" },
    FormatEntry { name: "Greyscale (12-bit)", format_type: FormatType::Grey, fourcc: "Y12 ", bit_depth: 12, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_Y12" },
    FormatEntry { name: "Greyscale (16-bit)", format_type: FormatType::Grey, fourcc: "Y16 ", bit_depth: 16, subsampling: (1, 1), bpp: 0, planes: 1, v4l2_name: "V4L2_PIX_FMT_Y16" },
];

static FORMAT_REGISTRY: LazyLock<Vec<VideoFormat>> = LazyLock::new(|| {
    FORMAT_DEFS
        .iter()
        .map(|e| {
            let display_name = format!("{} [{}]", e.name, e.fourcc);
            VideoFormat {
                name: e.name.to_string(),
                display_name,
                fourcc: e.fourcc.to_string(),
                format_type: e.format_type,
                bit_depth: e.bit_depth,
                subsampling: e.subsampling,
                bpp: e.bpp,
                planes: e.planes,
                v4l2_name: e.v4l2_name.to_string(),
            }
        })
        .collect()
});

static NAME_INDEX: LazyLock<HashMap<String, usize>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    for (i, fmt) in FORMAT_REGISTRY.iter().enumerate() {
        // Index by display name
        map.insert(fmt.display_name.clone(), i);
        // Index by short name (e.g., "I420 (4:2:0)")
        map.insert(fmt.name.clone(), i);
        // Index by first word of name (e.g., "I420", "NV12", "YUYV")
        if let Some(first) = fmt.name.split_whitespace().next() {
            map.entry(first.to_string()).or_insert(i);
        }
        // Index by fourcc
        map.entry(fmt.fourcc.clone()).or_insert(i);
        // Index by fourcc trimmed (for "Y10 " -> "Y10")
        let trimmed = fmt.fourcc.trim();
        if trimmed != fmt.fourcc {
            map.entry(trimmed.to_string()).or_insert(i);
        }
    }
    map
});

/// Look up a format by name, display name, short name prefix, or fourcc.
pub fn get_format_by_name(key: &str) -> Option<&'static VideoFormat> {
    NAME_INDEX.get(key).map(|&i| &FORMAT_REGISTRY[i])
}

/// Look up a format by fourcc code.
pub fn get_format_by_fourcc(fourcc: &str) -> Option<&'static VideoFormat> {
    FORMAT_REGISTRY.iter().find(|f| f.fourcc == fourcc)
}

/// Get all registered formats.
pub fn get_all_formats() -> &'static [VideoFormat] {
    &FORMAT_REGISTRY
}

/// Get formats filtered by type.
pub fn get_formats_by_type(format_type: FormatType) -> Vec<&'static VideoFormat> {
    FORMAT_REGISTRY
        .iter()
        .filter(|f| f.format_type == format_type)
        .collect()
}

/// Get all format display names, grouped by category.
pub fn get_formats_by_category() -> HashMap<&'static str, Vec<&'static str>> {
    let mut categories: HashMap<&'static str, Vec<&'static str>> = HashMap::new();
    for fmt in FORMAT_REGISTRY.iter() {
        let cat = match fmt.format_type {
            FormatType::YuvPlanar => "YUV Planar",
            FormatType::YuvSemiPlanar => "YUV Semi-Planar",
            FormatType::YuvPacked => "YUV Packed",
            FormatType::Rgb => "RGB",
            FormatType::Bayer => "Bayer",
            FormatType::Grey => "Grey",
            FormatType::Compressed => continue,
        };
        categories
            .entry(cat)
            .or_default()
            .push(&fmt.display_name);
    }
    categories
}
