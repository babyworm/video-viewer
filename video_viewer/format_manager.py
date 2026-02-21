import numpy as np
from enum import Enum, auto

class FormatType(Enum):
    YUV_PLANAR = auto()
    YUV_SEMI_PLANAR = auto()
    YUV_PACKED = auto()
    BAYER = auto()
    RGB = auto()
    GREY = auto()
    COMPRESSED = auto() # MJPEG, H264, etc. (Not fully supported for decoding yet, but listed)

class VideoFormat:
    def __init__(self, name, type, fourcc, bit_depth=8, subsampling=(1,1), bpp=0, planes=1, v4l2_name=""):
        self.name = name
        self.type = type
        self.fourcc = fourcc
        self.bit_depth = bit_depth
        self.subsampling = subsampling  # (h_sub, v_sub) relative to luminance
        self.bpp = bpp # Bits per pixel (average)
        self.planes = planes
        self.v4l2_name = v4l2_name  # e.g. "V4L2_PIX_FMT_YUV420"

    def calculate_frame_size(self, width, height):
        # Generic calculation based on type and bpp/planes if standard
        if self.type == FormatType.YUV_PLANAR:
            y_size = width * height
            # Common 4:2:0
            if self.fourcc in ["YU12", "YV12", "YM12", "YM21"]: return int(y_size * 1.5)
            # Common 4:2:2
            if self.fourcc in ["422P", "YM16", "YM61"]: return y_size * 2
            # Common 4:4:4
            if self.fourcc in ["Y444", "444P"]: return y_size * 3
            # Common 4:1:0 (9 bits per pixel)
            if self.fourcc == "YUV9": return y_size + (width//4)*(height//4)*2
            if self.fourcc == "YVU9": return y_size + (width//4)*(height//4)*2

            # Formatting fallback using subsampling
            u_size = (width // self.subsampling[0]) * (height // self.subsampling[1])
            v_size = u_size
            return y_size + u_size + v_size

        elif self.type == FormatType.YUV_SEMI_PLANAR:
            y_size = width * height
            if self.fourcc in ["NV12", "NV21", "NM12", "NM21"]: return int(y_size * 1.5)
            if self.fourcc in ["NV16", "NV61"]: return y_size * 2
            if self.fourcc in ["NV24", "NV42"]: return y_size * 3
            # P010/P016 - 10-bit/16-bit NV12 with 16-bit per sample storage
            if self.fourcc in ["P010", "P016"]: return y_size * 3
            return int(y_size * 1.5)

        elif self.type == FormatType.YUV_PACKED:
            # 8-bit usually 2 bytes/pixel for 4:2:2
            if self.fourcc in ["YUYV", "UYVY", "YVYU", "VYUY"]: return width * height * 2
            # 4:4:4 packed
            if self.fourcc in ["AYUV", "VUYA"]: return width * height * 4
            # Y41P (4:1:1 packed) -> 12 bits per pixel
            if self.fourcc == "Y41P": return int(width * height * 1.5)
            # Y210 - 10-bit packed YUYV (16-bit per component)
            if self.fourcc == "Y210": return width * height * 4

            return width * height * 2

        elif self.type == FormatType.BAYER:
            base = width * height
            if self.bit_depth == 8: return base
            # Packed 10-bit Bayer (5 bytes for 4 pixels)
            if self.fourcc in ["pRAA", "pBAA", "pGAA", "pGBA"]:
                return (base * 5) // 4
            # 16-bit container for higher bit depths
            if self.bit_depth > 8:
                return base * 2

        elif self.type == FormatType.RGB:
            if self.bpp > 0:
                return int(width * height * (self.bpp / 8.0))

        elif self.type == FormatType.GREY:
            if self.bit_depth == 8: return width * height
            if self.bit_depth > 8: return width * height * 2

        return 0

class FormatManager:
    def __init__(self):
        self.formats = {}
        self._init_formats()

    def _init_formats(self):
        # Reference: https://docs.kernel.org/userspace-api/media/v4l/pixfmt.html

        # --- Packed YUV ---
        self._add("YUYV (4:2:2)", FormatType.YUV_PACKED, "YUYV", subsampling=(2,1), v4l2_name="V4L2_PIX_FMT_YUYV")
        self._add("UYVY (4:2:2)", FormatType.YUV_PACKED, "UYVY", subsampling=(2,1), v4l2_name="V4L2_PIX_FMT_UYVY")
        self._add("YVYU (4:2:2)", FormatType.YUV_PACKED, "YVYU", subsampling=(2,1), v4l2_name="V4L2_PIX_FMT_YVYU")
        self._add("VYUY (4:2:2)", FormatType.YUV_PACKED, "VYUY", subsampling=(2,1), v4l2_name="V4L2_PIX_FMT_VYUY")
        self._add("Y41P (4:1:1)", FormatType.YUV_PACKED, "Y41P", subsampling=(4,1), v4l2_name="V4L2_PIX_FMT_Y41P")
        self._add("AYUV (4:4:4)", FormatType.YUV_PACKED, "AYUV", subsampling=(1,1), v4l2_name="V4L2_PIX_FMT_AYUV32")
        self._add("VUYA (4:4:4)", FormatType.YUV_PACKED, "VUYA", subsampling=(1,1), v4l2_name="V4L2_PIX_FMT_VUYA32")
        self._add("Y210 (10-bit 4:2:2)", FormatType.YUV_PACKED, "Y210", bit_depth=10, subsampling=(2,1), v4l2_name="V4L2_PIX_FMT_Y210")

        # --- Planar YUV ---
        self._add("I420 (4:2:0)", FormatType.YUV_PLANAR, "YU12", subsampling=(2,2), v4l2_name="V4L2_PIX_FMT_YUV420")
        self._add("YV12 (4:2:0)", FormatType.YUV_PLANAR, "YV12", subsampling=(2,2), v4l2_name="V4L2_PIX_FMT_YVU420")
        self._add("YUV422P (4:2:2)", FormatType.YUV_PLANAR, "422P", subsampling=(2,1), v4l2_name="V4L2_PIX_FMT_YUV422P")
        self._add("YUV411P (4:1:1)", FormatType.YUV_PLANAR, "411P", subsampling=(4,1), v4l2_name="V4L2_PIX_FMT_YUV411P")
        self._add("YUV444P (4:4:4)", FormatType.YUV_PLANAR, "444P", subsampling=(1,1), v4l2_name="V4L2_PIX_FMT_YUV444")
        self._add("YUV410 (4:1:0)", FormatType.YUV_PLANAR, "YUV9", subsampling=(4,4), v4l2_name="V4L2_PIX_FMT_YUV410")
        self._add("YVU410 (4:1:0)", FormatType.YUV_PLANAR, "YVU9", subsampling=(4,4), v4l2_name="V4L2_PIX_FMT_YVU410")
        # Multi-planar formats
        self._add("YUV420M (4:2:0 Multi)", FormatType.YUV_PLANAR, "YM12", subsampling=(2,2), planes=3, v4l2_name="V4L2_PIX_FMT_YUV420M")
        self._add("YUV422M (4:2:2 Multi)", FormatType.YUV_PLANAR, "YM16", subsampling=(2,1), planes=3, v4l2_name="V4L2_PIX_FMT_YUV422M")

        # --- Semi-Planar YUV ---
        self._add("NV12 (4:2:0)", FormatType.YUV_SEMI_PLANAR, "NV12", subsampling=(2,2), v4l2_name="V4L2_PIX_FMT_NV12")
        self._add("NV21 (4:2:0)", FormatType.YUV_SEMI_PLANAR, "NV21", subsampling=(2,2), v4l2_name="V4L2_PIX_FMT_NV21")
        self._add("NV16 (4:2:2)", FormatType.YUV_SEMI_PLANAR, "NV16", subsampling=(2,1), v4l2_name="V4L2_PIX_FMT_NV16")
        self._add("NV61 (4:2:2)", FormatType.YUV_SEMI_PLANAR, "NV61", subsampling=(2,1), v4l2_name="V4L2_PIX_FMT_NV61")
        self._add("NV24 (4:4:4)", FormatType.YUV_SEMI_PLANAR, "NV24", subsampling=(1,1), v4l2_name="V4L2_PIX_FMT_NV24")
        self._add("NV42 (4:4:4)", FormatType.YUV_SEMI_PLANAR, "NV42", subsampling=(1,1), v4l2_name="V4L2_PIX_FMT_NV42")
        # 10-bit/16-bit semi-planar
        self._add("P010 (10-bit NV12)", FormatType.YUV_SEMI_PLANAR, "P010", bit_depth=10, subsampling=(2,2), v4l2_name="V4L2_PIX_FMT_P010")
        self._add("P016 (16-bit NV12)", FormatType.YUV_SEMI_PLANAR, "P016", bit_depth=16, subsampling=(2,2), v4l2_name="V4L2_PIX_FMT_P016")
        # Multi-planar semi-planar
        self._add("NV12M (4:2:0 Multi)", FormatType.YUV_SEMI_PLANAR, "NM12", subsampling=(2,2), planes=2, v4l2_name="V4L2_PIX_FMT_NV12M")
        self._add("NV21M (4:2:0 Multi)", FormatType.YUV_SEMI_PLANAR, "NM21", subsampling=(2,2), planes=2, v4l2_name="V4L2_PIX_FMT_NV21M")

        # --- RGB ---
        self._add("RGB332 (8-bit)", FormatType.RGB, "RGB1", bpp=8, v4l2_name="V4L2_PIX_FMT_RGB332")
        self._add("RGB444 (16-bit)", FormatType.RGB, "R444", bpp=16, v4l2_name="V4L2_PIX_FMT_RGB444")
        self._add("ARGB444 (16-bit)", FormatType.RGB, "AR12", bpp=16, v4l2_name="V4L2_PIX_FMT_ARGB444")
        self._add("XRGB444 (16-bit)", FormatType.RGB, "XR12", bpp=16, v4l2_name="V4L2_PIX_FMT_XRGB444")
        self._add("RGB555 (16-bit)", FormatType.RGB, "RGBO", bpp=16, v4l2_name="V4L2_PIX_FMT_RGB555")
        self._add("ARGB555 (16-bit)", FormatType.RGB, "AR15", bpp=16, v4l2_name="V4L2_PIX_FMT_ARGB555")
        self._add("XRGB555 (16-bit)", FormatType.RGB, "XR15", bpp=16, v4l2_name="V4L2_PIX_FMT_XRGB555")
        self._add("RGB565 (16-bit)", FormatType.RGB, "RGBP", bpp=16, v4l2_name="V4L2_PIX_FMT_RGB565")
        self._add("RGB555X (16-bit)", FormatType.RGB, "RGBQ", bpp=16, v4l2_name="V4L2_PIX_FMT_RGB555X")
        self._add("RGB565X (16-bit)", FormatType.RGB, "RGBR", bpp=16, v4l2_name="V4L2_PIX_FMT_RGB565X")
        self._add("BGR24 (24-bit)", FormatType.RGB, "BGR3", bpp=24, v4l2_name="V4L2_PIX_FMT_BGR24")
        self._add("RGB24 (24-bit)", FormatType.RGB, "RGB3", bpp=24, v4l2_name="V4L2_PIX_FMT_RGB24")
        self._add("BGR32 (32-bit)", FormatType.RGB, "BGR4", bpp=32, v4l2_name="V4L2_PIX_FMT_BGR32")
        self._add("RGB32 (32-bit)", FormatType.RGB, "RGB4", bpp=32, v4l2_name="V4L2_PIX_FMT_RGB32")
        self._add("ABGR32 (32-bit)", FormatType.RGB, "AR24", bpp=32, v4l2_name="V4L2_PIX_FMT_ABGR32")
        self._add("ARGB32 (32-bit)", FormatType.RGB, "BA24", bpp=32, v4l2_name="V4L2_PIX_FMT_ARGB32")
        self._add("BGRA32 (32-bit)", FormatType.RGB, "RA24", bpp=32, v4l2_name="V4L2_PIX_FMT_BGRA32")
        self._add("RGBA32 (32-bit)", FormatType.RGB, "AB24", bpp=32, v4l2_name="V4L2_PIX_FMT_RGBA32")
        self._add("XBGR32 (32-bit)", FormatType.RGB, "XR24", bpp=32, v4l2_name="V4L2_PIX_FMT_XBGR32")
        self._add("XRGB32 (32-bit)", FormatType.RGB, "XB24", bpp=32, v4l2_name="V4L2_PIX_FMT_XRGB32")
        self._add("BGRX32 (32-bit)", FormatType.RGB, "RX24", bpp=32, v4l2_name="V4L2_PIX_FMT_BGRX32")
        self._add("RGBX32 (32-bit)", FormatType.RGB, "BX24", bpp=32, v4l2_name="V4L2_PIX_FMT_RGBX32")
        # HSV formats
        self._add("HSV24 (24-bit)", FormatType.RGB, "HSV3", bpp=24, v4l2_name="V4L2_PIX_FMT_HSV24")
        self._add("HSV32 (32-bit)", FormatType.RGB, "HSV4", bpp=32, v4l2_name="V4L2_PIX_FMT_HSV32")

        # --- Bayer ---
        # 8-bit
        self._add("Bayer RGGB (8-bit)", FormatType.BAYER, "RGGB", bit_depth=8, v4l2_name="V4L2_PIX_FMT_SRGGB8")
        self._add("Bayer BGGR (8-bit)", FormatType.BAYER, "BGGR", bit_depth=8, v4l2_name="V4L2_PIX_FMT_SBGGR8")
        self._add("Bayer GBRG (8-bit)", FormatType.BAYER, "GBRG", bit_depth=8, v4l2_name="V4L2_PIX_FMT_SGBRG8")
        self._add("Bayer GRBG (8-bit)", FormatType.BAYER, "GRBG", bit_depth=8, v4l2_name="V4L2_PIX_FMT_SGRBG8")

        # 10-bit (expanded/unpacked to 16-bit usually)
        self._add("Bayer RGGB (10-bit)", FormatType.BAYER, "RG10", bit_depth=10, v4l2_name="V4L2_PIX_FMT_SRGGB10")
        self._add("Bayer BGGR (10-bit)", FormatType.BAYER, "BG10", bit_depth=10, v4l2_name="V4L2_PIX_FMT_SBGGR10")
        self._add("Bayer GBRG (10-bit)", FormatType.BAYER, "GB10", bit_depth=10, v4l2_name="V4L2_PIX_FMT_SGBRG10")
        self._add("Bayer GRBG (10-bit)", FormatType.BAYER, "BA10", bit_depth=10, v4l2_name="V4L2_PIX_FMT_SGRBG10")

        # 10-bit packed (MIPI CSI-2)
        self._add("Bayer RGGB (10-bit packed)", FormatType.BAYER, "pRAA", bit_depth=10, v4l2_name="V4L2_PIX_FMT_SRGGB10P")
        self._add("Bayer BGGR (10-bit packed)", FormatType.BAYER, "pBAA", bit_depth=10, v4l2_name="V4L2_PIX_FMT_SBGGR10P")
        self._add("Bayer GBRG (10-bit packed)", FormatType.BAYER, "pGAA", bit_depth=10, v4l2_name="V4L2_PIX_FMT_SGBRG10P")
        self._add("Bayer GRBG (10-bit packed)", FormatType.BAYER, "pGBA", bit_depth=10, v4l2_name="V4L2_PIX_FMT_SGRBG10P")

        # 12-bit
        self._add("Bayer RGGB (12-bit)", FormatType.BAYER, "RG12", bit_depth=12, v4l2_name="V4L2_PIX_FMT_SRGGB12")
        self._add("Bayer BGGR (12-bit)", FormatType.BAYER, "BG12", bit_depth=12, v4l2_name="V4L2_PIX_FMT_SBGGR12")
        self._add("Bayer GBRG (12-bit)", FormatType.BAYER, "GB12", bit_depth=12, v4l2_name="V4L2_PIX_FMT_SGBRG12")
        self._add("Bayer GRBG (12-bit)", FormatType.BAYER, "BA12", bit_depth=12, v4l2_name="V4L2_PIX_FMT_SGRBG12")

        # 16-bit
        self._add("Bayer RGGB (16-bit)", FormatType.BAYER, "RG16", bit_depth=16, v4l2_name="V4L2_PIX_FMT_SRGGB16")
        self._add("Bayer BGGR (16-bit)", FormatType.BAYER, "BG16", bit_depth=16, v4l2_name="V4L2_PIX_FMT_SBGGR16")
        self._add("Bayer GBRG (16-bit)", FormatType.BAYER, "GB16", bit_depth=16, v4l2_name="V4L2_PIX_FMT_SGBRG16")
        self._add("Bayer GRBG (16-bit)", FormatType.BAYER, "GR16", bit_depth=16, v4l2_name="V4L2_PIX_FMT_SGRBG16")

        # --- Grey ---
        self._add("Greyscale (8-bit)", FormatType.GREY, "GREY", bit_depth=8, v4l2_name="V4L2_PIX_FMT_GREY")
        self._add("Greyscale (10-bit)", FormatType.GREY, "Y10 ", bit_depth=10, v4l2_name="V4L2_PIX_FMT_Y10")
        self._add("Greyscale (12-bit)", FormatType.GREY, "Y12 ", bit_depth=12, v4l2_name="V4L2_PIX_FMT_Y12")
        self._add("Greyscale (16-bit)", FormatType.GREY, "Y16 ", bit_depth=16, v4l2_name="V4L2_PIX_FMT_Y16")


    def _add(self, name, type, fourcc, bit_depth=8, subsampling=(1,1), bpp=0, planes=1, v4l2_name=""):
        # Name is now the User Facing Name
        # We store key as FourCC for lookup if possible?
        # But we also want formatting.
        # Let's map "Name (FourCC)" -> Format

        # User requested: "Most formats ... display fourCC next to it."
        # We will use the formatted string as the key for the dropdown.
        # But for internal logic we might want to lookup by straight FourCC or straight name.

        display_name = f"{name} [{fourcc}]"
        self.formats[display_name] = VideoFormat(name, type, fourcc, bit_depth, subsampling, bpp, planes, v4l2_name)

    def get_format(self, key):
        # 1. Exact match
        if key in self.formats:
            return self.formats[key]

        # 2. Fuzzy match by FourCC (content inside [])
        # Iterate keys: "Name [FOURCC]"
        for k, v in self.formats.items():
            if f"[{key}]" in k: # e.g. input "YU12" matches "... [YU12]"
                return v

        # 3. Fuzzy match by Name (start)
        for k, v in self.formats.items():
             # Check if key starts with input "I420" -> "I420 (4:2:0) [YU12]"
             if k.startswith(key + " ") or k == key:
                 return v

        return None

    def get_supported_formats(self):
        return list(self.formats.keys())

    def get_formats_by_category(self):
        """Returns dict of category_name -> list of format display names"""
        categories = {
            "YUV Planar": [],
            "YUV Semi-Planar": [],
            "YUV Packed": [],
            "RGB": [],
            "Bayer": [],
            "Grey": [],
        }
        for key, fmt in self.formats.items():
            if fmt.type == FormatType.YUV_PLANAR:
                categories["YUV Planar"].append(key)
            elif fmt.type == FormatType.YUV_SEMI_PLANAR:
                categories["YUV Semi-Planar"].append(key)
            elif fmt.type == FormatType.YUV_PACKED:
                categories["YUV Packed"].append(key)
            elif fmt.type == FormatType.RGB:
                categories["RGB"].append(key)
            elif fmt.type == FormatType.BAYER:
                categories["Bayer"].append(key)
            elif fmt.type == FormatType.GREY:
                categories["Grey"].append(key)
        return categories
