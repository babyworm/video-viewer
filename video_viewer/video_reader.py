import logging
import threading
import numpy as np
import cv2
import os
import mmap
from collections import OrderedDict
from PySide6.QtCore import QThread, Signal
from .format_manager import FormatType, FormatManager
from .constants import DEFAULT_CACHE_MAX_FRAMES, DEFAULT_CACHE_MAX_MEMORY_MB, DEFAULT_COLOR_MATRIX

logger = logging.getLogger(__name__)


class FrameCache:
    def __init__(self, max_memory_mb=DEFAULT_CACHE_MAX_MEMORY_MB):
        self.max_memory_mb = max_memory_mb
        self.max_frames = None  # Will be computed dynamically
        self._cache = OrderedDict()
        self._memory_usage = 0
        self._lock = threading.Lock()

    def _compute_max_frames(self, frame_size):
        """Compute max frames based on frame size and memory budget."""
        frame_size_mb = frame_size / (1024 * 1024)
        if frame_size_mb > 0:
            self.max_frames = max(4, min(256, int(self.max_memory_mb / frame_size_mb)))
        else:
            self.max_frames = 256

    def get(self, frame_idx):
        with self._lock:
            if frame_idx in self._cache:
                self._cache.move_to_end(frame_idx)
                return self._cache[frame_idx]
            return None

    def put(self, frame_idx, data):
        with self._lock:
            if frame_idx in self._cache:
                self._cache.move_to_end(frame_idx)
                return

            data_size = len(data)

            # Compute max_frames dynamically on first frame
            if self.max_frames is None:
                self._compute_max_frames(data_size)

            max_memory_bytes = self.max_memory_mb * 1024 * 1024

            # Evict oldest entries if over limit
            while (len(self._cache) >= self.max_frames or
                   self._memory_usage + data_size > max_memory_bytes) and self._cache:
                _, old_data = self._cache.popitem(last=False)
                self._memory_usage -= len(old_data)

            self._cache[frame_idx] = data
            self._memory_usage += data_size

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._memory_usage = 0

    @property
    def size(self):
        return len(self._cache)

    @property
    def memory_usage(self):
        return self._memory_usage


class VideoReader:
    def __init__(self, file_path, width, height, format_name, color_matrix=DEFAULT_COLOR_MATRIX):
        self.file_path = file_path
        self.is_y4m = file_path.lower().endswith('.y4m')

        self.width = width
        self.height = height
        self.color_matrix = color_matrix

        if not self.is_y4m and (self.width <= 0 or self.height <= 0):
            raise ValueError("Width and height must be positive")

        self.format_manager = FormatManager()
        self.y4m_header_len = 0
        self.y4m_frame_header_len = 6 # "FRAME\n"
        self.warnings = []  # collect warnings for UI display
        self.y4m_fps = None
        self.y4m_interlace = "progressive"
        self.y4m_par = (1, 1)
        self.y4m_extensions = []

        if self.is_y4m:
            self.parse_y4m_header()
        else:
            self.format = self.format_manager.get_format(format_name)
            self.frame_size = self.format.calculate_frame_size(width, height)

        self.file_size = os.path.getsize(file_path)
        logger.debug("Opening file: %s (size=%d bytes)", file_path, self.file_size)

        # Initialize mmap and cache
        self._file = None
        self._mmap = None
        self._cache = FrameCache(max_memory_mb=DEFAULT_CACHE_MAX_MEMORY_MB)
        self._frame_offsets = []  # For Y4M variable frame headers

        self._lock = threading.Lock()

        # Try to open mmap, fallback to regular file I/O if it fails
        self._use_mmap = False
        if self.file_size > 0:
            try:
                self._file = open(file_path, 'rb')
                self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
                self._use_mmap = True
                logger.debug("Using mmap for file: %s", file_path)
            except (OSError, ValueError):
                # mmap failed, clean up and fall back to regular I/O
                if self._mmap:
                    self._mmap.close()
                    self._mmap = None
                if self._file:
                    self._file.close()
                    self._file = None
                self._use_mmap = False
                logger.warning("mmap failed for %s, falling back to regular file I/O", file_path)

        if self.is_y4m:
            if self._use_mmap:
                # Build offset table for Y4M to handle variable frame headers
                self._build_y4m_offset_table()
            else:
                # Estimate total frames for Y4M (approximate if frame headers are constant)
                # Real Y4M might have variable frame headers or tags, but usually just "FRAME\n" (6 bytes)
                # payload = frame_size + 6
                self.payload_size = self.frame_size + self.y4m_frame_header_len
                self.total_frames = (self.file_size - self.y4m_header_len) // self.payload_size
        else:
            self.total_frames = self.file_size // self.frame_size if self.frame_size > 0 else 0

    def _build_y4m_offset_table(self):
        """Build offset table for Y4M frames (handles variable frame headers)"""
        self._frame_offsets = []
        offset = self.y4m_header_len

        while offset < self.file_size:
            # Read frame header
            header_end = self._mmap.find(b'\n', offset)
            if header_end == -1:
                break

            frame_header = self._mmap[offset:header_end]
            if not frame_header.startswith(b'FRAME'):
                break

            data_offset = header_end + 1
            self._frame_offsets.append(data_offset)
            offset = data_offset + self.frame_size

        self.total_frames = len(self._frame_offsets)

    def parse_y4m_header(self):
        with open(self.file_path, "rb") as f:
            # Header is up to newline
            # YUV4MPEG2 W720 H576 F25:1 Ip A1:1 C420mpeg2 XYSCSS=420JPEG
            header = b""
            while True:
                byte = f.read(1)
                if byte == b'\n':
                    break
                header += byte

            self.y4m_header_len = len(header) + 1
            header_str = header.decode('utf-8')

            parts = header_str.split(' ')
            if parts[0] != "YUV4MPEG2":
                raise ValueError("Invalid Y4M file")

            for part in parts:
                if part.startswith('W'):
                    self.width = int(part[1:])
                elif part.startswith('H'):
                    self.height = int(part[1:])
                elif part.startswith('C'):
                    colorspace = part[1:]
                    # Map y4m colorspace to our formats
                    if colorspace.startswith("420"):
                        # Y4M 4:2:0 is always planar I420
                        self.format = self.format_manager.get_format("I420 (4:2:0) [YU12]")
                    elif colorspace.startswith("422"):
                        # Y4M 4:2:2 is always planar, not packed
                        self.format = self.format_manager.get_format("YUV422P (4:2:2) [422P]")
                    elif colorspace.startswith("444"):
                        self.format = self.format_manager.get_format("YUV444P (4:4:4) [444P]")
                    elif colorspace.startswith("mono"):
                        self.format = self.format_manager.get_format("Greyscale (8-bit) [GREY]")
                    else:
                        msg = f"Unknown Y4M colorspace '{colorspace}', falling back to I420"
                        logger.warning(msg)
                        self.warnings.append(msg)
                elif part.startswith('F'):
                    # Frame rate: F25:1 or F30000:1001
                    try:
                        num_str, den_str = part[1:].split(':')
                        self.y4m_fps = int(num_str) / int(den_str)
                    except (ValueError, ZeroDivisionError):
                        pass
                elif part.startswith('I'):
                    # Interlace: Ip=progressive, It=tff, Ib=bff, Im=mixed
                    interlace_char = part[1:2]
                    interlace_map = {'p': 'progressive', 't': 'tff', 'b': 'bff', 'm': 'mixed'}
                    self.y4m_interlace = interlace_map.get(interlace_char, 'progressive')
                elif part.startswith('A'):
                    # Pixel aspect ratio: A1:1 or A10:11
                    try:
                        num_str, den_str = part[1:].split(':')
                        self.y4m_par = (int(num_str), int(den_str))
                    except ValueError:
                        pass
                elif part.startswith('X'):
                    # Extension field: store as-is (informational)
                    self.y4m_extensions.append(part[1:])

            if not getattr(self, 'format', None):
                # Default to I420 if colorspace unknown or unspecified
                msg = "Y4M colorspace not recognized, defaulting to I420"
                logger.warning(msg)
                self.warnings.append(msg)
                self.format = self.format_manager.get_format("I420 (4:2:0) [YU12]")

            self.frame_size = self.format.calculate_frame_size(self.width, self.height)
            logger.debug("Y4M header parsed: %dx%d format=%s fps=%s interlace=%s",
                         self.width, self.height, self.format.fourcc, self.y4m_fps, self.y4m_interlace)

    def seek_frame(self, frame_idx):
        with self._lock:
            if frame_idx < 0 or frame_idx >= self.total_frames:
                raise ValueError("Frame index out of bounds")

            # Check cache first
            cached = self._cache.get(frame_idx)
            if cached is not None:
                logger.debug("Cache hit for frame %d", frame_idx)
                return cached
            logger.debug("Cache miss for frame %d", frame_idx)

            # Read from mmap or file
            if self._use_mmap:
                if self.is_y4m:
                    if frame_idx < len(self._frame_offsets):
                        offset = self._frame_offsets[frame_idx]
                    else:
                        raise ValueError("Frame index out of bounds")
                else:
                    offset = frame_idx * self.frame_size

                raw_data = bytes(self._mmap[offset:offset + self.frame_size])
            else:
                # Fallback to regular file I/O
                if self.is_y4m:
                    offset = self.y4m_header_len + frame_idx * self.payload_size + self.y4m_frame_header_len
                else:
                    offset = frame_idx * self.frame_size

                with open(self.file_path, "rb") as f:
                    f.seek(offset)
                    raw_data = f.read(self.frame_size)

            # Cache the result
            self._cache.put(frame_idx, raw_data)
            return raw_data

    def _ycrcb_to_rgb(self, y, u, v):
        """Convert Y, U(Cb), V(Cr) planes to RGB using selected color matrix."""
        if self.color_matrix == "BT.709":
            # BT.709 manual conversion
            y_f = y.astype(np.float32)
            u_f = u.astype(np.float32) - 128.0
            v_f = v.astype(np.float32) - 128.0
            r = y_f + 1.5748 * v_f
            g = y_f - 0.1873 * u_f - 0.4681 * v_f
            b = y_f + 1.8556 * u_f
            rgb = np.stack([r, g, b], axis=-1)
            return np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            # BT.601 via OpenCV (default)
            yuv = cv2.merge([y, v, u])  # OpenCV YCrCb = Y, Cr(V), Cb(U)
            return cv2.cvtColor(yuv, cv2.COLOR_YCrCb2RGB)

    def convert_to_rgb(self, raw_data):
        if not raw_data:
            return None

        # Determine conversion based on format
        if self.format.type == FormatType.YUV_PLANAR:
            y_size = self.width * self.height
            y = np.frombuffer(raw_data, dtype=np.uint8, count=y_size, offset=0).reshape((self.height, self.width))

            if self.format.fourcc in ["YU12", "YV12"]:
                 uv_size = y_size // 4
                 u_offset = y_size
                 v_offset = y_size + uv_size
                 if self.format.fourcc == "YV12":
                     u_offset, v_offset = v_offset, u_offset # Swap

                 u = np.frombuffer(raw_data, dtype=np.uint8, count=uv_size, offset=u_offset).reshape((self.height//2, self.width//2))
                 v = np.frombuffer(raw_data, dtype=np.uint8, count=uv_size, offset=v_offset).reshape((self.height//2, self.width//2))

                 u = cv2.resize(u, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                 v = cv2.resize(v, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

            elif self.format.fourcc == "422P":
                 uv_size = y_size // 2
                 u = np.frombuffer(raw_data, dtype=np.uint8, count=uv_size, offset=y_size).reshape((self.height, self.width//2))
                 v = np.frombuffer(raw_data, dtype=np.uint8, count=uv_size, offset=y_size + uv_size).reshape((self.height, self.width//2))

                 u = cv2.resize(u, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                 v = cv2.resize(v, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

            elif self.format.fourcc == "444P":
                 # Y U V full size
                 u = np.frombuffer(raw_data, dtype=np.uint8, count=y_size, offset=y_size).reshape((self.height, self.width))
                 v = np.frombuffer(raw_data, dtype=np.uint8, count=y_size, offset=y_size*2).reshape((self.height, self.width))

            # Convert to RGB (using YCrCb logic for digital YUV)
            return self._ycrcb_to_rgb(y, u, v)

        elif self.format.type == FormatType.YUV_SEMI_PLANAR:
            y_size = self.width * self.height
            yuv_raw = np.frombuffer(raw_data, dtype=np.uint8)

            if self.format.fourcc in ["NV12", "NV21"]:
                # 4:2:0
                yuv_reshaped = yuv_raw.reshape((self.height * 3 // 2, self.width))
                if self.format.fourcc == "NV12":
                     bgr = cv2.cvtColor(yuv_reshaped, cv2.COLOR_YUV2BGR_NV12)
                else:
                     bgr = cv2.cvtColor(yuv_reshaped, cv2.COLOR_YUV2BGR_NV21)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            elif self.format.fourcc in ["NV16", "NV61"]:
                 # 4:2:2
                 # OpenCV might not support NV16 directly in all versions.
                 # Let's try UYVY approach? No, NV16 is Y plane then UV interleaved.
                 # Manual fallback for safety
                 y = yuv_raw[:y_size].reshape((self.height, self.width))
                 uv = yuv_raw[y_size:].reshape((self.height, self.width // 2, 2)) # UVUV...

                 if self.format.fourcc == "NV16":
                     u = uv[:,:,0]
                     v = uv[:,:,1]
                 else:
                     v = uv[:,:,0]
                     u = uv[:,:,1]

                 u = cv2.resize(u, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                 v = cv2.resize(v, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

                 return self._ycrcb_to_rgb(y, u, v)

        elif self.format.type == FormatType.YUV_PACKED:
            yuv_raw = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height, self.width, 2))
            code = None
            if self.format.fourcc == "YUYV": code = cv2.COLOR_YUV2BGR_YUYV
            elif self.format.fourcc == "UYVY": code = cv2.COLOR_YUV2BGR_UYVY
            elif self.format.fourcc == "YVYU": code = cv2.COLOR_YUV2BGR_YVYU
            elif self.format.fourcc == "VYUY": code = cv2.COLOR_YUV2BGR_VYUY

            if code is not None:
                bgr = cv2.cvtColor(yuv_raw, code)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        elif self.format.type == FormatType.BAYER:
            if self.format.bit_depth == 8:
                raw = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height, self.width))
                code = None
                if self.format.fourcc == "RGGB": code = cv2.COLOR_BayerRG2BGR
                elif self.format.fourcc == "BGGR": code = cv2.COLOR_BayerBG2BGR
                elif self.format.fourcc == "GBRG": code = cv2.COLOR_BayerGB2BGR
                elif self.format.fourcc == "GRBG": code = cv2.COLOR_BayerGR2BGR

                if code is not None:
                    bgr = cv2.cvtColor(raw, code)
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                 # 10/12/16 bit - Assuming 16-bit container
                 try:
                     raw = np.frombuffer(raw_data, dtype=np.uint16).reshape((self.height, self.width))
                 except ValueError:
                    # Fallback for when size mismatch e.g. packed
                    return None

                 # Scale to 8-bit for display
                 # Simplest: shift down
                 shift = self.format.bit_depth - 8
                 raw8 = (raw >> shift).astype(np.uint8)

                 # Demosaic
                 # We need to map 10/12 bit names to standard patterns
                 pattern = self.format.fourcc[:2] # RG, BG...
                 code = None
                 if pattern == "RG": code = cv2.COLOR_BayerRG2BGR
                 elif pattern == "BG": code = cv2.COLOR_BayerBG2BGR
                 elif pattern == "GB": code = cv2.COLOR_BayerGB2BGR
                 elif pattern == "GR": code = cv2.COLOR_BayerGR2BGR
                 elif self.format.fourcc.startswith("BA"): code = cv2.COLOR_BayerGR2BGR # V4L2 often uses BA?? Wait, BA10 is GRBG generally

                 if code is not None:
                    bgr = cv2.cvtColor(raw8, code)
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        elif self.format.type == FormatType.RGB:
            if self.format.fourcc == "RGB3": # RGB24
                return np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height, self.width, 3))
            elif self.format.fourcc == "BGR3": # BGR24
                bgr = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height, self.width, 3))
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif self.format.fourcc == "BA24": # ARGB
                bgra = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height, self.width, 4))
                return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
            elif self.format.fourcc in ["RGBP", "RGBO"]: # RGB565, 555
                # OpenCV 'cvtColor' for BGR565/555 expects uint8 with 2 channels, not uint16
                raw_packed = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height, self.width, 2))
                if self.format.fourcc == "RGBP":
                    bgr = cv2.cvtColor(raw_packed, cv2.COLOR_BGR5652BGR)
                else:
                    bgr = cv2.cvtColor(raw_packed, cv2.COLOR_BGR5552BGR)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif self.format.fourcc == "RGB1": # RGB332
                 # 8-bit RGB
                 raw = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height, self.width))
                 # R:3, G:3, B:2
                 # Manual conversion
                 img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                 img[:,:,0] = (raw & 0xE0) # R 11100000
                 img[:,:,1] = (raw & 0x1C) << 3 # G 00011100
                 img[:,:,2] = (raw & 0x03) << 6 # B 00000011
                 return img # It's RGB

        elif self.format.type == FormatType.GREY:
             if self.format.bit_depth == 8:
                 raw = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height, self.width))
                 return cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
             else:
                 try:
                     raw = np.frombuffer(raw_data, dtype=np.uint16).reshape((self.height, self.width))
                     shift = self.format.bit_depth - 8
                     raw8 = (raw >> shift).astype(np.uint8)
                     return cv2.cvtColor(raw8, cv2.COLOR_GRAY2RGB)
                 except: return None

        return None

    def get_pixel_info(self, frame_data, x, y, sub_grid_size=0):
        """Returns info about the pixel at x, y"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return None

        info = {
            "x": x,
            "y": y,
            "raw_hex": [],
            "components": {},
            "neighborhood": []
        }

        # 1. Extract Raw Value at X, Y and Neighborhood
        # This requires knowing pixel stride and layout.

        def get_raw_bytes(px, py):
             if px < 0 or px >= self.width or py < 0 or py >= self.height:
                 return None

             if self.format.type == FormatType.YUV_PLANAR:
                 # I420: Y is at y*w + x
                 y_idx = py * self.width + px
                 y_val = frame_data[y_idx]
                 return [y_val] # Just Y for "raw" view per pixel in planar? Or should we show UV?
                 # For planar, "raw" at a pixel coordinate is usually just the luminance plane value physically there?
                 # Or do we want the virtual pixel value found by combining planes?
                 # Request says "Original value and actual component".

             elif self.format.type == FormatType.YUV_PACKED:
                 # YUYV: 2 bytes per pixel
                 offset = (py * self.width + px) * 2
                 return [frame_data[offset], frame_data[offset+1]]

             elif self.format.type == FormatType.RGB:
                 bpp_bytes = self.format.bpp // 8
                 offset = (py * self.width + px) * bpp_bytes
                 return [frame_data[offset + i] for i in range(bpp_bytes)]

             elif self.format.type == FormatType.BAYER or self.format.type == FormatType.GREY:
                 if self.format.bit_depth <= 8:
                     offset = py * self.width + px
                     return [frame_data[offset]]
                 else:
                     offset = (py * self.width + px) * 2
                     return [frame_data[offset], frame_data[offset+1]]
             return []

        # Populate Neighborhood
        if sub_grid_size > 0:
            # Sub-grid cell origin
            cell_x = (x // sub_grid_size) * sub_grid_size
            cell_y = (y // sub_grid_size) * sub_grid_size
            # Range with 1-pixel padding
            x_start = cell_x - 1
            y_start = cell_y - 1
            x_end = cell_x + sub_grid_size  # inclusive via range
            y_end = cell_y + sub_grid_size
            nb = []
            for py in range(y_start, y_end + 1):
                row = []
                for px in range(x_start, x_end + 1):
                    raw = get_raw_bytes(px, py)
                    if raw:
                        hex_s = " ".join([f"{b:02X}" for b in raw])
                        row.append(hex_s)
                    else:
                        row.append("--")
                nb.append(row)
        else:
            # Default 3x3
            nb = []
            for dy in [-1, 0, 1]:
                row = []
                for dx in [-1, 0, 1]:
                    raw = get_raw_bytes(x + dx, y + dy)
                    if raw:
                        hex_s = " ".join([f"{b:02X}" for b in raw])
                        row.append(hex_s)
                    else:
                        row.append("--")
                nb.append(row)
        info["neighborhood"] = nb

        # 2. Extract Components
        # We need to compute the actual Y, U, V or R, G, B values
        if self.format.type == FormatType.YUV_PLANAR:
            y_size = self.width * self.height
            y_val = frame_data[y * self.width + x]
            info['components']['Y'] = y_val

            # Subsampling
            sx, sy = self.format.subsampling
            c_w = self.width // sx
            c_x = x // sx
            c_y = y // sy
            c_idx = c_y * c_w + c_x

            if self.format.fourcc == "Y444":
                info['components']['U'] = frame_data[y_size + y*self.width + x]
                info['components']['V'] = frame_data[y_size*2 + y*self.width + x]
            else:
               # Assumptions for I420/YV12
               uv_size = (self.width // sx) * (self.height // sy)
               if self.format.fourcc in ["YU12", "422P"]:
                   info['components']['U'] = frame_data[y_size + c_idx]
                   info['components']['V'] = frame_data[y_size + uv_size + c_idx]
               elif self.format.fourcc == "YV12":
                   info['components']['V'] = frame_data[y_size + c_idx]
                   info['components']['U'] = frame_data[y_size + uv_size + c_idx]

        elif self.format.type == FormatType.YUV_SEMI_PLANAR:
             y_val = frame_data[y * self.width + x]
             info['components']['Y'] = y_val

             sx, sy = self.format.subsampling
             c_w = self.width // sx
             c_x = x // sx
             c_y = y // sy
             # NV12/NV21 has interleaved UV: U V U V ...
             # Index in UV plane
             uv_idx = (c_y * c_w + c_x) * 2
             y_size = self.width * self.height

             if self.format.fourcc in ["NV12", "NV16"]: # U then V
                 info['components']['U'] = frame_data[y_size + uv_idx]
                 info['components']['V'] = frame_data[y_size + uv_idx + 1]
             else: # V then U
                 info['components']['V'] = frame_data[y_size + uv_idx]
                 info['components']['U'] = frame_data[y_size + uv_idx + 1]

        elif self.format.type == FormatType.YUV_PACKED:
             # YUYV: Y0 U0 Y1 V0
             # Pixel pair index
             pair_idx = x // 2
             offset = (y * self.width + pair_idx * 2) * 2 # 4 bytes per pair
             # Y value depends on if even or odd col
             # Y0 at offset, Y1 at offset+2
             is_odd = (x % 2) == 1

             if self.format.fourcc == "YUYV":
                 info['components']['Y'] = frame_data[offset + (2 if is_odd else 0)]
                 info['components']['U'] = frame_data[offset + 1]
                 info['components']['V'] = frame_data[offset + 3]
             elif self.format.fourcc == "UYVY":
                 info['components']['Y'] = frame_data[offset + (3 if is_odd else 1)]
                 info['components']['U'] = frame_data[offset]
                 info['components']['V'] = frame_data[offset + 2]

        elif self.format.type == FormatType.RGB:
             bpp = self.format.bpp
             offset = (y * self.width + x) * (bpp // 8)

             if bpp == 8: # RGB332
                 val = frame_data[offset]
                 info['components']['R'] = (val & 0xE0)
                 info['components']['G'] = (val & 0x1C) << 3
                 info['components']['B'] = (val & 0x03) << 6
             elif bpp == 24:
                 if self.format.fourcc == "RGB3":
                     info['components']['R'] = frame_data[offset]
                     info['components']['G'] = frame_data[offset+1]
                     info['components']['B'] = frame_data[offset+2]
                 elif self.format.fourcc == "BGR3":
                     info['components']['B'] = frame_data[offset]
                     info['components']['G'] = frame_data[offset+1]
                     info['components']['R'] = frame_data[offset+2]

        raw_here = get_raw_bytes(x, y)
        if raw_here:
            info["raw_hex"] = [f"{b:02X}" for b in raw_here]

        return info

    def get_channels(self, raw_data):
        """Returns a dict of channel name -> numpy image (grayscale)"""
        if not raw_data:
            return {}

        channels = {}

        if self.format.type in [FormatType.YUV_PLANAR, FormatType.YUV_SEMI_PLANAR, FormatType.YUV_PACKED]:
            # For simplicity, let's process based on converting to YUV planar logic or extracting if already planar
            # But wait, we want to see the *raw* components if possible?
            # Or just visual representation?
            # Let's convert to full sized Y, U, V images for visualization

            # Use OpenCV to convert everything to YUV I420 first if possible, then split
            # Actually, standard conversion to YCrCb (YUV) in OpenCV for RGB is RGB2YCrCb

            # We already implemented convert_to_rgb. Let's use that as intermediate if needed,
            # OR parse directly for planar formats which is faster and more "true" to data.

            if self.format.type == FormatType.YUV_PLANAR:
                y_size = self.width * self.height
                uv_size = y_size // 4
                y = np.frombuffer(raw_data, dtype=np.uint8, count=y_size, offset=0).reshape((self.height, self.width))
                channels['Y'] = y

                if self.format.fourcc == "YU12":
                    u = np.frombuffer(raw_data, dtype=np.uint8, count=uv_size, offset=y_size).reshape((self.height//2, self.width//2))
                    v = np.frombuffer(raw_data, dtype=np.uint8, count=uv_size, offset=y_size + uv_size).reshape((self.height//2, self.width//2))
                    # Resising for display if needed, but keeping raw is better?
                    # Request asked for "actual component per pixel", implied full size for visualization usually?
                    # Let's resize
                    channels['U'] = cv2.resize(u, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                    channels['V'] = cv2.resize(v, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                elif self.format.fourcc == "YV12":
                    v = np.frombuffer(raw_data, dtype=np.uint8, count=uv_size, offset=y_size).reshape((self.height//2, self.width//2))
                    u = np.frombuffer(raw_data, dtype=np.uint8, count=uv_size, offset=y_size + uv_size).reshape((self.height//2, self.width//2))
                    channels['V'] = cv2.resize(v, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                    channels['U'] = cv2.resize(u, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

            else:
                # Fallback: Convert to RGB then to YUV (YCrCb in OpenCV)
                rgb = self.convert_to_rgb(raw_data)
                if rgb is not None:
                    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb) # OpenCV uses YCrCb for YUV usually
                    y, cr, cb = cv2.split(yuv)
                    channels['Y'] = y
                    channels['U'] = cb
                    channels['V'] = cr

        elif self.format.type == FormatType.RGB or self.format.type == FormatType.BAYER:
             rgb = self.convert_to_rgb(raw_data)
             if rgb is not None:
                 r, g, b = cv2.split(rgb)
                 channels['R'] = r
                 channels['G'] = g
                 channels['B'] = b

        return channels

    def convert_rgb_to_bytes(self, rgb_image, target_fourcc):
        """
        Converts an RGB image (numpy array) to raw bytes in the specified format.
        Supports common formats: I420, NV12, YUYV, RGB3, BGR3.
        """
        h, w, _ = rgb_image.shape

        # 1. YUV Formats
        if target_fourcc in ["YU12", "YV12", "NV12", "NV21", "YUYV", "UYVY"]:
             # OpenCV usually expects BGR for its color conversions
             bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

             if target_fourcc == "YU12": # I420
                 yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
                 return yuv.tobytes()
             elif target_fourcc == "YV12":
                 yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_YV12)
                 return yuv.tobytes()
             elif target_fourcc == "NV12":
                 # Manual BGR -> I420 -> NV12
                 yuv_i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
                 yuv_flat = yuv_i420.flatten()
                 y_size = w * h
                 y = yuv_flat[:y_size]
                 uv_planar = yuv_flat[y_size:]
                 u_planar = uv_planar[:y_size//4]
                 v_planar = uv_planar[y_size//4:]

                 # Interleave
                 uv = np.empty((y_size//2,), dtype=np.uint8)
                 uv[0::2] = u_planar
                 uv[1::2] = v_planar

                 return y.tobytes() + uv.tobytes()
             elif target_fourcc == "NV21":
                 yuv_i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420) # Y U V
                 yuv_flat = yuv_i420.flatten()
                 y_size = w * h
                 y = yuv_flat[:y_size]
                 uv_planar = yuv_flat[y_size:]
                 u_planar = uv_planar[:y_size//4]
                 v_planar = uv_planar[y_size//4:]

                 # Interleave V U
                 vu = np.empty((y_size//2,), dtype=np.uint8)
                 vu[0::2] = v_planar
                 vu[1::2] = u_planar

                 return y.tobytes() + vu.tobytes()
             elif target_fourcc == "YUYV":
                 yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_YUYV) # YUYV aka YUY2
                 return yuv.tobytes()
             elif target_fourcc == "UYVY":
                 yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_UYVY)
                 return yuv.tobytes()

        # 2. RGB Formats
        elif target_fourcc == "RGB3": # RGB24
            return rgb_image.tobytes()
        elif target_fourcc == "BGR3": # BGR24
            bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            return bgr.tobytes()
        elif target_fourcc == "BA24": # ARGB
            bgra = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGRA) # Wait ARGB? BA24 is usually BGRA or ARGB depending on endian.
            # V4L2 ARGB32 is usually B G R A in memory (Little Endian integer ARGB) -> B, G, R, A bytes.
            return bgra.tobytes()
        elif target_fourcc == "RGBP": # RGB565
            bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            rgb565 = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGR565)
            return rgb565.tobytes()

        # Default/Fallback
        return None

    def close(self):
        """Clean up mmap and file resources"""
        if getattr(self, '_cache', None):
            self._cache.clear()
        if getattr(self, '_mmap', None):
            self._mmap.close()
            self._mmap = None
        if getattr(self, '_file', None):
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def cache_info(self):
        """Return cache statistics"""
        return {
            'size': self._cache.size,
            'memory_mb': self._cache.memory_usage / (1024 * 1024),
            'max_frames': self._cache.max_frames,
        }


class FrameDecodeWorker(QThread):
    """Background thread for frame decoding during playback."""
    frame_ready = Signal(int, int, object)  # (generation, frame_index, rgb_ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._reader = None
        self._frame_idx = -1
        self._generation = 0
        self._running = False

    @property
    def is_busy(self):
        return self._running

    def set_reader(self, reader):
        """Set the video reader (call from main thread when video changes)."""
        self._reader = reader

    def request_frame(self, frame_idx, generation=0):
        """Request decoding of a frame. Non-blocking. Returns False if busy."""
        if self._running or self.isRunning():
            return False
        self._frame_idx = frame_idx
        self._generation = generation
        self._running = True
        self.start()
        return True

    def run(self):
        try:
            if self._reader and self._frame_idx >= 0:
                raw_data = self._reader.seek_frame(self._frame_idx)
                if raw_data:
                    rgb = self._reader.convert_to_rgb(raw_data)
                    if rgb is not None:
                        self.frame_ready.emit(self._generation, self._frame_idx, rgb)
        except Exception:
            pass
        finally:
            self._running = False

    def stop_worker(self):
        """Wait for the worker thread to finish."""
        self.wait(2000)
