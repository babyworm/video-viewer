import logging
import cv2
import numpy as np
import os
from .video_reader import VideoReader
from .format_manager import FormatManager, FormatType

logger = logging.getLogger(__name__)


class VideoConverter:
    def __init__(self):
        self.format_manager = FormatManager()

    def convert(self, input_path, width, height, input_fmt, output_path, output_fmt,
                frame_callback=None):
        """Convert a video file from input_fmt to output_fmt.

        Args:
            frame_callback: Optional callable(frame_index, total_frames) called after
                each frame is written. If it returns False, conversion is cancelled.
        """
        reader = VideoReader(input_path, width, height, input_fmt)
        out_fmt_obj = self.format_manager.get_format(output_fmt)

        if not out_fmt_obj:
            raise ValueError(f"Unsupported output format: {output_fmt}")

        can_direct = self._can_direct_convert(reader.format, out_fmt_obj)
        logger.debug("Conversion start: %s -> %s (%dx%d), direct=%s, frames=%d",
                     input_fmt, output_fmt, width, height, can_direct, reader.total_frames)

        converted = 0
        cancelled = False
        with open(output_path, "wb") as out_f:
            for i in range(reader.total_frames):
                raw = reader.seek_frame(i)

                if reader.format.fourcc == out_fmt_obj.fourcc:
                    # Same format: direct copy
                    out_f.write(raw)
                    converted += 1
                elif can_direct:
                    # Direct YUV plane conversion (no RGB intermediate)
                    result = self._direct_convert(
                        raw, reader.width, reader.height, reader.format, out_fmt_obj)
                    if result:
                        out_f.write(result)
                        converted += 1
                    else:
                        break
                else:
                    # Fallback: RGB intermediate (for YUV↔RGB, YUV↔Bayer, etc.)
                    rgb = reader.convert_to_rgb(raw)
                    if rgb is not None:
                        out_data = reader.convert_rgb_to_bytes(rgb, out_fmt_obj.fourcc)
                        if out_data:
                            out_f.write(out_data)
                            converted += 1
                        else:
                            break

                if frame_callback is not None:
                    if frame_callback(i, reader.total_frames) is False:
                        cancelled = True
                        break

        logger.info("Conversion complete: %d frames, %s -> %s", converted, input_fmt, output_fmt)
        return converted, cancelled

    @staticmethod
    def _can_direct_convert(src_fmt, dst_fmt):
        """Check if direct YUV plane conversion is possible (no RGB needed)."""
        yuv_types = {FormatType.YUV_PLANAR, FormatType.YUV_SEMI_PLANAR, FormatType.YUV_PACKED}
        if src_fmt.type not in yuv_types or dst_fmt.type not in yuv_types:
            return False
        if src_fmt.bit_depth != dst_fmt.bit_depth or src_fmt.bit_depth != 8:
            return False
        # Skip exotic packed formats
        exotic = {"AYUV", "VUYA", "Y41P", "Y210"}
        if src_fmt.fourcc in exotic or dst_fmt.fourcc in exotic:
            return False
        return True

    @staticmethod
    def _extract_yuv_planes(raw_data, width, height, fmt):
        """Extract Y, U, V planes from raw frame data at native chroma resolution.

        Returns:
            (Y, U, V) numpy arrays. Y is (height, width).
            U, V are (height/v_sub, width/h_sub) at native chroma resolution.
        """
        y_size = width * height
        h_sub, v_sub = fmt.subsampling
        uv_w = width // h_sub
        uv_h = height // v_sub
        uv_size = uv_w * uv_h

        if fmt.type == FormatType.YUV_PLANAR:
            y = np.frombuffer(raw_data, dtype=np.uint8, count=y_size).reshape(height, width)
            if fmt.fourcc in ("YV12", "YVU9"):
                # V before U
                v = np.frombuffer(raw_data, dtype=np.uint8, count=uv_size,
                                  offset=y_size).reshape(uv_h, uv_w)
                u = np.frombuffer(raw_data, dtype=np.uint8, count=uv_size,
                                  offset=y_size + uv_size).reshape(uv_h, uv_w)
            else:
                # U before V (I420/YU12, 422P, 444P, 411P, YUV9, YM12, YM16)
                u = np.frombuffer(raw_data, dtype=np.uint8, count=uv_size,
                                  offset=y_size).reshape(uv_h, uv_w)
                v = np.frombuffer(raw_data, dtype=np.uint8, count=uv_size,
                                  offset=y_size + uv_size).reshape(uv_h, uv_w)
            return y.copy(), u.copy(), v.copy()

        elif fmt.type == FormatType.YUV_SEMI_PLANAR:
            y = np.frombuffer(raw_data, dtype=np.uint8, count=y_size).reshape(height, width)
            uv_data = np.frombuffer(raw_data, dtype=np.uint8,
                                    offset=y_size, count=uv_size * 2)
            if fmt.fourcc in ("NV12", "NV16", "NV24", "NM12"):
                # U first in interleaved pair
                u = uv_data[0::2].reshape(uv_h, uv_w).copy()
                v = uv_data[1::2].reshape(uv_h, uv_w).copy()
            else:
                # V first (NV21, NV61, NV42, NM21)
                v = uv_data[0::2].reshape(uv_h, uv_w).copy()
                u = uv_data[1::2].reshape(uv_h, uv_w).copy()
            return y.copy(), u, v

        elif fmt.type == FormatType.YUV_PACKED:
            # 4:2:2 packed: 2 bytes per pixel
            packed = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width * 2)
            if fmt.fourcc == "YUYV":
                y = packed[:, 0::2].copy()
                u = packed[:, 1::4].copy()
                v = packed[:, 3::4].copy()
            elif fmt.fourcc == "YVYU":
                y = packed[:, 0::2].copy()
                v = packed[:, 1::4].copy()
                u = packed[:, 3::4].copy()
            elif fmt.fourcc == "UYVY":
                y = packed[:, 1::2].copy()
                u = packed[:, 0::4].copy()
                v = packed[:, 2::4].copy()
            elif fmt.fourcc == "VYUY":
                y = packed[:, 1::2].copy()
                v = packed[:, 0::4].copy()
                u = packed[:, 2::4].copy()
            else:
                return None, None, None
            return y, u, v

        return None, None, None

    @staticmethod
    def _resample_chroma(u, v, src_sub, dst_sub):
        """Resample chroma planes when subsampling ratios differ.

        Uses INTER_AREA for downsampling, INTER_LINEAR for upsampling.
        """
        if src_sub == dst_sub:
            return u, v

        src_h, src_w = u.shape
        full_h = src_h * src_sub[1]
        full_w = src_w * src_sub[0]
        dst_w = full_w // dst_sub[0]
        dst_h = full_h // dst_sub[1]

        if (src_h, src_w) == (dst_h, dst_w):
            return u, v

        interp = cv2.INTER_AREA if (dst_h < src_h or dst_w < src_w) else cv2.INTER_LINEAR
        u_out = cv2.resize(u, (dst_w, dst_h), interpolation=interp)
        v_out = cv2.resize(v, (dst_w, dst_h), interpolation=interp)
        return u_out, v_out

    @staticmethod
    def _pack_yuv(y, u, v, fmt, width, height):
        """Pack Y, U, V planes into target format raw bytes."""
        if fmt.type == FormatType.YUV_PLANAR:
            if fmt.fourcc in ("YV12", "YVU9"):
                return y.tobytes() + v.tobytes() + u.tobytes()
            else:
                return y.tobytes() + u.tobytes() + v.tobytes()

        elif fmt.type == FormatType.YUV_SEMI_PLANAR:
            uv = np.empty(u.size + v.size, dtype=np.uint8)
            if fmt.fourcc in ("NV12", "NV16", "NV24", "NM12"):
                uv[0::2] = u.flatten()
                uv[1::2] = v.flatten()
            else:
                uv[0::2] = v.flatten()
                uv[1::2] = u.flatten()
            return y.tobytes() + uv.tobytes()

        elif fmt.type == FormatType.YUV_PACKED:
            out = np.empty((height, width * 2), dtype=np.uint8)
            if fmt.fourcc == "YUYV":
                out[:, 0::2] = y
                out[:, 1::4] = u
                out[:, 3::4] = v
            elif fmt.fourcc == "YVYU":
                out[:, 0::2] = y
                out[:, 1::4] = v
                out[:, 3::4] = u
            elif fmt.fourcc == "UYVY":
                out[:, 1::2] = y
                out[:, 0::4] = u
                out[:, 2::4] = v
            elif fmt.fourcc == "VYUY":
                out[:, 1::2] = y
                out[:, 0::4] = v
                out[:, 2::4] = u
            else:
                return None
            return out.tobytes()

        return None

    def _direct_convert(self, raw_data, width, height, src_fmt, dst_fmt):
        """Convert raw frame directly between YUV formats via plane manipulation."""
        logger.debug("Direct convert: %s -> %s (%dx%d)", src_fmt.fourcc, dst_fmt.fourcc, width, height)
        y, u, v = self._extract_yuv_planes(raw_data, width, height, src_fmt)
        if y is None:
            return None

        # Resample chroma if subsampling differs (e.g., 4:2:0 → 4:2:2)
        u, v = self._resample_chroma(u, v, src_fmt.subsampling, dst_fmt.subsampling)

        result = self._pack_yuv(y, u, v, dst_fmt, width, height)
        logger.debug("Direct convert done: %s -> %s", src_fmt.fourcc, dst_fmt.fourcc)
        return result
