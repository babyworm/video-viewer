import pytest
import numpy as np
import cv2
import os
import sys

from video_viewer.video_reader import VideoReader

# Helper function to create synthetic data
def create_synthetic_frame(width, height, fmt_name, color_rgb):
    """
    Creates a raw byte frame for the given format representing a solid color.
    color_rgb: tuple (R, G, B)
    """
    R, G, B = color_rgb

    # Use OpenCV to convert RGB to YUV to ensure matching coefficients with the reader
    # Create 1x1 pixel image
    rgb_pixel = np.array([[[R, G, B]]], dtype=np.uint8)

    # helper for specific formats
    def get_yuv_cv2(rgb_pix):
        # Use YCrCb because COLOR_YUV2BGR_... functions usually expect YCbCr (digital)
        # not the analog YUV.
        # RGB2YCrCb returns Y, Cr, Cb
        yuv = cv2.cvtColor(rgb_pix, cv2.COLOR_RGB2YCrCb)
        return yuv[0,0]

    yuv_pixel = get_yuv_cv2(rgb_pixel)
    Y, Cr, Cb = yuv_pixel[0], yuv_pixel[1], yuv_pixel[2]
    # Map to U, V
    U = Cb
    V = Cr

    if fmt_name == "I420":
        # Planar YUV 4:2:0
        y_plane = np.full((height, width), Y, dtype=np.uint8)
        u_plane = np.full((height//2, width//2), U, dtype=np.uint8)
        v_plane = np.full((height//2, width//2), V, dtype=np.uint8)
        return y_plane.tobytes() + u_plane.tobytes() + v_plane.tobytes()

    elif fmt_name == "NV12":
        # Semi-Planar YUV 4:2:0 (Y plane, then UV interleaved)
        y_plane = np.full((height, width), Y, dtype=np.uint8)
        # OpenCV NV12 expects UV order for the interleaved part
        uv_plane = np.zeros((height//2, width//2, 2), dtype=np.uint8)
        uv_plane[:, :, 0] = U
        uv_plane[:, :, 1] = V
        return y_plane.tobytes() + uv_plane.tobytes()

    elif fmt_name == "YUYV":
        # Packed YUV 4:2:2 (Y U Y V)
        # OpenCV COLOR_YUV2BGR_YUYV (YUY2) expects Y0 U0 Y1 V0
        row = np.zeros((width * 2,), dtype=np.uint8)
        row[0::4] = Y
        row[1::4] = U
        row[2::4] = Y
        row[3::4] = V
        frame = np.tile(row, (height, 1))
        return frame.tobytes()

    elif fmt_name == "UYVY":
        # Packed YUV 4:2:2 (U Y V Y)
        # OpenCV COLOR_YUV2BGR_UYVY expects U0 Y0 V0 Y1
        row = np.zeros((width * 2,), dtype=np.uint8)
        row[0::4] = U
        row[1::4] = Y
        row[2::4] = V
        row[3::4] = Y
        frame = np.tile(row, (height, 1))
        return frame.tobytes()

    elif fmt_name == "RGB888":
        # Packed RGB
        pixel = np.array([R, G, B], dtype=np.uint8)
        frame = np.tile(pixel, (height, width, 1))
        return frame.tobytes()

    elif fmt_name == "RGGB":
        # Bayer RGGB
        # R G R G
        # G B G B
        # Approximation: Fill R pos with R, G pos with G, B pos with B
        frame = np.zeros((height, width), dtype=np.uint8)
        frame[0::2, 0::2] = R # R
        frame[0::2, 1::2] = G # G
        frame[1::2, 0::2] = G # G
        frame[1::2, 1::2] = B # B
        return frame.tobytes()

    return b""

@pytest.mark.parametrize("fmt_key", [
    "I420 (4:2:0) [YU12]",
    "NV12 (4:2:0) [NV12]",
    "YUYV (4:2:2) [YUYV]",
    "UYVY (4:2:2) [UYVY]",
    "RGB24 (24-bit) [RGB3]",
    "Bayer RGGB (8-bit) [RGGB]"
])
def test_format_decoding(tmp_path, fmt_key):
    # Parse generic name from key for synthesis helper
    raw_name = fmt_key.split(" [")[1][:-1] # Extract 'YU12' etc.
    # Map FourCC back to internal helper names if needed or adapt helper
    # Helper uses: I420, NV12, YUYV, UYVY, RGB888, RGGB
    # We should normalize helper inputs

    helper_map = {
        "YU12": "I420",
        "NV12": "NV12",
        "YUYV": "YUYV",
        "UYVY": "UYVY",
        "RGB3": "RGB888",
        "RGGB": "RGGB"
    }

    fmt_internal = helper_map[raw_name]

    width, height = 128, 128

    # Test color: Greenish (low R, high G, low B)
    # RGB: (50, 200, 50)
    # Expect Y around ~130, U low, V low
    target_rgb = (50, 200, 50)

    raw_data = create_synthetic_frame(width, height, fmt_internal, target_rgb)

    file_path = tmp_path / f"test_{raw_name}.raw"
    with open(file_path, "wb") as f:
        f.write(raw_data)

    reader = VideoReader(str(file_path), width, height, fmt_key)
    raw_out = reader.seek_frame(0)
    rgb_out = reader.convert_to_rgb(raw_out)

    assert rgb_out is not None, f"Failed to convert {fmt_key} to RGB"
    assert rgb_out.shape == (height, width, 3)

    # Check center pixel
    center_pixel = rgb_out[height//2, width//2]

    # Allow some tolerance for color verify due to YUV<->RGB approximations
    # Bayer interpolation also introduces averaging
    tolerance = 25

    print(f"Format: {fmt_key}, Center Pixel: {center_pixel}, Target: {target_rgb}")

    assert abs(int(center_pixel[0]) - target_rgb[0]) < tolerance, f"R channel deviation too high for {fmt_key}"
    assert abs(int(center_pixel[1]) - target_rgb[1]) < tolerance, f"G channel deviation too high for {fmt_key}"
    assert abs(int(center_pixel[2]) - target_rgb[2]) < tolerance, f"B channel deviation too high for {fmt_key}"

def test_y4m_parsing(tmp_path):
    # Construct a valid Y4M header and frame
    header = b"YUV4MPEG2 W64 H64 F30:1 Ip A0:0 C420mpeg2\n"
    frame_header = b"FRAME\n"

    # pure white frame (Y=255, U=128, V=128)
    width, height = 64, 64
    y_plane = np.full((height, width), 255, dtype=np.uint8)
    u_plane = np.full((height//2, width//2), 128, dtype=np.uint8)
    v_plane = np.full((height//2, width//2), 128, dtype=np.uint8)
    payload = y_plane.tobytes() + u_plane.tobytes() + v_plane.tobytes()

    file_path = tmp_path / "test.y4m"
    with open(file_path, "wb") as f:
        f.write(header)
        f.write(frame_header)
        f.write(payload)

    reader = VideoReader(str(file_path), 0, 0, "") # Auto-detect

    assert reader.width == 64
    assert reader.height == 64
    assert reader.format.fourcc == "YU12"

    raw = reader.seek_frame(0)
    rgb = reader.convert_to_rgb(raw)

    # White should be close to 255, 255, 255
    center = rgb[32, 32]
    assert np.all(center > 240), f"Expected white, got {center}"

def test_reader_component_extraction(tmp_path):
    # Test YUV component extraction
    width, height = 64, 64
    fmt_key = "I420 (4:2:0) [YU12]"

    # Create distinct pattern: Y=100, U=50, V=200
    color_rgb = (0, 0, 0) # Placeholder, we construct manually

    y_plane = np.full((height, width), 100, dtype=np.uint8)
    u_plane = np.full((height//2, width//2), 50, dtype=np.uint8)
    v_plane = np.full((height//2, width//2), 200, dtype=np.uint8)
    raw_data = y_plane.tobytes() + u_plane.tobytes() + v_plane.tobytes()

    file_path = tmp_path / "test_components.yuv"
    with open(file_path, "wb") as f:
        f.write(raw_data)

    reader = VideoReader(str(file_path), width, height, fmt_key)
    raw = reader.seek_frame(0)
    channels = reader.get_channels(raw)

    assert 'Y' in channels
    assert 'U' in channels
    assert 'V' in channels

    # Verify values
    assert np.mean(channels['Y']) == 100
    assert np.mean(channels['U']) == 50
    assert np.mean(channels['V']) == 200
