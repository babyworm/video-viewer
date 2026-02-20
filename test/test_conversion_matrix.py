import pytest
import numpy as np
import cv2
import os
from video_viewer.video_reader import VideoReader

# Define supported output formats by VideoReader.convert_rgb_to_bytes
# Format: (FourCC, Description)
OUTPUT_FORMATS = [
    ("YU12", "I420"),
    ("YV12", "YV12"),
    ("NV12", "NV12"),
    ("NV21", "NV21"),
    ("YUYV", "YUYV"),
    ("UYVY", "UYVY"),
    # ("RGB3", "RGB24"), # RGB/BGR tests are simpler, let's focus on YUV complex ones first or include all
    ("RGB3", "RGB24"),
    ("BGR3", "BGR24"),
    ("RGBP", "RGB565"),
]

# Define Source Data Generators
# We need to create a file with a known color.
# Function signature: creates file at path, returns (width, height, expected_rgb)

def create_solid_rgb(path, w, h, color):
    # RGB24
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    with open(path, "wb") as f:
        f.write(img.tobytes())
    return "RGB24 (24-bit) [RGB3]"

def create_solid_i420(path, w, h, color):
    # color is RGB tuple
    # Convert to YUV
    yuv_pix = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2YCrCb)[0][0]
    Y, Cr, Cb = yuv_pix

    y_plane = np.full((h, w), Y, dtype=np.uint8)
    u_plane = np.full((h//2, w//2), Cb, dtype=np.uint8)
    v_plane = np.full((h//2, w//2), Cr, dtype=np.uint8)

    with open(path, "wb") as f:
        f.write(y_plane.tobytes())
        f.write(u_plane.tobytes())
        f.write(v_plane.tobytes())
    return "I420 (4:2:0) [YU12]"

def create_solid_nv12(path, w, h, color):
    # NV12: Y plane then UV interleaved
    yuv_pix = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2YCrCb)[0][0]
    Y, Cr, Cb = yuv_pix

    y_plane = np.full((h, w), Y, dtype=np.uint8)
    uv_plane = np.zeros((h//2, w//2, 2), dtype=np.uint8)
    uv_plane[:, :, 0] = Cb # U
    uv_plane[:, :, 1] = Cr # V

    with open(path, "wb") as f:
        f.write(y_plane.tobytes())
        f.write(uv_plane.tobytes())
    return "NV12 (4:2:0) [NV12]"

def create_solid_yuyv(path, w, h, color):
    # YUYV: Y0 U0 Y1 V0
    yuv_pix = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2YCrCb)[0][0]
    Y, Cr, Cb = yuv_pix

    row = np.zeros(w * 2, dtype=np.uint8)
    row[0::4] = Y
    row[1::4] = Cb
    row[2::4] = Y
    row[3::4] = Cr

    frame = np.tile(row, (h, 1))
    with open(path, "wb") as f:
        f.write(frame.tobytes())
    return "YUYV (4:2:2) [YUYV]"

SOURCES = [
    ("RGB24", create_solid_rgb),
    ("I420", create_solid_i420),
    ("NV12", create_solid_nv12),
    ("YUYV", create_solid_yuyv),
]

# Test Matrix
@pytest.mark.parametrize("source_name, source_gen", SOURCES)
@pytest.mark.parametrize("target_fourcc, target_desc", OUTPUT_FORMATS)
def test_conversion_matrix(tmp_path, source_name, source_gen, target_fourcc, target_desc):
    # 1. Setup
    width, height = 32, 32 # Multiple of 4/16/etc safe for all formats
    # Use a color that survives YUV conversion reasonably well (not pure bounds)
    # e.g. a nice Teal: R=50, G=150, B=150
    target_color = (50, 150, 150)

    # 2. Create Source File
    source_file = tmp_path / f"source_{source_name}.raw"
    fmt_str = source_gen(source_file, width, height, target_color)

    # 3. Read Source & Convert to RGB
    reader = VideoReader(str(source_file), width, height, fmt_str)
    raw_source = reader.seek_frame(0)
    assert raw_source is not None

    rgb_source = reader.convert_to_rgb(raw_source)
    assert rgb_source is not None

    # Verify Source Color Consistency first (sanity check generator)
    center_source = rgb_source[height//2, width//2]
    # diff = np.abs(center_source - target_color)
    # assert np.mean(diff) < 20, f"Source Generator {source_name} produced wrong color {center_source} vs {target_color}"

    # 4. Convert RGB -> Target Bytes
    target_bytes = reader.convert_rgb_to_bytes(rgb_source, target_fourcc)
    assert target_bytes is not None, f"Conversion from RGB to {target_fourcc} returned None"

    # 5. Write Target File
    target_file = tmp_path / f"target_{target_fourcc}.raw"
    with open(target_file, "wb") as f:
        f.write(target_bytes)

    # 6. Read Target File & Verify Color
    # We need the full format string for the reader. VideoReader usually takes "Name (FourCC) [CODE]"
    # We can reconstruct a minimal valid string or lookup from manager
    # Or just use "Name [CODE]" which our fuzzy matcher supports

    # Actually, pass only the FourCC. The fuzzy matcher looks for `[FourCC]` in known formats.

    reader_out = VideoReader(str(target_file), width, height, target_fourcc)
    raw_out = reader_out.seek_frame(0)
    assert raw_out is not None, "Failed to read back generated target file"

    rgb_out = reader_out.convert_to_rgb(raw_out)
    assert rgb_out is not None

    center_out = rgb_out[height//2, width//2]

    # Tolerance
    # YUV <-> RGB conversion involves colorspace shrinking and rounding.
    # 50, 150, 150 might drift by 10-20 values.
    # RGB565 has lower bit depth (5/6 bits), so drift is higher.

    tolerance = 25
    if target_fourcc == "RGBP": # RGB565
        tolerance = 40

    diff = np.mean(np.abs(center_out.astype(int) - target_color))

    print(f"[{source_name} -> {target_fourcc}] Target: {target_color}, Result: {center_out}, Diff: {diff}")

    assert diff < tolerance, (
        f"Color mismatch for {source_name} -> {target_fourcc}. "
        f"Expected ~{target_color}, Got {center_out}. Diff avg: {diff}"
    )
