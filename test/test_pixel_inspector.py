import pytest
import numpy as np
import sys
import os

from video_viewer.video_reader import VideoReader
from video_viewer.format_manager import FormatManager

def test_pixel_info_i420(tmp_path):
    width, height = 4, 4
    # I420: 4x4 Y, 2x2 U, 2x2 V
    fq = FormatManager().get_format("I420") # This might fail if get_format expects full key. But we don't use fq variable.

    # Create known pattern
    # Y: 0, 1, 2, ...
    y_plane = np.arange(16, dtype=np.uint8).reshape((4,4))
    # U: 100, 101, 102, 103
    u_plane = np.arange(100, 104, dtype=np.uint8).reshape((2,2))
    # V: 200, 201, 202, 203
    v_plane = np.arange(200, 204, dtype=np.uint8).reshape((2,2))

    raw = y_plane.tobytes() + u_plane.tobytes() + v_plane.tobytes()

    # Mock reader setup (bypass init file check if possible, or just write file)
    fpath = tmp_path / "test.i420"
    with open(fpath, "wb") as f:
        f.write(raw)

    reader = VideoReader(str(fpath), width, height, "I420 (4:2:0) [YU12]")

    # Check (0,0) -> Y=0, U=100, V=200
    info = reader.get_pixel_info(raw, 0, 0)
    assert info['components']['Y'] == 0
    assert info['components']['U'] == 100
    assert info['components']['V'] == 200
    # Raw hex should be just Y for I420 based on current impl
    assert info['raw_hex'] == ['00']

    # Check (1,0) -> Y=1, U=100, V=200 (same UV due to subsampling)
    info = reader.get_pixel_info(raw, 1, 0)
    assert info['components']['Y'] == 1
    assert info['components']['U'] == 100
    assert info['components']['V'] == 200

    # Check (2,2) -> Y=10, U=103, V=203 (col 1, row 1 of UV)
    # y index: 2*4 + 2 = 10
    # uv index: (2//2)*2 + (2//2) = 1*2 + 1 = 3 -> value 103/203
    info = reader.get_pixel_info(raw, 2, 2)
    assert info['components']['Y'] == 10
    assert info['components']['U'] == 103
    assert info['components']['V'] == 203

def test_pixel_info_nv12(tmp_path):
    width, height = 2, 2
    # NV12: Y(2x2), UV(1x1 interleaved)
    # Y: 10, 11, 12, 13
    y_plane = np.array([[10, 11], [12, 13]], dtype=np.uint8)
    # UV: U=50, V=60 -> sequence 50, 60
    uv_plane = np.array([50, 60], dtype=np.uint8)

    raw = y_plane.tobytes() + uv_plane.tobytes()

    fpath = tmp_path / "test.nv12"
    with open(fpath, "wb") as f:
        f.write(raw)

    fpath = tmp_path / "test.nv12"
    with open(fpath, "wb") as f:
        f.write(raw)

    reader = VideoReader(str(fpath), width, height, "NV12 (4:2:0) [NV12]")

    info = reader.get_pixel_info(raw, 0, 0)
    assert info['components']['Y'] == 10
    assert info['components']['U'] == 50
    assert info['components']['V'] == 60

    info = reader.get_pixel_info(raw, 1, 1)
    assert info['components']['Y'] == 13
    assert info['components']['U'] == 50
    assert info['components']['V'] == 60

def test_pixel_info_yuyv(tmp_path):
    width, height = 2, 1
    # YUYV: Y0 U0 Y1 V0
    # [10, 50, 20, 60] -> px0(Y=10, U=50, V=60), px1(Y=20, U=50, V=60)
    raw = bytes([10, 50, 20, 60])

    fpath = tmp_path / "test.yuyv"
    with open(fpath, "wb") as f:
        f.write(raw)

    fpath = tmp_path / "test.yuyv"
    with open(fpath, "wb") as f:
        f.write(raw)

    reader = VideoReader(str(fpath), width, height, "YUYV (4:2:2) [YUYV]")

    # Px 0
    info = reader.get_pixel_info(raw, 0, 0)
    assert info['components']['Y'] == 10
    assert info['components']['U'] == 50
    assert info['components']['V'] == 60
    assert info['raw_hex'] == ['0A', '32'] # 10, 50

    # Px 1
    info = reader.get_pixel_info(raw, 1, 0)
    assert info['components']['Y'] == 20
    assert info['components']['U'] == 50
    assert info['components']['V'] == 60
    assert info['raw_hex'] == ['14', '3C'] # 20, 60

def test_pixel_info_rgb888(tmp_path):
    width, height = 1, 1
    # R=10, G=20, B=30
    raw = bytes([10, 20, 30])

    fpath = tmp_path / "test.rgb"
    with open(fpath, "wb") as f:
        f.write(raw)

    fpath = tmp_path / "test.rgb"
    with open(fpath, "wb") as f:
        f.write(raw)

    reader = VideoReader(str(fpath), width, height, "RGB24 (24-bit) [RGB3]")

    info = reader.get_pixel_info(raw, 0, 0)
    assert info['components']['R'] == 10
    assert info['components']['G'] == 20
    assert info['components']['B'] == 30
    assert info['raw_hex'] == ['0A', '14', '1E']
