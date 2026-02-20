import pytest
import numpy as np
import os
import sys
from video_viewer.video_reader import VideoReader

def create_dummy_yuv_i420(width, height, frames, tmp_path):
    # Create a red frame in I420
    # Red in YUV (approx): Y=76, U=84, V=255
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    frame_size = y_size + 2 * uv_size
    data = bytearray()

    for i in range(frames):
        # Y plane
        data.extend([76] * y_size)
        # U plane
        data.extend([84] * uv_size)
        # V plane
        data.extend([255] * uv_size)

    filename = tmp_path / "test_i420.yuv"
    with open(filename, "wb") as f:
        f.write(data)
    return str(filename)

def test_reader(tmp_path):
    width = 64
    height = 64
    frames = 5
    file_path = create_dummy_yuv_i420(width, height, frames, tmp_path)

    reader = VideoReader(file_path, width, height, "I420 (4:2:0) [YU12]")

    assert reader.total_frames == frames, f"Expected {frames} frames, got {reader.total_frames}"

    # Test reading frame 0
    raw = reader.seek_frame(0)
    assert raw is not None, "Failed to read frame 0"
    assert len(raw) == width * height * 1.5, "Frame size mismatch"

    # Test Channels
    channels = reader.get_channels(raw)
    assert 'Y' in channels and 'U' in channels and 'V' in channels

    # Check values
    # Allow small tolerance if implementation changes, but integer check should be exact for synthesized data
    assert np.all(channels['Y'] == 76), "Y channel content mismatch"

    # U and V are resized to full resolution for display in get_channels
    # So we check mean or all values
    assert channels['U'].shape == (height, width)
    assert channels['V'].shape == (height, width)

    # Resize might introduce interpolation artifacts if not nearest neighbor,
    # but here input is constant color so output should be constant.
    assert np.mean(channels['U']) == 84
    assert np.mean(channels['V']) == 255
