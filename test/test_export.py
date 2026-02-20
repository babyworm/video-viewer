import numpy as np
import os
from video_viewer.video_reader import VideoReader

def test_export_logic(tmp_path):
    # generate a known RGB frame
    width, height = 4, 4
    file_path = tmp_path / "src.rgb"

    # Red frame
    src_data = np.zeros((height, width, 3), dtype=np.uint8)
    src_data[:,:,0] = 255 # R

    with open(file_path, "wb") as f:
        f.write(src_data.tobytes())

    reader = VideoReader(str(file_path), width, height, "RGB24 (24-bit) [RGB3]")

    # Test Read
    raw = reader.seek_frame(0)
    rgb = reader.convert_to_rgb(raw)

    # Test Convert to I420
    # Red in YUV (approx): Y=76, U=84, V=255
    yuv_bytes = reader.convert_rgb_to_bytes(rgb, "YU12")
    assert len(yuv_bytes) == width * height * 1.5

    y = np.frombuffer(yuv_bytes, dtype=np.uint8, count=16, offset=0)
    # Check mean Y
    # OpenCV's RGB2YUV might differ slightly but Red should have Y component around 76
    assert 70 < np.mean(y) < 85

    # Test Convert to NV12
    nv12_bytes = reader.convert_rgb_to_bytes(rgb, "NV12")
    assert len(nv12_bytes) == width * height * 1.5

    # Test Convert to YUYV
    yuyv_bytes = reader.convert_rgb_to_bytes(rgb, "YUYV")
    assert len(yuyv_bytes) == width * height * 2

def test_export_loop_integrity(tmp_path):
    # Create valid 2-frame RGB file
    width, height = 4, 4
    frames = [os.urandom(width*height*3) for _ in range(2)]

    fpath = tmp_path / "test_multi.rgb"
    with open(fpath, "wb") as f:
        for fr in frames:
            f.write(fr)

    reader = VideoReader(str(fpath), width, height, "RGB24 (24-bit) [RGB3]")

    # Simulate export loop for full range
    out_path = tmp_path / "out.i420"
    target_fourcc = "YU12"

    with open(out_path, "wb") as f_out:
        for i in range(2):
            raw = reader.seek_frame(i)
            rgb = reader.convert_to_rgb(raw)
            out_bytes = reader.convert_rgb_to_bytes(rgb, target_fourcc)
            f_out.write(out_bytes)

    assert out_path.exists()
    assert out_path.stat().st_size == (width*height*1.5) * 2
