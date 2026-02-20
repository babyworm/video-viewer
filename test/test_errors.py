import pytest
import os
from video_viewer.video_reader import VideoReader

def test_file_not_found():
    """Test that FileNotFoundError is raised for non-existent files."""
    with pytest.raises(FileNotFoundError):
        VideoReader("non_existent_file.yuv", 64, 64, "I420")

def test_invalid_dimensions(tmp_path):
    """Test behavior with invalid dimensions (0 or negative)."""
    # Create a dummy file
    fpath = tmp_path / "dummy.yuv"
    fpath.write_bytes(b'\x00' * 100)

    # Depending on implementation, this might raise ValueError or just result in 0 frames.
    # Let's assume 0 width/height is invalid for processing but might be accepted by init until seek.
    # Actually, VideoReader calculates frame size in init. Format frame size calc might fail or return 0.

    # If width=0, frame_size=0. total_frames = 100 / 0 -> ZeroDivisionError?
    with pytest.raises(ValueError):
         VideoReader(str(fpath), 0, 64, "I420")

def test_frame_out_of_bounds(tmp_path):
    """Test seeking to a frame index out of bounds."""
    width, height = 64, 64
    frame_size = int(width * height * 1.5) # I420

    # Create file with 2 frames
    fpath = tmp_path / "test.yuv"
    with open(fpath, "wb") as f:
        f.write(b'\x00' * frame_size * 2)

    reader = VideoReader(str(fpath), width, height, "I420 (4:2:0) [YU12]")

    assert reader.total_frames == 2

    # Seek to valid frames
    assert reader.seek_frame(0) is not None
    assert reader.seek_frame(1) is not None

    # Seek out of bounds
    with pytest.raises(ValueError):
        reader.seek_frame(2)

    with pytest.raises(ValueError):
        reader.seek_frame(-1)

def test_invalid_format_name(tmp_path):
    """Test that an invalid format name raises an error or is handled."""
    fpath = tmp_path / "dummy.yuv"
    fpath.write_bytes(b'\x00' * 100)

    # If format is unknown, VideoReader might default to something or fail.
    # Current implementation: FormatManager.get_format returns None.
    # VideoReader uses self.format.name later.

    with pytest.raises(AttributeError): # NoneType has no attribute 'name' or similar
        VideoReader(str(fpath), 64, 64, "INVALID_FMT")
