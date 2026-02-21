"""
Unit tests for parse_filename_hints() — pure function, no Qt dependency.
"""
import pytest
from video_viewer.video_reader import parse_filename_hints


# --- Resolution: explicit WxH ---

@pytest.mark.parametrize("filename, expected_w, expected_h", [
    ("video_1920x1080.yuv", 1920, 1080),
    ("video_320X240.yuv", 320, 240),          # case-insensitive X
    ("test_3840x2160_nv12.raw", 3840, 2160),
    ("clip_176x144.yuv", 176, 144),
])
def test_explicit_wxh(filename, expected_w, expected_h):
    hints = parse_filename_hints(filename)
    assert hints['width'] == expected_w
    assert hints['height'] == expected_h


def test_wxh_in_path_separator():
    """WxH in path directory is used as fallback when basename has no WxH."""
    hints = parse_filename_hints("/path/to/1920x1080/video.yuv")
    assert hints['width'] == 1920
    assert hints['height'] == 1080


def test_wxh_in_basename_with_path():
    """WxH in the basename is matched even when a full path is given."""
    hints = parse_filename_hints("/path/to/1920x1080_video.yuv")
    assert hints['width'] == 1920
    assert hints['height'] == 1080


def test_wxh_out_of_range_ignored():
    """WxH below 16 should be ignored."""
    hints = parse_filename_hints("video_1x1.yuv")
    assert 'width' not in hints
    assert 'height' not in hints


# --- Resolution: named aliases ---

@pytest.mark.parametrize("alias, expected_w, expected_h", [
    ("qcif", 176, 144),
    ("cif", 352, 288),
    ("vga", 640, 480),
    ("720p", 1280, 720),
    ("1080p", 1920, 1080),
    ("4k", 3840, 2160),
    ("pal", 720, 576),
    ("ntsc", 720, 480),
])
def test_named_resolution(alias, expected_w, expected_h):
    hints = parse_filename_hints(f"foreman_{alias}.yuv")
    assert hints['width'] == expected_w
    assert hints['height'] == expected_h


def test_wxh_priority_over_named():
    """Explicit WxH should take priority over named alias."""
    hints = parse_filename_hints("video_1920x1080_qcif.yuv")
    assert hints['width'] == 1920
    assert hints['height'] == 1080


# --- Format aliases ---

@pytest.mark.parametrize("token, expected_format", [
    ("i420", "I420"),
    ("nv12", "NV12"),
    ("nv21", "NV21"),
    ("yuyv", "YUYV"),
    ("yuy2", "YUYV"),
    ("rgb24", "RGB3"),
    ("bgr24", "BGR3"),
    ("grey", "GREY"),
    ("gray", "GREY"),
    ("yuv420p", "I420"),
    ("yuv422p", "422P"),
    ("yuv444p", "444P"),
])
def test_format_alias(token, expected_format):
    hints = parse_filename_hints(f"video_{token}.yuv")
    assert hints['format'] == expected_format


# --- FPS ---

@pytest.mark.parametrize("filename, expected_fps", [
    ("video_30fps.yuv", 30.0),
    ("video_15fps.raw", 15.0),
    ("video_29.97fps.yuv", 29.97),
])
def test_fps_suffix(filename, expected_fps):
    hints = parse_filename_hints(filename)
    assert hints['fps'] == pytest.approx(expected_fps)


@pytest.mark.parametrize("filename, expected_fps", [
    ("video_30_.yuv", 30.0),
    ("video_60_.yuv", 60.0),
    ("video_24_.yuv", 24.0),
])
def test_fps_bare_common(filename, expected_fps):
    hints = parse_filename_hints(filename)
    assert hints['fps'] == pytest.approx(expected_fps)


def test_non_fps_number_ignored():
    """Large numbers like resolution digits should not be treated as fps."""
    hints = parse_filename_hints("video_1920x1080.yuv")
    assert 'fps' not in hints


# --- Combined ---

def test_full_filename_all_hints():
    hints = parse_filename_hints("foreman_qcif_15fps_nv12.yuv")
    assert hints['width'] == 176
    assert hints['height'] == 144
    assert hints['format'] == "NV12"
    assert hints['fps'] == pytest.approx(15.0)


def test_no_hints():
    hints = parse_filename_hints("video.yuv")
    assert hints == {}


def test_extension_not_treated_as_token():
    """'.yuv' and '.raw' extensions should not produce format hints."""
    hints = parse_filename_hints("myfile.yuv")
    assert 'format' not in hints
    hints2 = parse_filename_hints("myfile.raw")
    assert 'format' not in hints2


# --- Edge cases ---

def test_multiple_wxh_first_wins():
    """When multiple WxH patterns exist, first match wins."""
    hints = parse_filename_hints("video_1920x1080_640x480.yuv")
    assert hints['width'] == 1920
    assert hints['height'] == 1080


def test_wxh_boundary_min():
    """Minimum valid WxH (16x16) should match."""
    hints = parse_filename_hints("test_16x16.yuv")
    assert hints['width'] == 16
    assert hints['height'] == 16


def test_wxh_boundary_max():
    """Maximum valid WxH (8192x8192) should match."""
    hints = parse_filename_hints("test_8192x8192.yuv")
    assert hints['width'] == 8192
    assert hints['height'] == 8192


def test_wxh_just_below_min_ignored():
    """15x15 is below minimum 16, should be ignored."""
    hints = parse_filename_hints("test_15x15.yuv")
    assert 'width' not in hints


def test_wxh_just_above_max_ignored():
    """8193x8193 is above maximum 8192, should be ignored."""
    hints = parse_filename_hints("test_8193x8193.yuv")
    assert 'width' not in hints


@pytest.mark.parametrize("alias, expected_w, expected_h", [
    ("qvga", 320, 240),
    ("wvga", 800, 480),
    ("svga", 800, 600),
    ("xga", 1024, 768),
    ("hd", 1280, 720),
    ("2k", 2560, 1440),
    ("sd", 720, 576),
])
def test_named_resolution_extended(alias, expected_w, expected_h):
    """Test remaining named aliases not in main parametrize set."""
    hints = parse_filename_hints(f"clip_{alias}.yuv")
    assert hints['width'] == expected_w
    assert hints['height'] == expected_h


def test_format_yv12():
    hints = parse_filename_hints("video_yv12.yuv")
    assert hints['format'] == "YV12"


def test_format_uyvy():
    hints = parse_filename_hints("video_uyvy.yuv")
    assert hints['format'] == "UYVY"


def test_format_yvyu():
    hints = parse_filename_hints("video_yvyu.yuv")
    assert hints['format'] == "YVYU"


def test_format_nv16():
    hints = parse_filename_hints("video_nv16.yuv")
    assert hints['format'] == "NV16"


def test_format_nv61():
    hints = parse_filename_hints("video_nv61.yuv")
    assert hints['format'] == "NV61"


def test_fps_suffix_case_insensitive():
    """FPS suffix pattern should be case-insensitive."""
    hints = parse_filename_hints("video_30FPS.yuv")
    assert hints['fps'] == pytest.approx(30.0)


def test_fps_120():
    hints = parse_filename_hints("video_120fps.yuv")
    assert hints['fps'] == pytest.approx(120.0)


def test_fps_out_of_range_ignored():
    """FPS above 240 should be ignored."""
    hints = parse_filename_hints("video_300fps.yuv")
    assert 'fps' not in hints


def test_combined_wxh_format_no_fps():
    hints = parse_filename_hints("test_640x480_nv21.raw")
    assert hints['width'] == 640
    assert hints['height'] == 480
    assert hints['format'] == "NV21"
    assert 'fps' not in hints


def test_dots_as_separators():
    """Dots should work as token separators."""
    hints = parse_filename_hints("clip.qcif.nv12.yuv")
    assert hints['width'] == 176
    assert hints['height'] == 144
    assert hints['format'] == "NV12"


def test_hyphens_as_separators():
    """Hyphens should work as token separators."""
    hints = parse_filename_hints("clip-cif-i420.yuv")
    assert hints['width'] == 352
    assert hints['height'] == 288
    assert hints['format'] == "I420"


# --- Path WxH fallback ---

def test_path_wxh_fallback():
    """WxH in directory path used as fallback when basename has no WxH."""
    hints = parse_filename_hints("/data/1920x1080/video.yuv")
    assert hints['width'] == 1920
    assert hints['height'] == 1080


def test_path_wxh_basename_takes_priority():
    """Basename WxH takes priority over path WxH."""
    hints = parse_filename_hints("/data/1920x1080/clip_640x480.yuv")
    assert hints['width'] == 640
    assert hints['height'] == 480


def test_path_wxh_out_of_range_ignored():
    """Out-of-range WxH in path should be ignored."""
    hints = parse_filename_hints("/data/1x1/video.yuv")
    assert 'width' not in hints


def test_path_wxh_with_named_alias():
    """Path WxH takes priority over named alias in basename."""
    hints = parse_filename_hints("/data/1920x1080/clip_qcif.yuv")
    assert hints['width'] == 1920
    assert hints['height'] == 1080


# --- Bit depth ---

@pytest.mark.parametrize("token, expected_bd", [
    ("8bit", 8),
    ("10bit", 10),
    ("12bit", 12),
    ("16bit", 16),
])
def test_bit_depth(token, expected_bd):
    hints = parse_filename_hints(f"video_{token}.raw")
    assert hints['bit_depth'] == expected_bd


def test_bit_depth_case_insensitive():
    hints = parse_filename_hints("video_10BIT.raw")
    assert hints['bit_depth'] == 10


def test_bit_depth_invalid_ignored():
    """Non-standard bit depths should be ignored."""
    hints = parse_filename_hints("video_14bit.raw")
    assert 'bit_depth' not in hints


def test_bit_depth_with_format():
    """Bit depth and format should both be extracted."""
    hints = parse_filename_hints("sensor_rggb_10bit.raw")
    assert hints['bit_depth'] == 10


def test_bit_depth_not_in_filename():
    hints = parse_filename_hints("video_1920x1080.yuv")
    assert 'bit_depth' not in hints


def test_full_combined_with_bit_depth():
    """Full filename with all hints including bit depth."""
    hints = parse_filename_hints("sensor_1920x1080_10bit_30fps_nv12.raw")
    assert hints['width'] == 1920
    assert hints['height'] == 1080
    assert hints['bit_depth'] == 10
    assert hints['fps'] == pytest.approx(30.0)
    assert hints['format'] == "NV12"
