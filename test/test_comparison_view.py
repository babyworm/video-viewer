"""
Test the ComparisonView module
"""
import pytest
import numpy as np
from PySide6.QtWidgets import QApplication
from video_viewer.comparison_view import ComparisonCanvas, ComparisonWindow, ComparisonMode
from video_viewer.video_reader import VideoReader
from video_viewer.format_manager import FormatManager
import sys

@pytest.fixture
def qapp():
    """Create QApplication instance for tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_comparison_canvas_creation(qapp):
    """Test that ComparisonCanvas can be instantiated"""
    canvas = ComparisonCanvas()
    assert canvas is not None
    assert canvas.mode == ComparisonMode.SPLIT
    assert canvas.split_position == 0.5
    assert canvas.overlay_opacity == 0.5


def test_comparison_canvas_set_images(qapp):
    """Test setting images on the canvas"""
    canvas = ComparisonCanvas()

    # Create dummy RGB images
    img_a = np.zeros((480, 640, 3), dtype=np.uint8)
    img_b = np.ones((480, 640, 3), dtype=np.uint8) * 255

    canvas.set_images(img_a, img_b)

    assert canvas.image_a is not None
    assert canvas.image_b is not None


def test_comparison_canvas_modes(qapp):
    """Test switching between comparison modes"""
    canvas = ComparisonCanvas()

    canvas.set_mode(ComparisonMode.OVERLAY)
    assert canvas.mode == ComparisonMode.OVERLAY

    canvas.set_mode(ComparisonMode.DIFF)
    assert canvas.mode == ComparisonMode.DIFF

    canvas.set_mode(ComparisonMode.SPLIT)
    assert canvas.mode == ComparisonMode.SPLIT


def test_comparison_canvas_split_position(qapp):
    """Test split position adjustment"""
    canvas = ComparisonCanvas()

    canvas.set_split_position(0.3)
    assert canvas.split_position == 0.3

    canvas.set_split_position(0.7)
    assert canvas.split_position == 0.7

    # Test bounds
    canvas.set_split_position(-0.5)
    assert canvas.split_position == 0.0

    canvas.set_split_position(1.5)
    assert canvas.split_position == 1.0


def test_comparison_canvas_overlay_opacity(qapp):
    """Test overlay opacity adjustment"""
    canvas = ComparisonCanvas()

    canvas.set_overlay_opacity(0.25)
    assert canvas.overlay_opacity == 0.25

    canvas.set_overlay_opacity(0.75)
    assert canvas.overlay_opacity == 0.75

    # Test bounds
    canvas.set_overlay_opacity(-0.5)
    assert canvas.overlay_opacity == 0.0

    canvas.set_overlay_opacity(1.5)
    assert canvas.overlay_opacity == 1.0


def test_comparison_window_creation(qapp):
    """Test that ComparisonWindow can be instantiated with a VideoReader"""
    # Create a dummy VideoReader (this assumes test video exists)
    import os
    test_file = os.path.join(os.path.dirname(__file__), "..", "bus_qcif_15fps.y4m")

    if not os.path.exists(test_file):
        pytest.skip("Test video file not found")

    # Y4M files auto-detect format/resolution
    reader = VideoReader(test_file, 176, 144, "I420")

    window = ComparisonWindow(reader)
    assert window is not None
    assert window.main_reader == reader
    assert window.ref_reader is None
    assert window.current_frame == 0


def test_comparison_mode_enum():
    """Test ComparisonMode enum values"""
    assert ComparisonMode.SPLIT.value == "Split View"
    assert ComparisonMode.OVERLAY.value == "Overlay"
    assert ComparisonMode.DIFF.value == "Difference"
