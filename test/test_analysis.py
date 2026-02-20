import pytest
import numpy as np
import cv2
from video_viewer.analysis import VideoAnalyzer

def test_histogram_rgb():
    # Create synthetic RGB image (Solid Red)
    width, height = 32, 32
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = 255 # R (OpenCV is usually BGR in cvtColor, but calcHist depends on input. Analyzer assumes standard order from VideoReader)
    # VideoReader converts to RGB. Analyzer assumes input is RGB.

    # Analyzer docs say: channels="RGB" -> assumes input is RGB.
    # calculate_histogram uses:
    # for i, color in enumerate(('r', 'g', 'b')):
    #    calcHist([image], [i]...)
    # So index 0 is R, 1 is G, 2 is B.

    hists = VideoAnalyzer.calculate_histogram(img, "RGB")

    assert 'r' in hists
    assert 'g' in hists
    assert 'b' in hists

    # Red channel should have all pixels at 255
    assert hists['r'][255] == width * height
    assert hists['r'][0] == 0

    # Green/Blue should be at 0
    assert hists['g'][0] == width * height
    assert hists['b'][0] == width * height

def test_histogram_gray():
    width, height = 32, 32
    img = np.zeros((height, width), dtype=np.uint8)
    img[:] = 128

    hists = VideoAnalyzer.calculate_histogram(img, "Y")
    assert 'y' in hists
    assert hists['y'][128] == width * height

def test_psnr_identical():
    img1 = np.zeros((32, 32, 3), dtype=np.uint8)
    # Perfect match -> Infinite PSNR (or very high capped by implementation)
    # OpenCV PSNR returns > 300 or similar for identical
    # If standard formula, div by zero mse -> inf
    # Let's see what OpenCV returns. Usually capped or Inf.

    psnr = VideoAnalyzer.calculate_psnr(img1, img1)
    # OpenCV return 0 if identical? No, it returns infinite, but represented as?
    # Actually cv2.PSNR(a, a) returns infinity? or just very large?
    # Let's check diff

    img2 = img1.copy()
    img2[0,0,0] = 1 # Small diff

    psnr_diff = VideoAnalyzer.calculate_psnr(img1, img2)
    assert psnr_diff > 0
    assert psnr_diff < 100 # Reasonable PSNR for 1 pixel diff on small image

def test_ssim_identical():
    img1 = np.zeros((32, 32, 3), dtype=np.uint8)
    ssim = VideoAnalyzer.calculate_ssim(img1, img1)
    assert ssim == 1.0

def test_vectorscope_rgb():
    # Solid Red in RGB is roughly Cr max, Cb min/mid.
    # RGB (255, 0, 0) -> Y=76, Cr=255, Cb=84? (Approx)

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)

    cb, cr = VideoAnalyzer.calculate_vectorscope_from_rgb(img)

    assert len(cb) == 32 * 32
    assert len(cr) == 32 * 32

    # Check values are consistent
    assert np.all(cb == cb[0])
    assert np.all(cr == cr[0])

    # Cr should be high for Red
    assert cr[0] > 128
    # Cb should be low/mid
    assert cb[0] < 128


def test_waveform_luma():
    """Test waveform monitor calculation for luma channel."""
    img = np.zeros((32, 64, 3), dtype=np.uint8)
    # Left half bright, right half dark
    img[:, :32, :] = 200
    img[:, 32:, :] = 50

    wf = VideoAnalyzer.calculate_waveform(img, channel="luma")
    assert wf is not None
    assert wf.shape[0] == 256  # 256 intensity levels
    assert wf.shape[1] == 64   # width preserved (<=720)


def test_waveform_channel_r():
    """Test waveform for red channel."""
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :, 0] = 128  # R=128

    wf = VideoAnalyzer.calculate_waveform(img, channel="r")
    assert wf is not None
    # All columns should have counts concentrated at level 128
    assert wf[128, :].sum() > 0
    assert wf[0, :].sum() == 0


def test_waveform_none():
    """Test waveform with None input."""
    assert VideoAnalyzer.calculate_waveform(None) is None


def test_frame_difference_identical():
    """Test frame difference for identical frames."""
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    diff = VideoAnalyzer.calculate_frame_difference(img, img)
    assert diff == 0.0


def test_frame_difference_opposite():
    """Test frame difference for maximally different frames."""
    img1 = np.zeros((32, 32, 3), dtype=np.uint8)
    img2 = np.full((32, 32, 3), 255, dtype=np.uint8)
    diff = VideoAnalyzer.calculate_frame_difference(img1, img2)
    assert diff == 255.0


def test_frame_difference_none():
    """Test frame difference with None input."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    assert VideoAnalyzer.calculate_frame_difference(None, img) == 0.0
    assert VideoAnalyzer.calculate_frame_difference(img, None) == 0.0


def test_frame_difference_shape_mismatch():
    """Test frame difference with different shapes."""
    img1 = np.zeros((32, 32, 3), dtype=np.uint8)
    img2 = np.zeros((64, 64, 3), dtype=np.uint8)
    diff = VideoAnalyzer.calculate_frame_difference(img1, img2)
    assert diff == 255.0
