import pytest
import subprocess
import sys
import shutil
import os

def test_cli_help():
    """Test that the CLI command exists and prints help."""
    # We use 'video_viewer' command. It should be in path if installed in venv.
    # Since we run tests via .venv/bin/pytest, PATH should include .venv/bin?
    # Or we explicitly use the executable path.

    executable = shutil.which("video_viewer")
    if not executable:
        # Fallback to sys.executable -m video_viewer.main if command not found (e.g. dev env issues)
        cmd = [sys.executable, "-m", "video_viewer.main", "--help"]
    else:
        cmd = [executable, "--help"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "YUV/RAW Video Viewer" in result.stdout

def test_cli_headless_conversion(tmp_path):
    """Test headless conversion via CLI."""
    infile = tmp_path / "input.i420"
    outfile = tmp_path / "output.nv12"

    # Create 1 frame I420 (4x4)
    width, height = 4, 4
    y_size = 16
    uv_size = 4
    file_size = y_size + 2 * uv_size # 24 bytes
    infile.write_bytes(b'\x00' * file_size)

    executable = shutil.which("video_viewer")
    if not executable:
        base_cmd = [sys.executable, "-m", "video_viewer.main"]
    else:
        base_cmd = [executable]

    cmd = base_cmd + [
        str(infile),
        "--width", str(width),
        "--height", str(height),
        "-vi", "I420",
        "-vo", "NV12",
        "-o", str(outfile)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0
    assert os.path.exists(outfile)
    assert outfile.stat().st_size == file_size # NV12 is same size as I420
    assert f"Converted 1 frames" in result.stdout

def test_cli_missing_args(tmp_path):
    """Test CLI behavior with missing arguments for raw files in headless mode."""
    infile = tmp_path / "input.raw"
    infile.write_bytes(b'\x00' * 10)

    executable = shutil.which("video_viewer")
    if not executable:
        base_cmd = [sys.executable, "-m", "video_viewer.main"]
    else:
        base_cmd = [executable]

    # Test headless mode with tiny file (defaults to 1920x1080 -> 0 frames)
    outfile = tmp_path / "out.yuv"
    cmd = base_cmd + [
        str(infile),
        "-o", str(outfile)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    assert "Converted 0 frames" in result.stdout
