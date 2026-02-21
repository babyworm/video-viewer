# YUV/RAW Video Viewer Walkthrough

A standalone GUI tool for viewing and analyzing raw YUV, RGB, and Bayer video files, built with PySide6 (Qt6).

## Features
- **Supported Formats**:
    - **YUV Planar**: I420, YV12, YUV422P, YUV444P, NV12, NV21, NV16, NV61, NV24, NV42, P010, P016
    - **YUV Packed**: YUYV, UYVY, YVYU, VYUY, AYUV, VUYA, Y210
    - **Bayer**: RGGB, BGGR, GBRG, GRBG (8/10/12/16-bit)
    - **RGB**: RGB24, BGR24, ARGB32, RGBA32, RGB565, RGB555, RGB332, and more
    - **Grayscale**: 8/10/12/16-bit
    - **Y4M**: Full header parsing (resolution, format, fps, interlace, PAR)
- **Visualization**:
    - Frame-by-frame navigation with playback
    - Channel separation (Y/U/V or R/G/B) with false color display
    - Split view (2x2 grid: Full + 3 channels)
    - Grid and sub-grid overlay for macroblock inspection
    - Pixel inspector with raw hex values and neighborhood display
    - Zooming (scroll wheel, Ctrl+/-)
- **Analysis**:
    - Histogram, waveform, vectorscope
    - PSNR, SSIM metrics
    - Scene change detection
- **Auto-detection**:
    - Y4M header parsing (resolution, format, fps)
    - Resolution guessing from file size
    - Filename metadata extraction (resolution, format, fps patterns)
- **Multi-tab**: Open multiple files in separate tabs
- **A/B Comparison**: Side-by-side, overlay, and diff modes
- **Conversion**: Single file and batch format conversion
- **Export**: Save frames as PNG/BMP, export clips
- **BT.601 / BT.709**: Selectable color matrix
- **Dark/Light theme**

## Setup

Requires Python 3.10+ with `numpy`, `opencv-python`, `PySide6`, `pyqtgraph`, and `scikit-image`.

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### GUI mode

```bash
# Auto-detect parameters (Y4M)
video_viewer input.y4m

# Specify parameters (raw YUV)
video_viewer input.yuv --width 1920 --height 1080 --format I420

# Filename hints are extracted automatically
# e.g. foreman_qcif_15fps_nv12.yuv → 176x144, NV12, 15fps
```

### Headless conversion

```bash
video_viewer input.yuv --width 1920 --height 1080 -vi I420 -vo NV12 -o output.nv12
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `Left` / `Right` | Previous / Next frame |
| `Ctrl+Left` / `Ctrl+Right` | Previous / Next scene change |
| `0` | Full (all channels) |
| `1` | Channel 1 (Y/R) |
| `2` | Channel 2 (U/G) |
| `3` | Channel 3 (V/B) |
| `4` | Split view (2x2) |
| `G` | Toggle grid overlay |
| `Shift+G` | Toggle sub-grid overlay |
| `B` | Toggle bookmark |
| `Ctrl+B` | Next bookmark |
| `Ctrl+Shift+B` | Previous bookmark |
| `Ctrl+C` | Copy frame to clipboard |

## Running Tests

```bash
pytest test/ -v
```
