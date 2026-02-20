# Video Viewer

Raw/YUV/RGB video viewer built with PySide6 (Qt6). Supports 30+ pixel formats for inspecting uncompressed video data.

## Features

- **30+ pixel formats**: YUV (I420, NV12, NV21, YUY2, UYVY, ...), RGB (RGB24, BGR24, RGB565, ...), Bayer, grayscale
- **Channel separation**: View individual Y/U/V or R/G/B channels with false color display
- **Split view**: 2x2 grid showing Full + 3 channels simultaneously (key `4`)
- **Keyboard shortcuts**: `0`-`4` for channel switching, `Space` for play/pause, arrow keys for navigation
- **A/B comparison**: Side-by-side, overlay, and diff modes for comparing two videos
- **Analysis tools**: Histogram, waveform, vectorscope
- **Pixel inspector**: Hover to see pixel values in all color spaces
- **Bookmarks & scene detection**: Mark frames and auto-detect scene changes
- **Y4M support**: Auto-detect parameters from Y4M headers
- **Resolution guessing**: Auto-detect resolution from file size
- **Frame export**: Save frames as PNG/BMP, export clips with format conversion
- **BT.601 / BT.709**: Selectable YUV-RGB color matrix
- **Dark/Light theme**: Toggle with menu

## Installation

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
```

### Headless conversion

```bash
video_viewer input.yuv --width 1920 --height 1080 -vi I420 -vo NV12 -o output.nv12
```

### Keyboard Shortcuts

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
| `B` | Toggle bookmark |
| `Ctrl+B` | Next bookmark |
| `Ctrl+Shift+B` | Previous bookmark |
| `Ctrl+C` | Copy frame to clipboard |

## Supported Formats

### YUV Planar
I420, YV12, I422, I444, NV12, NV21, NV16, NV61, NV24, NV42

### YUV Packed
YUY2 (YUYV), UYVY, YVYU, VYUY

### RGB
RGB24, BGR24, RGBA, BGRA, ARGB, ABGR, RGB565, BGR565, RGB555

### Bayer
BGGR8, GBRG8, GRBG8, RGGB8 (+ 10/12/16 bit variants)

### Grayscale
GRAY8, GRAY10, GRAY12, GRAY16

## Development

```bash
# Run tests
pytest test/ -v

# Run specific test
pytest test/test_main_window.py::test_colorize_channel -v
```

## Dependencies

| Package | License | Purpose |
|---------|---------|---------|
| PySide6 | LGPL v3 | Qt6 GUI framework |
| numpy | BSD 3-Clause | Array operations |
| opencv-python | Apache 2.0 | Image processing |
| pyqtgraph | MIT | Plot widgets |
| scikit-image | BSD 3-Clause | Image analysis |

All dependencies are compatible with the MIT license.

## License

MIT License. See [LICENSE](LICENSE) for details.
