# Rust Video Viewer — Full Rewrite Design Spec

## Overview

Full rewrite of the YUV/Raw Video Viewer from Python (PySide6 + numpy + opencv) to Rust (egui + opencv-rust). The goal is a functionally equivalent application with improved performance through native compilation and multicore parallelism via rayon.

## Technology Stack

| Layer | Python (current) | Rust (target) |
|-------|-------------------|---------------|
| GUI | PySide6 (Qt6) | egui + eframe |
| Image/CV | opencv-python, numpy | opencv-rust |
| Array ops | numpy | ndarray |
| Parallelism | QThread (GIL-bound) | rayon |
| Image export | opencv | image crate |
| Plots | pyqtgraph | egui_plot |
| Settings | QSettings | serde + toml (config file) |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  main.rs  (CLI parsing: clap)                       │
├─────────────────────────────────────────────────────┤
│  app.rs  (eframe::App — top-level state machine)    │
├──────────┬──────────┬───────────┬───────────────────┤
│  ui/     │  core/   │ analysis/ │ conversion/       │
│          │          │           │                   │
│ menu.rs  │reader.rs │histog.rs  │converter.rs       │
│ toolbar  │cache.rs  │waveform   │                   │
│ canvas   │formats.rs│vectorsc   │                   │
│ sidebar  │y4m.rs    │metrics.rs │                   │
│ dialogs  │pixel.rs  │           │                   │
│ compare  │hints.rs  │           │                   │
│ settings │decode.rs │           │                   │
└──────────┴──────────┴───────────┴───────────────────┘
```

### Module Breakdown

#### `core/` — Format definitions, frame I/O, caching

- **`formats.rs`** — FormatType enum (YUV_PLANAR, YUV_SEMI_PLANAR, YUV_PACKED, BAYER, RGB, GREY, COMPRESSED), VideoFormat struct, all 80+ pixel formats with frame size calculation, fourcc mapping, subsampling info. Direct port of `format_manager.py`. Note: `COMPRESSED` is a placeholder variant for future extension; no formats of that type are registered at launch.
- **`reader.rs`** — VideoReader: mmap-based frame access, `seek_frame()`, `convert_to_rgb()` (delegates to opencv), `get_channels()`, color matrix support (BT.601/BT.709). Thread-safe with `Arc<Mutex<>>`. Note: P010/P016 (10/16-bit NV12) require a 16-bit-per-sample unpack with right-shift before `cvt_color`. MIPI CSI-2 packed 10-bit Bayer formats (`pRAA` etc.) require a 5→4 pixel unpack step before demosaic (currently unimplemented in Python version; implement in Rust or mark as unsupported).
- **`cache.rs`** — LRU frame cache (lru crate or manual LinkedHashMap). Memory-budget based eviction (default 512MB). Uses `parking_lot::RwLock` for concurrent read access.
- **`y4m.rs`** — Y4M header parser, offset table builder, colorspace/fps/interlace extraction.
- **`pixel.rs`** — Pixel inspector: raw hex extraction, component values per format, neighborhood grid.
- **`hints.rs`** — `parse_filename_hints()`: resolution aliases, format aliases, fps/bit_depth token extraction. Pure function, direct port.
- **`decode.rs`** — Frame decode pipeline: background thread pool (rayon) for prefetch buffer. Replaces single-threaded QThread `FrameDecodeWorker`. Ring buffer of N pre-decoded frames via `crossbeam::ArrayQueue`. Color matrix and component mode are passed as copies into each rayon task. The prefetch buffer must be flushed when color_matrix or component mode changes mid-playback.

#### `ui/` — egui interface

- **`app.rs`** — Top-level `eframe::App` impl. Manages state: current file, frame index, playback, component mode, zoom, etc.
- **`menu.rs`** — Menu bar: File, View, Tools, Help. Maps to actions.
- **`toolbar.rs`** — Icon toolbar: play/pause, navigation, zoom, grid, component buttons, FPS selector.
- **`canvas.rs`** — Image display widget using `egui::TextureHandle`. Zoom (scroll), pan (drag), grid overlay rendering, sub-grid, crosshair. Pixel coordinate tracking for inspector. Channel button labels must reflect the current format type: Y/U/V for YUV formats, R/G/B for RGB/Bayer/Grey (matching Python behavior where `get_channels()` returns R/G/B keys for non-YUV formats).
- **`sidebar.rs`** — Right panel: pixel inspector display, analysis tabs (histogram, waveform, vectorscope, metrics).
- **`navigation.rs`** — Bottom bar: frame slider with markers (bookmarks=cyan, scenes=red), frame counter, FPS combo.
- **`dialogs.rs`** — Modal dialogs: ParametersDialog, ExportDialog, ConvertDialog, BatchConvertDialog, PngExportDialog, SettingsDialog, ShortcutsDialog, BookmarkDialog.
- **`comparison.rs`** — ComparisonWindow: split/overlay/diff modes, dual-canvas, opacity slider, per-pixel diff inspector. Can be a separate egui window or panel.
- **`settings.rs`** — Settings persistence via serde + TOML file (`~/.config/video-viewer/settings.toml`).

#### `analysis/` — Video analysis tools

- **`histogram.rs`** — RGB/Y histogram (256 bins). Rendered with `egui_plot`.
- **`waveform.rs`** — Waveform monitor (256 x width). Custom egui painting or texture.
- **`vectorscope.rs`** — Cb vs Cr scatter plot. `egui_plot` scatter.
- **`metrics.rs`** — PSNR (opencv), SSIM (manual implementation using Wang et al. 2004 formula: Gaussian 11x11 window, C1=(0.01*255)^2, C2=(0.03*255)^2), frame difference, histogram correlation.
- **`scene.rs`** — Scene change detection: mean diff + histogram correlation, threshold-based. Includes scene list save/load (plain text, one frame index per line, matching Python implementation).

#### `conversion/` — Format conversion engine

- **`converter.rs`** — VideoConverter: direct YUV-to-YUV (extract planes → resample chroma → repack), RGB intermediate fallback. Batch support. Progress callback.

## Key Design Decisions

### 1. Frame Prefetch with Rayon

Replace single-frame QThread decode with a ring buffer prefetch:

```rust
// Prefetch N frames ahead during playback
struct PrefetchBuffer {
    buffer: crossbeam::queue::ArrayQueue<(usize, DecodedFrame)>,
    prefetch_count: usize, // e.g., 8 frames
}
```

During playback, rayon spawns decode tasks for upcoming frames into a lock-free `crossbeam::ArrayQueue`. The render loop consumes from the buffer without contention. This eliminates frame drops at high FPS/resolution. The buffer is flushed on color_matrix, component mode, or seek changes.

### 2. egui Texture Pipeline

```
raw bytes → convert_to_rgb() → egui::ColorImage → TextureHandle → GPU
```

egui uploads textures to GPU via its renderer (glow/wgpu). Frame updates only re-upload the texture, keeping GPU-side rendering efficient. For channel views, the colorized channel image is uploaded instead.

### 3. Component View (Channel Selection)

Same logic as Python version:
- Mode 0: Full RGB
- Mode 1-3: Single channel with false color (colorize_channel)
- Mode 4: 2x2 split view (composite image)

Applied at decode time in the prefetch pipeline, not as a post-process, to avoid the flickering bug that existed in the Python version.

### 4. Grid Overlay

Rendered as egui painter lines on top of the image texture, in screen space. Grid sizes: [0, 16, 32, 64, 128], sub-grid: [0, 4, 8, 16]. Color: green (grid), yellow (sub-grid).

### 5. Comparison View

Implemented as a separate egui window (`ctx.show_viewport()`). Three modes:
- **Split**: vertical divider, draggable
- **Overlay**: alpha blending with opacity slider
- **Diff**: heatmap (amplified 10x, color-coded per component)

### 6. Settings Persistence

TOML file instead of QSettings:

```toml
[cache]
max_memory_mb = 512

[display]
zoom_min = 0.1
zoom_max = 50.0
dark_theme = true

[defaults]
fps = 30
color_matrix = "BT.601"
width = 1920
height = 1080
```

### 7. CLI (clap)

Same interface as Python version:

```bash
# GUI mode
video_viewer input.y4m
video_viewer input.yuv --width 1920 --height 1080 --format I420

# Headless conversion
video_viewer input.yuv -w 1920 -h 1080 -vi I420 -vo NV12 -o output.nv12
```

## Pixel Format Support

All 80+ formats from `format_manager.py` will be ported. Frame size calculations and conversion paths remain identical. opencv-rust provides the same `cvt_color`, Bayer demosaic, and resize functions.

### Format Categories

| Category | Count | Key Formats |
|----------|-------|-------------|
| YUV Planar | 9 | I420, YV12, 422P, 444P |
| YUV Semi-Planar | 10 | NV12, NV21, NV16, P010 |
| YUV Packed | 8 | YUYV, UYVY, YVYU, AYUV |
| RGB | 24 | RGB24, BGR24, RGBA, ARGB, RGB565 |
| Bayer | 16 | RGGB/BGGR/GBRG/GRBG × 8/10/12/16bit |
| Grey | 4 | GREY, Y10, Y12, Y16 |

## Keyboard Shortcuts

All shortcuts from the Python version will be preserved:

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| Left/Right | Prev/Next frame |
| Home/End | First/Last frame |
| Ctrl+Left/Right | Prev/Next scene change |
| 0-4 | Component view (Full/Y/U/V/Split) |
| G / Shift+G | Grid / Sub-grid toggle |
| B | Toggle bookmark |
| Ctrl+B / Ctrl+Shift+B | Next/Prev bookmark |
| F | Fit to view |
| 1 / 2 | Zoom 1:1 / 2:1 |
| +/- | Zoom in/out |
| Ctrl+O | Open file |
| Ctrl+S | Save frame |
| Ctrl+C | Copy to clipboard |

## Cargo Dependencies

```toml
[dependencies]
eframe = "0.31"
egui = "0.31"
egui_plot = "0.31"
opencv = { version = "0.93", features = ["clang-runtime"] }
ndarray = "0.16"
rayon = "1.10"
image = "0.25"
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
toml = "0.8"
memmap2 = "0.9"
lru = "0.12"
parking_lot = "0.12"
crossbeam = "0.8"      # lock-free prefetch queue
arboard = { version = "3", features = ["wayland-data-control"] }  # clipboard (Wayland support)
rfd = "0.15"           # native file dialogs
log = "0.4"
env_logger = "0.11"
```

## Performance Targets

| Metric | Python (current) | Rust (target) |
|--------|-------------------|---------------|
| Startup time | ~2s (Qt init) | <0.5s |
| 1080p frame decode | ~5-8ms | ~3-5ms |
| 4K frame decode | ~15-25ms | ~8-12ms |
| Playback 1080p@30fps | Occasional drops | Smooth (prefetch) |
| Playback 4K@30fps | Frame drops likely | Smooth with prefetch |
| Memory (idle) | ~80MB (Python+Qt) | ~20-30MB |

## File Structure

```
video-viewer-rs/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── app.rs
│   ├── core/
│   │   ├── mod.rs
│   │   ├── formats.rs
│   │   ├── reader.rs
│   │   ├── cache.rs
│   │   ├── y4m.rs
│   │   ├── pixel.rs
│   │   ├── hints.rs
│   │   └── decode.rs
│   ├── ui/
│   │   ├── mod.rs
│   │   ├── menu.rs
│   │   ├── toolbar.rs
│   │   ├── canvas.rs
│   │   ├── sidebar.rs
│   │   ├── navigation.rs
│   │   ├── dialogs.rs
│   │   ├── comparison.rs
│   │   └── settings.rs
│   ├── analysis/
│   │   ├── mod.rs
│   │   ├── histogram.rs
│   │   ├── waveform.rs
│   │   ├── vectorscope.rs
│   │   ├── metrics.rs
│   │   └── scene.rs
│   └── conversion/
│       ├── mod.rs
│       └── converter.rs
└── assets/
    └── icons/
```

## Migration Strategy

1. **Core first**: formats → reader → cache → y4m → hints (testable without GUI)
2. **Minimal GUI**: app shell with canvas, file open, frame display
3. **Navigation**: slider, playback, keyboard shortcuts
4. **Channel views**: component selection, split view, colorization
5. **Analysis**: histogram, waveform, vectorscope, metrics
6. **Comparison**: A/B window with split/overlay/diff
7. **Conversion**: format converter, export, batch
8. **Polish**: dialogs, settings, bookmarks, scene detection, pixel inspector

## Out of Scope

- Plugin system
- Network streaming
- Hardware-accelerated decode (future enhancement)
- Multi-tab (egui windows can substitute)
