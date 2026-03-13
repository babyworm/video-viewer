# YUV/Raw Video Viewer

## Version

- Current version: defined in `rust/Cargo.toml` (`version`)
- Follows Semantic Versioning (SemVer): `MAJOR.MINOR.PATCH`

### Versioning Rules

- **MAJOR**: Only the user (babyworm) decides when to bump the major version.
- **MINOR**: Increment when adding new features (e.g., new UI panel, new analysis tool, new format support).
- **PATCH**: Increment for bug fixes, small UI tweaks, refactoring without new features.
- Always update `rust/Cargo.toml` when the version changes.
- **Every commit that adds features or fixes bugs must bump the version.**

## Project Structure

- `rust/` - Main Rust codebase (eframe/egui)
  - `Cargo.toml` - Dependencies and version
  - `src/main.rs` - CLI entry point (clap, headless conversion)
  - `src/lib.rs` - Library root, GUI launch
  - `src/app.rs` - VideoViewerApp (main application state, keyboard shortcuts, frame logic)
  - `src/core/` - Core logic
    - `formats.rs` - FormatType, VideoFormat, FORMAT_DEFS (75+ pixel format definitions)
    - `reader.rs` - VideoReader (file I/O, frame seeking, RGB conversion, channel extraction)
    - `cache.rs` - FrameCache (LRU memory-bounded cache)
    - `hints.rs` - parse_filename_hints (resolution, format, fps, bit depth from filename)
    - `y4m.rs` - Y4M header parser and frame offset builder
    - `pixel.rs` - get_pixel_info (pixel inspector values, hex, neighborhood)
  - `src/ui/` - UI components
    - `canvas.rs` - ImageCanvas (rendering, zoom, grid overlay)
    - `toolbar.rs` - Toolbar (component selection, grid controls, colorize_channel)
    - `sidebar.rs` - Sidebar (analysis tabs: histogram, waveform, vectorscope, metrics)
    - `navigation.rs` - NavigationBar (frame slider, playback controls)
    - `dialogs.rs` - Open, Save, Parameters, Export, Convert, Settings dialogs
    - `comparison.rs` - ComparisonView (split, overlay, diff modes)
    - `settings.rs` - Settings persistence (toml)
  - `src/analysis/` - Analysis tools
    - `histogram.rs` - RGB and luma histograms
    - `waveform.rs` - Waveform display
    - `vectorscope.rs` - BT.709 YCbCr vectorscope
    - `metrics.rs` - PSNR, SSIM, frame difference
    - `scene.rs` - Scene change detection
  - `src/conversion/` - Format conversion
    - `converter.rs` - VideoConverter, extract/pack YUV planes, chroma resampling
  - `tests/` - Integration tests
- `scripts/generate_test_data.py` - Test data generator (Python/OpenCV)
- `test_data/` - Sample raw video files (QCIF I420, NV12, RGB565, YUYV)

## Testing Conventions

- Run tests: `cd rust && cargo test`
- All tests must pass before any release.
- Framework: Rust integration tests in `rust/tests/`

### Test Patterns

- Use `tempfile::NamedTempFile` or `tempfile::tempdir()` for temporary files
- Minimal I420 frame: 4x4 = 24 bytes (16 Y + 4 U + 4 V)
- Helper functions: `make_raw_i420(frames)`, `make_i420_frame(y, u, v)`
- Test naming: `test_<module>_<behavior>` (e.g., `test_pixel_info_yuyv_odd`)

### Test File Organization

| File | Scope |
|------|-------|
| `tests/formats_test.rs` | Format lookup, frame_size, categories (21 tests) |
| `tests/formats_extra_test.rs` | RGB16/32, semi-planar, packed frame sizes (9 tests) |
| `tests/reader_test.rs` | VideoReader open, seek, Y4M, RGB convert, channels (6 tests) |
| `tests/pixel_test.rs` | Pixel info: I420, YV12, NV12, NV21, RGB24, BGR24, Grey (12 tests) |
| `tests/pixel_packed_test.rs` | Pixel info: YUYV, UYVY, NV16 packed formats (5 tests) |
| `tests/hints_test.rs` | Filename hint parsing (10 tests) |
| `tests/y4m_test.rs` | Y4M header parsing, frame offsets (8 tests) |
| `tests/cache_test.rs` | LRU cache operations (6 tests) |
| `tests/converter_test.rs` | I420→NV12, identity, multi-frame, roundtrip, cancel (5 tests) |
| `tests/converter_extra_test.rs` | resample_chroma, I420→YV12, I420→422P, NV12 roundtrip (6 tests) |
| `tests/histogram_test.rs` | Histogram RGB/Y modes (2 tests) |
| `tests/waveform_test.rs` | Waveform luma/R/G/B, edge cases (6 tests) |
| `tests/vectorscope_test.rs` | Vectorscope neutral/red/blue, subsampling (5 tests) |
| `tests/metrics_test.rs` | PSNR, SSIM, frame difference (4 tests) |
| `tests/scene_test.rs` | Scene detection, threshold, save/load (4 tests) |
| `tests/integration_test.rs` | Real Y4M file (conditional) (1 test) |

## Filename Hint Reference

When adding new aliases, update `rust/src/core/hints.rs` tables AND this section.

### Named Resolutions

| Alias | Width | Height | Notes |
|-------|-------|--------|-------|
| `qcif` | 176 | 144 | Quarter CIF |
| `cif` | 352 | 288 | Common Intermediate Format |
| `qvga` | 320 | 240 | Quarter VGA |
| `vga` | 640 | 480 | |
| `wvga` | 800 | 480 | Wide VGA |
| `svga` | 800 | 600 | |
| `xga` | 1024 | 768 | |
| `hd` / `720p` | 1280 | 720 | |
| `1080p` | 1920 | 1080 | |
| `2k` | 2560 | 1440 | QHD |
| `4k` | 3840 | 2160 | UHD |
| `sd` / `pal` | 720 | 576 | PAL SD |
| `ntsc` | 720 | 480 | NTSC SD |

### Format Aliases

| Token(s) | Maps To | Type |
|----------|---------|------|
| `i420`, `yuv420p`, `yuv420` | I420 | YUV Planar 4:2:0 |
| `yv12` | YV12 | YUV Planar 4:2:0 |
| `nv12` | NV12 | YUV Semi-Planar 4:2:0 |
| `nv21` | NV21 | YUV Semi-Planar 4:2:0 |
| `nv16` | NV16 | YUV Semi-Planar 4:2:2 |
| `nv61` | NV61 | YUV Semi-Planar 4:2:2 |
| `yuv422p`, `yuv422` | YUV422P | YUV Planar 4:2:2 |
| `yuv444p`, `yuv444` | YUV444P | YUV Planar 4:4:4 |
| `yuyv`, `yuy2` | YUYV | YUV Packed 4:2:2 |
| `uyvy` | UYVY | YUV Packed 4:2:2 |
| `yvyu` | YVYU | YUV Packed 4:2:2 |
| `rgb24`, `rgb` | RGB24 | RGB 24-bit |
| `bgr24`, `bgr` | BGR24 | RGB 24-bit |
| `grey`, `gray` | Greyscale (8-bit) | Greyscale |

### Bit Depth Tokens (parsed from filename)

| Token | Value | Used For |
|-------|-------|----------|
| `8bit` | 8 | Default (usually omitted) |
| `10bit` | 10 | Bayer 10-bit, Grey 10-bit |
| `12bit` | 12 | Bayer 12-bit, Grey 12-bit |
| `16bit` | 16 | Bayer 16-bit, Grey 16-bit |

## Copyright

Copyright (c) babyworm (Hyun-Gyu Kim)
