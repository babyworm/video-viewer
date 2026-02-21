# YUV/Raw Video Viewer

## Version

- Current version: defined in `video_viewer/__init__.py` (`__version__`)
- Follows Semantic Versioning (SemVer): `MAJOR.MINOR.PATCH`

### Versioning Rules

- **MAJOR**: Only the user (babyworm) decides when to bump the major version.
- **MINOR**: Increment when adding new features (e.g., new UI panel, new analysis tool, new format support).
- **PATCH**: Increment for bug fixes, small UI tweaks, refactoring without new features.
- Always update `video_viewer/__init__.py` when the version changes.
- **Every commit that adds features or fixes bugs must bump the version.**
- The About dialog reads `__version__` automatically.

## Project Structure

- `video_viewer/` - Main package
  - `__init__.py` - Package version (`__version__`)
  - `main.py` - CLI entry point (argparse, headless conversion)
  - `main_window.py` - ImageCanvas, MainWindow (UI, toolbar, menus, shortcuts)
  - `video_reader.py` - VideoReader, FrameCache, parse_filename_hints (format decoding, pixel inspection, filename metadata extraction)
  - `format_manager.py` - FormatManager, FormatType (YUV/RGB/Bayer format definitions)
  - `constants.py` - Shared constants (resolutions, FPS options, theme, defaults)
  - `dialogs.py` - ParametersDialog, ExportDialog, ConvertDialog, SettingsDialog, etc.
  - `analysis.py` - VideoAnalyzer (PSNR, SSIM, histogram, waveform, vectorscope)
  - `comparison_view.py` - ComparisonWindow (A/B compare)
  - `video_converter.py` - VideoConverter (format conversion engine)
  - `log_config.py` - Logging configuration
- `test/` - pytest test suite

## Testing Conventions

- Run tests: `.venv/bin/python -m pytest test/ -v`
- All tests must pass before any release.
- Framework: pytest with PySide6 Qt integration

### Fixtures & Patterns

- `qapp` — QApplication fixture (defined in each test file, reuses existing instance)
- `tmp_path` — pytest built-in for temporary file creation
- Minimal I420 frame data: `b'\x80' * int(w * h * 1.5)` (4x4 default: 24 bytes)
- Y4M helper: `_make_y4m(tmp_path, name, header_params)` in `test_main_window.py`
- Use `pytest.mark.parametrize` for variant testing (resolution aliases, format tokens, etc.)
- Use `unittest.mock.patch` for dialog mocking and method interception

### Test File Organization

| File | Scope | Qt Required |
|------|-------|-------------|
| `test/test_filename_hints.py` | `parse_filename_hints()` pure function | No |
| `test/test_main_window.py` | MainWindow integration, `_apply_file_hints`, Y4M fps | Yes (`qapp`) |
| `test/test_pixel_inspector.py` | `get_pixel_info()` for various formats | No |
| `test/test_suite.py` | VideoReader core operations | No |
| `test/test_video_reader.py` | VideoReader file I/O | No |

## Filename Hint Reference

When adding new aliases, update BOTH `video_reader.py` dictionaries AND this table.

### Named Resolutions (`_NAMED_RESOLUTIONS`)

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

### Format Aliases (`_FORMAT_ALIASES`)

| Token(s) | Maps To | Type |
|----------|---------|------|
| `i420`, `yuv420p`, `yuv420` | I420 | YUV Planar 4:2:0 |
| `yv12` | YV12 | YUV Planar 4:2:0 |
| `nv12` | NV12 | YUV Semi-Planar 4:2:0 |
| `nv21` | NV21 | YUV Semi-Planar 4:2:0 |
| `nv16` | NV16 | YUV Semi-Planar 4:2:2 |
| `nv61` | NV61 | YUV Semi-Planar 4:2:2 |
| `yuv422p`, `yuv422` | 422P | YUV Planar 4:2:2 |
| `yuv444p`, `yuv444` | 444P | YUV Planar 4:4:4 |
| `yuyv`, `yuy2` | YUYV | YUV Packed 4:2:2 |
| `uyvy` | UYVY | YUV Packed 4:2:2 |
| `yvyu` | YVYU | YUV Packed 4:2:2 |
| `rgb24`, `rgb` | RGB3 | RGB 24-bit |
| `bgr24`, `bgr` | BGR3 | RGB 24-bit |
| `grey`, `gray` | GREY | Greyscale 8-bit |

### Bit Depth Tokens (parsed from filename)

| Token | Value | Used For |
|-------|-------|----------|
| `8bit` | 8 | Default (usually omitted) |
| `10bit` | 10 | Bayer 10-bit, Grey 10-bit |
| `12bit` | 12 | Bayer 12-bit, Grey 12-bit |
| `16bit` | 16 | Bayer 16-bit, Grey 16-bit |

## Copyright

Copyright (c) babyworm (Hyun-Gyu Kim)
