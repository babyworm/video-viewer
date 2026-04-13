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
    - `hints.rs` - Single source of truth for named resolutions (`NAMED_RESOLUTIONS`), filename hint parsing (`parse_filename_hints`), file-size-based resolution guess (`guess_resolution_from_size`), unified open-time resolver (`resolve_raw_params`)
    - `y4m.rs` - Y4M header parser and frame offset builder
    - `pixel.rs` - get_pixel_info (pixel inspector values, hex, neighborhood)
    - `sideband.rs` - Schema-driven sideband binary parser (ISP parameter overlay)
  - `src/ui/` - UI components
    - `canvas.rs` - ImageCanvas (rendering, zoom, grid overlay)
    - `toolbar.rs` - Toolbar (component selection, grid controls, colorize_channel)
    - `sidebar.rs` - Sidebar (analysis tabs: histogram, waveform, vectorscope, metrics)
    - `navigation.rs` - NavigationBar (frame slider, playback controls)
    - `dialogs.rs` - Open, Save, Parameters, Export, Convert, Settings dialogs
    - `comparison.rs` - ComparisonView (split, overlay, diff modes)
    - `settings.rs` - Settings persistence (toml)
    - `sideband_overlay.rs` - Sideband CTU heatmap overlay rendering
  - `src/analysis/` - Analysis tools
    - `histogram.rs` - RGB and luma histograms
    - `waveform.rs` - Waveform display
    - `vectorscope.rs` - BT.709 YCbCr vectorscope
    - `metrics.rs` - PSNR, SSIM, frame difference
    - `scene.rs` - Scene change detection
    - `isp_sideband.rs` - SidebandPanel UI (load/unload, overlay mode, opacity)
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
| `tests/sideband_test.rs` | Sideband binary parsing, extended header, signed fields, display (13 tests) |
| `tests/integration_test.rs` | Real Y4M file (conditional) (1 test) |

## Filename Hint Reference

When adding new aliases, update `rust/src/core/hints.rs` tables AND this section.

### Named Resolutions

| Alias | Width | Height | Notes |
|-------|-------|--------|-------|
| `sqcif` | 128 | 96 | Sub-QCIF |
| `qcif` | 176 | 144 | Quarter CIF |
| `cif` | 352 | 288 | Common Intermediate Format |
| `sif` | 352 | 240 | Source Input Format (NTSC-rate CIF) |
| `2cif` | 704 | 288 | Double CIF |
| `4cif` | 704 | 576 | 4× CIF (D1 PAL) |
| `16cif` | 1408 | 1152 | 16× CIF |
| `d1` | 720 | 480 | ITU-R BT.601 D1 (NTSC) |
| `sd` / `pal` | 720 | 576 | PAL SD |
| `ntsc` | 720 | 480 | NTSC SD |
| `qqvga` | 160 | 120 | Quarter QVGA |
| `qvga` | 320 | 240 | Quarter VGA |
| `hvga` | 480 | 320 | Half VGA |
| `vga` | 640 | 480 | |
| `wvga` | 800 | 480 | Wide VGA |
| `svga` | 800 | 600 | |
| `xga` | 1024 | 768 | |
| `wxga` | 1280 | 800 | Wide XGA |
| `sxga` | 1280 | 1024 | Super XGA |
| `wsxga` | 1680 | 1050 | Wide SXGA+ |
| `uxga` | 1600 | 1200 | Ultra XGA |
| `wuxga` | 1920 | 1200 | Wide UXGA |
| `qxga` | 2048 | 1536 | Quad XGA |
| `wqxga` | 2560 | 1600 | Wide QXGA |
| `qhd` | 960 | 540 | Quarter HD |
| `hd` / `720p` | 1280 | 720 | HD |
| `fhd` / `fullhd` / `1080p` | 1920 | 1080 | Full HD |
| `wqhd` / `1440p` / `2k` | 2560 | 1440 | WQHD |
| `uhd` / `4kuhd` / `4k` / `2160p` | 3840 | 2160 | UHD-1 |
| `8k` / `4320p` | 7680 | 4320 | UHD-2 |
| `240p` | 320 | 240 | NNNp shorthand |
| `360p` | 640 | 360 | NNNp shorthand |
| `480p` | 640 | 480 | NNNp shorthand |
| `576p` | 720 | 576 | NNNp shorthand |

### Resolution Resolver (single source of truth)

`rust/src/core/hints.rs::NAMED_RESOLUTIONS` is the canonical table. It simultaneously feeds:

1. **Filename alias lookup** — `parse_filename_hints()`
2. **View → Video Size menu** — entries with `show_in_menu: true`
3. **File-size-based resolution guess** — same menu set, sorted largest-first, crossed with `I420, NV12, YUYV, RGB24` (recovered from Python's `_guess_resolution`)

`resolve_raw_params(path, file_size, default_w, default_h, default_fmt)` applies this priority order:

1. Filename carries **both width and height** → use them (format from hint or default)
2. Else file-size guess succeeds → use it, emit info string for the status bar
3. Else fall back to configured defaults

When adding a new entry: edit `NAMED_RESOLUTIONS` only; the menu, HashMap lookup, and guess candidate set all update automatically.

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

## ISP Sideband Overlay (추가 기능, 계획)

**중요: 이 기능은 코어 뷰어와 분리된 추가 기능이다. 기존 코어 기능에 영향을 주지 않도록 메뉴/모듈을 분리할 것.**

### 개요

isp_emulator가 생성한 sideband.bin 파일을 로딩하여 CTU 단위 오버레이를 영상 위에 표시.
현재 64px 그리드 오버레이가 이미 존재하므로 이를 확장하는 형태.

### 연관 프로젝트

- **isp_emulator** (`/home/babyworm/work/isp_emulator/`)
  - sideband.bin 생성 (Y4M → CTU 분석 → 바이너리 출력)
  - `src/output/sideband_reader.rs`에 바이너리 파서 존재 — 이것을 복사하거나 공유 crate로 추출
  - dump 명령어로 text/hjson/csv 출력 가능

### Sideband 바이너리 형식 (v0)

```
프레임마다:
  Header:  "IP" + version(1B) + numCtus(1B 또는 0xFF + 2B extended)
  Frame:   20 bytes (scene_class, noise_class, motion_class, QP bias 등)
  CTUs:    16 bytes × numCtus (activity, flatness, edge, qp_delta, sao_prior 등)
```

- Big-endian, Q8.8 고정소수점 (256 = 1.0)
- CTU 순서: 래스터 스캔 (좌→우, 상→하)
- 1080p 64x64 CTU: 510개 (30 cols × 17 rows)

### 구현 방향

1. **모듈 분리**: `src/analysis/isp_sideband.rs` 또는 `src/core/sideband.rs` — 코어 뷰어 코드와 분리
2. **UI 분리**: Analysis 탭에 "ISP Sideband" 탭 추가 (기존 Histogram/Waveform/Vectorscope 옆)
3. **오버레이 종류**: QP delta 히트맵, activity/flatness 컬러맵, saliency 맵, confidence 맵
4. **프레임 동기화**: 영상 프레임 이동 시 sideband 프레임도 자동 연동
5. **로딩**: File 메뉴 또는 Analysis 패널에서 sideband.bin 파일 선택

## Copyright

Copyright (c) babyworm (Hyun-Gyu Kim)
