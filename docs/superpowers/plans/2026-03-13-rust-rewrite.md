# Rust Video Viewer Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Full rewrite of Python YUV/Raw Video Viewer to Rust with egui GUI, opencv-rust backend, and rayon-based multicore prefetch.

**Architecture:** Cargo workspace with four module groups: `core/` (formats, reader, cache, y4m, hints, pixel, decode), `ui/` (egui app, menu, toolbar, canvas, sidebar, navigation, dialogs, comparison, settings), `analysis/` (histogram, waveform, vectorscope, metrics, scene), `conversion/` (converter). The Python source at `video_viewer/` serves as the reference implementation.

**Tech Stack:** Rust, egui/eframe 0.31, opencv-rust 0.93, ndarray 0.16, rayon 1.10, crossbeam 0.8, clap 4, serde/toml, memmap2, image 0.25

**Spec:** `docs/superpowers/specs/2026-03-13-rust-rewrite-design.md`

**Python reference:** `video_viewer/*.py`

---

## File Structure

```
rust/
├── Cargo.toml
├── src/
│   ├── main.rs                  # CLI (clap), entry point
│   ├── app.rs                   # eframe::App top-level state
│   ├── core/
│   │   ├── mod.rs
│   │   ├── formats.rs           # FormatType enum, VideoFormat, 80+ formats
│   │   ├── reader.rs            # VideoReader: mmap, seek, convert_to_rgb, get_channels
│   │   ├── cache.rs             # LRU frame cache, memory-budget eviction
│   │   ├── y4m.rs               # Y4M header parser, offset table
│   │   ├── pixel.rs             # Pixel inspector: hex, components, neighborhood
│   │   ├── hints.rs             # parse_filename_hints(): resolution/format/fps aliases
│   │   └── decode.rs            # Prefetch pipeline: rayon + crossbeam::ArrayQueue
│   ├── ui/
│   │   ├── mod.rs
│   │   ├── menu.rs              # Menu bar (File, View, Tools, Help)
│   │   ├── toolbar.rs           # Icon toolbar, component buttons, FPS selector
│   │   ├── canvas.rs            # Image texture, zoom, pan, grid overlay
│   │   ├── sidebar.rs           # Pixel inspector, analysis tabs
│   │   ├── navigation.rs        # Frame slider with markers, frame counter
│   │   ├── dialogs.rs           # Parameters, Export, Convert, Batch, Settings, etc.
│   │   ├── comparison.rs        # A/B comparison: split/overlay/diff
│   │   └── settings.rs          # TOML persistence (~/.config/video-viewer/settings.toml)
│   ├── analysis/
│   │   ├── mod.rs
│   │   ├── histogram.rs         # RGB/Y histogram (256 bins)
│   │   ├── waveform.rs          # Waveform monitor
│   │   ├── vectorscope.rs       # Cb vs Cr scatter
│   │   ├── metrics.rs           # PSNR, SSIM, frame diff, histogram correlation
│   │   └── scene.rs             # Scene detection, scene list save/load
│   └── conversion/
│       ├── mod.rs
│       └── converter.rs         # Format conversion engine, batch support
└── tests/
    ├── formats_test.rs
    ├── reader_test.rs
    ├── cache_test.rs
    ├── y4m_test.rs
    ├── hints_test.rs
    ├── pixel_test.rs
    ├── metrics_test.rs
    ├── converter_test.rs
    └── scene_test.rs
```

---

## Chunk 1: Project Scaffold & Core Formats

### Task 1: Initialize Cargo project

**Files:**
- Create: `rust/Cargo.toml`
- Create: `rust/src/main.rs`

- [ ] **Step 1: Create Cargo project**

```bash
cd /home/babyworm/work/video-viewer
mkdir -p rust/src
```

- [ ] **Step 2: Write Cargo.toml**

Create `rust/Cargo.toml` with all dependencies from spec:
```toml
[package]
name = "video-viewer"
version = "0.1.0"
edition = "2021"

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
crossbeam = "0.8"
arboard = { version = "3", features = ["wayland-data-control"] }
rfd = "0.15"
log = "0.4"
env_logger = "0.11"
```

- [ ] **Step 3: Write minimal main.rs**

```rust
fn main() {
    println!("video-viewer-rs");
}
```

- [ ] **Step 4: Verify build**

Run: `cd rust && cargo build`
Expected: Compiles successfully (dependencies download)

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "feat: initialize Rust project with dependencies"
```

---

### Task 2: Core module structure & FormatType enum

**Files:**
- Create: `rust/src/core/mod.rs`
- Create: `rust/src/core/formats.rs`
- Create: `rust/tests/formats_test.rs`

**Reference:** `video_viewer/format_manager.py` — FormatType enum (line 8-15), VideoFormat class (line 17-47)

- [ ] **Step 1: Write failing test for FormatType and VideoFormat**

Create `rust/tests/formats_test.rs`:
```rust
use video_viewer::core::formats::{FormatType, VideoFormat, get_all_formats, get_format_by_name};

#[test]
fn test_format_type_variants() {
    // All 7 types from Python FormatType enum
    let types = [
        FormatType::YuvPlanar,
        FormatType::YuvSemiPlanar,
        FormatType::YuvPacked,
        FormatType::Bayer,
        FormatType::Rgb,
        FormatType::Grey,
        FormatType::Compressed,
    ];
    assert_eq!(types.len(), 7);
}

#[test]
fn test_i420_format() {
    let fmt = get_format_by_name("I420").unwrap();
    assert_eq!(fmt.fourcc, "YU12");
    assert_eq!(fmt.format_type, FormatType::YuvPlanar);
    assert_eq!(fmt.bits_per_pixel, 12); // 4:2:0 = 12 bpp
    // Frame size: W*H*1.5
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 3 / 2);
}

#[test]
fn test_nv12_format() {
    let fmt = get_format_by_name("NV12").unwrap();
    assert_eq!(fmt.format_type, FormatType::YuvSemiPlanar);
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 3 / 2);
}

#[test]
fn test_yuyv_format() {
    let fmt = get_format_by_name("YUYV").unwrap();
    assert_eq!(fmt.format_type, FormatType::YuvPacked);
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 2);
}

#[test]
fn test_rgb24_format() {
    let fmt = get_format_by_name("RGB24").unwrap();
    assert_eq!(fmt.format_type, FormatType::Rgb);
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080 * 3);
}

#[test]
fn test_bayer_rggb8_format() {
    let fmt = get_format_by_name("Bayer RGGB 8-bit").unwrap();
    assert_eq!(fmt.format_type, FormatType::Bayer);
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080);
}

#[test]
fn test_grey_format() {
    let fmt = get_format_by_name("Grey 8-bit").unwrap();
    assert_eq!(fmt.format_type, FormatType::Grey);
    assert_eq!(fmt.frame_size(1920, 1080), 1920 * 1080);
}

#[test]
fn test_all_formats_count() {
    let all = get_all_formats();
    // Python version has 75+ formats
    assert!(all.len() >= 70, "Expected at least 70 formats, got {}", all.len());
}

#[test]
fn test_format_lookup_not_found() {
    assert!(get_format_by_name("NONEXISTENT").is_none());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd rust && cargo test --test formats_test`
Expected: FAIL — module `core::formats` does not exist

- [ ] **Step 3: Create core module structure**

Create `rust/src/core/mod.rs`:
```rust
pub mod formats;
```

Update `rust/src/main.rs` to expose as library:
```rust
pub mod core;

fn main() {
    println!("video-viewer-rs");
}
```

Also create `rust/src/lib.rs`:
```rust
pub mod core;
```

- [ ] **Step 4: Implement FormatType and VideoFormat**

Create `rust/src/core/formats.rs`. Port all format definitions from `video_viewer/format_manager.py`:
- `FormatType` enum with 7 variants
- `VideoFormat` struct: name, fourcc, format_type, bits_per_pixel, subsampling (h_sub, v_sub), bit_depth, components
- `frame_size(width, height) -> usize` method
- `get_all_formats() -> Vec<VideoFormat>` — register all 80+ formats
- `get_format_by_name(name) -> Option<&VideoFormat>` — lookup by display name
- `get_format_by_fourcc(fourcc) -> Option<&VideoFormat>` — lookup by fourcc
- `get_formats_by_type(format_type) -> Vec<&VideoFormat>` — filter by type

Use `once_cell::sync::Lazy` or `std::sync::LazyLock` for the static format registry.

Key frame size formulas from Python (`_calculate_frame_size`):
- YUV Planar 4:2:0: `w * h * 3 / 2`
- YUV Planar 4:2:2: `w * h * 2`
- YUV Planar 4:4:4: `w * h * 3`
- Semi-Planar: same as equivalent planar
- P010/P016: `w * h * 3` (16-bit samples in 4:2:0)
- Packed YUYV: `w * h * 2`
- Packed AYUV: `w * h * 4`
- RGB 24-bit: `w * h * 3`
- RGB 32-bit: `w * h * 4`
- RGB 16-bit: `w * h * 2`
- RGB332: `w * h`
- Bayer 8-bit: `w * h`
- Bayer 10p (MIPI packed): `w * h * 5 / 4`
- Bayer 10/12/16-bit: `w * h * 2`
- Grey 8-bit: `w * h`
- Grey 10/12/16-bit: `w * h * 2`

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd rust && cargo test --test formats_test`
Expected: All 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add rust/src/core/ rust/src/lib.rs rust/tests/formats_test.rs
git commit -m "feat: add FormatType enum and 80+ pixel format definitions"
```

---

### Task 3: Filename hints parser

**Files:**
- Create: `rust/src/core/hints.rs`
- Create: `rust/tests/hints_test.rs`

**Reference:** `video_viewer/video_reader.py` — `parse_filename_hints()` (line ~63-170), `_NAMED_RESOLUTIONS` dict, `_FORMAT_ALIASES` dict

- [ ] **Step 1: Write failing tests**

Create `rust/tests/hints_test.rs`. Port test cases from `test/test_filename_hints.py`:
```rust
use video_viewer::core::hints::{parse_filename_hints, FilenameHints};

#[test]
fn test_resolution_wxh() {
    let h = parse_filename_hints("test_1920x1080_nv12.yuv");
    assert_eq!(h.width, Some(1920));
    assert_eq!(h.height, Some(1080));
}

#[test]
fn test_named_resolution_720p() {
    let h = parse_filename_hints("video_720p.yuv");
    assert_eq!(h.width, Some(1280));
    assert_eq!(h.height, Some(720));
}

#[test]
fn test_format_alias_nv12() {
    let h = parse_filename_hints("test_1920x1080_nv12.yuv");
    assert_eq!(h.format.as_deref(), Some("NV12"));
}

#[test]
fn test_format_alias_i420() {
    let h = parse_filename_hints("test_i420_cif.yuv");
    assert_eq!(h.format.as_deref(), Some("I420"));
}

#[test]
fn test_fps_suffix() {
    let h = parse_filename_hints("video_30fps.yuv");
    assert_eq!(h.fps, Some(30.0));
}

#[test]
fn test_bit_depth() {
    let h = parse_filename_hints("video_10bit.yuv");
    assert_eq!(h.bit_depth, Some(10));
}

#[test]
fn test_named_resolution_cif() {
    let h = parse_filename_hints("foreman_cif.yuv");
    assert_eq!(h.width, Some(352));
    assert_eq!(h.height, Some(288));
}

#[test]
fn test_named_resolution_4k() {
    let h = parse_filename_hints("video_4k_nv12.yuv");
    assert_eq!(h.width, Some(3840));
    assert_eq!(h.height, Some(2160));
}

#[test]
fn test_path_fallback_resolution() {
    let h = parse_filename_hints("/data/1920x1080/video.yuv");
    assert_eq!(h.width, Some(1920));
    assert_eq!(h.height, Some(1080));
}

#[test]
fn test_no_hints() {
    let h = parse_filename_hints("video.yuv");
    assert_eq!(h.width, None);
    assert_eq!(h.height, None);
    assert_eq!(h.format, None);
    assert_eq!(h.fps, None);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd rust && cargo test --test hints_test`
Expected: FAIL

- [ ] **Step 3: Implement hints parser**

Create `rust/src/core/hints.rs`:
- `FilenameHints` struct: `width: Option<u32>`, `height: Option<u32>`, `format: Option<String>`, `fps: Option<f64>`, `bit_depth: Option<u32>`
- `parse_filename_hints(filename: &str) -> FilenameHints`
- Static `NAMED_RESOLUTIONS`: HashMap of alias → (width, height) — all 14 entries from CLAUDE.md
- Static `FORMAT_ALIASES`: HashMap of token → format name — all entries from CLAUDE.md
- Parse order: WxH pattern → path WxH fallback → named resolution → format aliases → Nfps → Nbit

Add `pub mod hints;` to `rust/src/core/mod.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd rust && cargo test --test hints_test`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add rust/src/core/hints.rs rust/src/core/mod.rs rust/tests/hints_test.rs
git commit -m "feat: add filename hints parser with resolution/format/fps aliases"
```

---

### Task 4: Y4M header parser

**Files:**
- Create: `rust/src/core/y4m.rs`
- Create: `rust/tests/y4m_test.rs`

**Reference:** `video_viewer/video_reader.py` — Y4M parsing in `VideoReader.__init__()` (lines ~200-280)

- [ ] **Step 1: Write failing tests**

Create `rust/tests/y4m_test.rs`:
```rust
use video_viewer::core::y4m::{Y4mHeader, parse_y4m_header};

#[test]
fn test_basic_y4m_header() {
    let header = b"YUV4MPEG2 W720 H576 F25:1 Ip C420\n";
    let h = parse_y4m_header(header).unwrap();
    assert_eq!(h.width, 720);
    assert_eq!(h.height, 576);
    assert_eq!(h.fps_num, 25);
    assert_eq!(h.fps_den, 1);
    assert_eq!(h.colorspace, "420");
    assert_eq!(h.interlace, "progressive");
}

#[test]
fn test_y4m_30000_1001() {
    let header = b"YUV4MPEG2 W1920 H1080 F30000:1001 Ip C420\n";
    let h = parse_y4m_header(header).unwrap();
    assert_eq!(h.fps_num, 30000);
    assert_eq!(h.fps_den, 1001);
    let fps = h.fps_num as f64 / h.fps_den as f64;
    assert!((fps - 29.97).abs() < 0.01);
}

#[test]
fn test_y4m_422() {
    let header = b"YUV4MPEG2 W640 H480 F30:1 Ip C422\n";
    let h = parse_y4m_header(header).unwrap();
    assert_eq!(h.colorspace, "422");
}

#[test]
fn test_y4m_mono() {
    let header = b"YUV4MPEG2 W640 H480 F30:1 Ip Cmono\n";
    let h = parse_y4m_header(header).unwrap();
    assert_eq!(h.colorspace, "mono");
}

#[test]
fn test_y4m_colorspace_to_format() {
    let h420 = parse_y4m_header(b"YUV4MPEG2 W4 H4 F1:1 C420\n").unwrap();
    assert_eq!(h420.to_format_name(), "I420");

    let h422 = parse_y4m_header(b"YUV4MPEG2 W4 H4 F1:1 C422\n").unwrap();
    assert_eq!(h422.to_format_name(), "YUV422P");

    let h444 = parse_y4m_header(b"YUV4MPEG2 W4 H4 F1:1 C444\n").unwrap();
    assert_eq!(h444.to_format_name(), "YUV444P");

    let hmono = parse_y4m_header(b"YUV4MPEG2 W4 H4 F1:1 Cmono\n").unwrap();
    assert_eq!(hmono.to_format_name(), "Grey 8-bit");
}

#[test]
fn test_y4m_invalid() {
    assert!(parse_y4m_header(b"NOT_Y4M W4 H4\n").is_err());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd rust && cargo test --test y4m_test`
Expected: FAIL

- [ ] **Step 3: Implement Y4M parser**

Create `rust/src/core/y4m.rs`:
- `Y4mHeader` struct: width, height, fps_num, fps_den, colorspace, interlace, pixel_aspect
- `parse_y4m_header(data: &[u8]) -> Result<Y4mHeader, String>`
- `Y4mHeader::to_format_name() -> &str` — maps colorspace to internal format name
- `Y4mHeader::fps() -> f64`
- `build_frame_offsets(data: &[u8], header: &Y4mHeader) -> Vec<usize>` — scan for "FRAME\n" markers

Add `pub mod y4m;` to `rust/src/core/mod.rs`.

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test --test y4m_test`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add rust/src/core/y4m.rs rust/src/core/mod.rs rust/tests/y4m_test.rs
git commit -m "feat: add Y4M header parser with colorspace/fps/interlace support"
```

---

### Task 5: LRU frame cache

**Files:**
- Create: `rust/src/core/cache.rs`
- Create: `rust/tests/cache_test.rs`

**Reference:** `video_viewer/video_reader.py` — FrameCache class (lines ~20-60)

- [ ] **Step 1: Write failing tests**

Create `rust/tests/cache_test.rs`:
```rust
use video_viewer::core::cache::FrameCache;

#[test]
fn test_cache_put_get() {
    let mut cache = FrameCache::new(1024 * 1024); // 1MB
    let data = vec![0u8; 100];
    cache.put(0, data.clone());
    assert_eq!(cache.get(0), Some(&data));
}

#[test]
fn test_cache_miss() {
    let cache = FrameCache::new(1024 * 1024);
    assert_eq!(cache.get(5), None);
}

#[test]
fn test_cache_eviction() {
    // 300 bytes budget, 100-byte frames → max 3 frames
    let mut cache = FrameCache::new(300);
    for i in 0..5 {
        cache.put(i, vec![0u8; 100]);
    }
    // Oldest frames (0, 1) should be evicted
    assert!(cache.get(0).is_none());
    assert!(cache.get(1).is_none());
    assert!(cache.get(4).is_some());
}

#[test]
fn test_cache_clear() {
    let mut cache = FrameCache::new(1024 * 1024);
    cache.put(0, vec![0u8; 100]);
    cache.clear();
    assert!(cache.get(0).is_none());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_cache_memory_usage() {
    let mut cache = FrameCache::new(1024 * 1024);
    cache.put(0, vec![0u8; 1000]);
    cache.put(1, vec![0u8; 2000]);
    assert_eq!(cache.memory_usage(), 3000);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd rust && cargo test --test cache_test`
Expected: FAIL

- [ ] **Step 3: Implement FrameCache**

Create `rust/src/core/cache.rs`:
- `FrameCache` struct using `lru::LruCache` or manual `LinkedHashMap`
- `new(max_memory_bytes: usize) -> Self`
- `get(frame_idx: usize) -> Option<&Vec<u8>>`
- `put(frame_idx: usize, data: Vec<u8>)`
- `clear()`
- `len() -> usize`
- `memory_usage() -> usize`
- Evicts oldest entries when memory budget exceeded

Add `pub mod cache;` to `rust/src/core/mod.rs`.

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test --test cache_test`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add rust/src/core/cache.rs rust/src/core/mod.rs rust/tests/cache_test.rs
git commit -m "feat: add LRU frame cache with memory-budget eviction"
```

---

## Chunk 2: VideoReader & Pixel Inspector

### Task 6: VideoReader — file I/O and frame access

**Files:**
- Create: `rust/src/core/reader.rs`
- Create: `rust/tests/reader_test.rs`

**Reference:** `video_viewer/video_reader.py` — VideoReader class (lines ~70-200, seek_frame, mmap)

- [ ] **Step 1: Write failing tests**

Create `rust/tests/reader_test.rs`:
```rust
use video_viewer::core::reader::VideoReader;
use std::io::Write;

fn create_test_file(path: &std::path::Path, width: u32, height: u32, frames: usize) {
    let frame_size = (width as usize) * (height as usize) * 3 / 2; // I420
    let mut f = std::fs::File::create(path).unwrap();
    for i in 0..frames {
        let data: Vec<u8> = vec![(i & 0xFF) as u8; frame_size];
        f.write_all(&data).unwrap();
    }
}

#[test]
fn test_reader_open_raw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.yuv");
    create_test_file(&path, 4, 4, 3);
    let reader = VideoReader::open(path.to_str().unwrap(), 4, 4, "I420", "BT.601").unwrap();
    assert_eq!(reader.width(), 4);
    assert_eq!(reader.height(), 4);
    assert_eq!(reader.total_frames(), 3);
}

#[test]
fn test_reader_seek_frame() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.yuv");
    create_test_file(&path, 4, 4, 3);
    let mut reader = VideoReader::open(path.to_str().unwrap(), 4, 4, "I420", "BT.601").unwrap();
    let raw = reader.seek_frame(0).unwrap();
    assert_eq!(raw.len(), 24); // 4*4*1.5 = 24
    assert_eq!(raw[0], 0);

    let raw1 = reader.seek_frame(1).unwrap();
    assert_eq!(raw1[0], 1);
}

#[test]
fn test_reader_out_of_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.yuv");
    create_test_file(&path, 4, 4, 2);
    let mut reader = VideoReader::open(path.to_str().unwrap(), 4, 4, "I420", "BT.601").unwrap();
    assert!(reader.seek_frame(5).is_err());
}

#[test]
fn test_reader_y4m() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.y4m");
    let mut f = std::fs::File::create(&path).unwrap();
    let header = b"YUV4MPEG2 W4 H4 F25:1 Ip C420\n";
    f.write_all(header).unwrap();
    // 1 frame: "FRAME\n" + 24 bytes I420
    f.write_all(b"FRAME\n").unwrap();
    f.write_all(&vec![128u8; 24]).unwrap();
    drop(f);

    let reader = VideoReader::open(path.to_str().unwrap(), 0, 0, "", "BT.601").unwrap();
    assert_eq!(reader.width(), 4);
    assert_eq!(reader.height(), 4);
    assert_eq!(reader.total_frames(), 1);
    assert!(reader.is_y4m());
}
```

Add `tempfile = "3"` to `[dev-dependencies]` in `Cargo.toml`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd rust && cargo test --test reader_test`
Expected: FAIL

- [ ] **Step 3: Implement VideoReader**

Create `rust/src/core/reader.rs`:
- `VideoReader` struct: file handle, mmap, width, height, format, frame_size, total_frames, cache, y4m state
- `open(path, width, height, format_name, color_matrix) -> Result<Self>`
  - Detect Y4M by magic bytes → parse header → use Y4M params
  - Otherwise use provided params
  - Calculate frame_size, total_frames
  - Attempt mmap, fallback to File
- `seek_frame(idx) -> Result<Vec<u8>>` — check cache → read from mmap/file → cache → return
- `width()`, `height()`, `total_frames()`, `is_y4m()`, `format_name()` getters
- `close()` / Drop impl

Add `pub mod reader;` to `rust/src/core/mod.rs`.

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test --test reader_test`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add rust/src/core/reader.rs rust/src/core/mod.rs rust/tests/reader_test.rs rust/Cargo.toml
git commit -m "feat: add VideoReader with mmap, frame cache, Y4M auto-detection"
```

---

### Task 7: convert_to_rgb and get_channels

**Files:**
- Modify: `rust/src/core/reader.rs`
- Create: `rust/tests/reader_rgb_test.rs`

**Reference:** `video_viewer/video_reader.py` — `convert_to_rgb()` (lines ~280-400), `get_channels()` (lines ~700-810), `_colorize_channel()` from `main_window.py`

- [ ] **Step 1: Write failing tests**

Create `rust/tests/reader_rgb_test.rs`:
```rust
use video_viewer::core::reader::VideoReader;
use std::io::Write;

#[test]
fn test_convert_i420_to_rgb() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.yuv");
    // 4x4 I420: 16 Y + 4 U + 4 V = 24 bytes
    let mut data = vec![128u8; 16]; // Y=128 (mid gray)
    data.extend(vec![128u8; 4]);     // U=128 (no chroma)
    data.extend(vec![128u8; 4]);     // V=128 (no chroma)
    std::fs::write(&path, &data).unwrap();

    let mut reader = VideoReader::open(path.to_str().unwrap(), 4, 4, "I420", "BT.601").unwrap();
    let raw = reader.seek_frame(0).unwrap();
    let rgb = reader.convert_to_rgb(&raw).unwrap();
    assert_eq!(rgb.len(), 4 * 4 * 3); // 48 bytes RGB
    // Mid-gray YUV should produce near-gray RGB (~128 each channel)
    assert!((rgb[0] as i32 - 128).abs() < 5);
}

#[test]
fn test_get_channels_yuv() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.yuv");
    let mut data = vec![200u8; 16]; // Y=200
    data.extend(vec![100u8; 4]);     // U=100
    data.extend(vec![150u8; 4]);     // V=150
    std::fs::write(&path, &data).unwrap();

    let mut reader = VideoReader::open(path.to_str().unwrap(), 4, 4, "I420", "BT.601").unwrap();
    let raw = reader.seek_frame(0).unwrap();
    let channels = reader.get_channels(&raw);
    assert!(channels.contains_key("Y"));
    assert!(channels.contains_key("U"));
    assert!(channels.contains_key("V"));
    assert_eq!(channels["Y"].len(), 4 * 4); // full res
}

#[test]
fn test_get_channels_rgb() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.rgb");
    let data = vec![128u8; 4 * 4 * 3]; // RGB24
    std::fs::write(&path, &data).unwrap();

    let mut reader = VideoReader::open(path.to_str().unwrap(), 4, 4, "RGB24", "BT.601").unwrap();
    let raw = reader.seek_frame(0).unwrap();
    let channels = reader.get_channels(&raw);
    assert!(channels.contains_key("R"));
    assert!(channels.contains_key("G"));
    assert!(channels.contains_key("B"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd rust && cargo test --test reader_rgb_test`
Expected: FAIL — methods not implemented

- [ ] **Step 3: Implement convert_to_rgb and get_channels**

Add to `rust/src/core/reader.rs`:
- `convert_to_rgb(&self, raw: &[u8]) -> Result<Vec<u8>>` — dispatches by FormatType:
  - YUV Planar/Semi-Planar: construct opencv Mat, `cvt_color` (COLOR_YUV2RGB_I420, etc.)
  - YUV Packed: opencv cvt_color (COLOR_YUV2RGB_YUYV, etc.)
  - Bayer: opencv `cvt_color` with BayerXX2RGB
  - RGB: reorder channels if needed (BGR→RGB)
  - Grey: cvt_color GRAY2RGB
  - BT.709: manual matrix multiply (r = y + 1.5748*v, etc.)
- `get_channels(&self, raw: &[u8]) -> HashMap<String, Vec<u8>>` — extracts individual planes:
  - YUV: returns {"Y": y_plane, "U": u_plane_upsampled, "V": v_plane_upsampled}
  - RGB/Bayer/Grey: convert to RGB first, return {"R", "G", "B"}

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test --test reader_rgb_test`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add rust/src/core/reader.rs rust/tests/reader_rgb_test.rs
git commit -m "feat: add convert_to_rgb and get_channels for all format types"
```

---

### Task 8: Pixel inspector

**Files:**
- Create: `rust/src/core/pixel.rs`
- Create: `rust/tests/pixel_test.rs`

**Reference:** `video_viewer/video_reader.py` — `get_pixel_info()` (lines ~430-700), `test/test_pixel_inspector.py`

- [ ] **Step 1: Write failing tests**

Create `rust/tests/pixel_test.rs`:
```rust
use video_viewer::core::pixel::get_pixel_info;
use video_viewer::core::formats::get_format_by_name;

#[test]
fn test_pixel_info_i420() {
    let fmt = get_format_by_name("I420").unwrap();
    // 4x4 I420: 16Y + 4U + 4V
    let mut data = vec![0u8; 16];
    data[0] = 200; // Y at (0,0) = 200
    data.extend(vec![100u8; 4]); // U
    data.extend(vec![150u8; 4]); // V

    let info = get_pixel_info(&data, 4, 4, fmt, 0, 0, 0);
    assert_eq!(info.components["Y"], 200);
    assert_eq!(info.components["U"], 100);
    assert_eq!(info.components["V"], 150);
}

#[test]
fn test_pixel_info_rgb24() {
    let fmt = get_format_by_name("RGB24").unwrap();
    let mut data = vec![0u8; 4 * 4 * 3];
    data[0] = 255; // R at (0,0)
    data[1] = 128; // G at (0,0)
    data[2] = 64;  // B at (0,0)

    let info = get_pixel_info(&data, 4, 4, fmt, 0, 0, 0);
    assert_eq!(info.components["R"], 255);
    assert_eq!(info.components["G"], 128);
    assert_eq!(info.components["B"], 64);
}

#[test]
fn test_pixel_info_hex() {
    let fmt = get_format_by_name("I420").unwrap();
    let mut data = vec![0xABu8; 24];
    let info = get_pixel_info(&data, 4, 4, fmt, 0, 0, 0);
    assert!(info.raw_hex.contains("AB"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd rust && cargo test --test pixel_test`
Expected: FAIL

- [ ] **Step 3: Implement pixel inspector**

Create `rust/src/core/pixel.rs`:
- `PixelInfo` struct: x, y, raw_hex, components (HashMap<String, u8>), neighborhood (Vec<Vec<String>>)
- `get_pixel_info(data, width, height, format, x, y, sub_grid_size) -> PixelInfo`
  - Extract raw bytes at pixel position
  - Compute component values based on format type and subsampling
  - Build neighborhood grid (3x3 or sub_grid-sized)

Add `pub mod pixel;` to `rust/src/core/mod.rs`.

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test --test pixel_test`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add rust/src/core/pixel.rs rust/src/core/mod.rs rust/tests/pixel_test.rs
git commit -m "feat: add pixel inspector with format-aware component extraction"
```

---

## Chunk 3: Minimal GUI — App Shell, Canvas, File Open

### Task 9: eframe App shell with empty window

**Files:**
- Modify: `rust/src/main.rs`
- Create: `rust/src/app.rs`

**Reference:** egui/eframe docs — `eframe::run_native`, `eframe::App` trait

- [ ] **Step 1: Implement CLI with clap**

Update `rust/src/main.rs`:
```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "video_viewer", about = "YUV/Raw Video Viewer")]
struct Cli {
    /// Input file path
    input: Option<String>,

    /// Frame width
    #[arg(short = 'W', long)]
    width: Option<u32>,

    /// Frame height
    #[arg(short = 'H', long)]
    height: Option<u32>,

    /// Pixel format (e.g., I420, NV12, YUYV)
    #[arg(short, long)]
    format: Option<String>,

    /// Input format for conversion
    #[arg(long = "vi")]
    input_format: Option<String>,

    /// Output format for conversion
    #[arg(long = "vo")]
    output_format: Option<String>,

    /// Output file (enables headless conversion)
    #[arg(short, long)]
    output: Option<String>,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    if cli.output.is_some() {
        // Headless conversion mode
        todo!("Headless conversion");
    } else {
        // GUI mode
        video_viewer::run_gui(cli.input, cli.width, cli.height, cli.format);
    }
}
```

- [ ] **Step 2: Create App struct**

Create `rust/src/app.rs`:
```rust
use eframe::egui;

pub struct VideoViewerApp {
    // State fields will be added incrementally
    pub current_file: Option<String>,
}

impl VideoViewerApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            current_file: None,
        }
    }
}

impl eframe::App for VideoViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open").clicked() {
                        // Will be implemented in next task
                    }
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(ref path) = self.current_file {
                ui.label(format!("File: {}", path));
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Drag & drop a file or use File → Open");
                });
            }
        });
    }
}
```

- [ ] **Step 3: Add run_gui to lib.rs**

Update `rust/src/lib.rs`:
```rust
pub mod core;
pub mod app;

pub fn run_gui(
    input: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    format: Option<String>,
) {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("Video Viewer"),
        ..Default::default()
    };

    eframe::run_native(
        "Video Viewer",
        options,
        Box::new(|cc| Ok(Box::new(app::VideoViewerApp::new(cc)))),
    ).unwrap();
}
```

- [ ] **Step 4: Verify build and run**

Run: `cd rust && cargo build && cargo run`
Expected: Window opens with "File" menu and placeholder text. Close with File → Quit.

- [ ] **Step 5: Commit**

```bash
git add rust/src/main.rs rust/src/app.rs rust/src/lib.rs
git commit -m "feat: add eframe app shell with CLI parsing and empty window"
```

---

### Task 10: Canvas — image display with texture

**Files:**
- Create: `rust/src/ui/mod.rs`
- Create: `rust/src/ui/canvas.rs`
- Modify: `rust/src/app.rs`

**Reference:** `video_viewer/main_window.py` — ImageCanvas class (lines ~60-200)

- [ ] **Step 1: Create canvas module**

Create `rust/src/ui/mod.rs`:
```rust
pub mod canvas;
```

Create `rust/src/ui/canvas.rs`:
- `ImageCanvas` struct: texture_handle (Option<TextureHandle>), zoom, pan_offset, image_size
- `set_image(&mut self, ctx: &egui::Context, rgb: &[u8], width: u32, height: u32)`
  - Creates `egui::ColorImage` from RGB bytes
  - Uploads as texture via `ctx.load_texture()`
- `show(&mut self, ui: &mut egui::Ui)` — renders texture with zoom/pan:
  - Mouse wheel → zoom (0.1 to 50.0)
  - Middle-click drag → pan
  - Track mouse position in image coordinates for pixel inspector
- `fit_to_view(&mut self, available_size: egui::Vec2)`
- `zoom_level() -> f32`
- `image_pos_from_screen(screen_pos) -> Option<(u32, u32)>` — for pixel inspector

- [ ] **Step 2: Wire canvas into app**

Update `rust/src/app.rs`:
- Add `canvas: ui::canvas::ImageCanvas` field
- In `update()`, show canvas in CentralPanel
- Add `pub mod ui;` to `rust/src/lib.rs`

- [ ] **Step 3: Verify build**

Run: `cd rust && cargo build`
Expected: Compiles (no visual test needed yet — canvas shows nothing without a loaded file)

- [ ] **Step 4: Commit**

```bash
git add rust/src/ui/ rust/src/app.rs rust/src/lib.rs
git commit -m "feat: add ImageCanvas with texture upload, zoom, and pan"
```

---

### Task 11: File open dialog and frame display

**Files:**
- Modify: `rust/src/app.rs`
- Modify: `rust/src/ui/canvas.rs`

**Reference:** `video_viewer/main_window.py` — `open_file()`, `_load_file()`, ParametersDialog

- [ ] **Step 1: Implement file open flow**

Update `rust/src/app.rs`:
- Add fields: `reader: Option<VideoReader>`, `current_frame_idx: usize`, `current_rgb: Option<Vec<u8>>`
- File → Open: use `rfd::FileDialog::new().pick_file()` to get path
- Detect Y4M → open directly; raw → show simple parameters input (width/height/format combo)
- On load: `reader.seek_frame(0)` → `convert_to_rgb()` → `canvas.set_image()`
- Apply filename hints if no parameters specified

- [ ] **Step 2: Add simple parameters input panel**

When opening a raw file without Y4M header:
- Show an egui::Window with width/height spinboxes and format dropdown
- Pre-fill from filename hints
- On "OK", create VideoReader and display first frame

- [ ] **Step 3: Test manually**

Run: `cd rust && cargo run -- /path/to/test.y4m`
Expected: Y4M file loads and first frame displays in canvas. Zoom with scroll, pan with middle-click.

Run: `cd rust && cargo run -- /path/to/test.yuv --width 1920 --height 1080 --format I420`
Expected: Raw file loads with specified parameters.

- [ ] **Step 4: Commit**

```bash
git add rust/src/app.rs rust/src/ui/
git commit -m "feat: add file open dialog with Y4M auto-detect and parameter input"
```

---

## Chunk 4: Navigation, Playback, Keyboard Shortcuts

### Task 12: Frame slider and navigation

**Files:**
- Create: `rust/src/ui/navigation.rs`
- Modify: `rust/src/app.rs`

**Reference:** `video_viewer/main_window.py` — MarkerSlider, navigation toolbar (lines ~470-530)

- [ ] **Step 1: Create navigation bar**

Create `rust/src/ui/navigation.rs`:
- `NavigationBar` struct: bookmarks (HashSet<usize>), scene_changes (Vec<usize>)
- `show(&mut self, ui, current_frame, total_frames) -> Option<NavigationAction>`
- `NavigationAction` enum: `Seek(usize)`, `TogglePlay`, `NextFrame`, `PrevFrame`, `FirstFrame`, `LastFrame`
- Renders:
  - Frame slider (egui::Slider, 0..total_frames-1)
  - Frame counter label ("Frame 0/100")
  - FPS combo box
  - Play/Pause button

Add `pub mod navigation;` to `rust/src/ui/mod.rs`.

- [ ] **Step 2: Wire into app**

Update `rust/src/app.rs`:
- Add navigation bar to bottom panel (`egui::TopBottomPanel::bottom`)
- Handle NavigationAction: update `current_frame_idx`, call `reader.seek_frame()`, update canvas

- [ ] **Step 3: Add keyboard shortcuts**

In `app.rs` `update()`, handle `ctx.input()` for:
- Left/Right → prev/next frame
- Home/End → first/last frame
- Space → toggle playback (placeholder for now)
- F → fit to view
- +/- → zoom in/out

- [ ] **Step 4: Test manually**

Run: `cd rust && cargo run -- test.y4m`
Expected: Slider works, arrow keys navigate frames, frame counter updates.

- [ ] **Step 5: Commit**

```bash
git add rust/src/ui/navigation.rs rust/src/ui/mod.rs rust/src/app.rs
git commit -m "feat: add frame navigation bar with slider, keyboard shortcuts"
```

---

### Task 13: Playback timer and FPS control

**Files:**
- Modify: `rust/src/app.rs`
- Modify: `rust/src/ui/navigation.rs`

**Reference:** `video_viewer/main_window.py` — `toggle_playback()`, `_on_playback_tick()`, playback timer

- [ ] **Step 1: Add playback state to app**

Update `rust/src/app.rs`:
- Add fields: `is_playing: bool`, `fps: u32`, `last_frame_time: Instant`
- In `update()`: if playing, check elapsed time vs 1000/fps ms
  - If enough time elapsed: advance frame, decode, display
  - Request repaint: `ctx.request_repaint()`
- Space key toggles `is_playing`
- FPS combo changes `fps` value

- [ ] **Step 2: Add loop playback option**

- Add `loop_playback: bool` field
- When at last frame and playing: if loop → seek to 0, else stop

- [ ] **Step 3: Test manually**

Run: `cd rust && cargo run -- test.y4m`
Expected: Space starts/stops playback at selected FPS. FPS dropdown changes speed.

- [ ] **Step 4: Commit**

```bash
git add rust/src/app.rs rust/src/ui/navigation.rs
git commit -m "feat: add playback with FPS control and loop option"
```

---

### Task 14: Prefetch decode pipeline

**Files:**
- Create: `rust/src/core/decode.rs`
- Modify: `rust/src/app.rs`

**Reference:** `video_viewer/video_reader.py` — FrameDecodeWorker, design spec section on PrefetchBuffer

- [ ] **Step 1: Implement PrefetchBuffer**

Create `rust/src/core/decode.rs`:
```rust
use crossbeam::queue::ArrayQueue;
use std::sync::Arc;

pub struct DecodedFrame {
    pub frame_idx: usize,
    pub rgb: Vec<u8>,
    pub raw: Vec<u8>,
}

pub struct PrefetchBuffer {
    queue: Arc<ArrayQueue<DecodedFrame>>,
    prefetch_count: usize,
    generation: Arc<std::sync::atomic::AtomicU64>,
}
```

- `new(prefetch_count: usize) -> Self`
- `start_prefetch(reader: Arc<Mutex<VideoReader>>, start_idx: usize, component: u8, color_matrix: &str)`
  - Spawns rayon tasks for `start_idx..start_idx+prefetch_count`
  - Each task: `reader.seek_frame()` → `convert_to_rgb()` → push to queue
- `pop() -> Option<DecodedFrame>` — non-blocking consume from queue
- `flush()` — clear queue, increment generation (stale tasks check generation)

Add `pub mod decode;` to `rust/src/core/mod.rs`.

- [ ] **Step 2: Integrate with playback**

Update `rust/src/app.rs`:
- During playback: consume from prefetch buffer instead of synchronous decode
- On seek/stop: flush buffer
- On component/color_matrix change: flush buffer

- [ ] **Step 3: Test manually with large file**

Run: `cd rust && cargo run -- large_4k.yuv --width 3840 --height 2160 --format I420`
Expected: Smooth playback at 30fps without frame drops (vs Python version).

- [ ] **Step 4: Commit**

```bash
git add rust/src/core/decode.rs rust/src/core/mod.rs rust/src/app.rs
git commit -m "feat: add rayon prefetch decode pipeline with lock-free queue"
```

---

## Chunk 5: Channel Views, Grid, Toolbar

### Task 15: Component view (channel selection + split view)

**Files:**
- Modify: `rust/src/app.rs`
- Create: `rust/src/ui/toolbar.rs`

**Reference:** `video_viewer/main_window.py` — `set_component()`, `_colorize_channel()`, split view in `update_frame()` (lines ~1896-1960)

- [ ] **Step 1: Create toolbar with component buttons**

Create `rust/src/ui/toolbar.rs`:
- `Toolbar` struct: current_component (0-4), grid_size, sub_grid_size
- `show(&mut self, ui) -> Option<ToolbarAction>`
- `ToolbarAction` enum: `SetComponent(u8)`, `ToggleGrid`, `ToggleSubGrid`, `FitToView`, `Zoom(f32)`, etc.
- Component buttons: Full(0), Y/R(1), U/G(2), V/B(3), Split(4)
- Labels adapt to format type (YUV → Y/U/V, RGB/Bayer/Grey → R/G/B)

Add `pub mod toolbar;` to `rust/src/ui/mod.rs`.

- [ ] **Step 2: Implement colorize_channel**

Add to `rust/src/app.rs` or a helper:
- `colorize_channel(gray: &[u8], w: u32, h: u32, name: &str) -> Vec<u8>`
  - Y → green tint, U → blue tint, V → red tint
  - R → red channel only, G → green only, B → blue only

- [ ] **Step 3: Implement component rendering in frame update**

In the frame display path:
- Component 0: full RGB (existing)
- Component 1-3: `get_channels()` → select channel → `colorize_channel()` → display
- Component 4: composite 2x2 (full + 3 channels, each resized to half)

- [ ] **Step 4: Add keyboard shortcuts**

Keys 0-4 set component mode. Wire into app `update()`.

- [ ] **Step 5: Test manually**

Run: `cd rust && cargo run -- test.y4m`
Expected: Press 1/2/3 to see individual channels with false color. Press 4 for split view.

- [ ] **Step 6: Commit**

```bash
git add rust/src/ui/toolbar.rs rust/src/ui/mod.rs rust/src/app.rs
git commit -m "feat: add component view with channel selection and 2x2 split view"
```

---

### Task 16: Grid and sub-grid overlay

**Files:**
- Modify: `rust/src/ui/canvas.rs`

**Reference:** `video_viewer/main_window.py` — grid rendering in ImageCanvas.paintEvent() (lines ~130-180)

- [ ] **Step 1: Add grid rendering to canvas**

Update `rust/src/ui/canvas.rs`:
- `set_grid_size(&mut self, size: u32)` — 0 = off, 16/32/64/128
- `set_sub_grid_size(&mut self, size: u32)` — 0 = off, 4/8/16
- In `show()`, after rendering texture, use `ui.painter()` to draw:
  - Main grid: green dotted lines at grid_size intervals (in image coordinates, scaled by zoom)
  - Sub-grid: yellow dotted lines at sub_grid_size intervals
- G key cycles grid sizes [0, 16, 32, 64, 128]
- Shift+G cycles sub-grid sizes [0, 4, 8, 16]

- [ ] **Step 2: Test manually**

Run: `cd rust && cargo run -- test.y4m`
Expected: G toggles grid overlay. Shift+G toggles sub-grid. Grid lines scale with zoom.

- [ ] **Step 3: Commit**

```bash
git add rust/src/ui/canvas.rs
git commit -m "feat: add grid and sub-grid overlay with keyboard toggle"
```

---

### Task 17: Menu bar

**Files:**
- Create: `rust/src/ui/menu.rs`
- Modify: `rust/src/app.rs`

**Reference:** `video_viewer/main_window.py` — menu setup (lines ~430-700)

- [ ] **Step 1: Create full menu bar**

Create `rust/src/ui/menu.rs`:
- `MenuAction` enum covering all menu actions
- `show_menu_bar(ui, state) -> Option<MenuAction>`
- Menus: File (Open, Save Frame, Export Clip, Export PNG Sequence, Recent Files, Settings, Quit), View (Zoom submenu, Grid submenu, Component submenu, Show Analysis, Bookmarks, Dark Theme), Tools (Analysis, Convert, Compare, Batch Convert, Scene Detection, Save/Load Scene List, Video Parameters, Color Matrix), Help (Shortcuts, About)

Add `pub mod menu;` to `rust/src/ui/mod.rs`.

- [ ] **Step 2: Wire menu actions to app**

Update `rust/src/app.rs` to handle each `MenuAction`.

- [ ] **Step 3: Test manually**

Run: `cd rust && cargo run`
Expected: Full menu bar visible with all entries. Actions dispatch correctly.

- [ ] **Step 4: Commit**

```bash
git add rust/src/ui/menu.rs rust/src/ui/mod.rs rust/src/app.rs
git commit -m "feat: add full menu bar with File, View, Tools, Help menus"
```

---

## Chunk 6: Sidebar — Pixel Inspector & Analysis

### Task 18: Pixel inspector sidebar

**Files:**
- Create: `rust/src/ui/sidebar.rs`
- Modify: `rust/src/app.rs`

**Reference:** `video_viewer/main_window.py` — pixel inspector panel (lines ~820-900)

- [ ] **Step 1: Create sidebar panel**

Create `rust/src/ui/sidebar.rs`:
- `Sidebar` struct: pixel_info (Option<PixelInfo>), active_tab (Histogram/Waveform/Vectorscope/Metrics)
- `show(&mut self, ui)` — right side panel:
  - Pixel Inspector section: coordinates, raw hex, component values, neighborhood grid
  - Tabbed analysis section (placeholder for now)
- Update on canvas mouse move: `canvas.hovered_image_pos()` → `get_pixel_info()`

Add `pub mod sidebar;` to `rust/src/ui/mod.rs`.

- [ ] **Step 2: Wire pixel tracking**

Update `rust/src/app.rs`:
- On each frame, if canvas has hovered position, compute pixel info
- Pass to sidebar for display

- [ ] **Step 3: Test manually**

Run: `cd rust && cargo run -- test.y4m`
Expected: Hover over image shows pixel coordinates, hex values, and Y/U/V components in sidebar.

- [ ] **Step 4: Commit**

```bash
git add rust/src/ui/sidebar.rs rust/src/ui/mod.rs rust/src/app.rs
git commit -m "feat: add pixel inspector sidebar with format-aware component display"
```

---

### Task 19: Histogram analysis

**Files:**
- Create: `rust/src/analysis/mod.rs`
- Create: `rust/src/analysis/histogram.rs`
- Modify: `rust/src/ui/sidebar.rs`

**Reference:** `video_viewer/analysis.py` — `calculate_histogram()` (lines ~15-40)

- [ ] **Step 1: Write failing test**

Create `rust/tests/histogram_test.rs` (add to `rust/tests/`):
```rust
use video_viewer::analysis::histogram::calculate_histogram;

#[test]
fn test_histogram_uniform() {
    // All pixels = 128 → histogram peak at bin 128
    let rgb = vec![128u8; 100 * 100 * 3];
    let hist = calculate_histogram(&rgb, 100, 100, "RGB");
    assert_eq!(hist["R"][128], 10000); // 100x100 pixels
    assert_eq!(hist["R"][0], 0);
}

#[test]
fn test_histogram_y_mode() {
    let rgb = vec![128u8; 10 * 10 * 3];
    let hist = calculate_histogram(&rgb, 10, 10, "Y");
    assert!(hist.contains_key("Y"));
    assert_eq!(hist["Y"].len(), 256);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd rust && cargo test --test histogram_test`
Expected: FAIL

- [ ] **Step 3: Implement histogram calculation**

Create `rust/src/analysis/mod.rs` and `rust/src/analysis/histogram.rs`:
- `calculate_histogram(rgb: &[u8], w: u32, h: u32, mode: &str) -> HashMap<String, Vec<u32>>`
- Mode "RGB": 3 channels × 256 bins
- Mode "Y": convert to luma, 1 channel × 256 bins

Add `pub mod analysis;` to `rust/src/lib.rs`.

- [ ] **Step 4: Render histogram in sidebar**

Update `rust/src/ui/sidebar.rs`:
- In histogram tab, use `egui_plot::Plot` to draw histogram bars/lines
- R=red, G=green, B=blue, Y=white

- [ ] **Step 5: Run tests**

Run: `cd rust && cargo test --test histogram_test`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add rust/src/analysis/ rust/src/lib.rs rust/src/ui/sidebar.rs rust/tests/histogram_test.rs
git commit -m "feat: add histogram analysis with RGB/Y modes and egui_plot rendering"
```

---

### Task 20: Waveform monitor

**Files:**
- Create: `rust/src/analysis/waveform.rs`
- Modify: `rust/src/ui/sidebar.rs`

**Reference:** `video_viewer/analysis.py` — `calculate_waveform()` (lines ~80-120)

- [ ] **Step 1: Implement waveform calculation**

Create `rust/src/analysis/waveform.rs`:
- `calculate_waveform(rgb: &[u8], w: u32, h: u32, channel: &str) -> Vec<Vec<u32>>`
  - Returns 256 × width 2D histogram
  - Channel: "luma", "r", "g", "b"
  - Downsample width to max 720 for performance
  - BT.709 luma: 0.2126*R + 0.7152*G + 0.0722*B
- Render as custom egui texture (grayscale intensity image)

Add `pub mod waveform;` to `rust/src/analysis/mod.rs`.

- [ ] **Step 2: Wire to sidebar tab**

- [ ] **Step 3: Commit**

```bash
git add rust/src/analysis/waveform.rs rust/src/analysis/mod.rs rust/src/ui/sidebar.rs
git commit -m "feat: add waveform monitor with luma and per-channel modes"
```

---

### Task 21: Vectorscope

**Files:**
- Create: `rust/src/analysis/vectorscope.rs`
- Modify: `rust/src/ui/sidebar.rs`

**Reference:** `video_viewer/analysis.py` — `calculate_vectorscope_from_rgb()` (lines ~40-60)

- [ ] **Step 1: Implement vectorscope**

Create `rust/src/analysis/vectorscope.rs`:
- `calculate_vectorscope(rgb: &[u8], w: u32, h: u32) -> (Vec<f32>, Vec<f32>)`
  - Convert RGB to YCrCb
  - Return (Cb_values, Cr_values) for scatter plot
  - Downsample if width > 640
- Render as `egui_plot::Plot` scatter points

Add `pub mod vectorscope;` to `rust/src/analysis/mod.rs`.

- [ ] **Step 2: Wire to sidebar tab**

- [ ] **Step 3: Commit**

```bash
git add rust/src/analysis/vectorscope.rs rust/src/analysis/mod.rs rust/src/ui/sidebar.rs
git commit -m "feat: add vectorscope (Cb vs Cr scatter plot)"
```

---

### Task 22: Metrics — PSNR and SSIM

**Files:**
- Create: `rust/src/analysis/metrics.rs`
- Create: `rust/tests/metrics_test.rs`

**Reference:** `video_viewer/analysis.py` — `calculate_psnr()`, `calculate_ssim()`, `calculate_frame_difference()`, `calculate_histogram_difference()`

- [ ] **Step 1: Write failing tests**

Create `rust/tests/metrics_test.rs`:
```rust
use video_viewer::analysis::metrics::*;

#[test]
fn test_psnr_identical() {
    let img = vec![128u8; 100 * 100 * 3];
    let psnr = calculate_psnr(&img, &img, 100, 100);
    assert!(psnr.is_infinite() || psnr > 100.0);
}

#[test]
fn test_psnr_different() {
    let img1 = vec![100u8; 100 * 100 * 3];
    let img2 = vec![200u8; 100 * 100 * 3];
    let psnr = calculate_psnr(&img1, &img2, 100, 100);
    assert!(psnr > 0.0 && psnr < 100.0);
}

#[test]
fn test_ssim_identical() {
    let img = vec![128u8; 100 * 100 * 3];
    let ssim = calculate_ssim(&img, &img, 100, 100);
    assert!((ssim - 1.0).abs() < 0.01);
}

#[test]
fn test_frame_difference() {
    let img1 = vec![100u8; 10 * 10 * 3];
    let img2 = vec![150u8; 10 * 10 * 3];
    let diff = calculate_frame_difference(&img1, &img2);
    assert!((diff - 50.0).abs() < 1.0);
}
```

- [ ] **Step 2: Run test to verify fails**

Run: `cd rust && cargo test --test metrics_test`
Expected: FAIL

- [ ] **Step 3: Implement metrics**

Create `rust/src/analysis/metrics.rs`:
- `calculate_psnr(img1, img2, w, h) -> f64` — via opencv `core::psnr` or manual MSE
- `calculate_ssim(img1, img2, w, h) -> f64` — manual Wang et al. 2004:
  - Gaussian 11×11 window, C1=(0.01×255)², C2=(0.03×255)²
  - SSIM = (2μxμy + C1)(2σxy + C2) / (μx² + μy² + C1)(σx² + σy² + C2)
- `calculate_frame_difference(img1, img2) -> f64` — mean absolute diff
- `calculate_histogram_difference(img1, img2) -> f64` — histogram correlation

Add `pub mod metrics;` to `rust/src/analysis/mod.rs`.

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test --test metrics_test`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add rust/src/analysis/metrics.rs rust/src/analysis/mod.rs rust/tests/metrics_test.rs
git commit -m "feat: add PSNR, SSIM, frame diff, and histogram correlation metrics"
```

---

## Chunk 7: Comparison View & Conversion

### Task 23: A/B comparison window

**Files:**
- Create: `rust/src/ui/comparison.rs`
- Modify: `rust/src/app.rs`

**Reference:** `video_viewer/comparison_view.py` — ComparisonWindow, ComparisonCanvas, ComparisonMode

- [ ] **Step 1: Implement comparison view**

Create `rust/src/ui/comparison.rs`:
- `ComparisonMode` enum: Split, Overlay, Diff
- `ComparisonView` struct: ref_reader, mode, split_pos, overlay_opacity, ref_rgb, diff_rgb
- `show(&mut self, ctx, ui)`:
  - Mode selector (Split/Overlay/Diff)
  - Split: render two halves side by side with draggable divider
  - Overlay: alpha blend with opacity slider (0-100%)
  - Diff: heatmap (amplified 10×, R=V_diff, G=Y_diff, B=U_diff)
- `load_reference(path, format_name)` — opens second VideoReader
- `update_frame(main_raw, ref_raw)` — compute comparison
- PSNR/SSIM display in toolbar area
- Pixel inspector shows both source and reference values

Add `pub mod comparison;` to `rust/src/ui/mod.rs`.

- [ ] **Step 2: Wire to menu and app**

Tools → Compare opens comparison panel/window.

- [ ] **Step 3: Test manually**

Open two files, compare in split/overlay/diff modes.

- [ ] **Step 4: Commit**

```bash
git add rust/src/ui/comparison.rs rust/src/ui/mod.rs rust/src/app.rs
git commit -m "feat: add A/B comparison view with split, overlay, and diff modes"
```

---

### Task 24: Format conversion engine

**Files:**
- Create: `rust/src/conversion/mod.rs`
- Create: `rust/src/conversion/converter.rs`
- Create: `rust/tests/converter_test.rs`

**Reference:** `video_viewer/video_converter.py` — VideoConverter (all of it)

- [ ] **Step 1: Write failing tests**

Create `rust/tests/converter_test.rs`:
```rust
use video_viewer::conversion::converter::VideoConverter;
use std::io::Write;

#[test]
fn test_convert_i420_to_nv12() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.yuv");
    let output = dir.path().join("output.yuv");

    // 4x4 I420: 16Y + 4U + 4V
    let mut data = vec![128u8; 16];
    data.extend(vec![100u8; 4]); // U
    data.extend(vec![150u8; 4]); // V
    std::fs::write(&input, &data).unwrap();

    let converter = VideoConverter::new();
    let (count, cancelled) = converter.convert(
        input.to_str().unwrap(), 4, 4, "I420",
        output.to_str().unwrap(), "NV12", None,
    ).unwrap();

    assert_eq!(count, 1);
    assert!(!cancelled);

    let result = std::fs::read(&output).unwrap();
    assert_eq!(result.len(), 24); // NV12 same size as I420 for 4:2:0
    // Y plane should be preserved
    assert_eq!(&result[..16], &[128u8; 16]);
    // UV interleaved: U0,V0,U1,V1,...
    assert_eq!(result[16], 100); // U0
    assert_eq!(result[17], 150); // V0
}

#[test]
fn test_convert_same_format() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.yuv");
    let output = dir.path().join("output.yuv");

    let data = vec![42u8; 24];
    std::fs::write(&input, &data).unwrap();

    let converter = VideoConverter::new();
    let (count, _) = converter.convert(
        input.to_str().unwrap(), 4, 4, "I420",
        output.to_str().unwrap(), "I420", None,
    ).unwrap();

    assert_eq!(count, 1);
    assert_eq!(std::fs::read(&output).unwrap(), data);
}
```

- [ ] **Step 2: Run test to verify fails**

Run: `cd rust && cargo test --test converter_test`
Expected: FAIL

- [ ] **Step 3: Implement converter**

Create `rust/src/conversion/converter.rs`:
- `VideoConverter` struct
- `convert(input_path, w, h, input_fmt, output_path, output_fmt, callback) -> Result<(usize, bool)>`
  - Same format → direct copy
  - Both YUV → direct convert (extract planes → resample chroma → repack)
  - Otherwise → RGB intermediate
- `extract_yuv_planes(raw, w, h, fmt) -> (Vec<u8>, Vec<u8>, Vec<u8>)`
- `resample_chroma(u, v, src_sub, dst_sub) -> (Vec<u8>, Vec<u8>)`
- `pack_yuv(y, u, v, fmt, w, h) -> Vec<u8>`

Create `rust/src/conversion/mod.rs`:
```rust
pub mod converter;
```

Add `pub mod conversion;` to `rust/src/lib.rs`.

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test --test converter_test`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add rust/src/conversion/ rust/src/lib.rs rust/tests/converter_test.rs
git commit -m "feat: add format conversion engine with direct YUV and RGB intermediate paths"
```

---

### Task 25: Headless conversion CLI

**Files:**
- Modify: `rust/src/main.rs`

**Reference:** `video_viewer/main.py` — headless conversion mode

- [ ] **Step 1: Implement headless mode**

Update `rust/src/main.rs`:
- When `--output` is provided: run `VideoConverter::convert()` with progress to stderr
- Print frame count on completion

- [ ] **Step 2: Test manually**

```bash
cd rust && cargo run -- input.yuv -W 1920 -H 1080 --vi I420 --vo NV12 -o output.nv12
```
Expected: Converts file, prints progress, outputs to specified path.

- [ ] **Step 3: Commit**

```bash
git add rust/src/main.rs
git commit -m "feat: add headless format conversion CLI mode"
```

---

## Chunk 8: Dialogs, Settings, Polish

### Task 26: Dialogs

**Files:**
- Create: `rust/src/ui/dialogs.rs`
- Modify: `rust/src/app.rs`

**Reference:** `video_viewer/dialogs.py` — all dialog classes

- [ ] **Step 1: Implement dialog system**

Create `rust/src/ui/dialogs.rs`:
- `DialogState` enum tracking which dialog is open
- `ParametersDialog`: width/height spinbox, format dropdown, OK/Cancel
- `ExportDialog`: start/end frame, export format, file path
- `ConvertDialog`: input info (read-only), output format, output path
- `BatchConvertDialog`: file list, width/height, input/output format, output dir
- `PngExportDialog`: start/end frame, output dir, prefix
- `SettingsDialog`: cache memory, zoom bounds, defaults, dark theme
- `ShortcutsDialog`: table of all keyboard shortcuts
- `BookmarkDialog`: list of bookmarks with jump/delete
- `AboutDialog`: version, copyright

All rendered as `egui::Window` modals.

Add `pub mod dialogs;` to `rust/src/ui/mod.rs`.

- [ ] **Step 2: Wire dialogs to menu actions**

- [ ] **Step 3: Commit**

```bash
git add rust/src/ui/dialogs.rs rust/src/ui/mod.rs rust/src/app.rs
git commit -m "feat: add all dialog windows (parameters, export, convert, settings, etc.)"
```

---

### Task 27: Settings persistence

**Files:**
- Create: `rust/src/ui/settings.rs`
- Modify: `rust/src/app.rs`

**Reference:** `video_viewer/main_window.py` — QSettings usage, `video_viewer/constants.py`

- [ ] **Step 1: Implement settings**

Create `rust/src/ui/settings.rs`:
```rust
#[derive(Serialize, Deserialize)]
pub struct Settings {
    pub cache: CacheSettings,
    pub display: DisplaySettings,
    pub defaults: DefaultSettings,
}
```
- `CacheSettings`: max_memory_mb (default 512)
- `DisplaySettings`: zoom_min (0.1), zoom_max (50.0), dark_theme (true)
- `DefaultSettings`: fps (30), color_matrix ("BT.601"), width (1920), height (1080)
- `Settings::load() -> Self` — read from `~/.config/video-viewer/settings.toml`, fallback to defaults
- `Settings::save(&self)` — write TOML

Add `pub mod settings;` to `rust/src/ui/mod.rs`.

- [ ] **Step 2: Wire settings to app**

Load on startup, apply values, save on settings dialog OK.

- [ ] **Step 3: Commit**

```bash
git add rust/src/ui/settings.rs rust/src/ui/mod.rs rust/src/app.rs
git commit -m "feat: add TOML settings persistence"
```

---

### Task 28: Bookmarks and scene detection

**Files:**
- Create: `rust/src/analysis/scene.rs`
- Modify: `rust/src/app.rs`
- Modify: `rust/src/ui/navigation.rs`

**Reference:** `video_viewer/main_window.py` — bookmarks (lines ~750-830), scene detection (lines ~2680-2730)

- [ ] **Step 1: Write failing test for scene detection**

Create `rust/tests/scene_test.rs`:
```rust
use video_viewer::analysis::scene::detect_scene_changes;

#[test]
fn test_scene_change_identical() {
    let frame = vec![128u8; 10 * 10 * 3];
    let frames = vec![frame.clone(), frame.clone(), frame.clone()];
    let changes = detect_scene_changes(&frames, 10, 10, 45.0);
    assert!(changes.is_empty());
}

#[test]
fn test_scene_change_detected() {
    let frame_a = vec![50u8; 10 * 10 * 3];
    let frame_b = vec![200u8; 10 * 10 * 3];
    let frames = vec![frame_a.clone(), frame_a.clone(), frame_b.clone()];
    let changes = detect_scene_changes(&frames, 10, 10, 45.0);
    assert!(changes.contains(&2));
}
```

- [ ] **Step 2: Implement scene detection**

Create `rust/src/analysis/scene.rs`:
- `detect_scene_changes(frames, w, h, threshold) -> Vec<usize>` — frame indices where scene changes
- `save_scene_list(path, indices)` — write to text file, one index per line
- `load_scene_list(path) -> Vec<usize>` — read from text file

Add `pub mod scene;` to `rust/src/analysis/mod.rs`.

- [ ] **Step 3: Add bookmarks to app state**

In `app.rs`:
- `bookmarks: HashSet<usize>`
- B key toggles bookmark at current frame
- Ctrl+B / Ctrl+Shift+B navigate between bookmarks
- Ctrl+Left/Right navigate between scene changes
- Bookmark markers (cyan) and scene markers (red) on navigation slider

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test --test scene_test`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add rust/src/analysis/scene.rs rust/src/analysis/mod.rs rust/src/app.rs rust/src/ui/navigation.rs rust/tests/scene_test.rs
git commit -m "feat: add bookmarks, scene detection, and scene list save/load"
```

---

### Task 29: Frame export (PNG/BMP, clip, PNG sequence)

**Files:**
- Modify: `rust/src/app.rs`

**Reference:** `video_viewer/main_window.py` — save_frame, export_clip, export_png_sequence

- [ ] **Step 1: Implement frame export**

Add to `rust/src/app.rs`:
- Save Frame (Ctrl+S): use `image` crate to save current RGB as PNG
- Export Clip: iterate frame range, write raw data to output file
- Export PNG Sequence: iterate frame range, save each as numbered PNG
- Copy to Clipboard (Ctrl+C): use `arboard` to copy current frame as image

- [ ] **Step 2: Test manually**

Save a frame as PNG, verify it opens correctly.

- [ ] **Step 3: Commit**

```bash
git add rust/src/app.rs
git commit -m "feat: add frame export (PNG, clip, PNG sequence, clipboard)"
```

---

### Task 30: Dark/light theme and final polish

**Files:**
- Modify: `rust/src/app.rs`

- [ ] **Step 1: Add theme toggle**

- egui has built-in dark/light theme: `ctx.set_visuals(egui::Visuals::dark())` or `light()`
- View → Dark Theme toggle
- Persist in settings

- [ ] **Step 2: Add status bar**

`egui::TopBottomPanel::bottom("status")`:
- File info: format, resolution, total frames
- Current frame number
- Grid/sub-grid sizes
- Zoom level

- [ ] **Step 3: Add drag & drop support**

In `update()`, handle `ctx.input(|i| i.raw.dropped_files)`:
- Accept .yuv, .y4m, .rgb, .raw files
- Auto-detect or prompt for parameters

- [ ] **Step 4: Add recent files**

Store last 10 opened files in settings. Show in File → Recent Files submenu.

- [ ] **Step 5: Final build test**

Run: `cd rust && cargo build --release`
Expected: Release build succeeds.

Run: `cd rust && cargo test`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add rust/
git commit -m "feat: add theme toggle, status bar, drag-drop, recent files"
```

---

## Summary

| Chunk | Tasks | Focus |
|-------|-------|-------|
| 1 | 1-5 | Project scaffold, formats, hints, Y4M, cache |
| 2 | 6-8 | VideoReader, convert_to_rgb, pixel inspector |
| 3 | 9-11 | GUI shell, canvas, file open |
| 4 | 12-14 | Navigation, playback, prefetch pipeline |
| 5 | 15-17 | Channel views, grid, menu bar |
| 6 | 18-22 | Sidebar, histogram, waveform, vectorscope, metrics |
| 7 | 23-25 | Comparison view, converter, headless CLI |
| 8 | 26-30 | Dialogs, settings, bookmarks, scenes, export, polish |

Total: **30 tasks**, ~8 chunks, building from core → GUI → features → polish.
