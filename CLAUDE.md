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
    - `bmp.rs` - M-C minimal BMP reader for decoder-run stage images (self-contained, public-spec BITMAPINFOHEADER family: uncompressed 24/32-bit BGR(A), bottom-up/top-down, 4-byte row padding, bounds-checked — no panics; observed stage dumps are 24-bit bottom-up)
    - `decoder_run.rs` - M5 external decoder-run launcher (arm's-length: user-configured command template with `{input}`/`{workdir}`/`{telemetry}`/`{yuv}` placeholders, shell quoting per platform, `run_shell_command` with job-id gate + killable `Child` slot, `derive_workdir` = sibling `<name>.catb-run/`, `plan_yuv_open` I420 frame-count validation; 0.16.0 zero-config auto-detect `detect_decoder_command`/`probe_paths` — probe order: ① `<exe_dir>/codec-analyzer/codec-analyzer.cmd|.sh` bundle → ② unix `~/work/codec-analyzer` dev checkout (`.venv/bin/python` + `.local/bin/ffmpeg`, pinned via `CODEC_ANALYZER_FFMPEG`) → ③ `codec-analyzer` on PATH; empty Settings command falls back to the detected one; M-A 0.17.0: default template level block→**syntax**, `TelemetryLevel` {Block/Syntax/Full} with `set_telemetry_level` token rewrite — `--telemetry-level <v>` and `=<v>` forms, **token-only** (no token → `None`, template runs untouched))
    - `catb.rs` - `.catb` v4 bitstream telemetry reader (mmap, string table, FRAME/BLOCK/REF records; M4: lazy per-block SYNTAX records `syntax_for_block` with OOM clamp guard; M-A 0.17.0: structured meta parsing at open — `ParameterSetInfo` (kind/id/nal_index + sorted scalar fields), `CatbFrameMeta` per decode frame (`SliceHeaderInfo` with fields/syntax/ref_list0/1, `DpbRow` poc/slot/used_for_reference/long_term/output_mark/label, `exactness_missing`/`block_dropped_rows` parallel to BLOCK records) — all lenient, missing/mistyped → empty defaults; M-B 0.18.0: lazy per-block TRANSFORM (`tx_for_block` → `TxRow` with §9.1 presence bits — nonzero/last_sig_x/y/abs_sum `Option`, cbf luma/cb/cr helpers, has_detail/detail_dropped) and CABAC (`cabac_for_block` → `CabacRow` name/ctx/bin/bit_offset/bits) records, §11 indirect coefficient detail `coeffs_for_tx` (levels+scan+group_flags i32 sub-arrays, span bounds-checked before alloc), lightweight `tx_aggregate_for_block` (Σabs_sum, Σnonzero — no string resolve); all with the OOM clamp guard; slice-header syntax rows now carry bits: `(name, value, bits)`; M-C 0.19.0: `CatbFrameMeta.stage_images` — sorted (key, path) pairs of `frames_meta[i].stage_images`, observed keys residual/prediction/recon_unfiltered/final_recon with .catb-relative file names; M-E 0.21.0: lazy per-frame `frame_aux_for_frame` (§12 `{loop_filters, sao}` compact-JSON blob at FRAME `(aux_off, aux_n)`; aux_n≤0 = empty, range bounds-checked, truncated/invalid JSON = Err never panic, mistyped rows decay) → `LoopFilterRow` {x/y/w/h, vertical, boundary_strength, filter_strength strong/weak/none, reason, qp} + `SaoRow` (+`applied()`; fixtures all not_applied — parsed, no overlay yet))
    - `nal.rs` - M-E 0.21.0 Annex-B NAL scanner for raw .h264/.h265 elementary streams (standards-knowledge-only, `.catb`-independent, mmap single O(n) pass, no panics): `NalScan::open` / `scan_annexb` → `NalUnit` {offset, start_code_len(3/4), size(header+payload), nal_type, HEVC temporal_id/layer_id, AVC nal_ref_idc, forbidden, is_vcl, au_start, au_id}; AU estimation — HEVC `first_slice_segment_in_pic_flag` bit, AVC `first_mb_in_slice==0` leading-'1' heuristic, non-VCL AUD/param-set/prefix-SEI attributed to the *next* AU; codec detection `detect_codec` = header-violation scoring (forbidden bit / reserved types / HEVC tid_plus1==0) with extension hint tiebreak; Table 7-1 type names; 2M-unit cap (`truncated` flag); `infer_bitstream_source` — `<name>.catb-run/` workdir parent first, same-stem sibling ext second
    - `dropped.rs` - Drag&drop file classification (`classify_dropped_file` → `DroppedKind::{Telemetry,Bitstream,Other}`): CATB0001 magic (extension-independent) → `.catb` ext → `.h265/.hevc/.h264/.264/.265` ext → Annex-B start-code sniff in first 32 bytes (extensionless/`.bin` only, known video extensions never sniffed); read failure = Other so the normal open path surfaces the error
  - `src/ui/` - UI components
    - `canvas.rs` - ImageCanvas (rendering, zoom, grid overlay)
    - `toolbar.rs` - Toolbar (component selection, grid controls, colorize_channel)
    - `sidebar.rs` - Sidebar (analysis tabs: histogram, waveform, vectorscope, metrics, block, motion)
    - `navigation.rs` - NavigationBar (frame slider, playback controls)
    - `dialogs.rs` - Open, Save, Parameters, Export, Convert, Settings dialogs; M5 `DecoderRunDialog` (Tools ▸ Open Bitstream…; 0.16.0: "Auto-detected: <source>" label when the command comes from auto-detection, Run disabled + guidance only when neither configured nor detected; Settings shows a "Detected (<source>)" hint + [Use detected] fill button while the command field is empty; M-A 0.17.0: telemetry-level combo Fast(block)/Standard(syntax, default)/Deep(full) with per-item feature tooltips, disabled (+ guidance tooltip) when the effective template has no `--telemetry-level` token; M-E 0.21.0: `LocateBitstreamDialog` + `DialogState::LocateBitstream` — Structure-tab "Locate bitstream…" picker for the NAL browser source)
    - `comparison.rs` - ComparisonView (three-pane video diff, spatial metric labels, synchronized zoom/pan)
    - `settings.rs` - Settings persistence (toml, incl. `[bitstream]` view state with serde-default backward compat; M5 `[decoder]` `run_command` template — kept out of `[bitstream]` because that struct is rebuilt wholesale from the window's ViewConfig; 0.16.0: empty `run_command` = use auto-detected codec-analyzer, cached per session in `app.rs` `detected_decoder` and invalidated on Settings save — `effective_decoder_command`)
    - `sideband_overlay.rs` - Sideband CTU heatmap overlay rendering
    - `bitstream_window.rs` - Bitstream Analysis separate OS window (immediate viewport, `BitstreamShared`, fill heatmaps QP/bpp/Mode/MV-heat with LOD, presets, Inspector dock, filmstrip, Frame Graph tab, §8 shortcuts; M2 Correlation tab: X/Y/G combos with preset pairs, scatter plot, r/ρ/N/valid% readout, class table, CSV export, current-frame vs background range-scan modes; M3: Opportunity fill on G cells (diverging blue-white-red, symmetric ±max|z| legend, cell click → Inspector a/b/z), Opportunity preset + key 6, Top-N ranking table with [Jump] (Viewer tab + pan-to-cell + Sel highlight), scatter↔canvas bidirectional cell highlight, per-frame r timeline with current-frame VLine + scene-change markers + click-to-seek; M4: MV (V, per-PU arrows L0 orange/L1 purple, ref idx>0 dashed, MV/MVP/MVD source combo, zoom≥1.5 + block≥12px + 4000-arrow cap), Part (P, CU grey + per-PU magenta outlines), Intra (D, direction lines + P/DC badges at zoom≥2) layers; Motion preset = MV on, Mode preset = Part+Intra on; 0.15.0: drag&drop onto the window — child classifies and parks `drop_request` in `BitstreamShared`, root polls and routes it like a main-window drop; hover hint overlay; M-A 0.17.0: 4th "Structure" tab — one scrollable CollapsingHeader column (Parameter Sets with field-name filter TextEdit, Slice Headers of the current frame with full field grid + decoded syntax + ref_list0/1 tables, DPB table slot/POC/flags/label with active-ref rows in orange, Exactness frame badge) over the shared transport+filmstrip; filmstrip reference view (transport-bar "Refs" toggle, default on): current frame's ref arcs above the cells (L0 orange / L1 purple / long-term green quadratic beziers with chevron heads), red reference-usage dots in 3 size tiers, yellow top markers on exactness-problem frames — all from `FilmstripCache`-embedded `FilmstripRefs`; Inspector "exact" row for the selected block; M-B 0.18.0: TU layer (T, yellow depth-shaded outlines, cbf=0 dashed, transform_skip shaded, Coeffs:n at zoom≥2, zoom≥1.5 LOD), CoeffEnergy (key 7, orange→red, frame-max normalized abs_sum/px) + NonzeroCoeffs (key 8, magenta density) fills via the shared `FillSample` sample struct (block+grid constructors), Inspector TU table (size/depth/cbf/nz/abs/bits; row click → coefficient grid with scan-order indices, group flags line, detail-dropped guidance), 5th "Stats" tab (frame summary line, name-filtered syntax aggregate table with (slice) rows + % of data_bits, CABAC bins/bits-per-bin/ctx table with Deep-telemetry guidance, MV scatter L0 orange/L1 purple with symmetric axes + nearest-point block tooltip, cached per (file, display frame)); M-C 0.19.0: Viewer-tab "Pic" combo (VQA Pic analogue) — Source / Final recon / Recon unfiltered / Prediction / Residual background texture switch (per-frame availability disables items with re-run guidance tooltip; stage texture keyed (file, decode idx, kind); session-only, resets to Source on file change; **window-only** — the main-canvas overlay mirror keeps the viewer frame because the root pipeline reads `current_rgb`), BlockPSNR (key 9) / BlockSSIM (combo-only) fills — per-L0-block final_recon-vs-source quality rendered in a dedicated pass (worst block most opaque red, bit-exact = transparent + 'e' label/legend "(e=exact)"), quality legend shows the unmet precondition (no frame / no stage image / resolution mismatch) in the legend slot, per-frame quality cached in `FrameDerived` keyed on the image generation, `Snapshot.frame_pixels` always-Arc-cloned viewer pixels; M-D 0.20.0: Δ mode (X key / toolbar Δ toggle, visible only with a B loaded, session-only, outside ViewConfig): fill renders the A−B cell difference on the fixed comparison grids (L1 8px / LOD, Opportunity paint-pass convention — CU partitions differ between encodes so rects can't pair) — scalar fills diverging blue-white-red on ±max|Δ| (reused `opportunity_cell_color`/legend), Mode as grey=same / A's-colour=different (VQA Δ convention), tooltip + Δ loupe show A/B/Δ, missing-B-frame reason in the legend slot; Inspector "B (same position)" block via centre hit test with differing values highlighted; Frame Graph B series dashed (bits/avgQP, own background scan); `BitstreamShared.file_b`/`offset_b`/`frame_graph_b`; `reset_b_state` on B unload/replace and inside `reset_file_state`; M-E 0.21.0: Structure-tab **NAL browser** — `BitstreamShared.bitstream_source`/`nal_scan`/`nal_scanning`/`nal_error` + `locate_bitstream_request` (root-owned dialog, drop_request pattern), source auto-inferred on .catb load (`infer_bitstream_source`) with background scan thread (job-id guard), "NAL Units" CollapsingHeader: All/VCL/non-VCL filter + type-name search (`NalFilteredCache` keyed on scan ptr+filter+search), `show_rows` virtual table (#, hex offset, size, type name, TId/ref_idc, VCL), AU-parity striping, current-frame VCL rows orange via decode-order AU match **explicitly marked approximate**, forbidden-bit tooltip; "Hex view" CollapsingHeader: selected unit dump incl. start code, 4 KB pages, offset/hex/ASCII columns — **no bit_offset highlight**: BLOCK bit_offset is frame-local decode-domain (oracle bit_range verdict), noted in the UI; **LF layer** toggle (toolbar-only, no key — L=Label/F=fit; enabled when frame aux_n > 28 = larger than the writer's empty blob) drawn via `FrameDerived.lf` lazy `frame_aux_decode`; Inspector **CABAC bins** CollapsingHeader (name/ctx/bin/bits, 256-row cap + "… N more", Deep-telemetry hint when cabac_n=0))
    - `bitstream_overlay.rs` - M4 shared layer renderers (`draw_mv_layer`/`draw_part_layer`/`draw_intra_layer`, M-B `draw_tu_layer` (depth-brightness `tu_stroke`, `is_transform_skip`), M-E `draw_lf_layer` (deblocking edge segments, VQA colour code green=strong/yellow=weak/red=not-applied with dim-red BS 0, zoom≥1.5), `LayerGeom`, `MvSource`, `build_refs`/`build_intra`/`build_tx`) + main-canvas mirror overlay (`draw_bitstream_overlay` + `OverlayCache` incl. per-block TX aggregate + lazy TX rows; mirrors the window's ViewConfig, default off)
  - `src/analysis/` - Analysis tools
    - `histogram.rs` - RGB and luma histograms
    - `waveform.rs` - Waveform display
    - `vectorscope.rs` - BT.709 YCbCr vectorscope
    - `metrics.rs` - PSNR, SSIM, frame difference, MS-PSNR, MS-SSIM, VMAF-NEG proxy, spatial metric maps
    - `block_stats.rs` - Per-block luma/MS mean & variance grid (Block tab); shared `pixel_value` metric
    - `motion.rs` - Per-block inter-frame motion classification vs previous frame (Motion tab): 4-level class (none/slight/much/full), two methods (PixelDiff MAD, StatsDiff |Δmean|+|Δstd|), adjustable thresholds, block sizes down to 8px
    - `scene.rs` - Scene change detection
    - `isp_sideband.rs` - SidebandPanel UI (load/unload, overlay mode, opacity)
    - `bitstream_stats.rs` - `BitstreamFile` (CVS-aware display map, resolution, block LRU), L1 8px rasterization (`rasterize_blocks`, area-weighted qp/bpp/mv/mode/coverage), LOD rule (`use_lod`, `lod_cell_size`), min-area hit test, viewer↔catb offset mapping, frame-type classes; M4: intra direction extraction (`extract_intra_modes`, `IntraDir`, HEVC 0=Planar/1=DC/2..34 angular `225−(mode−2)·180/32`°, AVC 4x4/8x8 standard 0..8 table, AVC 16x16 FFmpeg-enum order — oracle-confirmed, raster-order sub-grid assumption `intra_grid_dim`); M-A: filmstrip reference graph (`build_filmstrip_refs`/`derive_ref_edges` pure over slices, `nearest_poc_match` decode-distance POC resolution for multi-IDR repeats, `ref_count_tier` 3-tier dots, out-of-range/offset-clipped refs dropped); M-B: `BlockTxAgg`/`aggregate_block_tx` (per-block Σcoeff_abs_sum/Σnonzero; blocks without TX = true 0, not missing — excluding them would bias correlations toward residual-carrying blocks), `rasterize_blocks_tx` adds area-weighted `coeff_energy`/`nz_density` grids, Stats-tab aggregation (`aggregate_syntax_rows` bits-desc, `aggregate_cabac_rows` bins/bits/ctx-count, `collect_mv_points` per-PU with block fallback, `compute_frame_stats` incl. "(slice) " prefixed slice-header rows); M-C: `BitstreamFile.path` — the .catb location, anchor for stage-image resolution; M-E 0.21.0: `frame_aux_decode` 1-frame LRU (`aux_cache`) — loop-filter rows outnumber blocks 12–15× (fixtures 228–457 KB JSON/frame), only the on-screen frame stays parsed
    - `bitstream_panel.rs` - Sidebar mini panel (§9): load/unload, summary, frame offset spinner, open/focus window, §7 error strings, M4 "Overlay on main canvas" checkbox (`BitstreamAction`); M5 "Decoding… Ns" spinner + Cancel while a decoder run is in flight; M-A: "level: <capture_level> · contract: <contract>" badge ("n/a" when unrecorded); M-D 0.20.0: "Comparison (.catb B)" section (shown once A is loaded): Load B / Unload B, codec·frames·resolution summary, B's own frame-offset spinner, ⚠ frame-count-vs-A warning (resolution mismatch is rejected at load, not warned); B loads via this button only — drag & drop keeps routing to A (ambiguity guard)
    - `stage.rs` - M-C stage images + block quality (VQA Reconstruction/YUV + Info-Overlay PSNR/SSIM analogue): `StageKind` {final_recon, recon_unfiltered, prediction, residual}, `resolve_stage_path` (absolute → .catb-relative → basename sidecar), `StageCache` (16-entry LRU keyed (file ptr, decode idx, kind), failures cached), BT.601-luma `region_psnr`/`region_ssim` (global single-window SSIM per block — the form `calculate_ssim` degenerates to below 11px; grey content bit-matches `calculate_psnr`, differential-tested), `compute_block_quality` (∞ PSNR kept for bit-exact blocks, finite legend range), `psnr_to_g` (G-cell PSNR grid for the ReconPsnr Y metric, bit-exact cells capped at 99 dB so Pearson/Spearman stay finite)
    - `correlation.rs` - M2 correlation engine (pure, egui-free): analysis↔bitstream grid alignment onto G ∈ {8,16,32,64} (`resample_scalar_to_g`, `resample_variance_to_g` total-variance law, `resample_classes_to_g` majority vote, `resample_orientation_to_g` purity-weighted circular doubled-angle mean, `aggregate_bitstream_to_g` with coverage/partial-cell valid masks), `AlignedPair`, Pearson r / Spearman ρ (tie-average ranks), motion-class × bitstream table, CSV dump, `compute_analysis_grids` (8px L1-aligned), range-scan request/result types; M3: `opportunity_grid` (valid-cell z-score normalize both sides, cell = z_b − z_a, positive = bits above what complexity explains, σ=0 guard), `top_n_ranking` (z descending, raster-order tiebreak), `per_frame_stats`/`FramePairStat` (per-frame r/ρ/N accumulated by `CorrScanResult::new` on the scan thread); M-B: `YMetric::CoeffEnergy` (BitstreamG `coeff_energy`, sum→renormalize like bpp; zero-TX blocks contribute 0, kept valid) + "variance ↔ coeff energy" preset pair — range-scan path computes the TX aggregate only when the Y metric needs it; M-C: `YMetric::ReconPsnr` — image-derived `BitstreamG.psnr` filled by the caller after `aggregate_bitstream_to_g` (empty ⇒ every cell invalid, non-finite cells masked); the range scan loads the per-frame stage BMP on the scan thread and **skips** frames without one (not pushed into `frames`), resolution mismatch fails the scan up front with the reason; M-D 0.20.0: `YMetric::DeltaBpp`/`DeltaQp` (`is_delta()`, B-gated in the combos, auto-revert to bpp when B unloads) via `subtract_bitstream_g` (A−B on every scalar slot, AND'd valid, zeroed invalid cells, psnr cleared), `CorrScanRequest.offset_b` staleness, "variance ↔ Δbpp" preset pair (B-gated)
    - `bitstream_diff.rs` - M-D 0.20.0 dual-.catb Δ engine (pure, egui-free): `DiffGrid` cell-wise A−B over two same-geometry `BitstreamGrid`s (d_qp/d_bpp/d_mv/d_coeff/d_nz + mode_a/mode_b agreement; valid = both sides covered — no fake A−0), `DiffMetric` selector + `max_abs` symmetric legend scale, `diff_grids` (geometry mismatch → None, never a misaligned subtraction), `validate_b_resolution` load gate (A≠B resolution or unknown → rejected)
    - `orientation.rs` - Per-block dominant gradient orientation (02 §3): Sobel → magnitude-weighted 16-bin histogram → dominant edge angle [0,180) + purity; angle = perpendicular-to-gradient (intra prediction direction)
  - `src/conversion/` - Format conversion
    - `converter.rs` - VideoConverter, extract/pack YUV planes, chroma resampling
  - `tests/` - Integration tests
- `scripts/generate_test_data.py` - Test data generator (Python/OpenCV)
- `test_data/` - Sample raw video files (QCIF I420, NV12, RGB565, YUYV)

## Testing Conventions

- Run tests: `cd rust && cargo test`
- All tests must pass before any release.
- Framework: Rust integration tests in `rust/tests/` plus focused inline unit tests in `rust/src/**`

### Test Patterns

- Use `tempfile::NamedTempFile` or `tempfile::tempdir()` for temporary files
- Minimal I420 frame: 4x4 = 24 bytes (16 Y + 4 U + 4 V)
- Helper functions: `make_raw_i420(frames)`, `make_i420_frame(y, u, v)`
- Test naming: `test_<module>_<behavior>` (e.g., `test_pixel_info_yuyv_odd`)

### Test File Organization

| File | Scope |
|------|-------|
| `src/*` inline tests | App comparison sync, comparison viewport math, toolbar grid selector, sideband schema, bitstream L1 rasterization/LOD/hit-test/offset/classification, bitstream window presets/view math/fill normalization + opportunity diverging color, correlation resample/stats/CSV/class-table (incl. circular purity-gated orientation resample) + opportunity z-score/σ-guard/Top-N tiebreak/per-frame stats, orientation Sobel angles/purity, M4 intra angle tables (HEVC anchors/AVC/badges) + MV-source & layer-geom overlay tests, dropped-file classification (magic/ext/Annex-B sniff/short files), 0.16.0 decoder auto-detect probes (bundle/dev-checkout/PATH priority, both-files gate, none) + settings command truncation, M-A filmstrip ref graph (nearest-POC multi-IDR backward-only, B-slice edge pattern, offset/range clipping, count tiers) + telemetry-level token rewrite/parse (space/`=` forms, token-only None), M-B CoeffEnergy/NonzeroCoeffs fill normalization + CoeffEnergy G-aggregation/align, M-C BMP parser (bottom-up/top-down/32-bit/truncation/unsupported), stage path resolution (absolute/relative/sidecar/missing) + block PSNR/SSIM differential vs metrics.rs on grey content + ∞/'e' path + psnr_to_g cap + quality fill color/text + Picture labels + ReconPsnr align masking (172 tests) |
| `tests/colorspace_test.rs` | RGB/YUV color conversion sanity checks (12 tests) |
| `tests/formats_test.rs` | Format lookup, frame_size, categories (21 tests) |
| `tests/formats_extra_test.rs` | RGB16/32, semi-planar, packed frame sizes (9 tests) |
| `tests/reader_test.rs` | VideoReader open, seek, Y4M, RGB convert, channels (8 tests) |
| `tests/pixel_test.rs` | Pixel info: I420, YV12, NV12, NV21, RGB24, BGR24, Grey (12 tests) |
| `tests/pixel_packed_test.rs` | Pixel info: YUYV, UYVY, NV16 packed formats (5 tests) |
| `tests/highbit_test.rs` | 10/12/16-bit YUV/Grey/Bayer format, reader, pixel, and Y4M coverage (50 tests) |
| `tests/hints_test.rs` | Filename hint parsing and file-size resolution guessing (44 tests) |
| `tests/image_test.rs` | Image/PPM/Y4M stills, image sequences, and interlace metadata (27 tests) |
| `tests/y4m_test.rs` | Y4M header parsing, frame offsets (8 tests) |
| `tests/cache_test.rs` | LRU cache operations (6 tests) |
| `tests/converter_test.rs` | I420→NV12, identity, multi-frame, roundtrip, cancel (5 tests) |
| `tests/converter_extra_test.rs` | resample_chroma, I420→YV12, I420→422P, NV12 roundtrip (6 tests) |
| `tests/histogram_test.rs` | Histogram RGB/Y modes (2 tests) |
| `tests/waveform_test.rs` | Waveform luma/R/G/B, edge cases (6 tests) |
| `tests/vectorscope_test.rs` | Vectorscope neutral/red/blue, subsampling (5 tests) |
| `tests/metrics_test.rs` | PSNR, SSIM, frame difference, MS metrics, spatial metric maps, empty images (12 tests) |
| `tests/block_stats_test.rs` | Per-block luma/MS mean & variance grid, edge/partial blocks (10 tests) |
| `tests/motion_test.rs` | Per-block motion classification: pixel-diff vs avg·std, 4-class bands, localized motion, 8px grid, edge cases (14 tests) |
| `tests/scene_test.rs` | Scene detection algorithms, thresholds, save/load (12 tests) |
| `tests/sideband_test.rs` | Sideband binary parsing, extended header, signed fields, display (20 tests) |
| `tests/catb_test.rs` | `.catb` v4 bitstream metadata: header/directory validation, frame/block/ref records, sentinels, malformed block/ref-count/syntax-count and block-extent guards, SYNTAX record roundtrip, M-B TRANSFORM/coeffs/CABAC roundtrip (presence-bit variants, §11 indirect coeff slices, detail-dropped) + tx_n/cabac_n/coeff-span OOM guards + frame-stats over hevc_bslice (slice rows, both MV lists), oracle differential (frames/blocks/intra modes/per-block+per-frame transform·syntax·cabac counts/coeff nonzero·abs_sum consistency), display_map; fixture writer lives in `tests/common/mod.rs` (26 tests) |
| `tests/structure_test.rs` | M-A structured meta: parameter sets (SPS/VPS/PPS presence, sorted fields, width/height), DPB differential vs `oracle_frames.json` (rows / reference / hold / long-term / output-mark counts, reference POC list, `reference_labels` = DPB labels + slice ref labels), slice-header differential (l0/l1/total ref rows, `slice_reference_pocs`, type/name/fields/syntax presence), exactness arrays parallel to BLOCK records, empty capture_level/contract tolerated (5 tests) |
| `tests/bitstream_ui_test.rs` | M1 window pure logic: settings backward compat (old settings.toml), preset apply/custom/cycle, offset mapping, rasterization bpp/coverage, LOD, min-area hit test, filmstrip colors, pre-0.13 `[bitstream]` serde defaults, M-B layer_tu/new-fill settings roundtrip + coeff-energy rasterization + Stats syntax/CABAC aggregation (15 tests) |
| `tests/correlation_test.rs` | M2 V19/V20: flat/noise synthetic pipeline r > 0.8, partial-block masking, motion↔bits + class table, CSV rows/flags/r re-computation, real hevc_bslice end-to-end smoke; M3 V21: opportunity Top-N concentrates on the flat+high-bpp half (z_b−z_a design), per-frame r sequence through `CorrScanResult` (7 tests) |
| `tests/bitstream_diff_test.rs` | M-D dual-.catb Δ: signed diff + antisymmetry + mode agreement on partition-mismatched synthetic pairs, B display-map/offset independence, ΔbppG/ΔQpG subtraction + align, missing-B-frame all-invalid, resolution gate, real hevc_inter vs hevc_multi_idr (identical leading CVS) Δ≡0 (7 tests) |
| `tests/nal_test.rs` | M-E Annex-B NAL scanner: synthetic HEVC/AVC streams (3/4-byte start codes, consecutive units, long zero runs, header fields, AU estimation), garbage/truncated/empty inputs no-panic, codec detection by header scoring (hint cannot override a clear score), all 5 real fixtures (unit counts/types/offsets hexdump-verified), `unit_bytes`, source inference (workdir/sibling/fixture layout), BLOCK bit_offset frame-local verdict vs oracle bit_range (16 tests) |
| `tests/frame_aux_test.rs` | M-E `frame_aux` §12: synthetic loop_filter/SAO rows, 28-byte empty blob + aux_n=0, truncated JSON = Err no-panic, out-of-range aux range, mistyped-row leniency, `frame_aux_decode` 1-frame LRU Arc identity, fixture structure (hevc_bslice 224 rows/112 vertical/52 weak + 4 not_applied SAO, hevc_inter strong BS-2 edges, AVC empty) (7 tests) |
| `tests/common/mod.rs` | Shared synthetic `.catb` v4 fixture writer (used by catb_test + correlation_test + launcher_test; no tests itself) |
| `tests/launcher_test.rs` | M5 decoder-run launcher: cp-based fake decoder end-to-end (quoted paths, telemetry load), stderr-last-line failure, cancel kill/reap, stale-job publish gate, YUV auto-open plan; env-gated `E2E_DECODER_CMD` real-decoder run (9 tests, unix) |
| `tests/dropped_test.rs` | Dropped-file classification against real `test_data/bitstream` fixtures: .catb/.h265/.h264/.yuv/.json, extensionless + .bin sniff, magic-beats-extension (5 tests) |
| `tests/ppm_test.rs` | PPM parsing, writing, reading, and conversion (12 tests) |
| `tests/diff_stats_test.rs` | Video Diff spatial statistics (9 tests) |
| `tests/e2e_hm_test.rs` | End-to-end HM B-pyramid stream vs codec-analyzer oracle differential; skips unless `E2E_CATB`/`E2E_ORACLE` are set (1 test) |
| `tests/stage_test.rs` | M-C stage images: `stage_images` meta parsing from synthetic .catb, relative + sidecar + missing BMP loading through `StageCache` (LRU Arc identity, out-of-range guard); env-gated `E2E_STAGE_RUN=<workdir>` real decoder-run check — every recorded stage BMP loads at the stream resolution (4 tests) |
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
| `classc` / `hevc_c` | 832 | 480 | HEVC CTC Class C (BasketballDrill, BQMall, PartyScene, RaceHorses) |
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
