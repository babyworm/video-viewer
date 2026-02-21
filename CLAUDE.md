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

## Testing

- Run tests: `.venv/bin/python -m pytest test/ -v`
- All tests must pass before any release.

## Copyright

Copyright (c) babyworm (Hyun-Gyu Kim)
