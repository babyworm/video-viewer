# YUV/Raw Video Viewer

## Version

- Current version: defined in `video_viewer/__init__.py` (`__version__`)
- Follows Semantic Versioning (SemVer): `MAJOR.MINOR.PATCH`

### Versioning Rules

- **MAJOR**: Only the user (babyworm) decides when to bump the major version.
- **MINOR**: Increment when adding new features (e.g., new UI panel, new analysis tool, new format support).
- **PATCH**: Increment for bug fixes, small UI tweaks, refactoring without new features.
- Always update `video_viewer/__init__.py` when the version changes.
- The About dialog reads `__version__` automatically.

## Project Structure

- `video_viewer/` - Main package
  - `main_window.py` - ImageCanvas, MainWindow (UI, toolbar, menus, shortcuts)
  - `video_reader.py` - VideoReader, FrameCache (format decoding, pixel inspection)
  - `format_manager.py` - FormatManager, FormatType (YUV/RGB/Bayer format definitions)
  - `analysis.py` - VideoAnalyzer (PSNR, SSIM, histogram)
  - `comparison_view.py` - ComparisonWindow (A/B compare)
  - `__init__.py` - Package version (`__version__`)
- `test/` - pytest test suite

## Testing

- Run tests: `.venv/bin/python -m pytest test/ -v`
- All tests must pass before any release.

## Copyright

Copyright (c) babyworm (Hyun-Gyu Kim)
