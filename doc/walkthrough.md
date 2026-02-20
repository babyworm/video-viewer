# YUV/RAW Video Viewer Walkthrough

I have implemented a standalone GUI tool for viewing raw YUV and Bayer video files.

## Features
- **Supported Formats**:
    - **YUV**: I420, NV12, NV21, YUYV, UYVY, YV12
    - **Bayer**: RGGB, BGGR, GBRG, GRBG (8-bit)
    - **RGB**: RGB888, ARGB, RGB565
- **Visualization**:
    - Frame-by-frame navigation.
    - Component view (Y, U, V or R, G, B separate channels).
    - Grid overlay (16x16, 32x32, 64x64, 128x128).
    - Zooming (via window resize).
- **Conversion**:
    - Save current frame as PNG or BMP.

## Setup
The tool requires `numpy`, `opencv-python`, and `PyQt6`. I have set up a virtual environment in `.venv`.

**To run the application:**

```bash
.venv/bin/python main.py
```

## Usage Guide
1.  **Launch the App**: Run the command above.
2.  **Open File**: Click "Open File" and select your `.yuv` or raw file.
3.  **Set Parameters**:
    - Enter the **Width** and **Height** (e.g., 1920 1080).
    - Select the **Format** from the dropdown.
    - Click **Apply Resolution** to reload the video with new settings.
4.  **Navigation**:
    - Use the slider or `<` `>` buttons to move between frames.
5.  **View Options**:
    - **Grid Overlay**: Select a grid size to check macroblocks.
    - **Component**: Switch between "Full" color (converted RGB) or individual raw components (Y, U, V).
6.  **Save**:
    - Click "Save Frame" to export the current view to an image file.

## Verification
I have verified the core logic using a test script `test_video_reader.py` which:
1.  Generates a synthetic I420 YUV file (Solid Red).
2.  Reads the file using the `VideoReader` class.
3.  Verifies the converted RGB values match expected Red.
4.  Verifies the extracted Y, U, V channels match expected raw values.

You can run the verification script yourself:
```bash
.venv/bin/python test_video_reader.py
```
