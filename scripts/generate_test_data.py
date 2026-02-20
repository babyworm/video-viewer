import numpy as np
import cv2
import os
import sys

# Define QCIF resolution
WIDTH = 176
HEIGHT = 144
FRAMES = 3

def generate_rgb_frames():
    """Generates 3 RGB frames with moving shapes."""
    frames = []
    for i in range(FRAMES):
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        # Background color change
        img[:, :] = (i * 20, 50, 100)

        # Moving rectangle
        start_x = 10 + i * 20
        start_y = 10 + i * 10
        cv2.rectangle(img, (start_x, start_y), (start_x+30, start_y+30), (255, 255, 0), -1)

        # Moving circle
        center_x = WIDTH - 30 - i * 20
        center_y = HEIGHT - 30 - i * 10
        cv2.circle(img, (center_x, center_y), 15, (0, 0, 255), -1)

        frames.append(img)
    return frames

def save_raw(filename, data):
    with open(filename, "wb") as f:
        f.write(data)
    print(f"Generated {filename}")

def main():
    os.makedirs("test_data", exist_ok=True)
    frames_rgb = generate_rgb_frames()

    # 1. RGB565 (Packed 16-bit)
    raw_rgb565 = b""
    for frame in frames_rgb:
        # Convert BGR (OpenCV default) to RGB first if using standard RGB565?
        # OpenCV's BGR2BGR565 creates Little Endian 16-bit
        # Let's assume input is BGR for OpenCV functions
        bgr = frame
        converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGR565)
        raw_rgb565 += converted.tobytes()
    save_raw("test_data/test_qcif.rgb565", raw_rgb565)

    # 2. I420 (Planar YUV 4:2:0)
    raw_i420 = b""
    for frame in frames_rgb:
        bgr = frame
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        raw_i420 += yuv.tobytes()
    save_raw("test_data/test_qcif.i420", raw_i420)

    # 3. NV12 (Semi-Planar YUV 4:2:0)
    raw_nv12 = b""
    for frame in frames_rgb:
        bgr = frame
        # OpenCV doesn't have direct BGR2NV12, but we can do BGR->I420 and then shuffle, or use specific codes if avail
        # Easier: BGR->YUV I420 then convert I420->NV12 manually or via cvtColor if supported
        # Actually cv2.COLOR_BGR2YUV_I420 returns Y then U then V planar.
        # NV12 is Y then UV interleaved.

        yuv_i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        yuv_flat = yuv_i420.flatten()
        y_size = WIDTH * HEIGHT
        uv_size = y_size // 4

        y = yuv_flat[:y_size]
        u = yuv_flat[y_size : y_size + uv_size]
        v = yuv_flat[y_size + uv_size :]

        # Interleave U and V
        uv = np.empty((uv_size * 2,), dtype=np.uint8)
        uv[0::2] = u
        uv[1::2] = v

        raw_nv12 += y.tobytes()
        raw_nv12 += uv.tobytes()
    save_raw("test_data/test_qcif.nv12", raw_nv12)

    # 4. YUYV (Packed YUV 4:2:2)
    raw_yuyv = b""
    for frame in frames_rgb:
        bgr = frame
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_YUYV) # YUY2
        raw_yuyv += yuv.tobytes()
    save_raw("test_data/test_qcif.yuyv", raw_yuyv)

if __name__ == "__main__":
    main()
