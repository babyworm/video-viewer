import unittest
import os
import sys
import subprocess

from video_viewer.video_reader import VideoReader

class TestAdvancedFeatures(unittest.TestCase):
    def create_dummy_y4m(self, filename, width, height, frames):
        header = f"YUV4MPEG2 W{width} H{height} F30:1 Ip A0:0 C420mpeg2 XYSCSS=420JPEG\n"
        frame_header = "FRAME\n"

        # I420 payload
        y_size = width * height
        uv_size = (width // 2) * (height // 2)
        payload_size = y_size + 2 * uv_size

        with open(filename, "wb") as f:
            f.write(header.encode('ascii'))
            for _ in range(frames):
                f.write(frame_header.encode('ascii'))
                f.write(bytearray([128] * payload_size)) # Gray
        return filename

    def test_y4m_reader(self):
        filename = "test.y4m"
        self.create_dummy_y4m(filename, 128, 128, 3)

        try:
            reader = VideoReader(filename, 0, 0, "") # Dims should be ignored/overwritten

            self.assertEqual(reader.width, 128)
            self.assertEqual(reader.height, 128)
            self.assertEqual(reader.total_frames, 3)
            self.assertEqual(reader.format.fourcc, "YU12")

            # Read frame
            raw = reader.seek_frame(1)
            self.assertEqual(len(raw), 128*128*1.5)

        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_headless_conversion(self):
        # Create input I420
        infile = os.path.abspath("in_c.yuv")
        outfile = os.path.abspath("out_c.yuv")
        width, height = 64, 64
        size = int(width * height * 1.5)

        with open(infile, "wb") as f:
            f.write(bytearray([100] * size))

        # Run conversion command (I420 -> NV12)
        # Note: Run as module to support relative imports
        cmd = [
            sys.executable, "-m", "video_viewer.main",
            infile,
            "--width", str(width),
            "--height", str(height),
            "-vi", "I420",
            "-vo", "NV12",
            "-o", outfile
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
            self.assertEqual(result.returncode, 0)
            self.assertTrue(os.path.exists(outfile))
            self.assertIn(f"Converting {infile} to {outfile}", result.stdout)

            # Verify output size (should be same for I420->NV12)
            self.assertEqual(os.path.getsize(outfile), size)

        finally:
            if os.path.exists(infile):
                os.remove(infile)
            if os.path.exists(outfile):
                os.remove(outfile)

if __name__ == '__main__':
    unittest.main()
