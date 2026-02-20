import unittest
import numpy as np
import os
import sys

from video_viewer.video_reader import VideoReader
from video_viewer.format_manager import FormatManager, FormatType

class TestVideoReader(unittest.TestCase):
    def setUp(self):
        self.width = 64
        self.height = 64
        self.frames = 2
        self.test_files = []

    def tearDown(self):
        for f in self.test_files:
            if os.path.exists(f):
                os.remove(f)

    def create_dummy_file(self, filename, size_bytes):
        with open(filename, "wb") as f:
            f.write(bytearray([0] * size_bytes))
        self.test_files.append(filename)
        return filename

    def test_frame_size_calculation(self):
        fm = FormatManager()

        # Test I420 (1.5 bpp)
        fmt = fm.get_format("I420 (4:2:0) [YU12]")
        size = fmt.calculate_frame_size(100, 100)
        self.assertEqual(size, 100*100 + 50*50*2) # 15000

        # Test NV12 (1.5 bpp)
        # Test NV12 (1.5 bpp)
        fmt = fm.get_format("NV12 (4:2:0) [NV12]")
        size = fmt.calculate_frame_size(100, 100)
        self.assertEqual(size, 100*100 + 50*50*2) # 15000

        # Test YUYV (2 bpp)
        # Test YUYV (2 bpp)
        fmt = fm.get_format("YUYV (4:2:2) [YUYV]")
        size = fmt.calculate_frame_size(100, 100)
        self.assertEqual(size, 100*100*2) # 20000

        # Test RGB888 (3 bpp)
        # Test RGB888 (3 bpp)
        fmt = fm.get_format("RGB24 (24-bit) [RGB3]")
        size = fmt.calculate_frame_size(100, 100)
        self.assertEqual(size, 100*100*3) # 30000

    def test_reader_bounds(self):
        # Create a file with exactly 2 frames of I420
        frame_size = 64*64 * 3 // 2
        file_path = self.create_dummy_file("test_bounds.yuv", frame_size * 2)

        reader = VideoReader(file_path, 64, 64, "I420 (4:2:0) [YU12]")
        self.assertEqual(reader.total_frames, 2)

        # Should succeed
        reader.seek_frame(0)
        reader.seek_frame(1)

        # Should fail
        with self.assertRaises(ValueError):
            reader.seek_frame(2)
        with self.assertRaises(ValueError):
            reader.seek_frame(-1)

    def test_bayer_support(self):
        # Bayer is 1 byte per pixel for 8 bit
        frame_size = 64*64
        file_path = self.create_dummy_file("test_bayer.raw", frame_size)

        file_path = self.create_dummy_file("test_bayer.raw", frame_size)

        reader = VideoReader(file_path, 64, 64, "Bayer RGGB (8-bit) [RGGB]")
        raw = reader.seek_frame(0)
        rgb = reader.convert_to_rgb(raw)

        self.assertIsNotNone(rgb)
        self.assertEqual(rgb.shape, (64, 64, 3))

    def test_rgb_support(self):
        # RGB888
        frame_size = 64*64*3
        file_path = self.create_dummy_file("test_rgb.raw", frame_size)

        file_path = self.create_dummy_file("test_rgb.raw", frame_size)

        reader = VideoReader(file_path, 64, 64, "RGB24 (24-bit) [RGB3]")
        raw = reader.seek_frame(0)
        rgb = reader.convert_to_rgb(raw)

        self.assertIsNotNone(rgb)
        self.assertEqual(rgb.shape, (64, 64, 3))

if __name__ == '__main__':
    unittest.main()
