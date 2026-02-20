import sys
import argparse
from PySide6.QtWidgets import QApplication
from .main_window import MainWindow
from .video_converter import VideoConverter

def main():
    parser = argparse.ArgumentParser(description="YUV/RAW Video Viewer and Converter")
    parser.add_argument("file", nargs="?", help="Path to input video file")
    parser.add_argument("--width", type=int, default=1920, help="Width of the video")
    parser.add_argument("--height", type=int, default=1080, help="Height of the video")
    parser.add_argument("--format", "-vi", dest="input_format", type=str, default="I420", help="Input Pixel format (e.g. I420, NV12)")

    parser.add_argument("-vo", "--output-format", dest="output_format", type=str, help="Output Pixel format (for conversion)")
    parser.add_argument("-o", "--output", dest="output_file", type=str, help="Output file path (triggers headless mode)")

    args = parser.parse_args()

    if args.output_file:
        # Headless mode
        if not args.file:
            print("Error: Input file required for conversion")
            sys.exit(1)

        output_fmt = args.output_format if args.output_format else args.input_format
        print(f"Converting {args.file} to {args.output_file} ({output_fmt})...")

        converter = VideoConverter()
        try:
             converter.convert(args.file, args.width, args.height, args.input_format, args.output_file, output_fmt)
        except Exception as e:
            print(f"Error converting video: {e}")
            sys.exit(1)

        sys.exit(0)

    # GUI Mode
    app = QApplication(sys.argv)
    window = MainWindow(args.file, args.width, args.height, args.input_format)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
