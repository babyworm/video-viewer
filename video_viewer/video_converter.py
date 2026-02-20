import cv2
import numpy as np
import os
from .video_reader import VideoReader
from .format_manager import FormatManager, FormatType

class VideoConverter:
    def __init__(self):
        self.format_manager = FormatManager()

    def convert(self, input_path, width, height, input_fmt, output_path, output_fmt):
        reader = VideoReader(input_path, width, height, input_fmt)
        out_fmt_obj = self.format_manager.get_format(output_fmt)

        if not out_fmt_obj:
            raise ValueError(f"Unsupported output format: {output_fmt}")

        with open(output_path, "wb") as out_f:
             for i in range(reader.total_frames):
                 raw = reader.seek_frame(i)

                 if reader.format.name == output_fmt:
                     # Direct copy if formats match
                     out_f.write(raw)
                 else:
                     # Convert to RGB then to target format using VideoReader's helper
                     # This ensures consistency with Export logic
                     rgb = reader.convert_to_rgb(raw)

                     if rgb is not None:
                         out_data = reader.convert_rgb_to_bytes(rgb, out_fmt_obj.fourcc)
                         if out_data:
                             out_f.write(out_data)
                         else:
                             print(f"Conversion to {output_fmt} (FourCC: {out_fmt_obj.fourcc}) not supported.")
                             break
                     else:
                        print(f"Failed to decode input frame {i}")

        print(f"Converted {reader.total_frames} frames to {output_path}")
