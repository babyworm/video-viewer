import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class VideoAnalyzer:
    """
    Provides methods for video analysis including Histograms, Vectorscope, and Quality Metrics.
    """

    @staticmethod
    def calculate_histogram(image_data, channels="RGB"):
        """
        Calculate histogram for the given image data.

        Args:
            image_data (np.ndarray): Image data (H, W, C) or (H, W).
            channels (str): "RGB" or "Y".

        Returns:
            dict: {channel_name: (hist_values, bin_edges)}
        """
        hists = {}
        if image_data is None:
            return hists

        # Ensure image is uint8 for standard histogram
        if image_data.dtype != np.uint8:
            # Normalize to 8-bit if needed, or just skip for now
            # For 16-bit, we might want 65536 bins, but let's stick to 256 for display
            if image_data.max() > 255:
                 image_data = (image_data / 256).astype(np.uint8)
            else:
                 image_data = image_data.astype(np.uint8)

        if channels == "RGB":
            # Assume image_data is RGB
            if len(image_data.shape) == 3 and image_data.shape[2] >= 3:
                colors = ('r', 'g', 'b')
                for i, color in enumerate(colors):
                    # cv2.calcHist(images, channels, mask, histSize, ranges)
                    hist = cv2.calcHist([image_data], [i], None, [256], [0, 256])
                    hists[color] = hist.flatten()
        elif channels == "Y":
            # Grayscale histogram
            if len(image_data.shape) == 3:
                # Convert to gray if RGB
                gray = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_data

            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hists['y'] = hist.flatten()

        return hists

    @staticmethod
    def calculate_vectorscope(yuv_data, width, height, format_type):
        """
        Extract Cb and Cr components for Vectorscope plotting.

        Args:
            yuv_data (bytes): Raw YUV data.
            width (int): Frame width.
            height (int): Frame height.
            format_type (str): Format FourCC/Type to determine how to extract UV.

        Returns:
            tuple: (Cb_values, Cr_values) as flat arrays.
        """
        # This requires parsing the raw YUV data similar to VideoReader
        # efficient sampling is key here. We don't want to plot all pixels.
        # Let's import VideoReader logic or just simple extraction for common formats.
        # For now, let's assume we can get a UV plane or U/V planes.

        # NOTE: This method might be best handled by the VideoReader returning planar views
        # independently, but let's implement a generic reader-helper or rely on parsed RGB?
        # Vectorscope is strictly strictly YCbCr. Converting RGB -> YCbCr is lossy but easiest if we already have RGB.
        # Reading raw U/V is better.

        pass

    @staticmethod
    def calculate_vectorscope_from_rgb(rgb_image):
        """
        Calculate CbCr values from an RGB image for Vectorscope.

        Args:
            rgb_image (np.ndarray): HxWx3 RGB image.

        Returns:
            tuple: (Cb, Cr) arrays, sampled/flattened.
        """
        if rgb_image is None:
            return None, None

        # Resize for performance if too large?
        h, w, _ = rgb_image.shape
        scale = 1
        if w > 640:
            scale = 4

        if scale > 1:
            small = rgb_image[::scale, ::scale, :]
        else:
            small = rgb_image

        ycrcb = cv2.cvtColor(small, cv2.COLOR_RGB2YCrCb)

        # Y=0, Cr=1, Cb=2 in OpenCV YCrCb?
        # Verify: OpenCV docs say Y, Cr, Cb.
        # Cr is V (Red-diff), Cb is U (Blue-diff).
        # Vectorscope usually plots Cb (X) vs Cr (Y).

        Cr = ycrcb[:,:,1].flatten()
        Cb = ycrcb[:,:,2].flatten()

        return Cb, Cr

    @staticmethod
    def calculate_psnr(img1, img2):
        """
        Calculate PSNR (Peak Signal-to-Noise Ratio).
        """
        if img1 is None or img2 is None:
            return 0.0
        if img1.shape != img2.shape:
            return 0.0

        return cv2.PSNR(img1, img2)

    @staticmethod
    def calculate_ssim(img1, img2):
        """
        Calculate SSIM (Structural Similarity Index).
        """
        if img1 is None or img2 is None:
            return 0.0
        if img1.shape != img2.shape:
            return 0.0

        # SSIM needs grayscale usually, or we average channel SSIMs
        # skimage ssim supports multichannel

        # Allow small window size for small images
        win_size = min(7, min(img1.shape[0], img1.shape[1]))
        if win_size % 2 == 0: win_size -= 1

        if win_size < 3:
            return 1.0 # Too small

        params = {"win_size": win_size, "channel_axis": 2}
        if len(img1.shape) < 3:
             del params["channel_axis"]

        return ssim(img1, img2, **params)

    @staticmethod
    def calculate_waveform(rgb_image, channel="luma"):
        """
        Calculate waveform monitor data.

        Args:
            rgb_image: HxWx3 RGB image.
            channel: "luma", "r", "g", or "b".

        Returns:
            np.ndarray: 2D histogram (256 x width) representing waveform intensity.
        """
        if rgb_image is None:
            return None

        h, w, _ = rgb_image.shape

        # Downsample width for performance
        target_w = min(w, 720)
        if w > target_w:
            step = w // target_w
            img = rgb_image[:, ::step, :]
            w = img.shape[1]
        else:
            img = rgb_image

        if channel == "luma":
            # BT.709 luma
            data = (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] +
                    0.0722 * img[:, :, 2]).astype(np.uint8)
        elif channel == "r":
            data = img[:, :, 0]
        elif channel == "g":
            data = img[:, :, 1]
        elif channel == "b":
            data = img[:, :, 2]
        else:
            data = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Build waveform: for each column x, count how many pixels have value y
        waveform = np.zeros((256, w), dtype=np.float32)
        for x in range(w):
            col = data[:, x]
            hist, _ = np.histogram(col, bins=256, range=(0, 256))
            waveform[:, x] = hist

        return waveform

    @staticmethod
    def calculate_frame_difference(frame_a_rgb, frame_b_rgb):
        """
        Calculate difference metric between two frames for scene change detection.

        Returns:
            float: Mean absolute difference (0-255 scale).
        """
        if frame_a_rgb is None or frame_b_rgb is None:
            return 0.0
        if frame_a_rgb.shape != frame_b_rgb.shape:
            return 255.0

        diff = np.abs(frame_a_rgb.astype(np.float32) - frame_b_rgb.astype(np.float32))
        return float(np.mean(diff))

    @staticmethod
    def calculate_histogram_difference(frame_a_rgb, frame_b_rgb):
        """
        Calculate histogram correlation difference between two frames.

        Returns:
            float: 0.0 = identical, 100.0 = completely different (scaled for threshold UI).
        """
        if frame_a_rgb is None or frame_b_rgb is None:
            return 0.0
        if frame_a_rgb.shape != frame_b_rgb.shape:
            return 100.0

        score = 0.0
        for ch in range(3):
            hist_a = cv2.calcHist([frame_a_rgb], [ch], None, [256], [0, 256])
            hist_b = cv2.calcHist([frame_b_rgb], [ch], None, [256], [0, 256])
            cv2.normalize(hist_a, hist_a)
            cv2.normalize(hist_b, hist_b)
            corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
            score += (1.0 - corr)  # 0=same, 1=different per channel
        # Average across 3 channels, scale to 0-100
        return float(score / 3.0 * 100.0)

    @staticmethod
    def calculate_ssim_difference(frame_a_rgb, frame_b_rgb):
        """
        Calculate SSIM-based difference between two frames.

        Returns:
            float: 0.0 = identical, 100.0 = completely different (scaled for threshold UI).
        """
        if frame_a_rgb is None or frame_b_rgb is None:
            return 0.0
        if frame_a_rgb.shape != frame_b_rgb.shape:
            return 100.0

        gray_a = cv2.cvtColor(frame_a_rgb, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(frame_b_rgb, cv2.COLOR_RGB2GRAY)
        score = ssim(gray_a, gray_b)
        # SSIM: 1.0=identical, 0.0=different → invert and scale to 0-100
        return float((1.0 - score) * 100.0)
