"""
A/B Comparison View for Video Viewer
Provides side-by-side, overlay, and diff comparison modes for two videos.
"""

import logging
from enum import Enum
from typing import Optional

import cv2

logger = logging.getLogger(__name__)
import numpy as np
from PySide6.QtCore import Qt, QPoint, QRect, Signal
from PySide6.QtGui import QPainter, QImage, QPixmap, QPen, QColor, QWheelEvent, QMouseEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QSlider, QLabel, QPushButton, QFileDialog,
    QMessageBox, QToolBar, QStatusBar, QDockWidget, QFrame
)

from .video_reader import VideoReader
from .analysis import VideoAnalyzer
from .format_manager import FormatManager, FormatType
from .video_converter import VideoConverter


class ComparisonMode(Enum):
    """Comparison display modes"""
    SPLIT = "Split View"
    OVERLAY = "Overlay"
    DIFF = "Difference"


class ComparisonCanvas(QWidget):
    """Canvas that renders two images in split/overlay/diff mode"""

    mouse_moved = Signal(int, int)  # image-space (x, y)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)

        # Images to compare
        self.image_a: Optional[QImage] = None
        self.image_b: Optional[QImage] = None

        # Display settings
        self.mode = ComparisonMode.SPLIT
        self.split_position = 0.5  # 0.0 to 1.0
        self.overlay_opacity = 0.5  # 0.0 to 1.0

        # Zoom and pan
        self.zoom_level = 1.0
        self.pan_offset = QPoint(0, 0)

        # Dragging state
        self.dragging_divider = False
        self.dragging_pan = False
        self.last_mouse_pos = QPoint()

        # Enable mouse tracking for divider hover
        self.setMouseTracking(True)

    def set_images(self, img_a: np.ndarray, img_b: np.ndarray):
        """Set the two images to compare (RGB numpy arrays)"""
        self.image_a = self._numpy_to_qimage(img_a)
        self.image_b = self._numpy_to_qimage(img_b)
        self.update()

    def _numpy_to_qimage(self, img: np.ndarray) -> QImage:
        """Convert RGB numpy array to QImage"""
        if img is None:
            return None
        h, w = img.shape[:2]
        if img.shape[2] == 3:
            bytes_per_line = 3 * w
            # .copy() so QImage owns its data (prevents corruption when numpy is GC'd)
            return QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        return None

    def set_mode(self, mode: ComparisonMode):
        """Change comparison mode"""
        self.mode = mode
        self.update()

    def set_split_position(self, position: float):
        """Set split divider position (0.0 to 1.0)"""
        self.split_position = max(0.0, min(1.0, position))
        if self.mode == ComparisonMode.SPLIT:
            self.update()

    def set_overlay_opacity(self, opacity: float):
        """Set overlay opacity (0.0 to 1.0)"""
        self.overlay_opacity = max(0.0, min(1.0, opacity))
        if self.mode == ComparisonMode.OVERLAY:
            self.update()

    def reset_view(self):
        """Reset zoom and pan to default"""
        self.zoom_level = 1.0
        self.pan_offset = QPoint(0, 0)
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming"""
        if event.angleDelta().y() > 0:
            self.zoom_level *= 1.1
        else:
            self.zoom_level /= 1.1
        self.zoom_level = max(0.1, min(10.0, self.zoom_level))
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for divider drag or pan"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode == ComparisonMode.SPLIT and self._is_near_divider(event.pos()):
                self.dragging_divider = True
            else:
                self.dragging_pan = True
            self.last_mouse_pos = event.pos()

    def _screen_to_image(self, pos: QPoint):
        """Convert screen coordinates to image coordinates. Returns (x, y) or None."""
        if self.image_a is None:
            return None
        img_w = int(self.image_a.width() * self.zoom_level)
        img_h = int(self.image_a.height() * self.zoom_level)
        ox = (self.width() - img_w) // 2 + self.pan_offset.x()
        oy = (self.height() - img_h) // 2 + self.pan_offset.y()
        ix = int((pos.x() - ox) / self.zoom_level)
        iy = int((pos.y() - oy) / self.zoom_level)
        if 0 <= ix < self.image_a.width() and 0 <= iy < self.image_a.height():
            return ix, iy
        return None

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for dragging"""
        if self.dragging_divider:
            # Update split position
            rel_x = event.pos().x() / self.width()
            self.set_split_position(rel_x)
        elif self.dragging_pan:
            # Pan the view
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = event.pos()
            self.update()
        else:
            if self.mode == ComparisonMode.SPLIT:
                if self._is_near_divider(event.pos()):
                    self.setCursor(Qt.CursorShape.SplitHCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
            # Emit image coordinates for pixel inspector
            coords = self._screen_to_image(event.pos())
            if coords:
                self.mouse_moved.emit(coords[0], coords[1])

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        self.dragging_divider = False
        self.dragging_pan = False

    def _is_near_divider(self, pos: QPoint) -> bool:
        """Check if mouse position is near split divider"""
        divider_x = int(self.width() * self.split_position)
        return abs(pos.x() - divider_x) < 10

    def paintEvent(self, event):
        """Render the comparison view"""
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        if self.image_a is None or self.image_b is None:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                           "Load reference video to begin comparison")
            return

        # Calculate scaled image size
        img_w = int(self.image_a.width() * self.zoom_level)
        img_h = int(self.image_a.height() * self.zoom_level)

        # Center image with pan offset
        x = (self.width() - img_w) // 2 + self.pan_offset.x()
        y = (self.height() - img_h) // 2 + self.pan_offset.y()

        if self.mode == ComparisonMode.SPLIT:
            self._paint_split(painter, x, y, img_w, img_h)
        elif self.mode == ComparisonMode.OVERLAY:
            self._paint_overlay(painter, x, y, img_w, img_h)
        elif self.mode == ComparisonMode.DIFF:
            self._paint_diff(painter, x, y, img_w, img_h)

    def _paint_split(self, painter: QPainter, x: int, y: int, w: int, h: int):
        """Render split view mode using clip regions for correctness."""
        divider_x = int(self.width() * self.split_position)
        dst_rect = QRect(x, y, w, h)

        # Left side (image A) - clip to left of divider
        painter.save()
        painter.setClipRect(QRect(0, 0, divider_x, self.height()))
        painter.drawImage(dst_rect, self.image_a)
        painter.restore()

        # Right side (image B) - clip to right of divider
        painter.save()
        painter.setClipRect(QRect(divider_x, 0, self.width() - divider_x, self.height()))
        painter.drawImage(dst_rect, self.image_b)
        painter.restore()

        # Draw divider line
        painter.setPen(QPen(QColor(255, 255, 0), 2))
        painter.drawLine(divider_x, 0, divider_x, self.height())

    def _paint_overlay(self, painter: QPainter, x: int, y: int, w: int, h: int):
        """Render overlay mode"""
        # Draw base image (A)
        dst_rect = QRect(x, y, w, h)
        painter.drawImage(dst_rect, self.image_a)

        # Draw overlay image (B) with opacity
        painter.setOpacity(self.overlay_opacity)
        painter.drawImage(dst_rect, self.image_b)
        painter.setOpacity(1.0)

    def _paint_diff(self, painter: QPainter, x: int, y: int, w: int, h: int):
        """Render difference heatmap mode"""
        dst_rect = QRect(x, y, w, h)
        # In diff mode, image_a holds the diff heatmap (set by ComparisonWindow)
        painter.drawImage(dst_rect, self.image_a)
        # Draw image boundary so all-zero diff is still visible
        painter.setPen(QPen(QColor(128, 128, 128), 1, Qt.PenStyle.DashLine))
        painter.drawRect(dst_rect)


class ComparisonWindow(QMainWindow):
    """Standalone window for A/B video comparison"""

    def __init__(self, main_reader: VideoReader, parent=None):
        super().__init__(parent)

        # Video readers
        self.main_reader = main_reader
        self.ref_reader: Optional[VideoReader] = None

        # Current frame index
        self.current_frame = 0
        self._diff_stats = None

        # Setup UI
        self.setWindowTitle("A/B Comparison")
        self.setGeometry(100, 100, 1200, 800)

        self._setup_ui()

    def _setup_ui(self):
        """Initialize UI components"""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # Canvas (created first so toolbar can reference it)
        self.canvas = ComparisonCanvas()

        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)

        layout.addWidget(self.canvas, stretch=1)

        # Navigation controls
        nav_layout = self._create_navigation()
        layout.addLayout(nav_layout)

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Load reference video to begin comparison")

        # Pixel inspector dock
        self._setup_inspector()

    def _create_toolbar(self) -> QWidget:
        """Create toolbar with controls"""
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)

        # Load reference button
        btn_load = QPushButton("Load Reference Video")
        btn_load.clicked.connect(self.load_reference)
        layout.addWidget(btn_load)

        layout.addSpacing(20)

        # Mode selector
        layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        for mode in ComparisonMode:
            self.mode_combo.addItem(mode.value, mode)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_combo)

        layout.addSpacing(20)

        # Opacity slider (for overlay mode)
        layout.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setMaximumWidth(150)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        layout.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("50%")
        layout.addWidget(self.opacity_label)

        layout.addSpacing(20)

        # Metrics display
        layout.addWidget(QLabel("PSNR:"))
        self.psnr_label = QLabel("--")
        self.psnr_label.setMinimumWidth(80)
        layout.addWidget(self.psnr_label)

        layout.addWidget(QLabel("SSIM:"))
        self.ssim_label = QLabel("--")
        self.ssim_label.setMinimumWidth(80)
        layout.addWidget(self.ssim_label)

        layout.addSpacing(20)

        # Reference format selector
        layout.addWidget(QLabel("Ref Format:"))
        self.ref_format_combo = QComboBox()
        self.ref_format_combo.setMinimumWidth(160)
        fm = FormatManager()
        self.ref_format_combo.addItem("(Same as source)", None)
        for name in fm.get_supported_formats():
            self.ref_format_combo.addItem(name, name)
        layout.addWidget(self.ref_format_combo)

        layout.addStretch()

        # Reset view button
        btn_reset = QPushButton("Reset View")
        btn_reset.clicked.connect(self.canvas.reset_view)
        layout.addWidget(btn_reset)

        return toolbar

    def _create_navigation(self) -> QHBoxLayout:
        """Create navigation controls"""
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)

        # Previous button
        btn_prev = QPushButton("◀ Prev")
        btn_prev.clicked.connect(self._prev_frame)
        layout.addWidget(btn_prev)

        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, max(0, self.main_reader.total_frames - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.frame_slider, stretch=1)

        # Next button
        btn_next = QPushButton("Next ▶")
        btn_next.clicked.connect(self._next_frame)
        layout.addWidget(btn_next)

        # Frame counter
        self.frame_label = QLabel(f"Frame: 0 / {self.main_reader.total_frames - 1}")
        self.frame_label.setMinimumWidth(150)
        layout.addWidget(self.frame_label)

        return layout

    def load_reference(self):
        """Load reference video for comparison"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Video", "",
            "Raw Video (*.yuv *.raw *.rgb *.y4m);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            # Use selected reference format, or main format if "(Same as source)"
            ref_fmt_name = self.ref_format_combo.currentData()
            if ref_fmt_name is None:
                ref_fmt_name = self.main_reader.format.name

            ref_reader = VideoReader(file_path, self.main_reader.width,
                                     self.main_reader.height, ref_fmt_name)

            # Validate dimensions match
            if (ref_reader.width != self.main_reader.width or
                ref_reader.height != self.main_reader.height):
                logger.warning("Dimension mismatch: reference %dx%d vs main %dx%d",
                               ref_reader.width, ref_reader.height,
                               self.main_reader.width, self.main_reader.height)
                QMessageBox.warning(
                    self, "Dimension Mismatch",
                    f"Reference video dimensions ({ref_reader.width}x{ref_reader.height}) "
                    f"must match main video ({self.main_reader.width}x{self.main_reader.height})"
                )
                return

            # Validate frame counts
            if ref_reader.total_frames != self.main_reader.total_frames:
                response = QMessageBox.question(
                    self, "Frame Count Mismatch",
                    f"Reference video has {ref_reader.total_frames} frames, "
                    f"main video has {self.main_reader.total_frames} frames. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if response != QMessageBox.StandardButton.Yes:
                    return

            self.ref_reader = ref_reader
            logger.debug("Reference file loaded: %s (format=%s, %dx%d, frames=%d)",
                         file_path, ref_reader.format.fourcc,
                         ref_reader.width, ref_reader.height, ref_reader.total_frames)

            # Update slider range to minimum of both videos
            max_frame = min(self.main_reader.total_frames, self.ref_reader.total_frames) - 1
            self.frame_slider.setRange(0, max(0, max_frame))

            # Update display
            self.current_frame = 0
            self.update_frame()

            self.statusBar.showMessage(f"Loaded reference: {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Reference",
                               f"Failed to load reference video:\n{str(e)}")

    def update_frame(self):
        """Update display for current frame"""
        if self.ref_reader is None:
            return

        try:
            # Read frames from both videos
            raw_main = self.main_reader.seek_frame(self.current_frame)
            raw_ref = self.ref_reader.seek_frame(self.current_frame)

            if raw_main is None or raw_ref is None:
                self.statusBar.showMessage("Error reading frames")
                return

            # Store raw data for pixel inspector
            self._raw_main = raw_main
            self._raw_ref = raw_ref

            # Convert to RGB
            img_main = self.main_reader.convert_to_rgb(raw_main)
            img_ref = self.ref_reader.convert_to_rgb(raw_ref)

            # Update canvas based on mode
            mode = self.mode_combo.currentData()
            if mode == ComparisonMode.DIFF:
                diff_result = self._compute_raw_diff(raw_main, raw_ref)
                if diff_result is not None:
                    diff_heatmap, self._diff_stats = diff_result
                else:
                    # Fallback: RGB-based diff
                    diff_heatmap, self._diff_stats = self._compute_rgb_diff(img_main, img_ref)
                self.canvas.set_images(diff_heatmap, diff_heatmap)
            else:
                self._diff_stats = None
                self.canvas.set_images(img_main, img_ref)

            # Calculate metrics (always on RGB)
            psnr = VideoAnalyzer.calculate_psnr(img_main, img_ref)
            ssim = VideoAnalyzer.calculate_ssim(img_main, img_ref)

            # Update labels
            self.psnr_label.setText(f"{psnr:.2f} dB" if psnr != float('inf') else "∞")
            self.ssim_label.setText(f"{ssim:.4f}")

            # Update frame counter
            max_frame = min(self.main_reader.total_frames, self.ref_reader.total_frames) - 1
            self.frame_label.setText(f"Frame: {self.current_frame} / {max_frame}")

            status = (f"Frame {self.current_frame} | "
                      f"Main: {self.main_reader.format.fourcc} "
                      f"{self.main_reader.width}x{self.main_reader.height} | "
                      f"Ref: {self.ref_reader.format.fourcc} "
                      f"{self.ref_reader.width}x{self.ref_reader.height}")
            if self._diff_stats:
                for comp, stats in self._diff_stats.items():
                    status += f" | {comp}: mean={stats['mean']:.2f} max={stats['max']}"
            self.statusBar.showMessage(status)

        except Exception as e:
            self.statusBar.showMessage(f"Error updating frame: {str(e)}")

    def _compute_raw_diff(self, raw_main, raw_ref):
        """Compute diff directly on raw pixel components (Y/U/V or R/G/B).

        Returns (diff_heatmap_rgb, stats_dict) or None if formats unsupported.
        """
        src_fmt = self.main_reader.format
        ref_fmt = self.ref_reader.format
        w, h = self.main_reader.width, self.main_reader.height

        yuv_types = {FormatType.YUV_PLANAR, FormatType.YUV_SEMI_PLANAR, FormatType.YUV_PACKED}

        if src_fmt.type in yuv_types and ref_fmt.type in yuv_types:
            return self._compute_yuv_diff(raw_main, raw_ref, w, h, src_fmt, ref_fmt)

        if src_fmt.type == FormatType.GREY and ref_fmt.type == FormatType.GREY:
            return self._compute_grey_diff(raw_main, raw_ref, w, h)

        # Mixed or unsupported types: fall back to RGB
        return None

    def _compute_yuv_diff(self, raw_main, raw_ref, w, h, src_fmt, ref_fmt):
        """Compare Y/U/V planes directly between two YUV frames."""
        y1, u1, v1 = VideoConverter._extract_yuv_planes(raw_main, w, h, src_fmt)
        y2, u2, v2 = VideoConverter._extract_yuv_planes(raw_ref, w, h, ref_fmt)

        if y1 is None or y2 is None:
            return None

        # Resample chroma to luma resolution for unified comparison
        u1_full = cv2.resize(u1, (w, h), interpolation=cv2.INTER_NEAREST) if u1.shape != (h, w) else u1
        v1_full = cv2.resize(v1, (w, h), interpolation=cv2.INTER_NEAREST) if v1.shape != (h, w) else v1
        u2_full = cv2.resize(u2, (w, h), interpolation=cv2.INTER_NEAREST) if u2.shape != (h, w) else u2
        v2_full = cv2.resize(v2, (w, h), interpolation=cv2.INTER_NEAREST) if v2.shape != (h, w) else v2

        # Per-component absolute diff
        y_diff = np.abs(y1.astype(np.int16) - y2.astype(np.int16)).astype(np.uint8)
        u_diff = np.abs(u1_full.astype(np.int16) - u2_full.astype(np.int16)).astype(np.uint8)
        v_diff = np.abs(v1_full.astype(np.int16) - v2_full.astype(np.int16)).astype(np.uint8)

        stats = {
            'Y': {'mean': float(np.mean(y_diff)), 'max': int(np.max(y_diff))},
            'U': {'mean': float(np.mean(u_diff)), 'max': int(np.max(u_diff))},
            'V': {'mean': float(np.mean(v_diff)), 'max': int(np.max(v_diff))},
        }

        # Combined diff: max across components per pixel
        combined = np.maximum(np.maximum(y_diff, u_diff), v_diff)

        # Create color-coded heatmap: Y=green, U=blue, V=red
        if int(np.max(combined)) == 0:
            heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            amplified_y = np.clip(y_diff.astype(np.float32) * 10, 0, 255).astype(np.uint8)
            amplified_u = np.clip(u_diff.astype(np.float32) * 10, 0, 255).astype(np.uint8)
            amplified_v = np.clip(v_diff.astype(np.float32) * 10, 0, 255).astype(np.uint8)
            # RGB: R=V_diff, G=Y_diff, B=U_diff
            heatmap = np.stack([amplified_v, amplified_y, amplified_u], axis=2)

        return heatmap, stats

    def _compute_grey_diff(self, raw_main, raw_ref, w, h):
        """Compare single-channel grey frames directly."""
        g1 = np.frombuffer(raw_main, dtype=np.uint8, count=w * h).reshape(h, w)
        g2 = np.frombuffer(raw_ref, dtype=np.uint8, count=w * h).reshape(h, w)
        g_diff = np.abs(g1.astype(np.int16) - g2.astype(np.int16)).astype(np.uint8)

        stats = {'Y': {'mean': float(np.mean(g_diff)), 'max': int(np.max(g_diff))}}

        if int(np.max(g_diff)) == 0:
            heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            amplified = np.clip(g_diff.astype(np.float32) * 10, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(amplified, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap[g_diff == 0] = 0

        return heatmap, stats

    @staticmethod
    def _compute_rgb_diff(img_main, img_ref):
        """Fallback: compute diff on RGB-converted images."""
        diff = np.abs(img_main.astype(np.int16) - img_ref.astype(np.int16)).astype(np.uint8)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        diff_max = int(np.max(diff_gray))

        stats = {
            'R': {'mean': float(np.mean(diff[:, :, 0])), 'max': int(np.max(diff[:, :, 0]))},
            'G': {'mean': float(np.mean(diff[:, :, 1])), 'max': int(np.max(diff[:, :, 1]))},
            'B': {'mean': float(np.mean(diff[:, :, 2])), 'max': int(np.max(diff[:, :, 2]))},
        }

        if diff_max == 0:
            heatmap = np.zeros_like(img_main)
        else:
            amplified = np.clip(diff_gray.astype(np.float32) * 10, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(amplified, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap[diff_gray == 0] = 0

        return heatmap, stats

    def _setup_inspector(self):
        """Create pixel inspector dock widget."""
        dock = QDockWidget("Pixel Inspector", self)
        dock.setObjectName("comparison_pixel_inspector")
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea)
        dock.setMinimumWidth(280)

        panel = QWidget()
        panel.setMinimumWidth(260)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        mono = "font-family: monospace; font-size: 12px;"

        # Coordinate
        self.insp_coord = QLabel("Pixel: -")
        self.insp_coord.setStyleSheet("font-weight: bold;")
        self.insp_coord.setMinimumHeight(20)
        layout.addWidget(self.insp_coord)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep1)

        # Source values
        self.insp_src_label = QLabel("Source:")
        self.insp_src_label.setStyleSheet("font-weight: bold; color: #4488ff;")
        layout.addWidget(self.insp_src_label)
        self.insp_src_comp = QLabel("-")
        self.insp_src_comp.setStyleSheet(mono)
        self.insp_src_comp.setMinimumHeight(18)
        layout.addWidget(self.insp_src_comp)

        # Reference values
        self.insp_ref_label = QLabel("Reference:")
        self.insp_ref_label.setStyleSheet("font-weight: bold; color: #44ff88;")
        layout.addWidget(self.insp_ref_label)
        self.insp_ref_comp = QLabel("-")
        self.insp_ref_comp.setStyleSheet(mono)
        self.insp_ref_comp.setMinimumHeight(18)
        layout.addWidget(self.insp_ref_comp)

        # Diff values
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep2)
        self.insp_diff_label = QLabel("Diff:")
        self.insp_diff_label.setStyleSheet("font-weight: bold; color: #ff8844;")
        layout.addWidget(self.insp_diff_label)
        self.insp_diff_comp = QLabel("-")
        self.insp_diff_comp.setStyleSheet(mono)
        self.insp_diff_comp.setTextFormat(Qt.TextFormat.RichText)
        self.insp_diff_comp.setMinimumHeight(18)
        layout.addWidget(self.insp_diff_comp)

        # Neighborhood: Source
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep3)
        self.insp_src_nb = QLabel("Src NB:")
        self.insp_src_nb.setTextFormat(Qt.TextFormat.RichText)
        self.insp_src_nb.setStyleSheet(mono)
        self.insp_src_nb.setMinimumHeight(60)
        layout.addWidget(self.insp_src_nb)

        # Neighborhood: Reference
        self.insp_ref_nb = QLabel("Ref NB:")
        self.insp_ref_nb.setTextFormat(Qt.TextFormat.RichText)
        self.insp_ref_nb.setStyleSheet(mono)
        self.insp_ref_nb.setMinimumHeight(60)
        layout.addWidget(self.insp_ref_nb)

        layout.addStretch()
        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # Connect canvas mouse signal
        self.canvas.mouse_moved.connect(self._update_inspector)

    def _update_inspector(self, x, y):
        """Update pixel inspector with source/ref/diff info at (x, y)."""
        if self.ref_reader is None:
            return

        try:
            if not hasattr(self, '_raw_main') or self._raw_main is None:
                return

            src_info = self.main_reader.get_pixel_info(self._raw_main, x, y)
            ref_info = self.ref_reader.get_pixel_info(self._raw_ref, x, y)

            if not src_info or not ref_info:
                return

            self.insp_coord.setText(f"Pixel: ({x}, {y})")

            # Source - components in hex
            src_comps = src_info['components']
            self.insp_src_label.setText(f"Source [{self.main_reader.format.fourcc}]:")
            self.insp_src_comp.setText(
                "  ".join(f"{k}=0x{v:02X}" for k, v in src_comps.items()))

            # Reference - components in hex
            ref_comps = ref_info['components']
            self.insp_ref_label.setText(f"Ref [{self.ref_reader.format.fourcc}]:")
            self.insp_ref_comp.setText(
                "  ".join(f"{k}=0x{v:02X}" for k, v in ref_comps.items()))

            # Diff per component (hex), color-coded
            diff_parts = []
            for k in src_comps:
                if k in ref_comps:
                    d = src_comps[k] - ref_comps[k]
                    ad = abs(d)
                    sign = "+" if d > 0 else ("-" if d < 0 else " ")
                    color = "#ff4444" if ad > 0 else "#44ff44"
                    diff_parts.append(
                        f'<span style="color:{color}">'
                        f'{k}={sign}0x{ad:02X}</span>')
            self.insp_diff_comp.setText("  ".join(diff_parts) if diff_parts else "-")

            # Neighborhoods
            self.insp_src_nb.setText(self._format_neighborhood("Src NB:", src_info['neighborhood']))
            self.insp_ref_nb.setText(self._format_neighborhood("Ref NB:", ref_info['neighborhood']))

        except Exception:
            pass

    @staticmethod
    def _format_neighborhood(title, nb):
        """Format neighborhood grid as HTML."""
        html = f'<span style="font-family:monospace;">{title}<br>'
        for r_idx, row in enumerate(nb):
            cells = []
            for c_idx, val in enumerate(row):
                if r_idx == 1 and c_idx == 1:
                    cells.append(
                        f'<span style="background-color:#ff4444;'
                        f'color:white;font-weight:bold;'
                        f'padding:1px 3px;">{val}</span>')
                else:
                    cells.append(f'<span style="padding:1px 3px;">{val}</span>')
            html += ' '.join(cells) + '<br>'
        html += '</span>'
        return html

    def _on_mode_changed(self, index):
        """Handle mode change"""
        mode = self.mode_combo.currentData()
        logger.debug("Comparison mode switched to: %s", mode.value if mode else mode)
        self.canvas.set_mode(mode)

        # Enable/disable opacity slider based on mode
        is_overlay = mode == ComparisonMode.OVERLAY
        self.opacity_slider.setEnabled(is_overlay)
        self.opacity_label.setEnabled(is_overlay)

        # Refresh display
        if self.ref_reader is not None:
            self.update_frame()

    def _on_opacity_changed(self, value):
        """Handle opacity slider change"""
        opacity = value / 100.0
        self.canvas.set_overlay_opacity(opacity)
        self.opacity_label.setText(f"{value}%")

    def _on_slider_changed(self, value):
        """Handle frame slider change"""
        self.current_frame = value
        self.update_frame()

    def _prev_frame(self):
        """Go to previous frame"""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.setValue(self.current_frame)

    def _next_frame(self):
        """Go to next frame"""
        max_frame = self.frame_slider.maximum()
        if self.current_frame < max_frame:
            self.current_frame += 1
            self.frame_slider.setValue(self.current_frame)
