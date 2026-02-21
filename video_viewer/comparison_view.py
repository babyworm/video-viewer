"""
A/B Comparison View for Video Viewer
Provides side-by-side, overlay, and diff comparison modes for two videos.
"""

from enum import Enum
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPoint, QRect, Signal
from PySide6.QtGui import QPainter, QImage, QPixmap, QPen, QColor, QWheelEvent, QMouseEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QSlider, QLabel, QPushButton, QFileDialog,
    QMessageBox, QToolBar, QStatusBar
)

from .video_reader import VideoReader
from .analysis import VideoAnalyzer


class ComparisonMode(Enum):
    """Comparison display modes"""
    SPLIT = "Split View"
    OVERLAY = "Overlay"
    DIFF = "Difference"


class ComparisonCanvas(QWidget):
    """Canvas that renders two images in split/overlay/diff mode"""

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
        elif self.mode == ComparisonMode.SPLIT:
            # Update cursor when near divider
            if self._is_near_divider(event.pos()):
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

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
        """Render split view mode"""
        divider_x = int(self.width() * self.split_position)

        # Draw left side (image A)
        src_rect_a = QRect(0, 0, int(self.image_a.width() * self.split_position), self.image_a.height())
        dst_rect_a = QRect(x, y, int(w * self.split_position), h)
        if dst_rect_a.right() > 0 and dst_rect_a.left() < divider_x:
            # Clip to divider
            dst_rect_a.setRight(min(dst_rect_a.right(), divider_x))
            src_w = int(src_rect_a.width() * (dst_rect_a.width() / (w * self.split_position)))
            src_rect_a.setWidth(src_w)
            painter.drawImage(dst_rect_a, self.image_a, src_rect_a)

        # Draw right side (image B)
        src_split = self.split_position
        src_rect_b = QRect(int(self.image_b.width() * src_split), 0,
                          int(self.image_b.width() * (1 - src_split)), self.image_b.height())
        dst_rect_b = QRect(divider_x, y, int(w * (1 - self.split_position)), h)
        if dst_rect_b.left() < self.width() and dst_rect_b.right() > divider_x:
            # Clip to divider
            dst_rect_b.setLeft(max(dst_rect_b.left(), divider_x))
            src_w = int(src_rect_b.width() * (dst_rect_b.width() / (w * (1 - self.split_position))))
            src_rect_b.setWidth(src_w)
            painter.drawImage(dst_rect_b, self.image_b, src_rect_b)

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


class ComparisonWindow(QMainWindow):
    """Standalone window for A/B video comparison"""

    def __init__(self, main_reader: VideoReader, parent=None):
        super().__init__(parent)

        # Video readers
        self.main_reader = main_reader
        self.ref_reader: Optional[VideoReader] = None

        # Current frame index
        self.current_frame = 0
        self._diff_mean = None
        self._diff_max = None

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
            # Create reader for reference video with same parameters as main
            fmt = self.main_reader.format

            ref_reader = VideoReader(file_path, self.main_reader.width,
                                     self.main_reader.height, fmt.name)

            # Validate dimensions match
            if (ref_reader.width != self.main_reader.width or
                ref_reader.height != self.main_reader.height):
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

            # Convert to RGB
            img_main = self.main_reader.convert_to_rgb(raw_main)
            img_ref = self.ref_reader.convert_to_rgb(raw_ref)

            # Update canvas based on mode
            mode = self.mode_combo.currentData()
            if mode == ComparisonMode.DIFF:
                # Compute difference
                diff = np.abs(img_main.astype(np.float32) - img_ref.astype(np.float32))
                diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                self._diff_mean = float(np.mean(diff_gray))
                self._diff_max = int(np.max(diff_gray))

                if self._diff_max == 0:
                    # Identical frames: pure black
                    diff_heatmap = np.zeros_like(img_main)
                else:
                    # Amplify 10x for visibility, clamp to 255
                    amplified = np.clip(diff_gray.astype(np.float32) * 10, 0, 255).astype(np.uint8)
                    diff_heatmap = cv2.applyColorMap(amplified, cv2.COLORMAP_JET)
                    diff_heatmap = cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB)
                    # Make zero-diff pixels black
                    mask = diff_gray == 0
                    diff_heatmap[mask] = 0

                self.canvas.set_images(diff_heatmap, diff_heatmap)
            else:
                self._diff_mean = None
                self._diff_max = None
                self.canvas.set_images(img_main, img_ref)

            # Calculate metrics
            psnr = VideoAnalyzer.calculate_psnr(img_main, img_ref)
            ssim = VideoAnalyzer.calculate_ssim(img_main, img_ref)

            # Update labels
            self.psnr_label.setText(f"{psnr:.2f} dB" if psnr != float('inf') else "∞")
            self.ssim_label.setText(f"{ssim:.4f}")

            # Update frame counter
            max_frame = min(self.main_reader.total_frames, self.ref_reader.total_frames) - 1
            self.frame_label.setText(f"Frame: {self.current_frame} / {max_frame}")

            status = (f"Frame {self.current_frame} | "
                      f"Main: {self.main_reader.width}x{self.main_reader.height} | "
                      f"Ref: {self.ref_reader.width}x{self.ref_reader.height}")
            if self._diff_mean is not None:
                status += f" | Diff mean={self._diff_mean:.2f} max={self._diff_max}"
            self.statusBar.showMessage(status)

        except Exception as e:
            self.statusBar.showMessage(f"Error updating frame: {str(e)}")

    def _on_mode_changed(self, index):
        """Handle mode change"""
        mode = self.mode_combo.currentData()
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
