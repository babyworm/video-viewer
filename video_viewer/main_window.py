import os
import logging
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QComboBox, QPushButton, QSlider,
                             QFileDialog, QGroupBox, QMenuBar, QStatusBar,
                             QDialog, QDialogButtonBox, QFormLayout, QMenu,
                             QToolBar, QStyle, QApplication, QTableWidget,
                             QTableWidgetItem, QHeaderView, QProgressDialog,
                             QSpinBox, QCheckBox, QListWidget, QListWidgetItem,
                             QSizePolicy, QTabWidget)
from PySide6.QtCore import Qt, QTimer, Signal, QSettings, QMimeData, QUrl
from PySide6.QtGui import (QImage, QPixmap, QPainter, QColor, QPen, QMouseEvent,
                          QAction, QKeySequence, QDragEnterEvent, QDropEvent,
                          QClipboard, QIcon)
import numpy as np
import cv2

logger = logging.getLogger(__name__)

from .video_reader import VideoReader
from .format_manager import FormatManager
from .analysis import VideoAnalyzer
from .comparison_view import ComparisonWindow

import pyqtgraph as pg

# Common resolutions for raw file size guessing (most likely first)
COMMON_RESOLUTIONS = [
    (3840, 2160),  # 4K UHD
    (2560, 1440),  # QHD
    (1920, 1080),  # FHD
    (1280, 720),   # HD
    (720, 576),    # PAL SD
    (720, 480),    # NTSC SD
    (640, 480),    # VGA
    (352, 288),    # CIF
    (320, 240),    # QVGA
    (176, 144),    # QCIF
]

# Format short names to try when guessing (most common first)
COMMON_GUESS_FORMATS = ["I420", "NV12", "YUYV", "RGB24"]


class MarkerSlider(QSlider):
    """QSlider with colored marker lines for scene changes and bookmarks."""

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._scene_markers = []   # list of frame indices
        self._bookmark_markers = []  # list of frame indices

    def set_scene_markers(self, markers):
        self._scene_markers = list(markers)
        self.update()

    def set_bookmark_markers(self, markers):
        self._bookmark_markers = list(markers)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.maximum() == 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate groove geometry
        handle_width = 14
        groove_start = handle_width // 2
        groove_width = self.width() - handle_width
        max_val = self.maximum()

        # Draw scene change markers (red)
        if self._scene_markers:
            painter.setPen(QPen(QColor(255, 60, 60, 200), 2))
            for idx in self._scene_markers:
                x = groove_start + int(idx / max_val * groove_width)
                painter.drawLine(x, 0, x, self.height())

        # Draw bookmark markers (cyan)
        if self._bookmark_markers:
            painter.setPen(QPen(QColor(0, 200, 255, 200), 2))
            for idx in self._bookmark_markers:
                x = groove_start + int(idx / max_val * groove_width)
                painter.drawLine(x, 0, x, self.height())

        painter.end()


class ImageCanvas(QWidget):
    mouse_moved = Signal(int, int) # Signal for coordinates
    roi_selected = Signal(int, int, int, int)  # x, y, w, h in image coords

    def __init__(self):
        super().__init__()
        self.image = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.grid_size = 0  # 0 means no grid
        self.sub_grid_size = 0  # 0=none, 2/3/4 = pixel interval
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.last_mouse_pos = None
        # ROI selection state
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None  # (x, y, w, h) in image coords

    def wheelEvent(self, event):
        if not self.image:
             return

        # Zoom centered on mouse cursor
        zoom_in = event.angleDelta().y() > 0
        scale_mult = 1.1 if zoom_in else 1/1.1

        old_scale = self.scale_factor
        new_scale = old_scale * scale_mult

        # Limit zoom? Maybe reasonable bounds
        if new_scale < 0.1: new_scale = 0.1
        if new_scale > 50.0: new_scale = 50.0

        self.scale_factor = new_scale

        # Adjust offset to keep mouse position stable
        # Mouse pos in widget coords
        mx = event.position().x()
        my = event.position().y()

        # Pos in image coords (before zoom)
        # x = (mx - offset_x) / old_scale
        # after zoom: mx = new_scale * x + new_offset_x
        # new_offset_x = mx - new_scale * x

        img_x = (mx - self.offset_x) / old_scale
        img_y = (my - self.offset_y) / old_scale

        self.offset_x = mx - new_scale * img_x
        self.offset_y = my - new_scale * img_y

        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self.image:
            # Start ROI selection
            ix = int((event.position().x() - self.offset_x) / self.scale_factor)
            iy = int((event.position().y() - self.offset_y) / self.scale_factor)
            self.roi_start = (ix, iy)
            self.roi_end = (ix, iy)
            return
        if event.button() == Qt.MouseButton.LeftButton or event.button() == Qt.MouseButton.MiddleButton:
            self.last_mouse_pos = event.position()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self.roi_start:
            ix = int((event.position().x() - self.offset_x) / self.scale_factor)
            iy = int((event.position().y() - self.offset_y) / self.scale_factor)
            self.roi_end = (ix, iy)
            x1 = max(0, min(self.roi_start[0], self.roi_end[0]))
            y1 = max(0, min(self.roi_start[1], self.roi_end[1]))
            x2 = min(self.image.width(), max(self.roi_start[0], self.roi_end[0]))
            y2 = min(self.image.height(), max(self.roi_start[1], self.roi_end[1]))
            w, h = x2 - x1, y2 - y1
            if w > 2 and h > 2:
                self.roi_rect = (x1, y1, w, h)
                self.roi_selected.emit(x1, y1, w, h)
            else:
                self.roi_rect = None
            self.roi_start = None
            self.update()
            return
        self.last_mouse_pos = None

    def mouseMoveEvent(self, event: QMouseEvent):
        # ROI drag
        if self.roi_start is not None and self.image:
            ix = int((event.position().x() - self.offset_x) / self.scale_factor)
            iy = int((event.position().y() - self.offset_y) / self.scale_factor)
            self.roi_end = (ix, iy)
            self.update()
            return

        # Pan
        if self.last_mouse_pos is not None:
             delta = event.position() - self.last_mouse_pos
             self.offset_x += delta.x()
             self.offset_y += delta.y()
             self.last_mouse_pos = event.position()
             self.update()
             return

        if self.image:
             # Calculate image coordinates with current transform
             # screen_x = img_x * scale + offset_x
             # img_x = (screen_x - offset_x) / scale

             x = int((event.position().x() - self.offset_x) / self.scale_factor)
             y = int((event.position().y() - self.offset_y) / self.scale_factor)

             # Bounds check
             if 0 <= x < self.image.width() and 0 <= y < self.image.height():
                 self.mouse_moved.emit(x, y)

    def set_image(self, cv_image):
        if cv_image is None:
            self.image = None
        else:
            height, width, channel = cv_image.shape
            bytes_per_line = 3 * width
            self.image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

            # Reset view on new image? No, keep user's zoom/pan if dimensions match?
            # Or reset? Usually reset if dimensions change significantly
            # Let's reset if first load
            if self.scale_factor == 1.0 and self.offset_x == 0 and self.offset_y == 0:
                 self.fit_to_view()

        self.update()

    def fit_to_view(self):
        if not self.image: return
        # Calculate scale to fit
        w_ratio = self.width() / self.image.width()
        h_ratio = self.height() / self.image.height()
        self.scale_factor = min(w_ratio, h_ratio) * 0.9 # 90% fit

        # Center
        scaled_w = self.image.width() * self.scale_factor
        scaled_h = self.image.height() * self.scale_factor
        self.offset_x = (self.width() - scaled_w) / 2
        self.offset_y = (self.height() - scaled_h) / 2
        self.update()

    def fit_to_width(self):
        if not self.image: return
        self.scale_factor = (self.width() / self.image.width()) * 0.95 # 95% fit

        # Center X, Top Y (or center Y?) - Let's center both
        scaled_w = self.image.width() * self.scale_factor
        scaled_h = self.image.height() * self.scale_factor
        self.offset_x = (self.width() - scaled_w) / 2
        self.offset_y = (self.height() - scaled_h) / 2
        self.update()

    def set_scale(self, scale):
        if not self.image: return
        self.scale_factor = scale

        # Center
        scaled_w = self.image.width() * self.scale_factor
        scaled_h = self.image.height() * self.scale_factor
        self.offset_x = (self.width() - scaled_w) / 2
        self.offset_y = (self.height() - scaled_h) / 2
        self.update()

    def set_grid(self, size):
        self.grid_size = size
        self.update()

    def set_sub_grid(self, size):
        self.sub_grid_size = size
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        if self.image:
            # We are manually managing pan/zoom, so we use coordinate transform
            painter.save()
            painter.translate(self.offset_x, self.offset_y)
            painter.scale(self.scale_factor, self.scale_factor)

            # Draw Image
            painter.drawImage(0, 0, self.image)

            # Draw Grid
            if self.grid_size > 0:
                # Use CosmicPen for grid - well, keeping it simple green dotted for now
                pen = QPen(QColor(0, 255, 0), 1 / self.scale_factor, Qt.PenStyle.DotLine) # constant width on screen
                painter.setPen(pen)

                w = self.image.width()
                h = self.image.height()

                # Vertical lines
                for x in range(0, w, self.grid_size):
                    painter.drawLine(x, 0, x, h)
                painter.drawLine(w, 0, w, h) # Last line

                # Horizontal lines
                for y in range(0, h, self.grid_size):
                    painter.drawLine(0, y, w, y)
                painter.drawLine(0, h, w, h) # Last line

            # Draw Sub-grid
            if self.sub_grid_size > 0:
                pen = QPen(QColor(255, 255, 0, 128), 1 / self.scale_factor, Qt.PenStyle.DotLine)
                painter.setPen(pen)
                w = self.image.width()
                h = self.image.height()
                for x in range(0, w + 1, self.sub_grid_size):
                    painter.drawLine(x, 0, x, h)
                for y in range(0, h + 1, self.sub_grid_size):
                    painter.drawLine(0, y, w, y)

            # Draw ROI rectangle
            if self.roi_rect:
                pen = QPen(QColor(0, 200, 255), 2 / self.scale_factor, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                rx, ry, rw, rh = self.roi_rect
                painter.drawRect(rx, ry, rw, rh)
            elif self.roi_start and self.roi_end:
                pen = QPen(QColor(0, 200, 255), 1 / self.scale_factor, Qt.PenStyle.DotLine)
                painter.setPen(pen)
                x1 = min(self.roi_start[0], self.roi_end[0])
                y1 = min(self.roi_start[1], self.roi_end[1])
                w = abs(self.roi_end[0] - self.roi_start[0])
                h = abs(self.roi_end[1] - self.roi_start[1])
                painter.drawRect(x1, y1, w, h)

            painter.restore()


class ParametersDialog(QDialog):
    """Dialog for setting width, height, and format parameters."""

    def __init__(self, parent=None, format_manager=None, current_width=1920, current_height=1080, current_format="I420", guess_info=None):
        super().__init__(parent)
        self.setWindowTitle("Video Parameters")
        self.format_manager = format_manager or FormatManager()

        layout = QFormLayout(self)

        # Guess info hint
        if guess_info:
            self.lbl_guess = QLabel(guess_info)
            self.lbl_guess.setStyleSheet("color: #2196F3; font-weight: bold; padding: 4px;")
            self.lbl_guess.setWordWrap(True)
            layout.addRow(self.lbl_guess)

        # Width
        self.txt_width = QLineEdit(str(current_width))
        layout.addRow("Width:", self.txt_width)

        # Height
        self.txt_height = QLineEdit(str(current_height))
        layout.addRow("Height:", self.txt_height)

        # Format
        self.combo_format = QComboBox()
        self.combo_format.addItems(self.format_manager.get_supported_formats())
        if current_format in self.format_manager.get_supported_formats():
            self.combo_format.setCurrentText(current_format)
        layout.addRow("Format:", self.combo_format)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_parameters(self):
        """Returns (width, height, format_name)"""
        try:
            width = int(self.txt_width.text())
            height = int(self.txt_height.text())
            format_name = self.combo_format.currentText()
            return width, height, format_name
        except ValueError:
            return None, None, None


class ExportDialog(QDialog):
    """Dialog for exporting video clips."""

    def __init__(self, parent=None, total_frames=0, current_frame=0):
        super().__init__(parent)
        self.setWindowTitle("Export Clip")

        layout = QFormLayout(self)

        # Start frame
        self.txt_start = QLineEdit(str(current_frame))
        layout.addRow("Start Frame:", self.txt_start)

        # End frame
        self.txt_end = QLineEdit(str(min(current_frame + 100, total_frames - 1)))
        layout.addRow("End Frame:", self.txt_end)

        # Export format
        self.combo_export_fmt = QComboBox()
        self.exportable_formats = [
            "I420 (4:2:0) [YU12]",
            "NV12 (4:2:0) [NV12]",
            "YUYV (4:2:2) [YUYV]",
            "RGB24 (24-bit) [RGB3]",
            "BGR24 (24-bit) [BGR3]"
        ]
        self.combo_export_fmt.addItems(self.exportable_formats)
        layout.addRow("Export Format:", self.combo_export_fmt)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_export_settings(self):
        """Returns (start_frame, end_frame, fourcc) or (None, None, None)"""
        try:
            start = int(self.txt_start.text())
            end = int(self.txt_end.text())
            fmt_str = self.combo_export_fmt.currentText()

            # Extract FourCC
            if "[" in fmt_str and "]" in fmt_str:
                fourcc = fmt_str.split("[")[1].replace("]", "")
            else:
                return None, None, None

            return start, end, fourcc
        except ValueError:
            return None, None, None


class ConvertDialog(QDialog):
    """Dialog for converting video format."""

    def __init__(self, parent=None, format_manager=None, current_width=1920,
                 current_height=1080, current_format="I420"):
        super().__init__(parent)
        self.setWindowTitle("Convert Format")
        self.format_manager = format_manager or FormatManager()

        layout = QFormLayout(self)

        # Input info (read-only)
        self.lbl_input = QLabel("(no file loaded)")
        layout.addRow("Input:", self.lbl_input)

        self.lbl_input_fmt = QLabel(f"{current_width}x{current_height} {current_format}")
        layout.addRow("Source:", self.lbl_input_fmt)

        # Output format
        self.combo_output_fmt = QComboBox()
        self.combo_output_fmt.addItems(self.format_manager.get_supported_formats())
        layout.addRow("Output Format:", self.combo_output_fmt)

        # Output path
        output_row = QHBoxLayout()
        self.txt_output = QLineEdit()
        self.txt_output.setPlaceholderText("Select output file...")
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_output)
        output_row.addWidget(self.txt_output, stretch=1)
        output_row.addWidget(btn_browse)
        layout.addRow("Output File:", output_row)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def set_input_info(self, file_path, width, height, fmt_name):
        self.lbl_input.setText(os.path.basename(file_path))
        self.lbl_input_fmt.setText(f"{width}x{height} {fmt_name}")

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Converted File", "",
            "Raw Video (*.yuv *.raw *.rgb);;All Files (*)")
        if path:
            self.txt_output.setText(path)

    def get_settings(self):
        """Returns (output_path, output_format) or (None, None)."""
        output_path = self.txt_output.text().strip()
        output_fmt = self.combo_output_fmt.currentText()
        if output_path:
            return output_path, output_fmt
        return None, None


class ShortcutsDialog(QDialog):
    """Dialog showing all keyboard shortcuts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(450, 400)

        layout = QVBoxLayout(self)
        table = QTableWidget()
        shortcuts = [
            ("Space", "Play / Pause"),
            ("Left", "Previous frame"),
            ("Right", "Next frame"),
            ("Ctrl+C", "Copy frame to clipboard"),
            ("Ctrl+O", "Open file"),
            ("Ctrl+S", "Save frame"),
            ("Ctrl+E", "Export clip"),
            ("Ctrl+Q", "Exit"),
            ("F", "Fit to view"),
            ("1", "Zoom 1:1"),
            ("G", "Toggle grid"),
            ("B", "Toggle bookmark"),
            ("Ctrl+Right", "Next scene change"),
            ("Ctrl+Left", "Prev scene change"),
            ("Ctrl+B", "Next bookmark"),
            ("Ctrl+Shift+B", "Prev bookmark"),
            ("Right-click drag", "Select ROI"),
        ]
        table.setRowCount(len(shortcuts))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Shortcut", "Action"])
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for i, (key, desc) in enumerate(shortcuts):
            table.setItem(i, 0, QTableWidgetItem(key))
            table.setItem(i, 1, QTableWidgetItem(desc))
        layout.addWidget(table)

        btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn.accepted.connect(self.accept)
        layout.addWidget(btn)


class BatchConvertDialog(QDialog):
    """Dialog for batch format conversion."""

    def __init__(self, parent=None, format_manager=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Convert")
        self.resize(500, 400)
        self.format_manager = format_manager or FormatManager()
        self.file_list = []

        layout = QVBoxLayout(self)

        # File list
        layout.addWidget(QLabel("Input Files:"))
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add Files...")
        btn_add.clicked.connect(self._add_files)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_files)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_clear)
        layout.addLayout(btn_row)

        # Parameters
        form = QFormLayout()
        self.txt_width = QLineEdit("1920")
        form.addRow("Width:", self.txt_width)
        self.txt_height = QLineEdit("1080")
        form.addRow("Height:", self.txt_height)
        self.combo_input_fmt = QComboBox()
        self.combo_input_fmt.addItems(self.format_manager.get_supported_formats())
        form.addRow("Input Format:", self.combo_input_fmt)
        self.combo_output_fmt = QComboBox()
        self.combo_output_fmt.addItems(self.format_manager.get_supported_formats())
        form.addRow("Output Format:", self.combo_output_fmt)
        layout.addLayout(form)

        # Output dir
        dir_row = QHBoxLayout()
        self.txt_output_dir = QLineEdit()
        self.txt_output_dir.setPlaceholderText("Output directory...")
        btn_dir = QPushButton("Browse...")
        btn_dir.clicked.connect(self._browse_dir)
        dir_row.addWidget(self.txt_output_dir, stretch=1)
        dir_row.addWidget(btn_dir)
        form.addRow("Output Dir:", dir_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "",
            "Raw Video (*.yuv *.raw *.rgb *.y4m);;All Files (*)")
        for f in files:
            self.file_list.append(f)
            self.list_widget.addItem(os.path.basename(f))

    def _clear_files(self):
        self.file_list.clear()
        self.list_widget.clear()

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.txt_output_dir.setText(d)

    def get_settings(self):
        try:
            w = int(self.txt_width.text())
            h = int(self.txt_height.text())
        except ValueError:
            return None
        return {
            "files": self.file_list,
            "width": w, "height": h,
            "input_fmt": self.combo_input_fmt.currentText(),
            "output_fmt": self.combo_output_fmt.currentText(),
            "output_dir": self.txt_output_dir.text().strip(),
        }


class PngExportDialog(QDialog):
    """Dialog for PNG sequence export."""

    def __init__(self, parent=None, total_frames=0, current_frame=0):
        super().__init__(parent)
        self.setWindowTitle("Export PNG Sequence")

        layout = QFormLayout(self)
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, max(0, total_frames - 1))
        self.spin_start.setValue(current_frame)
        layout.addRow("Start Frame:", self.spin_start)

        self.spin_end = QSpinBox()
        self.spin_end.setRange(0, max(0, total_frames - 1))
        self.spin_end.setValue(min(current_frame + 30, total_frames - 1))
        layout.addRow("End Frame:", self.spin_end)

        dir_row = QHBoxLayout()
        self.txt_dir = QLineEdit()
        self.txt_dir.setPlaceholderText("Output directory...")
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse)
        dir_row.addWidget(self.txt_dir, stretch=1)
        dir_row.addWidget(btn)
        layout.addRow("Directory:", dir_row)

        self.txt_prefix = QLineEdit("frame")
        layout.addRow("Prefix:", self.txt_prefix)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.txt_dir.setText(d)

    def get_settings(self):
        d = self.txt_dir.text().strip()
        if not d:
            return None
        return {
            "start": self.spin_start.value(),
            "end": self.spin_end.value(),
            "directory": d,
            "prefix": self.txt_prefix.text() or "frame",
        }


# Dark theme stylesheet
DARK_STYLE = """
QMainWindow, QDialog { background-color: #2b2b2b; color: #dcdcdc; }
QWidget { background-color: #2b2b2b; color: #dcdcdc; }
QMenuBar { background-color: #3c3f41; color: #dcdcdc; }
QMenuBar::item:selected { background-color: #4b6eaf; }
QMenu { background-color: #3c3f41; color: #dcdcdc; border: 1px solid #555; }
QMenu::item:selected { background-color: #4b6eaf; }
QToolBar { background-color: #3c3f41; border: none; spacing: 3px; }
QStatusBar { background-color: #3c3f41; color: #dcdcdc; }
QGroupBox { border: 1px solid #555; border-radius: 4px; margin-top: 8px;
            padding-top: 8px; color: #dcdcdc; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
QLabel { color: #dcdcdc; }
QLineEdit, QComboBox, QSpinBox { background-color: #45494a; color: #dcdcdc;
    border: 1px solid #646464; border-radius: 3px; padding: 2px; }
QPushButton { background-color: #4b6eaf; color: white; border: none;
    border-radius: 3px; padding: 4px 12px; }
QPushButton:hover { background-color: #5a7fbf; }
QPushButton:pressed { background-color: #3d5a8f; }
QSlider::groove:horizontal { background: #555; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background: #4b6eaf; width: 14px; margin: -4px 0;
    border-radius: 7px; }
QDockWidget { titlebar-close-icon: none; color: #dcdcdc; }
QDockWidget::title { background-color: #3c3f41; padding: 4px; }
QTabWidget::pane { border: 1px solid #555; }
QTabBar::tab { background-color: #3c3f41; color: #dcdcdc; padding: 6px 12px; }
QTabBar::tab:selected { background-color: #4b6eaf; }
QTableWidget { background-color: #2b2b2b; color: #dcdcdc; gridline-color: #555; }
QHeaderView::section { background-color: #3c3f41; color: #dcdcdc; border: 1px solid #555; }
QListWidget { background-color: #2b2b2b; color: #dcdcdc; }
QScrollBar:vertical { background: #2b2b2b; width: 12px; }
QScrollBar::handle:vertical { background: #555; border-radius: 6px; }
QScrollBar:horizontal { background: #2b2b2b; height: 12px; }
QScrollBar::handle:horizontal { background: #555; border-radius: 6px; }
"""


class MainWindow(QMainWindow):
    def __init__(self, file_path=None, width=1920, height=1080, fmt="I420"):
        super().__init__()
        self.setWindowTitle("YUV/RAW Video Viewer")
        self.resize(1200, 800)

        self.reader = None
        self.current_frame_idx = 0
        self.format_manager = FormatManager()

        # Initial params
        self.initial_file = file_path
        self.initial_width = width
        self.initial_height = height
        self.initial_fmt = fmt

        # Current parameters
        self.current_width = width
        self.current_height = height
        self.current_format = fmt

        # Grid state for keyboard toggle
        self.grid_sizes = [0, 16, 32, 64, 128]
        self.current_grid_idx = 0

        # Sub-grid state for keyboard toggle (Shift+G)
        self.sub_grid_sizes = [0, 4, 8, 16]
        self.current_sub_grid_idx = 0

        # Playback state (Phase 2)
        self.is_playing = False
        self.loop_playback = True
        self._programmatic_slider_change = False

        # Bookmarks
        self.bookmarks = set()

        # Scene changes
        self._scene_changes = []

        # Settings
        self._settings = QSettings("VideoViewer", "YUVViewer")

        # Dark theme
        self._dark_theme = self._settings.value("dark_theme", False, type=bool)

        # Accept drops
        self.setAcceptDrops(True)

        self.init_ui()

        # Apply theme
        if self._dark_theme:
            self.apply_dark_theme(True)

        # Restore window state
        self._restore_window_state()

        if self.initial_file:
            self.current_file_path = self.initial_file
            self.reload_video()

    def init_ui(self):
        # Create central widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        outer_layout = QVBoxLayout(main_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Top area: canvas + sidebar side-by-side
        top_widget = QWidget()
        layout = QHBoxLayout(top_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create canvas and navigation toolbar area
        canvas_area = QWidget()
        canvas_layout = QVBoxLayout(canvas_area)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        # Tab bar for multi-file viewing
        self.tab_bar = QTabWidget()
        self.tab_bar.setTabsClosable(True)
        self.tab_bar.setMovable(True)
        self.tab_bar.tabCloseRequested.connect(self._close_tab)
        self.tab_bar.currentChanged.connect(self._on_tab_changed)
        self.tab_bar.setMaximumHeight(30)
        self.tab_bar.setVisible(False)  # Hidden until 2+ tabs
        self._tab_states = []  # List of dicts per tab
        self._current_tab_idx = -1
        self._switching_tab = False  # Guard against recursive signals
        canvas_layout.addWidget(self.tab_bar)

        # Canvas (created before menu bar so menus can reference it)
        self.canvas = ImageCanvas()
        self.canvas.mouse_moved.connect(self.update_inspector)
        self.canvas.roi_selected.connect(self._on_roi_selected)
        canvas_layout.addWidget(self.canvas, stretch=1)

        # Create menu bar (after canvas exists)
        self.create_menu_bar()

        # Icon toolbar below menu bar
        self.create_icon_toolbar()

        # Navigation toolbar below canvas
        nav_toolbar = self.create_navigation_toolbar()
        canvas_layout.addWidget(nav_toolbar)

        layout.addWidget(canvas_area, stretch=1)

        # Right sidebar (250px fixed width)
        sidebar_widget = self.create_sidebar()
        layout.addWidget(sidebar_widget)

        outer_layout.addWidget(top_widget, stretch=1)

        # Console panel between content and status bar (full width, 2 lines)
        console_widget = QWidget()
        console_layout = QHBoxLayout(console_widget)
        console_layout.setContentsMargins(4, 0, 0, 0)
        console_layout.setSpacing(2)
        self.lbl_console = QLabel("")
        self.lbl_console.setFixedHeight(30)
        self.lbl_console.setWordWrap(True)
        self.lbl_console.setStyleSheet("color: #aaccff; font-size: 11px; background-color: #2b2b2b;")
        self.lbl_console.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        console_layout.addWidget(self.lbl_console, stretch=1)
        btn_console_up = QPushButton("\u25b2")
        btn_console_up.setFixedSize(18, 14)
        btn_console_up.setStyleSheet("font-size: 8px; padding: 0px; background-color: #3c3f41; color: #aaa; border: none;")
        btn_console_up.clicked.connect(self._console_scroll_up)
        btn_console_down = QPushButton("\u25bc")
        btn_console_down.setFixedSize(18, 14)
        btn_console_down.setStyleSheet("font-size: 8px; padding: 0px; background-color: #3c3f41; color: #aaa; border: none;")
        btn_console_down.clicked.connect(self._console_scroll_down)
        btn_col = QVBoxLayout()
        btn_col.setContentsMargins(0, 0, 2, 0)
        btn_col.setSpacing(1)
        btn_col.addWidget(btn_console_up)
        btn_col.addWidget(btn_console_down)
        console_layout.addLayout(btn_col)
        console_widget.setFixedHeight(34)
        console_widget.setStyleSheet("background-color: #2b2b2b;")
        self._console_history = []
        self._console_view_idx = -1  # -1 = latest
        outer_layout.addWidget(console_widget)

        # Create status bar
        self.create_status_bar()

        # Analysis Dock
        self.create_analysis_dock()

    def create_menu_bar(self):
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        save_action = QAction("Save Frame", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_frame)
        file_menu.addAction(save_action)

        export_action = QAction("Export Clip", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.show_export_dialog)
        file_menu.addAction(export_action)

        png_export_action = QAction("Export PNG Sequence", self)
        png_export_action.triggered.connect(self.show_png_export_dialog)
        file_menu.addAction(png_export_action)

        file_menu.addSeparator()

        # Recent Files submenu
        self.recent_menu = QMenu("Recent Files", self)
        file_menu.addMenu(self.recent_menu)
        self._update_recent_files_menu()

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View Menu
        view_menu = menubar.addMenu("View")

        # Zoom submenu
        zoom_menu = QMenu("Zoom", self)

        fit_action = QAction("Fit", self)
        fit_action.setShortcut(QKeySequence("F"))
        fit_action.triggered.connect(self.canvas.fit_to_view)
        zoom_menu.addAction(fit_action)

        fit_width_action = QAction("Fit Width", self)
        fit_width_action.triggered.connect(self.canvas.fit_to_width)
        zoom_menu.addAction(fit_width_action)

        zoom_100_action = QAction("1:1", self)
        zoom_100_action.setShortcut(QKeySequence("1"))
        zoom_100_action.triggered.connect(lambda: self.canvas.set_scale(1.0))
        zoom_menu.addAction(zoom_100_action)

        zoom_200_action = QAction("2:1", self)
        zoom_200_action.triggered.connect(lambda: self.canvas.set_scale(2.0))
        zoom_menu.addAction(zoom_200_action)

        view_menu.addMenu(zoom_menu)

        # Grid submenu
        grid_menu = QMenu("Grid", self)

        grid_none_action = QAction("None", self)
        grid_none_action.triggered.connect(lambda: self.set_grid_from_menu(0))
        grid_menu.addAction(grid_none_action)

        grid_16_action = QAction("16x16", self)
        grid_16_action.triggered.connect(lambda: self.set_grid_from_menu(16))
        grid_menu.addAction(grid_16_action)

        grid_32_action = QAction("32x32", self)
        grid_32_action.triggered.connect(lambda: self.set_grid_from_menu(32))
        grid_menu.addAction(grid_32_action)

        grid_64_action = QAction("64x64", self)
        grid_64_action.triggered.connect(lambda: self.set_grid_from_menu(64))
        grid_menu.addAction(grid_64_action)

        grid_128_action = QAction("128x128", self)
        grid_128_action.triggered.connect(lambda: self.set_grid_from_menu(128))
        grid_menu.addAction(grid_128_action)

        grid_menu.addSeparator()

        sub_grid_none_action = QAction("Sub-grid None", self)
        sub_grid_none_action.triggered.connect(lambda: self.set_sub_grid_from_menu(0))
        grid_menu.addAction(sub_grid_none_action)

        sub_grid_4_action = QAction("Sub-grid 4x4", self)
        sub_grid_4_action.triggered.connect(lambda: self.set_sub_grid_from_menu(4))
        grid_menu.addAction(sub_grid_4_action)

        sub_grid_8_action = QAction("Sub-grid 8x8", self)
        sub_grid_8_action.triggered.connect(lambda: self.set_sub_grid_from_menu(8))
        grid_menu.addAction(sub_grid_8_action)

        sub_grid_16_action = QAction("Sub-grid 16x16", self)
        sub_grid_16_action.triggered.connect(lambda: self.set_sub_grid_from_menu(16))
        grid_menu.addAction(sub_grid_16_action)

        view_menu.addMenu(grid_menu)

        # Component submenu
        component_menu = QMenu("Component", self)

        comp_full_action = QAction("Full (0)", self)
        comp_full_action.triggered.connect(lambda: self.set_component(0))
        component_menu.addAction(comp_full_action)

        comp_ch1_action = QAction("Channel 1 Y/R (1)", self)
        comp_ch1_action.triggered.connect(lambda: self.set_component(1))
        component_menu.addAction(comp_ch1_action)

        comp_ch2_action = QAction("Channel 2 U/G (2)", self)
        comp_ch2_action.triggered.connect(lambda: self.set_component(2))
        component_menu.addAction(comp_ch2_action)

        comp_ch3_action = QAction("Channel 3 V/B (3)", self)
        comp_ch3_action.triggered.connect(lambda: self.set_component(3))
        component_menu.addAction(comp_ch3_action)

        comp_split_action = QAction("Split View (4)", self)
        comp_split_action.triggered.connect(lambda: self.set_component(4))
        component_menu.addAction(comp_split_action)

        view_menu.addMenu(component_menu)

        view_menu.addSeparator()

        show_analysis_action = QAction("Show Analysis Dock", self)
        show_analysis_action.triggered.connect(self.toggle_analysis_dock)
        view_menu.addAction(show_analysis_action)

        show_bookmarks_action = QAction("Bookmarks... (B)", self)
        show_bookmarks_action.triggered.connect(self.show_bookmark_dialog)
        view_menu.addAction(show_bookmarks_action)

        view_menu.addSeparator()

        self.dark_theme_action = QAction("Dark Theme", self)
        self.dark_theme_action.setCheckable(True)
        self.dark_theme_action.setChecked(self._dark_theme)
        self.dark_theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(self.dark_theme_action)

        # Tools Menu
        tools_menu = menubar.addMenu("Tools")

        # Analysis submenu
        analysis_menu = QMenu("Analysis", self)

        histogram_action = QAction("Histogram", self)
        histogram_action.triggered.connect(lambda: self.show_analysis_tab(0))
        analysis_menu.addAction(histogram_action)

        vectorscope_action = QAction("Vectorscope", self)
        vectorscope_action.triggered.connect(lambda: self.show_analysis_tab(1))
        analysis_menu.addAction(vectorscope_action)

        metrics_action = QAction("Metrics", self)
        metrics_action.triggered.connect(lambda: self.show_analysis_tab(2))
        analysis_menu.addAction(metrics_action)

        tools_menu.addMenu(analysis_menu)

        # Convert
        convert_action = QAction("Convert", self)
        convert_action.triggered.connect(self.show_convert_dialog)
        tools_menu.addAction(convert_action)

        # Compare (A/B View)
        compare_action = QAction("Compare", self)
        compare_action.triggered.connect(self.show_comparison_window)
        tools_menu.addAction(compare_action)

        batch_convert_action = QAction("Batch Convert", self)
        batch_convert_action.triggered.connect(self.show_batch_convert_dialog)
        tools_menu.addAction(batch_convert_action)

        tools_menu.addSeparator()

        scene_detect_action = QAction("Detect Scene Changes", self)
        scene_detect_action.triggered.connect(self.detect_scene_changes)
        tools_menu.addAction(scene_detect_action)

        tools_menu.addSeparator()

        # Parameters
        params_action = QAction("Parameters", self)
        params_action.triggered.connect(self.show_parameters_dialog)
        tools_menu.addAction(params_action)

        # Color Matrix submenu
        color_menu = QMenu("Color Matrix", self)
        self.act_bt601 = QAction("BT.601 (SD)", self)
        self.act_bt601.setCheckable(True)
        self.act_bt601.setChecked(True)
        self.act_bt601.triggered.connect(lambda: self._set_color_matrix("BT.601"))
        color_menu.addAction(self.act_bt601)

        self.act_bt709 = QAction("BT.709 (HD)", self)
        self.act_bt709.setCheckable(True)
        self.act_bt709.triggered.connect(lambda: self._set_color_matrix("BT.709"))
        color_menu.addAction(self.act_bt709)
        tools_menu.addMenu(color_menu)

        # Help Menu
        help_menu = menubar.addMenu("Help")

        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self.show_shortcuts_dialog)
        help_menu.addAction(shortcuts_action)

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_icon_toolbar(self):
        """Create icon toolbar with common actions."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("MainToolBar")
        toolbar.setMovable(False)
        style = self.style()

        # Open
        act_open = toolbar.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "Open")
        act_open.setToolTip("Open File (Ctrl+O)")
        act_open.triggered.connect(self.open_file)

        # Save As (Convert)
        act_save_as = toolbar.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "Save As")
        act_save_as.setToolTip("Convert / Save As")
        act_save_as.triggered.connect(self.show_convert_dialog)

        toolbar.addSeparator()

        # First Frame
        act_first = toolbar.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward), "First")
        act_first.setToolTip("First Frame")
        act_first.triggered.connect(self.first_frame)

        # Previous Frame
        act_prev = toolbar.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaSeekBackward), "Prev")
        act_prev.setToolTip("Previous Frame (Left)")
        act_prev.triggered.connect(self.prev_frame)

        # Play/Pause
        self.act_play = toolbar.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "Play")
        self.act_play.setToolTip("Play/Pause (Space)")
        self.act_play.triggered.connect(self.toggle_playback)

        # Next Frame
        act_next = toolbar.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward), "Next")
        act_next.setToolTip("Next Frame (Right)")
        act_next.triggered.connect(self.next_frame)

        # Last Frame
        act_last = toolbar.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward), "Last")
        act_last.setToolTip("Last Frame")
        act_last.triggered.connect(self.last_frame)

        toolbar.addSeparator()

        # Fit to View
        act_fit = toolbar.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_TitleBarMaxButton), "Fit")
        act_fit.setToolTip("Fit to View (F)")
        act_fit.triggered.connect(self.canvas.fit_to_view)

        # Zoom 1:1 (100%)
        act_zoom_100 = toolbar.addAction(self._create_zoom_icon("1:1"), "1:1")
        act_zoom_100.setToolTip("Zoom 100% (1:1)")
        act_zoom_100.triggered.connect(lambda: self.canvas.set_scale(1.0))

        # Zoom 2:1 (200%)
        act_zoom_200 = toolbar.addAction(self._create_zoom_icon("2:1"), "2:1")
        act_zoom_200.setToolTip("Zoom 200% (2:1)")
        act_zoom_200.triggered.connect(lambda: self.canvas.set_scale(2.0))

        # Grid Toggle
        act_grid = toolbar.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_FileDialogListView), "Grid")
        act_grid.setToolTip("Toggle Grid (G)")
        act_grid.triggered.connect(self._toggle_grid)

        # Sub-grid Toggle
        act_sub_grid = toolbar.addAction(
            self._create_sub_grid_icon(), "Sub")
        act_sub_grid.setToolTip("Toggle Sub-grid (Shift+G)")
        act_sub_grid.triggered.connect(self._toggle_sub_grid)

        toolbar.addSeparator()

        # Compare
        act_compare = toolbar.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView), "Compare")
        act_compare.setToolTip("A/B Compare View")
        act_compare.triggered.connect(self.show_comparison_window)

        # Convert
        act_convert = toolbar.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "Convert")
        act_convert.setToolTip("Convert Format")
        act_convert.triggered.connect(self.show_convert_dialog)

        # Analysis
        act_analysis = toolbar.addAction(
            style.standardIcon(QStyle.StandardPixmap.SP_FileDialogInfoView), "Analysis")
        act_analysis.setToolTip("Show Analysis Dock")
        act_analysis.triggered.connect(self.toggle_analysis_dock)

        self.addToolBar(toolbar)

    def _create_zoom_icon(self, label):
        """Create a custom zoom icon with text label (e.g. '1:1', '2:1')."""
        size = 24
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(60, 63, 65))
        p = QPainter(pixmap)
        # Draw magnifier circle
        p.setPen(QPen(QColor(180, 200, 220), 1.5))
        p.drawEllipse(2, 2, 16, 16)
        # Draw handle
        p.setPen(QPen(QColor(140, 160, 180), 2))
        p.drawLine(16, 16, 22, 22)
        # Draw label text
        font = p.font()
        font.setPixelSize(8)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QColor(255, 255, 100))
        p.drawText(1, 3, 18, 15, Qt.AlignmentFlag.AlignCenter, label)
        p.end()
        return QIcon(pixmap)

    def _create_sub_grid_icon(self):
        """Create a custom sub-grid icon: small yellow dotted grid on dark background."""
        size = 24
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(60, 63, 65))
        p = QPainter(pixmap)
        # Outer border (green, like main grid)
        p.setPen(QPen(QColor(0, 200, 0, 180), 1))
        p.drawRect(1, 1, size - 3, size - 3)
        # Inner sub-grid lines (yellow dotted)
        pen = QPen(QColor(255, 220, 0, 200), 1, Qt.PenStyle.DotLine)
        p.setPen(pen)
        step = (size - 4) // 4
        for i in range(1, 4):
            x = 2 + i * step
            p.drawLine(x, 2, x, size - 3)
            y = 2 + i * step
            p.drawLine(2, y, size - 3, y)
        p.end()
        return QIcon(pixmap)

    def show_console_message(self, msg):
        """Show a message in the console panel. Persists until next message."""
        self._console_history.append(msg)
        if len(self._console_history) > 200:
            self._console_history = self._console_history[-200:]
        self._console_view_idx = -1
        self.lbl_console.setText(msg)

    def _console_scroll_up(self):
        if not self._console_history:
            return
        if self._console_view_idx == -1:
            self._console_view_idx = len(self._console_history) - 1
        if self._console_view_idx > 0:
            self._console_view_idx -= 1
        self.lbl_console.setText(self._console_history[self._console_view_idx])

    def _console_scroll_down(self):
        if not self._console_history:
            return
        if self._console_view_idx == -1:
            return
        if self._console_view_idx < len(self._console_history) - 1:
            self._console_view_idx += 1
            self.lbl_console.setText(self._console_history[self._console_view_idx])
        else:
            self._console_view_idx = -1
            self.lbl_console.setText(self._console_history[-1])

    def _toggle_grid(self):
        """Toggle grid through sizes (for toolbar button)."""
        self.current_grid_idx = (self.current_grid_idx + 1) % len(self.grid_sizes)
        size = self.grid_sizes[self.current_grid_idx]
        self.canvas.set_grid(size)
        self.show_console_message(f"Grid: {size}x{size}" if size > 0 else "Grid: Off")

    def _toggle_sub_grid(self):
        """Toggle sub-grid through sizes (for toolbar button)."""
        self.current_sub_grid_idx = (self.current_sub_grid_idx + 1) % len(self.sub_grid_sizes)
        size = self.sub_grid_sizes[self.current_sub_grid_idx]
        self.canvas.set_sub_grid(size)
        self.show_console_message(f"Sub-grid: {size}x{size}" if size > 0 else "Sub-grid: Off")

    def create_navigation_toolbar(self):
        """Create the navigation toolbar below the canvas."""
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 5, 0, 5)

        # First frame button
        self.btn_first = QPushButton("|<")
        self.btn_first.setFixedWidth(40)
        self.btn_first.clicked.connect(self.first_frame)
        nav_layout.addWidget(self.btn_first)

        # Previous frame button
        self.btn_prev = QPushButton("<")
        self.btn_prev.setFixedWidth(40)
        self.btn_prev.clicked.connect(self.prev_frame)
        nav_layout.addWidget(self.btn_prev)

        # Play/Pause button (placeholder for Phase 2)
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(40)
        self.btn_play.clicked.connect(self.toggle_playback)
        nav_layout.addWidget(self.btn_play)

        # Next frame button
        self.btn_next = QPushButton(">")
        self.btn_next.setFixedWidth(40)
        self.btn_next.clicked.connect(self.next_frame)
        nav_layout.addWidget(self.btn_next)

        # Last frame button
        self.btn_last = QPushButton(">|")
        self.btn_last.setFixedWidth(40)
        self.btn_last.clicked.connect(self.last_frame)
        nav_layout.addWidget(self.btn_last)

        # Frame slider
        self.slider_frame = MarkerSlider(Qt.Orientation.Horizontal)
        self.slider_frame.valueChanged.connect(self.seek_frame)
        nav_layout.addWidget(self.slider_frame, stretch=1)

        # Frame label
        self.lbl_nav_frame = QLabel("Frame 0/0")
        self.lbl_nav_frame.setMinimumWidth(100)
        nav_layout.addWidget(self.lbl_nav_frame)

        # FPS dropdown (for Phase 2)
        self.combo_fps = QComboBox()
        self.combo_fps.addItems(["1", "5", "10", "15", "24", "25", "30", "60"])
        self.combo_fps.setCurrentText("30")
        self.combo_fps.setFixedWidth(60)
        self.combo_fps.currentTextChanged.connect(self._update_playback_fps)
        nav_layout.addWidget(QLabel("FPS:"))
        nav_layout.addWidget(self.combo_fps)

        # Initialize playback timer (Phase 2)
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._on_playback_tick)

        return nav_widget

    def create_sidebar(self):
        """Create the right sidebar with file info and pixel inspector."""
        sidebar_widget = QWidget()
        sidebar_widget.setFixedWidth(250)
        sidebar_layout = QVBoxLayout(sidebar_widget)

        # File Info group
        grp_file_info = QGroupBox("File Info")
        file_info_layout = QVBoxLayout()

        self.lbl_path = QLabel("Path: -")
        self.lbl_path.setWordWrap(True)
        file_info_layout.addWidget(self.lbl_path)

        self.lbl_size = QLabel("Size: -")
        file_info_layout.addWidget(self.lbl_size)

        self.lbl_format = QLabel("Format: -")
        file_info_layout.addWidget(self.lbl_format)

        self.lbl_fourcc = QLabel("FourCC: -")
        file_info_layout.addWidget(self.lbl_fourcc)

        self.lbl_v4l2 = QLabel("V4L2: -")
        file_info_layout.addWidget(self.lbl_v4l2)

        self.lbl_resolution = QLabel("Resolution: -")
        file_info_layout.addWidget(self.lbl_resolution)

        self.lbl_frame_size = QLabel("Frame Size: -")
        file_info_layout.addWidget(self.lbl_frame_size)

        self.lbl_total_frames = QLabel("Total Frames: -")
        file_info_layout.addWidget(self.lbl_total_frames)

        self.lbl_bit_depth = QLabel("Bit Depth: 8")
        file_info_layout.addWidget(self.lbl_bit_depth)

        grp_file_info.setLayout(file_info_layout)
        sidebar_layout.addWidget(grp_file_info)

        # Pixel Inspector group
        grp_inspector = QGroupBox("Pixel Inspector")
        insp_layout = QVBoxLayout()

        self.lbl_coord = QLabel("X: - Y: -")
        insp_layout.addWidget(self.lbl_coord)

        self.lbl_raw = QLabel("Raw: -")
        insp_layout.addWidget(self.lbl_raw)

        self.lbl_comps = QLabel("Comp: -")
        insp_layout.addWidget(self.lbl_comps)

        self.lbl_neighborhood = QLabel("Neighborhood:<br>-")
        self.lbl_neighborhood.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_neighborhood.setStyleSheet("font-family: monospace;")
        insp_layout.addWidget(self.lbl_neighborhood)

        grp_inspector.setLayout(insp_layout)
        sidebar_layout.addWidget(grp_inspector)

        # Magnifier group
        grp_magnifier = QGroupBox("Magnifier")
        mag_layout = QVBoxLayout()
        self.lbl_magnifier = QLabel()
        self.lbl_magnifier.setFixedSize(210, 210)
        self.lbl_magnifier.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_magnifier.setStyleSheet("border: 1px solid #888; background-color: black;")
        mag_layout.addWidget(self.lbl_magnifier)
        grp_magnifier.setLayout(mag_layout)
        sidebar_layout.addWidget(grp_magnifier)

        # Bookmark list (hidden, used by bookmark dialog)
        self.list_bookmarks = QListWidget()
        self._bookmark_dialog = None

        sidebar_layout.addStretch()

        return sidebar_widget

    def create_status_bar(self):
        """Create the status bar at the bottom."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Status bar labels
        self.status_filename = QLabel("No file loaded")
        self.status_resolution = QLabel("")
        self.status_format = QLabel("")
        self.status_frame = QLabel("")
        self.status_filesize = QLabel("")
        self.status_framesize = QLabel("")

        # Add labels to status bar
        self.status_bar.addWidget(self.status_filename)
        self.status_bar.addWidget(QLabel("|"))
        self.status_bar.addWidget(self.status_resolution)
        self.status_bar.addWidget(QLabel("|"))
        self.status_bar.addWidget(self.status_format)
        self.status_bar.addWidget(QLabel("|"))
        self.status_bar.addWidget(self.status_frame)
        self.status_bar.addWidget(QLabel("|"))
        self.status_bar.addWidget(self.status_filesize)
        self.status_bar.addWidget(QLabel("|"))
        self.status_bar.addWidget(self.status_framesize)

    def create_analysis_dock(self):
        from PySide6.QtWidgets import QDockWidget, QTabWidget

        self.dock_analysis = QDockWidget("Analysis", self)
        self.dock_analysis.setObjectName("AnalysisDock")
        self.dock_analysis.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea)
        self.dock_analysis.setVisible(False)

        self.tabs_analysis = QTabWidget()
        self.dock_analysis.setWidget(self.tabs_analysis)

        # 1. Histogram Tab
        self.hist_widget = pg.PlotWidget()
        self.hist_widget.setBackground('k')
        self.hist_widget.showGrid(x=True, y=True)
        self.hist_widget.setTitle("Histogram")
        self.tabs_analysis.addTab(self.hist_widget, "Histogram")

        # 2. Vectorscope Tab
        self.vector_widget = pg.PlotWidget()
        self.vector_widget.setBackground('k')
        self.vector_widget.showGrid(x=True, y=True)
        self.vector_widget.setAspectLocked(True)
        self.vector_widget.setXRange(0, 255)
        self.vector_widget.setYRange(0, 255)
        self.vector_widget.setTitle("Vectorscope (Cb-Cr)")

        # Add reticule lines for vectorscope
        self.vector_widget.addLine(x=128, pen=pg.mkPen('w', style=Qt.PenStyle.DashLine))
        self.vector_widget.addLine(y=128, pen=pg.mkPen('w', style=Qt.PenStyle.DashLine))

        self.vectorscope_scatter = pg.ScatterPlotItem(size=2, pen=None, brush=pg.mkBrush(255, 255, 255, 50))
        self.vector_widget.addItem(self.vectorscope_scatter)

        self.tabs_analysis.addTab(self.vector_widget, "Vectorscope")

        # 3. Metrics Tab
        self.metrics_widget = QWidget()
        metrics_layout = QVBoxLayout()
        self.lbl_psnr = QLabel("PSNR: N/A")
        self.lbl_ssim = QLabel("SSIM: N/A")
        self.btn_load_ref = QPushButton("Load Reference Video")
        self.btn_load_ref.clicked.connect(self.load_reference)

        metrics_layout.addWidget(self.lbl_psnr)
        metrics_layout.addWidget(self.lbl_ssim)
        metrics_layout.addWidget(self.btn_load_ref)
        metrics_layout.addStretch()
        self.metrics_widget.setLayout(metrics_layout)

        self.tabs_analysis.addTab(self.metrics_widget, "Metrics")

        # 4. Waveform Monitor Tab
        self.waveform_widget = pg.PlotWidget()
        self.waveform_widget.setBackground('k')
        self.waveform_widget.showGrid(x=True, y=True)
        self.waveform_widget.setTitle("Waveform Monitor")
        self.waveform_widget.setYRange(0, 255)
        self.waveform_widget.setLabel('left', 'Level')
        self.waveform_widget.setLabel('bottom', 'Column')
        self.tabs_analysis.addTab(self.waveform_widget, "Waveform")

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_analysis)

    # Menu action handlers
    def show_parameters_dialog(self, guess_info=None):
        """Show parameters dialog to change width/height/format.

        Returns True if user accepted and video was reloaded, False otherwise.
        """
        dialog = ParametersDialog(self, self.format_manager,
                                  self.current_width, self.current_height, self.current_format,
                                  guess_info=guess_info)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            width, height, format_name = dialog.get_parameters()
            if width and height and format_name:
                self.current_width = width
                self.current_height = height
                self.current_format = format_name
                self.reload_video()
                return True
        return False

    def show_export_dialog(self):
        """Show export clip dialog."""
        if not self.reader:
            self.status_bar.showMessage("No video loaded", 2000)
            return

        dialog = ExportDialog(self, self.reader.total_frames, self.current_frame_idx)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            start, end, fourcc = dialog.get_export_settings()
            if start is not None and end is not None and fourcc:
                self.export_clip_with_settings(start, end, fourcc)

    def show_about(self):
        """Show about dialog."""
        from PySide6.QtWidgets import QMessageBox
        from . import __version__
        QMessageBox.about(self, "About YUV/Raw Video Viewer",
                         f"YUV/Raw Video Viewer v{__version__}\n\n"
                         "A professional tool for viewing and analyzing raw video formats.\n\n"
                         "Copyright (c) babyworm (Hyun-Gyu Kim)")

    def show_comparison_window(self):
        """Open the A/B comparison window."""
        if not self.reader:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Video", "Load a video first before comparing.")
            return
        self._comparison_window = ComparisonWindow(self.reader, self)
        self._comparison_window.show()

    def show_convert_dialog(self):
        """Open the format conversion dialog."""
        if not self.reader:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Video", "Load a video first before converting.")
            return

        dialog = ConvertDialog(self, self.format_manager,
                               self.current_width, self.current_height,
                               self.current_format)
        dialog.set_input_info(self.current_file_path,
                              self.current_width, self.current_height,
                              self.current_format)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            output_path, output_fmt = dialog.get_settings()
            if output_path and output_fmt:
                try:
                    from .video_converter import VideoConverter
                    converter = VideoConverter()
                    count = converter.convert(self.current_file_path,
                                      self.current_width, self.current_height,
                                      self.current_format,
                                      output_path, output_fmt)
                    self.show_console_message(
                        f"Converted {count} frames: {self.current_format} → {output_fmt}, "
                        f"{os.path.basename(output_path)}")
                except Exception as e:
                    self.show_console_message(f"Convert error: {e}")

    def toggle_analysis_dock(self):
        """Toggle analysis dock visibility."""
        self.dock_analysis.setVisible(not self.dock_analysis.isVisible())

    def show_analysis_tab(self, tab_index):
        """Show analysis dock and switch to specified tab."""
        self.dock_analysis.setVisible(True)
        self.tabs_analysis.setCurrentIndex(tab_index)
        self.update_analysis()

    def set_grid_from_menu(self, size):
        """Set grid size from menu action."""
        self.canvas.set_grid(size)
        # Update current grid index for keyboard toggle
        if size in self.grid_sizes:
            self.current_grid_idx = self.grid_sizes.index(size)

    def set_sub_grid_from_menu(self, size):
        """Set sub-grid size from menu action."""
        self.canvas.set_sub_grid(size)
        if size in self.sub_grid_sizes:
            self.current_sub_grid_idx = self.sub_grid_sizes.index(size)

    def _colorize_channel(self, gray, name):
        """Convert single-channel data to false color RGB.

        Args:
            gray: 2D numpy array (H, W) uint8
            name: channel name - 'R','G','B','Y','U','V'

        Returns:
            3D numpy array (H, W, 3) uint8 RGB
        """
        h, w = gray.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        if name == 'R':
            rgb[:, :, 0] = gray
        elif name == 'G':
            rgb[:, :, 1] = gray
        elif name == 'B':
            rgb[:, :, 2] = gray
        elif name == 'Y':
            rgb[:, :, 0] = gray
            rgb[:, :, 1] = gray
            rgb[:, :, 2] = gray
        elif name == 'U':
            rgb[:, :, 2] = gray  # blue
        elif name == 'V':
            rgb[:, :, 0] = gray  # red
        return rgb

    def set_component(self, index):
        """Set component view index."""
        self.current_component = index
        comp_names = {0: "Full", 1: "Y/R", 2: "U/G", 3: "V/B", 4: "Split"}
        name = comp_names.get(index, '?')
        self.status_bar.showMessage(f"Component: {name}", 2000)
        self.show_console_message(f"Component view: {name}")
        self.update_frame()

    # Keyboard shortcuts
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)

        if key == Qt.Key.Key_Space:
            self.toggle_playback()
        elif key == Qt.Key.Key_Left and ctrl:
            self.prev_scene_change()
        elif key == Qt.Key.Key_Right and ctrl:
            self.next_scene_change()
        elif key == Qt.Key.Key_Left:
            self.prev_frame()
        elif key == Qt.Key.Key_Right:
            self.next_frame()
        elif key == Qt.Key.Key_G and shift:
            self.current_sub_grid_idx = (self.current_sub_grid_idx + 1) % len(self.sub_grid_sizes)
            self.canvas.set_sub_grid(self.sub_grid_sizes[self.current_sub_grid_idx])
        elif key == Qt.Key.Key_G:
            self.current_grid_idx = (self.current_grid_idx + 1) % len(self.grid_sizes)
            self.canvas.set_grid(self.grid_sizes[self.current_grid_idx])
        elif key == Qt.Key.Key_C and ctrl:
            self.copy_frame_to_clipboard()
        elif key == Qt.Key.Key_B and ctrl and shift:
            self.prev_bookmark()
        elif key == Qt.Key.Key_B and ctrl:
            self.next_bookmark()
        elif key == Qt.Key.Key_B:
            self.toggle_bookmark()
        elif key == Qt.Key.Key_0:
            self.set_component(0)  # Full
        elif key == Qt.Key.Key_1:
            self.set_component(1)  # Y/R
        elif key == Qt.Key.Key_2:
            self.set_component(2)  # U/G
        elif key == Qt.Key.Key_3:
            self.set_component(3)  # V/B
        elif key == Qt.Key.Key_4:
            self.set_component(4)  # Split view
        else:
            super().keyPressEvent(event)

    # Navigation handlers
    def first_frame(self):
        """Jump to first frame."""
        self.slider_frame.setValue(0)

    def last_frame(self):
        """Jump to last frame."""
        if self.reader:
            self.slider_frame.setValue(self.reader.total_frames - 1)

    def prev_frame(self):
        """Go to previous frame."""
        if self.current_frame_idx > 0:
            self.slider_frame.setValue(self.current_frame_idx - 1)

    def next_frame(self):
        """Go to next frame."""
        if self.reader and self.current_frame_idx < self.reader.total_frames - 1:
            self.slider_frame.setValue(self.current_frame_idx + 1)

    def toggle_playback(self):
        """Toggle playback (Phase 2)."""
        if not self.reader:
            return

        style = self.style()
        if not self.is_playing:
            # Start playback
            self.is_playing = True
            self.btn_play.setText("❚❚")
            self.act_play.setIcon(
                style.standardIcon(QStyle.StandardPixmap.SP_MediaPause))

            # Get FPS and calculate interval
            fps = int(self.combo_fps.currentText())
            interval_ms = int(1000 / fps)

            self.playback_timer.start(interval_ms)
        else:
            # Pause playback
            self.is_playing = False
            self.btn_play.setText("▶")
            self.act_play.setIcon(
                style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.playback_timer.stop()

    def _on_playback_tick(self):
        """Handle playback timer tick - advance to next frame."""
        if not self.reader:
            self.playback_timer.stop()
            return

        # Advance to next frame
        next_idx = self.current_frame_idx + 1

        # Loop back to beginning if at end
        if next_idx >= self.reader.total_frames:
            if self.loop_playback:
                next_idx = 0
            else:
                # Stop playback if not looping
                self.toggle_playback()
                return

        # Use programmatic flag to avoid pausing on slider change
        self._programmatic_slider_change = True
        self.slider_frame.setValue(next_idx)
        self._programmatic_slider_change = False

    def _update_playback_fps(self):
        """Update playback timer interval when FPS changes."""
        if self.is_playing:
            # Update timer interval
            fps = int(self.combo_fps.currentText())
            interval_ms = int(1000 / fps)
            self.playback_timer.setInterval(interval_ms)

    def seek_frame(self, val):
        """Seek to specific frame."""
        # If user manually changes slider during playback, pause playback
        if self.is_playing and not self._programmatic_slider_change:
            self.toggle_playback()

        self.current_frame_idx = val
        self.update_frame()

    # File operations
    def open_file(self):
        """Open file dialog to select video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.yuv *.raw *.rgb *.y4m);;All Files (*)")
        if file_path:
            self._open_file_in_tab(file_path)

    def _open_file_in_tab(self, file_path):
        """Open a file, creating a new tab if one is already open."""
        self._add_to_recent_files(file_path)

        # If no reader yet, just load normally (first file)
        if not self.reader:
            self.current_file_path = file_path
            if file_path.lower().endswith('.y4m'):
                # Y4M: header contains W/H/C → skip dialog
                self.reload_video()
            else:
                # Raw: guess resolution and show dialog with pre-filled values
                guess = self._guess_resolution(file_path)
                guess_info = None
                if guess:
                    self.current_width, self.current_height, self.current_format = guess[:3]
                    fmt_short = guess[2].split(' [')[0]
                    guess_info = f"추정: {guess[0]}x{guess[1]} {fmt_short} ({guess[3]} frames) - 확인 후 OK를 눌러주세요"
                self.show_parameters_dialog(guess_info=guess_info)
            return

        # Save current tab state before opening new tab
        self._save_current_tab_state()

        self.current_file_path = file_path
        if file_path.lower().endswith('.y4m'):
            # Y4M: auto-detect from header
            self.reload_video()
            self._add_tab_for_current()
        else:
            # Raw: guess resolution and show dialog
            guess = self._guess_resolution(file_path)
            guess_info = None
            if guess:
                self.current_width, self.current_height, self.current_format = guess[:3]
                fmt_short = guess[2].split(' [')[0]
                guess_info = f"추정: {guess[0]}x{guess[1]} {fmt_short} ({guess[3]} frames) - 확인 후 OK를 눌러주세요"
            accepted = self.show_parameters_dialog(guess_info=guess_info)
            if not accepted:
                # User cancelled → restore previous tab state
                self._restore_tab_state(self._current_tab_idx)
                return
            self._add_tab_for_current()

    def _guess_resolution(self, file_path):
        """Guess resolution and format from file size.

        Returns (width, height, format_display_name, num_frames) or None.
        """
        try:
            file_size = os.path.getsize(file_path)
        except OSError:
            return None

        if file_size == 0:
            return None

        for width, height in COMMON_RESOLUTIONS:
            for fmt_short in COMMON_GUESS_FORMATS:
                fmt = self.format_manager.get_format(fmt_short)
                if fmt is None:
                    continue
                frame_size = fmt.calculate_frame_size(width, height)
                if frame_size > 0 and file_size % frame_size == 0:
                    num_frames = file_size // frame_size
                    if num_frames >= 1:
                        # Find the display name for this format
                        for display_name, f in self.format_manager.formats.items():
                            if f is fmt:
                                return (width, height, display_name, num_frames)
        return None

    def reload_video(self):
        """Reload video with current parameters."""
        if not hasattr(self, 'current_file_path'):
            return

        try:
            self.reader = VideoReader(self.current_file_path, self.current_width,
                                     self.current_height, self.current_format)

            # Update parameters if reader detected different properties (e.g. Y4M)
            if self.reader.is_y4m:
                self.current_width = self.reader.width
                self.current_height = self.reader.height
                if self.reader.format:
                    self.current_format = self.reader.format.name

            self.slider_frame.setMaximum(max(0, self.reader.total_frames - 1))
            self.current_frame_idx = 0
            self.bookmarks = set()
            self._scene_changes = []
            self.slider_frame.setValue(0)

            # Update sidebar file info
            self.update_file_info()

            # Update status bar
            self.update_status_bar()

            self.update_frame()

            # Ensure a tab exists for the first file
            if len(self._tab_states) == 0:
                self._add_tab_for_current()

            fname = os.path.basename(self.current_file_path)
            self.show_console_message(
                f"Opened: {fname} ({self.current_width}x{self.current_height}, "
                f"{self.current_format}, {self.reader.total_frames} frames)")

        except Exception as e:
            logger.error(f"Error loading video: {e}", exc_info=True)
            self.status_bar.showMessage(f"Error loading video: {e}", 3000)
            self.show_console_message(f"Error: {e}")

    def update_file_info(self):
        """Update file info in sidebar."""
        if not self.reader:
            return

        # Path
        self.lbl_path.setText(f"Path: {os.path.basename(self.current_file_path)}")

        # File size
        file_size = os.path.getsize(self.current_file_path)
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        elif file_size < 1024 * 1024 * 1024:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{file_size / (1024 * 1024 * 1024):.2f} GB"
        self.lbl_size.setText(f"Size: {size_str}")

        # Format
        fmt = self.reader.format
        if fmt:
            self.lbl_format.setText(f"Format: {fmt.name}")
            self.lbl_fourcc.setText(f"FourCC: {fmt.fourcc if hasattr(fmt, 'fourcc') else 'N/A'}")
            # V4L2 name (Phase 4 will add this to VideoFormat)
            v4l2_name = getattr(fmt, 'v4l2_name', 'N/A')
            self.lbl_v4l2.setText(f"V4L2: {v4l2_name}")
        else:
            self.lbl_format.setText("Format: -")
            self.lbl_fourcc.setText("FourCC: -")
            self.lbl_v4l2.setText("V4L2: -")

        # Resolution
        self.lbl_resolution.setText(f"Resolution: {self.reader.width}x{self.reader.height}")

        # Frame size
        frame_size = self.reader.frame_size
        self.lbl_frame_size.setText(f"Frame Size: {frame_size:,} bytes")

        # Total frames
        self.lbl_total_frames.setText(f"Total Frames: {self.reader.total_frames}")

    def update_status_bar(self):
        """Update status bar with current file/frame info."""
        if not self.reader:
            self.status_filename.setText("No file loaded")
            self.status_resolution.setText("")
            self.status_format.setText("")
            self.status_frame.setText("")
            self.status_filesize.setText("")
            self.status_framesize.setText("")
            return

        # Filename
        self.status_filename.setText(os.path.basename(self.current_file_path))

        # Resolution
        self.status_resolution.setText(f"{self.reader.width}x{self.reader.height}")

        # Format with FourCC and V4L2
        fmt = self.reader.format
        if fmt:
            fourcc = getattr(fmt, 'fourcc', 'N/A')
            v4l2_name = getattr(fmt, 'v4l2_name', 'N/A')
            format_str = f"{fmt.name} [{fourcc}] (V4L2: {v4l2_name})"
        else:
            format_str = "-"
        self.status_format.setText(format_str)

        # Frame info
        self.status_frame.setText(f"Frame {self.current_frame_idx}/{self.reader.total_frames}")

        # File size
        file_size = os.path.getsize(self.current_file_path)
        if file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        elif file_size < 1024 * 1024 * 1024:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{file_size / (1024 * 1024 * 1024):.2f} GB"
        self.status_filesize.setText(size_str)

        # Frame size
        self.status_framesize.setText(f"{self.reader.frame_size:,} bytes/frame")

    def update_frame(self):
        """Update the displayed frame."""
        if not self.reader:
            return

        try:
            raw = self.reader.seek_frame(self.current_frame_idx)

            comp_idx = getattr(self, 'current_component', 0)

            if comp_idx == 0: # Full
                # Render using CPU
                # Convert to RGB directly
                rgb = self.reader.convert_to_rgb(raw)
                if rgb is not None:
                     # Ensure data is contiguous for QImage
                     if not rgb.flags['C_CONTIGUOUS']:
                         rgb = np.ascontiguousarray(rgb)

                     self.canvas.set_image(rgb)
                     self.current_rgb_frame = rgb
                     self.current_raw_frame = raw
                     self.update_analysis()
                else:
                     self.canvas.set_image(None)

            elif comp_idx <= 3:
                channels = self.reader.get_channels(raw)
                # Map index to keys: 1->Y/R, 2->U/G, 3->V/B
                keys = []
                if 'Y' in channels: keys = ['Y', 'U', 'V']
                elif 'R' in channels: keys = ['R', 'G', 'B']

                if keys and (comp_idx - 1) < len(keys):
                    key = keys[comp_idx - 1]
                    if key in channels:
                        img = channels[key]
                        rgb = self._colorize_channel(img, key)
                        if not rgb.flags['C_CONTIGUOUS']:
                            rgb = np.ascontiguousarray(rgb)

                        self.canvas.set_image(rgb)
                        self.current_raw_frame = raw
                    else:
                        self.canvas.set_image(None)
                else:
                    self.canvas.set_image(None)

            elif comp_idx == 4:  # Split view (2x2)
                rgb_full = self.reader.convert_to_rgb(raw)
                channels = self.reader.get_channels(raw)

                keys = []
                if 'Y' in channels: keys = ['Y', 'U', 'V']
                elif 'R' in channels: keys = ['R', 'G', 'B']

                if rgb_full is not None and len(keys) == 3:
                    h, w = rgb_full.shape[:2]
                    half_h, half_w = h // 2, w // 2

                    tl = cv2.resize(rgb_full, (half_w, half_h))
                    tr = cv2.resize(self._colorize_channel(channels[keys[0]], keys[0]), (half_w, half_h))
                    bl = cv2.resize(self._colorize_channel(channels[keys[1]], keys[1]), (half_w, half_h))
                    br = cv2.resize(self._colorize_channel(channels[keys[2]], keys[2]), (half_w, half_h))

                    top = np.hstack([tl, tr])
                    bottom = np.hstack([bl, br])
                    composite = np.vstack([top, bottom])

                    self.canvas.set_image(np.ascontiguousarray(composite))
                    self.current_raw_frame = raw
                else:
                    self.canvas.set_image(None)

            # Update navigation label
            self.lbl_nav_frame.setText(f"Frame {self.current_frame_idx}/{self.reader.total_frames}")

            # Update status bar frame info
            if self.reader:
                self.status_frame.setText(f"Frame {self.current_frame_idx}/{self.reader.total_frames}")

        except Exception as e:
            logger.error(f"Error reading frame: {e}", exc_info=True)
            self.status_bar.showMessage(f"Error reading frame: {e}", 3000)

    def save_frame(self):
        """Save current frame as image."""
        if not self.canvas.image:
             return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Frame", "",
                                                   "PNG Files (*.png);;BMP Files (*.bmp);;All Files (*)")
        if file_path:
            self.canvas.image.save(file_path)
            self.show_console_message(f"Frame saved: {os.path.basename(file_path)}")

    def export_clip_with_settings(self, start_frame, end_frame, target_fourcc):
        """Export clip with given settings."""
        if not self.reader:
            return

        if start_frame < 0: start_frame = 0
        if end_frame >= self.reader.total_frames: end_frame = self.reader.total_frames - 1
        if start_frame > end_frame:
            self.status_bar.showMessage("Start frame > End frame", 3000)
            return

        # Save dialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Clip", "",
                                                   "Raw Video (*.yuv *.raw *.rgb);;All Files (*)")
        if not file_path:
            return

        try:
            with open(file_path, "wb") as f_out:
                for i in range(start_frame, end_frame + 1):
                    # 1. Read Raw
                    raw_in = self.reader.seek_frame(i)
                    if not raw_in: break

                    # 2. Convert to RGB (Intermediate)
                    rgb = self.reader.convert_to_rgb(raw_in)
                    if rgb is None: continue

                    # 3. Convert to Target Raw
                    raw_out = self.reader.convert_rgb_to_bytes(rgb, target_fourcc)

                    # 4. Write
                    if raw_out:
                        f_out.write(raw_out)

                    if i % 10 == 0:
                        self.status_bar.showMessage(f"Exporting frame {i}...", 1000)
                        QApplication.processEvents()

            self.status_bar.showMessage(f"Export complete: {file_path}", 3000)

        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            self.status_bar.showMessage(f"Export failed: {e}", 3000)

    # Pixel inspector
    def update_inspector(self, x, y):
        """Update pixel inspector with pixel info at (x, y)."""
        if not self.reader:
            return

        try:
             # Need raw data for the current frame
             if not hasattr(self, 'current_raw_frame'):
                 self.current_raw_frame = self.reader.seek_frame(self.current_frame_idx)

             sub_gs = self.canvas.sub_grid_size
             info = self.reader.get_pixel_info(self.current_raw_frame, x, y, sub_grid_size=sub_gs)
             if info:
                 self.lbl_coord.setText(f"X: {x} Y: {y}")

                 raw_s = " ".join(info['raw_hex'])
                 self.lbl_raw.setText(f"Raw: {raw_s}")

                 comps = [f"{k}:{v}" for k,v in info['components'].items()]
                 self.lbl_comps.setText("Comp: " + ", ".join(comps))

                 nb = info['neighborhood']
                 if sub_gs > 0:
                     cell_x = (x // sub_gs) * sub_gs
                     cell_y = (y // sub_gs) * sub_gs
                     cursor_row = y - cell_y + 1
                     cursor_col = x - cell_x + 1
                 else:
                     cursor_row = 1
                     cursor_col = 1

                 nb_html = '<span style="font-family:monospace;">Neighborhood:<br>'
                 for r_idx, row in enumerate(nb):
                     cells = []
                     for c_idx, val in enumerate(row):
                         if r_idx == cursor_row and c_idx == cursor_col:
                             cells.append(
                                 f'<span style="background-color:#ff4444;'
                                 f'color:white;font-weight:bold;'
                                 f'padding:1px 3px;">{val}</span>')
                         elif sub_gs > 0 and 1 <= r_idx <= sub_gs and 1 <= c_idx <= sub_gs:
                             cells.append(
                                 f'<span style="background-color:#334466;'
                                 f'color:white;padding:1px 3px;">{val}</span>')
                         else:
                             cells.append(f'<span style="padding:1px 3px;">{val}</span>')
                     nb_html += ' '.join(cells) + '<br>'
                 nb_html += '</span>'
                 self.lbl_neighborhood.setText(nb_html)

             # Update magnifier
             self._update_magnifier(x, y)
        except Exception as e:
            pass

    # Analysis functions
    def load_reference(self):
        """Load reference video for metrics comparison."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Reference Video")
        if file_path:
            try:
                if self.reader:
                    self.ref_reader = VideoReader(file_path, self.reader.width,
                                                  self.reader.height, self.reader.format.name)
                    self.status_bar.showMessage(f"Reference loaded: {os.path.basename(file_path)}", 3000)
                else:
                    self.status_bar.showMessage("Load main video first to establish format.", 3000)
            except Exception as e:
                self.status_bar.showMessage(f"Failed to load reference: {e}", 3000)

    def update_analysis(self):
        """Update analysis displays."""
        if not self.reader or not hasattr(self, 'current_raw_frame'):
            return

        # Only update if dock is visible to save CPU
        if not self.dock_analysis.isVisible():
            return

        current_tab = self.tabs_analysis.currentIndex()
        if current_tab == 0:
            self.update_histogram()
        elif current_tab == 1:
            self.update_vectorscope()
        elif current_tab == 2:
            self.update_metrics()
        elif current_tab == 3:
            self.update_waveform()

    def update_histogram(self):
        """Update histogram display."""
        if hasattr(self, 'current_rgb_frame'):
             # Calculate Histogram
             hists = VideoAnalyzer.calculate_histogram(self.current_rgb_frame, "RGB")
             self.hist_widget.clear()

             colors = {'r': (255, 0, 0), 'g': (0, 255, 0), 'b': (0, 0, 255), 'y': (255, 255, 255)}

             for channel, vals in hists.items():
                 self.hist_widget.plot(vals, pen=pg.mkPen(colors[channel]))

    def update_vectorscope(self):
        """Update vectorscope display."""
        if hasattr(self, 'current_rgb_frame'):
             # Downsample for Scatter Plot performance
             cb, cr = VideoAnalyzer.calculate_vectorscope_from_rgb(self.current_rgb_frame)

             # Random sample
             if len(cb) > 4096:
                 indices = np.random.choice(len(cb), 4096, replace=False)
                 cb = cb[indices]
                 cr = cr[indices]

             self.vectorscope_scatter.setData(cb, cr)

    def update_metrics(self):
        """Update metrics display."""
        if hasattr(self, 'ref_reader') and hasattr(self, 'current_rgb_frame'):
            # Get Reference Frame
            ref_raw = self.ref_reader.seek_frame(self.current_frame_idx)
            if ref_raw:
                ref_rgb = self.ref_reader.convert_to_rgb(ref_raw)

                psnr = VideoAnalyzer.calculate_psnr(self.current_rgb_frame, ref_rgb)
                ssim = VideoAnalyzer.calculate_ssim(self.current_rgb_frame, ref_rgb)

                self.lbl_psnr.setText(f"PSNR: {psnr:.2f} dB")
                self.lbl_ssim.setText(f"SSIM: {ssim:.4f}")
            else:
                 self.lbl_psnr.setText("PSNR: Ref End")

    def update_waveform(self):
        """Update waveform monitor display."""
        if hasattr(self, 'current_rgb_frame'):
            wf = VideoAnalyzer.calculate_waveform(self.current_rgb_frame, "luma")
            if wf is not None:
                self.waveform_widget.clear()
                # Plot max/avg per column for visualization
                h, w = wf.shape
                x = np.arange(w)
                # For each column, find weighted average level
                levels = np.arange(256)
                total = wf.sum(axis=0)
                total[total == 0] = 1
                avg = (wf * levels[:, np.newaxis]).sum(axis=0) / total
                self.waveform_widget.plot(x, avg, pen=pg.mkPen('g', width=1))
                # Also plot 10th/90th percentile
                cumsum = np.cumsum(wf, axis=0)
                norm = cumsum / total[np.newaxis, :]
                p10 = np.argmax(norm >= 0.1, axis=0).astype(float)
                p90 = np.argmax(norm >= 0.9, axis=0).astype(float)
                self.waveform_widget.plot(x, p10, pen=pg.mkPen('b', width=1))
                self.waveform_widget.plot(x, p90, pen=pg.mkPen('r', width=1))

    # ── Magnifier ──
    def _update_magnifier(self, x, y):
        """Update magnifier widget with zoomed view around (x, y)."""
        if not hasattr(self, 'current_rgb_frame') or self.current_rgb_frame is None:
            return

        h, w, _ = self.current_rgb_frame.shape
        radius = 7  # 15x15 pixel area
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(w, x + radius + 1)
        y2 = min(h, y + radius + 1)

        crop = self.current_rgb_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        # Scale up with nearest-neighbor
        zoom = 14
        zoomed = cv2.resize(crop, (crop.shape[1] * zoom, crop.shape[0] * zoom),
                            interpolation=cv2.INTER_NEAREST)

        # Draw crosshair at center
        ch, cw, _ = zoomed.shape
        cx = (x - x1) * zoom + zoom // 2
        cy = (y - y1) * zoom + zoom // 2
        zoomed = zoomed.copy()
        cv2.line(zoomed, (cx, 0), (cx, ch), (255, 0, 0), 1)
        cv2.line(zoomed, (0, cy), (cw, cy), (255, 0, 0), 1)

        # Convert to QPixmap
        qh, qw, qc = zoomed.shape
        qimg = QImage(zoomed.data, qw, qh, qw * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            210, 210, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation)
        self.lbl_magnifier.setPixmap(pixmap)

    # ── Drag & Drop ──
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path and os.path.isfile(file_path):
                self._open_file_in_tab(file_path)

    # ── Recent Files ──
    def _update_recent_files_menu(self):
        self.recent_menu.clear()
        recent = self._settings.value("recent_files", [], type=list)
        for path in recent[:10]:
            action = QAction(os.path.basename(path), self)
            action.setToolTip(path)
            action.triggered.connect(lambda checked, p=path: self._open_recent_file(p))
            self.recent_menu.addAction(action)
        if not recent:
            empty = QAction("(empty)", self)
            empty.setEnabled(False)
            self.recent_menu.addAction(empty)

    def _add_to_recent_files(self, file_path):
        recent = self._settings.value("recent_files", [], type=list)
        if file_path in recent:
            recent.remove(file_path)
        recent.insert(0, file_path)
        self._settings.setValue("recent_files", recent[:10])
        self._update_recent_files_menu()

    def _open_recent_file(self, path):
        if os.path.isfile(path):
            self._open_file_in_tab(path)
        else:
            self.status_bar.showMessage(f"File not found: {path}", 3000)

    # ── Dark Theme ──
    def apply_dark_theme(self, enable):
        self._dark_theme = enable
        app = QApplication.instance()
        if enable:
            app.setStyleSheet(DARK_STYLE)
        else:
            app.setStyleSheet("")
        self._settings.setValue("dark_theme", enable)

    def toggle_theme(self):
        self.apply_dark_theme(not self._dark_theme)
        self.dark_theme_action.setChecked(self._dark_theme)

    # ── Window State ──
    def _restore_window_state(self):
        geom = self._settings.value("geometry")
        if geom:
            self.restoreGeometry(geom)
        state = self._settings.value("windowState")
        if state:
            self.restoreState(state)

    def closeEvent(self, event):
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState())
        # Close all tab readers
        for state in self._tab_states:
            r = state.get("reader")
            if r:
                r.close()
        # Close current reader if not in tab_states
        if self.reader:
            self.reader.close()
        event.accept()

    # ── Keyboard Shortcuts Dialog ──
    def show_shortcuts_dialog(self):
        dialog = ShortcutsDialog(self)
        dialog.exec()

    # ── Clipboard Copy ──
    def copy_frame_to_clipboard(self):
        if not self.canvas.image:
            return
        clipboard = QApplication.clipboard()
        clipboard.setImage(self.canvas.image)
        self.status_bar.showMessage("Frame copied to clipboard", 2000)

    # ── Batch Convert ──
    def show_batch_convert_dialog(self):
        dialog = BatchConvertDialog(self, self.format_manager)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            if not settings or not settings["files"] or not settings["output_dir"]:
                return
            self._run_batch_convert(settings)

    def _run_batch_convert(self, settings):
        from .video_converter import VideoConverter
        converter = VideoConverter()
        total = len(settings["files"])
        progress = QProgressDialog("Converting...", "Cancel", 0, total, self)
        progress.setWindowTitle("Batch Convert")
        progress.setMinimumDuration(0)

        for i, fpath in enumerate(settings["files"]):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            progress.setLabelText(f"Converting {os.path.basename(fpath)}...")
            QApplication.processEvents()

            base = os.path.splitext(os.path.basename(fpath))[0]
            out_path = os.path.join(settings["output_dir"], f"{base}_converted.yuv")
            try:
                converter.convert(fpath, settings["width"], settings["height"],
                                  settings["input_fmt"], out_path, settings["output_fmt"])
            except Exception as e:
                logger.error(f"Batch convert error for {fpath}: {e}")

        progress.setValue(total)
        self.show_console_message(
            f"Batch convert complete: {total} files, "
            f"{settings['input_fmt']} → {settings['output_fmt']}")

    # ── PNG Sequence Export ──
    def show_png_export_dialog(self):
        if not self.reader:
            self.status_bar.showMessage("Load a video first", 2000)
            return
        dialog = PngExportDialog(self, self.reader.total_frames, self.current_frame_idx)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            if settings:
                self._export_png_sequence(settings)

    def _export_png_sequence(self, settings):
        start, end = settings["start"], settings["end"]
        directory = settings["directory"]
        prefix = settings["prefix"]

        os.makedirs(directory, exist_ok=True)
        total = end - start + 1
        progress = QProgressDialog("Exporting PNGs...", "Cancel", 0, total, self)
        progress.setWindowTitle("PNG Export")
        progress.setMinimumDuration(0)

        for i, idx in enumerate(range(start, end + 1)):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            QApplication.processEvents()

            raw = self.reader.seek_frame(idx)
            rgb = self.reader.convert_to_rgb(raw)
            if rgb is not None:
                out_path = os.path.join(directory, f"{prefix}_{idx:06d}.png")
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_path, bgr)

        progress.setValue(total)
        self.status_bar.showMessage(f"Exported {total} PNGs to {directory}", 3000)

    # ── Bookmarks ──
    def toggle_bookmark(self):
        if not self.reader:
            return
        idx = self.current_frame_idx
        if idx in self.bookmarks:
            self.bookmarks.discard(idx)
            self.status_bar.showMessage(f"Bookmark removed: frame {idx}", 2000)
            self.show_console_message(f"Bookmark removed: frame {idx}")
        else:
            self.bookmarks.add(idx)
            self.status_bar.showMessage(f"Bookmark added: frame {idx}", 2000)
            self.show_console_message(f"Bookmark added: frame {idx} (total: {len(self.bookmarks) + 1})")
        self._refresh_bookmark_list()

    def _refresh_bookmark_list(self):
        self.list_bookmarks.clear()
        for bm in sorted(self.bookmarks):
            self.list_bookmarks.addItem(f"Frame {bm}")
        self.slider_frame.set_bookmark_markers(sorted(self.bookmarks))

    def show_bookmark_dialog(self):
        """Show bookmark list in a separate dialog."""
        if not self.bookmarks:
            self.status_bar.showMessage("No bookmarks set", 2000)
            return
        if self._bookmark_dialog and self._bookmark_dialog.isVisible():
            self._bookmark_dialog.raise_()
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Bookmarks")
        dlg.setMinimumSize(250, 200)
        layout = QVBoxLayout(dlg)
        bm_list = QListWidget()
        for bm in sorted(self.bookmarks):
            bm_list.addItem(f"Frame {bm}")
        bm_list.itemDoubleClicked.connect(lambda item: self._jump_to_bookmark(item))
        layout.addWidget(bm_list)
        btn_row = QHBoxLayout()
        btn_del = QPushButton("Remove Selected")
        btn_del.clicked.connect(lambda: self._remove_bookmark_from_dialog(bm_list))
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.close)
        btn_row.addWidget(btn_del)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)
        self._bookmark_dialog = dlg
        dlg.show()

    def _remove_bookmark_from_dialog(self, bm_list):
        item = bm_list.currentItem()
        if item:
            idx = int(item.text().replace("Frame ", ""))
            self.bookmarks.discard(idx)
            bm_list.takeItem(bm_list.row(item))
            self._refresh_bookmark_list()
            if not self.bookmarks and self._bookmark_dialog:
                self._bookmark_dialog.close()

    def _jump_to_bookmark(self, item):
        text = item.text()
        idx = int(text.replace("Frame ", ""))
        self.slider_frame.setValue(idx)

    def _remove_selected_bookmark(self):
        item = self.list_bookmarks.currentItem()
        if item:
            idx = int(item.text().replace("Frame ", ""))
            self.bookmarks.discard(idx)
            self._refresh_bookmark_list()

    def next_bookmark(self):
        if not self.bookmarks:
            return
        higher = sorted(b for b in self.bookmarks if b > self.current_frame_idx)
        if higher:
            self.slider_frame.setValue(higher[0])
        else:
            self.slider_frame.setValue(min(self.bookmarks))

    def prev_bookmark(self):
        if not self.bookmarks:
            return
        lower = sorted((b for b in self.bookmarks if b < self.current_frame_idx), reverse=True)
        if lower:
            self.slider_frame.setValue(lower[0])
        else:
            self.slider_frame.setValue(max(self.bookmarks))

    # ── Scene Change Detection ──
    _SCENE_ALGORITHMS = [
        ("Mean Pixel Difference (MAD)", 45.0),
        ("Histogram Comparison", 30.0),
        ("SSIM (Structural Similarity)", 40.0),
    ]

    def detect_scene_changes(self):
        if not self.reader or self.reader.total_frames < 2:
            self.status_bar.showMessage("Need a loaded video with 2+ frames", 2000)
            return

        # Settings dialog: algorithm + threshold in one window
        dlg = QDialog(self)
        dlg.setWindowTitle("Scene Detection Settings")
        form = QFormLayout(dlg)

        combo_algo = QComboBox()
        algo_names = [a[0] for a in self._SCENE_ALGORITHMS]
        combo_algo.addItems(algo_names)
        form.addRow("Algorithm:", combo_algo)

        from PySide6.QtWidgets import QDoubleSpinBox
        spin_thresh = QDoubleSpinBox()
        spin_thresh.setRange(0.1, 255.0)
        spin_thresh.setDecimals(1)
        spin_thresh.setValue(self._SCENE_ALGORITHMS[0][1])
        form.addRow("Threshold:", spin_thresh)

        # Update threshold default when algorithm changes
        def on_algo_changed(idx):
            spin_thresh.setValue(self._SCENE_ALGORITHMS[idx][1])
        combo_algo.currentIndexChanged.connect(on_algo_changed)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        form.addRow(btn_box)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        algo_idx = combo_algo.currentIndex()
        algo_name = algo_names[algo_idx]
        threshold = spin_thresh.value()

        # Clear old markers before detection
        self._scene_changes = []
        self.slider_frame.set_scene_markers([])

        total = self.reader.total_frames
        progress = QProgressDialog(f"Detecting scene changes ({algo_name})...", "Cancel", 0, total - 1, self)
        progress.setWindowTitle("Scene Detection")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # Select comparison function
        if algo_idx == 0:
            compare_fn = VideoAnalyzer.calculate_frame_difference
        elif algo_idx == 1:
            compare_fn = VideoAnalyzer.calculate_histogram_difference
        else:
            compare_fn = VideoAnalyzer.calculate_ssim_difference

        prev_rgb = self.reader.convert_to_rgb(self.reader.seek_frame(0))

        for i in range(1, total):
            if progress.wasCanceled():
                break
            progress.setValue(i)

            raw = self.reader.seek_frame(i)
            cur_rgb = self.reader.convert_to_rgb(raw)
            if prev_rgb is not None and cur_rgb is not None:
                diff = compare_fn(prev_rgb, cur_rgb)
                if diff > threshold:
                    self._scene_changes.append(i)
            prev_rgb = cur_rgb

        progress.setValue(total - 1)
        progress.close()
        self.slider_frame.set_scene_markers(self._scene_changes)

        count = len(self._scene_changes)
        if count > 0:
            msg = f"[{algo_name}] Found {count} scene changes (threshold={threshold})"
        else:
            msg = f"[{algo_name}] No scene changes found (threshold={threshold}). Try lowering the threshold."
        self.status_bar.showMessage(msg, 5000)
        self.show_console_message(msg)

    def next_scene_change(self):
        if not self._scene_changes:
            return
        higher = [s for s in self._scene_changes if s > self.current_frame_idx]
        if higher:
            self.slider_frame.setValue(higher[0])

    def prev_scene_change(self):
        if not self._scene_changes:
            return
        lower = [s for s in self._scene_changes if s < self.current_frame_idx]
        if lower:
            self.slider_frame.setValue(lower[-1])

    # ── ROI ──
    def _on_roi_selected(self, x, y, w, h):
        if not hasattr(self, 'current_rgb_frame') or self.current_rgb_frame is None:
            return
        roi = self.current_rgb_frame[y:y+h, x:x+w]
        if roi.size == 0:
            return
        # Show ROI stats in status bar
        mean_r = np.mean(roi[:, :, 0])
        mean_g = np.mean(roi[:, :, 1])
        mean_b = np.mean(roi[:, :, 2])
        self.status_bar.showMessage(
            f"ROI ({w}x{h}) at ({x},{y}) | Mean R:{mean_r:.1f} G:{mean_g:.1f} B:{mean_b:.1f}", 5000)

        # Update histogram with ROI data if analysis dock is visible
        if self.dock_analysis.isVisible() and self.tabs_analysis.currentIndex() == 0:
            hists = VideoAnalyzer.calculate_histogram(roi, "RGB")
            self.hist_widget.clear()
            colors = {'r': (255, 0, 0), 'g': (0, 255, 0), 'b': (0, 0, 255)}
            for channel, vals in hists.items():
                if channel in colors:
                    self.hist_widget.plot(vals, pen=pg.mkPen(colors[channel]))

    # ── Color Matrix ──
    def _set_color_matrix(self, matrix):
        self.act_bt601.setChecked(matrix == "BT.601")
        self.act_bt709.setChecked(matrix == "BT.709")
        if self.reader:
            self.reader.color_matrix = matrix
            self.update_frame()
        self.status_bar.showMessage(f"Color matrix: {matrix}", 2000)

    # ── Multi-tab Management ──
    def _add_tab_for_current(self):
        """Add a tab for the currently loaded file."""
        if not hasattr(self, 'current_file_path') or not self.reader:
            return

        # Check if file is already open in a tab
        for i, state in enumerate(self._tab_states):
            if state["file_path"] == self.current_file_path:
                self._switching_tab = True
                self.tab_bar.setCurrentIndex(i)
                self._current_tab_idx = i
                self._switching_tab = False
                return

        tab_state = {
            "file_path": self.current_file_path,
            "reader": self.reader,
            "frame_idx": self.current_frame_idx,
            "width": self.current_width,
            "height": self.current_height,
            "format": self.current_format,
            "bookmarks": set(self.bookmarks),
            "scene_changes": list(self._scene_changes),
        }
        self._tab_states.append(tab_state)

        self._switching_tab = True
        label = os.path.basename(self.current_file_path)
        # Add an empty widget as page (we don't use QTabWidget pages, only the tab bar)
        self.tab_bar.addTab(QWidget(), label)
        self._current_tab_idx = len(self._tab_states) - 1
        self.tab_bar.setCurrentIndex(self._current_tab_idx)
        self._switching_tab = False

        # Show tab bar when 2+ tabs
        self.tab_bar.setVisible(len(self._tab_states) >= 2)

    def _save_current_tab_state(self):
        """Save current viewer state into the active tab's entry."""
        if self._current_tab_idx < 0 or self._current_tab_idx >= len(self._tab_states):
            return
        state = self._tab_states[self._current_tab_idx]
        state["reader"] = self.reader
        state["frame_idx"] = self.current_frame_idx
        state["width"] = self.current_width
        state["height"] = self.current_height
        state["format"] = self.current_format
        state["bookmarks"] = set(self.bookmarks)
        state["scene_changes"] = list(self._scene_changes)

    def _restore_tab_state(self, idx):
        """Restore viewer state from a tab entry."""
        if idx < 0 or idx >= len(self._tab_states):
            return
        state = self._tab_states[idx]
        self.current_file_path = state["file_path"]
        self.reader = state["reader"]
        self.current_width = state["width"]
        self.current_height = state["height"]
        self.current_format = state["format"]
        self.current_frame_idx = state["frame_idx"]
        self.bookmarks = set(state["bookmarks"])
        self._scene_changes = list(state["scene_changes"])

        # Update UI
        if self.reader:
            self.slider_frame.setMaximum(max(0, self.reader.total_frames - 1))
            self._programmatic_slider_change = True
            self.slider_frame.setValue(self.current_frame_idx)
            self._programmatic_slider_change = False
        self.update_file_info()
        self.update_status_bar()
        self._refresh_bookmark_list()
        self.update_frame()

    def _on_tab_changed(self, index):
        """Handle tab selection change."""
        if self._switching_tab or index < 0:
            return
        if index == self._current_tab_idx:
            return

        # Save old tab state
        self._save_current_tab_state()

        # Restore new tab state
        self._current_tab_idx = index
        self._restore_tab_state(index)

    def _close_tab(self, index):
        """Close a tab and clean up its resources."""
        if index < 0 or index >= len(self._tab_states):
            return

        # Don't close the last tab
        if len(self._tab_states) <= 1:
            self.status_bar.showMessage("Cannot close the last tab", 2000)
            return

        # Close the reader for the removed tab
        state = self._tab_states.pop(index)
        reader = state.get("reader")
        # Only close if it's not the current active reader
        if reader and reader is not self.reader:
            reader.close()

        self._switching_tab = True
        self.tab_bar.removeTab(index)
        self._switching_tab = False

        # Adjust current index
        if index == self._current_tab_idx:
            # Switched away: restore the now-current tab
            new_idx = min(index, len(self._tab_states) - 1)
            self._current_tab_idx = new_idx
            self._switching_tab = True
            self.tab_bar.setCurrentIndex(new_idx)
            self._switching_tab = False
            self._restore_tab_state(new_idx)
        elif index < self._current_tab_idx:
            self._current_tab_idx -= 1

        # Hide tab bar when only 1 tab
        self.tab_bar.setVisible(len(self._tab_states) >= 2)
        self.status_bar.showMessage(f"Closed: {os.path.basename(state['file_path'])}", 2000)
