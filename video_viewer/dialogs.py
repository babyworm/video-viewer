import os
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QLabel,
                               QLineEdit, QComboBox, QPushButton, QFileDialog,
                               QVBoxLayout, QHBoxLayout, QTableWidget,
                               QTableWidgetItem, QHeaderView, QListWidget,
                               QSpinBox, QGroupBox, QDoubleSpinBox, QCheckBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtCore import QSettings

from .constants import (COMMON_RESOLUTIONS, COMMON_GUESS_FORMATS,
                        DEFAULT_CACHE_MAX_MEMORY_MB, ZOOM_MIN, ZOOM_MAX,
                        DEFAULT_FPS_OPTIONS, DEFAULT_COLOR_MATRIX,
                        DEFAULT_RESOLUTION_WIDTH, DEFAULT_RESOLUTION_HEIGHT)
from .format_manager import FormatManager



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


class SettingsDialog(QDialog):
    """Dialog for application-wide settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Group: Cache
        cache_group = QGroupBox("Cache")
        cache_layout = QFormLayout()
        self.cache_memory_spin = QSpinBox()
        self.cache_memory_spin.setRange(64, 4096)
        self.cache_memory_spin.setSuffix(" MB")
        cache_layout.addRow("Max Memory:", self.cache_memory_spin)
        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)

        # Group: Display
        display_group = QGroupBox("Display")
        display_layout = QFormLayout()
        self.zoom_min_spin = QDoubleSpinBox()
        self.zoom_min_spin.setRange(0.01, 1.0)
        self.zoom_min_spin.setSingleStep(0.01)
        self.zoom_min_spin.setDecimals(2)
        self.zoom_max_spin = QDoubleSpinBox()
        self.zoom_max_spin.setRange(5.0, 200.0)
        self.zoom_max_spin.setSingleStep(1.0)
        self.zoom_max_spin.setDecimals(1)
        display_layout.addRow("Zoom Min:", self.zoom_min_spin)
        display_layout.addRow("Zoom Max:", self.zoom_max_spin)
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # Group: Defaults
        defaults_group = QGroupBox("Defaults")
        defaults_layout = QFormLayout()
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(DEFAULT_FPS_OPTIONS)

        self.color_matrix_combo = QComboBox()
        self.color_matrix_combo.addItems(["BT.601", "BT.709"])


        self.default_width_spin = QSpinBox()
        self.default_width_spin.setRange(1, 7680)
        self.default_height_spin = QSpinBox()
        self.default_height_spin.setRange(1, 4320)
        self.dark_theme_check = QCheckBox()
        defaults_layout.addRow("Default FPS:", self.fps_combo)
        defaults_layout.addRow("Color Matrix:", self.color_matrix_combo)
        defaults_layout.addRow("Default Width:", self.default_width_spin)
        defaults_layout.addRow("Default Height:", self.default_height_spin)
        defaults_layout.addRow("Dark Theme:", self.dark_theme_check)
        defaults_group.setLayout(defaults_layout)
        layout.addWidget(defaults_group)

        # Buttons: Restore Defaults, OK, Cancel
        button_layout = QHBoxLayout()
        restore_btn = QPushButton("Restore Defaults")
        restore_btn.clicked.connect(self._restore_defaults)
        button_layout.addWidget(restore_btn)
        button_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        # Load defaults initially
        self._restore_defaults()

    def load_settings(self, settings: QSettings):
        """Read values from QSettings 'preferences/' group into widgets."""
        settings.beginGroup("preferences")
        self.cache_memory_spin.setValue(
            settings.value("cache_max_memory_mb", DEFAULT_CACHE_MAX_MEMORY_MB, type=int))
        self.zoom_min_spin.setValue(
            settings.value("zoom_min", ZOOM_MIN, type=float))
        self.zoom_max_spin.setValue(
            settings.value("zoom_max", ZOOM_MAX, type=float))
        fps = settings.value("default_fps", DEFAULT_FPS_OPTIONS[6])  # "30"
        idx = self.fps_combo.findText(fps)
        if idx >= 0:
            self.fps_combo.setCurrentIndex(idx)
        matrix = settings.value("color_matrix", DEFAULT_COLOR_MATRIX)
        idx = self.color_matrix_combo.findText(matrix)
        if idx >= 0:
            self.color_matrix_combo.setCurrentIndex(idx)
        self.default_width_spin.setValue(
            settings.value("default_width", DEFAULT_RESOLUTION_WIDTH, type=int))
        self.default_height_spin.setValue(
            settings.value("default_height", DEFAULT_RESOLUTION_HEIGHT, type=int))
        self.dark_theme_check.setChecked(
            settings.value("dark_theme", False, type=bool))
        settings.endGroup()

    def save_settings(self, settings: QSettings):
        """Write current widget values to QSettings 'preferences/' group."""
        settings.beginGroup("preferences")
        settings.setValue("cache_max_memory_mb", self.cache_memory_spin.value())
        settings.setValue("zoom_min", self.zoom_min_spin.value())
        settings.setValue("zoom_max", self.zoom_max_spin.value())
        settings.setValue("default_fps", self.fps_combo.currentText())
        settings.setValue("color_matrix", self.color_matrix_combo.currentText())
        settings.setValue("default_width", self.default_width_spin.value())
        settings.setValue("default_height", self.default_height_spin.value())
        settings.setValue("dark_theme", self.dark_theme_check.isChecked())
        settings.endGroup()

    def _restore_defaults(self):
        """Reset all widgets to their constant default values."""
        self.cache_memory_spin.setValue(DEFAULT_CACHE_MAX_MEMORY_MB)
        self.zoom_min_spin.setValue(ZOOM_MIN)
        self.zoom_max_spin.setValue(ZOOM_MAX)
        idx = self.fps_combo.findText("30")
        if idx >= 0:
            self.fps_combo.setCurrentIndex(idx)
        idx = self.color_matrix_combo.findText(DEFAULT_COLOR_MATRIX)
        if idx >= 0:
            self.color_matrix_combo.setCurrentIndex(idx)
        self.default_width_spin.setValue(DEFAULT_RESOLUTION_WIDTH)
        self.default_height_spin.setValue(DEFAULT_RESOLUTION_HEIGHT)
        self.dark_theme_check.setChecked(False)

    def get_values(self) -> dict:
        """Return all current widget values as a dict."""
        return {
            "cache_max_memory_mb": self.cache_memory_spin.value(),
            "zoom_min": self.zoom_min_spin.value(),
            "zoom_max": self.zoom_max_spin.value(),
            "default_fps": self.fps_combo.currentText(),
            "color_matrix": self.color_matrix_combo.currentText(),
            "default_width": self.default_width_spin.value(),
            "default_height": self.default_height_spin.value(),
            "dark_theme": self.dark_theme_check.isChecked(),
        }
