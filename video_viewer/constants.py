# Cache defaults
DEFAULT_CACHE_MAX_FRAMES = 16
DEFAULT_CACHE_MAX_MEMORY_MB = 512

# Zoom bounds
ZOOM_MIN = 0.1
ZOOM_MAX = 50.0

# FPS options for playback combobox
DEFAULT_FPS_OPTIONS = ["1", "5", "10", "15", "24", "25", "30", "60"]

# Default color matrix
DEFAULT_COLOR_MATRIX = "BT.601"

# Grid sizes for keyboard toggle
DEFAULT_GRID_SIZES = [0, 16, 32, 64, 128]
DEFAULT_SUB_GRID_SIZES = [0, 4, 8, 16]

# Window defaults
DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 800
DEFAULT_RESOLUTION_WIDTH = 1920
DEFAULT_RESOLUTION_HEIGHT = 1080

# Scene detection default threshold (used for first algorithm: Mean Pixel Difference)
DEFAULT_SCENE_THRESHOLD = 45.0

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


# Dark theme stylesheet
DARK_STYLE = """
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
QComboBox QAbstractItemView { background-color: #45494a; color: #dcdcdc;
    selection-background-color: #4b6eaf; selection-color: white;
    border: 1px solid #646464; outline: none; }
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
