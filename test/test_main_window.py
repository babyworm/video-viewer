"""
Test MainWindow initialization and attribute ordering.
Ensures canvas is created before menu bar to prevent AttributeError.
"""
import pytest
import sys
import os
from PySide6.QtWidgets import QApplication

@pytest.fixture
def qapp():
    """Create QApplication instance for tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_main_window_creation(qapp):
    """Test that MainWindow can be created without errors (init ordering)."""
    from video_viewer.main_window import MainWindow
    window = MainWindow()
    assert window is not None
    assert hasattr(window, 'canvas')
    assert hasattr(window, 'slider_frame')
    assert hasattr(window, 'btn_play')
    assert hasattr(window, 'combo_fps')
    assert hasattr(window, 'status_bar')
    assert hasattr(window, 'dock_analysis')


def test_main_window_canvas_before_menu(qapp):
    """Verify canvas exists when menu bar references it."""
    from video_viewer.main_window import MainWindow
    window = MainWindow()
    # canvas must exist and be an ImageCanvas
    from video_viewer.main_window import ImageCanvas
    assert isinstance(window.canvas, ImageCanvas)
    # Menu bar should have menus
    menubar = window.menuBar()
    assert menubar is not None
    actions = menubar.actions()
    assert len(actions) >= 4  # File, View, Tools, Help


def test_main_window_playback_state(qapp):
    """Test that playback state is properly initialized."""
    from video_viewer.main_window import MainWindow
    window = MainWindow()
    assert window.is_playing is False
    assert window.loop_playback is True
    assert window._programmatic_slider_change is False
    assert hasattr(window, 'playback_timer')


def test_main_window_sidebar_labels(qapp):
    """Test sidebar info labels exist."""
    from video_viewer.main_window import MainWindow
    window = MainWindow()
    assert hasattr(window, 'lbl_path')
    assert hasattr(window, 'lbl_format')
    assert hasattr(window, 'lbl_fourcc')
    assert hasattr(window, 'lbl_v4l2')
    assert hasattr(window, 'lbl_coord')
    assert hasattr(window, 'lbl_neighborhood')


def test_multi_tab_init(qapp):
    """Test multi-tab state is properly initialized."""
    from video_viewer.main_window import MainWindow
    window = MainWindow()
    assert hasattr(window, 'tab_bar')
    assert hasattr(window, '_tab_states')
    assert hasattr(window, '_current_tab_idx')
    assert hasattr(window, '_switching_tab')
    assert window._tab_states == []
    assert window._current_tab_idx == -1
    assert window._switching_tab is False
    assert window.tab_bar.isHidden() is True


def test_multi_tab_add_and_close(qapp, tmp_path):
    """Test adding tabs and closing them."""
    from video_viewer.main_window import MainWindow
    window = MainWindow()

    # Create minimal I420 files
    width, height = 4, 4
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_bytes = b'\x80' * (y_size + 2 * uv_size)

    f1 = tmp_path / "video1.yuv"
    f1.write_bytes(frame_bytes)
    f2 = tmp_path / "video2.yuv"
    f2.write_bytes(frame_bytes * 2)  # 2 frames

    # Load first file
    window.current_file_path = str(f1)
    window.current_width = width
    window.current_height = height
    window.current_format = "I420"
    window.reload_video()

    assert len(window._tab_states) == 1
    assert window.tab_bar.count() == 1
    assert window.tab_bar.isHidden() is True  # 1 tab => hidden

    # Open second file as tab
    window._save_current_tab_state()
    window.current_file_path = str(f2)
    window.reload_video()
    window._add_tab_for_current()

    assert len(window._tab_states) == 2
    assert window.tab_bar.count() == 2
    assert window.tab_bar.isHidden() is False  # 2 tabs => not hidden
    assert window._current_tab_idx == 1

    # Close second tab
    window._close_tab(1)
    assert len(window._tab_states) == 1
    assert window.tab_bar.count() == 1
    assert window.tab_bar.isHidden() is True
    assert window._current_tab_idx == 0


def test_multi_tab_switch(qapp, tmp_path):
    """Test switching between tabs preserves state."""
    from video_viewer.main_window import MainWindow
    window = MainWindow()

    width, height = 4, 4
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_bytes = b'\x80' * (y_size + 2 * uv_size)

    f1 = tmp_path / "a.yuv"
    f1.write_bytes(frame_bytes * 3)  # 3 frames
    f2 = tmp_path / "b.yuv"
    f2.write_bytes(frame_bytes * 5)  # 5 frames

    # Load first file
    window.current_width = width
    window.current_height = height
    window.current_format = "I420"
    window.current_file_path = str(f1)
    window.reload_video()

    # Navigate to frame 2
    window.slider_frame.setValue(2)

    # Open second file
    window._save_current_tab_state()
    window.current_file_path = str(f2)
    window.reload_video()
    window._add_tab_for_current()

    # Switch back to first tab
    window._save_current_tab_state()
    window._current_tab_idx = 0
    window._restore_tab_state(0)

    assert window.current_file_path == str(f1)
    assert window.current_frame_idx == 2


def test_bookmarks(qapp, tmp_path):
    """Test bookmark add, remove, and navigation."""
    from video_viewer.main_window import MainWindow
    window = MainWindow()

    width, height = 4, 4
    frame_bytes = b'\x80' * (width * height + 2 * (width // 2) * (height // 2))
    f = tmp_path / "bm.yuv"
    f.write_bytes(frame_bytes * 10)  # 10 frames

    window.current_width = width
    window.current_height = height
    window.current_format = "I420"
    window.current_file_path = str(f)
    window.reload_video()

    # Add bookmarks
    window.slider_frame.setValue(3)
    window.toggle_bookmark()
    assert 3 in window.bookmarks

    window.slider_frame.setValue(7)
    window.toggle_bookmark()
    assert 7 in window.bookmarks
    assert len(window.bookmarks) == 2

    # Toggle again to remove
    window.toggle_bookmark()
    assert 7 not in window.bookmarks
    assert len(window.bookmarks) == 1

    # Navigation
    window.slider_frame.setValue(0)
    window.next_bookmark()
    assert window.slider_frame.value() == 3


def test_dark_theme_toggle(qapp):
    """Test dark theme toggle state."""
    from video_viewer.main_window import MainWindow
    window = MainWindow()

    initial = window._dark_theme
    window.toggle_theme()
    assert window._dark_theme != initial

    window.toggle_theme()
    assert window._dark_theme == initial


def test_bt709_color_matrix(qapp, tmp_path):
    """Test BT.709 color matrix produces different output than BT.601."""
    from video_viewer.video_reader import VideoReader
    import numpy as np

    width, height = 4, 4
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    # Create I420 with non-trivial color values
    y_data = np.full(y_size, 180, dtype=np.uint8).tobytes()
    u_data = np.full(uv_size, 100, dtype=np.uint8).tobytes()
    v_data = np.full(uv_size, 200, dtype=np.uint8).tobytes()
    frame_data = y_data + u_data + v_data

    f = tmp_path / "color.yuv"
    f.write_bytes(frame_data)

    reader601 = VideoReader(str(f), width, height, "I420")
    reader601.color_matrix = "BT.601"
    raw601 = reader601.seek_frame(0)
    frame601 = reader601.convert_to_rgb(raw601)

    reader709 = VideoReader(str(f), width, height, "I420")
    reader709.color_matrix = "BT.709"
    raw709 = reader709.seek_frame(0)
    frame709 = reader709.convert_to_rgb(raw709)

    # Different matrices should produce different RGB values
    assert frame601 is not None
    assert frame709 is not None
    assert not np.array_equal(frame601, frame709)

    reader601.close()
    reader709.close()


def test_recent_files(qapp):
    """Test recent files list management."""
    from video_viewer.main_window import MainWindow
    window = MainWindow()

    window._add_to_recent_files("/tmp/test_a.yuv")
    window._add_to_recent_files("/tmp/test_b.yuv")

    recent = window._settings.value("recent_files", [], type=list)
    assert "/tmp/test_b.yuv" in recent
    assert "/tmp/test_a.yuv" in recent
    # Most recent should be first
    assert recent[0] == "/tmp/test_b.yuv"

    # Clean up settings
    window._settings.remove("recent_files")


def test_scene_change_detection(qapp, tmp_path):
    """Test scene change detection finds changes between frames."""
    from video_viewer.main_window import MainWindow
    import numpy as np

    window = MainWindow()
    width, height = 4, 4
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    # 5 frames: 3 similar + 1 very different + 1 similar to #4
    frame_dark = b'\x10' * y_size + b'\x80' * (2 * uv_size)
    frame_bright = b'\xf0' * y_size + b'\x80' * (2 * uv_size)
    data = frame_dark * 3 + frame_bright + frame_dark

    f = tmp_path / "scene.yuv"
    f.write_bytes(data)

    window.current_width = width
    window.current_height = height
    window.current_format = "I420"
    window.current_file_path = str(f)
    window.reload_video()

    from unittest.mock import patch
    # Mock QInputDialog to return threshold=15.0 (low, to catch the big change)
    with patch('PySide6.QtWidgets.QInputDialog.getDouble', return_value=(15.0, True)):
        window.detect_scene_changes()
    # Should detect at least 1 scene change (frame 2→3 or 3→4)
    assert len(window._scene_changes) >= 1


def test_y4m_opens_without_dialog(qapp, tmp_path):
    """Test that Y4M files open without showing parameters dialog."""
    from video_viewer.main_window import MainWindow
    from unittest.mock import patch

    window = MainWindow()

    # Create a minimal Y4M file with header
    y4m_header = b"YUV4MPEG2 W4 H4 F30:1 Ip C420\n"
    frame_marker = b"FRAME\n"
    width, height = 4, 4
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_data = b'\x80' * (y_size + 2 * uv_size)
    y4m_content = y4m_header + frame_marker + frame_data

    f = tmp_path / "test.y4m"
    f.write_bytes(y4m_content)

    # show_parameters_dialog should NOT be called for Y4M
    with patch.object(window, 'show_parameters_dialog') as mock_dialog:
        window._open_file_in_tab(str(f))
        mock_dialog.assert_not_called()

    # Video should be loaded
    assert window.reader is not None


def test_guess_resolution(qapp, tmp_path):
    """Test file size → resolution guessing logic."""
    from video_viewer.main_window import MainWindow

    window = MainWindow()

    # Create a file that's exactly 1 frame of 1920x1080 I420
    # I420: w*h*1.5 bytes per frame
    width, height = 1920, 1080
    frame_size = int(width * height * 1.5)
    f = tmp_path / "test_1080p.yuv"
    f.write_bytes(b'\x00' * frame_size)

    result = window._guess_resolution(str(f))
    assert result is not None
    assert result[0] == 1920
    assert result[1] == 1080
    assert "I420" in result[2]
    assert result[3] == 1  # 1 frame


def test_guess_resolution_multiple_frames(qapp, tmp_path):
    """Test resolution guessing with multiple frames."""
    from video_viewer.main_window import MainWindow

    window = MainWindow()

    width, height = 640, 480
    frame_size = int(width * height * 1.5)  # I420
    num_frames = 10
    f = tmp_path / "test_vga.yuv"
    f.write_bytes(b'\x00' * (frame_size * num_frames))

    result = window._guess_resolution(str(f))
    assert result is not None
    assert result[0] == 640
    assert result[1] == 480
    assert result[3] == num_frames


def test_guess_resolution_no_match(qapp, tmp_path):
    """Test resolution guessing returns None for unrecognized sizes."""
    from video_viewer.main_window import MainWindow

    window = MainWindow()

    # Odd file size that doesn't match any resolution/format
    f = tmp_path / "test_odd.yuv"
    f.write_bytes(b'\x00' * 12345)

    result = window._guess_resolution(str(f))
    assert result is None


def test_guess_resolution_empty_file(qapp, tmp_path):
    """Test resolution guessing returns None for empty files."""
    from video_viewer.main_window import MainWindow

    window = MainWindow()

    f = tmp_path / "empty.yuv"
    f.write_bytes(b'')

    result = window._guess_resolution(str(f))
    assert result is None


def test_raw_file_opens_with_dialog(qapp, tmp_path):
    """Test that raw files show parameters dialog (not bypass it)."""
    from video_viewer.main_window import MainWindow
    from unittest.mock import patch

    window = MainWindow()

    width, height = 4, 4
    frame_bytes = b'\x80' * int(width * height * 1.5)
    f = tmp_path / "test.yuv"
    f.write_bytes(frame_bytes)

    # show_parameters_dialog SHOULD be called for raw files
    with patch.object(window, 'show_parameters_dialog', return_value=False) as mock_dialog:
        window._open_file_in_tab(str(f))
        mock_dialog.assert_called_once()


def test_colorize_channel(qapp):
    """Test false color conversion for each channel name."""
    import numpy as np
    from video_viewer.main_window import MainWindow

    window = MainWindow()
    gray = np.full((4, 4), 200, dtype=np.uint8)

    # R channel -> only red
    rgb = window._colorize_channel(gray, 'R')
    assert rgb.shape == (4, 4, 3)
    assert np.all(rgb[:, :, 0] == 200)
    assert np.all(rgb[:, :, 1] == 0)
    assert np.all(rgb[:, :, 2] == 0)

    # G channel -> only green
    rgb = window._colorize_channel(gray, 'G')
    assert np.all(rgb[:, :, 0] == 0)
    assert np.all(rgb[:, :, 1] == 200)
    assert np.all(rgb[:, :, 2] == 0)

    # B channel -> only blue
    rgb = window._colorize_channel(gray, 'B')
    assert np.all(rgb[:, :, 0] == 0)
    assert np.all(rgb[:, :, 1] == 0)
    assert np.all(rgb[:, :, 2] == 200)

    # Y channel -> grayscale (all equal)
    rgb = window._colorize_channel(gray, 'Y')
    assert np.all(rgb[:, :, 0] == 200)
    assert np.all(rgb[:, :, 1] == 200)
    assert np.all(rgb[:, :, 2] == 200)

    # U channel -> blue
    rgb = window._colorize_channel(gray, 'U')
    assert np.all(rgb[:, :, 0] == 0)
    assert np.all(rgb[:, :, 1] == 0)
    assert np.all(rgb[:, :, 2] == 200)

    # V channel -> red
    rgb = window._colorize_channel(gray, 'V')
    assert np.all(rgb[:, :, 0] == 200)
    assert np.all(rgb[:, :, 1] == 0)
    assert np.all(rgb[:, :, 2] == 0)


def test_split_view_component(qapp, tmp_path):
    """Test that comp_idx=4 produces a 2x2 composite image."""
    from video_viewer.main_window import MainWindow
    import numpy as np

    window = MainWindow()
    width, height = 8, 8
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_bytes = b'\x80' * (y_size + 2 * uv_size)

    f = tmp_path / "split.yuv"
    f.write_bytes(frame_bytes)

    window.current_width = width
    window.current_height = height
    window.current_format = "I420"
    window.current_file_path = str(f)
    window.reload_video()

    # Set to split view
    window.set_component(4)

    # Canvas should have an image set
    assert window.canvas.image is not None


def test_component_keyboard_shortcuts(qapp, tmp_path):
    """Test that keys 0-4 switch component view."""
    from video_viewer.main_window import MainWindow
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    window = MainWindow()
    width, height = 4, 4
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_bytes = b'\x80' * (y_size + 2 * uv_size)

    f = tmp_path / "keys.yuv"
    f.write_bytes(frame_bytes)

    window.current_width = width
    window.current_height = height
    window.current_format = "I420"
    window.current_file_path = str(f)
    window.reload_video()

    # Test each key
    for key_val, expected_comp in [
        (Qt.Key.Key_0, 0),
        (Qt.Key.Key_1, 1),
        (Qt.Key.Key_2, 2),
        (Qt.Key.Key_3, 3),
        (Qt.Key.Key_4, 4),
    ]:
        event = QKeyEvent(QKeyEvent.Type.KeyPress, key_val, Qt.KeyboardModifier.NoModifier)
        window.keyPressEvent(event)
        assert window.current_component == expected_comp, f"Key {key_val} should set component to {expected_comp}"


def test_sub_grid_toggle(qapp):
    """Test Shift+G cycles sub-grid sizes (0→2→3→4→0)."""
    from video_viewer.main_window import MainWindow
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    window = MainWindow()
    assert window.canvas.sub_grid_size == 0

    expected = [4, 8, 16, 0]
    for exp in expected:
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_G, Qt.KeyboardModifier.ShiftModifier)
        window.keyPressEvent(event)
        assert window.canvas.sub_grid_size == exp, f"Expected sub_grid_size={exp}, got {window.canvas.sub_grid_size}"


def test_sub_grid_neighborhood(qapp, tmp_path):
    """Test sub_grid_size=4 produces 6x6 neighborhood."""
    from video_viewer.video_reader import VideoReader
    import numpy as np

    width, height = 16, 16
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    y_data = np.arange(y_size, dtype=np.uint8).tobytes()
    uv_data = b'\x80' * (2 * uv_size)
    frame_data = y_data + uv_data

    f = tmp_path / "subgrid.yuv"
    f.write_bytes(frame_data)

    reader = VideoReader(str(f), width, height, "I420")
    raw = reader.seek_frame(0)

    # Pixel (6,6) with sub_grid_size=4 → cell origin (4,4), range (3,3)-(8,8) → 6x6
    info = reader.get_pixel_info(raw, 6, 6, sub_grid_size=4)
    nb = info['neighborhood']
    assert len(nb) == 6, f"Expected 6 rows, got {len(nb)}"
    assert len(nb[0]) == 6, f"Expected 6 cols, got {len(nb[0])}"

    # Without sub-grid → default 3x3
    info_default = reader.get_pixel_info(raw, 6, 6)
    nb_default = info_default['neighborhood']
    assert len(nb_default) == 3
    assert len(nb_default[0]) == 3

    reader.close()


def test_sub_grid_paint(qapp):
    """Test sub_grid_size property on ImageCanvas."""
    from video_viewer.main_window import ImageCanvas

    canvas = ImageCanvas()
    assert canvas.sub_grid_size == 0

    canvas.set_sub_grid(3)
    assert canvas.sub_grid_size == 3

    canvas.set_sub_grid(0)
    assert canvas.sub_grid_size == 0
