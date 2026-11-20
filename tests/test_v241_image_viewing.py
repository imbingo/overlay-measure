from __future__ import annotations

import os
from pathlib import Path
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QEvent, Qt
from PySide6.QtTest import QTest

from overlay_measure.recent_image_store import RecentImageStore
from overlay_measure.ui_main import MainWindow
from overlay_measure.ui_components import ImageCanvas, ImageViewerDialog
from overlay_measure.image_loader import load_image


def test_recent_image_store_persists_layer_and_filters_missing_paths(tmp_path):
    image = tmp_path / "array_upper.png"
    image.write_bytes(b"test")
    store = RecentImageStore(tmp_path / "recent.json", max_entries=3)
    store.add(str(image), "upper")
    store.add(str(image), "lower")
    assert [item["layer"] for item in store.entries()] == ["lower", "upper"]
    image.unlink()
    assert store.entries() == []


def test_image_viewer_opens_from_current_canvas():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    sample = Path(__file__).resolve().parents[1] / "sample_data" / "v2_2_3_hole_array" / "hole_array_upper.png"
    assert window.import_dropped_image(str(sample), "upper") is None
    window.open_image_viewer()
    app.processEvents()
    assert window._image_viewer_dialog is not None
    assert window._image_viewer_dialog.windowTitle() == "图像查看器"
    assert len(window._image_viewer_dialog._canvases) == 1
    window._image_viewer_dialog.close()
    window.close()
    app.processEvents()


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("maximized", [False, True])
def test_image_viewer_fullscreen_button_f11_and_escape_preserve_view(dual, maximized):
    app = QApplication.instance() or QApplication([])
    sample = Path(__file__).resolve().parents[1] / "sample_data" / "sample_upper.png"
    source = ImageCanvas("upper", fixed_layer="upper")
    source.set_image(load_image(str(sample)))
    viewer = ImageViewerDialog([source, source] if dual else [source])
    try:
        viewer.showMaximized() if maximized else viewer.show()
        viewer.activateWindow()
        app.processEvents()
        assert viewer.windowFlags() & Qt.WindowMaximizeButtonHint
        for canvas in viewer._canvases:
            canvas.user_zoom, canvas.pan_x, canvas.pan_y = 3.0, 17.0, -9.0
        QTest.mouseClick(viewer.fullscreen_btn, Qt.LeftButton)
        app.processEvents()
        assert viewer.isFullScreen()
        assert viewer.fullscreen_btn.text() == "退出全屏"
        canvas = viewer._canvases[0]
        canvas.setFocus()
        QTest.keyClick(canvas, Qt.Key_Escape)
        app.processEvents()
        assert viewer.isVisible() and not viewer.isFullScreen()
        assert viewer.isMaximized() == maximized
        assert viewer.fullscreen_btn.text() == "全屏"
        QTest.keyClick(canvas, Qt.Key_F11)
        app.processEvents()
        assert viewer.isFullScreen()
        QTest.mouseClick(viewer.fullscreen_btn, Qt.LeftButton)
        app.processEvents()
        assert not viewer.isFullScreen()
        assert viewer.isMaximized() == maximized
        for current in viewer._canvases:
            assert (current.user_zoom, current.pan_x, current.pan_y) == (3.0, 17.0, -9.0)
            assert current.image is source.image
            assert not current.roi_editing_enabled
    finally:
        viewer.close()
        viewer.deleteLater()
        source.close()
        source.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
