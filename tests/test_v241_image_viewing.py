from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from overlay_measure.recent_image_store import RecentImageStore
from overlay_measure.ui_main import MainWindow


def test_recent_image_store_persists_layer_and_filters_missing_paths(tmp_path):
    image = tmp_path / "array_upper.png"
    image.write_bytes(b"test")
    store = RecentImageStore(tmp_path / "recent.json", max_entries=3)
    store.add(str(image), "upper")
    store.add(str(image), "lower")
    assert [item["layer"] for item in store.entries()] == ["lower", "upper"]
    image.unlink()
    assert store.entries() == []


def test_actual_pixel_zoom_updates_visible_image_view():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    window.upper_canvas.fit_scale = 0.25
    window.upper_canvas.pixmap_cache = QPixmap(100, 100)
    window.set_canvas_actual_size()
    assert abs(window.upper_canvas.user_zoom * window.upper_canvas.fit_scale - 1.0) < 1e-9
    window.close()
    app.processEvents()
