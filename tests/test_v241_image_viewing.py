from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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


def test_magnifier_toggle_updates_both_canvases():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    window.magnifier_btn.setChecked(True)
    assert window.upper_canvas.magnifier_enabled
    assert window.lower_canvas.magnifier_enabled
    window.magnifier_btn.setChecked(False)
    assert not window.upper_canvas.magnifier_enabled
    assert not window.lower_canvas.magnifier_enabled
    window.close()
    app.processEvents()
