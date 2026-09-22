from __future__ import annotations

import os
from pathlib import Path

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
