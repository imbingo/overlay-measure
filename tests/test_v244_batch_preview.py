from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication

from overlay_measure.batch_image_store import BatchImageRef
from overlay_measure.ui_main import MainWindow
from overlay_measure.models import Roi


def test_batch_roi_selection_preserves_view_but_changing_image_resets_it(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    paths = [tmp_path / f"image_{i}.png" for i in range(2)]
    for i, path in enumerate(paths):
        Image.new("L", (240, 240), 20 + i).save(path)
    loads = []
    original_load = BatchImageRef.load

    def counted_load(ref):
        loads.append(ref.path)
        return original_load(ref)

    monkeypatch.setattr(BatchImageRef, "load", counted_load)
    window = MainWindow()
    try:
        window._set_operation_mode("Engineering", authenticated=True)
        window._set_combo_value(window.measurement_run_mode_combo, "Batch")
        refs = [BatchImageRef.from_path(str(path)) for path in paths]
        window.batch_images["Mark1"]["upper"] = refs
        mark = window.marks["Mark1"]
        first = mark.add_roi("upper", Roi(10, 10, 50, 50, "Circle"))
        second = mark.add_roi("upper", Roi(100, 100, 50, 50, "Circle"))
        window.select_roi_from_canvas("Mark1", "upper", first.roi_id)
        image = window.upper_canvas.image
        loads.clear()
        window.upper_canvas.user_zoom = 4.0
        window.upper_canvas.pan_x = 73.0
        window.upper_canvas.pan_y = -41.0

        window.select_roi_from_canvas("Mark1", "upper", second.roi_id)

        assert window.upper_canvas.image is image
        assert window.upper_canvas.user_zoom == 4.0
        assert (window.upper_canvas.pan_x, window.upper_canvas.pan_y) == (73.0, -41.0)
        assert loads == []

        window.show_next_batch_preview()
        assert Path(window.upper_canvas.image.path) == paths[1]
        assert window.upper_canvas.user_zoom == 1.0
        assert (window.upper_canvas.pan_x, window.upper_canvas.pan_y) == (0.0, 0.0)
        window.show_previous_batch_preview()
        assert Path(window.upper_canvas.image.path) == paths[0]

        # Reimporting even the same path is a new source and must reload pixels.
        previous = window.upper_canvas.image
        window.batch_images["Mark1"]["upper"][0] = BatchImageRef.from_path(str(paths[0]))
        window._sync_current_mark_images()
        assert window.upper_canvas.image is not previous
    finally:
        window.close()
        app.processEvents()


def test_batch_preview_buttons_switch_current_canvas_image(tmp_path):
    app = QApplication.instance() or QApplication([])
    paths = []
    for index in range(3):
        path = tmp_path / f"upper_{index + 1}.png"
        Image.new("L", (12, 12), 20 + index).save(path)
        paths.append(path)

    window = MainWindow()
    window._set_combo_value(window.measurement_run_mode_combo, "Batch")
    window.batch_images["Mark1"]["upper"] = [BatchImageRef.from_path(str(path)) for path in paths]
    window._batch_detail_last_single_index = 1
    window._sync_current_mark_images()
    window._refresh_all_widgets()

    assert not window.batch_preview_nav.isHidden()
    assert "1/3" in window.batch_preview_index_label.text()
    assert Path(window.upper_canvas.image.path) == paths[0]
    window.show_next_batch_preview()
    assert window._batch_detail_last_single_index == 2
    assert Path(window.upper_canvas.image.path) == paths[1]
    window.show_previous_batch_preview()
    assert window._batch_detail_last_single_index == 1
    window.close()
    app.processEvents()
