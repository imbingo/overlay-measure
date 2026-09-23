from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication

from overlay_measure.batch_image_store import BatchImageRef
from overlay_measure.ui_main import MainWindow


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
