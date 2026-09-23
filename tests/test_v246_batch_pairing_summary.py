from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication

from overlay_measure.batch_image_store import BatchImageRef
from overlay_measure.batch_pairing import summarize_batch_pairing
from overlay_measure.ui_main import MainWindow


def test_batch_pairing_summary_covers_single_dual_and_mismatch():
    assert summarize_batch_pairing(0, 0, True).status_text == "未导入"
    assert summarize_batch_pairing(3, 0, False).pairing_text == "单图，无需配对"
    assert summarize_batch_pairing(3, 3, True).pairing_text == "3组已配对"
    mismatch = summarize_batch_pairing(5, 3, True)
    assert mismatch.pairing_text == "2张未配对"
    assert mismatch.status_text == "需检查"
    assert not mismatch.ready


def test_batch_import_table_displays_pairing_status(tmp_path):
    app = QApplication.instance() or QApplication([])
    paths = []
    for index in range(3):
        path = tmp_path / f"upper_{index + 1}.png"
        Image.new("L", (12, 12), 40 + index).save(path)
        paths.append(path)

    window = MainWindow()
    window.mode_combo.setCurrentText("双图模式")
    window.batch_images["Mark1"]["upper"] = [BatchImageRef.from_path(str(path)) for path in paths]
    window.batch_images["Mark1"]["lower"] = [BatchImageRef.from_path(str(path)) for path in paths[:2]]
    window._refresh_batch_image_table()

    headers = [window.batch_image_table.horizontalHeaderItem(index).text()
               for index in range(window.batch_image_table.columnCount())]
    assert headers == ["Mark", "上层/单图", "下层", "配对", "状态"]
    assert window.batch_image_table.item(0, 3).text() == "1张未配对"
    assert window.batch_image_table.item(0, 4).text() == "需检查"
    window.close()
    app.processEvents()
