from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QMessageBox

from overlay_measure.ui_main import MainWindow


def test_main_window_algorithm_path_status_button_smoke(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()

    assert "V1.5.3" in window.windowTitle()
    assert window.algorithm_path_button.text() == "算法路径"
    assert not window.display_enhance_check.isChecked()
    assert window.result_tabs.count() == 3
    assert [window.result_tabs.tabText(i) for i in range(window.result_tabs.count())] == ["识别明细", "对位结果", "重复性分析"]
    assert "暂无测量结果" in window.algorithm_path_text
    assert "暂无测量结果" in window.algorithm_path_button.toolTip()
    assert not hasattr(window, "algorithm_path_label")
    label_texts = [label.text() for label in window.findChildren(QLabel)]
    assert "识别明细" not in label_texts
    assert "对位结果" not in label_texts

    captured = {}

    def fake_information(parent, title, text):
        captured["title"] = title
        captured["text"] = text
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "information", fake_information)
    window.show_algorithm_path_dialog()
    assert captured["title"] == "算法路径"
    assert "暂无测量结果" in captured["text"]

    window.close()
    app.processEvents()
