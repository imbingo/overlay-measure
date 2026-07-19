from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox

from overlay_measure.ui_main import MainWindow
from overlay_measure.image_loader import load_image
from overlay_measure.models import MarkRecipe
from overlay_measure.recipe_manager import load_recipe


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"


def test_main_window_algorithm_path_status_button_smoke(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()

    assert "V1.5.7" in window.windowTitle()
    assert window.windowFlags() & Qt.FramelessWindowHint
    assert window.title_bar.height() == 46
    assert window.command_bar.objectName() == "commandBar"
    assert window.version_label.text() == "V1.5.7"
    assert window.recipe_manage_btn.text() == "配方管理"
    assert window.load_recipe_btn.text().startswith("当前配方：未加载")
    window.show_recipe_quick_menu()
    app.processEvents()
    assert window.recipe_quick_menu is not None
    assert window.recipe_quick_menu.tree.columnCount() == 5
    window.recipe_quick_menu.hide()
    assert not window.progress_bar.isHidden()
    assert not window.cancel_progress_btn.isEnabled()
    assert window.algorithm_path_button.text() == "查看"
    assert window.algorithm_path_summary_label.text().startswith("算法路径：")
    assert window.current_recipe_label.text() == "当前配方：未加载"
    assert window.main_splitter.count() == 2
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


def test_background_calculation_keeps_qt_event_loop_responsive(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    config, params, marks = load_recipe(str(SAMPLE_DIR / "demo_recipe.json"))
    config.workflow_mode = "Manual"
    window.config = config
    window.params = params
    window.marks = {"Mark1": marks[0], "Mark2": MarkRecipe("Mark2")}
    window.mark_images["Mark1"] = {
        "upper": load_image(str(SAMPLE_DIR / "sample_upper.png")),
        "lower": load_image(str(SAMPLE_DIR / "sample_lower.png")),
    }
    window.roi_sources["Mark1"] = {"upper": "manual", "lower": "manual"}
    window._push_config_to_ui()
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: QMessageBox.Ok)

    window.analyze_all_marks()
    assert window._calculation_running
    deadline = time.monotonic() + 10.0
    while window._calculation_running and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert not window._calculation_running
    assert "Mark1" in window.overlays
    assert not window.progress_bar.isHidden()
    assert not window.cancel_progress_btn.isEnabled()
    window.close()
    app.processEvents()
