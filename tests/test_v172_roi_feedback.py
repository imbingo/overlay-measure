from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox

from overlay_measure.image_loader import load_image
from overlay_measure.models import MarkRecipe
from overlay_measure.recipe_manager import load_recipe
from overlay_measure.ui_main import MainWindow


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"


def test_analyze_roi_button_reports_detection_errors(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._set_operation_mode("Engineering", authenticated=True)
    config, params, marks = load_recipe(str(SAMPLE_DIR / "demo_recipe.json"))
    config.workflow_mode = "Manual"
    window.config = config
    window.params = params
    window.marks = {"Mark1": marks[0], "Mark2": MarkRecipe("Mark2")}
    window.mark_images["Mark1"] = {
        "upper": load_image(str(SAMPLE_DIR / "sample_upper.png")),
        "lower": load_image(str(SAMPLE_DIR / "sample_lower.png")),
    }
    window._push_config_to_ui()
    window._refresh_all_widgets()

    dialogs = []

    def fake_critical(parent, title, text):
        dialogs.append((title, text))
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "critical", fake_critical)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)

    def fail_detection(*args, **kwargs):
        raise ValueError("有效边缘点不足：2 < 60")

    monkeypatch.setattr(window, "_detect_one", fail_detection)

    # Exercise the real QPushButton signal. clicked(False) previously became
    # show_message=False and hid every failure from the operator.
    window.analyze_roi_btn.click()
    app.processEvents()

    assert dialogs
    assert dialogs[0][0] == "ROI 区域分析失败"
    assert "有效边缘点不足" in dialogs[0][1]
    assert "请确认 ROI 覆盖目标边缘" in dialogs[0][1]
    assert window.analyze_roi_btn.text() == "分析 ROI"
    assert window.analyze_roi_btn.isEnabled()
    assert "ROI 分析失败" in window.progress_stage_label.text()

    window.close()
    app.processEvents()
