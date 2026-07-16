from __future__ import annotations

import os
from datetime import datetime

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from overlay_measure.export_naming import build_export_filename
from overlay_measure.image_loader import display_to_uint8
from overlay_measure.models import ImageData, OverlayResult, Roi
from overlay_measure.ui_main import MainWindow


def _image(name: str = "sample.png") -> ImageData:
    gray = np.zeros((64, 64), dtype=np.float32)
    return ImageData(name, gray, name, "uint8", 0.0, 255.0)


def test_recipe_roi_is_explicit_and_auto_workflow_ignores_it():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.mark_images["Mark1"]["upper"] = _image()
    window.marks["Mark1"].upper_roi = Roi(5, 5, 30, 30)
    window.marks["Mark1"].lower_roi = Roi(7, 7, 26, 26)
    window.roi_sources["Mark1"] = {"upper": "recipe", "lower": "recipe"}

    window._set_combo_value(window.workflow_combo, "Manual")
    assert window._recipe_roi_usage() == ["Mark1 上层", "Mark1 下层"]

    window.set_roi("Mark1", "upper", Roi(10, 10, 20, 20))
    assert window._roi_source("Mark1", "upper") == "manual"
    assert window._recipe_roi_usage() == ["Mark1 下层"]

    window._set_combo_value(window.workflow_combo, "Auto")
    assert window._recipe_roi_usage() == []
    assert "仅限 ROI" in window.fit_mode_combo.itemText(2)
    window.close()
    app.processEvents()


def test_clear_recipe_rois_preserves_manual_rois():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.marks["Mark1"].upper_roi = Roi(1, 1, 10, 10)
    window.marks["Mark1"].lower_roi = Roi(2, 2, 10, 10)
    window.roi_sources["Mark1"] = {"upper": "manual", "lower": "recipe"}
    window.clear_all_recipe_rois()
    assert window.marks["Mark1"].upper_roi is not None
    assert window.marks["Mark1"].lower_roi is None
    assert window._roi_source("Mark1", "upper") == "manual"
    assert window._roi_source("Mark1", "lower") == "none"
    window.close()
    app.processEvents()


def test_native_display_range_and_export_filename():
    gray = np.asarray([[0, 32768, 65535]], dtype=np.float32)
    image = ImageData("C:/data/upper.tif", gray, "upper.tif", "uint16", 0.0, 65535.0)
    displayed = display_to_uint8(image, enhanced=False)
    assert displayed.tolist() == [[0, 128, 255]]
    assert build_export_filename(
        r"D:\data\upper:mark.tif",
        now=datetime(2026, 7, 16, 12, 34, 56),
    ) == "upper_mark_Misalignment_Result_20260716_123456.xlsx"


def test_repeatability_export_keeps_failed_runs_and_statistics():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    success = OverlayResult("Mark1", 0, 0, 0.1, -0.2, 0.224, "Pass")
    window.batch_overlays["Mark1"] = [success]
    window.batch_run_records["Mark1"] = [
        {"run_index": 1, "upper_file": "upper_1.tif", "lower_file": "", "overlay": success, "error": ""},
        {"run_index": 2, "upper_file": "upper_2.tif", "lower_file": "", "overlay": None, "error": "边缘点不足"},
    ]
    rows = window._build_repeatability_export_rows()
    assert rows[0]["判定"] == "通过"
    assert rows[1]["判定"] == "失败"
    assert rows[1]["提示"] == "边缘点不足"
    assert rows[2]["次数"] == "统计"
    window.close()
    app.processEvents()
