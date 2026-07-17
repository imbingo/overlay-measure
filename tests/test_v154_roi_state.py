from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from overlay_measure.export_naming import build_export_filename
from overlay_measure.image_loader import display_to_uint8
from overlay_measure.models import ImageData, OverlayResult, Roi
from overlay_measure.ui_main import ImageCanvas, MainWindow


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


def test_top_level_single_import_ignores_previous_batch_preview_images():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    batch_upper = _image("batch_upper.png")
    batch_lower = _image("batch_lower.png")
    single_upper = _image("single_upper.png")
    single_lower = _image("single_lower.png")

    window._set_combo_value(window.measurement_run_mode_combo, "Batch")
    window.batch_images["Mark1"]["upper"] = [batch_upper]
    window.batch_images["Mark1"]["lower"] = [batch_lower]
    window.batch_images["Mark2"]["upper"] = [batch_upper]
    window._set_image_for_layer("Mark1", "upper", batch_upper, "batch_preview")
    window._set_image_for_layer("Mark1", "lower", batch_lower, "batch_preview")
    window._set_image_for_layer("Mark2", "upper", batch_upper, "batch_preview")

    window._switch_to_single_measurement_after_top_import()
    window._set_image_for_layer("Mark1", "upper", single_upper, "single")
    window._set_image_for_layer("Mark1", "lower", single_lower, "single")

    snapshot = window._calculation_job_snapshot()
    assert snapshot["batch"] is False
    assert snapshot["mark_images"]["Mark1"]["upper"].path == "single_upper.png"
    assert snapshot["mark_images"]["Mark1"]["lower"].path == "single_lower.png"
    assert snapshot["mark_images"]["Mark2"]["upper"] is None
    assert window._is_batch_mode() is False
    window.close()
    app.processEvents()


def test_batch_images_can_be_appended_across_folders_without_duplicates():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    first = _image(r"D:\\run_01\\upper.tif")
    second = _image(r"E:\\run_02\\upper.tif")
    duplicate = _image(r"D:\\run_01\\upper.tif")

    assert window._append_batch_image_data("Mark1", "upper", [first]) == (1, 0)
    assert window._append_batch_image_data("Mark1", "upper", [second, duplicate]) == (1, 1)
    assert [image.path for image in window.batch_images["Mark1"]["upper"]] == [first.path, second.path]
    assert window._batch_source_folder_count(window.batch_images["Mark1"]["upper"]) == 2
    assert window._image_source("Mark1", "upper") == "batch_preview"
    window.close()
    app.processEvents()


def test_batch_folder_scan_supports_natural_sort_and_subfolders(tmp_path):
    root = tmp_path / "measurements"
    nested = root / "run_03"
    nested.mkdir(parents=True)
    (root / "upper_10.tif").touch()
    (root / "upper_2.tif").touch()
    (root / "notes.docx").touch()
    (nested / "upper_1.png").touch()

    direct = MainWindow._collect_batch_paths(str(root), recursive=False)
    recursive = MainWindow._collect_batch_paths(str(root), recursive=True)

    assert [Path(path).name for path in direct] == ["upper_2.tif", "upper_10.tif"]
    assert {Path(path).name for path in recursive} == {"upper_1.png", "upper_2.tif", "upper_10.tif"}
    assert all(not path.endswith("notes.docx") for path in recursive)


def test_reset_measurement_clears_images_results_and_rois():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    image = _image("upper.tif")
    overlay = OverlayResult("Mark1", 0, 0, 0.1, -0.1, 0.141, "Pass")
    window._set_image_for_layer("Mark1", "upper", image, "single")
    window.batch_images["Mark1"]["upper"] = [image]
    window.marks["Mark1"].upper_roi = Roi(1, 2, 20, 20)
    window.roi_sources["Mark1"]["upper"] = "manual"
    window.overlays["Mark1"] = overlay
    window.batch_overlays["Mark1"] = [overlay]
    window.batch_run_records["Mark1"] = [{"run_index": 1, "overlay": overlay}]

    window._clear_measurement_state()

    assert all(
        window.mark_images[mark_id][layer] is None
        for mark_id in ("Mark1", "Mark2")
        for layer in ("upper", "lower")
    )
    assert all(
        not window.batch_images[mark_id][layer]
        for mark_id in ("Mark1", "Mark2")
        for layer in ("upper", "lower")
    )
    assert not window.overlays
    assert not window.batch_overlays["Mark1"]
    assert not window.batch_run_records["Mark1"]
    assert window.marks["Mark1"].upper_roi is None
    assert window._roi_source("Mark1", "upper") == "none"
    assert window.upper_canvas.image is None
    assert window.lower_canvas.image is None
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


def _mouse_event(event_type, x, y, button, buttons):
    pos = QPointF(float(x), float(y))
    return QMouseEvent(event_type, pos, pos, pos, button, buttons, Qt.NoModifier)


def test_three_point_circle_keeps_selected_points_while_panning_for_third_point():
    app = QApplication.instance() or QApplication([])
    canvas = ImageCanvas("三点定圆测试")
    canvas.resize(600, 400)
    canvas.set_image(_image())
    canvas.show()
    app.processEvents()
    canvas.set_circle_pick_mode(True)
    canvas.circle_pick_points = [(12.0, 14.0), (38.0, 16.0)]

    start_x, start_y = canvas.pan_x, canvas.pan_y
    canvas.mousePressEvent(
        _mouse_event(QEvent.MouseButtonPress, 260, 190, Qt.MiddleButton, Qt.MiddleButton)
    )
    canvas.mouseMoveEvent(
        _mouse_event(QEvent.MouseMove, 315, 225, Qt.NoButton, Qt.MiddleButton)
    )
    canvas.mouseReleaseEvent(
        _mouse_event(QEvent.MouseButtonRelease, 315, 225, Qt.MiddleButton, Qt.NoButton)
    )

    assert canvas.pan_x == start_x + 55
    assert canvas.pan_y == start_y + 35
    assert canvas.circle_pick_points == [(12.0, 14.0), (38.0, 16.0)]
    assert canvas.circle_pick_mode
    assert canvas.cursor().shape() == Qt.CrossCursor
    canvas.close()
    app.processEvents()
