from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from overlay_measure.geometry_models import (
    CoordinateLabelDefinition,
    CoordinateSystemDefinition,
    GeometryFeatureDefinition,
    GeometryMeasurementDefinition,
)
from overlay_measure.batch_image_store import BatchImageRef
from overlay_measure.models import OverlayResult
from overlay_measure.result_table import ResultTableWidget
from overlay_measure.ui_main import MainWindow


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"


def _app():
    return QApplication.instance() or QApplication([])


def _overlay(mark_id: str, value: float) -> OverlayResult:
    return OverlayResult(mark_id, 0.0, 0.0, value, -value, abs(value), "Pass")


def test_result_table_copies_sparse_selection_as_tsv_and_headers():
    app = _app()
    table = ResultTableWidget()
    table.setColumnCount(3)
    table.setRowCount(2)
    table.setHorizontalHeaderLabels(["A", "B", "C"])
    for row in range(2):
        for column in range(3):
            from PySide6.QtWidgets import QTableWidgetItem
            table.setItem(row, column, QTableWidgetItem(f"{row}{column}"))
    table.item(0, 0).setSelected(True)
    table.item(1, 2).setSelected(True)

    table.copy_selection(False)
    assert app.clipboard().text() == "00\t\t\n\t\t12"
    table.copy_selection(True)
    assert app.clipboard().text() == "A\tB\tC\n00\t\t\n\t\t12"


def test_result_table_keeps_horizontal_scroll_and_unique_row_payloads():
    app = _app()
    table = ResultTableWidget()
    table.resize(260, 180)
    table.setColumnCount(8)
    table.setRowCount(2)
    table.setHorizontalHeaderLabels([f"Very wide column {index}" for index in range(8)])
    from PySide6.QtWidgets import QTableWidgetItem
    for row in range(2):
        for column in range(8):
            table.setItem(row, column, QTableWidgetItem("content-" * 8))
    table.set_row_payloads([
        {"kind": "batch_run", "run_index": 1},
        {"kind": "batch_run", "run_index": 2},
    ])
    table.resizeColumnsToContents()
    table.show()
    app.processEvents()

    assert table.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert table.horizontalScrollBar().maximum() > 0
    assert table.external_horizontal_bar.maximum() > 0
    table.selectRow(0)
    assert table.selected_row_payloads() == [{"kind": "batch_run", "run_index": 1}]
    table.close()


def test_all_main_result_tabs_expose_horizontal_scroll_when_needed():
    app = _app()
    window = MainWindow()
    window.resize(1120, 720)
    window.show()
    headers = [f"宽字段 {index}" for index in range(16)]
    rows = [["测量结果内容" * 5 for _ in headers]]
    tables = (window.det_table, window.overlay_table, window.geometry_table, window.repeat_table)
    for table in tables:
        window._fill_table(table, headers, rows, [{"kind": "test", "deletable": False}])
    for index, table in enumerate(tables):
        window.result_tabs.setCurrentIndex(index)
        app.processEvents()
        assert table.external_horizontal_bar.isVisible()
        assert table.external_horizontal_bar.maximum() > 0
    window.close()
    app.processEvents()


def test_batch_result_deletion_removes_whole_run_and_recomputes(monkeypatch):
    app = _app()
    window = MainWindow()
    window._set_operation_mode("Engineering", authenticated=True)
    window._set_combo_value(window.measurement_run_mode_combo, "Batch")
    for mark_id in ("Mark1", "Mark2"):
        records = []
        for index, value in enumerate((1.0, 2.0, 3.0), start=1):
            overlay = _overlay(mark_id, value)
            records.append({"run_index": index, "overlay": overlay, "detections": {}, "workflow": "Manual"})
        window.batch_run_records[mark_id] = records
        window.batch_overlays[mark_id] = [record["overlay"] for record in records]
        upper_paths = ["sample_upper.png", "sample_single.png", "sample_square_upper.png"]
        lower_paths = ["sample_lower.png", "sample_square_single.png", "sample_square_lower.png"]
        window.batch_images[mark_id] = {
            "upper": [BatchImageRef.from_path(SAMPLE_DIR / name) for name in upper_paths],
            "lower": [BatchImageRef.from_path(SAMPLE_DIR / name) for name in lower_paths],
        }
        window.overlays[mark_id] = window._mean_overlay(mark_id, window.batch_overlays[mark_id])
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    window._delete_result_rows([{"kind": "batch_run", "mark_id": "Mark1", "run_index": 2}])

    for mark_id in ("Mark1", "Mark2"):
        assert [record["run_index"] for record in window.batch_run_records[mark_id]] == [1, 2]
        assert [item.delta_x_um for item in window.batch_overlays[mark_id]] == [1.0, 3.0]
        assert window.overlays[mark_id].delta_x_um == 2.0
        assert [Path(item.path).name for item in window.batch_images[mark_id]["upper"]] == [
            "sample_upper.png", "sample_square_upper.png",
        ]
    window.close()
    app.processEvents()


def test_default_mode_allows_result_deletion_after_confirm(monkeypatch):
    app = _app()
    window = MainWindow()
    window.overlays["Mark1"] = _overlay("Mark1", 1.0)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)

    window._delete_result_rows([{"kind": "overlay", "mark_id": "Mark1", "workflow": "Manual"}])

    assert "Mark1" in window.overlays
    window.close()
    app.processEvents()


def test_geometry_batch_delete_cascades_by_stable_id(monkeypatch):
    app = _app()
    window = MainWindow()
    window._set_operation_mode("Engineering", authenticated=True)
    window.geometry_program.features = [GeometryFeatureDefinition("F1", "F1", "upper", "point")]
    window.geometry_program.coordinate_systems = [
        CoordinateSystemDefinition("CS1", "CS1", "upper", "two_points", ["F1", "F1"])
    ]
    window.geometry_program.measurements = [
        GeometryMeasurementDefinition("M1", "M1", "upper", "coordinate", ["F1"], "CS1")
    ]
    window.geometry_program.coordinate_labels = [
        CoordinateLabelDefinition("L1", "L1", "upper", "F1", "CS1")
    ]
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    window._delete_result_rows([
        {"kind": "geometry", "geometry_kind": "feature", "stable_id": "F1"}
    ])

    assert not window.geometry_program.features
    assert not window.geometry_program.coordinate_systems
    assert not window.geometry_program.measurements
    assert not window.geometry_program.coordinate_labels
    window.close()
    app.processEvents()
