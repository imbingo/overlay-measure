from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication

from overlay_measure.batch_image_store import BatchImageRef
from overlay_measure.geometry_engine import execute_geometry_program
from overlay_measure.geometry_models import (
    GeometryFeatureDefinition,
    GeometryMeasurementDefinition,
    GeometryProgram,
)
from overlay_measure.measurement_engine import _manual_overlay
from overlay_measure.measurement_units import minimum_enclosing_circle_um
from overlay_measure.models import (
    DetectionParams,
    DetectionResult,
    ImageData,
    MarkRecipe,
    MeasurementConfig,
    Roi,
)
from overlay_measure.runtime_support import RecoveryStore
from overlay_measure.ui_main import MainWindow


def _detection(mark_id: str, layer: str, roi_id: str) -> DetectionResult:
    return DetectionResult(
        mark_id, layer, 10.0, 10.0, 1.0, 1.0,
        20.0, 2.0, 0.1, 0.01, 100, 0.95, "Circle",
        shape_params={"roi_id": roi_id, "quality_status": "Valid", "radius_px": 10.0},
    )


def test_minimum_enclosing_circle_is_not_maximum_feret():
    points = np.asarray([(0.0, 0.0), (10.0, 0.0), (5.0, 8.660254)], dtype=float)
    config = MeasurementConfig(pixel_size_x_um=1.0, pixel_size_y_um=1.0)
    _, radius_um, _ = minimum_enclosing_circle_um(points, config)
    assert 2.0 * radius_um == pytest.approx(11.547005, rel=1e-5)


def test_physical_circle_is_correct_with_anisotropic_pixels():
    config = MeasurementConfig(pixel_size_x_um=0.5, pixel_size_y_um=2.0)
    angles = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    physical = np.column_stack((20.0 + 10.0 * np.cos(angles), 30.0 + 10.0 * np.sin(angles)))
    pixels = np.column_stack((physical[:, 0] / 0.5, physical[:, 1] / 2.0))
    center_px, radius_um, _ = minimum_enclosing_circle_um(pixels, config)
    assert center_px == pytest.approx((40.0, 15.0), abs=1e-4)
    assert radius_um == pytest.approx(10.0, abs=1e-4)


def test_geometry_minimum_circle_reports_physical_diameter():
    config = MeasurementConfig(pixel_size_x_um=1.0, pixel_size_y_um=1.0)
    contour = [(0.0, 0.0), (10.0, 0.0), (5.0, 8.660254)]
    detection = _detection("Mark1", "upper", "upper-1")
    detection.edge_points = contour
    program = GeometryProgram(
        features=[GeometryFeatureDefinition("f1", "最小圆", "upper", "outer_circle_min", source="detection", detection_key="d1")],
        measurements=[GeometryMeasurementDefinition("m1", "直径", "upper", "diameter", ["f1"])],
    )
    result = execute_geometry_program(program, {"d1": detection}, config)
    assert result.measurements["m1"].value == pytest.approx(11.547005, rel=1e-5)


def test_manual_overlay_keeps_good_roi_when_another_fails(monkeypatch):
    mark = MarkRecipe("Mark1")
    good = mark.add_roi("upper", Roi(0, 0, 20, 20, "Circle"), roi_id="good")
    mark.add_roi("upper", Roi(30, 0, 20, 20, "Circle"), roi_id="bad")
    image = ImageData("test.png", np.zeros((64, 64), np.float32), "test.png")

    def fake_detect(mark_id, layer, image, roi, params, config, cancelled=None):
        if roi.x > 20:
            raise ValueError("synthetic failure")
        return _detection(mark_id, layer, good.roi_id)

    monkeypatch.setattr("overlay_measure.measurement_engine.detect_manual_roi", fake_detect)
    measured = _manual_overlay(
        "Mark1", mark, {"upper": image, "lower": None}, DetectionParams(), MeasurementConfig(), {},
    )
    assert list(measured["detections"]) == ["good"]
    assert measured["failures"][0]["roi_id"] == "bad"
    assert measured["failures"][0]["status"] == "Error"


def test_production_mode_blocks_canvas_and_backend_roi_edits():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.operation_mode == "Production"
    assert window.upper_canvas.roi_editing_enabled is False
    window.set_roi("Mark1", "upper", Roi(0, 0, 20, 20, "Circle"))
    assert window.marks["Mark1"].upper_rois == []
    window.close()


def test_mode_change_invalidates_batch_and_manual_results():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.roi_detections["Mark1"] = {"r1": _detection("Mark1", "upper", "r1")}
    window.batch_run_records["Mark1"] = [{"run_index": 1}]
    window.batch_overlays["Mark1"] = []
    window.on_mode_changed()
    assert window.roi_detections == {"Mark1": {}, "Mark2": {}}
    assert window.batch_run_records == {"Mark1": [], "Mark2": []}
    window.close()


def test_batch_reference_is_lazy(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"1234")
    reference = BatchImageRef.from_path(str(path))
    assert reference.path == str(path)
    assert reference.size_bytes == 4
    assert not hasattr(reference, "gray")


def test_recovery_store_archives_terminal_state(monkeypatch, tmp_path):
    monkeypatch.setattr("overlay_measure.runtime_support.app_data_root", lambda: tmp_path)
    store = RecoveryStore()
    job_id = store.save({"recipe_path": "recipe.json"})
    assert store.load()["state"] == "running"
    store.finish("completed", measurement_id="M-1")
    assert store.load() is None
    history = [json.loads(line) for line in (tmp_path / "recovery_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert history[-1]["job_id"] == job_id
    assert history[-1]["state"] == "completed"


import pytest
