from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from overlay_measure.models import DetectionParams, DetectionResult, MeasurementConfig
from overlay_measure.measurement_service import describe_algorithm_path
from overlay_measure.production_measurement import refine_candidate
from overlay_measure.recipe_manager import load_recipe, save_recipe
from overlay_measure.ui_components import ImageCanvas


def _candidate() -> DetectionResult:
    return DetectionResult(
        mark_id="Mark1-上-1",
        layer="upper",
        center_x_px=100.0,
        center_y_px=90.0,
        center_x_um=100.0,
        center_y_um=90.0,
        diameter_px=60.0,
        diameter_um=60.0,
        residual_px=0.1,
        residual_um=0.1,
        edge_point_count=100,
        confidence=0.9,
        fitting_mode="AutoCircle",
        shape_params={"width_px": 64.0, "height_px": 48.0, "angle_deg": 22.0},
    )


def test_auto_ellipse_refinement_reuses_semantic_roi(monkeypatch):
    captured = {}

    def fake_manual(mark_id, layer, image, roi, params, config):
        captured["roi_type"] = roi.roi_type
        return DetectionResult(
            mark_id=mark_id,
            layer=layer,
            center_x_px=100.0,
            center_y_px=90.0,
            center_x_um=100.0,
            center_y_um=90.0,
            diameter_px=55.0,
            diameter_um=55.0,
            residual_px=0.1,
            residual_um=0.1,
            edge_point_count=120,
            confidence=0.95,
            fitting_mode="Ellipse",
            shape_params={"major_px": 60.0, "minor_px": 50.0, "angle_deg": 22.0},
        )

    monkeypatch.setattr("overlay_measure.measurement_service.detect_manual_roi", fake_manual)
    result = refine_candidate(
        np.zeros((220, 240), dtype=np.float32),
        _candidate(),
        DetectionParams(),
        MeasurementConfig(),
        roi_type="Ellipse",
    )
    assert captured["roi_type"] == "Ellipse"
    assert result.fitting_mode == "Ellipse"
    assert result.shape_params["roi_type"] == "Ellipse"
    assert result.shape_params["measurement_stage"] == "automatic_roi_semantic_refine"
    assert "椭圆拟合" in describe_algorithm_path(result, "Auto")


def test_line_profile_uses_raw_pixels_and_physical_distance():
    app = QApplication.instance() or QApplication([])
    canvas = ImageCanvas("profile")
    canvas.image = type("Image", (), {"gray": np.arange(100, dtype=np.float32).reshape(10, 10)})()
    canvas.pixel_size_x_um = 2.0
    canvas.pixel_size_y_um = 1.0
    received = []
    canvas.profileMeasured.connect(received.append)
    canvas.profile_start_img = (0.0, 0.0)
    canvas.profile_current_img = (9.0, 0.0)
    canvas._emit_profile()
    assert len(received) == 1
    assert received[0]["values"][0] == 0.0
    assert received[0]["values"][-1] == 9.0
    assert received[0]["distances_um"][-1] == 18.0
    app.processEvents()


def test_auto_refine_roi_type_round_trips_in_recipe(tmp_path):
    path = tmp_path / "ellipse_auto_recipe.json"
    params = DetectionParams(auto_refine_roi_type="Ellipse")
    save_recipe(str(path), MeasurementConfig(), params, [])
    _, loaded_params, _ = load_recipe(str(path))
    assert loaded_params.auto_refine_roi_type == "Ellipse"
