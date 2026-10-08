import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from overlay_measure.batch_results import compact_detection
from overlay_measure.measurement_service import detect_manual_roi
from overlay_measure.measurement_units import radial_roundness_um, radial_diameter_statistics_um
from overlay_measure.models import DetectionParams, DetectionResult, ImageData, MeasurementConfig, Roi
from overlay_measure.result_exporter import build_detection_rows, export_results
from overlay_measure.ui_main import MainWindow


def contour(radii, config):
    radii = np.asarray(radii)
    angles = np.linspace(0, 2 * np.pi, len(radii), endpoint=False)
    return np.column_stack((
        120 + radii * np.cos(angles) / config.pixel_size_x_um,
        230 + radii * np.sin(angles) / config.pixel_size_y_um,
    ))


@pytest.mark.parametrize("scales", [(1, 1), (0.2, 0.7)])
@pytest.mark.parametrize("radii,expected", [([50.] * 64, 0.), ([50.2, 49.9] + [50.] * 62, 0.3)])
def test_physical_radial_roundness(scales, radii, expected):
    config = MeasurementConfig(pixel_size_x_um=scales[0], pixel_size_y_um=scales[1])
    result = radial_roundness_um(contour(radii, config), 120, 230, config)
    assert isinstance(result, float)
    assert result == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("points", [[], [(0, 0)], [(0, 0), (1, 1)], [(0, 0), (1, 1), (np.nan, 0)], [(np.inf, 0)] * 3])
def test_insufficient_finite_points_are_invalid(points):
    assert radial_roundness_um(points, 0, 0, MeasurementConfig()) is None


def test_service_uses_final_inliers_and_preserves_diameter_statistics(monkeypatch):
    config = MeasurementConfig(pixel_size_x_um=0.2, pixel_size_y_um=0.7)
    points = contour([50.2, 49.9] + [50.] * 62, config)
    rejected = np.array([[10000., 10000.]])
    cal = SimpleNamespace(
        center_x_px=120., center_y_px=230., radius_px=100., residual_px=0.01,
        confidence=0.99, edge_points=points, rejected_points=rejected,
        gradients=np.ones(64), rejected_gradients=np.ones(1), caliper_windows=[],
        average_diameter_px=200., maximum_diameter_px=201., minimum_diameter_px=199.,
        diameter_pv_px=2., angular_coverage=1., maximum_gap_deg=6.,
    )
    monkeypatch.setattr("overlay_measure.measurement_service.detect_caliper_circle", lambda *args: cal)
    detection = detect_manual_roi(
        "Mark1", "upper", ImageData("", np.zeros((10, 10)), "test"),
        Roi(0, 0, 10, 10, roi_type="Caliper Circle"), DetectionParams(), config,
    )
    assert detection.shape_params["roundness_um"] == pytest.approx(0.3, abs=1e-12)
    assert radial_roundness_um(np.vstack((points, rejected)), 120, 230, config) > 100
    for key, value in radial_diameter_statistics_um(points, 120, 230, config).items():
        assert detection.shape_params[key] == value
    assert detection.ellipse_roundness_um is None
    assert "RANSAC有效轮廓点 → 标定物理半径 → 最大半径 - 最小半径" in detection.shape_params["algorithm_path"]
    assert compact_detection(detection).shape_params["roundness_um"] == detection.shape_params["roundness_um"]


def detection(mode="CaliperCircle", **shape):
    return DetectionResult("Mark1", "upper", 120, 230, 12, 23, 100, 10, .1, .01, 64, .99, mode, shape_params=shape)


def test_export_roundness_and_ellipse_remain_independent(tmp_path):
    config = MeasurementConfig()
    rows = build_detection_rows({
        "caliper": {"upper": detection(roundness_um=0.123)},
        "ellipse": {"upper": detection("Ellipse", major_px=100., minor_px=80., angle_deg=0.)},
        "old": {"upper": detection()},
    }, {}, config)
    assert rows[0]["roundness_um"] == 0.123
    assert rows[0]["ellipse_roundness_um"] is None
    assert rows[1]["roundness_um"] is None
    assert rows[1]["ellipse_roundness_um"] == pytest.approx((100-80)*config.pixel_size_x_um/2)
    assert rows[2]["roundness_um"] is None
    for suffix in ("xlsx", "csv"):
        path = tmp_path / f"results.{suffix}"
        export_results(str(path), rows, config)
        table = pd.read_excel(path, sheet_name="识别明细") if suffix == "xlsx" else pd.read_csv(path)
        assert table.loc[0, "圆度(μm)"] == 0.123
        assert pd.isna(table.loc[1, "圆度(μm)"])
        assert "椭圆圆度(μm)" in table.columns


@pytest.mark.parametrize("batch", [False, True])
def test_ui_roundness_column_and_missing_values(batch, monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        entries = [{"mark_id": "Mark1", "layer": "upper", "detection": d,
                    "run_index": 1 if batch else None} for d in (
            detection(roundness_um=0.1234), detection("Ellipse"), detection(), None,
        )]
        monkeypatch.setattr(window, "_display_detection_entries", lambda: entries)
        window._refresh_tables()
        headers = [window.det_table.horizontalHeaderItem(i).text() for i in range(window.det_table.columnCount())]
        column = headers.index("圆度(μm)")
        assert window.det_table.item(0, column).text() == "0.123"
        for row in (1, 2, 3):
            assert window.det_table.item(row, column).text() == ""
        assert window.det_table.item(3, headers.index("质量状态")).text() == "异常"
        assert "最大半径 - 最小半径" in window.det_table.horizontalHeaderItem(column).toolTip()
    finally:
        window.close()
        app.processEvents()
