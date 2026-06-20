from __future__ import annotations

import numpy as np
import pytest

from overlay_measure.auto_mark_detector import AutoDetectReport, detect_auto_marks_with_report
from overlay_measure.image_loader import load_image
from overlay_measure.measurement_units import radial_diameter_residual_um, rotated_rect_size_um
from overlay_measure.models import DetectionParams, MeasurementConfig, OverlayResult
from overlay_measure.rz_calculator import build_summary_rows


def test_non_square_pixel_radial_distances_use_xy_scales():
    config = MeasurementConfig(pixel_size_x_um=0.1, pixel_size_y_um=0.2)
    points = np.asarray([[10.0, 0.0], [0.0, 10.0]], dtype=np.float64)
    diameter_um, residual_um = radial_diameter_residual_um(points, 0.0, 0.0, 10.0, 0.0, config)

    assert diameter_um == pytest.approx(3.0)
    assert residual_um == pytest.approx(0.5)


def test_rotated_rectangle_size_uses_axis_angle():
    config = MeasurementConfig(pixel_size_x_um=0.1, pixel_size_y_um=0.2)

    width_um, height_um = rotated_rect_size_um(10.0, 20.0, 0.0, config)
    assert width_um == pytest.approx(1.0)
    assert height_um == pytest.approx(4.0)

    width_um, height_um = rotated_rect_size_um(10.0, 20.0, 90.0, config)
    assert width_um == pytest.approx(2.0)
    assert height_um == pytest.approx(2.0)


def test_auto_detect_report_exposes_limits():
    image = load_image("sample_data/sample_square_single.png")
    report = detect_auto_marks_with_report(
        image.gray,
        "upper",
        DetectionParams(canny_low=25, canny_high=90, min_gradient=2, residual_limit_px=0.6),
        0.1,
        0.1,
    )

    assert len(report.results) >= 1
    assert report.total_contours >= report.processed_contours
    assert report.max_results == 48


def test_auto_detect_report_warning_text():
    report = AutoDetectReport(
        results=[],
        processed_contours=0,
        selected_contours=96,
        total_contours=400,
        max_contours_per_mask=96,
        max_results=48,
        time_limit_s=4.0,
        truncated_by_time=True,
        truncated_by_contour_limit=True,
        truncated_by_result_limit=True,
    )

    text = report.warning_text()
    assert "时间上限" in text
    assert "轮廓过多" in text
    assert "候选达到上限" in text


def test_rz_summary_is_testable_outside_ui():
    config = MeasurementConfig(rz_layout="Y向前后分布", rz_distance_l_um=1000.0, rz_limit=600.0)
    rows = build_summary_rows(
        {
            "Mark1": OverlayResult("Mark1", 0, 0, 1.0, 0.0, 1.0, "Pass"),
            "Mark2": OverlayResult("Mark2", 0, 0, 1.5, 0.0, 1.5, "Pass"),
        },
        config,
    )

    rz_row = rows[-1]
    assert rz_row["项目"] == "Rz"
    assert rz_row["Rz(μrad)"] == pytest.approx(500.0)
    assert rz_row["判定"] == "通过"
