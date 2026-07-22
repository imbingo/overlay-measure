from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

from overlay_measure.caliper_circle_detector import detect_caliper_circle
from overlay_measure.measurement_service import detect_manual_roi
from overlay_measure.models import DetectionParams, ImageData, MarkRecipe, MeasurementConfig, Roi
from overlay_measure.recipe_manager import load_recipe, save_recipe
from overlay_measure.ui_main import ImageCanvas


def _concentric_ring_image(
    center: tuple[float, float] = (101.3, 96.7),
    inner_radius: float = 31.0,
    outer_radius: float = 43.0,
    size: int = 210,
) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(xx - center[0], yy - center[1])
    inner_step = 1.0 / (1.0 + np.exp(-(radius - inner_radius) / 0.45))
    outer_step = 1.0 / (1.0 + np.exp(-(radius - outer_radius) / 0.45))
    return (210.0 - 155.0 * inner_step + 155.0 * outer_step).astype(np.float32)


@pytest.mark.parametrize(
    ("target_edge", "expected_radius"),
    [
        ("Near Inner Boundary", 31.0),
        ("Near Outer Boundary", 43.0),
    ],
)
def test_caliper_circle_selects_one_requested_concentric_edge(target_edge, expected_radius):
    center = (101.3, 96.7)
    image = _concentric_ring_image(center=center)
    roi = Roi(
        center[0] - 50.0,
        center[1] - 50.0,
        100.0,
        100.0,
        "Caliper Circle",
        0.45,
        target_edge,
        0.0,
        72,
        5.0,
        "Inner to Outer",
    )
    params = DetectionParams(
        gaussian_sigma_px=0.6,
        min_gradient=4.0,
        profile_step_px=0.2,
        use_ransac=True,
        residual_limit_px=0.5,
    )

    result = detect_caliper_circle(image, roi, params)

    assert result.center_x_px == pytest.approx(center[0], abs=0.15)
    assert result.center_y_px == pytest.approx(center[1], abs=0.15)
    assert result.radius_px == pytest.approx(expected_radius, abs=0.25)
    assert result.angular_coverage > 0.90


def test_strongest_edge_does_not_mix_inner_and_outer_edges_by_quadrant():
    size = 220
    center = (108.3, 101.7)
    yy, xx = np.mgrid[:size, :size]
    dx = xx - center[0]
    dy = yy - center[1]
    radius = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    direction = (np.cos(angle) + np.sin(angle)) / np.sqrt(2.0)
    inner_gray = 170.0 + 45.0 * direction
    outer_gray = 170.0 - 45.0 * direction
    ring_gray = np.full_like(radius, 60.0)
    inner_step = 1.0 / (1.0 + np.exp(-(radius - 32.0) / 0.45))
    outer_step = 1.0 / (1.0 + np.exp(-(radius - 44.0) / 0.45))
    image = inner_gray * (1.0 - inner_step) + ring_gray * inner_step * (1.0 - outer_step) + outer_gray * outer_step
    roi = Roi(
        center[0] - 52.0,
        center[1] - 52.0,
        104.0,
        104.0,
        "Caliper Circle",
        0.45,
        "Strongest Edge",
        0.0,
        72,
        5.0,
        "Inner to Outer",
    )
    params = DetectionParams(
        gaussian_sigma_px=0.6,
        min_gradient=4.0,
        profile_step_px=0.2,
        use_ransac=True,
        residual_limit_px=0.5,
    )

    result = detect_caliper_circle(image.astype(np.float32), roi, params)

    assert result.center_x_px == pytest.approx(center[0], abs=0.15)
    assert result.center_y_px == pytest.approx(center[1], abs=0.15)
    assert min(abs(result.radius_px - 32.0), abs(result.radius_px - 44.0)) < 0.25
    assert result.residual_px < 0.15
    assert result.angular_coverage > 0.90


def test_solid_circle_roi_selects_main_target_and_rejects_noise_components():
    size = 240
    center = (126.4, 113.6)
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(xx - center[0], yy - center[1])
    image = (210.0 - 165.0 / (1.0 + np.exp((radius - 38.0) / 0.55))).astype(np.float32)
    rng = np.random.default_rng(20260722)
    for x, y in rng.integers(25, 215, size=(45, 2)):
        image[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = 30.0

    roi = Roi(20.0, 15.0, 205.0, 205.0, "Circle")
    params = DetectionParams(
        gaussian_sigma_px=0.7,
        min_gradient=3.0,
        profile_half_width_px=2.0,
        profile_step_px=0.2,
        fitting_mode="Circle",
        upper_fitting_mode="Circle",
        use_ransac=True,
        residual_limit_px=0.6,
        min_edge_points=40,
    )
    config = MeasurementConfig(pixel_size_x_um=0.1, pixel_size_y_um=0.1)

    detection = detect_manual_roi(
        "Mark1",
        "upper",
        ImageData("synthetic", image, "synthetic"),
        roi,
        params,
        config,
    )

    assert detection.fitting_mode == "Circle"
    assert detection.center_x_px == pytest.approx(center[0], abs=0.30)
    assert detection.center_y_px == pytest.approx(center[1], abs=0.30)
    assert "主目标轮廓" in detection.warning


def test_caliper_diameter_mode_changes_reported_size_without_moving_center():
    center = (101.3, 96.7)
    image = _concentric_ring_image(center=center, inner_radius=31.0, outer_radius=43.0)
    roi = Roi(
        center[0] - 50.0,
        center[1] - 50.0,
        100.0,
        100.0,
        "Caliper Circle",
        0.45,
        "Near Outer Boundary",
        0.0,
        72,
        5.0,
        "Inner to Outer",
        "Average",
    )
    params = DetectionParams(
        gaussian_sigma_px=0.6,
        min_gradient=4.0,
        profile_step_px=0.2,
        use_ransac=True,
        residual_limit_px=0.5,
    )
    config = MeasurementConfig(pixel_size_x_um=0.1, pixel_size_y_um=0.1)
    image_data = ImageData("synthetic", image, "synthetic")

    average = detect_manual_roi("Mark1", "upper", image_data, roi, params, config)
    maximum = detect_manual_roi(
        "Mark1",
        "upper",
        image_data,
        replace(roi, diameter_mode="Maximum"),
        params,
        config,
    )

    assert maximum.center_x_px == pytest.approx(average.center_x_px, abs=1e-9)
    assert maximum.center_y_px == pytest.approx(average.center_y_px, abs=1e-9)
    assert average.diameter_um == pytest.approx(average.shape_params["average_diameter_um"])
    assert maximum.diameter_um == pytest.approx(maximum.shape_params["maximum_diameter_um"])
    assert maximum.diameter_um >= average.diameter_um


def test_image_overlay_uses_pixel_center_coordinates():
    app = QApplication.instance() or QApplication([])
    canvas = ImageCanvas("测试")
    canvas.resize(200, 100)
    image = ImageData("synthetic", np.zeros((10, 20), dtype=np.uint8), "synthetic")
    canvas.set_image(image)
    canvas._update_transform()

    wx, wy = canvas.image_to_widget(0.0, 0.0)
    assert wx == pytest.approx(canvas.offset_x + 0.5 * canvas.scale)
    assert wy == pytest.approx(canvas.offset_y + 0.5 * canvas.scale)
    ix, iy = canvas.widget_to_image_float(QPoint(round(wx), round(wy)))
    assert ix == pytest.approx(0.0, abs=0.11)
    assert iy == pytest.approx(0.0, abs=0.11)
    canvas.close()
    app.processEvents()


def test_recipe_round_trip_preserves_circle_diameter_mode(tmp_path):
    roi = Roi(10, 20, 80, 80, "Caliper Circle", 0.6, "Strongest Edge", 0, 64, 8, "Inner to Outer", "Maximum")
    path = tmp_path / "diameter_mode.json"
    save_recipe(str(path), MeasurementConfig(), DetectionParams(), [MarkRecipe("Mark1", upper_roi=roi)])

    _config, _params, marks = load_recipe(str(path))

    assert marks[0].upper_roi.diameter_mode == "Maximum"
