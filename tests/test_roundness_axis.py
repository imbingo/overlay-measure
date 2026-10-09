import numpy as np
import pytest

from overlay_measure.measurement_units import (
    contour_ellipse_metrics_um, ellipse_major_axis_angle_deg, normalize_axis_angle_deg,
    detection_roundness_display,
)
from overlay_measure.measurement_service import _fit_to_detection
from overlay_measure.circle_ellipse_fitter import FitResult
from overlay_measure.models import MeasurementConfig, Roi


def ellipse_points(angle, config, major=100., minor=90.):
    t = np.linspace(0, 2*np.pi, 128, endpoint=False)
    theta = np.deg2rad(angle)
    xy = np.column_stack((major/2*np.cos(t), minor/2*np.sin(t)))
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (xy @ rotation.T) / [config.pixel_size_x_um, config.pixel_size_y_um] + [200., 300.]


@pytest.mark.parametrize("angle", [-90., -89., -45., 0., 35., 89., 90., 125., 180.])
@pytest.mark.parametrize("scales", [(1., 1.), (.2, .7)])
def test_ellipse_physical_axis_and_roundness(angle, scales):
    config = MeasurementConfig(pixel_size_x_um=scales[0], pixel_size_y_um=scales[1])
    metrics = contour_ellipse_metrics_um(ellipse_points(angle, config), config)
    assert metrics["ellipse_roundness_um"] == pytest.approx(5., abs=1e-4)
    actual = metrics["roundness_angle_deg"]
    assert -90 <= actual < 90
    assert normalize_axis_angle_deg(actual-angle) == pytest.approx(0., abs=1e-4)


@pytest.mark.parametrize("points", [[], [(0., 0.)]*5, [(0., 0.), (1., 1.), (2., 2.)], [(np.nan, 1.)]*6])
def test_invalid_contour_has_no_ellipse_metrics(points):
    assert all(value is None for value in contour_ellipse_metrics_um(points, MeasurementConfig()).values())


def test_perfect_circle_has_no_unique_axis():
    config = MeasurementConfig(pixel_size_x_um=.2, pixel_size_y_um=.7)
    metrics = contour_ellipse_metrics_um(ellipse_points(0, config, minor=100), config)
    assert metrics["ellipse_roundness_um"] == pytest.approx(0, abs=1e-4)
    assert metrics["roundness_angle_deg"] is None


def test_existing_ellipse_model_angle_is_calibrated_without_changing_model():
    shape = {"major_px": 100., "minor_px": 90., "angle_deg": 30.}
    config = MeasurementConfig(pixel_size_x_um=.2, pixel_size_y_um=.7)
    theta = np.deg2rad(30.)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mapping = np.diag([.2, .7]) @ rotation @ np.diag([50., 45.])
    t = np.linspace(0, 2*np.pi, 128, endpoint=False)
    physical = np.column_stack([np.cos(t), np.sin(t)]) @ mapping.T
    points = physical / [.2, .7]
    expected = contour_ellipse_metrics_um(points, config)["roundness_angle_deg"]
    actual = ellipse_major_axis_angle_deg(shape, config)
    assert normalize_axis_angle_deg(actual-expected) == pytest.approx(0., abs=1e-4)
    assert shape["angle_deg"] == 30.


def test_normal_circle_gets_ellipse_roundness_without_changing_circle_fit():
    config = MeasurementConfig(pixel_size_x_um=.2, pixel_size_y_um=.7)
    points = ellipse_points(-25., config)
    fit = FitResult(200., 300., 100., .01, "Circle", .99, {"radius_px": 50.})
    result = _fit_to_detection("Mark1", "upper", fit, points, config, Roi(0, 0, 400, 600))
    value, angle = detection_roundness_display(result, config)
    assert value == pytest.approx(5., abs=1e-4)
    assert angle == pytest.approx(-25., abs=1e-4)
    assert result.center_x_px == 200.
    assert result.center_y_px == 300.
    assert result.diameter_px == 100.
