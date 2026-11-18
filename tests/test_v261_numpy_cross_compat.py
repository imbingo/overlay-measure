from __future__ import annotations

import numpy as np
import pytest

from overlay_measure.geometry_engine import _line_intersection, cross2d
from overlay_measure.line_detector import detect_approximate_line
from overlay_measure.models import DetectionParams, ImageData, MeasurementConfig, Roi


def test_cross2d_matches_scalar_and_broadcast_forms():
    assert cross2d((1.0, 0.0), (0.0, 1.0)) == pytest.approx(1.0)
    values = cross2d(np.array([1.0, 0.0]), np.array([[0.0, 2.0], [3.0, -4.0]]))
    assert values == pytest.approx([2.0, -4.0])


def test_line_intersection_without_numpy_2d_cross():
    assert _line_intersection([(0.0, 0.0), (10.0, 10.0)], [(0.0, 10.0), (10.0, 0.0)]) == pytest.approx((5.0, 5.0))


def test_approximate_line_fits_horizontal_edge():
    gray = np.full((120, 200), 40.0, dtype=np.float32)
    gray[60:, :] = 200.0
    image = ImageData("synthetic.png", gray, "synthetic.png")
    roi = Roi(30.0, 40.0, 140.0, 40.0, roi_type="Approximate Line")
    detection = detect_approximate_line("Mark1", "upper", image, roi, DetectionParams(), MeasurementConfig())
    assert detection.fitting_mode == "Line"
    assert detection.center_y_px == pytest.approx(59.5, abs=0.5)
    assert detection.residual_px < 0.2
    assert abs(detection.shape_params["line_angle_deg"]) < 0.5
