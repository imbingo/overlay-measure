import numpy as np
import pytest

from overlay_measure.measurement_service import detect_manual_roi
from overlay_measure.models import DetectionParams, ImageData, MeasurementConfig, Roi
from overlay_measure.region_center_detector import _outer_roi_mask
from overlay_measure.result_exporter import build_detection_rows
from overlay_measure.measurement_units import ellipse_metrics_um


def halo_image(a=55, b=45, angle=27):
    yy, xx = np.mgrid[:240, :260]
    t = np.deg2rad(angle)
    dx, dy = xx - 130.3, yy - 119.7
    u, v = np.cos(t)*dx + np.sin(t)*dy, -np.sin(t)*dx + np.cos(t)*dy
    d = (np.sqrt((u/a)**2 + (v/b)**2) - 1) * b
    edge = 1 / (1 + np.exp(np.clip(-d / 0.4, -80, 80)))
    image = 8 + 210 * edge * np.exp(-np.maximum(d, 0) / 10)
    return (image + np.random.default_rng(227).normal(0, 0.3, image.shape)).astype(np.float32)


@pytest.mark.parametrize('polarity', ['Auto', 'Dark to Bright', 'Bright to Dark'])
@pytest.mark.parametrize('shape', ['Circle', 'Ellipse'])
def test_halo_center_axes_and_export(shape, polarity):
    a, b = (50, 50) if shape == 'Circle' else (55, 45)
    gray = halo_image(a, b)
    roi = Roi(40, 30, 180, 180, shape)
    config = MeasurementConfig(pixel_size_x_um=1, pixel_size_y_um=1)
    params = DetectionParams(polarity=polarity, gaussian_sigma_px=0.6)
    result = detect_manual_roi('Mark1', 'upper', ImageData('halo', gray, 'halo'), roi, params, config)
    assert result.center_x_px == pytest.approx(130.3, abs=0.2)
    assert result.center_y_px == pytest.approx(119.7, abs=0.2)
    assert result.diameter_um == pytest.approx(a+b, abs=0.8)
    if shape == 'Ellipse':
        assert result.ellipse_roundness_um == pytest.approx(a-b, abs=0.3)
        assert result.ellipse_major_um == pytest.approx(2*a, abs=0.8)
        row = build_detection_rows({'Mark1': {'upper': result}}, {}, config)[0]
        assert row['ellipse_roundness_um'] == result.ellipse_roundness_um
    else:
        # Circular ROIs now report the supplementary ellipse model too.
        assert result.ellipse_roundness_um == pytest.approx(0., abs=0.1)
        row = build_detection_rows({'Mark1': {'upper': result}}, {}, config)[0]
        assert row['ellipse_roundness_um'] == result.ellipse_roundness_um
        assert -90 <= row['roundness_angle_deg'] < 90


def test_rotated_ellipse_mask_matches_roi():
    roi = Roi(50, 60, 120, 60, 'Ellipse', angle_deg=35)
    mask, (x0, y0, x1, y1) = _outer_roi_mask((250, 250), roi)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    expected = roi.contains_points(np.column_stack((xx.ravel(), yy.ravel()))).reshape(mask.shape)
    assert np.array_equal(mask > 0, expected)


def test_no_target_still_fails():
    with pytest.raises(ValueError):
        detect_manual_roi('Mark1', 'upper', ImageData('blank', np.zeros((240, 260), np.float32), 'blank'),
                          Roi(40, 30, 180, 180, 'Ellipse'), DetectionParams(), MeasurementConfig())


def test_halo_non_square_pixels():
    config = MeasurementConfig(pixel_size_x_um=0.7, pixel_size_y_um=1.1)
    result = detect_manual_roi('Mark1', 'upper', ImageData('halo', halo_image(), 'halo'),
                              Roi(40, 30, 180, 180, 'Ellipse'),
                              DetectionParams(gaussian_sigma_px=0.6), config)
    expected = ellipse_metrics_um({'major_px': 110, 'minor_px': 90, 'angle_deg': 27}, config)
    assert result.ellipse_roundness_um == pytest.approx(expected['ellipse_roundness_um'], abs=0.3)
    assert result.ellipse_diameter_um == pytest.approx(expected['ellipse_diameter_um'], abs=0.8)
    assert '闭合边界' in result.shape_params['algorithm_path']
