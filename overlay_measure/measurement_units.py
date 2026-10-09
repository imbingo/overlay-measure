from __future__ import annotations

from typing import Iterable, Tuple

import cv2

import numpy as np

from .models import MeasurementConfig


def mean_pixel_size_um(config: MeasurementConfig) -> float:
    return 0.5 * (float(config.pixel_size_x_um) + float(config.pixel_size_y_um))


def points_px_to_um(
    points_xy: Iterable[tuple[float, float]] | np.ndarray,
    config: MeasurementConfig,
) -> np.ndarray:
    """Convert image points to calibrated physical coordinates (Y remains image-down)."""
    points = np.asarray(list(points_xy) if not isinstance(points_xy, np.ndarray) else points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)
    scaled = points.copy()
    scaled[:, 0] *= float(config.pixel_size_x_um)
    scaled[:, 1] *= float(config.pixel_size_y_um)
    return scaled


def point_um_to_px(point_xy_um: tuple[float, float], config: MeasurementConfig) -> tuple[float, float]:
    return (
        float(point_xy_um[0]) / float(config.pixel_size_x_um),
        float(point_xy_um[1]) / float(config.pixel_size_y_um),
    )


def minimum_enclosing_circle_um(
    points_xy: Iterable[tuple[float, float]] | np.ndarray,
    config: MeasurementConfig,
) -> tuple[tuple[float, float], float, float]:
    """Return calibrated center(px), radius(um) and radial RMS residual(um)."""
    physical = points_px_to_um(points_xy, config)
    if len(physical) < 3:
        raise ValueError("轮廓点不足，无法计算最小外接圆")
    (cx_um, cy_um), _ = cv2.minEnclosingCircle(physical.astype(np.float32))
    radial = np.hypot(physical[:, 0] - cx_um, physical[:, 1] - cy_um)
    # OpenCV intentionally pads the returned radius by a small epsilon. Use the
    # calibrated farthest-point radius around its center to avoid a systematic
    # positive diameter bias in metrology output.
    radius_um = float(np.max(radial))
    residual_um = float(np.sqrt(np.mean((radial - float(radius_um)) ** 2)))
    return point_um_to_px((cx_um, cy_um), config), radius_um, residual_um


def robust_circle_um(
    points_xy: Iterable[tuple[float, float]] | np.ndarray,
    config: MeasurementConfig,
) -> tuple[tuple[float, float], float, float]:
    """Fit a robust circle in calibrated coordinates, never in anisotropic pixels."""
    from .circle_ellipse_fitter import fit_circle_geometric_robust, fit_circle_least_squares

    physical = points_px_to_um(points_xy, config)
    if len(physical) < 3:
        raise ValueError("轮廓点不足，无法计算稳健外轮廓圆")
    initial = fit_circle_least_squares(physical)
    cx_um, cy_um, radius_um, residual_um = fit_circle_geometric_robust(physical, initial[:3])
    return point_um_to_px((cx_um, cy_um), config), float(radius_um), float(residual_um)


def axis_scale_um_per_px(config: MeasurementConfig, angle_deg: float) -> float:
    """Physical scale for a pixel-space vector pointing at angle_deg."""
    theta = np.deg2rad(float(angle_deg))
    dx = np.cos(theta) * float(config.pixel_size_x_um)
    dy = np.sin(theta) * float(config.pixel_size_y_um)
    return float(np.hypot(dx, dy))


def scalar_px_to_um(value_px: float, config: MeasurementConfig) -> float:
    return float(value_px) * mean_pixel_size_um(config)


def points_to_um_distances(
    points_xy: Iterable[tuple[float, float]] | np.ndarray,
    center_x_px: float,
    center_y_px: float,
    config: MeasurementConfig,
) -> np.ndarray:
    points = np.asarray(list(points_xy) if not isinstance(points_xy, np.ndarray) else points_xy, dtype=np.float64)
    if points.size == 0:
        return np.empty((0,), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        return np.empty((0,), dtype=np.float64)
    dx_um = (points[:, 0] - float(center_x_px)) * float(config.pixel_size_x_um)
    dy_um = (points[:, 1] - float(center_y_px)) * float(config.pixel_size_y_um)
    return np.hypot(dx_um, dy_um)


def radial_roundness_um(
    points_xy: Iterable[tuple[float, float]] | np.ndarray,
    center_x_px: float,
    center_y_px: float,
    config: MeasurementConfig,
) -> float | None:
    """Radial envelope width (Rmax - Rmin), in um, of final inlier points.

    Calibrate X/Y separately; fewer than three finite radii are invalid.
    This is neither diameter PV nor ellipse model roundness.
    """
    distances = points_to_um_distances(points_xy, center_x_px, center_y_px, config)
    distances = distances[np.isfinite(distances)]
    if len(distances) < 3:
        return None
    return float(np.max(distances) - np.min(distances))


def normalize_axis_angle_deg(angle_deg: float) -> float:
    """An undirected axis angle in [-90, 90), with image Y pointing down."""
    return float((angle_deg + 90.0) % 180.0 - 90.0)


def contour_ellipse_metrics_um(points_xy, config: MeasurementConfig) -> dict:
    """Supplement circle metrology with an ellipse fit of final physical inliers."""
    invalid = {"ellipse_roundness_um": None, "roundness_angle_deg": None,
               "contour_ellipse_major_um": None, "contour_ellipse_minor_um": None}
    points = points_px_to_um(points_xy, config)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 5:
        return invalid
    points = points - np.mean(points, axis=0)
    if np.linalg.matrix_rank(points) < 2:
        return invalid
    try:
        _, (first, second), angle = cv2.fitEllipse(points.astype(np.float32))
    except cv2.error:
        return invalid
    if not np.isfinite([first, second, angle]).all() or min(first, second) <= 0:
        return invalid
    return {
        "ellipse_roundness_um": float(abs(first - second) / 2.0),
        "contour_ellipse_major_um": float(max(first, second)),
        "contour_ellipse_minor_um": float(min(first, second)),
        "roundness_angle_deg": None if np.isclose(first, second, rtol=1e-5, atol=1e-8)
        else normalize_axis_angle_deg(angle + (90.0 if second > first else 0.0)),
    }


def ellipse_major_axis_angle_deg(shape_params: dict, config: MeasurementConfig) -> float | None:
    """Physical major-axis orientation of the existing pixel ellipse model."""
    if "major_px" not in shape_params or "minor_px" not in shape_params:
        return None
    theta = np.deg2rad(float(shape_params.get("angle_deg", 0.0)))
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    transform = np.diag([config.pixel_size_x_um, config.pixel_size_y_um]) @ rotation
    model = transform @ np.diag([float(shape_params["major_px"]) ** 2,
                                 float(shape_params["minor_px"]) ** 2]) @ transform.T
    if not np.isfinite(model).all():
        return None
    values, vectors = np.linalg.eigh(model)
    if values[0] <= 0 or np.isclose(values[0], values[1], rtol=1e-5, atol=1e-8):
        return None
    direction = vectors[:, -1]
    return normalize_axis_angle_deg(float(np.rad2deg(np.arctan2(direction[1], direction[0]))))


def detection_roundness_display(detection, config: MeasurementConfig) -> tuple[float | None, float | None]:
    """Select the ROI's own roundness and major-axis angle for one detail column."""
    if detection.fitting_mode == "Ellipse":
        value = detection.ellipse_roundness_um
        if value is None:
            value = ellipse_metrics_um(detection.shape_params, config).get("ellipse_roundness_um")
        return value, ellipse_major_axis_angle_deg(detection.shape_params, config)
    if detection.fitting_mode in {"Circle", "CaliperCircle", "ProductionCircle", "AutoCircle"}:
        return detection.shape_params.get("ellipse_roundness_um"), detection.shape_params.get("roundness_angle_deg")
    return None, None


def radial_diameter_residual_um(
    points_xy: Iterable[tuple[float, float]] | np.ndarray,
    center_x_px: float,
    center_y_px: float,
    fallback_radius_px: float,
    fallback_residual_px: float,
    config: MeasurementConfig,
) -> Tuple[float, float]:
    distances_um = points_to_um_distances(points_xy, center_x_px, center_y_px, config)
    if len(distances_um):
        return float(2.0 * np.mean(distances_um)), float(np.std(distances_um))
    return (
        2.0 * scalar_px_to_um(float(fallback_radius_px), config),
        scalar_px_to_um(float(fallback_residual_px), config),
    )


def maximum_feret_diameter_um(
    points_xy: Iterable[tuple[float, float]] | np.ndarray,
    config: MeasurementConfig,
) -> float:
    """Maximum distance between contour points in calibrated physical units."""
    points = np.asarray(list(points_xy) if not isinstance(points_xy, np.ndarray) else points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        return 0.0
    scaled = points.copy()
    scaled[:, 0] *= float(config.pixel_size_x_um)
    scaled[:, 1] *= float(config.pixel_size_y_um)
    maximum = 0.0
    for index in range(len(scaled)):
        distances = np.hypot(
            scaled[index + 1 :, 0] - scaled[index, 0],
            scaled[index + 1 :, 1] - scaled[index, 1],
        )
        if len(distances):
            maximum = max(maximum, float(np.max(distances)))
    return maximum


def radial_diameter_statistics_um(
    points_xy: Iterable[tuple[float, float]] | np.ndarray,
    center_x_px: float,
    center_y_px: float,
    config: MeasurementConfig,
) -> dict[str, float]:
    distances = points_to_um_distances(points_xy, center_x_px, center_y_px, config)
    if len(distances) == 0:
        return {
            "average_diameter_um": 0.0,
            "maximum_diameter_um": 0.0,
            "minimum_diameter_um": 0.0,
            "diameter_pv_um": 0.0,
        }
    average = 2.0 * float(np.mean(distances))
    minimum = 2.0 * float(np.min(distances))
    maximum = maximum_feret_diameter_um(points_xy, config)
    return {
        "average_diameter_um": average,
        "maximum_diameter_um": maximum,
        "minimum_diameter_um": minimum,
        "diameter_pv_um": max(0.0, maximum - minimum),
    }


def rotated_rect_size_um(
    width_px: float,
    height_px: float,
    angle_deg: float,
    config: MeasurementConfig,
) -> Tuple[float, float]:
    width_um = float(width_px) * axis_scale_um_per_px(config, angle_deg)
    height_um = float(height_px) * axis_scale_um_per_px(config, float(angle_deg) + 90.0)
    return width_um, height_um


def ellipse_metrics_um(shape_params: dict, config: MeasurementConfig) -> dict[str, float]:
    """Return calibrated ellipse axes and derived metrology values.

    OpenCV reports full axis lengths in pixel coordinates.  With anisotropic
    pixels each axis must be calibrated along its own fitted direction before
    the physical major/minor ordering is applied.
    """
    if "major_px" not in shape_params or "minor_px" not in shape_params:
        return {}
    angle_deg = float(shape_params.get("angle_deg", 0.0))
    first_axis_um = float(shape_params["major_px"]) * axis_scale_um_per_px(config, angle_deg)
    second_axis_um = float(shape_params["minor_px"]) * axis_scale_um_per_px(config, angle_deg + 90.0)
    major_um = max(first_axis_um, second_axis_um)
    minor_um = min(first_axis_um, second_axis_um)
    return {
        "ellipse_major_um": major_um,
        "ellipse_minor_um": minor_um,
        "ellipse_diameter_um": 0.5 * (major_um + minor_um),
        "ellipse_roundness_um": 0.5 * (major_um - minor_um),
    }


def equivalent_size_um_from_shape(shape_params: dict, diameter_px: float, config: MeasurementConfig) -> float:
    if "major_px" in shape_params and "minor_px" in shape_params:
        return ellipse_metrics_um(shape_params, config)["ellipse_diameter_um"]
    if "width_px" in shape_params and "height_px" in shape_params:
        width_um, height_um = rotated_rect_size_um(
            float(shape_params["width_px"]),
            float(shape_params["height_px"]),
            float(shape_params.get("angle_deg", 0.0)),
            config,
        )
        return 0.5 * (width_um + height_um)
    if "radius_px" in shape_params:
        return 2.0 * scalar_px_to_um(float(shape_params["radius_px"]), config)
    return scalar_px_to_um(float(diameter_px), config)
