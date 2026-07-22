from __future__ import annotations

from dataclasses import replace
from typing import List

import cv2
import numpy as np

from .caliper_circle_detector import _profile_edge, detect_caliper_circle
from .circle_ellipse_fitter import fit_rectangle
from .measurement_service import attach_algorithm_path
from .measurement_units import (
    points_to_um_distances,
    radial_diameter_statistics_um,
    radial_diameter_residual_um,
    rotated_rect_size_um,
    scalar_px_to_um,
)
from .models import DetectionParams, DetectionResult, MeasurementConfig, Roi
from .quality_profiles import annotate_detection_quality
from .subpixel_edge_detector import _bilinear_sample


def _measurement_warning(quality_status: str, reason: str, config: MeasurementConfig) -> str:
    if quality_status != "Valid":
        return reason
    if config.recipe_validation_status != "Validated":
        return "试测/未验证配方"
    return ""


def refine_circle_candidate(
    gray: np.ndarray,
    candidate: DetectionResult,
    params: DetectionParams,
    config: MeasurementConfig,
) -> DetectionResult:
    radius = float(candidate.shape_params.get("radius_px", candidate.diameter_px / 2.0))
    half_width = max(4.0, float(config.production_search_half_width_px))
    outer = radius + half_width
    inner = max(0.0, radius - half_width)
    roi = Roi(
        candidate.center_x_px - outer,
        candidate.center_y_px - outer,
        outer * 2.0,
        outer * 2.0,
        "Caliper Circle",
        inner / max(outer, 1e-9),
        "Strongest Edge",
        0.0,
        int(config.production_caliper_count),
        float(config.production_caliper_width_px),
        "Inner to Outer",
    )
    precise = detect_caliper_circle(gray, roi, params)
    count = max(1, int(config.production_caliper_count))
    found_count = len(precise.edge_points) + len(precise.rejected_points)
    # Count alone can look healthy even when all accepted points occupy one arc.
    # Use the stricter angular coverage to prevent an incomplete arc from
    # producing a deceptively valid center.
    coverage = min(len(precise.edge_points) / count, precise.angular_coverage)
    rejected_ratio = len(precise.rejected_points) / max(1, found_count)
    diameter_um, residual_um = radial_diameter_residual_um(
        precise.edge_points,
        precise.center_x_px,
        precise.center_y_px,
        precise.radius_px,
        precise.residual_px,
        config,
    )
    diameter_statistics_um = radial_diameter_statistics_um(
        precise.edge_points,
        precise.center_x_px,
        precise.center_y_px,
        config,
    )
    distances_um = points_to_um_distances(precise.edge_points, precise.center_x_px, precise.center_y_px, config)
    radius_um = 0.5 * diameter_um
    max_deviation_um = float(np.max(np.abs(distances_um - radius_um))) if len(distances_um) else float("inf")
    detection = DetectionResult(
        mark_id=candidate.mark_id,
        layer=candidate.layer,
        center_x_px=precise.center_x_px,
        center_y_px=precise.center_y_px,
        center_x_um=precise.center_x_px * config.pixel_size_x_um,
        center_y_um=precise.center_y_px * config.pixel_size_y_um,
        diameter_px=2.0 * precise.radius_px,
        diameter_um=diameter_um,
        residual_px=precise.residual_px,
        residual_um=residual_um,
        edge_point_count=len(precise.edge_points),
        confidence=precise.confidence,
        fitting_mode="ProductionCircle",
        warning="",
        edge_points=[(float(x), float(y)) for x, y in precise.edge_points],
        rejected_points=[(float(x), float(y)) for x, y in precise.rejected_points],
        edge_gradients=[float(value) for value in precise.gradients],
        rejected_gradients=[float(value) for value in precise.rejected_gradients],
        shape_params={
            "measurement_stage": "production_measurement",
            "shape_type": "Circle",
            "radius_px": precise.radius_px,
            "average_diameter_px": precise.average_diameter_px,
            "maximum_diameter_px": precise.maximum_diameter_px,
            "minimum_diameter_px": precise.minimum_diameter_px,
            "diameter_pv_px": precise.diameter_pv_px,
            **diameter_statistics_um,
            "angular_coverage": precise.angular_coverage,
            "maximum_gap_deg": precise.maximum_gap_deg,
            "diameter_definition": "robust_average_circle",
            "diameter_mode": "Average",
            "reported_diameter_um": diameter_statistics_um["average_diameter_um"],
            "width_px": 2.0 * precise.radius_px,
            "height_px": 2.0 * precise.radius_px,
            "roi_type": "Auto Caliper Circle",
            "roi_inner_radius_px": inner,
            "roi_outer_radius_px": outer,
            "caliper_count": count,
            "caliper_width_px": config.production_caliper_width_px,
            "search_direction": "Inner to Outer",
            "caliper_windows": precise.caliper_windows,
            "candidate_contour_points": candidate.shape_params.get("contour_points", candidate.edge_points),
            "candidate_mode": candidate.fitting_mode,
            "coverage": coverage,
            "rejected_count": len(precise.rejected_points),
            "rejected_ratio": rejected_ratio,
            "max_deviation_um": max_deviation_um,
            "recipe_validation_status": config.recipe_validation_status,
        },
    )
    assessment = annotate_detection_quality(detection, config)
    detection.warning = _measurement_warning(
        "Valid" if assessment.valid else "Invalid",
        "；".join(assessment.reasons),
        config,
    )
    return attach_algorithm_path(detection, "Auto")


def _rectangle_caliper_points(
    gray: np.ndarray,
    candidate: DetectionResult,
    params: DetectionParams,
    config: MeasurementConfig,
):
    image = gray.astype(np.float32)
    sigma = max(0.0, float(params.gaussian_sigma_px))
    if sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    cx = float(candidate.center_x_px)
    cy = float(candidate.center_y_px)
    width = float(candidate.shape_params.get("width_px", candidate.diameter_px))
    height = float(candidate.shape_params.get("height_px", candidate.diameter_px))
    angle = np.deg2rad(float(candidate.shape_params.get("angle_deg", 0.0)))
    x_axis = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
    y_axis = np.asarray([-np.sin(angle), np.cos(angle)], dtype=np.float64)
    center = np.asarray([cx, cy], dtype=np.float64)
    half_search = max(4.0, float(config.production_search_half_width_px))
    search_step = max(0.05, float(params.profile_step_px))
    radial_samples = np.arange(0.0, 2.0 * half_search + search_step * 0.5, search_step, dtype=np.float32)
    width_caliper = max(1.0, float(config.production_caliper_width_px))
    tangent_offsets = np.arange(-width_caliper / 2.0, width_caliper / 2.0 + 0.5, 1.0, dtype=np.float32)
    per_side = max(4, int(np.ceil(config.production_caliper_count / 4.0)))
    points: List[Tuple[float, float]] = []
    gradients: List[float] = []
    side_counts = []
    windows = []
    side_defs = [
        (x_axis, y_axis, width / 2.0, height),
        (-x_axis, y_axis, width / 2.0, height),
        (y_axis, x_axis, height / 2.0, width),
        (-y_axis, x_axis, height / 2.0, width),
    ]
    search_params = replace(params, polarity=params.polarity if params.polarity != "Auto" else "Auto")
    for normal, tangent, edge_distance, side_length in side_defs:
        accepted = 0
        positions = np.linspace(
            -side_length / 2.0 + width_caliper,
            side_length / 2.0 - width_caliper,
            per_side,
        )
        for position in positions:
            nominal = center + normal * edge_distance + tangent * float(position)
            start = nominal - normal * half_search
            profile = []
            for distance in radial_samples:
                sample_center = start + normal * float(distance)
                values = [
                    _bilinear_sample(
                        image,
                        float((sample_center + tangent * float(offset))[0]),
                        float((sample_center + tangent * float(offset))[1]),
                    )
                    for offset in tangent_offsets
                ]
                values = [value for value in values if np.isfinite(value)]
                if not values:
                    profile = []
                    break
                profile.append(float(np.mean(values)))
            candidate_edge = _profile_edge(np.asarray(profile, dtype=np.float32), search_step, search_params.polarity) if len(profile) >= 5 else None
            gradient = 0.0
            if candidate_edge is not None:
                index, gradient = candidate_edge
                if gradient >= params.min_gradient:
                    point = start + normal * (index * search_step)
                    points.append((float(point[0]), float(point[1])))
                    gradients.append(float(gradient))
                    accepted += 1
            windows.append({
                "center_x": float(nominal[0]),
                "center_y": float(nominal[1]),
                "length": 2.0 * half_search,
                "width": width_caliper,
                "direction_x": float(normal[0]),
                "direction_y": float(normal[1]),
                "gradient": float(gradient),
                "accepted": gradient >= params.min_gradient,
            })
        side_counts.append(accepted)
    return np.asarray(points, dtype=np.float64), np.asarray(gradients, dtype=np.float64), side_counts, windows


def refine_rectangle_candidate(
    gray: np.ndarray,
    candidate: DetectionResult,
    params: DetectionParams,
    config: MeasurementConfig,
) -> DetectionResult:
    points, gradients, side_counts, windows = _rectangle_caliper_points(gray, candidate, params, config)
    if len(points) < 8 or min(side_counts) < 2:
        raise ValueError("四边卡尺有效边缘点不足")
    initial = fit_rectangle(points)
    width = float(initial.shape_params["width_px"])
    height = float(initial.shape_params["height_px"])
    angle = np.deg2rad(float(initial.shape_params["angle_deg"]))
    ct, st = np.cos(angle), np.sin(angle)
    x = points[:, 0] - initial.center_x_px
    y = points[:, 1] - initial.center_y_px
    xr = ct * x + st * y
    yr = -st * x + ct * y
    errors = np.minimum(np.abs(np.abs(xr) - width / 2.0), np.abs(np.abs(yr) - height / 2.0))
    threshold = max(0.5, float(params.residual_limit_px) * 2.5)
    mask = errors <= threshold
    inliers = points[mask]
    rejected = points[~mask]
    fit = fit_rectangle(inliers) if len(inliers) >= 8 else initial
    if len(inliers) < 8:
        inliers = points
        rejected = np.empty((0, 2), dtype=np.float64)
        mask = np.ones(len(points), dtype=bool)
    total_expected = max(1, 4 * max(4, int(np.ceil(config.production_caliper_count / 4.0))))
    coverage = len(inliers) / total_expected
    rejected_ratio = len(rejected) / max(1, len(points))
    width_um, height_um = rotated_rect_size_um(
        float(fit.shape_params.get("width_px", fit.diameter_px)),
        float(fit.shape_params.get("height_px", fit.diameter_px)),
        float(fit.shape_params.get("angle_deg", 0.0)),
        config,
    )
    residual_um = scalar_px_to_um(fit.residual_px, config)
    max_deviation_um = scalar_px_to_um(float(np.max(errors[mask])), config) if np.any(mask) else float("inf")
    detection = DetectionResult(
        mark_id=candidate.mark_id,
        layer=candidate.layer,
        center_x_px=fit.center_x_px,
        center_y_px=fit.center_y_px,
        center_x_um=fit.center_x_px * config.pixel_size_x_um,
        center_y_um=fit.center_y_px * config.pixel_size_y_um,
        diameter_px=fit.diameter_px,
        diameter_um=0.5 * (width_um + height_um),
        residual_px=fit.residual_px,
        residual_um=residual_um,
        edge_point_count=len(inliers),
        confidence=fit.confidence,
        fitting_mode="ProductionRectangle",
        warning="",
        edge_points=[(float(x), float(y)) for x, y in inliers],
        rejected_points=[(float(x), float(y)) for x, y in rejected],
        edge_gradients=[float(value) for value in gradients[mask]],
        rejected_gradients=[float(value) for value in gradients[~mask]],
        shape_params={
            **fit.shape_params,
            "measurement_stage": "production_measurement",
            "shape_type": "Rectangle",
            "roi_type": "Auto Four-Side Caliper",
            "caliper_count": total_expected,
            "caliper_width_px": config.production_caliper_width_px,
            "caliper_windows": windows,
            "candidate_contour_points": candidate.shape_params.get("contour_points", candidate.edge_points),
            "candidate_mode": candidate.fitting_mode,
            "coverage": coverage,
            "rejected_count": len(rejected),
            "rejected_ratio": rejected_ratio,
            "max_deviation_um": max_deviation_um,
            "recipe_validation_status": config.recipe_validation_status,
        },
    )
    assessment = annotate_detection_quality(detection, config)
    detection.warning = _measurement_warning(
        "Valid" if assessment.valid else "Invalid",
        "；".join(assessment.reasons),
        config,
    )
    return attach_algorithm_path(detection, "Auto")


def refine_candidate(
    gray: np.ndarray,
    candidate: DetectionResult,
    params: DetectionParams,
    config: MeasurementConfig,
) -> DetectionResult:
    try:
        if candidate.fitting_mode == "AutoRectangle":
            return refine_rectangle_candidate(gray, candidate, params, config)
        return refine_circle_candidate(gray, candidate, params, config)
    except Exception as exc:
        failed = replace(candidate)
        failed.fitting_mode = "ProductionRectangle" if candidate.fitting_mode == "AutoRectangle" else "ProductionCircle"
        failed.warning = f"精测失败：{exc}"
        failed.shape_params = {
            **candidate.shape_params,
            "measurement_stage": "production_measurement",
            "shape_type": "Rectangle" if candidate.fitting_mode == "AutoRectangle" else "Circle",
            "candidate_contour_points": candidate.shape_params.get("contour_points", candidate.edge_points),
            "quality_hard_failure": True,
            "failure_reason": failed.warning,
            "recipe_validation_status": config.recipe_validation_status,
        }
        annotate_detection_quality(failed, config)
        return attach_algorithm_path(failed, "Auto")
