from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .circle_ellipse_fitter import (
    circle_diameter_statistics,
    fit_circle_geometric_robust,
    fit_circle_least_squares,
    fit_circle_ransac,
)
from .models import DetectionParams, Roi
from .subpixel_edge_detector import _bilinear_sample, _quadratic_peak_offset


@dataclass
class CaliperCircleResult:
    center_x_px: float
    center_y_px: float
    radius_px: float
    residual_px: float
    confidence: float
    edge_points: np.ndarray
    rejected_points: np.ndarray
    gradients: np.ndarray
    rejected_gradients: np.ndarray
    inlier_mask: np.ndarray
    caliper_windows: List[dict]
    average_diameter_px: float
    maximum_diameter_px: float
    minimum_diameter_px: float
    diameter_pv_px: float
    angular_coverage: float
    maximum_gap_deg: float


def _profile_edge_candidates(profile: np.ndarray, step: float, polarity: str):
    """Return every local gradient peak in one radial profile."""
    if len(profile) < 3:
        return []
    grad = np.gradient(profile, step)
    if polarity == "Dark to Bright":
        score = grad
    elif polarity == "Bright to Dark":
        score = -grad
    else:
        score = np.abs(grad)
    candidates = []
    for index in range(1, len(score) - 1):
        value = float(score[index])
        if value <= 0 or value < float(score[index - 1]) or value < float(score[index + 1]):
            continue
        offset = _quadratic_peak_offset(
            float(score[index - 1]),
            value,
            float(score[index + 1]),
        )
        candidates.append((float(index + offset), value))
    if not candidates:
        index = int(np.argmax(score))
        if float(score[index]) > 0:
            candidates.append((float(index), float(score[index])))
    return candidates


def _profile_edge(profile: np.ndarray, step: float, polarity: str):
    """Return the strongest radial gradient location and magnitude in one caliper."""
    candidates = _profile_edge_candidates(profile, step, polarity)
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[1])


def _initial_candidate(group: list[dict], target_edge: str) -> dict:
    if target_edge == "Near Inner Boundary":
        return min(group, key=lambda candidate: candidate["roi_radius"])
    if target_edge == "Near Outer Boundary":
        return max(group, key=lambda candidate: candidate["roi_radius"])
    return max(group, key=lambda candidate: candidate["gradient"])


def _select_consistent_strongest(groups: list[list[dict]], initial_points: np.ndarray, radial_step: float):
    """Keep one coherent circular edge instead of mixing local inner/outer peaks."""
    if len(initial_points) < 3:
        return [_initial_candidate(group, "Strongest Edge") for group in groups if group]
    # Start from radius relative to the user-defined ROI center. Fitting the
    # locally strongest points first is unsafe: if half of the calipers choose
    # an inner edge and the other half choose an outer edge, that mixed cloud
    # itself creates a shifted false circle.
    bin_width = max(0.35, 2.0 * float(radial_step))
    all_radii = []
    all_weights = []
    for group in groups:
        if not group:
            continue
        group_max = max(candidate["gradient"] for candidate in group)
        for candidate in group:
            all_radii.append(float(candidate["roi_radius"]))
            all_weights.append(candidate["gradient"] / max(group_max, 1e-12))
    if not all_radii:
        return []
    minimum = min(all_radii) - bin_width
    maximum = max(all_radii) + bin_width
    bins = max(3, int(np.ceil((maximum - minimum) / bin_width)))
    histogram, edges = np.histogram(all_radii, bins=bins, range=(minimum, maximum), weights=all_weights)
    smoothed = np.convolve(histogram, np.asarray([1.0, 2.0, 1.0]), mode="same")
    mode_index = int(np.argmax(smoothed))
    mode_radius = 0.5 * (float(edges[mode_index]) + float(edges[mode_index + 1]))

    selected = [
        min(group, key=lambda candidate: (abs(candidate["roi_radius"] - mode_radius), -candidate["gradient"]))
        for group in groups
        if group
    ]
    # Once one radius layer has been selected, refine the circle and reassign
    # each caliper to the candidate closest to that circle. This tolerates a
    # moderately off-center three-point initialization without mixing layers.
    for _ in range(3):
        if len(selected) < 3:
            break
        try:
            points = np.asarray([(candidate["x"], candidate["y"]) for candidate in selected], dtype=np.float64)
            fit_cx, fit_cy, fit_radius, _ = fit_circle_least_squares(points)
            fit_cx, fit_cy, fit_radius, _ = fit_circle_geometric_robust(
                points,
                (fit_cx, fit_cy, fit_radius),
                max_iterations=12,
            )
        except Exception:
            break
        reassigned = []
        for group in groups:
            if not group:
                continue
            group_max = max(candidate["gradient"] for candidate in group)

            def candidate_score(candidate):
                radial_error = abs(
                    float(np.hypot(candidate["x"] - fit_cx, candidate["y"] - fit_cy)) - fit_radius
                )
                gradient_bonus = 0.15 * bin_width * candidate["gradient"] / max(group_max, 1e-12)
                return radial_error - gradient_bonus

            reassigned.append(min(group, key=candidate_score))
        if all(left is right for left, right in zip(selected, reassigned)):
            break
        selected = reassigned
    return selected


def _angular_statistics(points: np.ndarray, center_x: float, center_y: float, expected_count: int):
    if len(points) == 0:
        return 0.0, 360.0
    angles = np.mod(np.arctan2(points[:, 1] - center_y, points[:, 0] - center_x), 2.0 * np.pi)
    sorted_angles = np.sort(angles)
    wrapped = np.concatenate([sorted_angles, sorted_angles[:1] + 2.0 * np.pi])
    maximum_gap = float(np.max(np.diff(wrapped))) if len(wrapped) > 1 else 2.0 * np.pi
    coverage = float(np.clip(1.0 - maximum_gap / (2.0 * np.pi), 0.0, 1.0))
    count_coverage = float(np.clip(len(points) / max(1, expected_count), 0.0, 1.0))
    return min(coverage, count_coverage), float(np.rad2deg(maximum_gap))


def detect_caliper_circle(gray: np.ndarray, roi: Roi, params: DetectionParams) -> CaliperCircleResult:
    """Extract one radial edge per caliper and fit the resulting circular contour."""
    r = roi.normalized()
    cx, cy = r.center()
    inner = r.inner_radius()
    outer = r.outer_radius()
    if outer <= inner + 2.0:
        raise ValueError("卡尺找圆 ROI 的内外圆间距太小")

    sigma = max(0.0, float(params.gaussian_sigma_px))
    image = gray.astype(np.float32)
    if sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    count = int(np.clip(getattr(r, "caliper_count", 64), 4, 720))
    width = float(max(1.0, getattr(r, "caliper_width_px", 8.0)))
    direction = getattr(r, "search_direction", "Inner to Outer")
    polarity = getattr(params, "polarity", "Auto")
    radial_step = max(0.05, float(getattr(params, "profile_step_px", 0.25)))
    tangent_step = 1.0

    length = outer - inner
    radial_samples = np.arange(0.0, length + radial_step * 0.5, radial_step, dtype=np.float32)
    tangent_offsets = np.arange(-width / 2.0, width / 2.0 + tangent_step * 0.5, tangent_step, dtype=np.float32)
    if len(tangent_offsets) < 2:
        tangent_offsets = np.array([0.0], dtype=np.float32)

    candidate_groups: list[list[dict]] = []
    windows: List[dict] = []

    for index in range(count):
        angle = 2.0 * np.pi * index / count
        radial = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        tangent = np.array([-np.sin(angle), np.cos(angle)], dtype=np.float64)
        if direction == "Outer to Inner":
            start = np.array([cx, cy], dtype=np.float64) + radial * outer
            search_vector = -radial
        else:
            start = np.array([cx, cy], dtype=np.float64) + radial * inner
            search_vector = radial

        profile = []
        for sample_distance in radial_samples:
            sample_center = start + search_vector * float(sample_distance)
            samples = [
                _bilinear_sample(
                    image,
                    float((sample_center + tangent * float(offset))[0]),
                    float((sample_center + tangent * float(offset))[1]),
                )
                for offset in tangent_offsets
            ]
            samples = [value for value in samples if np.isfinite(value)]
            if not samples:
                profile = []
                break
            profile.append(float(np.mean(samples)))

        profile_candidates = (
            _profile_edge_candidates(np.asarray(profile, dtype=np.float32), radial_step, polarity)
            if len(profile) >= 5
            else []
        )
        group = []
        for sub_index, gradient in profile_candidates:
            if gradient < float(params.min_gradient):
                continue
            point = start + search_vector * (sub_index * radial_step)
            roi_radius = inner + sub_index * radial_step if direction != "Outer to Inner" else outer - sub_index * radial_step
            group.append(
                {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "gradient": float(gradient),
                    "roi_radius": float(roi_radius),
                    "window_index": index,
                }
            )
        candidate_groups.append(group)
        strongest_gradient = max((candidate["gradient"] for candidate in group), default=0.0)

        windows.append(
            {
                "angle": float(angle),
                "center_x": float(cx + radial[0] * (inner + outer) * 0.5),
                "center_y": float(cy + radial[1] * (inner + outer) * 0.5),
                "length": float(length),
                "width": float(width),
                "gradient": float(strongest_gradient),
                "candidate_count": len(group),
                "accepted": False,
            }
        )

    target_edge = getattr(r, "target_edge", "Strongest Edge") or "Strongest Edge"
    initial = [_initial_candidate(group, target_edge) for group in candidate_groups if group]
    initial_points = np.asarray([(candidate["x"], candidate["y"]) for candidate in initial], dtype=np.float64)
    if target_edge in {"Strongest Edge", "All Edges"}:
        selected = _select_consistent_strongest(candidate_groups, initial_points, radial_step)
    else:
        selected = initial

    if len(selected) < max(3, min(8, count // 4)):
        raise ValueError(f"卡尺找圆有效边缘点不足：{len(selected)}")

    points = np.asarray([(candidate["x"], candidate["y"]) for candidate in selected], dtype=np.float64)
    gradients_arr = np.asarray([candidate["gradient"] for candidate in selected], dtype=np.float64)
    for candidate in selected:
        windows[int(candidate["window_index"])]["accepted"] = True
        windows[int(candidate["window_index"])]["selected_radius"] = candidate["roi_radius"]

    if params.use_ransac and len(points) >= 6:
        fit_cx, fit_cy, radius, residual, mask = fit_circle_ransac(
            points,
            params.residual_limit_px,
            iterations=400,
        )
    else:
        initial_cx, initial_cy, initial_radius, _ = fit_circle_least_squares(points)
        fit_cx, fit_cy, radius, residual = fit_circle_geometric_robust(
            points,
            (initial_cx, initial_cy, initial_radius),
        )
        mask = np.ones(len(points), dtype=bool)

    inlier_count = int(np.sum(mask))
    inlier_points = points[mask]
    stats = circle_diameter_statistics(inlier_points, fit_cx, fit_cy)
    radius = stats["average_radius_px"] or radius
    angular_coverage, maximum_gap_deg = _angular_statistics(inlier_points, fit_cx, fit_cy, count)
    confidence = float(
        np.clip(
            (inlier_count / max(1, count))
            * angular_coverage
            * np.exp(-max(0.0, residual) / 2.0),
            0.0,
            1.0,
        )
    )
    return CaliperCircleResult(
        center_x_px=float(fit_cx),
        center_y_px=float(fit_cy),
        radius_px=float(radius),
        residual_px=float(residual),
        confidence=confidence,
        edge_points=points[mask],
        rejected_points=points[~mask],
        gradients=gradients_arr[mask],
        rejected_gradients=gradients_arr[~mask],
        inlier_mask=mask,
        caliper_windows=windows,
        average_diameter_px=stats["average_diameter_px"],
        maximum_diameter_px=stats["maximum_diameter_px"],
        minimum_diameter_px=stats["minimum_diameter_px"],
        diameter_pv_px=stats["diameter_pv_px"],
        angular_coverage=angular_coverage,
        maximum_gap_deg=maximum_gap_deg,
    )
