from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .circle_ellipse_fitter import fit_circle_least_squares, fit_circle_ransac
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


def _profile_edge(profile: np.ndarray, step: float, polarity: str):
    """Return the strongest radial gradient location and magnitude in one caliper."""
    grad = np.gradient(profile, step)
    if polarity == "Dark to Bright":
        score = grad
    elif polarity == "Bright to Dark":
        score = -grad
    else:
        score = np.abs(grad)
    index = int(np.argmax(score))
    if float(score[index]) <= 0:
        return None
    offset = 0.0
    if 0 < index < len(score) - 1:
        offset = _quadratic_peak_offset(
            float(score[index - 1]),
            float(score[index]),
            float(score[index + 1]),
        )
    return float(index + offset), float(score[index])


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

    edge_points: List[Tuple[float, float]] = []
    gradients: List[float] = []
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

        candidate = _profile_edge(np.asarray(profile, dtype=np.float32), radial_step, polarity) if len(profile) >= 5 else None
        accepted = False
        gradient = 0.0
        if candidate is not None:
            sub_index, gradient = candidate
            if gradient >= float(params.min_gradient):
                point = start + search_vector * (sub_index * radial_step)
                edge_points.append((float(point[0]), float(point[1])))
                gradients.append(float(gradient))
                accepted = True

        windows.append(
            {
                "angle": float(angle),
                "center_x": float(cx + radial[0] * (inner + outer) * 0.5),
                "center_y": float(cy + radial[1] * (inner + outer) * 0.5),
                "length": float(length),
                "width": float(width),
                "gradient": float(gradient),
                "accepted": accepted,
            }
        )

    if len(edge_points) < max(3, min(8, count // 4)):
        raise ValueError(f"卡尺找圆有效边缘点不足：{len(edge_points)}")

    points = np.asarray(edge_points, dtype=np.float64)
    if params.use_ransac and len(points) >= 6:
        fit_cx, fit_cy, radius, residual, mask = fit_circle_ransac(
            points,
            params.residual_limit_px,
            iterations=400,
        )
    else:
        fit_cx, fit_cy, radius, residual = fit_circle_least_squares(points)
        mask = np.ones(len(points), dtype=bool)

    inlier_count = int(np.sum(mask))
    confidence = float(
        np.clip((inlier_count / max(1, count)) * np.exp(-max(0.0, residual) / 2.0), 0.0, 1.0)
    )
    gradients_arr = np.asarray(gradients, dtype=np.float64)
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
    )
