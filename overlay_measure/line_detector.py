from __future__ import annotations

import cv2
import numpy as np

from .models import DetectionParams, DetectionResult, ImageData, MeasurementConfig, Roi
from .quality_profiles import annotate_detection_quality
from .measurement_units import scalar_px_to_um
from .geometry_engine import cross2d


def _sample_bilinear(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    valid = (x0 >= 0) & (y0 >= 0) & (x0 < width - 1) & (y0 < height - 1)
    values = np.full(xs.shape, np.nan, dtype=np.float64)
    if not np.any(valid):
        return values
    xv, yv = xs[valid], ys[valid]
    xi, yi = x0[valid], y0[valid]
    dx, dy = xv - xi, yv - yi
    values[valid] = (
        image[yi, xi] * (1.0 - dx) * (1.0 - dy)
        + image[yi, xi + 1] * dx * (1.0 - dy)
        + image[yi + 1, xi] * (1.0 - dx) * dy
        + image[yi + 1, xi + 1] * dx * dy
    )
    return values


def _peak_offset(left: float, center: float, right: float) -> float:
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))


def detect_approximate_line(
    mark_id: str,
    layer: str,
    image: ImageData,
    roi: Roi,
    params: DetectionParams,
    config: MeasurementConfig,
    diagnostics: list | None = None,
) -> DetectionResult:
    """Find one subpixel edge in each normal profile of an oriented line ROI.

    ``diagnostics`` (optional) receives one record per normal profile.
    """
    r = roi.normalized()
    if r.w < 8.0 or r.h < 4.0:
        raise ValueError("近似直线 ROI 太小，请沿目标直线拉长并增加搜索带宽")
    gray = np.asarray(image.gray, dtype=np.float32)
    sigma = max(0.0, float(params.gaussian_sigma_px))
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma) if sigma > 0 else gray
    theta = np.deg2rad(float(r.angle_deg))
    along = np.asarray((np.cos(theta), np.sin(theta)), dtype=np.float64)
    normal = np.asarray((-np.sin(theta), np.cos(theta)), dtype=np.float64)
    center = np.asarray(r.center(), dtype=np.float64)
    sample_count = int(np.clip(round(r.w / 4.0), 12, 240))
    along_offsets = np.linspace(-0.46 * r.w, 0.46 * r.w, sample_count)
    step = max(0.05, float(params.profile_step_px))
    normal_offsets = np.arange(-r.h / 2.0, r.h / 2.0 + 0.5 * step, step)
    average_half = max(0.5, min(float(getattr(r, "caliper_width_px", 8.0)) / 2.0, r.w / sample_count))
    average_offsets = np.linspace(-average_half, average_half, 5)
    polarity = str(getattr(params, "polarity", "Auto"))
    points = []
    strengths = []
    for along_offset in along_offsets:
        profile_rows = []
        for average_offset in average_offsets:
            base = center + along * (along_offset + average_offset)
            positions = base[None, :] + normal_offsets[:, None] * normal[None, :]
            profile_rows.append(_sample_bilinear(blurred, positions[:, 0], positions[:, 1]))
        profile = np.nanmean(np.asarray(profile_rows), axis=0)
        record = None
        if diagnostics is not None:
            start = center + along * along_offset + normal * normal_offsets[0]
            record = {
                "kind": "normal", "window_index": len(diagnostics), "start": start,
                "search_vector": normal.copy(), "tangent": along.copy(),
                "tangent_offsets": np.asarray(average_offsets, dtype=np.float64),
                "radial_samples": normal_offsets - normal_offsets[0],
                "profile": np.asarray(profile, dtype=np.float32),
                "gradient": np.gradient(profile, step) if len(profile) >= 2 else np.empty(0),
                "selected": None, "selected_distance_px": None, "status": "outside_image",
                "inlier": False, "ransac_used": False, "fit_error_px": None, "fit_threshold_px": None,
            }
            diagnostics.append(record)
        if not np.isfinite(profile).all() or len(profile) < 5:
            continue
        gradient = np.gradient(profile, step)
        if polarity == "Dark to Bright":
            score = gradient
        elif polarity == "Bright to Dark":
            score = -gradient
        else:
            score = np.abs(gradient)
        index = int(np.argmax(score))
        strength = float(score[index])
        if strength < float(params.min_gradient) or index <= 0 or index >= len(score) - 1:
            if record is not None:
                record["status"] = "below_gradient_threshold" if strength < float(params.min_gradient) else "no_peak"
            continue
        offset = normal_offsets[index] + _peak_offset(score[index - 1], score[index], score[index + 1]) * step
        point = center + along * along_offset + normal * offset
        if record is not None:
            record["selected"] = {"x": float(point[0]), "y": float(point[1]), "gradient": strength}
            record["selected_distance_px"] = float(offset - normal_offsets[0])
            record["status"] = "pending_fit"
        points.append(point)
        strengths.append(strength)
    if len(points) < max(6, min(int(params.min_edge_points), sample_count) // 4):
        raise ValueError(f"近似直线有效卡尺点不足：{len(points)}，请加宽 ROI 或降低最小梯度")
    pts = np.asarray(points, dtype=np.float64)

    def fit(values: np.ndarray):
        vx, vy, x0, y0 = cv2.fitLine(values.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
        direction = np.asarray((float(vx), float(vy)), dtype=np.float64)
        direction /= max(np.linalg.norm(direction), 1e-12)
        origin = np.asarray((float(x0), float(y0)), dtype=np.float64)
        distances = np.abs(cross2d(direction, values - origin))
        return origin, direction, distances

    origin, direction, distances = fit(pts)
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    limit = max(float(params.residual_limit_px), median + 3.5 * max(mad, 0.05))
    inliers = distances <= limit
    if np.count_nonzero(inliers) >= 6:
        origin, direction, distances = fit(pts[inliers])
        used = pts[inliers]
        used_strengths = np.asarray(strengths, dtype=np.float64)[inliers]
    else:
        used = pts
        used_strengths = np.asarray(strengths, dtype=np.float64)
        inliers = np.ones(len(pts), dtype=bool)
    if np.dot(direction, along) < 0:
        direction = -direction
    projections = (used - origin) @ direction
    start = origin + direction * float(np.min(projections))
    end = origin + direction * float(np.max(projections))
    center_point = 0.5 * (start + end)
    residual_px = float(np.sqrt(np.mean(distances ** 2)))
    coverage = float(np.count_nonzero(inliers) / sample_count)
    confidence = float(np.clip(coverage * np.exp(-residual_px / 2.0), 0.0, 1.0))
    rejected = pts[~inliers]
    detection = DetectionResult(
        mark_id=mark_id,
        layer=layer,
        center_x_px=float(center_point[0]),
        center_y_px=float(center_point[1]),
        center_x_um=float(center_point[0] * config.pixel_size_x_um),
        center_y_um=float((float(image.gray.shape[0]) - center_point[1]) * config.pixel_size_y_um),
        diameter_px=float(np.linalg.norm(end - start)),
        diameter_um=float(np.hypot(
            (end[0] - start[0]) * config.pixel_size_x_um,
            (end[1] - start[1]) * config.pixel_size_y_um,
        )),
        residual_px=residual_px,
        residual_um=scalar_px_to_um(residual_px, config),
        edge_point_count=len(used),
        confidence=confidence,
        fitting_mode="Line",
        edge_points=[(float(x), float(y)) for x, y in used],
        rejected_points=[(float(x), float(y)) for x, y in rejected],
        edge_gradients=[float(value) for value in used_strengths],
        shape_params={
            "line_start_px": (float(start[0]), float(start[1])),
            "line_end_px": (float(end[0]), float(end[1])),
            "line_angle_deg": float(np.rad2deg(np.arctan2(-direction[1], direction[0]))),
            "coverage": coverage,
            "roi_type": "Approximate Line",
            "roi_angle_deg": float(r.angle_deg),
            "roi_width_px": float(r.h),
            "algorithm_path": "近似直线ROI → 法向灰度剖面 → 梯度峰值亚像素定位 → Huber直线拟合",
        },
    )
    annotate_detection_quality(detection, config)
    return detection
