from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .models import DetectionParams


@dataclass
class FitResult:
    center_x_px: float
    center_y_px: float
    diameter_px: float
    residual_px: float
    mode: str
    confidence: float
    shape_params: Dict[str, float]
    inlier_mask: Optional[np.ndarray] = None
    warning: str = ""


def _circle_from_3pts(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    temp = x2 * x2 + y2 * y2
    bc = (x1 * x1 + y1 * y1 - temp) / 2.0
    cd = (temp - x3 * x3 - y3 * y3) / 2.0
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
    if abs(det) < 1e-9:
        return None
    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    r = np.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)
    if not np.isfinite(r) or r <= 0:
        return None
    return float(cx), float(cy), float(r)


def fit_circle_least_squares(points: np.ndarray) -> Tuple[float, float, float, float]:
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c0 = c
    r2 = c0 + cx * cx + cy * cy
    r = np.sqrt(max(0.0, r2))
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    residual = float(np.sqrt(np.mean((d - r) ** 2))) if len(d) else float("inf")
    return float(cx), float(cy), float(r), residual


def fit_circle_geometric_robust(
    points: np.ndarray,
    initial: Optional[Tuple[float, float, float]] = None,
    max_iterations: int = 30,
) -> Tuple[float, float, float, float]:
    """Refine a circle by robust orthogonal-distance least squares.

    The algebraic solution is useful as an initializer, but it can move toward a
    densely sampled or incomplete arc. This IRLS refinement minimizes actual
    radial distance and limits the influence of local burrs and weak edge picks.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        raise ValueError("圆拟合至少需要 3 个点")
    if initial is None:
        cx, cy, radius, _ = fit_circle_least_squares(pts)
    else:
        cx, cy, radius = (float(value) for value in initial)

    for _ in range(max(1, int(max_iterations))):
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        distances = np.hypot(dx, dy)
        valid = distances > 1e-9
        if int(np.sum(valid)) < 3:
            break
        residuals = distances - radius
        median = float(np.median(residuals[valid]))
        mad = float(np.median(np.abs(residuals[valid] - median)))
        scale = max(0.05, 1.4826 * mad)
        normalized = np.abs(residuals) / (1.5 * scale)
        weights = np.ones_like(normalized)
        outliers = normalized > 1.0
        weights[outliers] = 1.0 / np.maximum(normalized[outliers], 1e-12)

        jacobian = np.column_stack(
            [
                -dx / np.maximum(distances, 1e-12),
                -dy / np.maximum(distances, 1e-12),
                -np.ones(len(pts), dtype=np.float64),
            ]
        )
        sqrt_weights = np.sqrt(weights)
        lhs = jacobian * sqrt_weights[:, None]
        rhs = -residuals * sqrt_weights
        delta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        cx += float(delta[0])
        cy += float(delta[1])
        radius += float(delta[2])
        radius = max(radius, 1e-9)
        if float(np.linalg.norm(delta)) < 1e-7:
            break

    distances = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    residual = float(np.sqrt(np.mean(np.square(distances - radius))))
    return float(cx), float(cy), float(radius), residual


def circle_diameter_statistics(
    points: np.ndarray,
    center_x: float,
    center_y: float,
) -> Dict[str, float]:
    """Return representative and maximum-Feret diameter statistics."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return {
            "average_radius_px": 0.0,
            "average_diameter_px": 0.0,
            "maximum_diameter_px": 0.0,
            "minimum_diameter_px": 0.0,
            "diameter_pv_px": 0.0,
        }
    radii = np.hypot(pts[:, 0] - float(center_x), pts[:, 1] - float(center_y))
    average_diameter = 2.0 * float(np.mean(radii))
    minimum_diameter = 2.0 * float(np.min(radii))

    hull = cv2.convexHull(pts.astype(np.float32).reshape(-1, 1, 2)).reshape(-1, 2).astype(np.float64)
    maximum_diameter = 0.0
    for index in range(len(hull)):
        distances = np.hypot(hull[index + 1 :, 0] - hull[index, 0], hull[index + 1 :, 1] - hull[index, 1])
        if len(distances):
            maximum_diameter = max(maximum_diameter, float(np.max(distances)))
    if len(hull) == 1:
        maximum_diameter = 0.0
    return {
        "average_radius_px": float(np.mean(radii)),
        "average_diameter_px": average_diameter,
        "maximum_diameter_px": maximum_diameter,
        "minimum_diameter_px": minimum_diameter,
        "diameter_pv_px": max(0.0, maximum_diameter - minimum_diameter),
    }


def fit_circle_ransac(points: np.ndarray, residual_limit_px: float, iterations: int = 250) -> Tuple[float, float, float, float, np.ndarray]:
    n = len(points)
    if n < 3:
        raise ValueError("圆拟合至少需要 3 个点")
    rng = np.random.default_rng(12345)
    best_mask = np.ones(n, dtype=bool)
    best_score = -1
    best_residual = float("inf")

    # The UI labels this value as the rejection threshold, so use it directly.
    # Earlier versions silently multiplied it by 2.5, allowing mixed inner and
    # outer edges to survive as one circle.
    thresh = max(0.10, float(residual_limit_px))
    for _ in range(iterations):
        idx = rng.choice(n, 3, replace=False)
        circle = _circle_from_3pts(points[idx[0]], points[idx[1]], points[idx[2]])
        if circle is None:
            continue
        cx, cy, r = circle
        d = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        err = np.abs(d - r)
        mask = err < thresh
        score = int(np.sum(mask))
        residual = float(np.sqrt(np.mean(err[mask] ** 2))) if score > 0 else float("inf")
        if score > best_score or (score == best_score and residual < best_residual):
            best_score = score
            best_mask = mask
            best_residual = residual

    if int(np.sum(best_mask)) < max(3, min(20, int(0.2 * n))):
        best_mask = np.ones(n, dtype=bool)
    initial_cx, initial_cy, initial_r, _ = fit_circle_least_squares(points[best_mask])
    cx, cy, r, residual = fit_circle_geometric_robust(
        points[best_mask],
        (initial_cx, initial_cy, initial_r),
    )
    return cx, cy, r, residual, best_mask


def fit_ellipse(points: np.ndarray) -> FitResult:
    if len(points) < 5:
        raise ValueError("椭圆拟合至少需要 5 个点")
    pts = points.astype(np.float32).reshape(-1, 1, 2)
    (cx, cy), (major, minor), angle = cv2.fitEllipse(pts)
    # Normalize major >= minor
    if minor > major:
        major, minor = minor, major
        angle += 90.0
    a = max(major, 1e-9) / 2.0
    b = max(minor, 1e-9) / 2.0
    theta = np.deg2rad(angle)
    ct, st = np.cos(theta), np.sin(theta)
    x = points[:, 0] - cx
    y = points[:, 1] - cy
    xr = ct * x + st * y
    yr = -st * x + ct * y
    rho = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
    r_mean = 0.5 * (a + b)
    residual = float(np.sqrt(np.mean(((rho - 1.0) * r_mean) ** 2)))
    roundness = float(minor / major) if major > 0 else 0.0
    confidence = _confidence(len(points), residual, roundness)
    return FitResult(
        center_x_px=float(cx),
        center_y_px=float(cy),
        diameter_px=float((major + minor) / 2.0),
        residual_px=residual,
        mode="Ellipse",
        confidence=confidence,
        shape_params={"major_px": float(major), "minor_px": float(minor), "angle_deg": float(angle), "roundness": roundness},
        inlier_mask=np.ones(len(points), dtype=bool),
    )


def _rotated_rect_residual(points: np.ndarray, cx: float, cy: float, width: float, height: float, angle_deg: float) -> float:
    """Approximate distance of each point to the nearest side of a rotated rectangle."""
    if len(points) == 0 or width <= 0 or height <= 0:
        return float("inf")
    theta = np.deg2rad(angle_deg)
    ct, st = np.cos(theta), np.sin(theta)
    x = points[:, 0].astype(np.float64) - cx
    y = points[:, 1].astype(np.float64) - cy
    # Rotate points into rectangle local coordinates.
    xr = ct * x + st * y
    yr = -st * x + ct * y
    dx = np.abs(np.abs(xr) - width / 2.0)
    dy = np.abs(np.abs(yr) - height / 2.0)
    # Points on a rectangle contour should be close to either a vertical or horizontal side.
    dist = np.minimum(dx, dy)
    return float(np.sqrt(np.mean(dist * dist)))


def fit_rectangle(points: np.ndarray) -> FitResult:
    """Fit a square/rectangle mark by a subpixel rotated minimum-area rectangle.

    This is intended for square holes / rectangular holes. The edge points are still
    generated by the same subpixel edge locator; only the shape model changes.
    """
    if len(points) < 4:
        raise ValueError("方孔/矩形拟合至少需要 4 个点")

    pts = points.astype(np.float32).reshape(-1, 1, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    w = float(max(w, 1e-9))
    h = float(max(h, 1e-9))
    angle = float(angle)

    # Normalize width >= height for reporting consistency.
    width, height = w, h
    report_angle = angle
    if height > width:
        width, height = height, width
        report_angle = angle + 90.0

    residual = _rotated_rect_residual(points, float(cx), float(cy), width, height, report_angle)
    aspect_ratio = float(min(width, height) / max(width, height)) if max(width, height) > 0 else 0.0
    confidence = _confidence_rectangle(len(points), residual, aspect_ratio)
    return FitResult(
        center_x_px=float(cx),
        center_y_px=float(cy),
        diameter_px=float((width + height) / 2.0),  # for UI/export compatibility; means equivalent side size
        residual_px=residual,
        mode="Rectangle",
        confidence=confidence,
        shape_params={
            "width_px": float(width),
            "height_px": float(height),
            "angle_deg": float(report_angle),
            "aspect_ratio": aspect_ratio,
            "roundness": aspect_ratio,
        },
        inlier_mask=np.ones(len(points), dtype=bool),
    )


def _confidence(n_points: int, residual_px: float, roundness: float = 1.0) -> float:
    point_score = min(1.0, n_points / 250.0)
    residual_score = float(np.exp(-max(0.0, residual_px) / 0.35))
    round_score = max(0.0, min(1.0, roundness))
    return float(np.clip(0.45 * point_score + 0.40 * residual_score + 0.15 * round_score, 0.0, 1.0))


def _confidence_rectangle(n_points: int, residual_px: float, aspect_ratio: float = 1.0) -> float:
    point_score = min(1.0, n_points / 220.0)
    residual_score = float(np.exp(-max(0.0, residual_px) / 0.45))
    aspect_score = max(0.0, min(1.0, aspect_ratio))
    return float(np.clip(0.45 * point_score + 0.40 * residual_score + 0.15 * aspect_score, 0.0, 1.0))



def fit_region_center(points: np.ndarray) -> FitResult:
    """Calculate a robust region-style center from the selected contour points.

    This mode is intended for rounded or slightly irregular square/rectangular
    marks. It uses the convex hull / contour moments of the selected points,
    so it behaves more like a region-centroid measurement than a strict
    circle/rectangle fit. It does not force the target into an ideal primitive.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts is None or len(pts) < 3:
        raise ValueError("有效边缘点数量不足，无法计算区域中心")

    pts32 = pts.astype(np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts32)
    moments = cv2.moments(hull)
    if abs(moments.get("m00", 0.0)) > 1e-9:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))

    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    bbox_width = float(x_max - x_min)
    bbox_height = float(y_max - y_min)

    (rect_cx, rect_cy), (rw, rh), angle = cv2.minAreaRect(pts32)
    rw = float(max(rw, 1e-9))
    rh = float(max(rh, 1e-9))
    width, height = rw, rh
    report_angle = float(angle)
    if height > width:
        width, height = height, width
        report_angle += 90.0

    radial = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    radius = float(np.mean(radial)) if len(radial) else 0.0
    residual = float(np.std(radial)) if len(radial) else 0.0
    diameter = float((bbox_width + bbox_height) / 2.0) if (bbox_width > 0 and bbox_height > 0) else float(2.0 * radius)
    area = abs(float(cv2.contourArea(hull)))
    rect_area = max(width * height, 1e-9)
    rectangularity = float(np.clip(area / rect_area, 0.0, 1.0))
    confidence = float(np.clip(0.50 * min(1.0, len(pts) / 180.0) + 0.35 * rectangularity + 0.15 * np.exp(-residual / 3.0), 0.0, 1.0))

    return FitResult(
        center_x_px=cx,
        center_y_px=cy,
        diameter_px=diameter,
        residual_px=residual,
        mode="RegionCenter",
        confidence=confidence,
        shape_params={
            "radius_px": radius,
            "width_px": width,
            "height_px": height,
            "bbox_width_px": bbox_width,
            "bbox_height_px": bbox_height,
            "angle_deg": float(report_angle),
            "region_area_px2": area,
            "rectangularity": rectangularity,
            "center_method": "convex_hull_moments",
        },
        inlier_mask=np.ones(len(pts), dtype=bool),
    )

def fit_edge_center(points: np.ndarray) -> FitResult:
    """Use the extracted edge contour itself as the measured geometry.

    The ROI defines where edge points are accepted. The center is the centroid
    of the subpixel edge point cloud, so non-ideal round/square holes are not
    forced into an ideal primitive.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts is None or len(pts) < 3:
        raise ValueError("有效边缘点数量不足，无法计算边缘中心")
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    radius_values = np.sqrt(dx * dx + dy * dy)
    radius = float(np.mean(radius_values))
    residual = float(np.std(radius_values))
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    diameter = float(2.0 * radius)
    confidence = float(np.clip(len(pts) / 160.0, 0.0, 1.0))
    return FitResult(
        center_x_px=cx,
        center_y_px=cy,
        diameter_px=diameter,
        residual_px=residual,
        mode="EdgeCenter",
        confidence=confidence,
        shape_params={
            "radius_px": radius,
            "width_px": width,
            "height_px": height,
            "edge_center_method": "point_centroid",
        },
        inlier_mask=np.ones(len(pts), dtype=bool),
    )


def fit_mark_shape(points: np.ndarray, params: DetectionParams) -> FitResult:
    if points is None or len(points) < 3:
        raise ValueError("有效边缘点数量不足，无法拟合")

    mode = params.fitting_mode
    warnings = []

    if mode == "EdgeCenter":
        return fit_edge_center(points)
    if mode == "RegionCenter":
        return fit_region_center(points)

    circle_result = None
    if mode in {"Auto", "Circle"} and len(points) >= 3:
        try:
            if params.use_ransac:
                cx, cy, r, residual, mask = fit_circle_ransac(points, params.residual_limit_px)
            else:
                initial_cx, initial_cy, initial_r, _ = fit_circle_least_squares(points)
                cx, cy, r, residual = fit_circle_geometric_robust(
                    points,
                    (initial_cx, initial_cy, initial_r),
                )
                mask = np.ones(len(points), dtype=bool)
            statistics = circle_diameter_statistics(points[mask], cx, cy)
            r = statistics["average_radius_px"] or r
            roundness = 1.0
            conf = _confidence(int(np.sum(mask)), residual, roundness)
            circle_result = FitResult(
                center_x_px=cx,
                center_y_px=cy,
                diameter_px=2.0 * r,
                residual_px=residual,
                mode="Circle",
                confidence=conf,
                shape_params={
                    "radius_px": float(r),
                    "roundness": 1.0,
                    "average_diameter_px": statistics["average_diameter_px"],
                    "maximum_diameter_px": statistics["maximum_diameter_px"],
                    "minimum_diameter_px": statistics["minimum_diameter_px"],
                    "diameter_pv_px": statistics["diameter_pv_px"],
                    "diameter_definition": "robust_average_circle",
                },
                inlier_mask=mask,
            )
        except Exception as exc:
            if mode == "Circle":
                raise
            warnings.append(f"圆拟合失败：{exc}")

    ellipse_result = None
    if mode in {"Auto", "Ellipse"} and len(points) >= 5:
        try:
            ellipse_result = fit_ellipse(points)
        except Exception as exc:
            if mode == "Ellipse":
                raise
            warnings.append(f"椭圆拟合失败：{exc}")

    rectangle_result = None
    if mode in {"Auto", "Rectangle"} and len(points) >= 4:
        try:
            rectangle_result = fit_rectangle(points)
        except Exception as exc:
            if mode == "Rectangle":
                raise
            warnings.append(f"方孔/矩形拟合失败：{exc}")

    if mode == "Circle":
        result = circle_result
    elif mode == "Ellipse":
        result = ellipse_result
    elif mode == "Rectangle":
        result = rectangle_result
    else:
        # Auto: choose the model with the best confidence, but penalize circle/ellipse
        # when a rectangle clearly explains the contour better.
        candidates = [r for r in (circle_result, ellipse_result, rectangle_result) if r is not None]
        if not candidates:
            raise ValueError("拟合失败：没有可用结果")

        def score(r: FitResult) -> float:
            # Favor lower residual and higher confidence. Rectangle gets a slight bonus when
            # the edge contour is not well represented by circle/ellipse.
            base = r.confidence
            if (
                r.mode == "Circle"
                and ellipse_result is not None
                and float(ellipse_result.shape_params.get("roundness", 0.0)) >= 0.97
                and r.residual_px <= ellipse_result.residual_px * 1.25 + 0.02
            ):
                # A nearly circular ellipse is the same physical hypothesis with
                # two unnecessary degrees of freedom. Prefer the simpler circle.
                base += 0.04
            if r.mode == "Rectangle":
                base += 0.03
            return base

        result = max(candidates, key=score)

    if result is None:
        raise ValueError("拟合失败：没有可用结果")

    result.warning = "; ".join(warnings)
    return result
