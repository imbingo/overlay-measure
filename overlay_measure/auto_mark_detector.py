from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .circle_ellipse_fitter import fit_circle_ransac
from .image_loader import normalize_to_uint8
from .models import DetectionParams, DetectionResult, Roi
from .subpixel_edge_detector import _bilinear_sample, _quadratic_peak_offset, detect_subpixel_edges


def _refine_contour_points(gray: np.ndarray, contour: np.ndarray, params: DetectionParams) -> Tuple[np.ndarray, np.ndarray]:
    """Move contour samples to the local maximum gray-gradient position."""
    image = gray.astype(np.float32)
    sigma = max(0.0, float(params.gaussian_sigma_px))
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma) if sigma > 0 else image
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    points = contour.reshape(-1, 2).astype(np.float64)
    stride = max(1, len(points) // 720)
    points = points[::stride]
    half_width = max(1.0, float(params.profile_half_width_px))
    step = max(0.05, float(params.profile_step_px))
    offsets = np.arange(-half_width, half_width + step * 0.5, step, dtype=np.float32)
    if len(offsets) < 5:
        offsets = np.linspace(-half_width, half_width, 9, dtype=np.float32)

    refined = []
    gradients = []
    height, width = blurred.shape[:2]
    for x, y in points:
        ix = int(np.clip(round(x), 0, width - 1))
        iy = int(np.clip(round(y), 0, height - 1))
        magnitude = float(np.hypot(gx[iy, ix], gy[iy, ix]))
        if magnitude < float(params.min_gradient):
            continue
        nx = float(gx[iy, ix]) / max(magnitude, 1e-12)
        ny = float(gy[iy, ix]) / max(magnitude, 1e-12)
        samples = np.asarray(
            [_bilinear_sample(blurred, float(x + s * nx), float(y + s * ny)) for s in offsets],
            dtype=np.float32,
        )
        if not np.isfinite(samples).all():
            continue
        score = np.abs(np.gradient(samples, step))
        index = int(np.argmax(score))
        if float(score[index]) < float(params.min_gradient):
            continue
        peak_offset = 0.0
        if 0 < index < len(score) - 1:
            peak_offset = _quadratic_peak_offset(
                float(score[index - 1]),
                float(score[index]),
                float(score[index + 1]),
            )
        distance = float(offsets[index] + peak_offset * step)
        refined.append((float(x + distance * nx), float(y + distance * ny)))
        gradients.append(float(score[index]))
    return np.asarray(refined, dtype=np.float64), np.asarray(gradients, dtype=np.float64)


def _contour_geometry(points: np.ndarray) -> dict:
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    area = abs(float(cv2.contourArea(contour)))
    perimeter = max(float(cv2.arcLength(contour, True)), 1e-9)
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-9:
        center_x = float(moments["m10"] / moments["m00"])
        center_y = float(moments["m01"] / moments["m00"])
    else:
        center_x = float(np.mean(points[:, 0]))
        center_y = float(np.mean(points[:, 1]))

    rectangle = cv2.minAreaRect(contour)
    (_, _), (rect_width, rect_height), angle = rectangle
    rect_width = float(max(rect_width, 1e-9))
    rect_height = float(max(rect_height, 1e-9))
    width = max(rect_width, rect_height)
    height = min(rect_width, rect_height)
    rectangularity = area / max(rect_width * rect_height, 1e-9)
    circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
    approximate = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    is_rectangle = rectangularity > 0.82 and circularity < 0.84 and len(approximate) <= 8

    radial = np.hypot(points[:, 0] - center_x, points[:, 1] - center_y)
    if is_rectangle:
        equivalent_radius = width / 2.0
        box = cv2.boxPoints(rectangle).astype(np.float32)
        distances = [
            abs(float(cv2.pointPolygonTest(box, (float(point[0]), float(point[1])), True)))
            for point in points
        ]
        residual = float(np.sqrt(np.mean(np.square(distances))))
        mode = "AutoRectangle"
    else:
        equivalent_radius = float(np.mean(radial))
        residual = float(np.std(radial))
        mode = "AutoCircle"
    return {
        "center_x": center_x,
        "center_y": center_y,
        "radius_px": equivalent_radius,
        "width_px": width,
        "height_px": height,
        "angle_deg": float(angle),
        "residual_px": residual,
        "circularity": circularity,
        "rectangularity": rectangularity,
        "mode": mode,
    }


def _is_duplicate(candidate: dict, existing: List[dict]) -> bool:
    for prior in existing:
        center_distance = float(
            np.hypot(candidate["center_x"] - prior["center_x"], candidate["center_y"] - prior["center_y"])
        )
        radius_distance = abs(candidate["radius_px"] - prior["radius_px"])
        if center_distance < 2.0 and radius_distance < 2.0:
            return True
    return False


def _sample_polygon(corners: np.ndarray, count_per_side: int = 48) -> np.ndarray:
    samples = []
    for index in range(len(corners)):
        start = corners[index]
        end = corners[(index + 1) % len(corners)]
        for ratio in np.linspace(0.0, 1.0, count_per_side, endpoint=False):
            samples.append(start * (1.0 - ratio) + end * ratio)
    return np.asarray(samples, dtype=np.float64)


def _split_overlapping_rectangles(points: np.ndarray) -> List[np.ndarray]:
    """Recover two similarly oriented rectangles from their non-convex union outline."""
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    perimeter = max(float(cv2.arcLength(contour, True)), 1e-9)
    approximate = cv2.approxPolyDP(contour, 0.015 * perimeter, True).reshape(-1, 2).astype(np.float64)
    if len(approximate) != 8 or cv2.isContourConvex(approximate.astype(np.float32).reshape(-1, 1, 2)):
        return []
    cross_products = []
    for index in range(len(approximate)):
        previous = approximate[index - 1]
        current = approximate[index]
        following = approximate[(index + 1) % len(approximate)]
        cross_products.append(float(np.cross(current - previous, following - current)))
    nonzero = [value for value in cross_products if abs(value) > 1e-6]
    if not nonzero:
        return []
    convex_sign = 1.0 if sum(1 for value in nonzero if value > 0) >= len(nonzero) / 2 else -1.0
    concave = [index for index, value in enumerate(cross_products) if value * convex_sign < 0]
    if len(concave) != 2:
        return []

    paths = []
    for start, stop in ((concave[0], concave[1]), (concave[1], concave[0])):
        path = []
        index = (start + 1) % len(approximate)
        while index != stop:
            path.append(approximate[index])
            index = (index + 1) % len(approximate)
        if len(path) != 3:
            return []
        missing = path[0] + path[2] - path[1]
        paths.append(np.asarray([path[0], path[1], path[2], missing], dtype=np.float64))
    dimensions = []
    for corners in paths:
        side_a = float(np.linalg.norm(corners[1] - corners[0]))
        side_b = float(np.linalg.norm(corners[2] - corners[1]))
        dimensions.append(sorted([side_a, side_b]))
    if max(abs(dimensions[0][i] - dimensions[1][i]) for i in (0, 1)) > max(dimensions[0] + dimensions[1]) * 0.18:
        return []
    return [_sample_polygon(corners) for corners in paths]


def _detect_complete_circles(
    gray: np.ndarray,
    layer: str,
    params: DetectionParams,
    pixel_size_x_um: float,
    pixel_size_y_um: float,
) -> List[DetectionResult]:
    """Extract multiple full circular boundaries, including intersecting circles."""
    height, width = gray.shape[:2]
    whole_image_roi = Roi(2.0, 2.0, float(width - 4), float(height - 4), "Rectangle")
    edges = detect_subpixel_edges(gray, whole_image_roi, params)
    remaining = edges.points_xy.astype(np.float64)
    if len(remaining) < 30:
        return []
    mean_pixel_size = 0.5 * (pixel_size_x_um + pixel_size_y_um)
    results = []
    for _ in range(32):
        if len(remaining) < 30:
            break
        center_x, center_y, radius, residual, inlier_mask = fit_circle_ransac(
            remaining,
            0.5,
            iterations=1200,
        )
        inliers = remaining[inlier_mask]
        angles = (np.arctan2(inliers[:, 1] - center_y, inliers[:, 0] - center_x) + 2.0 * np.pi) % (2.0 * np.pi)
        coverage = len(np.unique((angles / (2.0 * np.pi) * 36).astype(int)))
        if len(inliers) < 28 or coverage < 30 or residual > 0.45 or radius < 5.0:
            break
        order = np.argsort(angles)
        contour_points = inliers[order]
        confidence = float(np.clip((coverage / 36.0) * np.exp(-residual), 0.0, 1.0))
        results.append(
            DetectionResult(
                mark_id="",
                layer=layer,
                center_x_px=center_x,
                center_y_px=center_y,
                center_x_um=center_x * pixel_size_x_um,
                center_y_um=center_y * pixel_size_y_um,
                diameter_px=2.0 * radius,
                diameter_um=2.0 * radius * mean_pixel_size,
                residual_px=residual,
                residual_um=residual * mean_pixel_size,
                edge_point_count=len(inliers),
                confidence=confidence,
                fitting_mode="AutoCircle",
                edge_points=[(float(px), float(py)) for px, py in contour_points],
                shape_params={
                    "radius_px": float(radius),
                    "width_px": float(2.0 * radius),
                    "height_px": float(2.0 * radius),
                    "roi_type": "Auto Full Image",
                    "contour_points": [(float(px), float(py)) for px, py in contour_points],
                    "model_source": "multi_circle_ransac",
                },
            )
        )
        remaining = remaining[~inlier_mask]
    return results


def detect_auto_marks(
    gray: np.ndarray,
    layer: str,
    params: DetectionParams,
    pixel_size_x_um: float,
    pixel_size_y_um: float,
) -> List[DetectionResult]:
    """Find closed dark or bright mark contours and refine their edge coordinates."""
    image_u8 = normalize_to_uint8(gray)
    blurred = cv2.GaussianBlur(image_u8, (5, 5), 1.0)
    image_area = float(image_u8.shape[0] * image_u8.shape[1])
    minimum_area = max(24.0, image_area * 0.00008)
    masks = []
    for threshold_type in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        _, mask = cv2.threshold(blurred, 0, 255, threshold_type + cv2.THRESH_OTSU)
        masks.append(mask)

    results = _detect_complete_circles(gray, layer, params, pixel_size_x_um, pixel_size_y_um)
    geometries: List[dict] = [
        {
            "center_x": result.center_x_px,
            "center_y": result.center_y_px,
            "radius_px": result.shape_params.get("radius_px", result.diameter_px / 2.0),
        }
        for result in results
    ]
    has_complete_circles = bool(results)
    mean_pixel_size = 0.5 * (pixel_size_x_um + pixel_size_y_um)
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = abs(float(cv2.contourArea(contour)))
            perimeter = float(cv2.arcLength(contour, True))
            x, y, width, height = cv2.boundingRect(contour)
            if area < minimum_area or area > image_area * 0.75 or perimeter < 20:
                continue
            if x <= 1 or y <= 1 or x + width >= image_u8.shape[1] - 1 or y + height >= image_u8.shape[0] - 1:
                continue
            refined, gradients = _refine_contour_points(gray, contour, params)
            if len(refined) < 12:
                continue
            base_geometry = _contour_geometry(refined)
            point_sets = _split_overlapping_rectangles(refined) if base_geometry["mode"] == "AutoRectangle" else []
            if not point_sets:
                point_sets = [refined]
            for candidate_points in point_sets:
                geometry = _contour_geometry(candidate_points)
                if has_complete_circles and geometry["mode"] == "AutoCircle":
                    continue
                if geometry["mode"] == "AutoCircle" and geometry["circularity"] < 0.68:
                    continue
                if _is_duplicate(geometry, geometries):
                    continue
                geometries.append(geometry)
                confidence = float(
                    np.clip(
                        0.45
                        + min(0.25, len(candidate_points) / 800.0)
                        + (0.20 * max(geometry["circularity"], geometry["rectangularity"])),
                        0.0,
                        1.0,
                    )
                )
                point_gradients = gradients if candidate_points is refined else np.zeros(len(candidate_points))
                results.append(
                    DetectionResult(
                        mark_id="",
                        layer=layer,
                        center_x_px=geometry["center_x"],
                        center_y_px=geometry["center_y"],
                        center_x_um=geometry["center_x"] * pixel_size_x_um,
                        center_y_um=geometry["center_y"] * pixel_size_y_um,
                        diameter_px=2.0 * geometry["radius_px"],
                        diameter_um=2.0 * geometry["radius_px"] * mean_pixel_size,
                        residual_px=geometry["residual_px"],
                        residual_um=geometry["residual_px"] * mean_pixel_size,
                        edge_point_count=len(candidate_points),
                        confidence=confidence,
                        fitting_mode=geometry["mode"],
                        warning="由重叠外轮廓恢复" if candidate_points is not refined else "",
                        edge_points=[(float(px), float(py)) for px, py in candidate_points],
                        edge_gradients=[float(value) for value in point_gradients],
                        shape_params={
                            "radius_px": geometry["radius_px"],
                            "width_px": geometry["width_px"],
                            "height_px": geometry["height_px"],
                            "angle_deg": geometry["angle_deg"],
                            "circularity": geometry["circularity"],
                            "rectangularity": geometry["rectangularity"],
                            "roi_type": "Auto Full Image",
                            "contour_points": [(float(px), float(py)) for px, py in candidate_points],
                            "inferred_overlap_split": candidate_points is not refined,
                        },
                    )
                )
    return sorted(results, key=lambda result: result.shape_params.get("radius_px", 0.0), reverse=True)
