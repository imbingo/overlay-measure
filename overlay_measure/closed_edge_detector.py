"""Closed edge selection for circular/elliptical holes surrounded by halos."""
from __future__ import annotations

import cv2
import numpy as np

from .image_loader import normalize_to_uint8
from .subpixel_edge_detector import SubpixelEdges, refine_contour_edges


def closed_round_edges(gray, roi, params, *, require_nested=False):
    roi = roi.normalized()
    x0, y0, x1, y1 = roi.to_int_bounds(gray.shape)
    crop = np.asarray(gray[y0:y1, x0:x1], dtype=np.float32)
    if crop.size < 25:
        return None
    sigma = max(0.0, float(params.gaussian_sigma_px))
    blurred = cv2.GaussianBlur(crop, (0, 0), sigma) if sigma else crop
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    precomputed = (blurred, gx, gy)
    edges = cv2.Canny(normalize_to_uint8(blurred), params.canny_low,
                     params.canny_high, L2gradient=True)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    candidates = []
    roi_span = max(float(roi.w), float(roi.h), 1e-6)
    for contour in contours:
        if len(contour) < max(12, params.min_edge_points):
            continue
        area = abs(cv2.contourArea(contour))
        if area < max(30, crop.size * 0.003):
            continue
        local = contour.reshape(-1, 2).astype(np.float64)
        # Open arcs trace back on themselves and have negligible enclosed area.
        hull_area = max(cv2.contourArea(cv2.convexHull(contour)), 1)
        if area / hull_area < 0.85:
            continue
        points = local + [x0, y0]
        if not roi.contains_points(points).all():
            continue
        if (local[:, 0].min() <= 1 or local[:, 1].min() <= 1
                or local[:, 0].max() >= crop.shape[1] - 2
                or local[:, 1].max() >= crop.shape[0] - 2):
            continue
        try:
            (cx, cy), (a, b), _ = cv2.fitEllipse(contour)
        except cv2.error:
            continue
        if not np.isfinite([cx, cy, a, b]).all() or min(a, b) <= 0 or max(a, b) / min(a, b) > 4:
            continue
        if abs(area / (np.pi * a * b / 4) - 1) > 0.15:
            continue
        refined, gradients = refine_contour_edges(
            gray,
            points,
            params,
            precomputed=precomputed,
            origin_xy=(float(x0), float(y0)),
        )
        if len(refined) < params.min_edge_points or len(refined) < 0.75 * min(len(points), 1200):
            continue
        if not roi.contains_points(refined).all():
            continue
        center = np.array([cx + x0, cy + y0])
        center_distance = np.linalg.norm(center - roi.center()) / roi_span
        score = float(np.median(gradients)) / (1 + 4 * center_distance)
        candidates.append((score, area, center, refined, gradients))
    if not candidates:
        return None
    if require_nested:
        # Ignore duplicate sides of a one-pixel Canny trace.
        nested = any(
            np.linalg.norm(a[2] - b[2]) < 0.1 * np.sqrt(max(a[1], b[1]))
            and min(a[1], b[1]) / max(a[1], b[1]) < 0.8
            for i, a in enumerate(candidates) for b in candidates[i + 1:]
        )
        if not nested:
            return None
    _, _, _, points, gradients = max(candidates, key=lambda item: item[0])
    return SubpixelEdges(points, gradients, (x0, y0),
                         "主目标轮廓：闭合边缘筛选，使用最强完整边界（排除光晕混边）")
