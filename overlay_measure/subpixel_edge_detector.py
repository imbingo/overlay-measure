from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .models import DetectionParams, Roi
from .image_loader import normalize_to_uint8


@dataclass
class SubpixelEdges:
    points_xy: np.ndarray  # shape [N, 2] in full image pixel coordinates
    gradients: np.ndarray  # shape [N]
    roi_origin: Tuple[int, int]
    warning: str = ""


def _bilinear_sample(img: np.ndarray, x: float, y: float) -> float:
    h, w = img.shape[:2]
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return float("nan")
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    dx = x - x0
    dy = y - y0
    v00 = img[y0, x0]
    v10 = img[y0, x0 + 1]
    v01 = img[y0 + 1, x0]
    v11 = img[y0 + 1, x0 + 1]
    return float((1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 + (1 - dx) * dy * v01 + dx * dy * v11)


def _quadratic_peak_offset(y_minus: float, y0: float, y_plus: float) -> float:
    """Return sub-step peak offset in [-1, 1] for three samples."""
    denom = y_minus - 2.0 * y0 + y_plus
    if abs(denom) < 1e-9:
        return 0.0
    offset = 0.5 * (y_minus - y_plus) / denom
    return float(np.clip(offset, -1.0, 1.0))


def normal_profile_record(index, x, y, nx, ny, offsets, step, samples, status):
    """Diagnostic record for one 1D normal profile (parameter comparison only)."""
    offsets = np.asarray(offsets, dtype=np.float64)
    normal = np.asarray((nx, ny), dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float32)
    return {
        "kind": "normal", "window_index": int(index),
        "start": np.asarray((x, y), dtype=np.float64) + normal * offsets[0],
        "search_vector": normal, "tangent": np.asarray((-ny, nx), dtype=np.float64),
        "tangent_offsets": np.asarray([0.0]), "radial_samples": offsets - offsets[0],
        "profile": samples,
        "gradient": np.gradient(samples, step) if len(samples) >= 2 else np.empty(0),
        "selected": None, "selected_distance_px": None, "status": status,
        "inlier": False, "ransac_used": False, "fit_error_px": None, "fit_threshold_px": None,
        "source_xy": (float(x), float(y)),
    }


def _diag_normal(gx: float, gy: float, magnitude: float) -> tuple[float, float]:
    if magnitude <= 0 or not np.isfinite(magnitude):
        return 1.0, 0.0
    return gx / max(magnitude, 1e-12), gy / max(magnitude, 1e-12)


def refine_contour_edges(
    gray: np.ndarray,
    contour_points_xy: np.ndarray,
    params: DetectionParams,
    max_points: int = 1200,
    precomputed: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    origin_xy: Tuple[float, float] = (0.0, 0.0),
    diagnostics: Optional[list] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine one selected closed contour to subpixel gradient-peak positions.

    ``diagnostics`` (optional) receives one record per sampled contour point;
    it never changes the returned points.
    """
    points = np.asarray(contour_points_xy, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
    stride = max(1, len(points) // max(3, int(max_points)))
    points = points[::stride]
    if precomputed is None:
        image = np.asarray(gray, dtype=np.float32)
        sigma = max(0.0, float(params.gaussian_sigma_px))
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma) if sigma > 0 else image
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    else:
        blurred, gx, gy = precomputed
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    height, width = blurred.shape[:2]
    half_width = max(1.0, float(params.profile_half_width_px))
    step = max(0.05, float(params.profile_step_px))
    offsets = np.arange(-half_width, half_width + step * 0.5, step, dtype=np.float32)
    if len(offsets) < 5:
        offsets = np.linspace(-half_width, half_width, 9, dtype=np.float32)
        step = float(offsets[1] - offsets[0])

    refined = []
    gradients = []
    for x, y in points:
        lx = float(x) - ox
        ly = float(y) - oy
        ix = int(np.clip(round(lx), 0, width - 1))
        iy = int(np.clip(round(ly), 0, height - 1))
        magnitude = float(np.hypot(gx[iy, ix], gy[iy, ix]))
        record = None
        if diagnostics is not None:
            dnx, dny = _diag_normal(float(gx[iy, ix]), float(gy[iy, ix]), magnitude)
            dsamples = [_bilinear_sample(blurred, float(lx + o * dnx), float(ly + o * dny)) for o in offsets]
            record = normal_profile_record(len(diagnostics), x, y, dnx, dny, offsets, step, dsamples, "pending")
            diagnostics.append(record)
        if magnitude < float(params.min_gradient):
            if record is not None:
                record["status"] = "below_gradient_threshold"
            continue
        nx = float(gx[iy, ix]) / max(magnitude, 1e-12)
        ny = float(gy[iy, ix]) / max(magnitude, 1e-12)
        samples = np.asarray(
            [_bilinear_sample(blurred, float(lx + offset * nx), float(ly + offset * ny)) for offset in offsets],
            dtype=np.float32,
        )
        if not np.isfinite(samples).all():
            if record is not None:
                record["status"] = "outside_image"
            continue
        score = np.abs(np.gradient(samples, step))
        index = int(np.argmax(score))
        if float(score[index]) < float(params.min_gradient):
            if record is not None:
                record["status"] = "below_gradient_threshold"
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
        if record is not None:
            record["selected"] = {"x": refined[-1][0], "y": refined[-1][1], "gradient": float(score[index])}
            record["selected_distance_px"] = distance - float(offsets[0])
            record["status"] = "pending_fit"
    return np.asarray(refined, dtype=np.float64), np.asarray(gradients, dtype=np.float64)


def detect_subpixel_edges(
    gray: np.ndarray,
    roi: Roi,
    params: DetectionParams,
    cancelled: Optional[Callable[[], bool]] = None,
    diagnostics: Optional[list] = None,
) -> SubpixelEdges:
    """Detect subpixel edge points inside ROI.

    Strategy:
    - normalize ROI to uint8 for Canny
    - compute Sobel gradients on lightly blurred float ROI
    - for every coarse Canny edge point, sample intensity along gradient normal
    - choose max absolute gradient in 1D profile and refine with quadratic interpolation
    """
    x0, y0, x1, y1 = roi.to_int_bounds(gray.shape)
    roi_img = gray[y0:y1, x0:x1].astype(np.float32)
    if roi_img.size < 25:
        return SubpixelEdges(np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), (x0, y0), "ROI 太小")

    sigma = max(0.0, float(params.gaussian_sigma_px))
    if sigma > 0:
        blurred = cv2.GaussianBlur(roi_img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    else:
        blurred = roi_img.copy()

    roi_u8 = normalize_to_uint8(blurred)
    low = max(0, min(255, int(params.canny_low)))
    high = max(low + 1, min(255, int(params.canny_high)))
    coarse = cv2.Canny(roi_u8, low, high, L2gradient=True)

    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)

    ys, xs = np.nonzero(coarse > 0)
    points: List[Tuple[float, float]] = []
    weights: List[float] = []
    point_records: list = []

    half = float(params.profile_half_width_px)
    step = max(0.1, float(params.profile_step_px))
    offsets = np.arange(-half, half + 0.5 * step, step, dtype=np.float32)
    if len(offsets) < 5:
        offsets = np.linspace(-half, half, 9, dtype=np.float32)
        step = float(offsets[1] - offsets[0])

    for point_index, (xi, yi) in enumerate(zip(xs, ys)):
        if point_index % 128 == 0 and cancelled and cancelled():
            raise InterruptedError("用户取消计算")
        grad = float(gmag[yi, xi])
        record = None
        if diagnostics is not None:
            dnx, dny = _diag_normal(float(gx[yi, xi]), float(gy[yi, xi]), grad)
            dsamples = [_bilinear_sample(blurred, float(xi) + float(o) * dnx, float(yi) + float(o) * dny) for o in offsets]
            record = normal_profile_record(len(diagnostics), float(xi) + x0, float(yi) + y0,
                                           dnx, dny, offsets, step, dsamples, "pending")
            diagnostics.append(record)
        if grad < params.min_gradient:
            if record is not None:
                record["status"] = "below_gradient_threshold"
            continue
        nx = float(gx[yi, xi]) / (grad + 1e-12)
        ny = float(gy[yi, xi]) / (grad + 1e-12)
        if not np.isfinite(nx) or not np.isfinite(ny):
            if record is not None:
                record["status"] = "no_peak"
            continue

        samples = []
        valid = True
        for s in offsets:
            v = _bilinear_sample(blurred, float(xi) + float(s) * nx, float(yi) + float(s) * ny)
            if not np.isfinite(v):
                valid = False
                break
            samples.append(v)
        if not valid:
            if record is not None:
                record["status"] = "outside_image"
            continue
        samples = np.asarray(samples, dtype=np.float32)
        deriv = np.gradient(samples, step)

        # Generic closed-contour extraction has no globally consistent normal
        # direction, so polarity is intentionally not applied here. Directional
        # polarity remains available in caliper-circle and line detectors.
        score = np.abs(deriv)

        k = int(np.argmax(score))
        if k <= 0 or k >= len(score) - 1:
            refined_s = float(offsets[k])
        else:
            sub = _quadratic_peak_offset(float(score[k - 1]), float(score[k]), float(score[k + 1]))
            refined_s = float(offsets[k] + sub * step)

        px = float(xi) + refined_s * nx + x0
        py = float(yi) + refined_s * ny + y0
        points.append((px, py))
        weights.append(grad)
        if record is not None:
            record["selected"] = {"x": px, "y": py, "gradient": float(score[k])}
            record["selected_distance_px"] = refined_s - float(offsets[0])
            record["status"] = "pending_fit"
            point_records.append(record)

    if not points:
        return SubpixelEdges(np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), (x0, y0), "未找到有效边缘点")

    pts = np.asarray(points, dtype=np.float32)
    wg = np.asarray(weights, dtype=np.float32)

    # V1.0.5: real ROI masking for Circle / Annulus / Rectangular Ring.
    # The edge detector first finds subpixel candidate points in the outer
    # bounding box, then this mask keeps only the effective ROI band.
    try:
        inside_mask = roi.contains_points(pts)
        target_mask = roi.edge_target_mask(pts)
        mask = inside_mask & target_mask
        pts = pts[mask]
        wg = wg[mask]
        if point_records:
            for record, inside, target in zip(point_records, inside_mask, target_mask):
                if not inside:
                    record["status"] = "outside_roi_mask"
                elif not target:
                    record["status"] = "target_edge_filtered"
            point_records = [record for record, keep in zip(point_records, mask) if keep]
    except Exception:
        # Backward compatibility for older ROI objects.
        pass

    if len(pts) == 0:
        return SubpixelEdges(np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), (x0, y0), "ROI 有效区域内未找到边缘点")

    max_points = max(0, int(getattr(params, "max_edge_points", 2000)))
    if max_points and len(pts) > max_points:
        # Preserve the complete perimeter distribution for fitting.  Taking the
        # first N scan-line points biases circles and ellipses toward the top.
        indices = np.linspace(0, len(pts) - 1, max_points, dtype=np.intp)
        pts = pts[indices]
        wg = wg[indices]
        if point_records and len(point_records) == len(mask[mask]):
            kept = set(int(i) for i in indices)
            for i, record in enumerate(point_records):
                if i not in kept:
                    record["status"] = "subsampled"
    return SubpixelEdges(pts, wg, (x0, y0), "")
