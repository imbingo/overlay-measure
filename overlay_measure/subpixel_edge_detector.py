from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

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


def detect_subpixel_edges(gray: np.ndarray, roi: Roi, params: DetectionParams) -> SubpixelEdges:
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

    half = float(params.profile_half_width_px)
    step = max(0.1, float(params.profile_step_px))
    offsets = np.arange(-half, half + 0.5 * step, step, dtype=np.float32)
    if len(offsets) < 5:
        offsets = np.linspace(-half, half, 9, dtype=np.float32)

    for xi, yi in zip(xs, ys):
        grad = float(gmag[yi, xi])
        if grad < params.min_gradient:
            continue
        nx = float(gx[yi, xi]) / (grad + 1e-12)
        ny = float(gy[yi, xi]) / (grad + 1e-12)
        if not np.isfinite(nx) or not np.isfinite(ny):
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
            continue
        samples = np.asarray(samples, dtype=np.float32)
        deriv = np.gradient(samples, step)

        if params.polarity == "Dark hole":
            # dark hole inner boundary usually bright->dark or dark->bright depending side;
            # use absolute gradient to avoid assuming which side of the contour this point is on.
            score = np.abs(deriv)
        elif params.polarity == "Bright hole":
            score = np.abs(deriv)
        else:
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

    if not points:
        return SubpixelEdges(np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), (x0, y0), "未找到有效边缘点")

    pts = np.asarray(points, dtype=np.float32)
    wg = np.asarray(weights, dtype=np.float32)

    # V1.0.5: real ROI masking for Circle / Annulus / Rectangular Ring.
    # The edge detector first finds subpixel candidate points in the outer
    # bounding box, then this mask keeps only the effective ROI band.
    try:
        mask = roi.contains_points(pts)
        target_mask = roi.edge_target_mask(pts)
        mask = mask & target_mask
        pts = pts[mask]
        wg = wg[mask]
    except Exception:
        # Backward compatibility for older ROI objects.
        pass

    if len(pts) == 0:
        return SubpixelEdges(np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), (x0, y0), "ROI 有效区域内未找到边缘点")

    return SubpixelEdges(pts, wg, (x0, y0), "")
