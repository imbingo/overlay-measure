from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .circle_ellipse_fitter import FitResult
from .models import DetectionParams, Roi
from .image_loader import normalize_to_uint8
from .subpixel_edge_detector import SubpixelEdges, refine_contour_edges


@dataclass
class RegionCandidate:
    mask: np.ndarray
    contour: np.ndarray
    polarity: str
    area: float
    score: float
    rectangularity: float
    circularity: float
    solidity: float
    area_ratio: float
    center_distance_norm: float
    touches_border: bool


def _outer_roi_mask(shape: Tuple[int, int], roi: Roi) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Build a local mask for the ROI outer support.

    Region-center mode is a Z-stack-like area detector: it segments the target
    area inside the ROI, then calculates the center from that segmented region.
    For annular / rectangular-ring ROI we deliberately use the outer support,
    otherwise filled rounded-square marks would be partially removed by the ring.
    """
    r = roi.normalized()
    x0, y0, x1, y1 = r.to_int_bounds(shape)
    h = max(1, y1 - y0)
    w = max(1, x1 - x0)
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = r.center()
    local_cx = cx - x0
    local_cy = cy - y0
    typ = getattr(r, "roi_type", "Rectangle")

    if typ in {"Circle", "Annulus", "Caliper Circle"}:
        radius = max(1, int(round(r.outer_radius())))
        cv2.circle(mask, (int(round(local_cx)), int(round(local_cy))), radius, 255, -1)
    elif typ == "Rectangular Ring":
        theta = np.deg2rad(float(getattr(r, "angle_deg", 0.0)))
        ct, st = np.cos(theta), np.sin(theta)
        pts = []
        for lx, ly in ((-r.w / 2, -r.h / 2), (r.w / 2, -r.h / 2), (r.w / 2, r.h / 2), (-r.w / 2, r.h / 2)):
            px = local_cx + ct * lx - st * ly
            py = local_cy + st * lx + ct * ly
            pts.append([int(round(px)), int(round(py))])
        cv2.fillPoly(mask, [np.asarray(pts, dtype=np.int32)], 255)
    else:
        mask[:, :] = 255
    return mask, (x0, y0, x1, y1)


def _remove_small_components(binary: np.ndarray, min_area: float, max_area: float) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for idx in range(1, num):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            out[labels == idx] = 255
    return out


def _clean_binary(binary: np.ndarray, roi_mask: np.ndarray, min_area: float, max_area: float) -> np.ndarray:
    """Clean segmentation without letting isolated speckles attach to the mark."""
    valid = cv2.bitwise_and(binary, roi_mask)
    # Small median blur style opening removes salt-and-pepper dots.
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(valid, cv2.MORPH_OPEN, kernel3, iterations=1)
    cleaned = _remove_small_components(cleaned, min_area, max_area)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel5, iterations=1)
    cleaned = cv2.bitwise_and(cleaned, roi_mask)

    # Fill holes per component. This improves rounded-square holes while not
    # reintroducing remote speckles.
    flood = cleaned.copy()
    h, w = flood.shape
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(cleaned, cv2.bitwise_and(holes, roi_mask))
    filled = _remove_small_components(filled, min_area, max_area)
    return filled


def _candidate_from_binary(
    binary: np.ndarray,
    roi_mask: np.ndarray,
    polarity: str,
    roi_center_local: tuple[float, float],
    expected_shape: str = "Any",
) -> RegionCandidate | None:
    roi_area = max(1.0, float(np.count_nonzero(roi_mask)))
    # Area limits are intentionally conservative: reject dust speckles and the
    # opposite-polarity background that fills almost the whole ROI.
    min_area = max(30.0, 0.003 * roi_area)
    max_area = 0.86 * roi_area
    valid = _clean_binary(binary, roi_mask, min_area, max_area)
    contours, _ = cv2.findContours(valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    rcx, rcy = roi_center_local
    h, w = valid.shape[:2]
    diag = max(1.0, float(np.hypot(w, h)))
    best: RegionCandidate | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        moments = cv2.moments(contour)
        if abs(moments.get("m00", 0.0)) > 1e-9:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            pts_mean = contour.reshape(-1, 2).astype(float)
            cx, cy = float(np.mean(pts_mean[:, 0])), float(np.mean(pts_mean[:, 1]))

        dist = float(np.hypot(cx - rcx, cy - rcy))
        dist_norm = dist / diag
        if dist_norm > 0.45:
            # In production use the ROI is meant to surround one target; far-away
            # blobs are almost always dust/noise/background remnants.
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        touches_border = bx <= 1 or by <= 1 or (bx + bw) >= (w - 2) or (by + bh) >= (h - 2)
        if touches_border and area / roi_area > 0.30:
            # Opposite-polarity background region often touches ROI/crop border.
            continue

        hull = cv2.convexHull(contour)
        hull_area = max(float(cv2.contourArea(hull)), 1e-9)
        solidity = float(np.clip(area / hull_area, 0.0, 1.0))
        rect = cv2.minAreaRect(contour.astype(np.float32))
        (_, _), (rw, rh), _ = rect
        rect_area = max(float(rw) * float(rh), 1e-9)
        rectangularity = float(np.clip(area / rect_area, 0.0, 1.0))
        perimeter = max(float(cv2.arcLength(contour, True)), 1e-9)
        circularity = float(np.clip(4.0 * np.pi * area / (perimeter * perimeter), 0.0, 1.0))
        aspect = max(float(rw), float(rh)) / max(min(float(rw), float(rh)), 1e-9)
        if aspect > 8.0:
            continue

        area_ratio = float(np.clip(area / roi_area, 0.0, 1.0))
        # Prefer a sizeable, compact, centered component. Rounded squares have
        # high rectangularity and solidity; dust has low area and often poor score.
        area_score = min(1.0, area / max(300.0, 0.03 * roi_area))
        center_score = 1.0 / (1.0 + 7.0 * dist_norm)
        border_score = 0.55 if touches_border else 1.0
        if expected_shape == "Circle":
            shape_score = 0.65 * circularity + 0.35 * solidity
        elif expected_shape == "Rectangle":
            shape_score = 0.65 * rectangularity + 0.35 * solidity
        else:
            shape_score = 0.35 * circularity + 0.35 * rectangularity + 0.30 * solidity
        score = area * area_score * center_score * border_score * shape_score

        component_mask = np.zeros_like(valid)
        cv2.drawContours(component_mask, [contour], -1, 255, -1)
        cand = RegionCandidate(
            component_mask,
            contour,
            polarity,
            area,
            score,
            rectangularity,
            circularity,
            solidity,
            area_ratio,
            dist_norm,
            touches_border,
        )
        if best is None or cand.score > best.score:
            best = cand
    return best


def _sample_contour(contour_global: np.ndarray, max_points: int = 1200) -> list[tuple[float, float]]:
    if len(contour_global) == 0:
        return []
    step = max(1, len(contour_global) // max_points)
    return [(float(x), float(y)) for x, y in contour_global[::step]]


def _segment_primary_candidate(
    gray: np.ndarray,
    roi: Roi,
    params: DetectionParams,
    expected_shape: str = "Any",
) -> tuple[RegionCandidate, tuple[int, int, int, int], float]:
    """Segment and select one main target inside a solid search ROI."""
    if gray is None:
        raise ValueError("未导入图像，无法识别目标")
    r = roi.normalized()
    h_img, w_img = gray.shape[:2]
    roi_mask, bounds = _outer_roi_mask((h_img, w_img), r)
    x0, y0, x1, y1 = bounds
    crop = normalize_to_uint8(gray[y0:y1, x0:x1])
    if crop.size == 0:
        raise ValueError("ROI 区域为空，无法识别目标")

    sigma = max(0.0, float(getattr(params, "gaussian_sigma_px", 1.0)))
    blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma, sigmaY=sigma) if sigma > 0 else crop
    if min(blurred.shape[:2]) >= 5:
        blurred = cv2.medianBlur(blurred, 3)

    valid_pixels = blurred[roi_mask > 0]
    if valid_pixels.size < 20:
        raise ValueError("ROI 有效区域太小，无法识别目标")
    contrast = float(np.percentile(valid_pixels, 95) - np.percentile(valid_pixels, 5))
    if contrast < 3.0:
        raise ValueError("ROI 内灰度对比度过低，无法分割目标区域")

    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rcx, rcy = r.center()[0] - x0, r.center()[1] - y0
    requested = getattr(params, "polarity", "Auto")
    candidates: list[RegionCandidate | None] = []
    if requested == "Dark to Bright":
        candidates.append(_candidate_from_binary(dark, roi_mask, "暗目标", (rcx, rcy), expected_shape))
    elif requested == "Bright to Dark":
        candidates.append(_candidate_from_binary(bright, roi_mask, "亮目标", (rcx, rcy), expected_shape))
    else:
        candidates.extend(
            [
                _candidate_from_binary(dark, roi_mask, "暗目标", (rcx, rcy), expected_shape),
                _candidate_from_binary(bright, roi_mask, "亮目标", (rcx, rcy), expected_shape),
            ]
        )
    valid_candidates = [candidate for candidate in candidates if candidate is not None]
    if not valid_candidates:
        raise ValueError(
            "ROI 内未找到可用主目标。请让 ROI 只覆盖一个 Mark、检查亮暗极性，或提高目标与背景的对比度。"
        )
    return max(valid_candidates, key=lambda candidate: candidate.score), bounds, contrast


def detect_primary_contour_edges(
    gray: np.ndarray,
    roi: Roi,
    params: DetectionParams,
    expected_shape: str = "Any",
) -> SubpixelEdges:
    """Select one connected target, then refine only its contour to subpixels."""
    candidate, (x0, y0, _x1, _y1), _contrast = _segment_primary_candidate(
        gray,
        roi,
        params,
        expected_shape,
    )
    contour_global = candidate.contour.reshape(-1, 2).astype(np.float64)
    contour_global[:, 0] += x0
    contour_global[:, 1] += y0
    points, gradients = refine_contour_edges(gray, contour_global, params)
    if len(points) < 3:
        raise ValueError("主目标轮廓的有效亚像素边缘点不足")
    return SubpixelEdges(
        points_xy=points,
        gradients=gradients,
        roi_origin=(x0, y0),
        warning=f"主目标轮廓：{candidate.polarity}，已过滤 ROI 内其他噪声与轮廓",
    )


def detect_region_center(gray: np.ndarray, roi: Roi, params: DetectionParams) -> FitResult:
    """Detect center by ROI area segmentation, not edge fitting.

    Compared with strict rectangle/circle fitting, this mode is more tolerant of
    rounded corners and weak/discontinuous edges. It displays only the final main
    segmented region; isolated noise components are filtered and not reported.
    """
    r = roi.normalized()
    cand, (x0, y0, _x1, _y1), contrast = _segment_primary_candidate(gray, r, params, "Any")

    contour = cand.contour
    moments = cv2.moments(contour)
    if abs(moments.get("m00", 0.0)) > 1e-9:
        centroid_x = float(moments["m10"] / moments["m00"])
        centroid_y = float(moments["m01"] / moments["m00"])
    else:
        pts_local = contour.reshape(-1, 2).astype(float)
        centroid_x = float(np.mean(pts_local[:, 0]))
        centroid_y = float(np.mean(pts_local[:, 1]))

    pts = contour.reshape(-1, 2).astype(np.float32)
    (rect_cx, rect_cy), (rw, rh), angle = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    width, height = float(rw), float(rh)
    report_angle = float(angle)
    if height > width:
        width, height = height, width
        report_angle += 90.0

    center_x_local = float(rect_cx)
    center_y_local = float(rect_cy)
    center_x = center_x_local + x0
    center_y = center_y_local + y0

    area = float(cand.area)
    equivalent_diameter = float(2.0 * np.sqrt(max(area, 0.0) / np.pi))
    contour_pts = pts.astype(np.float64)
    radial = np.hypot(contour_pts[:, 0] - center_x_local, contour_pts[:, 1] - center_y_local)
    residual = float(np.std(radial)) if len(radial) else 0.0

    confidence = float(np.clip(
        0.25 * min(1.0, area / 500.0)
        + 0.25 * cand.rectangularity
        + 0.20 * cand.solidity
        + 0.20 * min(1.0, contrast / 60.0)
        + 0.10 * (1.0 - min(1.0, cand.center_distance_norm / 0.45)),
        0.0,
        1.0,
    ))

    contour_global = contour.reshape(-1, 2).astype(np.float64)
    contour_global[:, 0] += x0
    contour_global[:, 1] += y0
    sampled_contour = _sample_contour(contour_global)

    box_local = cv2.boxPoints(((rect_cx, rect_cy), (rw, rh), angle)).astype(np.float64)
    box_global = box_local.copy()
    box_global[:, 0] += x0
    box_global[:, 1] += y0

    # Do not include radius_px in shape_params. Region-center marks are not
    # circles; drawing an equivalent circle is visually misleading for rounded
    # square holes.
    return FitResult(
        center_x_px=center_x,
        center_y_px=center_y,
        diameter_px=equivalent_diameter,
        residual_px=residual,
        mode="RegionCenter",
        confidence=confidence,
        shape_params={
            "width_px": width,
            "height_px": height,
            "angle_deg": report_angle,
            "region_area_px2": area,
            "rectangularity": cand.rectangularity,
            "circularity": cand.circularity,
            "solidity": cand.solidity,
            "area_ratio": cand.area_ratio,
            "contrast": contrast,
            "region_polarity": cand.polarity,
            "center_method": "min_area_rect_center_from_segmented_main_region",
            "centroid_x_px": centroid_x + x0,
            "centroid_y_px": centroid_y + y0,
            "contour_points": sampled_contour,
            "region_box_points": [(float(x), float(y)) for x, y in box_global],
            "region_noise_filtered": True,
        },
        inlier_mask=np.ones(len(sampled_contour), dtype=bool),
        warning=f"区域中心：{cand.polarity}，已过滤小噪点",
    )
