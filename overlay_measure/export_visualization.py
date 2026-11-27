from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np
from PIL import Image

from .image_loader import display_to_uint8
from .models import DetectionResult, ImageData, Roi


ROI_COLOR = (255, 149, 0)
FIT_COLOR = (52, 199, 89)
TEXT_COLOR = (255, 255, 255)


def _points(value) -> np.ndarray:
    points = np.asarray(list(value or []), dtype=np.float64)
    return points.reshape((-1, 2)) if points.size else np.empty((0, 2), dtype=np.float64)


def _rotated_box(roi: Roi) -> np.ndarray:
    normalized = roi.normalized()
    box = cv2.boxPoints((normalized.center(), (normalized.w, normalized.h), normalized.angle_deg))
    return np.rint(box).astype(np.int32)


def _draw_roi(canvas: np.ndarray, roi: Roi, label: str) -> None:
    roi = roi.normalized()
    thickness = max(1, int(round(min(canvas.shape[:2]) / 700.0)))
    if roi.roi_type in {"Circle", "Annulus", "Caliper Circle"}:
        center = tuple(np.rint(roi.center()).astype(int))
        outer = max(1, int(round(roi.outer_radius())))
        cv2.circle(canvas, center, outer, ROI_COLOR, thickness, cv2.LINE_AA)
        if roi.roi_type != "Circle":
            cv2.circle(canvas, center, max(1, int(round(roi.inner_radius()))), ROI_COLOR, thickness, cv2.LINE_AA)
        anchor = (center[0] + 6, center[1] - outer - 8)
    else:
        box = _rotated_box(roi)
        cv2.polylines(canvas, [box], True, ROI_COLOR, thickness, cv2.LINE_AA)
        if roi.roi_type == "Rectangular Ring":
            inner_w, inner_h = roi.inner_size()
            inner = replace(roi, w=inner_w, h=inner_h, x=roi.center()[0] - inner_w / 2, y=roi.center()[1] - inner_h / 2)
            cv2.polylines(canvas, [_rotated_box(inner)], True, ROI_COLOR, thickness, cv2.LINE_AA)
        anchor = tuple(box[np.argmin(box[:, 1])])
    cv2.putText(canvas, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_COLOR, 1, cv2.LINE_AA)


def _draw_detection(canvas: np.ndarray, detection: DetectionResult, label: str) -> None:
    if not (np.isfinite(detection.center_x_px) and np.isfinite(detection.center_y_px)):
        return
    center = (int(round(detection.center_x_px)), int(round(detection.center_y_px)))
    thickness = max(2, int(round(min(canvas.shape[:2]) / 600.0)))
    mode = detection.fitting_mode
    params = detection.shape_params
    if mode in {"Circle", "EdgeCenter", "CaliperCircle", "ProductionCircle", "AutoCircle"}:
        raw_radius = float(params.get("radius_px", detection.diameter_px / 2.0))
        if np.isfinite(raw_radius) and raw_radius > 0:
            radius = max(1, int(round(raw_radius)))
            cv2.circle(canvas, center, radius, FIT_COLOR, thickness, cv2.LINE_AA)
    elif mode == "Ellipse":
        major = max(1, int(round(float(params.get("major_px", detection.diameter_px)) / 2.0)))
        minor = max(1, int(round(float(params.get("minor_px", detection.diameter_px)) / 2.0)))
        cv2.ellipse(canvas, center, (major, minor), float(params.get("angle_deg", 0.0)), 0, 360, FIT_COLOR, thickness, cv2.LINE_AA)
    elif mode in {"Rectangle", "ProductionRectangle"}:
        box = cv2.boxPoints((center, (float(params.get("width_px", detection.diameter_px)), float(params.get("height_px", detection.diameter_px))), float(params.get("angle_deg", 0.0))))
        cv2.polylines(canvas, [np.rint(box).astype(np.int32)], True, FIT_COLOR, thickness, cv2.LINE_AA)
    elif mode == "Line":
        start, end = params.get("line_start_px"), params.get("line_end_px")
        if start is not None and end is not None:
            cv2.line(canvas, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)), FIT_COLOR, thickness, cv2.LINE_AA)
    else:
        contour = _points(params.get("contour_points") or params.get("region_box_points") or detection.edge_points)
        if len(contour) >= 2:
            cv2.polylines(canvas, [np.rint(contour).astype(np.int32)], True, FIT_COLOR, thickness, cv2.LINE_AA)
    arm = max(5, thickness * 3)
    cv2.line(canvas, (center[0] - arm, center[1]), (center[0] + arm, center[1]), FIT_COLOR, thickness, cv2.LINE_AA)
    cv2.line(canvas, (center[0], center[1] - arm), (center[0], center[1] + arm), FIT_COLOR, thickness, cv2.LINE_AA)
    cv2.putText(canvas, label, (center[0] + arm + 3, center[1] - arm), cv2.FONT_HERSHEY_SIMPLEX, 0.48, FIT_COLOR, 1, cv2.LINE_AA)


def render_measurement_image(
    image: ImageData,
    path: str | Path,
    rois: Iterable[tuple[str, Roi]] = (),
    detections: Mapping[str, DetectionResult] | None = None,
) -> bool:
    if image is None:
        return False
    gray = display_to_uint8(image, enhanced=False)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for label, roi in rois:
        _draw_roi(canvas, roi, label)
    for label, detection in (detections or {}).items():
        _draw_detection(canvas, detection, label)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(output)
    return True
