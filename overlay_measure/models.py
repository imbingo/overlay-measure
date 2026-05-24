from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


LayerName = str  # "upper" or "lower"


@dataclass
class ImageData:
    path: str
    gray: object
    display_name: str


@dataclass
class Roi:
    """ROI definition in full image pixel coordinates.

    x/y/w/h represent the outer bounding box. Annular and caliper-circle
    modes use inner_ratio for the inner boundary and add caliper settings.
    """

    x: float
    y: float
    w: float
    h: float
    roi_type: str = "Rectangle"
    inner_ratio: float = 0.60
    target_edge: str = "All Edges"
    angle_deg: float = 0.0
    caliper_count: int = 64
    caliper_width_px: float = 8.0
    search_direction: str = "Inner to Outer"

    def normalized(self) -> "Roi":
        x0 = min(self.x, self.x + self.w)
        y0 = min(self.y, self.y + self.h)
        return Roi(
            x0,
            y0,
            abs(self.w),
            abs(self.h),
            self.roi_type,
            float(np.clip(self.inner_ratio, 0.0, 0.98)),
            self.target_edge,
            self.angle_deg,
            int(np.clip(getattr(self, "caliper_count", 64), 4, 720)),
            float(max(1.0, getattr(self, "caliper_width_px", 8.0))),
            getattr(self, "search_direction", "Inner to Outer"),
        )

    def center(self) -> Tuple[float, float]:
        roi = self.normalized()
        return roi.x + roi.w / 2.0, roi.y + roi.h / 2.0

    def outer_radius(self) -> float:
        roi = self.normalized()
        return max(0.5, min(roi.w, roi.h) / 2.0)

    def inner_radius(self) -> float:
        return max(0.0, self.outer_radius() * float(np.clip(self.inner_ratio, 0.0, 0.98)))

    def inner_size(self) -> Tuple[float, float]:
        roi = self.normalized()
        ratio = float(np.clip(roi.inner_ratio, 0.0, 0.98))
        return roi.w * ratio, roi.h * ratio

    def to_int_bounds(self, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        roi = self.normalized()
        height, width = shape[:2]
        x0 = max(0, min(width - 1, int(np.floor(roi.x))))
        y0 = max(0, min(height - 1, int(np.floor(roi.y))))
        x1 = max(0, min(width, int(np.ceil(roi.x + roi.w))))
        y1 = max(0, min(height, int(np.ceil(roi.y + roi.h))))
        if x1 <= x0:
            x1 = min(width, x0 + 1)
        if y1 <= y0:
            y1 = min(height, y0 + 1)
        return x0, y0, x1, y1

    def _local_rotated(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        roi = self.normalized()
        cx, cy = roi.center()
        theta = np.deg2rad(roi.angle_deg)
        cosine, sine = np.cos(theta), np.sin(theta)
        x = xs.astype(np.float64) - cx
        y = ys.astype(np.float64) - cy
        return cosine * x + sine * y, -sine * x + cosine * y

    def contains_points(self, points_xy: np.ndarray) -> np.ndarray:
        if points_xy is None or len(points_xy) == 0:
            return np.zeros((0,), dtype=bool)
        roi = self.normalized()
        xs = points_xy[:, 0].astype(np.float64)
        ys = points_xy[:, 1].astype(np.float64)
        roi_type = roi.roi_type

        if roi_type == "Rectangle":
            return (xs >= roi.x) & (xs <= roi.x + roi.w) & (ys >= roi.y) & (ys <= roi.y + roi.h)

        if roi_type in {"Circle", "Annulus", "Caliper Circle"}:
            cx, cy = roi.center()
            radius = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
            outer = roi.outer_radius()
            if roi_type == "Circle":
                return radius <= outer
            return (radius >= roi.inner_radius()) & (radius <= outer)

        if roi_type == "Rectangular Ring":
            xr, yr = roi._local_rotated(xs, ys)
            inner_w, inner_h = roi.inner_size()
            inside_outer = (np.abs(xr) <= roi.w / 2.0) & (np.abs(yr) <= roi.h / 2.0)
            inside_inner = (np.abs(xr) <= inner_w / 2.0) & (np.abs(yr) <= inner_h / 2.0)
            return inside_outer & (~inside_inner)

        return (xs >= roi.x) & (xs <= roi.x + roi.w) & (ys >= roi.y) & (ys <= roi.y + roi.h)

    def edge_target_mask(self, points_xy: np.ndarray) -> np.ndarray:
        if points_xy is None or len(points_xy) == 0:
            return np.zeros((0,), dtype=bool)
        roi = self.normalized()
        target = roi.target_edge
        if target in {"All Edges", "Strongest Edge", ""}:
            return np.ones(len(points_xy), dtype=bool)

        xs = points_xy[:, 0].astype(np.float64)
        ys = points_xy[:, 1].astype(np.float64)
        if roi.roi_type in {"Annulus", "Caliper Circle"}:
            cx, cy = roi.center()
            radius = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
            inner = roi.inner_radius()
            outer = roi.outer_radius()
            band = 0.40 * max(1e-9, outer - inner)
            if target == "Near Inner Boundary":
                return radius <= inner + band
            if target == "Near Outer Boundary":
                return radius >= outer - band

        if roi.roi_type == "Rectangular Ring":
            xr, yr = roi._local_rotated(xs, ys)
            inner_w, inner_h = roi.inner_size()
            distance_outer = np.minimum(roi.w / 2.0 - np.abs(xr), roi.h / 2.0 - np.abs(yr))
            distance_inner = np.maximum(np.abs(xr) - inner_w / 2.0, np.abs(yr) - inner_h / 2.0)
            distance_outer = np.maximum(distance_outer, 0.0)
            distance_inner = np.maximum(distance_inner, 0.0)
            if target == "Near Inner Boundary":
                return distance_inner <= distance_outer
            if target == "Near Outer Boundary":
                return distance_outer <= distance_inner

        return np.ones(len(points_xy), dtype=bool)


@dataclass
class DetectionParams:
    gaussian_sigma_px: float = 1.0
    canny_low: float = 40.0
    canny_high: float = 120.0
    min_gradient: float = 5.0
    profile_half_width_px: float = 2.0
    profile_step_px: float = 0.25
    fitting_mode: str = "EdgeCenter"
    upper_fitting_mode: str = "EdgeCenter"
    lower_fitting_mode: str = "EdgeCenter"
    use_ransac: bool = True
    residual_limit_px: float = 2.0
    min_edge_points: int = 60
    diameter_min_um: float = 0.0
    diameter_max_um: float = 999999.0
    polarity: str = "Auto"


@dataclass
class MeasurementConfig:
    mode: str = "Single Image"
    workflow_mode: str = "Manual"
    recipe_name: str = ""
    recipe_version: str = "1.0"
    recipe_validation_status: str = "Draft"
    auto_reference_label: str = ""
    auto_target_label: str = ""
    material_code: str = ""
    process_name: str = ""
    equipment_model: str = ""
    calibration_date: str = ""
    operator_name: str = ""
    pixel_size_x_um: float = 0.1
    pixel_size_y_um: float = 0.1
    registration_offset_x_um: float = 0.0
    registration_offset_y_um: float = 0.0
    overlay_definition: str = "upper_minus_lower"
    delta_x_limit_um: float = 0.5
    delta_y_limit_um: float = 0.5
    overlay_r_limit_um: float = 0.7
    confidence_min: float = 0.7
    rz_layout: str = "Y向前后分布"
    rz_distance_l_um: float = 1.0
    rz_limit: float = 999999.0
    production_caliper_count: int = 64
    production_caliper_width_px: float = 8.0
    production_search_half_width_px: float = 8.0
    production_min_coverage: float = 0.65
    production_max_rejected_ratio: float = 0.40
    production_max_residual_um: float = 0.30
    production_max_radial_deviation_um: float = 0.60


@dataclass
class MarkRecipe:
    mark_id: str
    upper_roi: Optional[Roi] = None
    lower_roi: Optional[Roi] = None
    reference_shape: str = "Any"
    target_shape: str = "Any"
    reference_size_min_um: float = 0.0
    reference_size_max_um: float = 999999.0
    target_size_min_um: float = 0.0
    target_size_max_um: float = 999999.0


@dataclass
class DetectionResult:
    mark_id: str
    layer: LayerName
    center_x_px: float
    center_y_px: float
    center_x_um: float
    center_y_um: float
    diameter_px: float
    diameter_um: float
    residual_px: float
    residual_um: float
    edge_point_count: int
    confidence: float
    fitting_mode: str
    warning: str = ""
    edge_points: List[Tuple[float, float]] = field(default_factory=list)
    rejected_points: List[Tuple[float, float]] = field(default_factory=list)
    edge_gradients: List[float] = field(default_factory=list)
    rejected_gradients: List[float] = field(default_factory=list)
    shape_params: Dict[str, object] = field(default_factory=dict)


@dataclass
class OverlayResult:
    mark_id: str
    delta_x_px: float
    delta_y_px: float
    delta_x_um: float
    delta_y_um: float
    overlay_r_um: float
    result: str
    warning: str = ""


def dataclass_to_dict(obj):
    return asdict(obj)
