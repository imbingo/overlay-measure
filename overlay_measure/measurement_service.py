from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from .caliper_circle_detector import detect_caliper_circle
from .circle_ellipse_fitter import FitResult, fit_mark_shape
from .models import DetectionParams, DetectionResult, ImageData, MeasurementConfig, Roi
from .measurement_units import equivalent_size_um_from_shape, radial_diameter_residual_um, scalar_px_to_um
from .region_center_detector import detect_region_center
from .subpixel_edge_detector import detect_subpixel_edges


LAYER_LABELS = {"upper": "上层", "lower": "下层"}


def fitting_mode_for_layer(params: DetectionParams, layer: str) -> str:
    return (
        getattr(params, "upper_fitting_mode", params.fitting_mode)
        if layer == "upper"
        else getattr(params, "lower_fitting_mode", params.fitting_mode)
    )


def _point_list(points: np.ndarray) -> list[tuple[float, float]]:
    if points is None or len(points) == 0:
        return []
    return [(float(x), float(y)) for x, y in np.asarray(points, dtype=np.float64)]


def _algorithm_path_for_detection(detection: DetectionResult, workflow: str = "Manual") -> str:
    if workflow == "Auto":
        candidate = detection.shape_params.get("candidate_mode", "")
        if detection.fitting_mode == "ProductionCircle":
            return "自动识别 → Otsu阈值/闭合轮廓候选 → AutoCircle → 径向卡尺精测 → RANSAC圆拟合 → 中心差计算"
        if detection.fitting_mode == "ProductionRectangle":
            return "自动识别 → Otsu阈值/闭合轮廓候选 → AutoRectangle → 四边卡尺精测 → 旋转矩形拟合 → 中心差计算"
        if candidate:
            return f"自动识别 → Otsu阈值/闭合轮廓候选 → {candidate} → 候选中心 → 中心差计算"
        return "自动识别 → Otsu阈值/闭合轮廓候选 → 候选中心 → 中心差计算"

    roi_type = detection.shape_params.get("roi_type", "ROI")
    if detection.fitting_mode == "CaliperCircle":
        return "手动ROI → 卡尺圆ROI → 径向灰度卡尺找边缘 → RANSAC圆拟合 → 中心差计算"
    if detection.fitting_mode == "RegionCenter":
        return "手动ROI → 区域分割 → 主区域最小外接矩形中心 → 中心差计算"
    if detection.fitting_mode == "EdgeCenter":
        return "手动ROI → 亚像素边缘 → 边缘点云质心/稳健中心 → 中心差计算"
    if detection.fitting_mode == "Circle":
        method = "RANSAC圆拟合" if detection.shape_params.get("use_ransac", True) else "最小二乘圆拟合"
        return f"手动ROI({roi_type}) → 亚像素边缘 → {method} → 中心差计算"
    if detection.fitting_mode == "Ellipse":
        return f"手动ROI({roi_type}) → 亚像素边缘 → OpenCV椭圆拟合 → 中心差计算"
    if detection.fitting_mode == "Rectangle":
        return f"手动ROI({roi_type}) → 亚像素边缘 → 旋转最小外接矩形拟合 → 中心差计算"
    return f"手动ROI({roi_type}) → 亚像素边缘 → {detection.fitting_mode} → 中心差计算"


def attach_algorithm_path(detection: DetectionResult, workflow: str = "Manual") -> DetectionResult:
    detection.shape_params["algorithm_path"] = _algorithm_path_for_detection(detection, workflow)
    return detection


def describe_algorithm_path(detection: Optional[DetectionResult], workflow: str = "Manual") -> str:
    if detection is None:
        return "未生成检测结果"
    return str(detection.shape_params.get("algorithm_path") or _algorithm_path_for_detection(detection, workflow))


def _fit_to_detection(
    mark_id: str,
    layer: str,
    fit: FitResult,
    used_points: np.ndarray,
    config: MeasurementConfig,
    roi: Roi,
    warning: str = "",
    use_ransac: bool = True,
) -> DetectionResult:
    shape_params = {
        **fit.shape_params,
        "roi_type": getattr(roi, "roi_type", "Rectangle"),
        "roi_inner_ratio": float(getattr(roi, "inner_ratio", 0.0)),
        "roi_target_edge": getattr(roi, "target_edge", "All Edges"),
        "roi_angle_deg": float(getattr(roi, "angle_deg", 0.0)),
        "use_ransac": bool(use_ransac),
    }
    if "radius_px" in fit.shape_params or fit.mode in {"Circle", "EdgeCenter"}:
        diameter_um, residual_um = radial_diameter_residual_um(
            used_points,
            fit.center_x_px,
            fit.center_y_px,
            float(fit.shape_params.get("radius_px", fit.diameter_px / 2.0)),
            fit.residual_px,
            config,
        )
    else:
        diameter_um = equivalent_size_um_from_shape(shape_params, fit.diameter_px, config)
        residual_um = scalar_px_to_um(fit.residual_px, config)

    detection = DetectionResult(
        mark_id=mark_id,
        layer=layer,
        center_x_px=fit.center_x_px,
        center_y_px=fit.center_y_px,
        center_x_um=fit.center_x_px * config.pixel_size_x_um,
        center_y_um=fit.center_y_px * config.pixel_size_y_um,
        diameter_px=fit.diameter_px,
        diameter_um=diameter_um,
        residual_px=fit.residual_px,
        residual_um=residual_um,
        edge_point_count=len(used_points),
        confidence=fit.confidence,
        fitting_mode=fit.mode,
        warning=fit.warning or warning,
        edge_points=_point_list(used_points),
        shape_params=shape_params,
    )
    return attach_algorithm_path(detection, "Manual")


def detect_manual_roi(
    mark_id: str,
    layer: str,
    image: ImageData,
    roi: Roi,
    params: DetectionParams,
    config: MeasurementConfig,
) -> DetectionResult:
    if getattr(roi, "roi_type", "") == "Caliper Circle":
        cal = detect_caliper_circle(image.gray, roi, params)
        diameter_um, residual_um = radial_diameter_residual_um(
            cal.edge_points,
            cal.center_x_px,
            cal.center_y_px,
            cal.radius_px,
            cal.residual_px,
            config,
        )
        detection = DetectionResult(
            mark_id=mark_id,
            layer=layer,
            center_x_px=cal.center_x_px,
            center_y_px=cal.center_y_px,
            center_x_um=cal.center_x_px * config.pixel_size_x_um,
            center_y_um=cal.center_y_px * config.pixel_size_y_um,
            diameter_px=2.0 * cal.radius_px,
            diameter_um=diameter_um,
            residual_px=cal.residual_px,
            residual_um=residual_um,
            edge_point_count=len(cal.edge_points),
            confidence=cal.confidence,
            fitting_mode="CaliperCircle",
            warning="",
            edge_points=_point_list(cal.edge_points),
            rejected_points=_point_list(cal.rejected_points),
            edge_gradients=[float(g) for g in cal.gradients],
            rejected_gradients=[float(g) for g in cal.rejected_gradients],
            shape_params={
                "radius_px": cal.radius_px,
                "inlier_count": int(len(cal.edge_points)),
                "rejected_count": int(len(cal.rejected_points)),
                "caliper_count": int(getattr(roi, "caliper_count", 64)),
                "caliper_width_px": float(getattr(roi, "caliper_width_px", 8.0)),
                "search_direction": getattr(roi, "search_direction", "Inner to Outer"),
                "roi_type": getattr(roi, "roi_type", "Caliper Circle"),
                "roi_inner_ratio": float(getattr(roi, "inner_ratio", 0.0)),
                "roi_inner_radius_px": float(roi.inner_radius()),
                "roi_outer_radius_px": float(roi.outer_radius()),
                "roi_target_edge": getattr(roi, "target_edge", "All Edges"),
                "roi_angle_deg": float(getattr(roi, "angle_deg", 0.0)),
                "caliper_windows": cal.caliper_windows,
                "use_ransac": bool(getattr(params, "use_ransac", True)),
            },
        )
        return attach_algorithm_path(detection, "Manual")

    layer_fit_mode = fitting_mode_for_layer(params, layer)
    detect_params = replace(params, fitting_mode=layer_fit_mode)
    if detect_params.fitting_mode == "RegionCenter":
        fit = detect_region_center(image.gray, roi, detect_params)
        used_points = np.asarray(fit.shape_params.get("contour_points", []), dtype=np.float64)
        return _fit_to_detection(mark_id, layer, fit, used_points, config, roi, use_ransac=detect_params.use_ransac)

    edges = detect_subpixel_edges(image.gray, roi, detect_params)
    if len(edges.points_xy) < detect_params.min_edge_points:
        raise ValueError(
            f"{mark_id} {LAYER_LABELS.get(layer, layer)} 有效边缘点不足："
            f"{len(edges.points_xy)} < {detect_params.min_edge_points}. "
            "可以尝试放大 ROI、降低 Canny/最小梯度，或检查焦面和对比度。"
        )
    fit = fit_mark_shape(edges.points_xy, detect_params)
    used_points = edges.points_xy
    if fit.inlier_mask is not None and len(fit.inlier_mask) == len(edges.points_xy):
        used_points = edges.points_xy[fit.inlier_mask]
    return _fit_to_detection(mark_id, layer, fit, used_points, config, roi, edges.warning, detect_params.use_ransac)
