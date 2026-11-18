from __future__ import annotations

from math import acos, atan2, degrees
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .geometry_models import (
    CoordinateLabelResult,
    CoordinateSystemDefinition,
    CoordinateSystemResult,
    GeometryFeatureDefinition,
    GeometryFeatureResult,
    GeometryMeasurementDefinition,
    GeometryMeasurementResult,
    GeometryProgram,
    GeometryRunResult,
)
from .models import DetectionResult, MeasurementConfig
from .measurement_units import mean_pixel_size_um, minimum_enclosing_circle_um, robust_circle_um


Point = Tuple[float, float]


def _point(value: Iterable[float]) -> Point:
    x, y = value
    return float(x), float(y)


def _circle_from_three_points(points: list[Point]) -> tuple[Point, float]:
    if len(points) != 3:
        raise ValueError("三点定圆需要恰好 3 个点")
    p1, p2, p3 = np.asarray(points, dtype=np.float64)
    matrix = np.array(
        [[2.0 * (p2[0] - p1[0]), 2.0 * (p2[1] - p1[1])],
         [2.0 * (p3[0] - p1[0]), 2.0 * (p3[1] - p1[1])]],
        dtype=np.float64,
    )
    rhs = np.array(
        [np.dot(p2, p2) - np.dot(p1, p1), np.dot(p3, p3) - np.dot(p1, p1)],
        dtype=np.float64,
    )
    if abs(float(np.linalg.det(matrix))) < 1e-10:
        raise ValueError("三个点接近共线，无法建立圆")
    center = np.linalg.solve(matrix, rhs)
    radius = float(np.linalg.norm(center - p1))
    return (float(center[0]), float(center[1])), radius


def cross2d(a, b):
    """z component of a 2-D cross product; np.cross dropped 2-D vectors in NumPy 2.5."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _line_intersection(a: list[Point], b: list[Point]) -> Point:
    if len(a) < 2 or len(b) < 2:
        raise ValueError("直线交点需要两条有效直线")
    p = np.asarray(a[0], dtype=np.float64)
    r = np.asarray(a[1], dtype=np.float64) - p
    q = np.asarray(b[0], dtype=np.float64)
    s = np.asarray(b[1], dtype=np.float64) - q
    cross = float(cross2d(r, s))
    if abs(cross) < 1e-10:
        raise ValueError("两条直线平行，无法计算交点")
    t = float(cross2d(q - p, s) / cross)
    result = p + t * r
    return float(result[0]), float(result[1])


def _project_point(point: Point, line: list[Point]) -> Point:
    if len(line) < 2:
        raise ValueError("投影点需要一条有效直线")
    p = np.asarray(point, dtype=np.float64)
    a = np.asarray(line[0], dtype=np.float64)
    direction = np.asarray(line[1], dtype=np.float64) - a
    denominator = float(np.dot(direction, direction))
    if denominator < 1e-12:
        raise ValueError("直线长度为零")
    projected = a + float(np.dot(p - a, direction) / denominator) * direction
    return float(projected[0]), float(projected[1])


def _feature_anchor(feature: GeometryFeatureResult) -> Point:
    if feature.center_px is not None:
        return feature.center_px
    if feature.points_px:
        return feature.points_px[0]
    raise ValueError(f"要素 {feature.name} 没有可用定位点")


def _feature_line(feature: GeometryFeatureResult) -> list[Point]:
    if len(feature.points_px) >= 2:
        return feature.points_px[:2]
    raise ValueError(f"要素 {feature.name} 不是有效直线")


def _detection_feature(
    definition: GeometryFeatureDefinition,
    detection: DetectionResult,
    config: MeasurementConfig,
) -> GeometryFeatureResult:
    contour = detection.shape_params.get("contour_points", detection.edge_points)
    points = [_point(value) for value in contour] if contour else []
    if definition.feature_type == "line":
        start = detection.shape_params.get("line_start_px")
        end = detection.shape_params.get("line_end_px")
        if start is not None and end is not None:
            points = [_point(start), _point(end)]
    radius = float(detection.shape_params.get("radius_px", detection.diameter_px / 2.0))
    result = GeometryFeatureResult(
        definition.feature_id,
        definition.name,
        definition.layer,
        definition.feature_type,
        "Valid",
        (float(detection.center_x_px), float(detection.center_y_px)),
        points,
        radius,
        float(detection.residual_px),
        str(detection.shape_params.get("quality_grade", "有效")),
        str(detection.shape_params.get("algorithm_path", "识别轮廓复用")),
    )
    result.radius_um = float(detection.shape_params.get("radius_um", detection.diameter_um / 2.0))
    return result


def _resolve_feature(
    definition: GeometryFeatureDefinition,
    resolved: Dict[str, GeometryFeatureResult],
    detections: Dict[str, DetectionResult],
    config: MeasurementConfig,
) -> GeometryFeatureResult:
    if definition.source == "detection":
        detection = detections.get(definition.detection_key)
        if detection is None:
            raise ValueError(f"找不到识别轮廓：{definition.detection_key}")
        base = _detection_feature(definition, detection, config)
        if definition.feature_type not in {"outer_circle_min", "outer_circle_robust"}:
            return base
        contour = np.asarray(base.points_px, dtype=np.float64)
        if len(contour) < 3:
            raise ValueError("轮廓点不足，无法计算外轮廓圆")
        if definition.feature_type == "outer_circle_min":
            (cx, cy), radius_um, residual_um = minimum_enclosing_circle_um(contour, config)
            path = "识别轮廓 → 凸包 → 最小外接圆"
        elif definition.feature_type == "outer_circle_robust":
            (cx, cy), radius_um, residual_um = robust_circle_um(contour, config)
            path = "识别轮廓 → 稳健正交距离拟合 → 外轮廓圆"
        else:
            raise ValueError(f"识别轮廓不支持要素类型：{definition.feature_type}")
        base.center_px = (float(cx), float(cy))
        base.radius_um = float(radius_um)
        base.radius_px = float(radius_um / mean_pixel_size_um(config))
        base.residual_px = float(residual_um / mean_pixel_size_um(config))
        base.algorithm_path = path
        return base

    if definition.feature_type == "point":
        if not definition.points_px:
            raise ValueError("点要素缺少坐标")
        point = _point(definition.points_px[0])
        return GeometryFeatureResult(definition.feature_id, definition.name, definition.layer, "point", "Valid", point, [point], algorithm_path="手动点")
    if definition.feature_type == "line":
        if len(definition.points_px) < 2:
            raise ValueError("直线要素至少需要 2 个点")
        points = [_point(value) for value in definition.points_px[:2]]
        center = tuple(np.mean(np.asarray(points), axis=0))
        return GeometryFeatureResult(definition.feature_id, definition.name, definition.layer, "line", "Valid", _point(center), points, algorithm_path="两点定直线")
    if definition.feature_type == "circle":
        center, radius = _circle_from_three_points([_point(value) for value in definition.points_px])
        return GeometryFeatureResult(definition.feature_id, definition.name, definition.layer, "circle", "Valid", center, list(definition.points_px), radius, 0.0, algorithm_path="三点定圆")

    references = [resolved[item] for item in definition.reference_ids]
    if definition.feature_type == "intersection":
        center = _line_intersection(_feature_line(references[0]), _feature_line(references[1]))
        path = "两直线交点"
    elif definition.feature_type == "midpoint":
        first, second = _feature_anchor(references[0]), _feature_anchor(references[1])
        center = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
        path = "两要素中心中点"
    elif definition.feature_type == "projection":
        center = _project_point(_feature_anchor(references[0]), _feature_line(references[1]))
        path = "点到直线垂足"
    else:
        raise ValueError(f"未知要素类型：{definition.feature_type}")
    return GeometryFeatureResult(definition.feature_id, definition.name, definition.layer, definition.feature_type, "Valid", center, [center], algorithm_path=path)


def _coordinate_system(
    definition: CoordinateSystemDefinition,
    features: Dict[str, GeometryFeatureResult],
    config: MeasurementConfig,
) -> CoordinateSystemResult:
    refs = [features[item] for item in definition.reference_ids]
    origin = _feature_anchor(refs[0])
    if definition.method in {"two_centers", "two_points"}:
        target = _feature_anchor(refs[1])
        vector = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    elif definition.method == "point_line":
        line = _feature_line(refs[1])
        vector = np.asarray(line[1], dtype=np.float64) - np.asarray(line[0], dtype=np.float64)
    else:
        raise ValueError(f"未知坐标系建立方式：{definition.method}")
    vector = np.asarray(
        [vector[0] * config.pixel_size_x_um, -vector[1] * config.pixel_size_y_um],
        dtype=np.float64,
    )
    length = float(np.linalg.norm(vector))
    if length < 1e-10:
        raise ValueError("坐标轴参考方向长度为零")
    vector /= length
    angle = np.deg2rad(float(definition.rotation_deg))
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    # Rotate in calibrated physical coordinates (Y up), then convert the axes
    # back to image-pixel directions for rendering.
    x_axis_physical = np.asarray(
        (cosine * vector[0] - sine * vector[1], sine * vector[0] + cosine * vector[1]),
        dtype=np.float64,
    )
    x_axis_physical /= np.linalg.norm(x_axis_physical)
    y_axis_physical = np.asarray((-x_axis_physical[1], x_axis_physical[0]), dtype=np.float64)
    x_axis = np.asarray(
        (x_axis_physical[0] / config.pixel_size_x_um, -x_axis_physical[1] / config.pixel_size_y_um),
        dtype=np.float64,
    )
    y_axis = np.asarray(
        (y_axis_physical[0] / config.pixel_size_x_um, -y_axis_physical[1] / config.pixel_size_y_um),
        dtype=np.float64,
    )
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return CoordinateSystemResult(
        definition.coordinate_id,
        definition.name,
        definition.layer,
        "Valid",
        origin,
        _point(x_axis),
        _point(y_axis),
        float(definition.rotation_deg),
    )


def coordinate_of_point(point_px: Point, coordinate: CoordinateSystemResult, config: MeasurementConfig) -> Point:
    if coordinate.origin_px is None or coordinate.x_axis_image is None or coordinate.y_axis_image is None:
        raise ValueError("坐标系无效")
    delta_px = np.asarray(point_px, dtype=np.float64) - np.asarray(coordinate.origin_px, dtype=np.float64)
    delta_um = np.asarray(
        [delta_px[0] * float(config.pixel_size_x_um), -delta_px[1] * float(config.pixel_size_y_um)],
        dtype=np.float64,
    )
    x_axis_px = np.asarray(coordinate.x_axis_image, dtype=np.float64)
    y_axis_px = np.asarray(coordinate.y_axis_image, dtype=np.float64)
    x_axis_um = np.asarray([x_axis_px[0] * config.pixel_size_x_um, -x_axis_px[1] * config.pixel_size_y_um])
    y_axis_um = np.asarray([y_axis_px[0] * config.pixel_size_x_um, -y_axis_px[1] * config.pixel_size_y_um])
    x_axis_um /= np.linalg.norm(x_axis_um)
    y_axis_um /= np.linalg.norm(y_axis_um)
    return float(np.dot(delta_um, x_axis_um)), float(np.dot(delta_um, y_axis_um))


def _distance_um(first: Point, second: Point, config: MeasurementConfig) -> float:
    return float(np.hypot(
        (second[0] - first[0]) * config.pixel_size_x_um,
        (second[1] - first[1]) * config.pixel_size_y_um,
    ))


def _line_angle_deg(line: list[Point], config: MeasurementConfig) -> float:
    dx = (line[1][0] - line[0][0]) * config.pixel_size_x_um
    dy = -(line[1][1] - line[0][1]) * config.pixel_size_y_um
    return float(degrees(atan2(dy, dx)))


def _measurement(
    definition: GeometryMeasurementDefinition,
    features: Dict[str, GeometryFeatureResult],
    coordinates: Dict[str, CoordinateSystemResult],
    config: MeasurementConfig,
) -> GeometryMeasurementResult:
    refs = [features[item] for item in definition.reference_ids]
    kind = definition.measurement_type
    unit = "μm"
    if kind in {"point_distance", "center_distance"}:
        value = _distance_um(_feature_anchor(refs[0]), _feature_anchor(refs[1]), config)
        path = "两要素中心 → 物理坐标距离"
    elif kind == "point_line_distance":
        point = np.asarray(_feature_anchor(refs[0]), dtype=np.float64)
        line = _feature_line(refs[1])
        a = np.asarray(line[0], dtype=np.float64)
        b = np.asarray(line[1], dtype=np.float64)
        scaled_point = np.asarray([point[0] * config.pixel_size_x_um, point[1] * config.pixel_size_y_um])
        scaled_a = np.asarray([a[0] * config.pixel_size_x_um, a[1] * config.pixel_size_y_um])
        scaled_b = np.asarray([b[0] * config.pixel_size_x_um, b[1] * config.pixel_size_y_um])
        direction = scaled_b - scaled_a
        value = float(abs(cross2d(direction, scaled_point - scaled_a)) / max(np.linalg.norm(direction), 1e-12))
        path = "点到直线正交距离"
    elif kind == "line_angle":
        value = _line_angle_deg(_feature_line(refs[0]), config)
        unit = "°"
        path = "直线方向角（物理坐标 Y 向上）"
    elif kind == "two_line_angle":
        first = np.deg2rad(_line_angle_deg(_feature_line(refs[0]), config))
        second = np.deg2rad(_line_angle_deg(_feature_line(refs[1]), config))
        cosine = float(np.clip(np.cos(first - second), -1.0, 1.0))
        value = float(degrees(acos(abs(cosine))))
        unit = "°"
        path = "两直线最小夹角"
    elif kind == "diameter":
        if refs[0].radius_um is None and refs[0].radius_px is None:
            raise ValueError("所选要素没有直径")
        value = 2.0 * (
            float(refs[0].radius_um)
            if refs[0].radius_um is not None
            else float(refs[0].radius_px) * mean_pixel_size_um(config)
        )
        path = "物理坐标圆半径 → 标定直径"
    elif kind in {"coordinate_x", "coordinate_y"}:
        coordinate = coordinates[definition.coordinate_id]
        x_value, y_value = coordinate_of_point(_feature_anchor(refs[0]), coordinate, config)
        value = x_value if kind == "coordinate_x" else y_value
        path = "要素中心 → 自定义坐标系"
    else:
        raise ValueError(f"未知测量类型：{kind}")
    return GeometryMeasurementResult(
        definition.measurement_id,
        definition.name,
        definition.layer,
        kind,
        "Valid",
        float(value),
        unit,
        "有效",
        path,
        reference_ids=list(definition.reference_ids),
    )


def execute_geometry_program(
    program: GeometryProgram,
    detections: Dict[str, DetectionResult],
    config: MeasurementConfig,
) -> GeometryRunResult:
    result = GeometryRunResult()
    pending = [item for item in program.features if item.enabled]
    while pending:
        progress = False
        for definition in list(pending):
            if any(reference not in result.features for reference in definition.reference_ids):
                continue
            try:
                feature = _resolve_feature(definition, result.features, detections, config)
            except Exception as exc:
                feature = GeometryFeatureResult(
                    definition.feature_id, definition.name, definition.layer,
                    definition.feature_type, "Invalid", error=str(exc), quality="无效",
                )
            result.features[definition.feature_id] = feature
            pending.remove(definition)
            progress = True
        if not progress:
            for definition in pending:
                result.features[definition.feature_id] = GeometryFeatureResult(
                    definition.feature_id, definition.name, definition.layer,
                    definition.feature_type, "Invalid", error="要素引用缺失或存在循环依赖", quality="无效",
                )
            break

    for definition in program.coordinate_systems:
        if not definition.enabled:
            continue
        try:
            if any(result.features[item].status != "Valid" for item in definition.reference_ids):
                raise ValueError("坐标系引用的要素无效")
            coordinate = _coordinate_system(definition, result.features, config)
        except Exception as exc:
            coordinate = CoordinateSystemResult(
                definition.coordinate_id, definition.name, definition.layer, "Invalid", error=str(exc)
            )
        result.coordinate_systems[definition.coordinate_id] = coordinate

    for definition in program.measurements:
        if not definition.enabled:
            continue
        try:
            if any(result.features[item].status != "Valid" for item in definition.reference_ids):
                raise ValueError("测量项目引用的要素无效")
            if definition.coordinate_id and result.coordinate_systems[definition.coordinate_id].status != "Valid":
                raise ValueError("测量项目引用的坐标系无效")
            measurement = _measurement(definition, result.features, result.coordinate_systems, config)
        except Exception as exc:
            measurement = GeometryMeasurementResult(
                definition.measurement_id, definition.name, definition.layer,
                definition.measurement_type, "Invalid", error=str(exc), quality="无效",
                reference_ids=list(definition.reference_ids),
            )
        result.measurements[definition.measurement_id] = measurement

    for definition in program.coordinate_labels:
        if not definition.enabled:
            continue
        try:
            feature = result.features[definition.feature_id]
            coordinate = result.coordinate_systems[definition.coordinate_id]
            if feature.status != "Valid" or coordinate.status != "Valid":
                raise ValueError("坐标标注引用无效")
            anchor = _feature_anchor(feature)
            x_um, y_um = coordinate_of_point(anchor, coordinate, config)
            label_px = (anchor[0] + definition.offset_px[0], anchor[1] + definition.offset_px[1])
            label = CoordinateLabelResult(
                definition.label_id, definition.name, definition.layer,
                definition.feature_id, definition.coordinate_id, "Valid",
                x_um, y_um, anchor, label_px,
            )
        except Exception as exc:
            label = CoordinateLabelResult(
                definition.label_id, definition.name, definition.layer,
                definition.feature_id, definition.coordinate_id, "Invalid", error=str(exc),
            )
        result.coordinate_labels[definition.label_id] = label
    return result


def flatten_detection_map(detections: dict) -> Dict[str, DetectionResult]:
    flattened: Dict[str, DetectionResult] = {}
    for identity, layer_map in detections.items():
        if isinstance(layer_map, DetectionResult):
            flattened[f"{identity}:{layer_map.layer}"] = layer_map
            continue
        for layer, detection in (layer_map or {}).items():
            if isinstance(detection, DetectionResult):
                flattened[f"{identity}:{layer}"] = detection
    return flattened
