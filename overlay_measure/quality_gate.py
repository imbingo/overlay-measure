from __future__ import annotations

from typing import Iterable

from .models import DetectionResult, MeasurementConfig, OverlayResult


def detection_invalid_reasons(detection: DetectionResult, config: MeasurementConfig) -> list[str]:
    shape = detection.shape_params or {}
    reasons: list[str] = []
    if shape.get("quality_status") == "Invalid":
        reasons.append(str(shape.get("failure_reason") or detection.warning or "识别质量无效"))
    if detection.confidence < config.confidence_min:
        reasons.append(f"置信度 {detection.confidence:.3f} 低于 {config.confidence_min:.3f}")
    if detection.residual_um > config.production_max_residual_um:
        reasons.append(
            f"残差 {detection.residual_um:.3f} μm 超过 {config.production_max_residual_um:.3f} μm"
        )
    coverage = shape.get("coverage")
    if coverage is not None and float(coverage) < config.production_min_coverage:
        reasons.append(f"边缘覆盖率 {float(coverage):.1%} 不足")
    rejected_ratio = shape.get("rejected_ratio")
    if rejected_ratio is not None and float(rejected_ratio) > config.production_max_rejected_ratio:
        reasons.append(f"异常点比例 {float(rejected_ratio):.1%} 过高")
    max_deviation = shape.get("max_deviation_um")
    if max_deviation is not None and float(max_deviation) > config.production_max_radial_deviation_um:
        reasons.append(f"最大轮廓偏差 {float(max_deviation):.3f} μm 过大")
    return list(dict.fromkeys(reason for reason in reasons if reason))


def apply_quality_gate(
    overlay: OverlayResult,
    detections: Iterable[DetectionResult],
    config: MeasurementConfig,
) -> OverlayResult:
    invalid_reasons: list[str] = []
    for detection in detections:
        invalid_reasons.extend(detection_invalid_reasons(detection, config))
    if invalid_reasons:
        overlay.result = "Invalid"
        overlay.warning = "；".join(dict.fromkeys(invalid_reasons))
    elif config.recipe_validation_status != "Validated":
        overlay.result = "Trial"
        overlay.warning = "试测/未验证配方，不作正式判定"
    return overlay
