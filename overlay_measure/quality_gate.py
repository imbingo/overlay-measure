from __future__ import annotations

from typing import Iterable

from .models import DetectionResult, MeasurementConfig, OverlayResult
from .quality_profiles import annotate_detection_quality, quality_profile_display, worst_quality_grade


def detection_invalid_reasons(detection: DetectionResult, config: MeasurementConfig) -> list[str]:
    assessment = annotate_detection_quality(detection, config)
    return list(assessment.reasons)


def apply_quality_gate(
    overlay: OverlayResult,
    detections: Iterable[DetectionResult],
    config: MeasurementConfig,
) -> OverlayResult:
    detections = list(detections)
    invalid_reasons: list[str] = []
    for detection in detections:
        invalid_reasons.extend(detection_invalid_reasons(detection, config))
    overlay.quality_profile = quality_profile_display(config)
    overlay.quality_grade = worst_quality_grade(detections)
    overlay.quality_summary = "；".join(
        f"{'上层' if detection.layer == 'upper' else '下层'}："
        f"{detection.shape_params.get('quality_grade', '')}（{detection.shape_params.get('quality_details', '')}）"
        for detection in detections
    )
    if invalid_reasons:
        overlay.result = "Invalid"
        overlay.warning = "；".join(dict.fromkeys(invalid_reasons))
    elif config.recipe_validation_status != "Validated":
        overlay.result = "Trial"
        overlay.warning = "试测/未验证配方，不作正式判定"
    return overlay
