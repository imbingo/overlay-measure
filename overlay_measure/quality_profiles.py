from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from .models import DetectionResult, MeasurementConfig


@dataclass(frozen=True)
class QualityThresholds:
    confidence_min: float
    coverage_min: float
    rejected_ratio_max: float
    residual_um_max: float
    radial_deviation_um_max: float


QUALITY_PROFILES = {
    "Tolerant": QualityThresholds(0.45, 0.45, 0.65, 0.60, 1.20),
    "Standard": QualityThresholds(0.70, 0.65, 0.40, 0.30, 0.60),
    "UltraPrecise": QualityThresholds(0.85, 0.80, 0.20, 0.15, 0.30),
}

QUALITY_PROFILE_LABELS = {
    "Tolerant": "宽容",
    "Standard": "标准",
    "UltraPrecise": "超精确",
}

QUALITY_GRADE_ORDER = {
    "优秀（满足超精确）": 0,
    "合格（满足标准）": 1,
    "较差（仅宽容可用）": 2,
    "无效（低于宽容门槛）": 3,
}


@dataclass(frozen=True)
class QualityAssessment:
    profile: str
    profile_label: str
    grade: str
    valid: bool
    reasons: tuple[str, ...]
    details: str


def normalized_quality_profile(value: str) -> str:
    return value if value in QUALITY_PROFILES else "Standard"


def quality_profile_label(value: str) -> str:
    return QUALITY_PROFILE_LABELS[normalized_quality_profile(value)]


def configured_thresholds(config: MeasurementConfig) -> QualityThresholds:
    return QualityThresholds(
        float(config.confidence_min),
        float(config.production_min_coverage),
        float(config.production_max_rejected_ratio),
        float(config.production_max_residual_um),
        float(config.production_max_radial_deviation_um),
    )


def apply_quality_profile(config: MeasurementConfig, profile: str) -> None:
    profile = normalized_quality_profile(profile)
    thresholds = QUALITY_PROFILES[profile]
    config.quality_profile = profile
    config.confidence_min = thresholds.confidence_min
    config.production_min_coverage = thresholds.coverage_min
    config.production_max_rejected_ratio = thresholds.rejected_ratio_max
    config.production_max_residual_um = thresholds.residual_um_max
    config.production_max_radial_deviation_um = thresholds.radial_deviation_um_max


def quality_profile_is_modified(config: MeasurementConfig) -> bool:
    expected = QUALITY_PROFILES[normalized_quality_profile(getattr(config, "quality_profile", "Standard"))]
    actual = configured_thresholds(config)
    return any(
        abs(left - right) > 1e-9
        for left, right in zip(expected.__dict__.values(), actual.__dict__.values())
    )


def quality_profile_display(config: MeasurementConfig) -> str:
    label = quality_profile_label(getattr(config, "quality_profile", "Standard"))
    return f"{label}（参数已调整）" if quality_profile_is_modified(config) else label


def _metric_values(detection: DetectionResult) -> dict[str, float | None]:
    shape = detection.shape_params or {}
    coverage = shape.get("coverage", shape.get("angular_coverage"))
    rejected_ratio = shape.get("rejected_ratio")
    if rejected_ratio is None:
        rejected = len(detection.rejected_points)
        accepted = len(detection.edge_points)
        if rejected + accepted:
            rejected_ratio = rejected / (rejected + accepted)
    return {
        "confidence": float(detection.confidence),
        "residual_um": float(detection.residual_um),
        "coverage": float(coverage) if coverage is not None else None,
        "rejected_ratio": float(rejected_ratio) if rejected_ratio is not None else None,
        "max_deviation_um": (
            float(shape["max_deviation_um"]) if shape.get("max_deviation_um") is not None else None
        ),
    }


def _failures(metrics: dict[str, float | None], thresholds: QualityThresholds) -> list[str]:
    reasons: list[str] = []
    confidence = metrics["confidence"]
    residual = metrics["residual_um"]
    coverage = metrics["coverage"]
    rejected_ratio = metrics["rejected_ratio"]
    max_deviation = metrics["max_deviation_um"]
    if confidence is None or not isfinite(confidence) or confidence < thresholds.confidence_min:
        reasons.append(f"置信度 {confidence or 0.0:.3f} < {thresholds.confidence_min:.3f}")
    if residual is None or not isfinite(residual) or residual > thresholds.residual_um_max:
        value = residual if residual is not None and isfinite(residual) else float("inf")
        reasons.append(f"残差 {value:.3f} μm > {thresholds.residual_um_max:.3f} μm")
    if coverage is not None and (not isfinite(coverage) or coverage < thresholds.coverage_min):
        reasons.append(f"覆盖率 {coverage:.1%} < {thresholds.coverage_min:.1%}")
    if rejected_ratio is not None and (
        not isfinite(rejected_ratio) or rejected_ratio > thresholds.rejected_ratio_max
    ):
        reasons.append(f"异常点比例 {rejected_ratio:.1%} > {thresholds.rejected_ratio_max:.1%}")
    if max_deviation is not None and (
        not isfinite(max_deviation) or max_deviation > thresholds.radial_deviation_um_max
    ):
        reasons.append(
            f"最大轮廓偏差 {max_deviation:.3f} μm > {thresholds.radial_deviation_um_max:.3f} μm"
        )
    return reasons


def _hard_failure(detection: DetectionResult) -> str:
    shape = detection.shape_params or {}
    if shape.get("quality_hard_failure"):
        return str(shape.get("failure_reason") or detection.warning or "识别失败")
    # Results created before the unified evaluator may already carry a hard
    # invalid state. New gate failures always include quality_evaluation_version.
    if shape.get("quality_status") == "Invalid" and not shape.get("quality_evaluation_version"):
        reason = str(shape.get("failure_reason") or detection.warning or "")
        if reason:
            return reason
    return ""


def _actual_grade(metrics: dict[str, float | None], hard_failure: str) -> str:
    if hard_failure:
        return "无效（低于宽容门槛）"
    if not _failures(metrics, QUALITY_PROFILES["UltraPrecise"]):
        return "优秀（满足超精确）"
    if not _failures(metrics, QUALITY_PROFILES["Standard"]):
        return "合格（满足标准）"
    if not _failures(metrics, QUALITY_PROFILES["Tolerant"]):
        return "较差（仅宽容可用）"
    return "无效（低于宽容门槛）"


def _details(metrics: dict[str, float | None]) -> str:
    parts = [
        f"置信度={metrics['confidence']:.3f}",
        f"残差={metrics['residual_um']:.3f} μm",
    ]
    if metrics["coverage"] is not None:
        parts.append(f"覆盖率={metrics['coverage']:.1%}")
    if metrics["rejected_ratio"] is not None:
        parts.append(f"异常点={metrics['rejected_ratio']:.1%}")
    if metrics["max_deviation_um"] is not None:
        parts.append(f"最大偏差={metrics['max_deviation_um']:.3f} μm")
    return "；".join(parts)


def assess_detection_quality(detection: DetectionResult, config: MeasurementConfig) -> QualityAssessment:
    profile = normalized_quality_profile(getattr(config, "quality_profile", "Standard"))
    profile_text = quality_profile_display(config)
    metrics = _metric_values(detection)
    hard_failure = _hard_failure(detection)
    reasons = [hard_failure] if hard_failure else _failures(metrics, configured_thresholds(config))
    reasons = list(dict.fromkeys(reason for reason in reasons if reason))
    return QualityAssessment(
        profile=profile,
        profile_label=profile_text,
        grade=_actual_grade(metrics, hard_failure),
        valid=not reasons,
        reasons=tuple(reasons),
        details=_details(metrics),
    )


def annotate_detection_quality(detection: DetectionResult, config: MeasurementConfig) -> QualityAssessment:
    previous_gate_warning = str(detection.shape_params.get("quality_gate_warning", ""))
    if previous_gate_warning and detection.warning == previous_gate_warning:
        detection.warning = ""
    assessment = assess_detection_quality(detection, config)
    shape = detection.shape_params
    shape["quality_profile"] = assessment.profile
    shape["quality_profile_label"] = assessment.profile_label
    shape["quality_grade"] = assessment.grade
    shape["quality_details"] = assessment.details
    shape["quality_status"] = "Valid" if assessment.valid else "Invalid"
    shape["quality_gate_reasons"] = list(assessment.reasons)
    shape["quality_gate_warning"] = "；".join(assessment.reasons)
    shape["quality_evaluation_version"] = 1
    if assessment.reasons:
        shape["failure_reason"] = "；".join(assessment.reasons)
    elif not shape.get("quality_hard_failure"):
        shape["failure_reason"] = ""
    if assessment.reasons and not detection.warning:
        detection.warning = shape["quality_gate_warning"]
    return assessment


def worst_quality_grade(detections: Iterable[DetectionResult]) -> str:
    grades = [
        str(detection.shape_params.get("quality_grade", "无效（低于宽容门槛）"))
        for detection in detections
    ]
    return max(grades, key=lambda grade: QUALITY_GRADE_ORDER.get(grade, 99)) if grades else ""
