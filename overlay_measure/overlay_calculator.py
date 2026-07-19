from __future__ import annotations

import math
from typing import Dict, Optional

from .models import DetectionResult, MeasurementConfig, OverlayResult
from .quality_gate import apply_quality_gate


def _angle_compensation_um(config: MeasurementConfig) -> tuple[float, float]:
    """Return equipment-coordinate compensation from material tilt.

    thickness(mm) * angle(urad) / 1000 = displacement(um).
    Rx tilts compensate Y with the opposite sign; Ry tilts compensate X.
    """
    thickness_mm = float(getattr(config, "material_thickness_mm", 0.0) or 0.0)
    rx_urad = float(getattr(config, "rx_angle_urad", 0.0) or 0.0)
    ry_urad = float(getattr(config, "ry_angle_urad", 0.0) or 0.0)
    return thickness_mm * ry_urad / 1000.0, -thickness_mm * rx_urad / 1000.0


def _um_to_px(value_um: float, pixel_size_um: float) -> float:
    return value_um / pixel_size_um if pixel_size_um else 0.0


def calculate_overlay(mark_id: str, upper: DetectionResult, lower: DetectionResult, config: MeasurementConfig) -> OverlayResult:
    off_x_px = config.registration_offset_x_um / config.pixel_size_x_um if config.pixel_size_x_um else 0.0
    off_y_px = config.registration_offset_y_um / config.pixel_size_y_um if config.pixel_size_y_um else 0.0

    lower_x_corr_px = lower.center_x_px + off_x_px
    lower_y_corr_px = lower.center_y_px + off_y_px

    # Internal image coordinates use X right-positive and Y down-positive.
    # Measurement output follows equipment/Keyence-style coordinates:
    # X right-positive, Y up-positive. Therefore only the final Y delta is inverted.
    delta_x_px = upper.center_x_px - lower_x_corr_px
    delta_y_image_px = upper.center_y_px - lower_y_corr_px
    delta_y_px = -delta_y_image_px
    comp_x_um, comp_y_um = _angle_compensation_um(config)
    delta_x_um = delta_x_px * config.pixel_size_x_um + comp_x_um
    delta_y_um = delta_y_px * config.pixel_size_y_um + comp_y_um
    delta_x_px += _um_to_px(comp_x_um, config.pixel_size_x_um)
    delta_y_px += _um_to_px(comp_y_um, config.pixel_size_y_um)
    r_um = math.sqrt(delta_x_um * delta_x_um + delta_y_um * delta_y_um)

    warnings = []
    if abs(delta_x_um) > config.delta_x_limit_um:
        warnings.append("ΔX 超限")
    if abs(delta_y_um) > config.delta_y_limit_um:
        warnings.append("ΔY 超限")
    if r_um > config.overlay_r_limit_um:
        warnings.append("R 超限")
    result = "Fail" if warnings else "Pass"
    overlay = OverlayResult(
        mark_id=mark_id,
        delta_x_px=delta_x_px,
        delta_y_px=delta_y_px,
        delta_x_um=delta_x_um,
        delta_y_um=delta_y_um,
        overlay_r_um=r_um,
        result=result,
        warning="; ".join(warnings),
    )
    return apply_quality_gate(overlay, (upper, lower), config)


def calculate_relative_overlay(
    mark_id: str,
    reference: DetectionResult,
    target: DetectionResult,
    config: MeasurementConfig,
) -> OverlayResult:
    """Calculate target position relative to the selected reference contour."""
    offset_x_px = config.registration_offset_x_um / config.pixel_size_x_um if config.pixel_size_x_um else 0.0
    offset_y_px = config.registration_offset_y_um / config.pixel_size_y_um if config.pixel_size_y_um else 0.0

    def corrected_center(detection: DetectionResult):
        if config.mode == "Dual Image" and detection.layer == "lower":
            return detection.center_x_px + offset_x_px, detection.center_y_px + offset_y_px
        return detection.center_x_px, detection.center_y_px

    reference_x, reference_y = corrected_center(reference)
    target_x, target_y = corrected_center(target)
    # Internal image coordinates use X right-positive and Y down-positive.
    # Measurement output follows equipment/Keyence-style coordinates:
    # X right-positive, Y up-positive. Therefore only the final Y delta is inverted.
    delta_x_px = target_x - reference_x
    delta_y_image_px = target_y - reference_y
    delta_y_px = -delta_y_image_px
    comp_x_um, comp_y_um = _angle_compensation_um(config)
    delta_x_um = delta_x_px * config.pixel_size_x_um + comp_x_um
    delta_y_um = delta_y_px * config.pixel_size_y_um + comp_y_um
    delta_x_px += _um_to_px(comp_x_um, config.pixel_size_x_um)
    delta_y_px += _um_to_px(comp_y_um, config.pixel_size_y_um)
    distance_um = math.hypot(delta_x_um, delta_y_um)
    warnings = []
    if abs(delta_x_um) > config.delta_x_limit_um:
        warnings.append("Dx 超限")
    if abs(delta_y_um) > config.delta_y_limit_um:
        warnings.append("Dy 超限")
    if distance_um > config.overlay_r_limit_um:
        warnings.append("Dxy 超限")
    overlay = OverlayResult(
        mark_id=mark_id,
        delta_x_px=delta_x_px,
        delta_y_px=delta_y_px,
        delta_x_um=delta_x_um,
        delta_y_um=delta_y_um,
        overlay_r_um=distance_um,
        result="Fail" if warnings else "Pass",
        warning="；".join(warnings),
    )
    return apply_quality_gate(overlay, (reference, target), config)
