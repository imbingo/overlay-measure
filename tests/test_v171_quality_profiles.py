from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor

from overlay_measure.models import DetectionResult, MeasurementConfig, OverlayResult
from overlay_measure.quality_gate import apply_quality_gate
from overlay_measure.quality_profiles import apply_quality_profile
from overlay_measure.result_exporter import build_detection_rows
from overlay_measure.ui_main import MainWindow


def _detection(
    *,
    confidence: float,
    residual_um: float,
    coverage: float,
    rejected_ratio: float,
    max_deviation_um: float,
) -> DetectionResult:
    return DetectionResult(
        mark_id="Mark1",
        layer="upper",
        center_x_px=10.0,
        center_y_px=10.0,
        center_x_um=1.0,
        center_y_um=1.0,
        diameter_px=20.0,
        diameter_um=2.0,
        residual_px=residual_um / 0.1,
        residual_um=residual_um,
        edge_point_count=64,
        confidence=confidence,
        fitting_mode="CaliperCircle",
        shape_params={
            "coverage": coverage,
            "rejected_ratio": rejected_ratio,
            "max_deviation_um": max_deviation_um,
        },
    )


def _overlay() -> OverlayResult:
    return OverlayResult("Mark1", 0.0, 0.0, 0.1, 0.1, 0.141, "Pass")


def test_tolerant_profile_accepts_poor_material_without_hiding_actual_quality():
    detection = _detection(
        confidence=0.55,
        residual_um=0.45,
        coverage=0.55,
        rejected_ratio=0.50,
        max_deviation_um=0.90,
    )
    standard = MeasurementConfig(recipe_validation_status="Validated")
    standard_result = apply_quality_gate(_overlay(), [detection], standard)
    assert standard_result.result == "Invalid"
    assert detection.shape_params["quality_grade"] == "较差（仅宽容可用）"

    tolerant = MeasurementConfig(recipe_validation_status="Validated")
    apply_quality_profile(tolerant, "Tolerant")
    tolerant_result = apply_quality_gate(_overlay(), [detection], tolerant)
    assert tolerant_result.result == "Pass"
    assert tolerant_result.quality_profile == "宽容"
    assert tolerant_result.quality_grade == "较差（仅宽容可用）"
    assert "置信度=0.550" in tolerant_result.quality_summary


def test_ultra_precise_profile_reports_excellent_quality():
    config = MeasurementConfig(recipe_validation_status="Validated")
    apply_quality_profile(config, "UltraPrecise")
    detection = _detection(
        confidence=0.92,
        residual_um=0.10,
        coverage=0.90,
        rejected_ratio=0.10,
        max_deviation_um=0.20,
    )
    result = apply_quality_gate(_overlay(), [detection], config)
    assert result.result == "Pass"
    assert result.quality_profile == "超精确"
    assert result.quality_grade == "优秀（满足超精确）"


def test_quality_profile_combo_applies_presets_and_marks_manual_adjustment():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._set_operation_mode("Engineering", authenticated=True)

    window._set_combo_value(window.quality_profile_combo, "Tolerant")
    app.processEvents()
    assert window.config.quality_profile == "Tolerant"
    assert window.conf_min_spin.value() == 0.45
    assert window.production_residual_spin.value() == 0.60
    assert "当前：宽容" in window.quality_profile_hint.text()

    window.production_residual_spin.setValue(0.75)
    app.processEvents()
    assert "参数已调整" in window.quality_profile_hint.text()
    window.close()
    app.processEvents()


def test_quality_is_explicit_in_detail_overlay_and_export_rows():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    apply_quality_profile(window.config, "Tolerant")
    window.config.recipe_validation_status = "Validated"
    detection = _detection(
        confidence=0.55,
        residual_um=0.45,
        coverage=0.55,
        rejected_ratio=0.50,
        max_deviation_um=0.90,
    )
    overlay = apply_quality_gate(_overlay(), [detection], window.config)
    window.detections = {"Mark1": {"upper": detection}}
    window.overlays = {"Mark1": overlay}
    window._refresh_tables()

    detail_headers = [
        window.det_table.horizontalHeaderItem(index).text()
        for index in range(window.det_table.columnCount())
    ]
    overlay_headers = [
        window.overlay_table.horizontalHeaderItem(index).text()
        for index in range(window.overlay_table.columnCount())
    ]
    assert {"质量门槛", "实际质量", "质量详情"}.issubset(detail_headers)
    assert {"质量门槛", "实际质量", "质量详情"}.issubset(overlay_headers)
    assert "较差（仅宽容可用）" in " ".join(
        window.det_table.item(0, index).text() for index in range(window.det_table.columnCount())
    )
    assert window.det_table.item(0, 0).background().color() == QColor(255, 243, 205)

    rows = build_detection_rows(
        window.detections,
        window.overlays,
        window.config,
    )
    assert rows[0]["quality_profile"] == "宽容"
    assert rows[0]["quality_grade"] == "较差（仅宽容可用）"
    assert "覆盖率=55.0%" in rows[0]["quality_details"]
    window.close()
    app.processEvents()
