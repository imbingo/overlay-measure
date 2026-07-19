from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

from overlay_measure.access_control import AccessController
from overlay_measure.batch_pairing import validate_batch_pairing
from overlay_measure.models import (
    DetectionParams,
    DetectionResult,
    ImageData,
    MarkRecipe,
    MeasurementConfig,
    OverlayResult,
)
from overlay_measure.quality_gate import apply_quality_gate
from overlay_measure.recipe_integrity import seal_recipe, verify_recipe
from overlay_measure.traceability import create_measurement_archive
from overlay_measure.ui_main import MainWindow
from overlay_measure.measurement_engine import run_measurement_job


def _detection(*, confidence: float = 0.95, residual_um: float = 0.02) -> DetectionResult:
    return DetectionResult(
        mark_id="Mark1",
        layer="upper",
        center_x_px=10.0,
        center_y_px=11.0,
        center_x_um=1.0,
        center_y_um=1.1,
        diameter_px=20.0,
        diameter_um=2.0,
        residual_px=0.2,
        residual_um=residual_um,
        edge_point_count=64,
        confidence=confidence,
        fitting_mode="Circle",
        shape_params={"coverage": 0.95, "rejected_ratio": 0.05},
    )


def test_default_engineering_password_is_hashed(tmp_path):
    controller = AccessController(tmp_path)
    assert controller.verify("admin123")
    assert not controller.verify("wrong-password")
    assert "admin123" not in controller.settings_path.read_text(encoding="utf-8")


def test_engineering_mode_requires_password(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.access_controller = AccessController(tmp_path)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Ok)

    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("wrong", True))
    window.operation_mode_combo.setCurrentIndex(window.operation_mode_combo.findData("Engineering"))
    assert window.operation_mode == "Production"

    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("admin123", True))
    window.operation_mode_combo.setCurrentIndex(window.operation_mode_combo.findData("Engineering"))
    assert window.operation_mode == "Engineering"
    assert window.side_tabs.isTabEnabled(2)
    assert window.side_tabs.isTabEnabled(3)
    assert window.change_engineering_password_btn.isEnabled()
    window.close()
    app.processEvents()


def test_recipe_integrity_detects_changes(tmp_path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text('{"recipe": 1}', encoding="utf-8")
    digest = seal_recipe(recipe)
    assert verify_recipe(recipe) == ("Verified", digest)

    recipe.write_text('{"recipe": 2}', encoding="utf-8")
    status, changed_digest = verify_recipe(recipe)
    assert status == "Mismatch"
    assert changed_digest != digest


def test_quality_gate_produces_four_state_verdicts():
    config = MeasurementConfig(recipe_validation_status="Validated")
    passed = apply_quality_gate(OverlayResult("Mark1", 0, 0, 0.1, 0.1, 0.141, "Pass"), [_detection()], config)
    assert passed.result == "Pass"

    exceeded = apply_quality_gate(OverlayResult("Mark1", 0, 0, 1.0, 0.0, 1.0, "Fail"), [_detection()], config)
    assert exceeded.result == "Fail"

    invalid = apply_quality_gate(
        OverlayResult("Mark1", 0, 0, 0.1, 0.1, 0.141, "Pass"),
        [_detection(confidence=0.1)],
        config,
    )
    assert invalid.result == "Invalid"

    trial_config = MeasurementConfig(recipe_validation_status="Draft")
    trial = apply_quality_gate(OverlayResult("Mark1", 0, 0, 0.1, 0.1, 0.141, "Pass"), [_detection()], trial_config)
    assert trial.result == "Trial"


def test_batch_pairing_rejects_mismatch_and_same_file(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    image1 = ImageData(str(first), None, first.name)
    image2 = ImageData(str(second), None, second.name)

    mismatched = {"Mark1": {"upper": [image1, image2], "lower": [image1]}, "Mark2": {"upper": [], "lower": []}}
    assert any("数量不一致" in item for item in validate_batch_pairing(mismatched, True))

    duplicated = {"Mark1": {"upper": [image1], "lower": [image1]}, "Mark2": {"upper": [], "lower": []}}
    assert any("同一个文件" in item for item in validate_batch_pairing(duplicated, True))


def test_traceability_archive_contains_hashes_and_results(monkeypatch, tmp_path):
    import overlay_measure.traceability as traceability

    monkeypatch.setattr(traceability, "app_data_root", lambda: tmp_path)
    image = tmp_path / "upper.png"
    image.write_bytes(b"image-data")
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}", encoding="utf-8")
    recipe_hash = seal_recipe(recipe)
    config = MeasurementConfig(
        recipe_name="Production Recipe",
        recipe_version="1.0",
        recipe_validation_status="Validated",
        material_code="MAT-001",
    )
    overlay = OverlayResult("Mark1", 0, 0, 0.1, -0.1, 0.141, "Pass")
    measurement_id, archive = create_measurement_archive(
        config,
        DetectionParams(),
        str(recipe),
        recipe_hash,
        [str(image)],
        {"Mark1": overlay},
        {"Mark1": [], "Mark2": []},
        "Production",
    )

    manifest = json.loads((archive / "measurement.json").read_text(encoding="utf-8"))
    assert manifest["measurement_id"] == measurement_id
    assert manifest["operation_mode"] == "Production"
    assert manifest["recipe"]["sha256"] == recipe_hash
    assert manifest["inputs"][0]["sha256"]
    assert manifest["results"]["Mark1"]["result"] == "Pass"


def test_production_preflight_requires_verified_recipe_and_operator(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window._production_preflight_errors()

    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}", encoding="utf-8")
    window.loaded_recipe_path = str(recipe)
    window.loaded_recipe_hash = seal_recipe(recipe)
    window.recipe_integrity_status = "Verified"
    window.config.recipe_validation_status = "Validated"
    window.config.material_code = "MAT-001"
    window.config.operator_name = "OP-01"
    assert window._production_preflight_errors() == []
    window.close()
    app.processEvents()


def test_single_measurement_exception_is_reported_as_error():
    job = {
        "config": MeasurementConfig(workflow_mode="Manual", recipe_validation_status="Validated"),
        "params": DetectionParams(),
        "marks": {"Mark1": MarkRecipe("Mark1"), "Mark2": MarkRecipe("Mark2")},
        "mark_images": {"Mark1": {"upper": ImageData("missing", None, "missing"), "lower": None},
                        "Mark2": {"upper": None, "lower": None}},
        "batch_images": {"Mark1": {"upper": [], "lower": []}, "Mark2": {"upper": [], "lower": []}},
        "selections": {},
        "batch": False,
    }
    payload = run_measurement_job(job, lambda *args: None, lambda: False)
    assert payload["overlays"]["Mark1"].result == "Error"
