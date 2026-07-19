from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from overlay_measure.models import DetectionParams, ImageData, MarkRecipe, MeasurementConfig, OverlayResult, Roi
from overlay_measure.recipe_library import RecipeLibrary
from overlay_measure.recipe_manager import save_recipe
from overlay_measure.ui_main import MainWindow


def _save_test_recipe(path: Path, name: str, status: str = "Draft", material: str = "MAT-01") -> None:
    config = MeasurementConfig(
        recipe_name=name,
        recipe_version="2.1",
        recipe_validation_status=status,
        material_code=material,
    )
    marks = [MarkRecipe("Mark1", upper_roi=Roi(10, 12, 30, 30)), MarkRecipe("Mark2")]
    save_recipe(str(path), config, DetectionParams(), marks)


def _image(path: str) -> ImageData:
    gray = np.zeros((32, 32), dtype=np.float32)
    return ImageData(path, gray, Path(path).name, "uint8", 0.0, 255.0)


def test_recipe_library_import_scan_favorite_and_recent_are_persistent(tmp_path):
    source = tmp_path / "external_recipe.json"
    _save_test_recipe(source, "CP 对位配方", "Validated", "CP-100")
    library = RecipeLibrary(tmp_path / "library")

    managed = library.import_recipe(source)

    assert managed.parent == library.validated_dir
    assert source.exists()
    entries = library.scan()
    assert len(entries) == 1
    assert entries[0].name == "CP 对位配方"
    assert entries[0].material_code == "CP-100"
    assert entries[0].version == "2.1"
    assert entries[0].source == "本机"

    assert library.toggle_favorite(managed) is True
    library.mark_used(managed)
    reloaded = RecipeLibrary(tmp_path / "library").scan()[0]
    assert reloaded.favorite
    assert reloaded.last_used


def test_recipe_library_discovers_shared_recipes_without_copying(tmp_path):
    library = RecipeLibrary(tmp_path / "library")
    shared = tmp_path / "company" / "recipes"
    shared.mkdir(parents=True)
    shared_recipe = shared / "shared.json"
    _save_test_recipe(shared_recipe, "公司标准配方", "Validated", "STD-8")
    library.set_shared_library(shared)

    entries = library.scan()

    assert len(entries) == 1
    assert entries[0].source == "共享"
    assert entries[0].path == shared_recipe
    assert not any(library.validated_dir.glob("*.json"))


def test_recipe_switch_preserves_images_and_replaces_rois_and_results(tmp_path, monkeypatch):
    monkeypatch.setenv("OVERLAY_MEASURE_RECIPE_LIBRARY", str(tmp_path / "library"))
    recipe_path = tmp_path / "new_recipe.json"
    _save_test_recipe(recipe_path, "新配方", "Draft")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    image = _image("upper.tif")
    window._set_image_for_layer("Mark1", "upper", image, "single")
    window.batch_images["Mark1"]["upper"] = [image]
    window.marks["Mark1"].upper_roi = Roi(1, 1, 8, 8)
    window.roi_sources["Mark1"]["upper"] = "manual"
    window.overlays["Mark1"] = OverlayResult("Mark1", 0, 0, 0.1, 0.2, 0.224, "Pass")

    assert window._load_recipe_from_path(str(recipe_path), confirm_switch=False, show_message=False)

    assert window.mark_images["Mark1"]["upper"] is image
    assert window.batch_images["Mark1"]["upper"] == [image]
    assert not window.overlays
    assert window.marks["Mark1"].upper_roi == Roi(10, 12, 30, 30)
    assert window.roi_sources["Mark1"]["upper"] == "recipe"
    assert window.loaded_recipe_display_name == "新配方"
    assert "新配方" in window.load_recipe_btn.text()
    window.close()
    app.processEvents()
