from __future__ import annotations

from pathlib import Path

import pytest

from overlay_measure.image_loader import load_image
from overlay_measure.measurement_service import detect_manual_roi, describe_algorithm_path
from overlay_measure.measurement_engine import run_measurement_job
from overlay_measure.models import MarkRecipe
from overlay_measure.overlay_calculator import calculate_overlay
from overlay_measure.recipe_manager import load_recipe


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"


@pytest.mark.parametrize(
    ("recipe", "upper_image", "lower_image", "upper_mode", "lower_mode", "dx_um", "dy_um", "dxy_um"),
    [
        ("demo_recipe.json", "sample_upper.png", "sample_lower.png", "Circle", "Circle", 0.2417, 0.2528, 0.3498),
        ("demo_square_recipe.json", "sample_square_upper.png", "sample_square_lower.png", "Rectangle", "Rectangle", -2.7998, 2.1995, 3.5604),
        ("demo_mixed_square_circle_recipe.json", "sample_square_upper.png", "sample_lower.png", "Rectangle", "Circle", 5.6555, -5.6398, 7.9870),
        ("demo_annulus_recipe.json", "sample_concentric_single.png", "sample_concentric_single.png", "Circle", "Circle", -5.9855, 2.7836, 6.6011),
        ("demo_rect_ring_recipe.json", "sample_concentric_square_single.png", "sample_concentric_square_single.png", "Rectangle", "Rectangle", -5.5120, 2.9641, 6.2585),
    ],
)
def test_golden_manual_roi_measurements(recipe, upper_image, lower_image, upper_mode, lower_mode, dx_um, dy_um, dxy_um):
    config, params, marks = load_recipe(str(SAMPLE_DIR / recipe))
    mark = marks[0]
    upper = detect_manual_roi(
        mark.mark_id,
        "upper",
        load_image(str(SAMPLE_DIR / upper_image)),
        mark.upper_roi,
        params,
        config,
    )
    lower = detect_manual_roi(
        mark.mark_id,
        "lower",
        load_image(str(SAMPLE_DIR / lower_image)),
        mark.lower_roi,
        params,
        config,
    )
    overlay = calculate_overlay(mark.mark_id, upper, lower, config)

    assert upper.fitting_mode == upper_mode
    assert lower.fitting_mode == lower_mode
    assert overlay.delta_x_um == pytest.approx(dx_um, abs=0.02)
    assert overlay.delta_y_um == pytest.approx(dy_um, abs=0.02)
    assert overlay.overlay_r_um == pytest.approx(dxy_um, abs=0.02)


def test_algorithm_path_explains_manual_circle_pipeline():
    config, params, marks = load_recipe(str(SAMPLE_DIR / "demo_recipe.json"))
    mark = marks[0]
    detection = detect_manual_roi(
        mark.mark_id,
        "upper",
        load_image(str(SAMPLE_DIR / "sample_upper.png")),
        mark.upper_roi,
        params,
        config,
    )

    path = describe_algorithm_path(detection, "Manual")
    assert "手动ROI" in path
    assert "亚像素边缘" in path
    assert "RANSAC圆拟合" in path


def test_background_engine_reuses_main_measurement_service():
    config, params, marks = load_recipe(str(SAMPLE_DIR / "demo_recipe.json"))
    config.workflow_mode = "Manual"
    upper = load_image(str(SAMPLE_DIR / "sample_upper.png"))
    lower = load_image(str(SAMPLE_DIR / "sample_lower.png"))
    payload = run_measurement_job(
        {
            "config": config,
            "params": params,
            "marks": {"Mark1": marks[0], "Mark2": MarkRecipe("Mark2")},
            "mark_images": {
                "Mark1": {"upper": upper, "lower": lower},
                "Mark2": {"upper": None, "lower": None},
            },
            "batch_images": {
                "Mark1": {"upper": [], "lower": []},
                "Mark2": {"upper": [], "lower": []},
            },
            "selections": {"Mark1": {}, "Mark2": {}},
            "batch": False,
        },
        lambda done, total, text: None,
        lambda: False,
    )
    overlay = payload["overlays"]["Mark1"]
    # The production UI defines overlay as selected target minus selected reference.
    assert overlay.delta_x_um == pytest.approx(-0.2417, abs=0.02)
    assert overlay.delta_y_um == pytest.approx(-0.2528, abs=0.02)
    assert "亚像素边缘" in payload["detections"]["Mark1"]["upper"].shape_params["algorithm_path"]
