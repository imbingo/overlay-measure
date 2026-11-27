from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import openpyxl
import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from overlay_measure.auto_mark_detector import detect_auto_marks_with_report
from overlay_measure.batch_image_store import BatchImageRef
from overlay_measure.circle_ellipse_fitter import fit_ellipse
from overlay_measure.diagnostics.measurement_bridge import align_measurement, snapshot_main
from overlay_measure.export_visualization import render_measurement_image
from overlay_measure.geometry_engine import execute_geometry_program
from overlay_measure.geometry_models import GeometryFeatureDefinition, GeometryMeasurementDefinition, GeometryProgram
from overlay_measure.image_loader import display_to_uint8, load_image
from overlay_measure.measurement_engine import run_measurement_job
from overlay_measure.measurement_service import detect_manual_roi
from overlay_measure.measurement_units import contour_ellipse_metrics_um, ellipse_metrics_um
from overlay_measure.models import (
    DetectionParams,
    DetectionResult,
    ImageData,
    MarkRecipe,
    MeasurementConfig,
    OverlayResult,
    Roi,
)
from overlay_measure.overlay_calculator import calculate_overlay, calculate_relative_overlay
from overlay_measure.production_measurement import refine_candidate
from overlay_measure.recipe_integrity import seal_recipe, verify_recipe
from overlay_measure.recipe_library import RecipeLibrary
from overlay_measure.recipe_manager import load_recipe, save_recipe
from overlay_measure.region_center_detector import detect_region_center
from overlay_measure.result_exporter import build_detection_rows, export_data_tables, export_results
from overlay_measure.rz_calculator import build_summary_rows
from overlay_measure.subpixel_edge_detector import refine_contour_edges
from overlay_measure.ui_main import MainWindow


def _app():
    return QApplication.instance() or QApplication([])


def test_loading_dual_recipe_from_single_startup_preserves_all_fields(tmp_path):
    app = _app()
    window = MainWindow()
    try:
        recipe_path = tmp_path / "dual_recipe.json"
        cfg = MeasurementConfig(
            mode="Dual Image",
            workflow_mode="Auto",
            material_code="MAT-999",
            recipe_name="DualTest",
            recipe_version="2.0",
            recipe_validation_status="Validated",
            pixel_size_x_um=0.25,
            pixel_size_y_um=0.35,
            delta_x_limit_um=1.5,
            delta_y_limit_um=1.8,
        )
        prm = DetectionParams(gaussian_sigma_px=2.2, min_gradient=25.0, use_ransac=False)
        mark1 = MarkRecipe("Mark1", reference_contour_id="roi_ref_1", target_contour_id="roi_tgt_2")
        save_recipe(str(recipe_path), cfg, prm, [mark1])
        seal_recipe(recipe_path)

        ok = window._load_recipe_from_path(str(recipe_path), confirm_switch=False, show_message=False)
        assert ok is True
        assert window.config.mode == "Dual Image"
        assert window.config.workflow_mode == "Auto"
        assert window.config.material_code == "MAT-999"
        assert window.config.recipe_validation_status == "Validated"
        assert window.config.pixel_size_x_um == pytest.approx(0.25)
        assert window.config.pixel_size_y_um == pytest.approx(0.35)
        assert window.config.delta_x_limit_um == pytest.approx(1.5)
        assert window.params.gaussian_sigma_px == pytest.approx(2.2)
        assert window.params.use_ransac is False
        assert window.marks["Mark1"].reference_contour_id == "roi_ref_1"
        assert window.marks["Mark1"].target_contour_id == "roi_tgt_2"
        assert window.recipe_integrity_status == "Verified"

        # Editing mode/params marks loaded recipe dirty and clears stale hash.
        window.invalidate_measurement_state("参数已修改")
        assert window.recipe_integrity_status == "Modified"
        assert window.loaded_recipe_hash == ""
    finally:
        window.close()
        app.processEvents()


def test_registration_offset_y_has_consistent_equipment_sign():
    cfg = MeasurementConfig(
        mode="Dual Image",
        pixel_size_x_um=1.0,
        pixel_size_y_um=1.0,
        registration_offset_x_um=5.0,
        registration_offset_y_um=5.0,
        recipe_validation_status="Validated",
    )
    upper = DetectionResult("Mark1", "upper", 100.0, 100.0, 100.0, 100.0, 20.0, 20.0, 0.01, 0.01, 64, 0.99, "Circle")
    lower = DetectionResult("Mark1", "lower", 100.0, 100.0, 100.0, 100.0, 20.0, 20.0, 0.01, 0.01, 64, 0.99, "Circle")

    ov = calculate_overlay("Mark1", upper, lower, cfg)
    assert ov.delta_x_um == pytest.approx(-5.0)
    assert ov.delta_y_um == pytest.approx(-5.0)

    rel = calculate_relative_overlay("Mark1", upper, lower, cfg)
    assert rel.delta_x_um == pytest.approx(-5.0)
    assert rel.delta_y_um == pytest.approx(-5.0)

    # Verify lower - upper direction and Y-up physical center_y_um consistency.
    upper_shift = DetectionResult("Mark1", "upper", 100.0, 150.0, 100.0, 50.0, 20.0, 20.0, 0.01, 0.01, 64, 0.99, "Circle")
    lower_shift = DetectionResult("Mark1", "lower", 103.4, 147.4, 103.4, 52.6, 20.0, 20.0, 0.01, 0.01, 64, 0.99, "Circle")
    cfg_zero = MeasurementConfig(mode="Dual Image", pixel_size_x_um=1.0, pixel_size_y_um=1.0, recipe_validation_status="Validated")
    ov_dir = calculate_overlay("Mark1", upper_shift, lower_shift, cfg_zero)
    rel_dir = calculate_relative_overlay("Mark1", upper_shift, lower_shift, cfg_zero)
    assert ov_dir.delta_x_um == pytest.approx(3.4)
    assert ov_dir.delta_y_um == pytest.approx(2.6)
    assert rel_dir.delta_x_um == pytest.approx(ov_dir.delta_x_um)
    assert rel_dir.delta_y_um == pytest.approx(ov_dir.delta_y_um)
    assert (lower_shift.center_y_um - upper_shift.center_y_um) == pytest.approx(ov_dir.delta_y_um)


def test_rz_rejects_invalid_error_trial_and_nan_marks():
    cfg = MeasurementConfig(rz_distance_l_um=1000.0, rz_limit=100.0, recipe_validation_status="Validated")
    err = OverlayResult("Mark1", 0.0, 0.0, 0.0, 0.0, 0.0, "Error", "失败")
    inv = OverlayResult("Mark2", 0.0, 0.0, 0.0, 0.0, 0.0, "Invalid", "质量无效")
    rows = build_summary_rows({"Mark1": err, "Mark2": inv}, cfg)
    rz_row = next(r for r in rows if r["项目"] == "Rz")
    assert rz_row["Rz(μrad)"] is None
    assert rz_row["判定"] == "异常"

    inv1 = OverlayResult("Mark1", 0.0, 0.0, 0.0, 0.0, 0.0, "Invalid", "质量无效")
    rows_inv = build_summary_rows({"Mark1": inv1, "Mark2": inv}, cfg)
    rz_inv = next(r for r in rows_inv if r["项目"] == "Rz")
    assert rz_inv["Rz(μrad)"] is None
    assert rz_inv["判定"] == "无效"

    t1 = OverlayResult("Mark1", 0.0, 0.0, 0.1, 0.0, 0.1, "Trial", "试测")
    t2 = OverlayResult("Mark2", 0.0, 0.0, 0.2, 0.0, 0.2, "Trial", "试测")
    rows_trial = build_summary_rows({"Mark1": t1, "Mark2": t2}, cfg, is_trial=False)
    rz_trial = next(r for r in rows_trial if r["项目"] == "Rz")
    assert rz_trial["判定"] == "试测"

    nan_det = DetectionResult("Mark1", "upper", float("nan"), 10.0, float("nan"), 1.0, 20.0, 2.0, 0.0, 0.0, 64, 0.99, "Circle")
    ok_det = DetectionResult("Mark1", "lower", 10.0, 10.0, 1.0, 1.0, 20.0, 2.0, 0.0, 0.0, 64, 0.99, "Circle")
    ov_nan = calculate_overlay("Mark1", nan_det, ok_det, cfg)
    assert ov_nan.result == "Error"


def test_batch_all_error_runs_produce_error_overlay():
    job = {
        "config": MeasurementConfig(mode="Single Image", workflow_mode="Manual", recipe_validation_status="Validated"),
        "params": DetectionParams(),
        "marks": {"Mark1": MarkRecipe("Mark1"), "Mark2": MarkRecipe("Mark2")},
        "mark_images": {"Mark1": {"upper": None, "lower": None}, "Mark2": {"upper": None, "lower": None}},
        "batch": True,
        "batch_images": {
            "Mark1": {
                "upper": [ImageData("missing1", None, "missing1"), ImageData("missing2", None, "missing2")],
                "lower": [],
            },
            "Mark2": {"upper": [], "lower": []},
        },
        "selections": {"Mark1": {}, "Mark2": {}},
    }
    payload = run_measurement_job(job, lambda *_: None, lambda: False)
    assert "Mark1" in payload["overlays"]
    assert payload["overlays"]["Mark1"].result == "Error"


def test_anisotropic_ellipse_and_circle_and_geometry_metrics():
    cfg = MeasurementConfig(pixel_size_x_um=0.2, pixel_size_y_um=0.7)
    theta = np.deg2rad(30.0)
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    a, b = 50.0, 45.0
    xs = 150.0 + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    ys = 150.0 + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
    pts = np.column_stack((xs, ys))
    expected = contour_ellipse_metrics_um(pts, cfg)
    actual = ellipse_metrics_um({"major_px": 100.0, "minor_px": 90.0, "angle_deg": 30.0}, cfg)
    assert actual["ellipse_major_um"] == pytest.approx(expected["contour_ellipse_major_um"], rel=1e-3)
    assert actual["ellipse_minor_um"] == pytest.approx(expected["contour_ellipse_minor_um"], rel=1e-3)
    assert actual["ellipse_roundness_um"] == pytest.approx(expected["ellipse_roundness_um"], rel=1e-3)

    # Anisotropic caliper circle: true physical circle R=10 um with sx=0.2, sy=0.1 -> 50x100 px ellipse.
    cfg_c = MeasurementConfig(pixel_size_x_um=0.2, pixel_size_y_um=0.1)
    yy, xx = np.mgrid[0:300, 0:300].astype(np.float32)
    dist_um = np.hypot((xx - 150.0) * 0.2, (yy - 150.0) * 0.1)
    img = np.where(dist_um <= 10.0, 40.0, 210.0).astype(np.float32)
    roi = Roi(40, 40, 220, 220, "Caliper Circle", inner_ratio=0.35, caliper_count=48)
    det = detect_manual_roi("Mark1", "upper", ImageData("c", img, "c"), roi, DetectionParams(residual_limit_px=2.0), cfg_c)
    assert det.center_x_px == pytest.approx(150.0, abs=0.8)
    assert det.center_y_px == pytest.approx(150.0, abs=0.8)
    assert det.diameter_um == pytest.approx(20.0, abs=0.4)

    # Geometry projection & 3-point circle in physical coords.
    program = GeometryProgram(
        features=[
            GeometryFeatureDefinition("F1", "P", "upper", "point", points_px=[(0.0, 100.0)]),
            GeometryFeatureDefinition("F2", "L", "upper", "line", points_px=[(0.0, 0.0), (100.0, 100.0)]),
            GeometryFeatureDefinition("F3", "H", "upper", "projection", reference_ids=["F1", "F2"]),
            GeometryFeatureDefinition("F4", "C", "upper", "circle", points_px=[(50.0, 0.0), (-50.0, 0.0), (0.0, 100.0)]),
        ],
        measurements=[
            GeometryMeasurementDefinition("M1", "D_PH", "upper", "point_distance", ["F1", "F3"]),
            GeometryMeasurementDefinition("M2", "D_PL", "upper", "point_line_distance", ["F1", "F2"]),
            GeometryMeasurementDefinition("M3", "Dia", "upper", "diameter", ["F4"]),
        ],
    )
    gres = execute_geometry_program(program, {}, cfg_c)
    assert gres.measurements["M1"].value == pytest.approx(gres.measurements["M2"].value, rel=1e-4)
    assert gres.features["F4"].center_px[0] == pytest.approx(0.0, abs=1e-3)
    assert gres.features["F4"].center_px[1] == pytest.approx(0.0, abs=1e-3)
    assert gres.measurements["M3"].value == pytest.approx(20.0, abs=1e-3)


def test_rotated_rectangle_auto_detect_and_refine():
    img = np.full((300, 300), 220.0, dtype=np.float32)
    box = cv2.boxPoints(((150.0, 150.0), (100.0, 40.0), 30.0))
    cv2.fillConvexPoly(img, np.rint(box).astype(np.int32), 30.0)
    params = DetectionParams(gaussian_sigma_px=1.0, min_gradient=5.0)
    cfg = MeasurementConfig(pixel_size_x_um=0.1, pixel_size_y_um=0.1, recipe_validation_status="Validated")
    report = detect_auto_marks_with_report(img, "upper", params, 0.1, 0.1)
    rect_candidates = [c for c in report.results if c.fitting_mode == "AutoRectangle"]
    assert rect_candidates
    refined = refine_candidate(img, rect_candidates[0], params, cfg, roi_type="Caliper Circle")
    assert refined.fitting_mode == "ProductionRectangle"
    assert refined.shape_params.get("coverage", 0.0) > 0.9
    assert refined.shape_params.get("rejected_ratio", 1.0) < 0.05
    assert refined.center_x_px == pytest.approx(150.0, abs=1.0)
    assert refined.center_y_px == pytest.approx(150.0, abs=1.0)

    # Also verify semantic refine preserves shape_type.
    sem = refine_candidate(img, rect_candidates[0], params, cfg, roi_type="Ellipse")
    assert sem.shape_params.get("shape_type") in {"Circle", "Rectangle"}


def test_rotated_ellipse_bounds_and_region_center_corner_foreground():
    roi = Roi(80, 120, 140, 60, "Ellipse", angle_deg=90.0)
    x0, y0, x1, y1 = roi.to_int_bounds((300, 300))
    assert (x0, y0, x1, y1) == (120, 80, 180, 220)

    img = np.full((200, 200), 220.0, dtype=np.float32)
    img[80:120, 80:120] = 30.0
    img[48:58, 48:58] = 30.0  # overlaps ROI top-left corner (50, 50)
    reg_roi = Roi(50, 50, 100, 100, "Region Center")
    fit = detect_region_center(img, reg_roi, DetectionParams())
    assert fit.center_x_px == pytest.approx(100.0, abs=1.0)
    assert fit.center_y_px == pytest.approx(100.0, abs=1.0)

    # Degenerate collinear points rejected by fit_ellipse.
    collinear = np.column_stack((np.linspace(0, 50, 10), np.linspace(0, 50, 10)))
    with pytest.raises(ValueError):
        fit_ellipse(collinear)

    # Step fallback updates step for gradient & subpixel offset.
    edge_img = np.tile(np.where(np.arange(100) < 50, 20.0, 220.0).astype(np.float32), (100, 1))
    contour = np.column_stack((np.full(20, 50.0), np.linspace(30.0, 70.0, 20)))
    refined, grads = refine_contour_edges(
        edge_img, contour, DetectionParams(gaussian_sigma_px=1.0, profile_half_width_px=1.0, profile_step_px=0.6, min_gradient=5.0)
    )
    assert len(refined) > 10
    assert float(np.median(grads)) > 40.0


def test_recipe_import_blocks_tampered_and_handles_utf8_bom_sidecar(tmp_path):
    lib = RecipeLibrary(tmp_path / "lib", config_path=tmp_path / "cfg.json")
    src = tmp_path / "src.json"
    save_recipe(str(src), MeasurementConfig(recipe_name="R1", recipe_validation_status="Validated"), DetectionParams(), [])
    digest = seal_recipe(src)

    # UTF-8 BOM in .sha256 should still verify cleanly.
    sidecar = src.with_suffix(".json.sha256")
    sidecar.write_bytes(b"\xef\xbb\xbf" + (digest + "\n").encode("ascii"))
    assert verify_recipe(src) == ("Verified", digest)

    # Non-ASCII corrupt sidecar returns Mismatch instead of UnicodeDecodeError.
    sidecar.write_bytes(b"\xff\xfeinvalid")
    assert verify_recipe(src)[0] == "Mismatch"
    with pytest.raises(ValueError):
        lib.import_recipe(src)

    # Unknown future fields in recipe JSON are ignored safely.
    data = json.loads(src.read_text(encoding="utf-8"))
    data["measurement_config"]["future_unknown_field"] = 123
    data["detection_params"]["future_param"] = "x"
    src.write_text(json.dumps(data), encoding="utf-8")
    cfg_loaded, _, _ = load_recipe(str(src))
    assert cfg_loaded.recipe_name == "R1"


def test_close_window_during_running_thread_exits_quickly_and_batch_preview_survives_bad_txt(tmp_path):
    app = _app()
    window = MainWindow()
    try:
        # Bad .txt in batch_images should not raise during preview switching.
        bad_txt = tmp_path / "README.txt"
        bad_txt.write_text("not a numeric matrix", encoding="utf-8")
        good_png = tmp_path / "ok.png"
        cv2.imwrite(str(good_png), np.full((32, 32), 120, dtype=np.uint8))
        window._set_combo_value(window.measurement_run_mode_combo, "Batch")
        window.batch_images["Mark1"]["upper"] = [
            BatchImageRef.from_path(str(good_png)),
            BatchImageRef.from_path(str(bad_txt)),
        ]
        window._set_batch_preview_index(2)
        assert window.upper_image is None

        # Running QThread exits promptly on closeEvent without 5s deadlock.
        thread = QThread(window)
        thread.start()
        window._preview_thread = thread
        t0 = time.perf_counter()
        assert window.close()
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0
    finally:
        app.processEvents()


def test_image_loader_display_and_diagnostics_bridge_and_exporter_safety(tmp_path):
    # Default-metadata ImageData should not clip to solid white.
    arr = np.linspace(20.0, 200.0, 100, dtype=np.float32).reshape(10, 10)
    img_default = ImageData("", arr, "mem")
    u8 = display_to_uint8(img_default, enhanced=False)
    assert int(u8.min()) == 0
    assert int(u8.max()) == 255

    # Multi-channel float64/int16 TIFF loads cleanly as 2D float32.
    from PIL import Image
    rgb_path = tmp_path / "rgb.tif"
    Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8)).save(rgb_path)
    loaded = load_image(str(rgb_path))
    assert loaded.gray.shape == (16, 16)

    # Diagnostics bridge catches manual quality_status == Invalid and supports 16-caliper circles.
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    pts = [(100.0 + 40.0 * np.cos(a), 100.0 + 40.0 * np.sin(a)) for a in angles]
    det_a = DetectionResult(
        "Mark1", "upper", 100.0, 100.0, 10.0, 10.0, 80.0, 8.0, 0.05, 0.005, 16, 0.95,
        "CaliperCircle", edge_points=pts, shape_params={"quality_status": "Invalid", "roi_label": "ROI 1"},
    )
    det_b = DetectionResult(
        "Mark1", "lower", 100.0, 100.0, 10.0, 10.0, 80.0, 8.0, 0.05, 0.005, 16, 0.95,
        "CaliperCircle", edge_points=pts, shape_params={"quality_status": "Valid", "roi_label": "ROI 1"},
    )
    img_data = ImageData(str(rgb_path), np.full((200, 200), 120.0, dtype=np.float32), "rgb.tif", "uint8", 0.0, 255.0)
    app = _app()
    win = MainWindow()
    try:
        win._set_mode_ui("Dual Image")
        win.mark_images["Mark1"] = {"upper": img_data, "lower": img_data}
        win.roi_detections["Mark1"] = {"r1": det_a, "r2": det_b}
        snap = snapshot_main(win)
        result = align_measurement(snap, "r1", "r2")
        assert result.registered.metrics["coverage"] > 0.9
        assert result.reliable is False
    finally:
        win.close()
        app.processEvents()

    # Exporter preserves 6-decimal pixel size, neutralizes formula injection, and handles NaN visualization.
    cfg = MeasurementConfig(
        pixel_size_x_um=0.068512,
        pixel_size_y_um=0.068512,
        material_code="=CMD|' /C calc'!A0",
        recipe_validation_status="Validated",
    )
    det_nan = DetectionResult("Mark1", "upper", float("nan"), float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, "ProductionCircle")
    assert render_measurement_image(img_data, tmp_path / "vis.png", detections={"d": det_nan}) is True

    rows = build_detection_rows({"Mark1": {"r2": det_b}}, {}, cfg, upper_file="=1+1.tif")
    xlsx_path = tmp_path / "report.xlsx"
    export_results(
        str(xlsx_path),
        rows,
        config=cfg,
        summary_rows=[{"项目": "Rz", "Rz(μrad)": 999.0, "判定": "不通过", "提示": "Rz超限"}],
    )
    wb = openpyxl.load_workbook(xlsx_path)
    info_ws = wb["基础信息"]
    mat_val = next(r[1].value for r in info_ws.iter_rows(min_row=2) if r[0].value == "物料编码")
    assert str(mat_val).startswith("'")
    det_ws = wb["识别明细"]
    headers = [c.value for c in det_ws[1]]
    px_col = headers.index("像素尺寸X(μm/px)") + 1
    assert det_ws.cell(2, px_col).value == pytest.approx(0.068512, abs=1e-6)
    sum_ws = wb["结果汇总"]
    s_headers = [c.value for c in sum_ws[1]]
    verdict_cell = sum_ws.cell(2, s_headers.index("判定") + 1)
    assert verdict_cell.fill.fgColor.rgb.endswith("FFC7CE")

    # export_data_tables sanitizes sheet/csv names.
    export_data_tables(str(tmp_path / "csv_out"), {"../bad:name": [{"val": "=SUM(A1:A2)"}]})
    assert not (tmp_path / "bad_name.csv").exists()
