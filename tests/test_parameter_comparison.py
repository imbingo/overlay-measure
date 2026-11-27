from __future__ import annotations

import os
import time
from dataclasses import asdict, replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from overlay_measure.caliper_circle_detector import detect_caliper_circle
from overlay_measure.circle_ellipse_fitter import fit_circle_ransac
from overlay_measure.models import ImageData, Roi
from overlay_measure.parameter_comparison import ComparisonSource, calculate_preview, point_explanation
from overlay_measure.ui_main import MainWindow
from overlay_measure.ui_parameter_comparison import ParameterComparisonDialog


@pytest.fixture
def main():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._set_operation_mode("Engineering", authenticated=True)
    y, x = np.mgrid[:180, :180]
    radii = np.hypot(x - 90.2, y - 88.7)
    gray = (20 + 180 / (1 + np.exp(np.clip(-(radii - 45) / 0.9, -60, 60)))
            + np.random.default_rng(31).normal(0, 1, radii.shape)).astype(np.float32)
    image = ImageData("synthetic", gray, "synthetic.png", "float32", float(gray.min()), float(gray.max()))
    window._set_image_for_layer("Mark1", "upper", image, "single")
    entry = window.marks["Mark1"].add_roi("upper", Roi(25, 25, 130, 130, "Caliper Circle", 0.5, "Strongest Edge"))
    window._refresh_roi_index_combo(entry.roi_id)
    window._sync_current_mark_images()
    yield app, window, entry
    dialog = getattr(window, "_parameter_comparison_dialog", None)
    if dialog is not None:
        dialog.close()
    window.close()
    app.processEvents()


def settle(app, dialog):
    deadline = time.monotonic() + 15
    while dialog.worker is not None or dialog._pending is not None:
        assert time.monotonic() < deadline, "preview did not finish"
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()


def source(main):
    _, window, entry = main
    return ComparisonSource.capture(window, "Mark1", "upper", entry.roi_id)


def test_preview_matches_production_detector_exactly(main):
    captured = source(main)
    plain = detect_caliper_circle(captured.image.gray, captured.roi, captured.params)
    preview = calculate_preview(captured, captured.params)
    assert preview.error == ""
    for field in ("center_x_px", "center_y_px", "radius_px", "residual_px", "confidence", "angular_coverage"):
        assert getattr(preview.circle, field) == getattr(plain, field)
    np.testing.assert_array_equal(preview.circle.edge_points, plain.edge_points)
    np.testing.assert_array_equal(preview.circle.rejected_points, plain.rejected_points)
    assert sum(r["status"] == "valid" for r in preview.records) == preview.valid_count
    assert all(r.get("selected_distance_px") is not None for r in preview.records if r.get("selected"))


def test_ransac_trace_explains_the_actual_consensus_mask():
    angle = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    points = np.column_stack((np.cos(angle), np.sin(angle))) * 30
    points[::8] *= 1.1
    trace = {}
    plain = fit_circle_ransac(points, 0.4)
    observed = fit_circle_ransac(points, 0.4, diagnostics=trace)
    assert not trace["fallback"]
    np.testing.assert_array_equal(observed[-1], trace["errors_px"] < trace["threshold_px"])
    np.testing.assert_array_equal(observed[-1], plain[-1])
    assert observed[:4] == plain[:4]
    assert np.count_nonzero(~observed[-1]) == 8


def test_weak_peaks_are_missing_not_fit_rejections_and_canny_is_unused(main):
    captured = source(main)
    failed = calculate_preview(captured, replace(captured.params, min_gradient=1000000))
    assert failed.circle is None and failed.error
    assert len(failed.records) == 64
    assert all(r["status"] == "below_gradient_threshold" for r in failed.records)
    assert "梯度" in point_explanation(failed, 0)
    a = calculate_preview(captured, captured.params)
    b = calculate_preview(captured, replace(captured.params, canny_low=250, canny_high=255))
    np.testing.assert_array_equal(a.circle.edge_points, b.circle.edge_points)


def test_disabled_ransac_and_cancelled_preview_are_explained(main):
    captured = source(main)
    result = calculate_preview(captured, replace(captured.params, use_ransac=False))
    assert all(not r["ransac_used"] for r in result.records if r.get("selected"))
    assert "RANSAC 已关闭" in point_explanation(result, 0)
    with pytest.raises(InterruptedError):
        calculate_preview(captured, captured.params, cancelled=lambda: True)


def test_source_is_frozen_and_stale_roi_or_batch_cannot_be_applied(main):
    _, window, entry = main
    captured = source(main)
    assert captured.is_current(window)
    assert not captured.image.gray.flags.writeable
    assert not np.shares_memory(captured.image.gray, captured.image_identity.gray)
    entry.roi.x += 1
    assert not captured.is_current(window)
    entry.roi.x -= 1
    window.params.min_gradient += 1
    assert not captured.is_current(window)


def test_batch_snapshot_is_current_preview_only_and_detects_index_change(main):
    _, window, _ = main
    first = window.mark_images["Mark1"]["upper"]
    second = replace(first, path="second", display_name="second.png", gray=first.gray.copy())
    window._set_combo_value(window.measurement_run_mode_combo, "Batch")
    window.batch_images["Mark1"]["upper"] = [first, second]
    captured = source(main)
    assert captured.batch_index == 1
    assert captured.image_identity is first
    assert captured.is_current(window)
    window._batch_detail_last_single_index = 2
    assert not captured.is_current(window)


def test_edit_preview_rebase_and_restore_do_not_change_main_or_zoom(main):
    app, window, _ = main
    original_params = asdict(window.params)
    window.detections["sentinel"] = {"original": True}
    dialog = ParameterComparisonDialog(window)
    try:
        dialog.show()
        settle(app, dialog)
        original = dialog.baseline
        dialog.canvas.user_zoom, dialog.canvas.pan_x = 4.0, 23.0
        dialog.controls["gaussian_sigma_px"].setValue(1.5)
        assert not dialog.apply_button.isEnabled()
        dialog.request_preview()
        settle(app, dialog)
        assert dialog.baseline is original
        assert dialog.current.params.gaussian_sigma_px == 1.5
        assert dialog.canvas.user_zoom == 4.0 and dialog.canvas.pan_x == 23.0
        assert asdict(window.params) == original_params
        assert "sentinel" in window.detections
        dialog.rebase()
        assert dialog.baseline is dialog.current
        dialog.controls["gaussian_sigma_px"].setValue(2)
        dialog.restore_baseline()
        settle(app, dialog)
        assert dialog.current.params.gaussian_sigma_px == 1.5
        assert asdict(window.params) == original_params
    finally:
        dialog.close()


def test_apply_only_tuned_fields_invalidates_results_and_blocks_stale(main):
    app, window, entry = main
    original = asdict(window.params)
    dialog = ParameterComparisonDialog(window)
    try:
        settle(app, dialog)
        dialog.controls["gaussian_sigma_px"].setValue(1.75)
        dialog.request_preview()
        settle(app, dialog)
        window.detections["old"] = {}
        assert dialog.apply_parameters()
        assert window.params.gaussian_sigma_px == window.sigma_spin.value() == 1.75
        assert not window.detections
        for name, value in original.items():
            if name != "gaussian_sigma_px":
                assert getattr(window.params, name) == value
        entry.roi.x += 2
        dialog.check_freshness()
        assert not dialog.apply_button.isEnabled()
        assert not dialog.apply_parameters()
    finally:
        dialog.close()


def test_repeated_preview_requests_keep_only_latest_and_close_is_safe(main):
    app, window, _ = main
    dialog = ParameterComparisonDialog(window)
    for value in (1.2, 1.5, 2.0):
        dialog.controls["gaussian_sigma_px"].setValue(value)
        dialog.request_preview()
    settle(app, dialog)
    assert dialog.current.params.gaussian_sigma_px == 2.0
    assert dialog.baseline.params.gaussian_sigma_px == 1.0
    dialog.request_preview()
    worker = dialog.worker
    dialog.close()
    assert worker is None or not worker.isRunning()
    app.processEvents()


def test_escape_or_reject_stops_preview_thread(main):
    app, window, _ = main
    dialog = ParameterComparisonDialog(window)
    dialog.show()
    worker = dialog.worker
    dialog.reject()
    assert dialog._closing
    assert worker is None or not worker.isRunning()
    app.processEvents()


def test_entry_permission_and_main_close_stop_dialog(main, monkeypatch):
    app, window, _ = main
    assert window.parameter_comparison_action.isEnabled()
    window.start_parameter_comparison()
    dialog = window._parameter_comparison_dialog
    assert dialog is not None
    assert window.close()
    assert dialog._closing
    app.processEvents()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_side_by_side_views_share_actual_image_scale(main):
    app, window, _ = main
    dialog = ParameterComparisonDialog(window)
    try:
        dialog.show()
        settle(app, dialog)
        dialog.canvas.zoom_by(2)
        dialog.canvas._update_transform()
        old_scale = dialog.canvas.scale
        dialog.view_mode.setCurrentIndex(1)
        app.processEvents()
        dialog.canvas._update_transform()
        dialog.second_canvas._update_transform()
        assert dialog.canvas.scale == pytest.approx(old_scale)
        dialog.canvas.pan_x = 17
        dialog.canvas.viewChanged.emit()
        assert dialog.second_canvas.pan_x == 17
    finally:
        dialog.close()


def _rect_and_line_image():
    y, x = np.mgrid[:180, :180]
    rect = ((x >= 45) & (x <= 135) & (y >= 50) & (y <= 130)).astype(np.float32)
    gray = 30.0 + 170.0 * rect + np.random.default_rng(7).normal(0, 0.8, rect.shape).astype(np.float32)
    return ImageData("rect", gray, "rect.png", "float32", float(gray.min()), float(gray.max()))


def test_all_manual_roi_types_and_single_image_lower_layer_zero_deviation(main):
    _, window, _ = main
    from overlay_measure.measurement_service import detect_manual_roi
    from overlay_measure.parameter_comparison import available_rois

    window._set_mode_ui("Single Image")
    window.mark_images["Mark1"]["lower"] = None
    circle_rois = [
        ("Annulus", Roi(25, 25, 130, 130, "Annulus", 0.5, "Strongest Edge")),
        ("Circle", Roi(25, 25, 130, 130, "Circle", 0.0, "All Edges")),
        ("Ellipse", Roi(25, 25, 130, 130, "Ellipse", 0.0, "All Edges")),
        ("Robust Center", Roi(25, 25, 130, 130, "Robust Center", 0.0, "All Edges")),
        ("Region Center", Roi(25, 25, 130, 130, "Region Center", 0.0, "All Edges")),
    ]
    created = []
    for name, roi in circle_rois:
        layer = "lower" if name in {"Annulus", "Circle"} else "upper"
        entry = window.marks["Mark1"].add_roi(layer, roi)
        created.append((layer, entry, name))

    choices = available_rois(window)
    choice_ids = {(layer, rid) for _, layer, rid, _ in choices}
    for layer, entry, _ in created:
        assert (layer, entry.roi_id) in choice_ids
        captured = ComparisonSource.capture(window, "Mark1", layer, entry.roi_id)
        assert captured.is_current(window)
        formal = detect_manual_roi("Mark1", layer, captured.image, captured.roi, captured.params, captured.config)
        preview = calculate_preview(captured, captured.params)
        assert preview.ok and preview.error == ""
        assert preview.detection.center_x_px == formal.center_x_px
        assert preview.detection.center_y_px == formal.center_y_px
        assert preview.diameter_um == formal.diameter_um
        assert preview.residual_um == formal.residual_um
        assert preview.valid_count == formal.edge_point_count
        assert len(preview.records) > 0

    rect_img = _rect_and_line_image()
    window._set_image_for_layer("Mark1", "upper", rect_img, "single")
    for roi in (
        Roi(30, 35, 120, 110, "Rectangle", 0.0, "All Edges"),
        Roi(30, 35, 120, 110, "Rectangular Ring", 0.45, "All Edges"),
        Roi(55, 38, 70, 24, "Approximate Line", 0.0, "All Edges"),
    ):
        entry = window.marks["Mark1"].add_roi("upper", roi)
        captured = ComparisonSource.capture(window, "Mark1", "upper", entry.roi_id)
        formal = detect_manual_roi("Mark1", "upper", captured.image, captured.roi, captured.params, captured.config)
        preview = calculate_preview(captured, captured.params)
        assert preview.ok and preview.error == ""
        assert preview.detection.center_x_px == formal.center_x_px
        assert preview.detection.center_y_px == formal.center_y_px
        assert preview.diameter_um == formal.diameter_um
        assert preview.residual_um == formal.residual_um
        assert preview.valid_count == formal.edge_point_count
        assert len(preview.records) > 0


def test_non_caliper_canny_profile_controls_point_click_and_full_apply(main):
    app, window, _ = main
    ann_entry = window.marks["Mark1"].add_roi(
        "upper", Roi(25, 25, 130, 130, "Annulus", 0.5, "Strongest Edge")
    )
    window._refresh_roi_index_combo(ann_entry.roi_id)
    dialog = ParameterComparisonDialog(window)
    try:
        dialog.show()
        settle(app, dialog)
        assert dialog.source.roi_type == "Annulus"
        assert dialog.controls["canny_low"].isEnabled()
        assert dialog.controls["canny_high"].isEnabled()
        assert dialog.controls["profile_half_width_px"].isEnabled()
        assert dialog.controls["profile_step_px"].isEnabled()
        assert not dialog.controls["polarity"].isEnabled()

        dialog.controls["canny_low"].setValue(180)
        assert dialog.controls["canny_high"].minimum() == 181
        assert dialog.controls["canny_high"].value() >= 181
        dialog.controls["canny_low"].setValue(40)
        dialog.controls["canny_high"].setValue(120)
        dialog.controls["profile_half_width_px"].setValue(2.5)
        dialog.controls["profile_step_px"].setValue(0.2)
        dialog.controls["min_gradient"].setValue(12.0)
        dialog.polarity.setCurrentIndex(dialog.polarity.findData("Dark to Bright"))
        dialog.request_preview()
        settle(app, dialog)
        assert dialog.current.ok
        assert dialog.current.records

        rec = dialog.current.records[len(dialog.current.records) // 2]
        sel = rec.get("selected") or {"x": float(rec["start"][0]), "y": float(rec["start"][1])}
        dialog.canvas.pointPicked.emit(sel["x"], sel["y"])
        app.processEvents()
        assert "采样剖面" in dialog.selected_label.text()
        assert len(dialog.plot.raw) > 0
        assert "梯度" in point_explanation(dialog.current, dialog.canvas.selected) or "拟合" in point_explanation(dialog.current, dialog.canvas.selected)

        window.detections["stale_check"] = {}
        assert dialog.apply_parameters()
        assert window.params.canny_low == window.canny_low_spin.value() == 40
        assert window.params.canny_high == window.canny_high_spin.value() == 120
        assert window.params.profile_half_width_px == pytest.approx(2.5)
        assert window.profile_half_spin.value() == pytest.approx(2.5)
        assert window.params.profile_step_px == pytest.approx(0.2)
        assert window.profile_step_spin.value() == pytest.approx(0.2)
        assert window.params.min_gradient == pytest.approx(12.0)
        assert window.params.polarity == "Dark to Bright"
        assert window._combo_value(window.polarity_combo) == "Dark to Bright"
        assert not window.detections
    finally:
        dialog.close()

