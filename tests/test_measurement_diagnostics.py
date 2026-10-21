import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from copy import deepcopy
from dataclasses import asdict
import json
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QPushButton
from overlay_measure.models import ImageData, DetectionResult, MarkRecipe, Roi
from overlay_measure.ui_main import MainWindow
from overlay_measure.diagnostics.measurement_bridge import (
    snapshot_main,
    align_measurement,
    export_measurement,
)
from overlay_measure.diagnostics.measurement_profile import measurement_profile


def detection(layer, center, radius, count=120):
    theta = np.linspace(0, 2 * np.pi, count, endpoint=False)
    points = np.array(center) + radius * np.column_stack([np.cos(theta), np.sin(theta)])
    return DetectionResult(
        "Mark1",
        layer,
        *center,
        center[0] * 0.1,
        center[1] * 0.1,
        2 * radius,
        2 * radius * 0.1,
        0.01,
        0.001,
        count,
        0.99,
        "Circle",
        edge_points=points.tolist(),
    )


@pytest.fixture
def main_pair():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._set_operation_mode("Engineering", authenticated=True)
    window._set_mode_ui("Dual Image")
    y, x = np.mgrid[:128, :136]
    for layer, center, radius in [
        ("upper", (64.0, 64.0), 34.0),
        ("lower", (72.0, 61.0), 34.4),
    ]:
        gray = (
            1000
            + 50000
            / (
                1
                + np.exp(
                    np.clip(
                        -(np.hypot(x - center[0], y - center[1]) - radius) / 0.9,
                        -60,
                        60,
                    )
                )
            )
        ).astype(np.float32)
        window._set_image_for_layer(
            "Mark1", layer, ImageData("", gray, layer + ".tif", "uint16"), "single"
        )
    window.marks["Mark1"] = MarkRecipe(
        "Mark1",
        Roi(20, 20, 88, 88, "Circle"),
        Roi(28, 17, 88, 88, "Circle"),
        reference_contour_id="upper-1",
        target_contour_id="lower-1",
    )
    window.roi_detections["Mark1"] = {
        "upper-1": detection("upper", (64.0, 64.0), 34.0, 80),
        "lower-1": detection("lower", (72.0, 61.0), 34.4, 140),
    }
    window._rebuild_legacy_manual_detections("Mark1")
    window._refresh_all_widgets()
    yield window, app
    dialog = getattr(window, "_soma_diagnostics_dialog", None)
    if dialog:
        dialog.close()
    window.close()
    app.processEvents()


def test_main_snapshot_reuses_original_points_and_is_read_only(main_pair, monkeypatch):
    window, app = main_pair
    before_params = deepcopy(asdict(window.params))
    before_marks = deepcopy(window.marks)
    before = deepcopy(window.roi_detections)

    def forbidden(*args, **kwargs):
        raise AssertionError("Must not re-detect already measured edges")

    monkeypatch.setattr(
        "overlay_measure.caliper_circle_detector.detect_caliper_circle", forbidden
    )
    monkeypatch.setattr(
        "overlay_measure.measurement_service.detect_manual_roi", forbidden
    )
    snap = snapshot_main(window)
    result = align_measurement(snap, "upper-1", "lower-1")
    np.testing.assert_array_equal(
        result.a.edge_points, before["Mark1"]["upper-1"].edge_points
    )
    assert result.registered.metrics["mean_um"] == pytest.approx(0.04, abs=1e-12)
    assert result.registered.metrics["coverage"] == 1.0
    assert (
        result.circle_metrics["diameter_a_um"] == before["Mark1"]["upper-1"].diameter_um
    )
    assert result.circle_metrics["center_translation_removed"]
    assert asdict(window.params) == before_params and window.marks == before_marks
    assert window.roi_detections == before
    assert not snap.session.a.original.gray.flags.writeable
    window.mark_images["Mark1"]["upper"].gray[0, 0] = 123
    assert snap.session.a.original.gray[0, 0] != 123


def test_one_click_from_main_large_view_and_three_tabs(main_pair):
    window, app = main_pair
    window.show()
    app.processEvents()
    assert window.soma_diagnostics_btn.isVisible()
    window.soma_diagnostics_btn.click()
    app.processEvents()
    dialog = window._soma_diagnostics_dialog
    assert dialog.session.a is not None and dialog.session.b is not None
    assert dialog.align_button.isEnabled()
    assert not dialog.pairs.isVisible()
    text = [b.text() for b in dialog.findChildren(QPushButton)]
    assert not any(
        "导入" in name or "开始分析" in name or "基准点" in name for name in text
    )
    dialog.align_button.click()
    app.processEvents()
    assert dialog.session.result is not None
    assert dialog.overlay.height() > dialog.height() * 0.4
    for mode in range(3):
        dialog.set_mode(mode)
        app.processEvents()
        assert dialog.views.checkedId() == mode
    assert dialog.profile is not None
    assert "主页面边缘位置" in dialog.selected_label.text()
    assert not hasattr(dialog, "source_a")


def test_same_physical_profile_axis_and_main_edge_markers(main_pair):
    window, _ = main_pair
    snap = snapshot_main(window)
    r = align_measurement(snap, "upper-1", "lower-1")
    cache = {}
    profile = measurement_profile(snap, 0, cache)
    assert profile["a"]["selected_um"] == 0.0
    assert profile["b"]["selected_um"] == pytest.approx(0.04, abs=1e-12)
    axis_a = profile["a"]["sampling_xy_native"]
    axis_b = profile["b"]["sampling_xy_native"]
    np.testing.assert_allclose(
        axis_b - axis_a, np.tile([8.0, -3.0], (len(axis_a), 1)), atol=1e-12
    )
    assert profile["a"]["raw"].max() > 40000
    assert profile["b"]["gradient"].max() > 0
    measurement_profile(snap, 30, cache)
    assert len(cache) == 2


def test_missing_arc_not_filled_with_different_point_counts(main_pair):
    window, _ = main_pair
    d = window.roi_detections["Mark1"]["lower-1"]
    points = np.asarray(d.edge_points)
    v = points - [72, 61]
    angles = np.mod(np.degrees(np.arctan2(v[:, 1], v[:, 0])), 360)
    d.edge_points = points[(angles < 90) | (angles > 130)].tolist()
    result = align_measurement(snapshot_main(window), "upper-1", "lower-1")
    assert not result.registered.valid[110]
    assert 0.8 < result.registered.metrics["coverage"] < 0.91
    with pytest.raises(ValueError, match="缺少有效对应"):
        measurement_profile(snapshot_with_result(window), 110, {})


def snapshot_with_result(window):
    snap = snapshot_main(window)
    align_measurement(snap, "upper-1", "lower-1")
    return snap


def test_snapshot_sync_tracks_main_without_writing_back(main_pair):
    window, app = main_pair
    window.start_soma_diagnostics()
    dialog = window._soma_diagnostics_dialog
    dialog.auto_align()
    old = dialog.session
    window.roi_detections["Mark1"]["lower-1"].diameter_um = 7.1
    dialog.sync_main()
    assert dialog.session is not old and dialog.session.result is None
    dialog.auto_align()
    assert dialog.session.result.circle_metrics["diameter_b_um"] == 7.1
    assert window.roi_detections["Mark1"]["lower-1"].diameter_um == 7.1


def test_quality_failed_contour_can_be_inspected_but_not_validated(main_pair):
    window, _ = main_pair
    window.roi_detections["Mark1"]["lower-1"].shape_params[
        "measurement_eligible"
    ] = False
    r = align_measurement(snapshot_main(window), "upper-1", "lower-1")
    assert not r.reliable and any("质量门槛" in s for s in r.warnings)
    assert not any("有效对应不足" in s for s in r.warnings)


def test_export_preserves_main_points_and_roi_trace(main_pair, tmp_path):
    window, _ = main_pair
    snap = snapshot_with_result(window)
    out = export_measurement(snap, "upper-1", "lower-1", tmp_path / "result")
    data = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert data["source"] == "main_page_read_only_snapshot"
    assert (
        data["upper_detection"]["edge_points"]
        == window.roi_detections["Mark1"]["upper-1"].edge_points
    )
    assert data["main_page_rois"]["upper"][0]["roi"]["roi_type"] == "Circle"
    assert len(data["upper"]["gray_sha256"]) == 64
    assert (out / "analysis.xlsx").exists() and (out / "edge_displacement.png").exists()


def test_auto_candidate_keys_shared_across_layers_are_not_lost(main_pair, monkeypatch):
    window, _ = main_pair
    items = window.roi_detections["Mark1"]
    monkeypatch.setattr(window, "_is_auto_workflow", lambda: True)
    monkeypatch.setattr(
        window,
        "_current_auto_detections",
        lambda: {"same-key": {"upper": items["upper-1"], "lower": items["lower-1"]}},
    )
    snap = snapshot_main(window)
    assert "same-key" in snap.upper and "same-key" in snap.lower


def test_known_quarter_turn_uses_coordinates_not_warped_edges(main_pair):
    from overlay_measure.diagnostics.models import clockwise_matrix, transform_points

    window, _ = main_pair
    a = window.roi_detections["Mark1"]["upper-1"]
    b = deepcopy(a)
    b.layer = "lower"
    shape = window.mark_images["Mark1"]["upper"].gray.shape
    b.edge_points = transform_points(a.edge_points, clockwise_matrix(shape, 3)).tolist()
    b.center_x_px, b.center_y_px = transform_points(
        [[a.center_x_px, a.center_y_px]], clockwise_matrix(shape, 3)
    )[0]
    old = deepcopy(b.edge_points)
    window.roi_detections["Mark1"]["lower-1"] = b
    original = window.mark_images["Mark1"]["lower"].gray.copy()
    r = align_measurement(snapshot_main(window), "upper-1", "lower-1", turns=1)
    assert r.registered.metrics["rms_um"] < 1e-12
    np.testing.assert_array_equal(b.edge_points, old)
    np.testing.assert_array_equal(window.mark_images["Mark1"]["lower"].gray, original)


def test_multiple_contours_default_to_main_selection(main_pair):
    window, app = main_pair
    window.roi_detections["Mark1"]["upper-extra"] = detection(
        "upper", (40.0, 40.0), 20.0
    )
    window.roi_detections["Mark1"]["lower-extra"] = detection(
        "lower", (50.0, 41.0), 20.1
    )
    for combo, key in [
        (window.auto_reference_combo, "upper-extra"),
        (window.auto_target_combo, "lower-extra"),
    ]:
        combo.blockSignals(True)
        combo.addItem(key, key)
        combo.setCurrentIndex(combo.findData(key))
        combo.blockSignals(False)
    window.start_soma_diagnostics()
    dialog = window._soma_diagnostics_dialog
    app.processEvents()
    assert dialog.pairs.isVisible()
    assert dialog.upper_choice.currentData() == "upper-extra"
    assert dialog.lower_choice.currentData() == "lower-extra"


def test_sync_uses_current_batch_pair_and_results(main_pair, monkeypatch):
    window, _ = main_pair
    images = window.mark_images["Mark1"]
    window.batch_images["Mark1"] = {
        "upper": [deepcopy(images["upper"]), deepcopy(images["upper"])],
        "lower": [deepcopy(images["lower"]), deepcopy(images["lower"])],
    }
    window.batch_images["Mark1"]["upper"][1].display_name = "batch-upper-2.tif"
    measured = deepcopy(window.roi_detections["Mark1"])
    measured["lower-1"].diameter_um = 7.5
    window.batch_run_records["Mark1"] = [
        {"workflow": "Manual", "detections": {}},
        {"workflow": "Manual", "detections": measured},
    ]
    window._batch_detail_last_single_index = 2
    monkeypatch.setattr(window, "_is_batch_mode", lambda: True)
    snapshot = snapshot_main(window)
    assert snapshot.session.a.original.display_name == "batch-upper-2.tif"
    assert snapshot.lower["lower-1"].diameter_um == 7.5
    assert "第 2 组" in snapshot.label


def test_compacted_main_contours_are_not_misrepresented_as_full_precision(main_pair):
    window, _ = main_pair
    window.roi_detections["Mark1"]["lower-1"].edge_point_count = 1000
    result = align_measurement(snapshot_main(window), "upper-1", "lower-1")
    assert not result.reliable
    assert any("抽样轮廓" in s for s in result.warnings)
    assert not any("有效对应不足" in s for s in result.warnings)
