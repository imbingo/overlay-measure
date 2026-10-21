from __future__ import annotations
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication
from overlay_measure.models import DetectionParams, ImageData, MeasurementConfig, Roi
from overlay_measure.image_loader import load_image
from overlay_measure.caliper_circle_detector import detect_caliper_circle
from overlay_measure.diagnostics.models import (
    Calibration,
    DiagnosticImage,
    DiagnosticsSession,
    Registration,
    clockwise_matrix,
    transform_points,
)
from overlay_measure.diagnostics.registration import (
    rigid_from_points,
    translation_from_region,
)
from overlay_measure.diagnostics.diagnostics_service import analyze, center_align
from overlay_measure.diagnostics.edge_comparison import compare_edges
from overlay_measure.diagnostics.profile_comparison import compare_profiles, edge_width
from overlay_measure.diagnostics.export import export_session


def circle_image(radius=35.0, center=(62.3, 63.7), shape=(128, 130)):
    y, x = np.mgrid[: shape[0], : shape[1]]
    r = np.hypot(x - center[0], y - center[1])
    return (1000 + 50000 / (1 + np.exp(np.clip(-(r - radius) / 0.85, -60, 60)))).astype(
        np.float32
    )


def session(a=None, b=None):
    if a is None:
        a = circle_image()
    if b is None:
        b = a.copy()
    return DiagnosticsSession(
        a=DiagnosticImage(ImageData("", a, "A", "uint16")),
        b=DiagnosticImage(ImageData("", b, "B", "uint16")),
        params=DetectionParams(
            gaussian_sigma_px=0.7,
            profile_step_px=0.5,
            min_gradient=20,
            use_ransac=False,
        ),
        roi=Roi(
            18,
            19,
            90,
            90,
            "Caliper Circle",
            0.6,
            "Strongest Edge",
            caliper_count=40,
            caliper_width_px=3,
        ),
        registration=Registration(
            method="manual_rigid", features="test independent fiducials", reliable=True
        ),
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("suffix", [".tif", ".png"])
def test_gray_fidelity_and_unicode(tmp_path, dtype, suffix):
    info = np.iinfo(dtype)
    data = np.array([[0, 1, info.max], [info.max - 1, 17, 42]], dtype)
    path = tmp_path / ("中文圆孔" + suffix)
    ok, encoded = cv2.imencode(suffix, data)
    assert ok
    encoded.tofile(str(path))
    image = DiagnosticImage.load(path)
    assert image.original.source_dtype == str(data.dtype)
    np.testing.assert_array_equal(image.original.gray, data.astype(np.float32))
    assert image.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not image.original.gray.flags.writeable


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_quarter_turn_pixel_centers_calibration_and_original_unchanged(turns):
    data = np.arange(35, dtype=np.float32).reshape(5, 7)
    backup = data.copy()
    slot = DiagnosticImage(ImageData("", data, "test"), Calibration(0.2, 0.7), turns)
    oriented = slot.oriented.gray
    np.testing.assert_array_equal(oriented, np.rot90(data, -turns))
    np.testing.assert_array_equal(data, backup)
    assert oriented.flags.c_contiguous
    for y in range(5):
        for x in range(7):
            xx, yy = transform_points([[x, y]], slot.native_to_oriented)[0].astype(int)
            assert oriented[yy, xx] == data[y, x]
    assert slot.oriented_calibration == (
        Calibration(0.7, 0.2) if turns % 2 else Calibration(0.2, 0.7)
    )


def test_known_rigid_transform_anisotropic_points():
    a = np.array([[4.0, 5.0], [16.0, 5.0], [5.0, 24.0], [20.0, 28.0]])
    theta = 0.13
    expected = np.array(
        [
            [np.cos(theta), -np.sin(theta), 2.2],
            [np.sin(theta), np.cos(theta), -1.4],
            [0, 0, 1.0],
        ]
    )
    ca = Calibration(0.2, 0.7)
    cb = Calibration(0.4, 0.3)
    b_um = transform_points(ca.physical(a), np.linalg.inv(expected))
    b = b_um / [cb.x, cb.y]
    reg = rigid_from_points(a, b, ca, cb, max_residual_um=0.01)
    assert reg.reliable
    assert reg.residual_um < 1e-12
    np.testing.assert_allclose(reg.matrix, expected, atol=1e-13)
    bad = b.copy()
    bad[0] += [7, 4]
    assert not rigid_from_points(a, bad, ca, cb, max_residual_um=0.01).reliable
    with pytest.raises(ValueError):
        rigid_from_points(a[:2], b[:2], ca, cb)
    with pytest.raises(ValueError):
        rigid_from_points([[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]], ca, cb)


def test_known_translation_on_independent_features():
    a = np.zeros((140, 160), np.float32)
    rng = np.random.default_rng(5)
    a[25:105, 30:130] = rng.uniform(0, 1000, (80, 100))
    b = cv2.warpAffine(a, np.array([[1, 0, 3.2], [0, 1, -2.1]], float), (160, 140))
    reg = translation_from_region(
        a, b, Calibration(0.2, 0.3), Calibration(0.2, 0.3), (15, 10, 135, 120)
    )
    assert reg.reliable
    np.testing.assert_allclose(reg.matrix[:2, 2], [-0.64, 0.63], atol=0.08)
    with pytest.raises(ValueError):
        translation_from_region(
            a * 0, b * 0, Calibration(), Calibration(), (0, 0, 100, 100)
        )


def test_collector_does_not_change_production_results():
    from tools.capture_diagnostics_baseline import inputs

    gray, roi, params, config = inputs()
    plain = detect_caliper_circle(gray, roi, params)
    traces = []
    traced = detect_caliper_circle(gray, roi, params, diagnostics=traces)
    assert len(traces) == roi.caliper_count
    for key, value in asdict(plain).items():
        actual = getattr(traced, key)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, actual)
        else:
            assert actual == value
    assert sum(t["inlier"] for t in traces) == len(plain.edge_points)
    for trace in traces:
        assert (
            len(trace["profile"])
            == len(trace["gradient"])
            == len(trace["radial_samples"])
        )
        assert trace["selected"] is not None


def test_frozen_v248_measurement_baseline_exact():
    from tools.capture_diagnostics_baseline import capture, serialize

    baseline = json.loads(
        (Path(__file__).parent / "data/diagnostics_v248_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    current = json.loads(json.dumps(capture(), default=serialize))
    # Wall clock export timestamps are intentionally excluded; no numeric field is normalized.
    for snapshot in (current, baseline):
        for row in snapshot["export_rows"]:
            row.pop("timestamp")
    assert (
        current == baseline
    ), "Original v2.4.8 measurements/points/roundness/overlay/export changed"


def test_identical_ab_equals_production_same_input():
    s = session()
    result = analyze(s)
    plain = detect_caliper_circle(s.a.original.gray, s.roi, s.params)
    np.testing.assert_array_equal(result.a.edge_points, plain.edge_points)
    np.testing.assert_allclose(result.b.edge_points, plain.edge_points, atol=1e-12)
    assert result.registered.metrics["rms_um"] < 1e-12
    assert result.reliable


def test_known_radial_shift_and_profile_peak():
    s = session(b=circle_image(35.5))
    r = analyze(s)
    assert r.registered.metrics["mean_um"] == pytest.approx(0.05, abs=0.002)
    assert r.circle_metrics["diameter_delta_um"] == pytest.approx(0.1, abs=0.004)
    data = compare_profiles(s, 135)
    assert data["local_shift_um"] == pytest.approx(0.05, abs=0.005)
    assert data["a"]["selected_gradient_dn_per_um"] > 0
    assert data["b"]["inlier"]


def test_translated_sampling_uses_native_gray_not_warped_detection():
    a = circle_image()
    b = circle_image(center=(67.3, 60.7), shape=(140, 150))
    s = session(a, b)
    s.registration.matrix[:2, 2] = [-0.5, 0.3]
    r = analyze(s)
    assert r.registered.metrics["rms_um"] < 1e-8
    np.testing.assert_array_equal(s.a.original.gray, a)
    np.testing.assert_array_equal(s.b.original.gray, b)
    np.testing.assert_allclose(r.sampling_b[:2, 2], [5, -3])


def test_center_alignment_removes_translation_with_warning():
    s = session(b=circle_image(center=(65.3, 61.7)))
    s.seed_b_roi = deepcopy(s.roi)
    s.registration = center_align(s)
    assert "消除整体平移" in s.registration.warning
    result = analyze(s)
    assert result.circle_metrics["center_translation_removed"]
    assert result.registered.metrics["rms_um"] < 1e-5


def circle_points(count, radius=10, center=(0, 0)):
    t = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.array(center) + radius * np.column_stack([np.cos(t), np.sin(t)])


def test_different_point_counts_are_angular_correspondence():
    a = circle_points(64)
    b = circle_points(97, 10.3)
    comp = compare_edges(a, b, [0, 0], max_gap_deg=7)
    assert comp.metrics["coverage"] == 1
    assert comp.metrics["mean_um"] == pytest.approx(0.3, abs=1e-13)
    assert comp.metrics["pv_um"] < 1e-13


def test_ellipse_displacement_and_signed_statistics():
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    radius = 1 / np.sqrt((np.cos(theta) / 10.2) ** 2 + (np.sin(theta) / 9.8) ** 2)
    ellipse = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    comp = compare_edges(circle_points(720), ellipse, [0, 0])
    assert comp.radius_b_um[0] - comp.radius_a_um[0] == pytest.approx(0.2)
    assert comp.radius_b_um[90] - comp.radius_a_um[90] == pytest.approx(-0.2)
    assert comp.metrics["pv_um"] == pytest.approx(0.4)
    assert comp.metrics["rms_um"] == pytest.approx(0.2 / np.sqrt(2), abs=0.002)


def test_rejected_calipers_not_filled_even_when_gap_is_small():
    points = circle_points(72)
    keep = np.arange(72) != 20
    comp = compare_edges(
        points,
        points[keep],
        [0, 0],
        max_gap_deg=20,
        ids_a=np.arange(72),
        ids_b=np.arange(72)[keep],
        planned_count=72,
    )
    assert not comp.valid[100]
    assert comp.valid[95] and comp.valid[105]
    assert 0.96 < comp.metrics["coverage"] < 1
    assert np.isnan(comp.points_b_um[100]).all()


def test_large_missing_arc_and_shape_center_frame():
    points = circle_points(72)
    keep = np.arange(72) < 50
    comp = compare_edges(points, points[keep], [0, 0], max_gap_deg=8)
    assert comp.metrics["coverage"] < 0.72
    translated = points + [0.4, -0.3]
    shape = compare_edges(
        points, translated, [0, 0], [0.4, -0.3], center_aligned=True, max_gap_deg=8
    )
    assert shape.metrics["rms_um"] < 1e-13
    assert (
        compare_edges(points, translated, [0, 0], max_gap_deg=8).metrics["rms_um"] > 0.2
    )


def test_coordinate_chain_round_trip_and_no_affine_registration():
    s = session()
    s.a.calibration = Calibration(0.2, 0.4)
    s.b.calibration = Calibration(0.3, 0.5)
    s.b.turns = 1
    s.registration.matrix[:2, 2] = [2, -1]
    points = np.array([[4.0, 5.0], [15.0, 8.0]])
    b = transform_points(points, s.sampling_b())
    np.testing.assert_allclose(
        transform_points(s.b.oriented_calibration.physical(b), s.registration.matrix),
        s.a.oriented_calibration.physical(points),
        atol=1e-14,
    )
    s.registration.matrix[0, 0] = 1.01
    with pytest.raises(ValueError, match="禁止仿射"):
        analyze(s)


def test_actual_detection_with_known_rotation_and_different_pixel_sizes():
    ca = Calibration(0.1, 0.1)
    cb = Calibration(0.12, 0.09)
    theta = np.deg2rad(14)
    matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.3],
            [np.sin(theta), np.cos(theta), -0.5],
            [0, 0, 1.0],
        ]
    )

    def render(shape, cal, mapping):
        y, x = np.mgrid[: shape[0], : shape[1]]
        p = transform_points(
            np.column_stack([x.ravel() * cal.x, y.ravel() * cal.y]), mapping
        )
        d = p - [6.23, 6.37]
        angle = np.arctan2(d[:, 1], d[:, 0])
        r = np.linalg.norm(d, axis=1)
        edge = 3.5 + 0.06 * np.cos(2 * angle) + 0.03 * np.sin(3 * angle)
        return (
            (1000 + 50000 / (1 + np.exp(np.clip(-(r - edge) / 0.085, -60, 60))))
            .reshape(shape)
            .astype(np.float32)
        )

    a = render((128, 130), ca, np.eye(3))
    b = render((150, 155), cb, matrix)
    s = session(a, b)
    s.a.calibration = ca
    s.b.calibration = cb
    s.registration.matrix = matrix
    result = analyze(s)
    assert result.reliable
    assert result.registered.metrics["rms_um"] < 0.006
    assert any("物理平滑宽度" in warning for warning in result.warnings)
    np.testing.assert_array_equal(s.b.original.gray, b)


def test_final_rejection_is_distinct_from_selected_peak_and_missing_correspondence():
    y, x = np.mgrid[:128, :130]
    angle = np.arctan2(y - 63.7, x - 62.3)
    radius = np.hypot(x - 62.3, y - 63.7)
    edge = 35 + 4 * np.exp(-((angle / 0.12) ** 2))
    b = (1000 + 50000 / (1 + np.exp(np.clip(-(radius - edge) / 0.85, -60, 60)))).astype(
        np.float32
    )
    s = session(b=b)
    s.params.use_ransac = True
    s.params.residual_limit_px = 0.3
    result = analyze(s)
    rejected = [trace for trace in result.trace_b if trace["status"] == "fit_rejected"]
    assert rejected
    assert all(t["selected"] is not None and not t["inlier"] for t in rejected)
    assert len(result.b.rejected_points) == len(rejected)
    assert result.registered.metrics["coverage"] < 1
    profile = compare_profiles(s, 0)
    assert profile["b"]["selected_um"] is not None
    assert not profile["local_shift_valid"]


def test_disjoint_fields_rejected_in_feature_registration():
    with pytest.raises(ValueError, match="共同视场"):
        translation_from_region(
            np.ones((200, 200), np.float32),
            np.ones((30, 30), np.float32),
            Calibration(),
            Calibration(),
            (40, 40, 80, 80),
        )


def test_unreliable_registration_prevents_numeric_analysis():
    s = session()
    s.registration.reliable = False
    with pytest.raises(ValueError, match="可靠性"):
        analyze(s)
    assert s.result is None


def test_width_requires_single_monotonic_transition():
    x = np.linspace(-2, 2, 301)
    y = 1 / (1 + np.exp(-x / 0.1))
    width, reason = edge_width(x, y, [0.0])
    assert width == pytest.approx(2 * np.log(9) * 0.1, abs=0.002)
    assert not reason
    assert edge_width(x, y, [0.0, 1.0])[0] is None
    assert edge_width(x, y + 0.2 * np.exp(-(((x - 0.8) / 0.1) ** 2)), [0.0])[0] is None


def test_full_precision_traceable_exports(tmp_path):
    path = tmp_path / "历史中文.tif"
    ok, encoded = cv2.imencode(".tif", circle_image().astype(np.uint16))
    encoded.tofile(str(path))
    s = session()
    s.a = DiagnosticImage.load(path)
    s.b = DiagnosticImage.load(path)
    r = analyze(s)
    output = export_session(s, tmp_path / "结果", excel=True)
    export_session(s, output, excel=False)
    manifest = json.loads((output / "session.json").read_text(encoding="utf-8"))
    assert (
        manifest["a"]["sha256_import"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert not manifest["a"]["file_changed_since_import"]
    assert manifest["traces_a"][0]["candidates"]
    edges = pd.read_csv(output / "csv/edges_A.csv")
    assert edges.iloc[0]["x_native_px"] == pytest.approx(
        r.a.edge_points[0, 0], abs=1e-12
    )
    assert (
        abs(edges.iloc[0]["x_native_px"] - round(edges.iloc[0]["x_native_px"], 3))
        > 1e-6
    )
    book = pd.ExcelFile(output / "analysis.xlsx")
    assert "registration" in book.sheet_names
    assert cv2.imdecode(
        np.fromfile(str(output / "edge_displacement.png"), np.uint8),
        cv2.IMREAD_UNCHANGED,
    ).shape == (620, 1280, 3)


def test_diagnostics_ui_state_isolation_and_rotation(tmp_path):
    from overlay_measure.ui_main import MainWindow
    from overlay_measure.diagnostics.ui_diagnostics import DiagnosticsDialog

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    original_params = deepcopy(asdict(window.params))
    original_config = deepcopy(asdict(window.config))
    original_marks = deepcopy(window.marks)
    dialog = DiagnosticsDialog(window.params, window.config, window)
    try:
        path = tmp_path / "圆孔.tif"
        _, encoded = cv2.imencode(".tif", circle_image().astype(np.uint16))
        encoded.tofile(str(path))
        dialog.load_path("A", str(path))
        dialog.load_path("B", str(path))
        dialog.show()
        app.processEvents()

        dialog.rotation.setCurrentIndex(1)
        app.processEvents()
        dialog.editors["sigma"].setValue(2.3)
        dialog.editors["bx"].setValue(0.2)
        assert dialog.session.params.gaussian_sigma_px == 2.3
        assert asdict(window.params) == original_params
        assert asdict(window.config) == original_config
        assert window.marks == original_marks
        assert dialog.session.b.original.gray.shape == (128, 130)
        assert dialog.source_b.image.gray.shape == (130, 128)
        assert window.soma_diagnostics_action.text().startswith("SOMA Diagnostics")
        assert window.image_diagnostics_action.text() == "图像诊断与边缘分析"
    finally:
        dialog.close()
        window.close()
        app.processEvents()


def test_background_center_registration_and_analysis_lifecycle(tmp_path):
    import time
    from overlay_measure.diagnostics.ui_diagnostics import DiagnosticsDialog

    app = QApplication.instance() or QApplication([])
    dialog = DiagnosticsDialog(
        DetectionParams(use_ransac=False, profile_step_px=0.5), MeasurementConfig()
    )
    try:
        for name, center in [("A", (62.3, 63.7)), ("B", (65.3, 61.7))]:
            path = tmp_path / (name + ".tif")
            _, encoded = cv2.imencode(
                ".tif", circle_image(center=center).astype(np.uint16)
            )
            encoded.tofile(str(path))
            dialog.load_path(name, str(path))
        dialog.count.setValue(40)
        dialog.editors["width"].setValue(3)
        dialog.registration_method.setCurrentIndex(2)
        dialog.register()
        assert dialog.busy and not dialog.editors["sigma"].isEnabled()
        deadline = time.monotonic() + 20
        while dialog.busy and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert not dialog.busy and dialog.session.registration.reliable
        assert dialog.worker is None
        dialog.start_analysis()
        deadline = time.monotonic() + 20
        while dialog.busy and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert not dialog.busy and dialog.session.result is not None
        assert dialog.worker is None and dialog.editors["sigma"].isEnabled()
        dialog.set_mode(2)
        dialog.show()
        app.processEvents()
        assert dialog.profile is not None
        assert dialog.grab().width() >= 1120
    finally:
        if dialog.worker is not None:
            dialog.worker.wait(20000)
            app.processEvents()
        dialog.close()
        app.processEvents()
