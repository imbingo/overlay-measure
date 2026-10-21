"""Read-only bridge from production images/contours to a small diagnostic viewer.

No detector is run here: the displayed edge points ARE the main-page results.
"""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, replace, asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np
from ..models import DetectionResult
from ..result_exporter import export_data_tables
from .models import (
    Calibration,
    DiagnosticImage,
    DiagnosticsSession,
    Registration,
    transform_points,
    file_sha256,
)
from .diagnostics_service import AnalysisResult
from .edge_comparison import compare_edges
from .export import export_curve_png, json_value


@dataclass
class MeasurementSnapshot:
    session: DiagnosticsSession
    upper: dict[str, DetectionResult]
    lower: dict[str, DetectionResult]
    selected_upper: str = ""
    selected_lower: str = ""
    label: str = ""
    roi_snapshot: dict | None = None


def snapshot_main(window):
    mark = window._current_mark_id()
    calibration = Calibration(
        window.config.pixel_size_x_um, window.config.pixel_size_y_um
    )
    session = DiagnosticsSession(params=deepcopy(window.params))
    # The currently displayed batch pair is also resolved by this existing API.
    for layer, name in [("upper", "a"), ("lower", "b")]:
        image = (
            window._active_image_for_layer(mark, layer)
            if window._current_mode() == "Dual Image" or layer == "upper"
            else None
        )
        if image is not None:
            gray = np.array(image.gray, copy=True, order="C")
            gray.setflags(write=False)
            setattr(
                session, name, DiagnosticImage(replace(image, gray=gray), calibration)
            )
    if window._is_auto_workflow():
        entries = [
            (key, d)
            for key, layers in window._current_auto_detections().items()
            for d in layers.values()
        ]
    else:
        entries = list(window._current_manual_detection_map().get(mark, {}).items())
    entries = [
        (key, deepcopy(d))
        for key, d in entries
        if d.fitting_mode in {"Circle", "CaliperCircle", "ProductionCircle", "Ellipse"}
        and len(d.edge_points) >= 3
    ]
    upper = {k: d for k, d in entries if d.layer == "upper"}
    lower = {k: d for k, d in entries if d.layer == "lower"}
    reference = window.auto_reference_combo.currentData() or ""
    target = window.auto_target_combo.currentData() or ""
    upper_preferred = (
        reference if reference in upper else target if target in upper else ""
    )
    lower_preferred = (
        target if target in lower else reference if reference in lower else ""
    )
    record = window._selected_batch_record(mark)
    label = mark + (
        f" / 第 {window._batch_detail_last_single_index} 组" if record else ""
    )
    recipe = window.marks.get(mark)
    rois = (
        {
            layer: [asdict(entry) for entry in recipe.roi_entries(layer)]
            for layer in ("upper", "lower")
        }
        if recipe
        else {}
    )
    return MeasurementSnapshot(
        session,
        upper,
        lower,
        str(upper_preferred),
        str(lower_preferred),
        label,
        deepcopy(rois),
    )


def align_measurement(snapshot, upper_key, lower_key, turns=0):
    """One-click, known quarter-turn then translation of EXISTING measured centers."""
    s = snapshot.session
    s.require_pair()
    s.result = None
    if upper_key not in snapshot.upper or lower_key not in snapshot.lower:
        raise ValueError("请先在主页面识别上下层圆孔，再点击“同步主页面”。")
    a = snapshot.upper[upper_key]
    b = snapshot.lower[lower_key]
    s.b.turns = int(turns) % 4
    ca = s.a.calibration.physical([[a.center_x_px, a.center_y_px]])[0]
    b_center = transform_points(
        [[b.center_x_px, b.center_y_px]], s.b.native_to_oriented
    )
    cb = s.b.oriented_calibration.physical(b_center)[0]
    matrix = np.eye(3)
    matrix[:2, 2] = ca - cb
    for d in (a, b):
        if not np.isfinite([d.center_x_px, d.center_y_px, d.diameter_um]).all():
            raise ValueError("主页面检测结果无效，请重新识别。")
    s.registration = Registration(
        matrix,
        "circle_center",
        "主页面已识别圆心，一键平移对齐",
        reliable=True,
        warning="已按圆心自动对齐：整体平移已消除；单个圆孔不能确定未知旋转，不能据此判断真实圆心漂移。",
        details={
            "source": "main_page_detection",
            "upper_key": upper_key,
            "lower_key": lower_key,
            "raw_center_delta_um_before_alignment": (cb - ca).tolist(),
        },
    )
    cal = s.a.oriented_calibration
    ap = cal.physical(a.edge_points)

    def b_physical(points):
        return transform_points(
            s.b.oriented_calibration.physical(
                transform_points(points, s.b.native_to_oriented)
            ),
            matrix,
        )

    bp = b_physical(b.edge_points)

    # Gaps are bounded by observed spacing; do not bridge rejected edge sectors.
    def gap(points, center):
        v = points - center
        t = np.sort(np.mod(np.degrees(np.arctan2(v[:, 1], v[:, 0])), 360))
        spacing = np.diff(np.r_[t, t[0] + 360])
        spacing = spacing[spacing > 1e-7]
        return min(12.0, max(1e-6, float(np.median(spacing)) * 1.5))

    comp = compare_edges(
        ap, bp, ca, ca, max_gap_a_deg=gap(ap, ca), max_gap_b_deg=gap(bp, ca)
    )
    rejected_a = np.asarray(a.rejected_points, float).reshape(-1, 2)
    rejected_b = b_physical(b.rejected_points)
    view_a = SimpleNamespace(
        edge_points=np.asarray(a.edge_points, float).reshape(-1, 2),
        rejected_points=rejected_a,
    )
    view_b = SimpleNamespace(
        edge_points=bp / [cal.x, cal.y], rejected_points=rejected_b / [cal.x, cal.y]
    )
    circle = dict(
        center_a_um=ca.tolist(),
        center_b_registered_um=ca.tolist(),
        center_delta_registered_um=[0.0, 0.0],
        center_a_native_px=[a.center_x_px, a.center_y_px],
        center_b_native_px=[b.center_x_px, b.center_y_px],
        diameter_a_um=a.diameter_um,
        diameter_b_um=b.diameter_um,
        diameter_delta_um=b.diameter_um - a.diameter_um,
        center_translation_removed=True,
        definition="Main-page detection results, not refitted",
    )
    warnings = [
        s.registration.warning,
        "曲线按物理角度比较；不重新检测或修改主页面边缘点。",
    ]
    reliable = comp.metrics["coverage"] >= 0.7
    if any(d.edge_point_count > len(d.edge_points) for d in (a, b)):
        reliable = False
        warnings.append(
            "主页面批量记录保存的是抽样轮廓，仅供预览；完整诊断请在主页面对当前图像重新识别。"
        )
    if any(d.shape_params.get("measurement_eligible") is False for d in (a, b)):
        reliable = False
        warnings.append("主页面轮廓未通过质量门槛，仅供诊断查看，不作量测结论。")
    if comp.metrics["coverage"] < 0.7:
        warnings.append("有效对应不足 70%，结果仅供查看，请回主页面检查轮廓。")
    s.result = AnalysisResult(
        view_a,
        view_b,
        [],
        [],
        comp,
        comp,
        ca,
        ca,
        circle,
        reliable,
        warnings,
        s.sampling_b(),
    )
    return s.result


def export_measurement(snapshot, upper_key, lower_key, directory):
    from .. import __version__

    s = snapshot.session
    r = s.result
    if r is None:
        raise ValueError("请先点击自动配准")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    c = r.registered
    tables = {
        "metrics": [dict(c.metrics, **r.circle_metrics, reliable=r.reliable)],
        "angles": [
            dict(
                angle_deg=a,
                valid=bool(v),
                displacement_um=float(rb - ra) if v else None,
            )
            for a, ra, rb, v in zip(c.angles_deg, c.radius_a_um, c.radius_b_um, c.valid)
        ],
    }
    for name, d in [
        ("upper", snapshot.upper[upper_key]),
        ("lower", snapshot.lower[lower_key]),
    ]:
        rows = []
        for valid, points in [(True, d.edge_points), (False, d.rejected_points)]:
            rows.extend(
                dict(x_native_px=x, y_native_px=y, valid=valid) for x, y in points
            )
        tables[name + "_edges"] = rows
    # Convert structured metrics to JSON cells for the shared table writer.
    for rows in tables.values():
        for row in rows:
            for key, value in list(row.items()):
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, ensure_ascii=False)
    export_data_tables(str(out / "analysis.xlsx"), tables)
    export_data_tables(str(out / "csv"), tables)

    def source(slot):
        path = Path(slot.original.path)
        return dict(
            path=slot.original.path,
            source_file_sha256_at_export=(
                file_sha256(path) if path.is_file() else None
            ),
            source_matches_original_import="not verified by main ImageData model",
            gray_sha256=hashlib.sha256(slot.original.gray.tobytes()).hexdigest(),
            gray_shape=list(slot.original.gray.shape),
            gray_dtype=str(slot.original.gray.dtype),
            source_dtype=slot.original.source_dtype,
            calibration=asdict(slot.calibration),
            turns_clockwise=slot.turns,
        )

    manifest = dict(
        version=__version__,
        source="main_page_read_only_snapshot",
        upper=source(s.a),
        lower=source(s.b),
        params=asdict(s.params),
        registration=asdict(s.registration),
        warnings=r.warnings,
        main_page_rois=snapshot.roi_snapshot,
        upper_detection=asdict(snapshot.upper[upper_key]),
        lower_detection=asdict(snapshot.lower[lower_key]),
        valid_intervals_deg=c.valid_intervals_deg,
        point_policy="original main-page points, never refitted",
    )
    (out / "session.json").write_text(
        json.dumps(manifest, default=json_value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    export_curve_png(c, out / "edge_displacement.png", r.reliable)
    return out
