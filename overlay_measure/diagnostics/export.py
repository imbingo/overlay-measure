from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import cv2
from .. import __version__
from ..result_exporter import export_data_tables
from .models import transform_points, file_sha256


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def export_session(session, directory, *, excel=True):
    result = session.result
    if result is None:
        raise ValueError("没有可导出的分析")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, comp in [
        ("registered", result.registered),
        ("center_aligned", result.shape),
    ]:
        tables[name + "_metrics"] = [
            dict(comp.metrics, reliable=result.reliable, **result.circle_metrics)
        ]
        cal = session.a.oriented_calibration
        tables[name + "_angles"] = [
            dict(
                angle_deg=float(a),
                valid=bool(v),
                radius_a_um=float(ra) if np.isfinite(ra) else None,
                radius_b_um=float(rb) if np.isfinite(rb) else None,
                displacement_um=float(rb - ra) if v else None,
                signed_displacement_reference_px=(
                    float(np.sign(rb - ra) * np.linalg.norm((pb - pa) / [cal.x, cal.y]))
                    if v
                    else None
                ),
                correspondence="local radial interpolation between adjacent valid calipers",
            )
            for a, ra, rb, v, pa, pb in zip(
                comp.angles_deg,
                comp.radius_a_um,
                comp.radius_b_um,
                comp.valid,
                comp.points_a_um,
                comp.points_b_um,
            )
        ]
    for name, slot, traces, matrix in [
        ("A", session.a, result.trace_a, np.eye(3)),
        ("B", session.b, result.trace_b, result.sampling_b),
    ]:
        rows = []
        for trace in traces:
            selected = trace["selected"]
            row = dict(
                caliper_id=trace["caliper_id"],
                angle_frame_rad=trace["angle_rad"],
                valid=trace["inlier"],
                status=trace["status"],
            )
            if selected:
                common = np.array([[selected["x"], selected["y"]]])
                oriented = transform_points(common, matrix)
                native = transform_points(
                    oriented, np.linalg.inv(slot.native_to_oriented)
                )[0]
                physical = session.a.oriented_calibration.physical(common)[0]
                row.update(
                    x_native_px=native[0],
                    y_native_px=native[1],
                    x_registered_um=physical[0],
                    y_registered_um=physical[1],
                    gradient_dn_per_frame_px=selected["gradient"],
                )
            rows.append(row)
        tables["edges_" + name] = rows
    tables["registration"] = [
        dict(
            method=session.registration.method,
            features=session.registration.features,
            translation_x_um=session.registration.matrix[0, 2],
            translation_y_um=session.registration.matrix[1, 2],
            translation_x_reference_px=session.registration.matrix[0, 2]
            / session.a.oriented_calibration.x,
            translation_y_reference_px=session.registration.matrix[1, 2]
            / session.a.oriented_calibration.y,
            rotation_deg=session.registration.rotation_deg,
            residual_um=session.registration.residual_um,
            response=session.registration.response,
            reliable=session.registration.reliable,
        )
    ]

    def source(slot):
        path = Path(slot.original.path)
        current = file_sha256(path) if path.is_file() else None
        return dict(
            path=slot.original.path,
            sha256_import=slot.sha256,
            sha256_current=current,
            file_changed_since_import=current != slot.sha256,
            source_dtype=slot.original.source_dtype,
            native_shape=list(slot.original.gray.shape),
            calibration=asdict(slot.calibration),
            oriented_calibration=asdict(slot.oriented_calibration),
            turns_clockwise=slot.turns,
            native_to_oriented=slot.native_to_oriented.tolist(),
        )

    manifest = dict(
        software="SOMA Diagnostics",
        version=__version__,
        utc=datetime.now(timezone.utc).isoformat(),
        a=source(session.a),
        b=source(session.b),
        roi=asdict(session.roi),
        roi_coordinate_frame="Image A oriented pixel frame; mapped directly to B native samples",
        params=asdict(session.params),
        registration=asdict(session.registration),
        sampling_frame_A_to_B=result.sampling_b,
        reliable=result.reliable,
        warnings=result.warnings,
        circle_metrics=result.circle_metrics,
        coordinates="native px; oriented px; calibrated um (X right/Y down); B->A rigid physical; outward positive",
        coverage_definition="valid planned physical angles / all planned physical angles; no bridging rejected calipers",
        valid_intervals_registered_deg=result.registered.valid_intervals_deg,
        valid_intervals_center_aligned_deg=result.shape.valid_intervals_deg,
        units="angles degrees unless named rad; position um; native pixel positions px",
        traces_a=result.trace_a,
        traces_b=result.trace_b,
    )
    # Lists in metrics become explicit JSON cells rather than Python repr in spreadsheets.
    for rows in tables.values():
        for row in rows:
            for key, val in list(row.items()):
                if isinstance(val, (list, dict)):
                    row[key] = json.dumps(val, ensure_ascii=False)
    export_data_tables(
        str(target / "analysis.xlsx") if excel else str(target / "csv"), tables
    )
    (target / "session.json").write_text(
        json.dumps(
            manifest, default=json_value, ensure_ascii=False, indent=2, allow_nan=False
        ),
        encoding="utf-8",
    )
    for name, comp in [
        ("edge_displacement", result.registered),
        ("center_aligned_displacement", result.shape),
    ]:
        export_curve_png(comp, target / (name + ".png"), result.reliable)
    return target


def export_curve_png(comparison, path, reliable):
    """Portable full-resolution plot; OpenCV already ships with SOMA."""
    img = np.full((620, 1280, 3), 255, np.uint8)
    left, top, right, bottom = 110, 85, 1230, 540
    d = comparison.radius_b_um - comparison.radius_a_um
    finite = d[comparison.valid]
    low = min(float(finite.min()), 0.0) if len(finite) else -1.0
    high = max(float(finite.max()), 0.0) if len(finite) else 1.0
    padding = max((high - low) * 0.1, 1e-6)
    low -= padding
    high += padding

    def text(s, xy, scale=0.55, color=(65, 65, 65)):
        cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    text("SOMA Diagnostics - Signed edge displacement (um)", (110, 32), 0.8)
    text(
        ("Center-aligned shape" if comparison.center_aligned else "Registered geometry")
        + (" / VALID" if reliable else " / UNRELIABLE - inspection only"),
        (110, 60),
    )
    for val in np.linspace(low, high, 7):
        yy = int(bottom - (val - low) / (high - low) * (bottom - top))
        cv2.line(img, (left, yy), (right, yy), (230, 230, 230), 1)
        text(f"{val:.6f}", (5, yy + 5), 0.45)
    zero = int(bottom - (0 - low) / (high - low) * (bottom - top))
    cv2.line(img, (left, zero), (right, zero), (120, 120, 120), 1)
    previous = None
    for angle, value, valid in zip(comparison.angles_deg, d, comparison.valid):
        if not valid:
            previous = None
            continue
        pt = (
            int(left + angle / 360 * (right - left)),
            int(bottom - (value - low) / (high - low) * (bottom - top)),
        )
        if previous is not None:
            cv2.line(img, previous, pt, (65, 140, 240), 2, cv2.LINE_AA)
        previous = pt
    for angle in range(0, 361, 60):
        text(str(angle), (int(left + angle / 360 * (right - left)) - 10, 570))
    text(
        "Angle (deg) / gaps excluded; outward positive; not measurement uncertainty",
        (240, 605),
    )
    success, encoded = cv2.imencode(".png", img)
    if not success:
        raise ValueError("PNG curve export failed")
    encoded.tofile(str(path))
