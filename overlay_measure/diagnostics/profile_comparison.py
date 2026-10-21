from __future__ import annotations
import numpy as np
from ..subpixel_edge_detector import _bilinear_sample
from .models import transform_points


def edge_width(distances, values, candidates):
    """10–90% only for a single, monotonic, well-contained transition."""
    x = np.asarray(distances, float)
    v = np.asarray(values, float)
    if len(v) < 10 or not np.isfinite(v).all() or len(candidates) != 1:
        return None, "多峰、缺失或采样不足"
    n = max(2, len(v) // 8)
    lo = float(np.median(v[:n]))
    hi = float(np.median(v[-n:]))
    span = hi - lo
    if abs(span) < 1e-9:
        return None, "灰度跃迁不足"
    z = (v - lo) / span
    if np.any(np.diff(z) < -0.015):
        return None, "剖面非单调"
    if np.ptp(v[:n]) > abs(span) * 0.08 or np.ptp(v[-n:]) > abs(span) * 0.08:
        return None, "端点平台不充分"
    if z.min() > 0.1 or z.max() < 0.9:
        return None, "跃迁未完整包含"
    return float(np.interp(0.9, z, x) - np.interp(0.1, z, x)), ""


def compare_profiles(session, angle_deg):
    """Resample both raw images on EXACTLY the selected A caliper's physical axis.

    Detector-selected peaks and native collected profiles are returned separately
    and never recomputed by a second peak-selection algorithm.
    """
    result = session.result
    if result is None:
        raise ValueError("请先完成边缘分析")
    target = np.deg2rad(float(angle_deg) % 360)
    # Physical angle: nearest A selected edge about the physical fitted center.
    candidates = []
    calibration = session.a.oriented_calibration
    for trace in result.trace_a:
        if trace["selected"] is None:
            continue
        xy = (
            calibration.physical([[trace["selected"]["x"], trace["selected"]["y"]]])[0]
            - result.center_a_um
        )
        angle = np.arctan2(xy[1], xy[0])
        distance = abs(np.angle(np.exp(1j * (angle - target))))
        candidates.append((distance, trace))
    if not candidates:
        raise ValueError("没有可选卡尺")
    _, ta = min(candidates, key=lambda x: x[0])
    tb = result.trace_b[ta["caliper_id"]]
    actual = (
        calibration.physical([[ta["selected"]["x"], ta["selected"]["y"]]])[0]
        - result.center_a_um
    )
    actual_angle = float(np.degrees(np.arctan2(actual[1], actual[0])) % 360)
    # The two collectors used this same ROI frame and radial index during detection.
    distances = ta["radial_samples"].astype(np.float64)
    axis = ta["start"] + distances[:, None] * ta["search_vector"]
    tangent = ta["tangent"]
    offsets = ta["tangent_offsets"]
    scale = float(
        np.linalg.norm(ta["search_vector"] * np.array([calibration.x, calibration.y]))
    )
    origin = 0.0
    if ta["selected"] is not None:
        origin = float(
            np.dot(
                np.array([ta["selected"]["x"], ta["selected"]["y"]]) - ta["start"],
                ta["search_vector"],
            )
        )
    x = (distances - origin) * scale
    output = {
        "distance_um": x,
        "caliper_id": ta["caliper_id"],
        "direction": ta["search_vector"].tolist(),
        "selected_physical_angle_deg": actual_angle,
        "step_um": ta["step_px"] * scale,
        "tangent_width_px": float(np.ptp(offsets)),
        "interpolation": ta["interpolation"],
        "a": {},
        "b": {},
    }
    for name, slot, trace, matrix in [
        ("a", session.a, ta, np.eye(3)),
        ("b", session.b, tb, result.sampling_b),
    ]:
        raw = []
        for center in axis:
            coords = transform_points(center + offsets[:, None] * tangent, matrix)
            samples = np.array(
                [
                    _bilinear_sample(slot.oriented.gray, float(xx), float(yy))
                    for xx, yy in coords
                ]
            )
            finite = samples[np.isfinite(samples)]
            raw.append(float(finite.mean()) if len(finite) else np.nan)
        profile = np.asarray(trace["profile"], float)
        grad = np.asarray(trace["gradient"], float) / scale
        peak_positions = [
            (c["sub_index"] * trace["step_px"] - origin) * scale
            for c in trace["candidates"]
            if c["above_threshold"]
        ]
        all_peak_positions = [
            (c["sub_index"] * trace["step_px"] - origin) * scale
            for c in trace["candidates"]
        ]
        selected = trace["selected"]
        selected_pos = (
            None
            if selected is None
            else float(
                (
                    np.dot(
                        np.array([selected["x"], selected["y"]]) - ta["start"],
                        ta["search_vector"],
                    )
                    - origin
                )
                * scale
            )
        )
        width, reason = (
            edge_width(x, profile, all_peak_positions)
            if len(profile) == len(x)
            else (None, "剖面越界")
        )
        output[name] = dict(
            raw=np.asarray(raw),
            filtered=profile,
            gradient=grad,
            candidates_um=peak_positions,
            all_candidates_um=all_peak_positions,
            selected_um=selected_pos,
            selected_gradient_dn_per_um=(
                None if selected is None else selected["gradient"] / scale
            ),
            inlier=trace["inlier"],
            status=trace["status"],
            width_um=width,
            width_reason=reason,
            sampling_xy_native=transform_points(axis, matrix),
        )
    output["local_shift_um"] = (
        output["b"]["selected_um"] - output["a"]["selected_um"]
        if output["b"]["selected_um"] is not None
        and output["a"]["selected_um"] is not None
        else None
    )
    output["local_shift_valid"] = bool(ta["inlier"] and tb["inlier"])
    return output
