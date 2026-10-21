"""Local intensity explanation of EXISTING main-page edge coordinates."""

import cv2
import numpy as np
from ..subpixel_edge_detector import _bilinear_sample
from ..caliper_circle_detector import _profile_edge_candidates
from .models import transform_points
from .profile_comparison import edge_width


def measurement_profile(snapshot, angle_deg, cache):
    s = snapshot.session
    r = s.result
    c = r.registered
    i = int(round(angle_deg)) % 360
    if not c.valid[i]:
        raise ValueError("该角度缺少有效对应，请选择有曲线的位置。")
    cal = s.a.oriented_calibration
    pa = c.points_a_um[i]
    pb = c.points_b_um[i]
    direction = pa - r.center_a_um
    direction /= np.linalg.norm(direction)
    step_px = max(0.05, s.params.profile_step_px)
    um_per_frame_px = 1 / np.linalg.norm(direction / [cal.x, cal.y])
    step_um = step_px * um_per_frame_px
    half_um = max(8.0, float(s.params.profile_half_width_px)) * um_per_frame_px
    x = np.arange(-half_um, half_um + step_um * 0.5, step_um)
    physical = pa + x[:, None] * direction
    axis = physical / [cal.x, cal.y]
    output = dict(
        distance_um=x,
        selected_physical_angle_deg=float(i),
        step_um=step_um,
        sampling_line_frame_px=np.array([axis[0], axis[-1]]),
        edge_a_frame_px=pa / [cal.x, cal.y],
        local_shift_um=float(np.dot(pb - pa, direction)),
        local_shift_valid=True,
        interpolation="bilinear; same registered physical line; no edge re-detection",
    )
    for name, slot, matrix in [("a", s.a, np.eye(3)), ("b", s.b, r.sampling_b)]:
        native = transform_points(axis, matrix)
        gray = slot.oriented.gray
        key = (id(slot.original.gray), slot.turns, float(s.params.gaussian_sigma_px))
        if key not in cache:
            sigma = max(0.0, s.params.gaussian_sigma_px)
            cache[key] = (
                cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
                if sigma
                else gray
            )
        filtered_image = cache[key]
        raw = np.array(
            [_bilinear_sample(gray, float(xx), float(yy)) for xx, yy in native]
        )
        filtered = np.array(
            [
                _bilinear_sample(filtered_image, float(xx), float(yy))
                for xx, yy in native
            ],
            np.float32,
        )
        finite = np.isfinite(filtered).all()
        peaks = (
            _profile_edge_candidates(filtered, step_px, s.params.polarity)
            if finite
            else []
        )
        candidates = [float(x[0] + p * step_um) for p, g in peaks]
        above = [
            float(x[0] + p * step_um) for p, g in peaks if g >= s.params.min_gradient
        ]
        gradient = np.gradient(filtered, step_um)
        selected = 0.0 if name == "a" else output["local_shift_um"]
        index = int(np.clip(np.argmin(abs(x - selected)), 0, len(x) - 1))
        width, reason = edge_width(x, filtered, candidates)
        output[name] = dict(
            raw=raw,
            filtered=filtered,
            gradient=gradient,
            all_candidates_um=candidates,
            candidates_um=above,
            selected_um=selected,
            inlier=True,
            status="main-page edge coordinate",
            selected_gradient_dn_per_um=float(abs(gradient[index])),
            width_um=width,
            width_reason=reason,
            sampling_xy_native=native,
        )
    return output
