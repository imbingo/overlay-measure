from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class EdgeComparison:
    angles_deg: np.ndarray
    radius_a_um: np.ndarray
    radius_b_um: np.ndarray
    valid: np.ndarray
    points_a_um: np.ndarray
    points_b_um: np.ndarray
    metrics: dict
    center_aligned: bool = False

    @property
    def valid_intervals_deg(self):
        intervals = []
        start = None
        for i, valid in enumerate(self.valid):
            if valid and start is None:
                start = i
            if start is not None and (not valid or i == len(self.valid) - 1):
                end = i if valid else i - 1
                intervals.append(
                    [float(self.angles_deg[start]), float(self.angles_deg[end])]
                )
                start = None
        return intervals


def radial_intersections(
    points, center, angles_deg, max_gap_deg, caliper_ids=None, planned_count=None
):
    """Local radial interpolation, restricted to adjacent valid planned calipers."""
    points = np.asarray(points, np.float64).reshape(-1, 2)
    finite = np.isfinite(points).all(1)
    points = points[finite]
    ids = np.asarray(caliper_ids)[finite] if caliper_ids is not None else None
    out = np.full((len(angles_deg), 2), np.nan)
    if len(points) < 3:
        return out
    v = points - np.asarray(center)
    theta = np.mod(np.degrees(np.arctan2(v[:, 1], v[:, 0])), 360)
    order = np.argsort(theta)
    theta = theta[order]
    v = v[order]
    if ids is not None:
        ids = np.r_[ids[order], ids[order][0]]
    # Coincident angles are ambiguous (e.g. two physical edges); don't silently pair them.
    if np.any(np.diff(theta) < 1e-7):
        raise ValueError("同一角度存在多个边缘，请检查是否选中同一物理边缘")
    theta = np.r_[theta, theta[0] + 360]
    v = np.vstack([v, v[0]])
    for i, angle in enumerate(angles_deg):
        a = float(angle) % 360
        if a < theta[0]:
            a += 360
        j = min(int(np.searchsorted(theta, a, side="right") - 1), len(v) - 2)
        if abs(a - theta[j]) < 1e-7:
            out[i] = np.asarray(center) + v[j]
            continue
        if abs(a - theta[j + 1]) < 1e-7:
            out[i] = np.asarray(center) + v[j + 1]
            continue
        if theta[j + 1] - theta[j] > max_gap_deg + 1e-9:
            continue
        if ids is not None and (int(ids[j + 1]) - int(ids[j])) % planned_count != 1:
            continue
        d = np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        t = (a - theta[j]) / (theta[j + 1] - theta[j])
        radius = (1 - t) * np.linalg.norm(v[j]) + t * np.linalg.norm(v[j + 1])
        if radius > 0:
            out[i] = np.asarray(center) + radius * d
    return out


def compare_edges(
    a_um,
    b_um,
    center_a,
    center_b=None,
    *,
    count=360,
    max_gap_deg=12.0,
    center_aligned=False,
    ids_a=None,
    ids_b=None,
    planned_count=None,
):
    if count < 8 or not 0 < max_gap_deg <= 30:
        raise ValueError("角采样和最大角间隙不合法")
    a = np.asarray(a_um, np.float64)
    b = np.asarray(b_um, np.float64)
    center_a = np.asarray(center_a, np.float64)
    if center_aligned:
        if center_b is None:
            raise ValueError("形状比较需要 B 圆心")
        b = b + center_a - np.asarray(center_b, np.float64)
    angles = np.linspace(0, 360, count, endpoint=False)
    pa = radial_intersections(a, center_a, angles, max_gap_deg, ids_a, planned_count)
    pb = radial_intersections(b, center_a, angles, max_gap_deg, ids_b, planned_count)
    ra = np.linalg.norm(pa - center_a, axis=1)
    rb = np.linalg.norm(pb - center_a, axis=1)
    valid = np.isfinite(ra) & np.isfinite(rb)
    d = (rb - ra)[valid]
    metrics = {
        "coverage": float(valid.mean()),
        "valid_count": int(valid.sum()),
        "planned_count": count,
        "mean_um": None,
        "rms_um": None,
        "pv_um": None,
        "p95_abs_um": None,
        "max_abs_um": None,
    }
    if len(d):
        metrics.update(
            mean_um=float(d.mean()),
            rms_um=float(np.sqrt(np.mean(d * d))),
            pv_um=float(np.ptp(d)),
            p95_abs_um=float(np.percentile(abs(d), 95)),
            max_abs_um=float(np.max(abs(d))),
        )
    return EdgeComparison(angles, ra, rb, valid, pa, pb, metrics, center_aligned)
