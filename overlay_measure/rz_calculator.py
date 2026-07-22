from __future__ import annotations

import numpy as np

from .models import MeasurementConfig, OverlayResult


def build_summary_rows(
    overlays: dict[str, OverlayResult],
    config: MeasurementConfig,
    is_trial: bool = False,
) -> list[dict]:
    rows = []
    deltas: dict[str, tuple[float, float]] = {}
    for mark_id in sorted(overlays.keys()):
        o = overlays[mark_id]
        idx = "".join(ch for ch in mark_id if ch.isdigit()) or mark_id
        dx = o.delta_x_um
        dy = o.delta_y_um
        dxy = float(np.hypot(dx, dy))
        deltas[mark_id] = (dx, dy)
        trial = is_trial or o.result == "Trial"
        warnings = []
        if not trial and abs(dx) > config.delta_x_limit_um:
            warnings.append(f"Dx{idx}超限")
        if not trial and abs(dy) > config.delta_y_limit_um:
            warnings.append(f"Dy{idx}超限")
        if not trial and dxy > config.overlay_r_limit_um:
            warnings.append(f"Dxy{idx}超限")
        if o.warning and not trial:
            warnings.append(o.warning)
        if o.result == "Invalid":
            verdict = "无效"
            hint = o.warning or "识别质量不满足量测要求"
        elif o.result == "Error":
            verdict = "异常"
            hint = o.warning or "计算异常"
        elif trial:
            verdict = "试测"
            hint = "未验证配方，不作正式判定"
        elif warnings:
            verdict = "超限"
            hint = "；".join(warnings)
        else:
            verdict = "通过"
            hint = ""
        rows.append({
            "项目": mark_id,
            f"Dx{idx}(μm)": dx,
            f"Dy{idx}(μm)": dy,
            f"Dxy{idx}(μm)": dxy,
            "判定": verdict,
            "质量门槛": o.quality_profile,
            "实际质量": o.quality_grade,
            "质量详情": o.quality_summary,
            "提示": hint,
        })

    if "Mark1" in deltas and "Mark2" in deltas:
        dx1, dy1 = deltas["Mark1"]
        dx2, dy2 = deltas["Mark2"]
        l_value = max(config.rz_distance_l_um, 1e-12)
        if config.rz_layout == "Y向前后分布":
            rz_rad = (dx2 - dx1) / l_value
            formula = "Rz=(Dx2-Dx1)/L"
        else:
            rz_rad = (dy2 - dy1) / l_value
            formula = "Rz=(Dy2-Dy1)/L"
        rz_urad = rz_rad * 1_000_000.0
        rz_trial = is_trial
        rows.append({
            "项目": "Rz",
            "Rz(μrad)": rz_urad,
            "公式": formula,
            "L(μm)": l_value,
            "判定": "试测" if rz_trial else ("不通过" if abs(rz_urad) > config.rz_limit else "通过"),
            "质量门槛": "—",
            "实际质量": "—",
            "质量详情": "由 Mark1/Mark2 对位结果计算",
            "提示": "未验证配方，不作正式判定" if rz_trial else ("Rz超限" if abs(rz_urad) > config.rz_limit else ""),
        })
    return rows
