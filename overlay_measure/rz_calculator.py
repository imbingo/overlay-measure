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
    mark_statuses: dict[str, str] = {}
    for mark_id in sorted(overlays.keys()):
        o = overlays[mark_id]
        idx = "".join(ch for ch in mark_id if ch.isdigit()) or mark_id
        dx = float(o.delta_x_um)
        dy = float(o.delta_y_um)
        finite_xy = bool(np.isfinite(dx) and np.isfinite(dy))
        dxy = float(np.hypot(dx, dy)) if finite_xy else float("nan")
        deltas[mark_id] = (dx, dy)
        mark_statuses[mark_id] = o.result
        trial = is_trial or o.result == "Trial"
        warnings = []
        if finite_xy and not trial and abs(dx) > config.delta_x_limit_um:
            warnings.append(f"Dx{idx}超限")
        if finite_xy and not trial and abs(dy) > config.delta_y_limit_um:
            warnings.append(f"Dy{idx}超限")
        if finite_xy and not trial and dxy > config.overlay_r_limit_um:
            warnings.append(f"Dxy{idx}超限")
        if o.warning and not trial:
            warnings.append(o.warning)
        if o.result == "Invalid":
            verdict = "无效"
            hint = o.warning or "识别质量不满足量测要求"
        elif o.result == "Error" or not finite_xy:
            verdict = "异常"
            hint = o.warning or "计算异常"
            mark_statuses[mark_id] = "Error"
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
            "基准轮廓": o.reference_contour_name,
            "基准ROI稳定ID": o.reference_contour_id,
            "待测轮廓": o.target_contour_name,
            "待测ROI稳定ID": o.target_contour_id,
            "项目": mark_id,
            f"Dx{idx}(μm)": dx if finite_xy else None,
            f"Dy{idx}(μm)": dy if finite_xy else None,
            f"Dxy{idx}(μm)": dxy if finite_xy else None,
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
            formula = "Rz=(Dx2-Dx1)/L"
            rz_rad = (dx2 - dx1) / l_value
        else:
            formula = "Rz=(Dy2-Dy1)/L"
            rz_rad = (dy2 - dy1) / l_value
        status1 = mark_statuses.get("Mark1", "")
        status2 = mark_statuses.get("Mark2", "")
        rz_urad = rz_rad * 1_000_000.0
        finite_rz = bool(
            np.isfinite(dx1) and np.isfinite(dy1) and np.isfinite(dx2) and np.isfinite(dy2) and np.isfinite(rz_urad)
        )
        if "Error" in (status1, status2) or not finite_rz:
            rz_value = None
            rz_verdict = "异常"
            rz_hint = "Mark1 或 Mark2 计算异常，无法计算 Rz"
        elif "Invalid" in (status1, status2):
            rz_value = None
            rz_verdict = "无效"
            rz_hint = "Mark1 或 Mark2 识别质量无效，无法计算 Rz"
        else:
            rz_value = rz_urad
            rz_trial = is_trial or "Trial" in (status1, status2)
            rz_verdict = "试测" if rz_trial else ("不通过" if abs(rz_urad) > config.rz_limit else "通过")
            rz_hint = (
                "未验证配方，不作正式判定"
                if rz_trial
                else ("Rz超限" if abs(rz_urad) > config.rz_limit else "")
            )
        rows.append({
            "项目": "Rz",
            "Rz(μrad)": rz_value,
            "公式": formula,
            "L(μm)": l_value,
            "判定": rz_verdict,
            "质量门槛": "—",
            "实际质量": "—",
            "质量详情": "由 Mark1/Mark2 对位结果计算",
            "提示": rz_hint,
        })
    return rows
