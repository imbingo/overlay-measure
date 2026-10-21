from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from ..caliper_circle_detector import detect_caliper_circle
from ..measurement_units import robust_circle_um, radial_diameter_statistics_um
from .models import Registration, transform_points
from .edge_comparison import compare_edges


@dataclass
class AnalysisResult:
    a: object
    b: object
    trace_a: list
    trace_b: list
    registered: object
    shape: object
    center_a_um: np.ndarray
    center_b_um: np.ndarray
    circle_metrics: dict
    reliable: bool
    warnings: list
    sampling_b: np.ndarray


def center_align(session):
    session.require_pair()
    if session.roi is None or session.seed_b_roi is None:
        raise ValueError("圆心平移需分别指定 A 参考环带和 B 初始环带")
    aa = detect_caliper_circle(
        session.a.oriented.gray, session.roi, deepcopy(session.params)
    )
    bb = detect_caliper_circle(
        session.b.oriented.gray, session.seed_b_roi, deepcopy(session.params)
    )
    ca_px, _, _ = robust_circle_um(
        aa.edge_points, session.a.oriented_calibration.config()
    )
    cb_px, _, _ = robust_circle_um(
        bb.edge_points, session.b.oriented_calibration.config()
    )
    ca = session.a.oriented_calibration.physical([ca_px])[0]
    cb = session.b.oriented_calibration.physical([cb_px])[0]
    matrix = np.eye(3)
    matrix[:2, 2] = ca - cb
    # Refine in the common sampling frame so different initial ROI centers do not
    # leave tangential averaging biases disguised as shape changes.
    correction = np.inf
    for _ in range(8):
        sampling = (
            np.linalg.inv(session.b.oriented_calibration.matrix)
            @ np.linalg.inv(matrix)
            @ session.a.oriented_calibration.matrix
        )
        aligned = detect_caliper_circle(
            session.b.oriented.gray,
            session.roi,
            deepcopy(session.params),
            sampling_transform=sampling,
        )
        aligned_center, _, _ = robust_circle_um(
            aligned.edge_points, session.a.oriented_calibration.config()
        )
        delta = ca - session.a.oriented_calibration.physical([aligned_center])[0]
        correction = float(np.linalg.norm(delta))
        matrix[:2, 2] += delta
        if correction < 1e-7:
            break
    return Registration(
        matrix,
        "circle_center",
        "被测孔圆心，仅供相对形状分析",
        reliable=correction < 1e-4,
        warning="圆心对齐消除整体平移；单个圆不能确定旋转，不能据此判断真实圆心漂移",
        details={
            "center_alignment_convergence_um": correction,
            "not_independent_registration_residual": True,
        },
    )


def analyze(session):
    session.require_pair()
    session.result = None
    if session.roi is None or session.roi.roi_type != "Caliper Circle":
        raise ValueError("请在 A 图指定 Caliper Circle 环带")
    if not session.registration.reliable:
        raise ValueError("配准尚未通过可靠性检查，无法定量比较")
    sampling = session.sampling_b()
    ta = []
    tb = []
    a = detect_caliper_circle(
        session.a.oriented.gray,
        deepcopy(session.roi),
        deepcopy(session.params),
        diagnostics=ta,
    )
    b = detect_caliper_circle(
        session.b.oriented.gray,
        deepcopy(session.roi),
        deepcopy(session.params),
        diagnostics=tb,
        sampling_transform=sampling,
    )
    calibration = session.a.oriented_calibration
    config = calibration.config()
    ap = calibration.physical(a.edge_points)
    bp = calibration.physical(b.edge_points)
    # Existing physical fitter handles anisotropic reference pixel axes.
    ca_px, ra, _ = robust_circle_um(a.edge_points, config)
    cb_px, rb, _ = robust_circle_um(b.edge_points, config)
    ca = calibration.physical([ca_px])[0]
    cb = calibration.physical([cb_px])[0]
    ids_a = [t["caliper_id"] for t in ta if t["inlier"]]
    ids_b = [t["caliper_id"] for t in tb if t["inlier"]]
    kwargs = dict(max_gap_deg=30, ids_a=ids_a, ids_b=ids_b, planned_count=len(ta))
    registered = compare_edges(ap, bp, ca, cb, **kwargs)
    shape = compare_edges(ap, bp, ca, cb, center_aligned=True, **kwargs)
    stat_a = radial_diameter_statistics_um(a.edge_points, *ca_px, config)
    stat_b = radial_diameter_statistics_um(b.edge_points, *cb_px, config)
    da = stat_a["average_diameter_um"]
    db = stat_b["average_diameter_um"]
    cb_native = transform_points([cb_px], sampling)[0]
    ca_native = transform_points([ca_px], np.linalg.inv(session.a.native_to_oriented))[
        0
    ]
    cb_original = transform_points(
        [cb_native], np.linalg.inv(session.b.native_to_oriented)
    )[0]
    circle = dict(
        definition="shared robust_circle_um physical fit of final inlier points; average radial diameter",
        detector_a_center_roi_frame_px=[a.center_x_px, a.center_y_px],
        detector_b_center_roi_frame_px=[b.center_x_px, b.center_y_px],
        detector_a_radius_frame_px=a.radius_px,
        detector_b_radius_frame_px=b.radius_px,
        center_a_um=ca.tolist(),
        center_b_registered_um=cb.tolist(),
        center_delta_registered_um=(cb - ca).tolist(),
        center_a_native_px=ca_native.tolist(),
        center_b_native_px=cb_original.tolist(),
        diameter_a_um=da,
        diameter_b_um=db,
        diameter_delta_um=db - da,
        center_translation_removed=session.registration.method == "circle_center",
    )
    warnings = [session.registration.warning] if session.registration.warning else []
    if not np.allclose(
        [session.a.oriented_calibration.x, session.a.oriented_calibration.y],
        [session.b.oriented_calibration.x, session.b.oriented_calibration.y],
    ):
        warnings.append(
            "两图原生像素标定不同：同一 Gaussian px 参数的物理平滑宽度不同；已记录，需确认可比性"
        )
    warnings.append(
        "径向正值为向孔外；局部角插值仅连接相邻有效卡尺。数值精度不代表测量不确定度。"
    )
    reliable = (
        min(registered.metrics["coverage"], shape.metrics["coverage"]) >= 0.7
        and min(a.angular_coverage, b.angular_coverage) >= 0.7
    )
    if not reliable:
        warnings.append("有效覆盖率不足 70%，禁止将统计作为可靠定量结论")
    # Large shifts relative to ROI width often mean a different physical edge was selected.
    available = (session.roi.outer_radius() - session.roi.inner_radius()) * min(
        calibration.x, calibration.y
    )
    if (
        registered.metrics["max_abs_um"] is not None
        and registered.metrics["max_abs_um"] > available * 0.4
    ):
        reliable = False
        warnings.append("边缘差异接近搜索带宽，可能选中了不同物理边缘，请核查剖面")
    result = AnalysisResult(
        a, b, ta, tb, registered, shape, ca, cb, circle, reliable, warnings, sampling
    )
    session.result = result
    return result
