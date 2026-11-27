"""Frozen single-ROI previews that reuse the production manual-ROI detectors.

Caliper circles keep their dedicated caliper diagnostics.  Every other manual
ROI type is measured through :func:`measurement_service.detect_manual_roi`
(the authoritative result, so values never deviate from formal measurement),
while a separate diagnostics replay of the same edge extractor collects 1D
normal profiles for visualisation only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import numpy as np

from .caliper_circle_detector import CaliperCircleResult, detect_caliper_circle
from .measurement_units import radial_diameter_residual_um, radial_diameter_statistics_um
from .models import DetectionParams, ImageData, MeasurementConfig, Roi
from .subpixel_edge_detector import _bilinear_sample

TUNABLE_FIELDS = (
    "gaussian_sigma_px", "canny_low", "canny_high", "min_gradient",
    "profile_half_width_px", "profile_step_px", "residual_limit_px",
    "use_ransac", "polarity",
)
MAX_RECORDS = 600

ROI_LABELS = {
    "Caliper Circle": "卡尺圆", "Annulus": "圆环", "Rectangular Ring": "矩形环",
    "Circle": "圆形", "Ellipse": "椭圆", "Rectangle": "矩形",
    "Robust Center": "稳健中心", "Region Center": "区域中心", "Approximate Line": "近似直线",
}
SUPPORTED_ROI_TYPES = tuple(ROI_LABELS)
CONTOUR_TYPES = {"Circle", "Ellipse", "Rectangle", "Region Center", "Robust Center"}

ALL_ENABLED = set(TUNABLE_FIELDS)
APPLICABILITY = {
    "Caliper Circle": (ALL_ENABLED - {"canny_low", "canny_high", "profile_half_width_px", "profile_step_px"},
                       "卡尺圆沿径向卡尺采样，不使用 Canny 与法向短剖面"),
    "Annulus": (ALL_ENABLED - {"polarity"}, "通用闭合边缘提取无固定法向极性"),
    "Rectangular Ring": (ALL_ENABLED - {"polarity"}, "通用闭合边缘提取无固定法向极性"),
    "Circle": (ALL_ENABLED - {"polarity"}, "主轮廓法向精化按梯度幅值选峰，无固定极性"),
    "Rectangle": (ALL_ENABLED - {"polarity"}, "主轮廓法向精化按梯度幅值选峰，无固定极性"),
    "Ellipse": (ALL_ENABLED - {"polarity"}, "椭圆拟合不使用 RANSAC 阈值；极性不适用"),
    "Robust Center": (ALL_ENABLED - {"polarity"}, "稳健中心按边缘中心统计，不使用 RANSAC 阈值；极性不适用"),
    "Approximate Line": (ALL_ENABLED - {"canny_low", "canny_high"}, "近似直线沿法向剖面搜索，不使用 Canny"),
    "Region Center": (ALL_ENABLED - {"polarity", "use_ransac", "residual_limit_px"},
                      "区域中心按区域分割主轮廓计算中心；Canny/剖面参数仅影响诊断剖面"),
}


def roi_label(roi_type):
    return ROI_LABELS.get(roi_type, roi_type)


def _source_layer(main, layer):
    """Single Image mode stores only the upper image; lower ROIs reuse it."""
    mode = main._current_mode() if hasattr(main, "_current_mode") else getattr(main, "image_mode", "")
    if layer == "lower" and (mode == "Single Image" or getattr(getattr(main, "config", None), "mode", "") == "Single Image"):
        return "upper"
    return layer


def _source_image(main, mark_id, layer):
    getter = getattr(main, "_image_for_layer", None)
    if getter is not None:
        try:
            return getter(layer, mark_id)
        except TypeError:
            pass
    return main._active_image_for_layer(mark_id, _source_layer(main, layer))


def available_rois(main):
    """List metadata only; never eagerly load the whole batch."""
    choices = []
    for mark_id, mark in main.marks.items():
        for layer in ("upper", "lower"):
            src = _source_layer(main, layer)
            if main._is_batch_mode():
                present = bool(main.batch_images.get(mark_id, {}).get(src))
            else:
                present = main.mark_images.get(mark_id, {}).get(src) is not None
            if not present:
                continue
            for index, entry in enumerate(mark.roi_entries(layer), 1):
                if entry.roi.roi_type in ROI_LABELS:
                    choices.append((mark_id, layer, entry.roi_id,
                                    f"ROI {index} · {roi_label(entry.roi.roi_type)}"))
    return choices


@dataclass
class ComparisonSource:
    mark_id: str
    layer: str
    roi_id: str
    image: ImageData
    roi: Roi
    params: DetectionParams
    config: MeasurementConfig
    image_identity: ImageData
    batch_index: int

    @classmethod
    def capture(cls, main, mark_id, layer, roi_id):
        entry = main.marks[mark_id].roi_entry(layer, roi_id)
        if entry is None or entry.roi.roi_type not in ROI_LABELS:
            raise ValueError("请选择已创建的手动 ROI")
        image = _source_image(main, mark_id, layer)
        if image is None:
            raise ValueError("当前层尚未导入图像")
        # Freeze just the active image, not all batch inputs.
        pixels = np.asarray(image.gray).copy()
        pixels.setflags(write=False)
        return cls(mark_id, layer, roi_id, replace(image, gray=pixels), replace(entry.roi),
                   replace(main.params), replace(main.config), image,
                   main._batch_detail_last_single_index if main._is_batch_mode() else 0)

    @property
    def roi_type(self):
        return self.roi.roi_type

    def is_current(self, main):
        if main.operation_mode != "Engineering" or main._calculation_running:
            return False
        mark = main.marks.get(self.mark_id)
        entry = mark.roi_entry(self.layer, self.roi_id) if mark else None
        if entry is None or asdict(entry.roi) != asdict(self.roi):
            return False
        if asdict(main.params) != asdict(self.params) or asdict(main.config) != asdict(self.config):
            return False
        index = main._batch_detail_last_single_index if main._is_batch_mode() else 0
        try:
            return index == self.batch_index and _source_image(main, self.mark_id, self.layer) is self.image_identity
        except (OSError, ValueError, KeyError):
            return False


@dataclass
class ComparisonResult:
    params: DetectionParams
    records: list[dict]
    circle: CaliperCircleResult | None
    error: str = ""
    diameter_um: float | None = None
    residual_um: float | None = None
    detection: object = None
    stats: dict = field(default_factory=dict)
    roi_type: str = "Caliper Circle"
    path: str = ""
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    rejected: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    primary: tuple = ("", None, None)

    @property
    def is_caliper(self):
        return self.roi_type == "Caliper Circle"

    @property
    def ok(self):
        return self.circle is not None if self.is_caliper else self.detection is not None

    @property
    def valid_count(self):
        if self.is_caliper:
            return len(self.circle.edge_points) if self.circle is not None else 0
        return int(self.detection.edge_point_count) if self.detection is not None else 0

    @property
    def candidate_count(self):
        if self.is_caliper:
            return len(self.records)
        return int(self.stats.get("candidates", 0))

    @property
    def coverage(self):
        if self.is_caliper:
            return self.circle.angular_coverage if self.circle is not None else None
        if not self.ok or not self.candidate_count:
            return None
        return min(1.0, self.valid_count / self.candidate_count)


def _caliper_preview(source, params, cancelled):
    records = []
    try:
        circle = detect_caliper_circle(source.image.gray, source.roi, params,
                                       diagnostics=records, cancelled=cancelled)
    except InterruptedError:
        raise
    except Exception as exc:
        return ComparisonResult(params, records, None, str(exc), path="径向卡尺采样 → 卡尺圆拟合")
    diameter, residual = radial_diameter_residual_um(
        circle.edge_points, circle.center_x_px, circle.center_y_px,
        circle.radius_px, circle.residual_px, source.config)
    mode = getattr(source.roi, "diameter_mode", "Average")
    stats_um = radial_diameter_statistics_um(circle.edge_points, circle.center_x_px,
                                             circle.center_y_px, source.config)
    if mode == "Maximum":
        primary = ("最大直径 (μm)", stats_um["maximum_diameter_um"], None)
    else:
        primary = ("平均直径 (μm)", stats_um["average_diameter_um"], None)
    counts = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return ComparisonResult(params, records, circle, diameter_um=diameter, residual_um=residual,
                            stats={"candidates": len(records), "by_status": counts},
                            path="径向卡尺采样 → 卡尺圆拟合", primary=primary,
                            points=np.asarray(circle.edge_points, dtype=float).reshape(-1, 2),
                            rejected=np.asarray(circle.rejected_points, dtype=float).reshape(-1, 2))


def _trace(source, params, records, cancelled):
    """Replay the production edge extractor with diagnostics; returns path text."""
    from .line_detector import detect_approximate_line
    from .measurement_service import fitting_mode_for_roi
    from .region_center_detector import detect_primary_contour_edges, detect_region_center
    from .subpixel_edge_detector import detect_subpixel_edges, refine_contour_edges

    roi, gray = source.roi, source.image.gray
    if roi.roi_type == "Approximate Line":
        detect_approximate_line(source.mark_id, source.layer, source.image, roi, params,
                                source.config, diagnostics=records)
        return "法向灰度剖面 → 亚像素梯度峰 → 直线拟合"
    dp = replace(params, fitting_mode=fitting_mode_for_roi(roi, params, source.layer))
    if dp.fitting_mode == "RegionCenter":
        fit = detect_region_center(gray, roi, dp)
        contour = np.asarray(fit.shape_params.get("contour_points", []), dtype=np.float64).reshape(-1, 2)
        refine_contour_edges(gray, contour, dp, diagnostics=records)
        for r in records:
            if r.get("source_xy") is not None:
                x, y = r["source_xy"]
                grad = r["selected"]["gradient"] if r.get("selected") else 0.0
                r["selected"] = {"x": float(x), "y": float(y), "gradient": float(grad)}
                r["status"] = "pending_fit"
        return "区域分割主轮廓 → 区域中心（法向剖面仅供诊断）"
    if roi.roi_type in CONTOUR_TYPES:
        expected = "Circle" if dp.fitting_mode in {"Circle", "Ellipse"} else "Rectangle"
        if dp.fitting_mode in {"Auto", "EdgeCenter", "RegionCenter"}:
            expected = "Any"
        edges = detect_primary_contour_edges(gray, roi, dp, expected, diagnostics=records)
        if edges is not None and "闭合边缘筛选" in (edges.warning or ""):
            return "Canny 闭合边界筛选 → 法向亚像素插值"
        return "区域分割主轮廓 → 法向亚像素插值"
    detect_subpixel_edges(gray, roi, dp, cancelled, diagnostics=records)
    return "Canny 粗边缘 → 法向亚像素插值 → ROI/环形掩膜筛选"


def _key(x, y):
    return (round(float(np.float32(x)), 3), round(float(np.float32(y)), 3))


def _finalize(records, detection, params, roi_type):
    used = set()
    if detection is not None:
        used = {_key(x, y) for x, y in detection.edge_points}
    sp = detection.shape_params if detection is not None else {}
    mode = detection.fitting_mode if detection is not None else ""
    ransac_used = bool(params.use_ransac) and mode == "Circle"
    threshold = max(0.10, float(params.residual_limit_px)) if ransac_used else None
    radius = sp.get("radius_px") or (0.5 * detection.diameter_px if detection is not None else None)
    line = (sp.get("line_start_px"), sp.get("line_end_px"))
    for r in records:
        r["ransac_used"] = ransac_used
        r["fit_threshold_px"] = threshold
        if r.get("status") != "pending_fit":
            continue
        sel = r["selected"]
        if detection is None:
            r["status"] = "not_fitted"
            continue
        r["inlier"] = _key(sel["x"], sel["y"]) in used
        r["status"] = "valid" if r["inlier"] else "fit_rejected"
        if mode == "Line" and line[0] is not None:
            p0, p1 = np.asarray(line[0], float), np.asarray(line[1], float)
            d = p1 - p0
            n = np.hypot(*d)
            if n > 0:
                v = np.asarray((sel["x"], sel["y"]), float) - p0
                r["fit_error_px"] = abs(float(d[0] * v[1] - d[1] * v[0]) / float(n))
        elif radius:
            r["fit_error_px"] = abs(float(np.hypot(sel["x"] - detection.center_x_px,
                                                   sel["y"] - detection.center_y_px)) - float(radius))


def _thin(records, limit=MAX_RECORDS):
    if len(records) <= limit:
        kept = records
    else:
        idx = np.unique(np.linspace(0, len(records) - 1, limit).astype(int))
        kept = [records[i] for i in idx]
    for i, r in enumerate(kept):
        r["window_index"] = i
    return kept


def _primary(detection, roi):
    if detection is None:
        return ("", None, None)
    t, sp = roi.roi_type, detection.shape_params
    if t == "Ellipse":
        return ("椭圆直径 / 圆度 (μm)", detection.ellipse_diameter_um or detection.diameter_um,
                detection.ellipse_roundness_um)
    if t in {"Rectangle", "Rectangular Ring", "Region Center"}:
        return ("等效尺寸 (μm)", detection.diameter_um, None)
    if t == "Approximate Line":
        return ("线段长度 (μm) / 角度 (°)", sp.get("line_length_um", detection.diameter_um),
                sp.get("line_angle_deg", sp.get("angle_deg")))
    if t == "Robust Center":
        return ("等效直径 (μm)", detection.diameter_um, None)
    label = "最大直径 (μm)" if getattr(roi, "diameter_mode", "Average") == "Maximum" else "平均直径 (μm)"
    return (label, detection.diameter_um, None)


def calculate_preview(source: ComparisonSource, params: DetectionParams, cancelled=None):
    params = replace(params)
    if source.roi.roi_type == "Caliper Circle":
        return _caliper_preview(source, params, cancelled)
    from .measurement_service import detect_manual_roi
    records, error, detection, path = [], "", None, ""
    try:
        path = _trace(source, params, records, cancelled)
    except InterruptedError:
        raise
    except Exception:
        pass
    try:
        detection = detect_manual_roi(source.mark_id, source.layer, source.image, source.roi,
                                      params, source.config, cancelled)
    except InterruptedError:
        raise
    except Exception as exc:
        error = str(exc)
    _finalize(records, detection, params, source.roi.roi_type)
    records = [r for r in records if r.get("status") != "subsampled"]
    counts = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    stats = {"candidates": len(records), "by_status": counts}
    points = np.asarray(detection.edge_points, float).reshape(-1, 2) if detection else np.empty((0, 2))
    rejected = np.asarray([(r["selected"]["x"], r["selected"]["y"]) for r in records
                           if r["status"] == "fit_rejected"], float).reshape(-1, 2)
    return ComparisonResult(
        params, _thin(records), None, error,
        diameter_um=detection.diameter_um if detection else None,
        residual_um=detection.residual_um if detection else None,
        detection=detection, stats=stats, roi_type=source.roi.roi_type, path=path,
        points=points, rejected=rejected, primary=_primary(detection, source.roi))


def nearest_record(result: ComparisonResult, x, y):
    best, best_d = None, None
    for i, r in enumerate(result.records):
        sel = r.get("selected")
        if sel:
            px, py = sel["x"], sel["y"]
        else:
            px, py = r["start"] + r["search_vector"] * float(r["radial_samples"][-1] if len(r["radial_samples"]) else 0) * 0.5
        d = (px - x) ** 2 + (py - y) ** 2
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best


def raw_profile(source: ComparisonSource, record):
    values = []
    for distance in record["radial_samples"]:
        point = record["start"] + record["search_vector"] * float(distance)
        xy = point + record["tangent_offsets"][:, None] * record["tangent"]
        samples = [_bilinear_sample(source.image.gray, float(x), float(y)) for x, y in xy]
        finite = [value for value in samples if np.isfinite(value)]
        if not finite:
            break
        values.append(float(np.mean(finite)))
    return np.asarray(values)


def _normal_explanation(result, r):
    status = r["status"]
    g = result.params.min_gradient
    if status == "below_gradient_threshold":
        return f"Canny 粗边缘命中 → 法向梯度峰低于最小梯度 {g:g}，筛除"
    if status == "outside_image":
        return "法向剖面超出图像，无法取得完整采样"
    if status == "no_peak":
        return "法向剖面上没有可用梯度峰"
    if status == "outside_roi_mask":
        return "亚像素偏移后超出 ROI/环形掩膜，筛除"
    if status == "target_edge_filtered":
        return "被内外边界策略（目标边缘）过滤"
    sel = r.get("selected")
    if not sel:
        return result.error or "尚未完成拟合"
    text = f"法向梯度峰 {sel['gradient']:.3f} ≥ {g:g} · 已找到亚像素边缘"
    if status == "not_fitted":
        return text + "\n拟合未完成：" + result.error
    err, thr = r.get("fit_error_px"), r.get("fit_threshold_px")
    if status == "fit_rejected":
        detail = f"（偏差 {err:.4f} px ≥ {thr:g} px）" if err is not None and thr is not None else ""
        return text + "\nRANSAC 拟合残差超限剔除" + detail
    detail = f"\n拟合偏差 {err:.4f} px" if err is not None else ""
    return text + detail + "\n参与最终拟合 · 保留"


def point_explanation(result: ComparisonResult, index: int):
    if index is None or index >= len(result.records):
        return "尚无剖面数据"
    r = result.records[index]
    if r.get("kind") == "normal":
        return _normal_explanation(result, r)
    status = r["status"]
    if status == "outside_image":
        return "搜索剖面超出图像，无法取得完整采样"
    if status == "no_peak":
        return "没有找到符合边缘极性的梯度峰"
    if status == "below_gradient_threshold":
        return f"梯度峰低于最小梯度 {result.params.min_gradient:g}，未找到可用边缘"
    selected = r.get("selected")
    if not selected:
        return result.error or "尚未完成拟合"
    text = f"梯度 {selected['gradient']:.3f} ≥ {result.params.min_gradient:g} · 已找到边缘"
    if status == "not_fitted":
        return text + "\n拟合未完成：" + result.error
    if not r.get("ransac_used"):
        reason = "RANSAC 已关闭" if not result.params.use_ransac else "候选点少于 6 个，未启用 RANSAC"
        return text + "\n" + reason + "；此点参与稳健拟合"
    if not r.get("fit_model_available", True):
        return text + "\nRANSAC 未形成候选模型，按现有流程保留候选点；需检查边界"
    if r.get("fit_fallback"):
        return text + "\nRANSAC 共识点不足，已回退保留候选点；需检查边界"
    error, threshold = r.get("fit_error_px"), r.get("fit_threshold_px")
    if error is not None and threshold is not None:
        relation = "<" if r["inlier"] else "≥"
        text += f"\n筛点模型径向偏差 {error:.4f} px {relation} {threshold:g} px"
    return text + (" · 保留" if r["inlier"] else " · RANSAC 剔除")
