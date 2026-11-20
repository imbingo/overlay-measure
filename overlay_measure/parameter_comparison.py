"""Frozen single-ROI previews using the production caliper-circle detector."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import numpy as np

from .caliper_circle_detector import CaliperCircleResult, detect_caliper_circle
from .measurement_units import radial_diameter_residual_um
from .models import DetectionParams, ImageData, MeasurementConfig, Roi
from .subpixel_edge_detector import _bilinear_sample

TUNABLE_FIELDS = ("gaussian_sigma_px", "min_gradient", "residual_limit_px", "use_ransac")


def available_rois(main):
    """List metadata only; never eagerly load the whole batch."""
    choices = []
    for mark_id, mark in main.marks.items():
        for layer in ("upper", "lower"):
            if main._is_batch_mode():
                present = bool(main.batch_images.get(mark_id, {}).get(layer))
            else:
                present = main.mark_images.get(mark_id, {}).get(layer) is not None
            if not present:
                continue
            for index, entry in enumerate(mark.roi_entries(layer), 1):
                if entry.roi.roi_type == "Caliper Circle":
                    choices.append((mark_id, layer, entry.roi_id, f"ROI {index} · 卡尺圆"))
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
        if entry is None or entry.roi.roi_type != "Caliper Circle":
            raise ValueError("请选择已创建的卡尺圆 ROI")
        image = main._active_image_for_layer(mark_id, layer)
        if image is None:
            raise ValueError("当前层尚未导入图像")
        # Freeze just the active image, not all batch inputs.
        pixels = np.asarray(image.gray).copy()
        pixels.setflags(write=False)
        return cls(mark_id, layer, roi_id, replace(image, gray=pixels), replace(entry.roi),
                   replace(main.params), replace(main.config), image,
                   main._batch_detail_last_single_index if main._is_batch_mode() else 0)

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
            return index == self.batch_index and main._active_image_for_layer(self.mark_id, self.layer) is self.image_identity
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

    @property
    def valid_count(self):
        return len(self.circle.edge_points) if self.circle is not None else 0

    @property
    def coverage(self):
        return self.circle.angular_coverage if self.circle is not None else None


def calculate_preview(source: ComparisonSource, params: DetectionParams, cancelled=None):
    records = []
    params = replace(params)
    try:
        circle = detect_caliper_circle(source.image.gray, source.roi, params,
                                       diagnostics=records, cancelled=cancelled)
    except InterruptedError:
        raise
    except Exception as exc:
        return ComparisonResult(params, records, None, str(exc))
    diameter, residual = radial_diameter_residual_um(
        circle.edge_points, circle.center_x_px, circle.center_y_px,
        circle.radius_px, circle.residual_px, source.config)
    return ComparisonResult(params, records, circle, diameter_um=diameter, residual_um=residual)


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


def point_explanation(result: ComparisonResult, index: int):
    if index >= len(result.records):
        return "尚无剖面数据"
    r = result.records[index]
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
