from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .models import ImageData


@dataclass(frozen=True)
class BatchPairingSummary:
    """Small, presentation-neutral summary for one Mark's batch inputs."""

    pairing_text: str
    status_text: str
    ready: bool


def summarize_batch_pairing(upper_count: int, lower_count: int, dual_image: bool) -> BatchPairingSummary:
    """Describe whether a Mark's imported batch images are runnable.

    Pairing remains positional and is validated separately before a run.  This
    helper only gives the operator a compact, unambiguous table summary.
    """
    if not upper_count and not lower_count:
        return BatchPairingSummary("—", "未导入", False)
    if not dual_image:
        if upper_count:
            return BatchPairingSummary("单图，无需配对", "就绪", True)
        return BatchPairingSummary("—", "缺少单图", False)
    if not upper_count:
        return BatchPairingSummary("缺少上层", "需检查", False)
    if not lower_count:
        return BatchPairingSummary("缺少下层", "需检查", False)
    if upper_count != lower_count:
        return BatchPairingSummary(f"{abs(upper_count - lower_count)}张未配对", "需检查", False)
    return BatchPairingSummary(f"{upper_count}组已配对", "就绪", True)


def validate_batch_pairing(
    batch_images: dict[str, dict[str, list[ImageData]]],
    dual_image: bool,
) -> list[str]:
    errors: list[str] = []
    for mark_id in ("Mark1", "Mark2"):
        layers = batch_images.get(mark_id, {})
        upper = list(layers.get("upper", []))
        lower = list(layers.get("lower", []))
        if not upper and not lower:
            continue
        if not upper:
            errors.append(f"{mark_id} 缺少上层图像")
            continue
        if dual_image and len(upper) != len(lower):
            errors.append(f"{mark_id} 上下层数量不一致：上层 {len(upper)} 张，下层 {len(lower)} 张")
            continue
        if dual_image:
            for index, (upper_image, lower_image) in enumerate(zip(upper, lower), start=1):
                upper_path = os.path.normcase(str(Path(upper_image.path).resolve()))
                lower_path = os.path.normcase(str(Path(lower_image.path).resolve()))
                if upper_path == lower_path:
                    errors.append(f"{mark_id} 第 {index} 组上下层使用了同一个文件")
    return errors
