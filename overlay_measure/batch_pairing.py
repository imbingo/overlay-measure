from __future__ import annotations

import os
from pathlib import Path

from .models import ImageData


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
