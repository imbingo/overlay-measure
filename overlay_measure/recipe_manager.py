from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .models import DetectionParams, MarkRecipe, MeasurementConfig, Roi


def _roi_to_dict(roi):
    return None if roi is None else asdict(roi)


def _roi_from_dict(data):
    if not data:
        return None
    return Roi(**data)


def save_recipe(path: str, config: MeasurementConfig, params: DetectionParams, marks: List[MarkRecipe]) -> None:
    data = {
        "software_name": "Overlay Mark Measurement Software",
        "version": "1.5.5",
        "measurement_config": asdict(config),
        "detection_params": asdict(params),
        "marks": [
            {
                "mark_id": m.mark_id,
                "upper_roi": _roi_to_dict(m.upper_roi),
                "lower_roi": _roi_to_dict(m.lower_roi),
                "reference_shape": m.reference_shape,
                "target_shape": m.target_shape,
                "reference_size_min_um": m.reference_size_min_um,
                "reference_size_max_um": m.reference_size_max_um,
                "target_size_min_um": m.target_size_min_um,
                "target_size_max_um": m.target_size_max_um,
            }
            for m in marks
        ],
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_recipe(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    config = MeasurementConfig(**data.get("measurement_config", {}))
    params_data = data.get("detection_params", {})
    if "upper_fitting_mode" not in params_data and "fitting_mode" in params_data:
        params_data["upper_fitting_mode"] = params_data["fitting_mode"]
    if "lower_fitting_mode" not in params_data and "fitting_mode" in params_data:
        params_data["lower_fitting_mode"] = params_data["fitting_mode"]
    params = DetectionParams(**params_data)
    marks = []
    for item in data.get("marks", []):
        marks.append(
            MarkRecipe(
                mark_id=item.get("mark_id", f"Mark{len(marks)+1}"),
                upper_roi=_roi_from_dict(item.get("upper_roi")),
                lower_roi=_roi_from_dict(item.get("lower_roi")),
                reference_shape=item.get("reference_shape", "Any"),
                target_shape=item.get("target_shape", "Any"),
                reference_size_min_um=float(item.get("reference_size_min_um", 0.0)),
                reference_size_max_um=float(item.get("reference_size_max_um", 999999.0)),
                target_size_min_um=float(item.get("target_size_min_um", 0.0)),
                target_size_max_um=float(item.get("target_size_max_um", 999999.0)),
            )
        )
    return config, params, marks
