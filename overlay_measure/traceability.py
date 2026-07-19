from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from . import __version__
from .models import DetectionParams, MeasurementConfig, OverlayResult
from .recipe_integrity import file_sha256
from .runtime_support import app_data_root


def _input_record(path: str) -> dict:
    file_path = Path(path)
    record = {"path": str(file_path), "name": file_path.name, "sha256": "", "size": None}
    if file_path.exists() and file_path.is_file():
        record["size"] = file_path.stat().st_size
        record["sha256"] = file_sha256(file_path)
    return record


def create_measurement_archive(
    config: MeasurementConfig,
    params: DetectionParams,
    recipe_path: str,
    recipe_hash: str,
    input_paths: list[str],
    overlays: dict[str, OverlayResult],
    batch_records: dict[str, list[dict]],
    operation_mode: str,
) -> tuple[str, Path]:
    now = datetime.now()
    measurement_id = f"M-{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8].upper()}"
    directory = app_data_root() / "records" / now.strftime("%Y-%m-%d") / measurement_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "measurement_id": measurement_id,
        "created_at": now.isoformat(timespec="seconds"),
        "software_version": __version__,
        "operation_mode": operation_mode,
        "recipe": {
            "path": recipe_path,
            "sha256": recipe_hash,
            "name": config.recipe_name,
            "version": config.recipe_version,
            "validation_status": config.recipe_validation_status,
        },
        "config_snapshot": asdict(config),
        "detection_params_snapshot": asdict(params),
        "inputs": [_input_record(path) for path in dict.fromkeys(path for path in input_paths if path)],
        "results": {mark_id: asdict(result) for mark_id, result in overlays.items()},
        "batch_runs": {
            mark_id: [
                {
                    "run_index": item.get("run_index"),
                    "upper_file": item.get("upper_file", ""),
                    "lower_file": item.get("lower_file", ""),
                    "result": asdict(item["overlay"]) if item.get("overlay") else None,
                    "error": item.get("error", ""),
                }
                for item in records
            ]
            for mark_id, records in batch_records.items()
        },
    }
    (directory / "measurement.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    index_path = app_data_root() / "records" / "measurement_index.jsonl"
    with index_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "measurement_id": measurement_id,
            "created_at": manifest["created_at"],
            "recipe_name": config.recipe_name,
            "material_code": config.material_code,
            "archive": str(directory),
        }, ensure_ascii=False) + "\n")
    return measurement_id, directory
