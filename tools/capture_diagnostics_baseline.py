"""Capture immutable pre-Diagnostics measurements; never run to fix regressions."""

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from overlay_measure.caliper_circle_detector import detect_caliper_circle
from overlay_measure.models import DetectionParams, ImageData, MeasurementConfig, Roi
from overlay_measure.measurement_service import detect_manual_roi
from overlay_measure.production_measurement import refine_circle_candidate
from overlay_measure.overlay_calculator import calculate_overlay
from overlay_measure.result_exporter import build_detection_rows


def inputs():
    y, x = np.mgrid[:160, :180]
    r = np.hypot(x - 88.3, y - 77.7)
    gray = (
        1000
        + 50000 / (1 + np.exp(np.clip(-(r - 45.2) / 1.1, -60, 60)))
        + np.random.default_rng(402).normal(0, 15, r.shape)
    ).astype(np.float32)
    return (
        gray,
        Roi(
            35,
            24,
            108,
            108,
            "Caliper Circle",
            0.65,
            "Strongest Edge",
            caliper_count=48,
            caliper_width_px=4,
        ),
        DetectionParams(min_gradient=20),
        MeasurementConfig(),
    )


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def capture():
    gray, roi, params, config = inputs()
    image = ImageData("synthetic-v248", gray, "baseline", "uint16")
    cal = detect_caliper_circle(gray, roi, params)
    manual = detect_manual_roi("baseline", "upper", image, roi, params, config)
    auto = refine_circle_candidate(gray, manual, params, config)
    other = detect_manual_roi("baseline", "lower", image, roi, params, config)
    overlay = calculate_overlay("baseline", manual, other, config)
    return dict(
        source_commit="d4f7a3e70999f0f99a6207a40f8d047bf7b8eb9e",
        input_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),
        roi=asdict(roi),
        params=asdict(params),
        config=asdict(config),
        caliper=asdict(cal),
        manual=asdict(manual),
        production=asdict(auto),
        overlay=asdict(overlay),
        export_rows=build_detection_rows(
            {"baseline": {"upper": manual, "lower": other}},
            {"baseline": overlay},
            config,
        ),
    )


if __name__ == "__main__":
    target = ROOT / "tests/data/diagnostics_v248_baseline.json"
    if target.exists():
        raise SystemExit("Baseline exists; refusing to overwrite")
    target.parent.mkdir(exist_ok=True)
    target.write_text(
        json.dumps(capture(), ensure_ascii=False, indent=2, default=serialize),
        encoding="utf-8",
    )
    print(target)
