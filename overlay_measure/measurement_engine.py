from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import numpy as np

from .auto_mark_detector import detect_auto_marks_with_report
from .batch_pairing import validate_batch_pairing
from .measurement_service import attach_algorithm_path, detect_manual_roi
from .models import DetectionParams, DetectionResult, ImageData, MarkRecipe, MeasurementConfig, OverlayResult
from .overlay_calculator import calculate_relative_overlay
from .production_measurement import refine_candidate
from .traceability import create_measurement_archive


ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


def _alpha_label(index: int) -> str:
    chars = []
    value = int(index)
    while True:
        chars.append(chr(ord("a") + value % 26))
        value = value // 26 - 1
        if value < 0:
            return "".join(reversed(chars))


def _matches_auto_rule(detection: DetectionResult, role: str, mark: MarkRecipe) -> bool:
    shape = detection.shape_params.get("shape_type", "")
    expected = mark.reference_shape if role == "reference" else mark.target_shape
    minimum = mark.reference_size_min_um if role == "reference" else mark.target_size_min_um
    maximum = mark.reference_size_max_um if role == "reference" else mark.target_size_max_um
    return (expected == "Any" or expected == shape) and minimum <= detection.diameter_um <= maximum


def _choose_auto_selection(
    detections: Dict[str, Dict[str, DetectionResult]],
    mark: MarkRecipe,
    preferred: Optional[dict],
) -> tuple[str, str]:
    preferred = preferred or {}
    references = []
    targets = []
    for label, layer_map in detections.items():
        if not layer_map:
            continue
        detection = next(iter(layer_map.values()))
        if detection.shape_params.get("quality_status") != "Valid":
            continue
        if _matches_auto_rule(detection, "reference", mark):
            references.append(label)
        if _matches_auto_rule(detection, "target", mark):
            targets.append(label)
    reference = preferred.get("reference_label", "")
    target = preferred.get("target_label", "")
    if reference not in references:
        reference = references[0] if references else ""
    if target not in targets or target == reference:
        target = next((label for label in targets if label != reference), "")
    return reference, target


def detect_auto_set(
    mark_id: str,
    mark: MarkRecipe,
    images: Iterable[tuple[str, ImageData]],
    params: DetectionParams,
    config: MeasurementConfig,
    preferred_selection: Optional[dict] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    cancelled: Optional[CancelCallback] = None,
) -> dict:
    image_pairs = list(images)
    results_all = []
    warnings = []
    search_count = max(1, len(image_pairs))
    for image_index, (layer, image) in enumerate(image_pairs):
        if cancelled and cancelled():
            raise InterruptedError("用户取消计算")
        layer_name = "上层" if layer == "upper" else "下层"
        if progress:
            progress(30.0 * image_index / search_count, f"正在搜索{layer_name}候选轮廓")
        report = detect_auto_marks_with_report(
            image.gray,
            layer,
            params,
            config.pixel_size_x_um,
            config.pixel_size_y_um,
        )
        results_all.extend(report.results)
        warning = report.warning_text()
        if warning:
            warnings.append(f"{mark_id} {layer_name}：{warning}")
        if progress:
            progress(30.0 * (image_index + 1) / search_count, f"{layer_name}候选搜索完成")

    results_all.sort(key=lambda result: -result.diameter_px)
    results_all = results_all[:32]
    candidates: Dict[str, Dict[str, DetectionResult]] = {}
    detected: Dict[str, Dict[str, DetectionResult]] = {}
    image_map = dict(image_pairs)
    candidate_count = max(1, len(results_all))
    for label_index, result in enumerate(results_all):
        if cancelled and cancelled():
            raise InterruptedError("用户取消计算")
        label = _alpha_label(label_index)
        result.mark_id = f"{mark_id}-{label}"
        candidates[label] = {result.layer: result}
        if progress:
            progress(30.0 + 60.0 * label_index / candidate_count, f"正在精测候选 {label_index + 1}/{len(results_all)}")
        try:
            measured = refine_candidate(image_map[result.layer].gray, result, params, config)
        except Exception as exc:
            measured = result
            measured.shape_params["quality_status"] = "Invalid"
            measured.shape_params["failure_reason"] = f"精测失败：{exc}"
            measured.warning = measured.shape_params["failure_reason"]
        if not (params.diameter_min_um <= measured.diameter_um <= params.diameter_max_um):
            measured.shape_params["quality_status"] = "Invalid"
            measured.shape_params["failure_reason"] = "尺寸超出配方范围"
            measured.warning = "尺寸超出配方范围"
        attach_algorithm_path(measured, "Auto")
        detected[label] = {result.layer: measured}

    reference_label, target_label = _choose_auto_selection(detected, mark, preferred_selection)
    overlay = None
    if reference_label and target_label:
        if progress:
            progress(95.0, "正在计算对位偏差")
        reference = next(iter(detected[reference_label].values()))
        target = next(iter(detected[target_label].values()))
        overlay = calculate_relative_overlay(mark_id, reference, target, config)
        if config.recipe_validation_status != "Validated" and overlay.result != "Invalid":
            overlay.result = "Trial"
            overlay.warning = "试测/未验证配方，不作正式判定"
    return {
        "candidates": candidates,
        "detections": detected,
        "selection": {"reference_label": reference_label, "target_label": target_label},
        "overlay": overlay,
        "warnings": warnings,
    }


def _manual_overlay(
    mark_id: str,
    mark: MarkRecipe,
    images: Dict[str, Optional[ImageData]],
    params: DetectionParams,
    config: MeasurementConfig,
    preferred_selection: Optional[dict],
    progress: Optional[Callable[[float, str], None]] = None,
    cancelled: Optional[CancelCallback] = None,
) -> dict:
    detections: Dict[str, DetectionResult] = {}
    tasks = []
    for layer in ("upper", "lower"):
        roi = mark.upper_roi if layer == "upper" else mark.lower_roi
        image = images.get("upper") if config.mode == "Single Image" else images.get(layer)
        if roi is not None and image is not None:
            tasks.append((layer, image, roi))
    for index, (layer, image, roi) in enumerate(tasks):
        if cancelled and cancelled():
            raise InterruptedError("用户取消计算")
        layer_name = "上层" if layer == "upper" else "下层"
        if progress:
            progress(80.0 * index / max(1, len(tasks)), f"正在分析{layer_name} ROI")
        detections[layer] = detect_manual_roi(mark_id, layer, image, roi, params, config)

    preferred_selection = preferred_selection or {}
    available = [layer for layer in ("upper", "lower") if layer in detections]
    reference = preferred_selection.get("reference_label", "")
    target = preferred_selection.get("target_label", "")
    if reference not in detections:
        reference = available[0] if available else ""
    if target not in detections or target == reference:
        target = next((layer for layer in available if layer != reference), "")
    overlay = None
    if reference and target:
        if progress:
            progress(95.0, "正在计算对位偏差")
        overlay = calculate_relative_overlay(mark_id, detections[reference], detections[target], config)
        if config.recipe_validation_status != "Validated" and overlay.result != "Invalid":
            overlay.result = "Trial"
            overlay.warning = "试测/未验证配方，不作正式判定"
    return {
        "detections": detections,
        "selection": {"reference_label": reference, "target_label": target},
        "overlay": overlay,
    }


def _run_count(job: dict, mark_id: str) -> int:
    if not job["batch"]:
        return 1 if job["mark_images"].get(mark_id, {}).get("upper") is not None else 0
    upper = job["batch_images"].get(mark_id, {}).get("upper", [])
    lower = job["batch_images"].get(mark_id, {}).get("lower", [])
    return min(len(upper), len(lower)) if job["config"].mode == "Dual Image" else len(upper)


def _mean_overlay(mark_id: str, overlays: list[OverlayResult], config: MeasurementConfig) -> OverlayResult:
    dx = float(np.mean([item.delta_x_um for item in overlays]))
    dy = float(np.mean([item.delta_y_um for item in overlays]))
    radius = float(np.hypot(dx, dy))
    warnings = []
    if abs(dx) > config.delta_x_limit_um:
        warnings.append("Dx均值超限")
    if abs(dy) > config.delta_y_limit_um:
        warnings.append("Dy均值超限")
    if radius > config.overlay_r_limit_um:
        warnings.append("Dxy均值超限")
    result = OverlayResult(mark_id, 0.0, 0.0, dx, dy, radius, "Fail" if warnings else "Pass", "；".join(warnings))
    if config.recipe_validation_status != "Validated":
        result.result = "Trial"
        result.warning = "试测/未验证配方，不作正式判定"
    return result


def _terminal_overlay(mark_id: str, result: str, warning: str) -> OverlayResult:
    return OverlayResult(mark_id, 0.0, 0.0, 0.0, 0.0, 0.0, result, warning)


def run_measurement_job(job: dict, progress: ProgressCallback, cancelled: CancelCallback) -> dict:
    config: MeasurementConfig = job["config"]
    params: DetectionParams = job["params"]
    marks: Dict[str, MarkRecipe] = job["marks"]
    selections = job.get("selections", {})
    is_auto = config.workflow_mode == "Auto"
    is_batch = bool(job.get("batch"))
    if is_batch:
        pairing_errors = validate_batch_pairing(job["batch_images"], config.mode == "Dual Image")
        if pairing_errors:
            raise ValueError("批量图像配对检查失败：\n" + "\n".join(pairing_errors))
    total = max(1, sum(_run_count(job, mark_id) for mark_id in ("Mark1", "Mark2")))
    done = 0
    payload = {
        "batch": is_batch,
        "detections": {}, "overlays": {},
        "auto_candidates": {"Mark1": {}, "Mark2": {}},
        "auto_detections": {"Mark1": {}, "Mark2": {}},
        "auto_overlays": {},
        "selections": {mark_id: dict(selections.get(mark_id, {})) for mark_id in ("Mark1", "Mark2")},
        "batch_overlays": {"Mark1": [], "Mark2": []},
        "batch_records": {"Mark1": [], "Mark2": []},
        "skipped": [], "warnings": [],
    }
    for mark_id in ("Mark1", "Mark2"):
        count = _run_count(job, mark_id)
        for run_index in range(count):
            if cancelled():
                raise InterruptedError("用户取消计算")
            if is_batch:
                upper = job["batch_images"][mark_id]["upper"][run_index]
                lower = job["batch_images"][mark_id]["lower"][run_index] if config.mode == "Dual Image" else None
            else:
                upper = job["mark_images"][mark_id].get("upper")
                lower = job["mark_images"][mark_id].get("lower")
            images = {"upper": upper, "lower": lower}
            prefix = f"{mark_id} 第{run_index + 1}/{count}次" if is_batch else mark_id
            record = {
                "run_index": run_index + 1,
                "upper_file": upper.path if upper else "",
                "lower_file": lower.path if lower else "",
                "overlay": None,
                "error": "",
            }
            try:
                stage = lambda percent, message: progress(
                    done * 100 + int(max(0.0, min(99.0, percent))), total * 100, f"{prefix}：{message}"
                )
                if is_auto:
                    pairs = [("upper", upper)]
                    if config.mode == "Dual Image":
                        pairs.append(("lower", lower))
                    if any(image is None for _, image in pairs):
                        raise ValueError("缺少当前测量模式需要的图像")
                    measured = detect_auto_set(mark_id, marks[mark_id], pairs, params, config, selections.get(mark_id), stage, cancelled)
                    payload["auto_candidates"][mark_id] = measured["candidates"]
                    payload["auto_detections"][mark_id] = measured["detections"]
                    payload["selections"][mark_id] = measured["selection"]
                    payload["warnings"].extend(measured["warnings"])
                    overlay = measured["overlay"]
                else:
                    measured = _manual_overlay(mark_id, marks[mark_id], images, params, config, selections.get(mark_id), stage, cancelled)
                    payload["detections"][mark_id] = measured["detections"]
                    payload["selections"][mark_id] = measured["selection"]
                    overlay = measured["overlay"]
                if overlay is None:
                    raise ValueError("未选择到两个有效轮廓，未生成对位结果")
                record["overlay"] = overlay
                if is_batch and overlay.result != "Invalid":
                    payload["batch_overlays"][mark_id].append(overlay)
                else:
                    (payload["auto_overlays"] if is_auto else payload["overlays"])[mark_id] = overlay
            except InterruptedError:
                raise
            except Exception as exc:
                record["error"] = str(exc)
                payload["skipped"].append(f"{prefix}：{exc}")
                if not is_batch:
                    target = payload["auto_overlays"] if is_auto else payload["overlays"]
                    target[mark_id] = _terminal_overlay(mark_id, "Error", str(exc))
            if is_batch:
                payload["batch_records"][mark_id].append(record)
            done += 1
            progress(done * 100, total * 100, f"{prefix}：完成")
        if is_batch and payload["batch_overlays"][mark_id]:
            target = payload["auto_overlays"] if is_auto else payload["overlays"]
            target[mark_id] = _mean_overlay(mark_id, payload["batch_overlays"][mark_id], config)
        elif is_batch and payload["batch_records"][mark_id]:
            target = payload["auto_overlays"] if is_auto else payload["overlays"]
            invalid = [
                item["overlay"] for item in payload["batch_records"][mark_id]
                if item.get("overlay") is not None and item["overlay"].result == "Invalid"
            ]
            if invalid:
                warning = "；".join(dict.fromkeys(item.warning or "识别质量无效" for item in invalid))
                target[mark_id] = _terminal_overlay(mark_id, "Invalid", warning)
            else:
                errors = [item.get("error", "") for item in payload["batch_records"][mark_id] if item.get("error")]
                target[mark_id] = _terminal_overlay(mark_id, "Error", "；".join(dict.fromkeys(errors)))
    traceability = job.get("traceability") or {}
    if traceability:
        try:
            progress(total * 100, total * 100, "正在生成追溯档案")
            result_map = payload["auto_overlays"] if is_auto else payload["overlays"]
            measurement_id, archive_path = create_measurement_archive(
                config,
                params,
                traceability.get("recipe_path", ""),
                traceability.get("recipe_hash", ""),
                traceability.get("input_paths", []),
                result_map,
                payload["batch_records"],
                traceability.get("operation_mode", ""),
            )
            payload["measurement_id"] = measurement_id
            payload["archive_path"] = str(archive_path)
        except Exception as exc:
            payload["archive_error"] = str(exc)
    return payload
