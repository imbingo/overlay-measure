from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import numpy as np

from .auto_mark_detector import detect_auto_marks_with_report
from .batch_image_store import image_path, resolve_image
from .batch_results import compact_detection_map
from .batch_pairing import validate_batch_pairing
from .candidate_ordering import assign_spatial_candidate_ids, resolve_preferred_candidate
from .geometry_engine import execute_geometry_program
from .geometry_models import GeometryProgram, GeometryRunResult
from .measurement_service import attach_algorithm_path, detect_manual_roi
from .models import DetectionParams, DetectionResult, ImageData, MarkRecipe, MeasurementConfig, OverlayResult
from .overlay_calculator import calculate_relative_overlay
from .production_measurement import refine_candidate
from .quality_profiles import annotate_detection_quality, quality_profile_display
from .traceability import create_measurement_archive


ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


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
        if not detection.shape_params.get(
            "measurement_eligible",
            detection.shape_params.get("quality_status") == "Valid",
        ):
            continue
        if _matches_auto_rule(detection, "reference", mark):
            references.append(label)
        if _matches_auto_rule(detection, "target", mark):
            targets.append(label)
    reference = resolve_preferred_candidate(preferred.get("reference_label", ""), detections)
    target = resolve_preferred_candidate(preferred.get("target_label", ""), detections)
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
    # Do not silently discard half of a regular hole array here.  Each image
    # already applies the recipe's explicit candidate limit and reports any
    # truncation.  A shared cap would also let the first layer crowd out the
    # second one in dual-image mode.
    candidates: Dict[str, Dict[str, DetectionResult]] = {}
    detected: Dict[str, Dict[str, DetectionResult]] = {}
    image_map = dict(image_pairs)
    candidate_count = max(1, len(results_all))
    spatial_entries = assign_spatial_candidate_ids(results_all, config.mode == "Dual Image")
    for label_index, (label, result) in enumerate(spatial_entries):
        if cancelled and cancelled():
            raise InterruptedError("用户取消计算")
        result.mark_id = f"{mark_id}-{label}"
        candidates[label] = {result.layer: result}
        if progress:
            progress(30.0 + 60.0 * label_index / candidate_count, f"正在精测候选 {label_index + 1}/{len(results_all)}")
        try:
            measured = refine_candidate(image_map[result.layer].gray, result, params, config)
        except Exception as exc:
            measured = result
            measured.shape_params["quality_hard_failure"] = True
            measured.shape_params["measurement_eligible"] = False
            measured.shape_params["candidate_status"] = "DiagnosticOnly"
            measured.shape_params["failure_reason"] = f"精测失败：{exc}"
            measured.warning = measured.shape_params["failure_reason"]
        if not (params.diameter_min_um <= measured.diameter_um <= params.diameter_max_um):
            measured.shape_params["quality_hard_failure"] = True
            measured.shape_params["failure_reason"] = "尺寸超出配方范围"
            measured.warning = "尺寸超出配方范围"
        annotate_detection_quality(measured, config)
        measured.shape_params["measurement_eligible"] = (
            measured.shape_params.get("quality_status") == "Valid"
            and not measured.shape_params.get("quality_hard_failure", False)
        )
        measured.shape_params.setdefault(
            "candidate_status",
            "MeasurementEligible" if measured.shape_params["measurement_eligible"] else "DiagnosticOnly",
        )
        attach_algorithm_path(measured, "Auto")
        detected[label] = {result.layer: measured}

    for role_key, role_name in (("reference_label", "基准"), ("target_label", "待测")):
        legacy_value = str((preferred_selection or {}).get(role_key, "")).strip()
        if legacy_value and not resolve_preferred_candidate(legacy_value, detected):
            warnings.append(f"{mark_id}：旧配方{role_name}候选 {legacy_value} 无法映射，已按当前形状和尺寸规则回退")
    reference_label, target_label = _choose_auto_selection(detected, mark, preferred_selection)
    overlay = None
    if reference_label and target_label:
        if progress:
            progress(95.0, "正在计算对位偏差")
        reference = next(iter(detected[reference_label].values()))
        target = next(iter(detected[target_label].values()))
        overlay = calculate_relative_overlay(mark_id, reference, target, config)
        overlay.reference_contour_id = reference_label
        overlay.target_contour_id = target_label
        overlay.reference_contour_name = reference_label
        overlay.target_contour_name = target_label
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
    failures: list[dict] = []
    tasks = []
    for layer in ("upper", "lower"):
        image = images.get("upper") if config.mode == "Single Image" else images.get(layer)
        if image is None:
            continue
        for roi_index, entry in enumerate(mark.roi_entries(layer), start=1):
            tasks.append((layer, image, entry, roi_index))
    for index, (layer, image, entry, roi_index) in enumerate(tasks):
        if cancelled and cancelled():
            raise InterruptedError("用户取消计算")
        layer_name = "上层" if layer == "upper" else "下层"
        if progress:
            progress(80.0 * index / max(1, len(tasks)), f"正在分析{layer_name} ROI")
        try:
            detection = detect_manual_roi(
                mark_id, layer, image, entry.roi, params, config, cancelled
            )
            detection.shape_params["roi_id"] = entry.roi_id
            detection.shape_params["roi_index"] = roi_index
            detection.shape_params["roi_label"] = f"ROI {roi_index}"
            detections[entry.roi_id] = detection
        except InterruptedError:
            raise
        except Exception as exc:
            failures.append({
                "roi_id": entry.roi_id,
                "roi_index": roi_index,
                "roi_label": f"ROI {roi_index}",
                "layer": layer,
                "status": "Error",
                "error": str(exc),
            })

    preferred_selection = preferred_selection or {}
    available = list(detections)

    def resolve_legacy(value: str) -> str:
        if value in detections:
            return value
        if value in {"upper", "lower"}:
            return next(
                (roi_id for roi_id, detection in detections.items() if detection.layer == value),
                "",
            )
        return ""

    requested_reference = str(preferred_selection.get("reference_label", "") or "")
    requested_target = str(preferred_selection.get("target_label", "") or "")
    reference = resolve_legacy(requested_reference)
    target = resolve_legacy(requested_target)
    if not requested_reference:
        reference = next(
            (roi_id for roi_id, detection in detections.items() if detection.layer == "upper"),
            available[0] if available else "",
        )
    if not requested_target:
        target = next(
            (
                roi_id
                for roi_id, detection in detections.items()
                if detection.layer == "lower" and roi_id != reference
            ),
            next((roi_id for roi_id in available if roi_id != reference), ""),
        )
    if target == reference:
        target = ""
    overlay = None
    if reference and target:
        if progress:
            progress(95.0, "正在计算对位偏差")
        overlay = calculate_relative_overlay(mark_id, detections[reference], detections[target], config)
        overlay.reference_contour_id = reference
        overlay.target_contour_id = target
        overlay.reference_contour_name = detections[reference].shape_params.get("roi_label", reference)
        overlay.target_contour_name = detections[target].shape_params.get("roi_label", target)
        if config.recipe_validation_status != "Validated" and overlay.result != "Invalid":
            overlay.result = "Trial"
            overlay.warning = "试测/未验证配方，不作正式判定"
    return {
        "detections": detections,
        "selection": {"reference_label": reference, "target_label": target},
        "overlay": overlay,
        "failures": failures,
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
    first = overlays[0]
    result.reference_contour_id = first.reference_contour_id
    result.target_contour_id = first.target_contour_id
    result.reference_contour_name = first.reference_contour_name
    result.target_contour_name = first.target_contour_name
    result.quality_profile = quality_profile_display(config)
    grade_order = {
        "优秀（满足超精确）": 0,
        "合格（满足标准）": 1,
        "较差（仅宽容可用）": 2,
        "无效（低于宽容门槛）": 3,
    }
    grades = [item.quality_grade for item in overlays if item.quality_grade]
    if grades:
        result.quality_grade = max(grades, key=lambda grade: grade_order.get(grade, 99))
    result.quality_summary = f"{len(overlays)} 次有效测量；最差质量={result.quality_grade or '未评估'}"
    if config.recipe_validation_status != "Validated":
        result.result = "Trial"
        result.warning = "试测/未验证配方，不作正式判定"
    return result


def _terminal_overlay(mark_id: str, result: str, warning: str) -> OverlayResult:
    return OverlayResult(mark_id, 0.0, 0.0, 0.0, 0.0, 0.0, result, warning)


def run_preview_job(job: dict, progress: ProgressCallback, cancelled: CancelCallback) -> dict:
    """Run auto-search or manual ROI preview without touching Qt/UI state."""
    kind = str(job["kind"])
    mark_id = str(job["mark_id"])
    config: MeasurementConfig = job["config"]
    params: DetectionParams = job["params"]
    mark: MarkRecipe = job["mark"]
    images = {
        layer: resolve_image(value)
        for layer, value in (job.get("images") or {}).items()
    }
    stage = lambda percent, message: progress(int(percent), 100, message)
    if kind == "auto":
        pairs = [("upper", images.get("upper"))]
        if config.mode == "Dual Image":
            pairs.append(("lower", images.get("lower")))
        if any(image is None for _, image in pairs):
            raise ValueError("缺少当前模式需要的图像")
        measured = detect_auto_set(
            mark_id,
            mark,
            pairs,
            params,
            config,
            job.get("selection"),
            lambda percent, message: stage(percent, message),
            cancelled,
        )
    elif kind == "manual":
        measured = _manual_overlay(
            mark_id,
            mark,
            images,
            params,
            config,
            job.get("selection"),
            lambda percent, message: stage(percent, message),
            cancelled,
        )
    else:
        raise ValueError(f"未知预览任务：{kind}")
    progress(100, 100, "预览分析完成")
    return {"kind": kind, "mark_id": mark_id, "measured": measured}


def run_measurement_job(job: dict, progress: ProgressCallback, cancelled: CancelCallback) -> dict:
    config: MeasurementConfig = job["config"]
    params: DetectionParams = job["params"]
    marks: Dict[str, MarkRecipe] = job["marks"]
    selections = job.get("selections", {})
    is_auto = config.workflow_mode == "Auto"
    is_batch = bool(job.get("batch"))
    geometry_program: GeometryProgram = job.get("geometry_program") or GeometryProgram()
    geometry_configured = bool(
        geometry_program.features
        or geometry_program.coordinate_systems
        or geometry_program.measurements
        or geometry_program.coordinate_labels
    )
    geometry_detection_runs: Dict[int, Dict[str, DetectionResult]] = {}
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
        "roi_failures": {"Mark1": [], "Mark2": []},
        "geometry_result": GeometryRunResult(),
        "batch_geometry_results": [],
        "skipped": [], "warnings": [],
    }
    for mark_id in ("Mark1", "Mark2"):
        count = _run_count(job, mark_id)
        for run_index in range(count):
            if cancelled():
                raise InterruptedError("用户取消计算")
            if is_batch:
                upper_source = job["batch_images"][mark_id]["upper"][run_index]
                lower_source = job["batch_images"][mark_id]["lower"][run_index] if config.mode == "Dual Image" else None
            else:
                upper_source = job["mark_images"][mark_id].get("upper")
                lower_source = job["mark_images"][mark_id].get("lower")
            upper_file = image_path(upper_source)
            lower_file = image_path(lower_source)
            upper = resolve_image(upper_source)
            lower = resolve_image(lower_source)
            images = {"upper": upper, "lower": lower}
            prefix = f"{mark_id} 第{run_index + 1}/{count}次" if is_batch else mark_id
            record = {
                "run_index": run_index + 1,
                "upper_file": upper_file,
                "lower_file": lower_file,
                "overlay": None,
                "error": "",
                "workflow": "Auto" if is_auto else "Manual",
                "detections": {},
                "candidates": {},
                "selection": {},
                "roi_failures": [],
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
                    record["detections"] = compact_detection_map(measured["detections"])
                    record["candidates"] = compact_detection_map(measured["candidates"])
                    record["selection"] = dict(measured["selection"])
                    run_geometry_detections = geometry_detection_runs.setdefault(run_index, {})
                    for label, layer_map in measured["detections"].items():
                        for layer, detection in layer_map.items():
                            run_geometry_detections[f"{mark_id}/{label}:{layer}"] = detection
                else:
                    measured = _manual_overlay(mark_id, marks[mark_id], images, params, config, selections.get(mark_id), stage, cancelled)
                    payload["detections"][mark_id] = measured["detections"]
                    payload["roi_failures"][mark_id] = list(measured.get("failures", []))
                    payload["selections"][mark_id] = measured["selection"]
                    overlay = measured["overlay"]
                    record["detections"] = compact_detection_map(measured["detections"])
                    record["selection"] = dict(measured["selection"])
                    record["roi_failures"] = list(measured.get("failures", []))
                    for failure in record["roi_failures"]:
                        payload["skipped"].append(
                            f"{prefix} {failure['layer']} {failure['roi_label']}：{failure['error']}"
                        )
                    run_geometry_detections = geometry_detection_runs.setdefault(run_index, {})
                    for roi_id, detection in measured["detections"].items():
                        run_geometry_detections[f"{mark_id}/{roi_id}:{detection.layer}"] = detection
                if overlay is None:
                    if geometry_configured:
                        record["overlay_skipped"] = "未配置成对的基准/待测轮廓，已跳过对位偏差"
                    elif record["roi_failures"] and record["detections"]:
                        raise ValueError("部分 ROI 识别失败，且未选择到两个有效轮廓")
                    else:
                        raise ValueError("未选择到两个有效轮廓，未生成对位结果")
                else:
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
        elif is_batch and any(
            record.get("overlay") is not None for record in payload["batch_records"][mark_id]
        ):
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
    if geometry_configured:
        geometry_runs = []
        run_indexes = sorted(geometry_detection_runs) or [0]
        for index in run_indexes:
            geometry_runs.append(
                execute_geometry_program(geometry_program, geometry_detection_runs.get(index, {}), config)
            )
        payload["geometry_result"] = geometry_runs[0]
        payload["batch_geometry_results"] = geometry_runs if is_batch else []
        if is_batch:
            for records in payload["batch_records"].values():
                for record in records:
                    index = int(record.get("run_index", 1)) - 1
                    if 0 <= index < len(geometry_runs):
                        record["geometry_result"] = geometry_runs[index].to_dict()

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
