from __future__ import annotations

import sys
import re
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QPoint, QPointF, QRectF, QThread, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QFontDatabase, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGridLayout,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStyle,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from .auto_mark_detector import detect_auto_marks_with_report
from .access_control import AccessController
from .batch_pairing import validate_batch_pairing
from .batch_image_store import BatchImageRef, resolve_image
from .candidate_ordering import assign_spatial_candidate_ids, candidate_display_label, resolve_preferred_candidate
from .export_naming import build_export_filename
from .export_visualization import render_measurement_image
from .image_loader import SUPPORTED_EXTENSIONS, display_to_uint8, load_image
from .measurement_engine import run_measurement_job
from .geometry_models import GeometryProgram, GeometryRunResult
from .measurement_service import attach_algorithm_path, describe_algorithm_path, detect_manual_roi
from .measurement_units import axis_scale_um_per_px, rotated_rect_size_um
from .models import DetectionParams, DetectionResult, ImageData, MarkRecipe, MeasurementConfig, OverlayResult, Roi
from .overlay_calculator import calculate_overlay, calculate_relative_overlay
from .production_measurement import refine_candidate
from .quality_profiles import (
    QUALITY_PROFILE_LABELS,
    annotate_detection_quality,
    apply_quality_profile,
    quality_profile_display,
    quality_profile_is_modified,
)
from .recipe_manager import load_recipe, save_recipe
from .recipe_library import RecipeLibrary, RecipeLibraryEntry
from .recipe_integrity import seal_recipe, verify_recipe
from .result_exporter import build_detection_failure_row, build_detection_rows, build_geometry_rows, export_results
from .rz_calculator import build_summary_rows
from .runtime_support import RecoveryStore, build_runtime_logger

from .ui_constants import LAYER_LABELS, RESULT_LABELS, STEP_TITLES
from .ui_components import ImageDiagnosticsDialog, ImageViewerDialog, MultiFolderSelectionDialog
from .ui_components import (
    CollapsibleSection,
    FramelessTitleBar,
    ImageCanvas,
    RepeatabilityPlot,
    SidebarComboBox,
    SidebarDoubleSpinBox,
    SidebarSpinBox,
)
from .ui_recipe_views import RecipeLibraryDialog, RecipeQuickMenu
from .ui_workers import MeasurementWorker, PreviewWorker


class MainWindowWorkflowMixin:
        def open_image_viewer(self):
            sources = [self.upper_canvas]
            if self._current_mode() == "Dual Image" and self.lower_canvas.image is not None:
                sources.append(self.lower_canvas)
            if not any(canvas.image is not None for canvas in sources):
                QMessageBox.information(self, "图像查看器", "请先导入图像。")
                return
            dialog = getattr(self, "_image_viewer_dialog", None)
            if dialog is not None and dialog.isVisible():
                dialog.raise_()
                dialog.activateWindow()
                return
            self._image_viewer_dialog = ImageViewerDialog(sources, self)
            self._image_viewer_dialog.setAttribute(Qt.WA_DeleteOnClose)
            self._image_viewer_dialog.destroyed.connect(
                lambda *_: setattr(self, "_image_viewer_dialog", None)
            )
            self._image_viewer_dialog.show()

        def refresh_recent_images_menu(self):
            menu = self.recent_images_menu
            menu.clear()
            entries = self.recent_image_store.entries(existing_only=True)
            if not entries:
                empty = menu.addAction("暂无可用最近文件")
                empty.setEnabled(False)
                return
            for entry in entries:
                path = Path(str(entry["path"]))
                layer = str(entry.get("layer", "upper"))
                layer_name = "下层" if layer == "lower" else "上层/单图"
                action = menu.addAction(f"{layer_name}  ·  {path.parent.name} / {path.name}")
                action.setToolTip(str(path))
                action.triggered.connect(
                    lambda checked=False, p=str(path), target_layer=layer: self.import_recent_image(p, target_layer)
                )
            menu.addSeparator()
            clear_action = menu.addAction("清除最近文件记录")
            clear_action.triggered.connect(self.clear_recent_images)

        def clear_recent_images(self):
            self.recent_image_store.clear()
            self.show_canvas_interaction_message("已清除最近文件记录。")

        def import_recent_image(self, path: str, layer: str):
            if not Path(path).is_file():
                self.show_canvas_interaction_message("最近文件已不存在，已跳过。")
                return
            self._import_image_path(path, layer, "已从最近文件导入")

        def _import_image_path(self, path: str, layer: str, success_prefix: str) -> bool:
            if self._calculation_running:
                self._append_log("正在计算，请等待结束后导入图像。")
                return False
            if self._current_mode() != "Dual Image" and self.config.mode != "Dual Image":
                layer = "upper"
            mark_id = self._current_mark_id()
            try:
                image = load_image(path)
                self._ensure_mark_runtime(mark_id)
                self._switch_to_single_measurement_after_top_import()
                self._set_image_for_layer(mark_id, layer, image, "single")
                self._invalidate_image_dependent_results(mark_id, layer)
                self.recent_image_store.add(path, layer)
                self.dialog_state_store.remember_file("image_import", path)
                self._sync_current_mark_images()
                self._refresh_auto_selection_combos()
                layer_name = "上层/单图" if layer == "upper" else "下层"
                self._append_log(f"{success_prefix}{layer_name}：{Path(path).name}")
                return True
            except Exception as exc:
                self._append_log(f"导入图像失败：{exc}")
                QMessageBox.critical(self, "导入失败", str(exc))
                return False
            finally:
                self._refresh_all_widgets()

        def _diagnostic_canvas(self):
            if self._current_mode() == "Dual Image" and self._current_layer() == "lower":
                return self.lower_canvas
            return self.upper_canvas

        def start_image_diagnostics(self):
            if self.operation_mode != "Engineering":
                QMessageBox.information(self, "图像诊断", "图像诊断仅在工程模式下可用。")
                return
            canvas = self._diagnostic_canvas()
            if canvas.image is None:
                QMessageBox.information(self, "图像诊断", "请先导入当前层图像。")
                return
            dialog = getattr(self, "_image_diagnostics_dialog", None)
            if dialog is None:
                dialog = ImageDiagnosticsDialog(self)
                dialog.finished.connect(lambda *_: self._stop_image_diagnostics())
                self._image_diagnostics_dialog = dialog
            self.upper_canvas.set_profile_capture_enabled(canvas is self.upper_canvas)
            self.lower_canvas.set_profile_capture_enabled(canvas is self.lower_canvas)
            dialog.instruction_label.setText(
                f"当前：{canvas.title}。在图像上拖动一条线；使用原始灰度，不受显示增强影响。"
            )
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            self.show_canvas_interaction_message("图像诊断已开启：在图像上拖动一条线查看灰度剖面。")

        def _stop_image_diagnostics(self):
            self.upper_canvas.set_profile_capture_enabled(False)
            self.lower_canvas.set_profile_capture_enabled(False)

        def receive_image_profile(self, payload: dict):
            dialog = getattr(self, "_image_diagnostics_dialog", None)
            if dialog is not None:
                dialog.set_profile(payload)

        def _crop_roi_image(self, image: Optional[ImageData], roi: Optional[Roi], path: Path):
            if image is None or roi is None:
                return False
            r = roi.normalized()
            h, w = image.gray.shape[:2]
            margin = 20
            x0 = max(0, int(np.floor(r.x - margin)))
            y0 = max(0, int(np.floor(r.y - margin)))
            x1 = min(w, int(np.ceil(r.x + r.w + margin)))
            y1 = min(h, int(np.ceil(r.y + r.h + margin)))
            if x1 <= x0 or y1 <= y0:
                return False
            crop = display_to_uint8(image, enhanced=False)[y0:y1, x0:x1]
            Image.fromarray(crop).save(path)
            return True

        def _build_mark_image_exports(self, tmp_dir: str):
            items = []
            jobs = []
            if self._is_batch_mode() and any(self.batch_run_records.values()):
                for mark_id, records in self.batch_run_records.items():
                    for record in records:
                        run_index = int(record.get("run_index", len(jobs) + 1))
                        detected = record.get("detections", {})
                        layer_files = (("upper", "upper_file"), ("lower", "lower_file"))
                        if self._current_mode() != "Dual Image":
                            layer_files = (("single", "upper_file"),)
                        for layer, file_key in layer_files:
                            source = record.get(file_key, "")
                            if not source:
                                continue
                            layer_detections = {}
                            if record.get("workflow") == "Auto":
                                for label, layer_map in detected.items():
                                    if layer == "single":
                                        for detected_layer, detection in layer_map.items():
                                            layer_detections[f"{candidate_display_label(label, detection)}-{LAYER_LABELS.get(detected_layer, detected_layer)}"] = detection
                                    elif layer in layer_map:
                                        layer_detections[candidate_display_label(label, layer_map[layer])] = layer_map[layer]
                            else:
                                layer_detections = {
                                    key: value for key, value in detected.items()
                                    if layer == "single" or value.layer == layer
                                }
                            try:
                                image = load_image(source)
                            except Exception:
                                continue
                            jobs.append((mark_id, run_index, layer, image, layer_detections))
            else:
                for mark_id, mark in self.marks.items():
                    export_layers = ("upper", "lower") if self._current_mode() == "Dual Image" else ("single",)
                    for layer in export_layers:
                        image_layer = "upper" if layer == "single" else layer
                        image = self._image_for_layer(image_layer, mark_id)
                        if image is None:
                            continue
                        if self._is_auto_workflow():
                            detected = {}
                            for label, layer_map in self.auto_detections_by_mark.get(mark_id, {}).items():
                                for detected_layer, detection in layer_map.items():
                                    if layer == "single" or detected_layer == layer:
                                        detected[f"{candidate_display_label(label, detection)}-{LAYER_LABELS.get(detected_layer, detected_layer)}"] = detection
                        else:
                            detected = {
                                key: value for key, value in self.roi_detections.get(mark_id, {}).items()
                                if layer == "single" or value.layer == layer
                            }
                        jobs.append((mark_id, 1, layer, image, detected))
            for mark_id, run_index, layer, image, detected in jobs:
                mark = self.marks.get(mark_id)
                if mark is not None and not self._is_auto_workflow() and layer == "single":
                    rois = []
                    for roi_layer in ("upper", "lower"):
                        rois.extend(
                            (f"{LAYER_LABELS.get(roi_layer, roi_layer)} ROI {index}", entry.roi)
                            for index, entry in enumerate(mark.roi_entries(roi_layer), start=1)
                        )
                else:
                    entries = mark.roi_entries(layer) if mark is not None and not self._is_auto_workflow() else []
                    rois = [(f"ROI {index}", entry.roi) for index, entry in enumerate(entries, start=1)]
                out_path = Path(tmp_dir) / f"{mark_id}_run_{run_index}_{layer}.png"
                if render_measurement_image(image, out_path, rois, detected):
                    items.append({
                        "mark_id": f"{mark_id} 第{run_index}次",
                        "layer": "单图" if layer == "single" else LAYER_LABELS.get(layer, layer),
                        "path": str(out_path),
                        "note": f"完整图像与全部 ROI/绿色识别轮廓；来源：{Path(image.path).parent.name} / {Path(image.path).name}",
                    })
            return items

        def _first_upper_image_for_export(self) -> Optional[ImageData]:
            if self._is_batch_mode():
                for mark_id in ("Mark1", "Mark2"):
                    images = self.batch_images.get(mark_id, {}).get("upper", [])
                    if images:
                        return images[0]
            current = self._image_for_layer("upper", self._current_mark_id())
            if current is not None:
                return current
            for mark_id in ("Mark1", "Mark2"):
                image = self._active_image_for_layer(mark_id, "upper")
                if image is not None:
                    return image
            return None

        def _default_export_filename(self) -> str:
            image = self._first_upper_image_for_export()
            return build_export_filename(image.path if image is not None else "")

        def _build_repeatability_export_rows(self) -> list[dict]:
            rows: list[dict] = []
            for mark_id in ("Mark1", "Mark2"):
                overlays = self.batch_overlays.get(mark_id, [])
                records = list(self.batch_run_records.get(mark_id, []))
                if records:
                    for record in records:
                        overlay = record.get("overlay")
                        error = record.get("error", "")
                        rows.append({
                            "Mark": mark_id,
                            "次数": record.get("run_index", ""),
                            "上层/单图文件": Path(record.get("upper_file", "")).name,
                            "下层文件": Path(record.get("lower_file", "")).name,
                            "Dx(μm)": overlay.delta_x_um if overlay else None,
                            "Dy(μm)": overlay.delta_y_um if overlay else None,
                            "Dxy(μm)": overlay.overlay_r_um if overlay else None,
                            "判定": RESULT_LABELS.get(overlay.result, overlay.result) if overlay else "异常",
                            "提示": overlay.warning if overlay else error,
                        })
                else:
                    upper_images = self.batch_images.get(mark_id, {}).get("upper", [])
                    lower_images = self.batch_images.get(mark_id, {}).get("lower", [])
                    for index, overlay in enumerate(overlays):
                        rows.append({
                            "Mark": mark_id,
                            "次数": index + 1,
                            "上层/单图文件": Path(upper_images[index].path).name if index < len(upper_images) else "",
                            "下层文件": Path(lower_images[index].path).name if index < len(lower_images) else "",
                            "Dx(μm)": overlay.delta_x_um,
                            "Dy(μm)": overlay.delta_y_um,
                            "Dxy(μm)": overlay.overlay_r_um,
                            "判定": RESULT_LABELS.get(overlay.result, overlay.result),
                            "提示": overlay.warning,
                        })
                if overlays:
                    dxs = np.asarray([o.delta_x_um for o in overlays], dtype=float)
                    dys = np.asarray([o.delta_y_um for o in overlays], dtype=float)
                    rs = np.asarray([o.overlay_r_um for o in overlays], dtype=float)
                    if len(overlays) >= 2:
                        dx_3sigma = float(3.0 * np.std(dxs, ddof=1))
                        dy_3sigma = float(3.0 * np.std(dys, ddof=1))
                        r_3sigma = float(3.0 * np.std(rs, ddof=1))
                    else:
                        dx_3sigma = dy_3sigma = r_3sigma = 0.0
                    rows.append({
                        "Mark": mark_id,
                        "次数": "统计",
                        "上层/单图文件": "",
                        "下层文件": "",
                        "Dx(μm)": float(np.mean(dxs)),
                        "Dy(μm)": float(np.mean(dys)),
                        "Dxy(μm)": float(np.mean(rs)),
                        "均值向量Dxy(μm)": float(np.hypot(np.mean(dxs), np.mean(dys))),
                        "3σ-Dx(μm)": dx_3sigma,
                        "3σ-Dy(μm)": dy_3sigma,
                        "3σ-Dxy(μm)": r_3sigma,
                        "PV-Dx(μm)": float(np.max(dxs) - np.min(dxs)),
                        "PV-Dy(μm)": float(np.max(dys) - np.min(dys)),
                        "PV-Dxy(μm)": float(np.max(rs) - np.min(rs)),
                        "判定": "-",
                        "提示": "多次测量统计",
                    })
            return rows

        def on_mode_changed(self, *args):
            self._pull_config_from_ui()
            self.invalidate_measurement_state("图像模式已切换")
            self._sync_current_mark_images()
            self._refresh_auto_selection_combos()
            self._refresh_all_widgets()

        def on_workflow_mode_changed(self, *args):
            self._pull_config_from_ui()
            if self._is_auto_workflow():
                # Auto mode always performs a fresh full-image search. Remove stale
                # candidate overlays so a previous run cannot look current.
                self.auto_detections_by_mark = {"Mark1": {}, "Mark2": {}}
                self.auto_candidates_by_mark = {"Mark1": {}, "Mark2": {}}
                self.auto_overlays.clear()
                self._append_log("已切换为全图自动识别；本次计算不会使用配方或手动 ROI。")
            else:
                self._append_log("已切换为手动 ROI 测量；计算前会确认仍在使用的配方 ROI。")
            self._refresh_auto_selection_combos()
            self._refresh_all_widgets()

        def on_auto_selection_changed(self, *args):
            if not hasattr(self, "auto_reference_combo"):
                return
            mark_id = self._current_mark_id()
            self._ensure_mark_runtime(mark_id)
            self.auto_selections[mark_id] = {
                "reference_label": self.auto_reference_combo.currentData() or "",
                "target_label": self.auto_target_combo.currentData() or "",
            }
            if self.auto_selections[mark_id]["reference_label"] or self.auto_selections[mark_id]["target_label"]:
                self._manual_selection_requires_review.discard(mark_id)
            mark = self.marks.get(mark_id)
            if mark is not None:
                mark.reference_contour_id = self.auto_selections[mark_id]["reference_label"]
                mark.target_contour_id = self.auto_selections[mark_id]["target_label"]
            self.auto_overlays.pop(mark_id, None)
            self._refresh_all_widgets()

        def _push_current_auto_match_rules(self):
            if not hasattr(self, "auto_ref_shape_combo"):
                return
            mark = self.marks.get(self._current_mark_id())
            if mark is None:
                return
            widgets = (
                self.auto_ref_shape_combo,
                self.auto_target_shape_combo,
                self.auto_ref_size_min_spin,
                self.auto_ref_size_max_spin,
                self.auto_target_size_min_spin,
                self.auto_target_size_max_spin,
            )
            for widget in widgets:
                widget.blockSignals(True)
            self._set_combo_value(self.auto_ref_shape_combo, mark.reference_shape)
            self._set_combo_value(self.auto_target_shape_combo, mark.target_shape)
            self.auto_ref_size_min_spin.setValue(mark.reference_size_min_um)
            self.auto_ref_size_max_spin.setValue(mark.reference_size_max_um)
            self.auto_target_size_min_spin.setValue(mark.target_size_min_um)
            self.auto_target_size_max_spin.setValue(mark.target_size_max_um)
            for widget in widgets:
                widget.blockSignals(False)

        def on_auto_match_rule_changed(self, *args):
            mark = self.marks.get(self._current_mark_id())
            if mark is None:
                return
            mark.reference_shape = self._combo_value(self.auto_ref_shape_combo)
            mark.target_shape = self._combo_value(self.auto_target_shape_combo)
            mark.reference_size_min_um = self.auto_ref_size_min_spin.value()
            mark.reference_size_max_um = max(mark.reference_size_min_um, self.auto_ref_size_max_spin.value())
            mark.target_size_min_um = self.auto_target_size_min_spin.value()
            mark.target_size_max_um = max(mark.target_size_min_um, self.auto_target_size_max_spin.value())
            self.auto_candidates_by_mark[mark.mark_id] = {}
            self.auto_detections_by_mark[mark.mark_id] = {}
            self.auto_selections[mark.mark_id] = {"reference_label": "", "target_label": ""}
            self.auto_overlays.pop(mark.mark_id, None)
            self.overlays.pop(mark.mark_id, None)
            self._refresh_auto_selection_combos()
            self._refresh_all_widgets()
            self._append_log("自动识别规则已修改，旧结果已失效，请重新识别。")

        def _matches_auto_rule(self, detection: DetectionResult, role: str, mark: MarkRecipe) -> bool:
            shape = detection.shape_params.get("shape_type", "")
            expected = mark.reference_shape if role == "reference" else mark.target_shape
            minimum = mark.reference_size_min_um if role == "reference" else mark.target_size_min_um
            maximum = mark.reference_size_max_um if role == "reference" else mark.target_size_max_um
            return (expected == "Any" or expected == shape) and minimum <= detection.diameter_um <= maximum

        def _refresh_auto_selection_combos(self):
            """Refresh reference/target contour selectors for both Auto and Manual workflows.

            Auto workflow uses contours detected by auto_mark_detector and displays row-major spatial numbers.
            Manual workflow uses the currently analyzed manual ROI results, so users can
            explicitly choose which ROI is the reference contour and which is the target contour.
            """
            if not hasattr(self, "auto_reference_combo"):
                return
            mark_id = self._current_mark_id()
            self._ensure_mark_runtime(mark_id)
            batch_record = self._selected_batch_record(mark_id) if hasattr(self, "_selected_batch_record") else None
            selection = (
                batch_record.get("selection", {})
                if batch_record
                else self.auto_selections[mark_id]
            )
            previous_reference = selection.get("reference_label", "")
            previous_target = selection.get("target_label", "")
            self.auto_reference_combo.blockSignals(True)
            self.auto_target_combo.blockSignals(True)
            self.auto_reference_combo.clear()
            self.auto_target_combo.clear()

            if self._is_auto_workflow():
                mark = self.marks[mark_id]
                current_detections = self._current_auto_detections()
                previous_reference = resolve_preferred_candidate(previous_reference, current_detections)
                previous_target = resolve_preferred_candidate(previous_target, current_detections)
                for label, layer_map in current_detections.items():
                    detection = next(iter(layer_map.values()))
                    if not detection.shape_params.get(
                        "measurement_eligible",
                        detection.shape_params.get("quality_status") == "Valid",
                    ):
                        continue
                    shape = "方" if detection.fitting_mode == "ProductionRectangle" else "圆"
                    size_name = "半尺寸" if detection.fitting_mode in {"AutoRectangle", "ProductionRectangle"} else "半径"
                    display_label = candidate_display_label(label, detection)
                    text = f"{mark_id}-{display_label} - {shape} - {size_name}={detection.diameter_um / 2.0:.3f} μm"
                    if self._matches_auto_rule(detection, "reference", mark):
                        self.auto_reference_combo.addItem(text, label)
                    if self._matches_auto_rule(detection, "target", mark):
                        self.auto_target_combo.addItem(text, label)
            else:
                layer_map = self._current_manual_detection_map().get(mark_id, {})
                for label, detection in layer_map.items():
                    layer = detection.layer
                    layer_name = LAYER_LABELS.get(layer, layer)
                    fit_name = {
                        "Circle": "圆拟合",
                        "Ellipse": "椭圆拟合",
                        "Rectangle": "矩形拟合",
                        "EdgeCenter": "稳健中心",
                        "RegionCenter": "区域中心",
                        "CaliperCircle": "卡尺圆",
                    }.get(detection.fitting_mode, detection.fitting_mode)
                    roi_index = int(detection.shape_params.get("roi_index", 1))
                    text = f"{layer_name} ROI {roi_index} - " + (
                        f"{mark_id}-{layer_name}轮廓 - {fit_name} - "
                        f"中心=({detection.center_x_um:.3f}, {detection.center_y_um:.3f}) μm"
                    )
                    self.auto_reference_combo.addItem(text, label)
                    self.auto_target_combo.addItem(text, label)

            if self.auto_reference_combo.count() > 0:
                reference_index = self.auto_reference_combo.findData(previous_reference) if previous_reference else -1
                if reference_index >= 0:
                    self.auto_reference_combo.setCurrentIndex(reference_index)
                elif previous_reference:
                    self.auto_reference_combo.setCurrentIndex(-1)
                elif self._is_auto_workflow():
                    self.auto_reference_combo.setCurrentIndex(0)
                elif mark_id in self._manual_selection_requires_review:
                    self.auto_reference_combo.setCurrentIndex(-1)
                else:
                    manual_map = self._current_manual_detection_map().get(mark_id, {})
                    default_id = next(
                        (roi_id for roi_id, detection in manual_map.items() if detection.layer == "upper"),
                        self.auto_reference_combo.itemData(0),
                    )
                    self.auto_reference_combo.setCurrentIndex(self.auto_reference_combo.findData(default_id))
            if self.auto_target_combo.count() > 0:
                target_index = self.auto_target_combo.findData(previous_target) if previous_target else -1
                if target_index >= 0:
                    self.auto_target_combo.setCurrentIndex(target_index)
                elif previous_target:
                    self.auto_target_combo.setCurrentIndex(-1)
                elif self._is_auto_workflow():
                    default_index = 1 if self.auto_target_combo.count() > 1 else -1
                    self.auto_target_combo.setCurrentIndex(default_index)
                elif mark_id in self._manual_selection_requires_review:
                    self.auto_target_combo.setCurrentIndex(-1)
                else:
                    manual_map = self._current_manual_detection_map().get(mark_id, {})
                    reference_id = self.auto_reference_combo.currentData() or ""
                    default_id = next(
                        (
                            roi_id for roi_id, detection in manual_map.items()
                            if detection.layer == "lower" and roi_id != reference_id
                        ),
                        next((roi_id for roi_id in manual_map if roi_id != reference_id), ""),
                    )
                    self.auto_target_combo.setCurrentIndex(self.auto_target_combo.findData(default_id))
            self.auto_reference_combo.blockSignals(False)
            self.auto_target_combo.blockSignals(False)
            self.auto_selections[mark_id] = {
                "reference_label": self.auto_reference_combo.currentData() or "",
                "target_label": self.auto_target_combo.currentData() or "",
            }
            mark = self.marks.get(mark_id)
            if mark is not None:
                mark.reference_contour_id = self.auto_selections[mark_id]["reference_label"]
                mark.target_contour_id = self.auto_selections[mark_id]["target_label"]

        def auto_identify_marks(self, show_message: bool = True):
            self._pull_config_from_ui()
            mark_id = self._current_mark_id()
            images = {
                "upper": self._image_for_layer("upper", mark_id),
                "lower": self._image_for_layer("lower", mark_id),
            }
            required = ("upper", "lower") if self._current_mode() == "Dual Image" else ("upper",)
            if any(images[layer] is None for layer in required):
                if show_message:
                    QMessageBox.warning(self, "自动识别", "请先导入当前测量模式需要的图像。")
                return 0
            return self._start_preview_job("auto", mark_id, images, show_message)

        def _find_manual_detection(self, mark_id: str, label: str) -> Optional[DetectionResult]:
            detections = self._current_manual_detection_map().get(mark_id, {})
            if label in detections:
                return detections[label]
            if label in {"upper", "lower"}:
                return next((item for item in detections.values() if item.layer == label), None)
            return None

        def calculate_auto_overlay(self, show_message: bool = True):
            """Calculate overlay from the selected reference/target contours.

            The historical name is kept to avoid changing signal connections, but the
            function now supports both 自动识别测量 and 手动 ROI 测量.
            """
            mark_id = self._current_mark_id()
            reference_label = self.auto_reference_combo.currentData() or ""
            target_label = self.auto_target_combo.currentData() or ""
            if not reference_label or not target_label or reference_label == target_label:
                if show_message:
                    QMessageBox.warning(self, "计算对位偏差", "请选择不同的基准轮廓和待测轮廓。")
                return None
            self._pull_config_from_ui()

            if self._is_auto_workflow():
                reference = self._find_auto_detection(mark_id, reference_label)
                target = self._find_auto_detection(mark_id, target_label)
                missing_message = "所选轮廓不存在，请重新执行自动识别。"
            else:
                reference = self._find_manual_detection(mark_id, reference_label)
                target = self._find_manual_detection(mark_id, target_label)
                missing_message = "所选手动 ROI 轮廓不存在，请先分析对应 ROI。"

            if reference is None or target is None:
                if show_message:
                    QMessageBox.warning(self, "计算对位偏差", missing_message)
                return None
            invalid = [
                label
                for label, detection in ((reference_label, reference), (target_label, target))
                if detection.shape_params.get("quality_status") == "Invalid"
            ]
            if invalid:
                if show_message:
                    QMessageBox.warning(self, "计算对位偏差", "所选轮廓未通过质量门槛，不能用于对位判定。")
                return None

            if self._is_auto_workflow():
                reference_name = candidate_display_label(reference_label, reference)
                target_name = candidate_display_label(target_label, target)
            else:
                reference_name, target_name = reference_label, target_label
            name = f"{mark_id}: {target_name} 相对 {reference_name}"
            overlay = calculate_relative_overlay(mark_id, reference, target, self.config)
            overlay.reference_contour_id = reference_label
            overlay.target_contour_id = target_label
            overlay.reference_contour_name = reference_name
            overlay.target_contour_name = target_name
            if self.config.recipe_validation_status != "Validated":
                overlay.result = "Trial"
                overlay.warning = "试测/未验证配方，不作正式判定"

            if self._is_auto_workflow():
                self.auto_overlays[mark_id] = overlay
            else:
                self.overlays[mark_id] = overlay
            self.auto_selections[mark_id] = {
                "reference_label": reference_label,
                "target_label": target_label,
            }
            self._refresh_all_widgets()
            if show_message:
                QMessageBox.information(
                    self,
                    "计算完成",
                    f"{name}：Dx={overlay.delta_x_um:.3f} μm，Dy={overlay.delta_y_um:.3f} μm，Dxy={overlay.overlay_r_um:.3f} μm",
                )
            return overlay

        def on_three_point_circle_toggled(self, checked: bool):
            if checked:
                self._set_combo_value(self.roi_type_combo, "Caliper Circle")
            self.upper_canvas.set_circle_pick_mode(checked)
            self.lower_canvas.set_circle_pick_mode(checked)

        def _is_batch_mode(self) -> bool:
            return hasattr(self, "measurement_run_mode_combo") and self._combo_value(self.measurement_run_mode_combo) == "Batch"

        @staticmethod
        def _batch_image_path_key(image: ImageData) -> str:
            path = str(getattr(image, "path", "") or "")
            if not path:
                return f"<memory:{id(image)}>"
            return str(Path(path).resolve(strict=False)).casefold()

        @staticmethod
        def _natural_path_sort_key(path: Path, root: Optional[Path] = None) -> tuple:
            try:
                text = str(path.relative_to(root)) if root is not None else path.name
            except ValueError:
                text = str(path)
            parts = re.split(r"(\d+)", text.replace("\\", "/").casefold())
            return tuple((1, int(part)) if part.isdigit() else (0, part) for part in parts if part)

        @classmethod
        def _collect_batch_paths(cls, folder: str, recursive: bool) -> list[str]:
            root = Path(folder)
            iterator = root.rglob("*") if recursive else root.iterdir()
            paths = [
                path
                for path in iterator
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            paths.sort(key=lambda path: cls._natural_path_sort_key(path, root))
            return [str(path) for path in paths]

        @staticmethod
        def _batch_source_folder_count(images: list[ImageData]) -> int:
            folders = {
                str(Path(image.path).resolve(strict=False).parent).casefold()
                for image in images
                if getattr(image, "path", "")
            }
            return len(folders)

        def _append_batch_image_data(self, mark_id: str, layer: str, images: list) -> tuple[int, int]:
            self._ensure_mark_runtime(mark_id)
            target = self.batch_images.setdefault(mark_id, {}).setdefault(layer, [])
            known_paths = {self._batch_image_path_key(image) for image in target}
            added = 0
            duplicates = 0
            for image in images:
                path_key = self._batch_image_path_key(image)
                if path_key in known_paths:
                    duplicates += 1
                    continue
                target.append(image)
                known_paths.add(path_key)
                added += 1

            if added:
                self.batch_run_records[mark_id] = []
                self.batch_overlays[mark_id] = []
                self._set_image_for_layer(mark_id, layer, resolve_image(target[0]), "batch_preview")
                self._invalidate_image_dependent_results(mark_id, layer)
            return added, duplicates

        def import_batch_images(self, mark_id: str, layer: str):
            title = f"追加批量 {mark_id} {LAYER_LABELS.get(layer, layer)}图像"
            import_source = self._combo_value(self.batch_import_source_combo)
            if import_source == "Folder":
                folder = QFileDialog.getExistingDirectory(
                    self, title, self.dialog_state_store.directory("batch_folder"), QFileDialog.ShowDirsOnly
                )
                paths = self._collect_batch_paths(folder, self.batch_recursive_check.isChecked()) if folder else []
                if folder:
                    self.dialog_state_store.remember_directory("batch_folder", folder, [folder])
                if folder and not paths:
                    QMessageBox.warning(self, "未找到图像", "所选文件夹中没有可导入的图像或矩阵文件。")
            elif import_source == "Folders":
                dialog = MultiFolderSelectionDialog(
                    self.dialog_state_store.directory("batch_folders"),
                    self.dialog_state_store.paths("batch_folders"),
                    self,
                )
                if dialog.exec() != QDialog.Accepted:
                    paths = []
                else:
                    folders = dialog.selected_directories()
                    paths = []
                    for folder in folders:
                        paths.extend(self._collect_batch_paths(folder, self.batch_recursive_check.isChecked()))
                    paths = sorted(set(paths), key=lambda path: self._natural_path_sort_key(Path(path)))
                    if folders:
                        self.dialog_state_store.remember_directory("batch_folders", folders[0], folders)
                    if folders and not paths:
                        QMessageBox.warning(self, "未找到图像", "所选文件夹中没有可导入的图像或矩阵文件。")
            else:
                dialog = QFileDialog(self, title, self.dialog_state_store.directory("batch_files"))
                dialog.setNameFilter("图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;矩阵 (*.csv *.txt *.npy);;全部文件 (*)")
                dialog.setFileMode(QFileDialog.ExistingFiles)
                last_path = self.dialog_state_store.last_path("batch_files")
                if last_path:
                    dialog.selectFile(last_path)
                paths = dialog.selectedFiles() if dialog.exec() == QDialog.Accepted else []
                paths = sorted(paths, key=lambda path: self._natural_path_sort_key(Path(path)))
                if paths:
                    self.dialog_state_store.remember_file("batch_files", paths[-1])
            if not paths:
                self._append_log(f"取消{title}。")
                return
            try:
                images = [BatchImageRef.from_path(path) for path in paths]
                added, duplicates = self._append_batch_image_data(mark_id, layer, images)
                self._set_combo_value(self.measurement_run_mode_combo, "Batch")
                self._sync_current_mark_images()
                self._refresh_auto_selection_combos()
                self._refresh_all_widgets()
                total = len(self.batch_images[mark_id][layer])
                folder_count = self._batch_source_folder_count(self.batch_images[mark_id][layer])
                message = f"{title}完成：新增 {added} 张，当前共 {total} 张，来自 {folder_count} 个文件夹"
                if duplicates:
                    message += f"；已跳过 {duplicates} 张重复文件"
                total_bytes = sum(getattr(item, "size_bytes", 0) for item in self.batch_images[mark_id][layer])
                if total >= 1000 or total_bytes >= 500 * 1024 * 1024:
                    message += f"；数据量较大（{total} 张，{total_bytes / 1024**2:.1f} MB），测量将按文件逐张加载"
                self._append_log(message + "。")
            except Exception as exc:
                self._append_log(f"{title}失败：{exc}")
                QMessageBox.critical(self, "批量导入失败", str(exc))

        def clear_batch_images(self):
            self.batch_images = {"Mark1": {"upper": [], "lower": []}, "Mark2": {"upper": [], "lower": []}}
            self.batch_overlays = {"Mark1": [], "Mark2": []}
            self.batch_run_records = {"Mark1": [], "Mark2": []}
            self._batch_detail_run_index = 1
            self._batch_detail_last_single_index = 1
            self._refresh_batch_detail_selector()
            for mark_id in ("Mark1", "Mark2"):
                for layer in ("upper", "lower"):
                    if self._image_source(mark_id, layer) == "batch_preview":
                        self._set_image_for_layer(mark_id, layer, None, "none")
            self._sync_current_mark_images()
            self._refresh_all_widgets()
            self._append_log("已清空批量图像和重复性结果。")

        def _refresh_batch_image_table(self):
            if not hasattr(self, "batch_image_table"):
                return
            headers = ["Mark", "上层/单图", "下层", "状态"]
            rows = []
            is_dual = self._current_mode() == "Dual Image"
            for mark_id in ("Mark1", "Mark2"):
                upper_images = self.batch_images.get(mark_id, {}).get("upper", [])
                lower_images = self.batch_images.get(mark_id, {}).get("lower", [])
                upper_count = len(upper_images)
                lower_count = len(lower_images)
                if upper_count == 0:
                    status = "未导入"
                elif is_dual and lower_count == 0:
                    status = "缺少下层"
                elif is_dual and upper_count != lower_count:
                    status = "上下数量不一致"
                else:
                    status = "可批量计算"
                rows.append([
                    mark_id,
                    f"{upper_count}张/{self._batch_source_folder_count(upper_images)}目录",
                    f"{lower_count}张/{self._batch_source_folder_count(lower_images)}目录",
                    status,
                ])
            self._fill_table(self.batch_image_table, headers, rows)

        def _refresh_repeatability_table(self):
            if not hasattr(self, "repeat_table"):
                return
            headers = ["Mark", "次数", "Dx(μm)", "Dy(μm)", "Dxy(μm)", "判定", "提示"]
            rows = []
            row_payloads = []
            for mark_id in ("Mark1", "Mark2"):
                overlays = self.batch_overlays.get(mark_id, [])
                records = self.batch_run_records.get(mark_id, [])
                if records:
                    for record in records:
                        overlay = record.get("overlay")
                        rows.append([
                            mark_id,
                            str(record.get("run_index", "")),
                            f"{overlay.delta_x_um:+.3f}" if overlay else "-",
                            f"{overlay.delta_y_um:+.3f}" if overlay else "-",
                            f"{overlay.overlay_r_um:.3f}" if overlay else "-",
                            RESULT_LABELS.get(overlay.result, overlay.result) if overlay else "异常",
                            overlay.warning if overlay else record.get("error", ""),
                        ])
                        row_payloads.append({
                            "kind": "batch_run", "run_index": record.get("run_index"),
                        })
                else:
                    for idx, overlay in enumerate(overlays, start=1):
                        rows.append([
                            mark_id,
                            str(idx),
                            f"{overlay.delta_x_um:+.3f}",
                            f"{overlay.delta_y_um:+.3f}",
                            f"{overlay.overlay_r_um:.3f}",
                            RESULT_LABELS.get(overlay.result, overlay.result),
                            overlay.warning,
                        ])
                        row_payloads.append({
                            "kind": "batch_run", "run_index": idx,
                        })
                if overlays:
                    dxs = np.asarray([o.delta_x_um for o in overlays], dtype=float)
                    dys = np.asarray([o.delta_y_um for o in overlays], dtype=float)
                    rs = np.asarray([o.overlay_r_um for o in overlays], dtype=float)
                    if len(overlays) >= 2:
                        dx_3sigma = float(3.0 * np.std(dxs, ddof=1))
                        dy_3sigma = float(3.0 * np.std(dys, ddof=1))
                        r_3sigma = float(3.0 * np.std(rs, ddof=1))
                    else:
                        dx_3sigma = dy_3sigma = r_3sigma = 0.0
                    dx_pv = float(np.max(dxs) - np.min(dxs))
                    dy_pv = float(np.max(dys) - np.min(dys))
                    r_pv = float(np.max(rs) - np.min(rs))
                    rows.append([
                        mark_id,
                        "统计",
                        f"均值={np.mean(dxs):+.3f}; 3σ={dx_3sigma:.3f}; PV={dx_pv:.3f}",
                        f"均值={np.mean(dys):+.3f}; 3σ={dy_3sigma:.3f}; PV={dy_pv:.3f}",
                        f"均值={np.mean(rs):.3f}; 3σ={r_3sigma:.3f}; PV={r_pv:.3f}",
                        "-",
                        "重复性统计",
                    ])
                    row_payloads.append({
                        "kind": "repeat_stat", "mark_id": mark_id, "deletable": False,
                    })
            self._fill_table(self.repeat_table, headers, rows, row_payloads)

        def _mean_overlay(self, mark_id: str, overlays: list[OverlayResult]) -> Optional[OverlayResult]:
            if not overlays:
                return None
            dx = float(np.mean([o.delta_x_um for o in overlays]))
            dy = float(np.mean([o.delta_y_um for o in overlays]))
            r = float(np.hypot(dx, dy))
            warnings = []
            if abs(dx) > self.config.delta_x_limit_um:
                warnings.append("Dx均值超限")
            if abs(dy) > self.config.delta_y_limit_um:
                warnings.append("Dy均值超限")
            if r > self.config.overlay_r_limit_um:
                warnings.append("Dxy均值超限")
            return OverlayResult(
                mark_id=mark_id,
                delta_x_px=0.0,
                delta_y_px=0.0,
                delta_x_um=dx,
                delta_y_um=dy,
                overlay_r_um=r,
                result="Fail" if warnings else "Pass",
                warning="；".join(warnings),
            )

        def calculate_batch_overlays(self):
            return self._start_measurement_job()

        def import_dropped_image(self, path: str, layer: str):
            self._import_image_path(path, layer, "已拖入")

        def import_upper_image(self):
            path = self._choose_image_file("导入上层/单张图像")
            if not path:
                self._append_log("取消导入上层/单图。")
                return
            self._import_image_path(path, "upper", "已导入")

        def import_lower_image(self):
            path = self._choose_image_file("导入下层图像")
            if not path:
                self._append_log("取消导入下层图像。")
                return
            self._import_image_path(path, "lower", "已导入")

        def _choose_image_file(self, title: str) -> str:
            dialog = QFileDialog(self, title, self.dialog_state_store.directory("image_import"))
            dialog.setNameFilter("图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;矩阵 (*.csv *.txt *.npy);;全部文件 (*)")
            dialog.setFileMode(QFileDialog.ExistingFile)
            last_path = self.dialog_state_store.last_path("image_import")
            if last_path:
                dialog.selectFile(last_path)
            return dialog.selectedFiles()[0] if dialog.exec() == QDialog.Accepted and dialog.selectedFiles() else ""

        def add_mark(self):
            self.marks = {
                mark_id: self.marks.get(mark_id, MarkRecipe(mark_id))
                for mark_id in ("Mark1", "Mark2")
            }
            QMessageBox.information(self, "标记数量", "本版本仅支持 Mark1 和 Mark2。")
            self._refresh_all_widgets()

        def reset_measurement(self):
            answer = QMessageBox.question(
                self,
                "重置测量",
                "将清除所有已导入的单次和批量图像、全部测量结果以及所有 ROI。\n\n确定继续吗？",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return
            self._clear_measurement_state()
            self._append_log("测量已重置：图像、测量结果和 ROI 已全部清除。")

        def _clear_measurement_state(self):
            self.marks = {"Mark1": MarkRecipe("Mark1"), "Mark2": MarkRecipe("Mark2")}
            self.mark_images = {
                "Mark1": {"upper": None, "lower": None},
                "Mark2": {"upper": None, "lower": None},
            }
            self.mark_image_sources = self._empty_image_sources()
            self.detections.clear()
            self.roi_detections = {"Mark1": {}, "Mark2": {}}
            self.roi_detection_failures = {"Mark1": [], "Mark2": []}
            self._pending_new_roi = False
            self.overlays.clear()
            self.auto_detections_by_mark = {"Mark1": {}, "Mark2": {}}
            self.auto_candidates_by_mark = {"Mark1": {}, "Mark2": {}}
            self.auto_selections = {
                "Mark1": {"reference_label": "", "target_label": ""},
                "Mark2": {"reference_label": "", "target_label": ""},
            }
            self._manual_selection_requires_review.clear()
            self.auto_overlays.clear()
            self.batch_images = {"Mark1": {"upper": [], "lower": []}, "Mark2": {"upper": [], "lower": []}}
            self.batch_overlays = {"Mark1": [], "Mark2": []}
            self.batch_run_records = {"Mark1": [], "Mark2": []}
            self.geometry_program = GeometryProgram()
            self.geometry_result = GeometryRunResult()
            self.batch_geometry_results = []
            self._geometry_interaction = None
            self._batch_detail_run_index = 1
            self._batch_detail_last_single_index = 1
            self.roi_sources = self._empty_roi_sources()
            self.loaded_recipe_path = ""
            self.loaded_recipe_display_name = ""
            self.loaded_recipe_hash = ""
            self.recipe_integrity_status = "Unsealed"
            self._recipe_roi_confirmation_signature = None
            self.config.recipe_name = ""
            if hasattr(self, "recipe_name_edit"):
                self.recipe_name_edit.clear()
            if hasattr(self, "measurement_run_mode_combo"):
                self._set_combo_value(self.measurement_run_mode_combo, "Single")
            self.upper_canvas.set_circle_pick_mode(False)
            self.lower_canvas.set_circle_pick_mode(False)
            self.three_point_circle_btn.blockSignals(True)
            self.three_point_circle_btn.setChecked(False)
            self.three_point_circle_btn.blockSignals(False)
            self.mark_combo.setCurrentText("Mark1")
            self.layer_combo.setCurrentIndex(0)
            self._sync_current_mark_images()
            self.reset_canvas_views()
            self._refresh_auto_selection_combos()
            self._refresh_all_widgets()

        def commit_canvas_roi_edit(self, mark_id: str, layer: str, roi_id: str, roi: Roi):
            if self.operation_mode != "Engineering" or self._calculation_running:
                self._append_log("当前模式不允许修改 ROI。")
                return
            mark = self.marks.get(mark_id)
            entry = mark.roi_entry(layer, roi_id) if mark is not None else None
            if entry is None or roi is None:
                self._append_log("ROI 编辑未提交：目标 ROI 已不存在。")
                return
            self._push_roi_undo()
            entry.roi = roi.normalized()
            entry.source = "manual"
            self.roi_sources.setdefault(mark_id, {})[layer] = "manual"
            self._recipe_roi_confirmation_signature = None
            self._invalidate_manual_roi(mark_id, roi_id)
            if mark_id == self._current_mark_id() and layer == self._current_layer():
                self._refresh_roi_index_combo(roi_id)
            self._refresh_all_widgets()

        def set_roi(self, mark_id: str, layer: str, roi: Roi, source: str = "manual"):
            if self.operation_mode != "Engineering":
                self._append_log("生产模式不允许修改 ROI，请先进入工程模式。")
                return
            if hasattr(self, "three_point_circle_btn") and self.three_point_circle_btn.isChecked():
                self.three_point_circle_btn.blockSignals(True)
                self.three_point_circle_btn.setChecked(False)
                self.three_point_circle_btn.blockSignals(False)
            if mark_id not in {"Mark1", "Mark2"}:
                return
            self._push_roi_undo()
            if mark_id not in self.marks:
                self.marks[mark_id] = MarkRecipe(mark_id)
            mark = self.marks[mark_id]
            if roi is not None:
                roi = self._coerce_roi_to_auto_ring(roi, layer)
            current_entry = (
                self._current_roi_entry()
                if mark_id == self._current_mark_id() and layer == self._current_layer()
                else None
            )
            if roi is None:
                if current_entry is not None:
                    mark.remove_roi(layer, current_entry.roi_id)
                    self._invalidate_manual_roi(mark_id, current_entry.roi_id, remove_selection=True)
                self._pending_new_roi = False
                self._refresh_roi_index_combo()
                self._refresh_all_widgets()
                return
            if self._pending_new_roi or current_entry is None:
                current_entry = mark.add_roi(layer, roi, source)
                self._pending_new_roi = False
                self._refresh_roi_index_combo(current_entry.roi_id)
                self.upper_canvas.setCursor(Qt.ArrowCursor)
                self.lower_canvas.setCursor(Qt.ArrowCursor)
            else:
                current_entry.roi = roi
                current_entry.source = source
            self.roi_sources.setdefault(mark_id, {})[layer] = source if roi is not None else "none"
            self._recipe_roi_confirmation_signature = None
            self._invalidate_manual_roi(mark_id, current_entry.roi_id)
            if roi is not None and mark_id == (self.mark_combo.currentText() or "Mark1") and layer == self._current_layer():
                widgets = (
                    self.roi_type_combo,
                    self.center_x_spin,
                    self.center_y_spin,
                    self.inner_radius_spin,
                    self.outer_radius_spin,
                    self.caliper_count_spin,
                    self.caliper_width_spin,
                    self.search_direction_combo,
                    self.target_edge_combo,
                    self.diameter_mode_combo,
                    self.inner_ratio_spin,
                    self.roi_angle_spin,
                )
                for widget in widgets:
                    widget.blockSignals(True)
                self._set_combo_value(self.roi_type_combo, getattr(roi, "roi_type", "Annulus"))
                cx, cy = roi.center()
                self.center_x_spin.setValue(cx)
                self.center_y_spin.setValue(cy)
                self.inner_radius_spin.setValue(roi.inner_radius())
                self.outer_radius_spin.setValue(roi.outer_radius())
                self.caliper_count_spin.setValue(int(getattr(roi, "caliper_count", 64)))
                self.caliper_width_spin.setValue(float(getattr(roi, "caliper_width_px", 8.0)))
                self._set_combo_value(self.search_direction_combo, getattr(roi, "search_direction", "Inner to Outer"))
                self._set_combo_value(self.target_edge_combo, getattr(roi, "target_edge", "All Edges"))
                self._set_combo_value(self.diameter_mode_combo, getattr(roi, "diameter_mode", "Average"))
                self.inner_ratio_spin.setValue(float(getattr(roi, "inner_ratio", 0.60)))
                self.roi_angle_spin.setValue(float(getattr(roi, "angle_deg", 0.0)))
                for widget in widgets:
                    widget.blockSignals(False)
            self._refresh_all_widgets()

        def clear_current_roi(self):
            mark_id = self._current_mark_id()
            layer = self._current_layer()
            entry = self._current_roi_entry()
            if entry is not None:
                self.marks[mark_id].remove_roi(layer, entry.roi_id)
                self._invalidate_manual_roi(mark_id, entry.roi_id, remove_selection=True)
                self._refresh_roi_index_combo()
                self._refresh_all_widgets()
            self._append_log(f"已清除 {mark_id} {LAYER_LABELS.get(layer, layer)} ROI。")

        def clear_all_recipe_rois(self):
            cleared = []
            for mark_id in ("Mark1", "Mark2"):
                mark = self.marks.get(mark_id)
                if mark is None:
                    continue
                for layer in ("upper", "lower"):
                    retained = []
                    for entry_index, entry in enumerate(mark.roi_entries(layer)):
                        legacy_source = self.roi_sources.get(mark_id, {}).get(layer, "none")
                        is_recipe_entry = entry.source == "recipe" or (
                            entry_index == 0 and legacy_source == "recipe"
                        )
                        if is_recipe_entry:
                            self._invalidate_manual_roi(mark_id, entry.roi_id, remove_selection=True)
                            cleared.append(f"{mark_id}-{LAYER_LABELS.get(layer, layer)}-{entry.roi_id}")
                        else:
                            retained.append(entry)
                    if layer == "upper":
                        mark.upper_rois = retained
                    else:
                        mark.lower_rois = retained
                    if not retained:
                        self.roi_sources.setdefault(mark_id, {})[layer] = "none"
            self._recipe_roi_confirmation_signature = None
            if cleared:
                self._append_log("已清除配方 ROI：" + "、".join(cleared))
            else:
                self._append_log("当前没有配方来源的 ROI。")
            self._refresh_all_widgets()

        def _image_for_layer(self, layer: str, mark_id: Optional[str] = None) -> Optional[ImageData]:
            mark_id = mark_id or self._current_mark_id()
            self._ensure_mark_runtime(mark_id)
            if self._current_mode() == "Single Image":
                return self._active_image_for_layer(mark_id, "upper")
            return self._active_image_for_layer(mark_id, layer)

        def _detect_one(self, mark: MarkRecipe, layer: str, roi_entry=None) -> DetectionResult:
            self._pull_config_from_ui()
            img = self._image_for_layer(layer, mark.mark_id)
            if img is None:
                raise ValueError(f"{LAYER_LABELS[layer]} 图像未导入")
            entries = mark.roi_entries(layer)
            roi_entry = roi_entry or (entries[0] if entries else None)
            roi = roi_entry.roi if roi_entry is not None else None
            if roi is None:
                raise ValueError(f"{mark.mark_id} {LAYER_LABELS[layer]} ROI 未设置")
            roi = self._coerce_roi_to_auto_ring(roi, layer)
            detection = detect_manual_roi(mark.mark_id, layer, img, roi, self.params, self.config)
            roi_index = entries.index(roi_entry) + 1
            detection.shape_params["roi_id"] = roi_entry.roi_id
            detection.shape_params["roi_index"] = roi_index
            detection.shape_params["roi_label"] = f"{LAYER_LABELS.get(layer, layer)} ROI {roi_index}"
            return detection

        def analyze_current_mark(self):
            mark_id = self.mark_combo.currentText()
            if not mark_id:
                return
            return self.analyze_roi_regions(show_message=True)

        def analyze_current_roi(self):
            # Backward-compatible entry. V1.2.5 uses batch ROI-region analysis.
            return self.analyze_roi_regions()

        def analyze_roi_regions(self, show_message: bool = True):
            mark_id = self.mark_combo.currentText() or self._current_mark_id()
            mark = self.marks[mark_id]
            images = {}
            roi_count = 0
            for layer in ("upper", "lower"):
                images[layer] = self._image_for_layer(layer, mark_id)
                if images[layer] is not None:
                    roi_count += len(mark.roi_entries(layer))
            if not roi_count:
                if show_message:
                    QMessageBox.warning(self, "分析 ROI 区域", "当前 Mark 没有可分析的 ROI 区域。请先导入图像并框选 ROI。")
                self._append_log(f"{mark_id} 分析 ROI 未开始：缺少图像或 ROI。")
                return 0
            self._pull_config_from_ui()
            return self._start_preview_job("manual", mark_id, images, show_message)

        def _start_preview_job(self, kind: str, mark_id: str, images: dict, show_message: bool):
            if self._calculation_running or self._preview_thread is not None:
                self._append_log("已有分析任务正在运行。")
                return 0
            job = {
                "kind": kind,
                "mark_id": mark_id,
                "config": deepcopy(self.config),
                "params": deepcopy(self.params),
                "mark": deepcopy(self.marks[mark_id]),
                "images": dict(images),
                "selection": self._selection_snapshot(mark_id),
            }
            self._preview_show_message = bool(show_message)
            self._preview_final_stage = ""
            self._set_calculation_running(True)
            self.progress_stage_label.setText("当前阶段：正在准备预览分析")
            thread = QThread(self)
            worker = PreviewWorker(job)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.progress.connect(self._on_calculation_progress, Qt.QueuedConnection)
            worker.finished.connect(self._on_preview_completed, Qt.QueuedConnection)
            worker.failed.connect(self._on_preview_failed, Qt.QueuedConnection)
            worker.cancelled.connect(self._on_preview_cancelled, Qt.QueuedConnection)
            for signal in (worker.finished, worker.failed, worker.cancelled):
                signal.connect(thread.quit)
                signal.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._on_preview_thread_finished, Qt.QueuedConnection)
            self._preview_thread = thread
            self._preview_worker = worker
            thread.start()
            return 1

        def _on_preview_completed(self, payload: dict):
            mark_id = payload["mark_id"]
            measured = payload["measured"]
            if payload["kind"] == "auto":
                self.auto_candidates_by_mark[mark_id] = measured["candidates"]
                self.auto_detections_by_mark[mark_id] = measured["detections"]
                self.auto_selections[mark_id] = measured["selection"]
                self.auto_overlays.pop(mark_id, None)
                warnings = measured.get("warnings", [])
                valid_count = sum(
                    next(iter(layer_map.values())).shape_params.get("measurement_eligible", False)
                    for layer_map in measured["detections"].values()
                )
                self._append_log(
                    f"自动识别完成：{len(measured['detections'])} 个候选，{valid_count} 个可用于测量。"
                )
                if warnings:
                    self._append_log("；".join(warnings))
                self._preview_final_stage = f"当前阶段：{mark_id} 自动识别完成"
            else:
                self.roi_detections[mark_id] = measured["detections"]
                self._rebuild_legacy_manual_detections(mark_id)
                self.auto_selections[mark_id] = measured["selection"]
                self.overlays.pop(mark_id, None)
                failures = measured.get("failures", [])
                self.roi_detection_failures[mark_id] = failures
                if failures and self._preview_show_message:
                    dialog = QMessageBox.warning if measured["detections"] else QMessageBox.critical
                    dialog(
                        self,
                        "ROI 区域部分完成" if measured["detections"] else "ROI 区域分析失败",
                        "\n".join(
                            f"{LAYER_LABELS.get(item['layer'], item['layer'])} {item['roi_label']}："
                            f"{self._friendly_error(Exception(item['error']))}"
                            for item in failures
                        ),
                    )
                self._append_log(
                    f"{mark_id} ROI 分析完成：{len(measured['detections'])} 个成功，{len(failures)} 个失败。"
                )
                self._preview_final_stage = (
                    f"当前阶段：{mark_id} ROI 分析失败"
                    if not measured["detections"]
                    else f"当前阶段：{mark_id} ROI 部分完成" if failures
                    else f"当前阶段：{mark_id} ROI 分析完成"
                )
            self._refresh_auto_selection_combos()
            self._refresh_all_widgets()

        def _on_preview_failed(self, message: str):
            self.runtime_logger.error("Preview analysis failed: %s", message)
            self._preview_final_stage = "当前阶段：ROI 分析失败"
            QMessageBox.critical(self, "分析失败", self._friendly_error(Exception(message)))

        def _on_preview_cancelled(self):
            self._preview_final_stage = "当前阶段：预览分析已取消"
            self._append_log("预览分析已取消。")

        def _on_preview_thread_finished(self):
            self._preview_worker = None
            self._preview_thread = None
            self._set_calculation_running(False)
            self._refresh_all_widgets()
            if getattr(self, "_preview_final_stage", ""):
                self.progress_stage_label.setText(self._preview_final_stage)
                self._preview_final_stage = ""

        def _recipe_roi_usage(self) -> list[str]:
            if self._is_auto_workflow():
                return []
            usage = []
            for mark_id in ("Mark1", "Mark2"):
                if self._is_batch_mode():
                    images = self.batch_images.get(mark_id, {})
                    has_upper = bool(images.get("upper"))
                else:
                    has_upper = self._active_image_for_layer(mark_id, "upper") is not None
                if not has_upper:
                    continue
                for layer in ("upper", "lower"):
                    mark = self.marks.get(mark_id)
                    if mark is None:
                        continue
                    for roi_index, entry in enumerate(mark.roi_entries(layer), start=1):
                        legacy_source = self.roi_sources.get(mark_id, {}).get(layer, "none")
                        if entry.source == "recipe" or (roi_index == 1 and legacy_source == "recipe"):
                            usage.append(f"{mark_id} {LAYER_LABELS.get(layer, layer)} ROI {roi_index}")
            return usage

        def _confirm_recipe_rois(self) -> bool:
            usage = self._recipe_roi_usage()
            if not usage:
                return True
            signature = tuple(usage)
            if signature == self._recipe_roi_confirmation_signature:
                return True
            answer = QMessageBox.question(
                self,
                "确认配方 ROI",
                "本次手动测量仍会使用以下配方 ROI：\n\n"
                + "\n".join(f"• {item}" for item in usage)
                + "\n\n继续使用这些 ROI 计算吗？\n如不需要，请取消后清除或重新框选对应 ROI。",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer == QMessageBox.Yes:
                self._recipe_roi_confirmation_signature = signature
                return True
            return False

        def _calculation_job_snapshot(self) -> dict:
            self._pull_config_from_ui()
            return {
                "config": deepcopy(self.config),
                "params": deepcopy(self.params),
                "marks": deepcopy(self.marks),
                "mark_images": self._mark_images_snapshot_for_current_run(),
                "batch_images": {
                    mark_id: {layer: list(self.batch_images[mark_id][layer]) for layer in ("upper", "lower")}
                    for mark_id in ("Mark1", "Mark2")
                },
                "selections": {
                    mark_id: self._selection_snapshot(mark_id)
                    for mark_id in self.marks
                },
                "batch": self._is_batch_mode(),
                "roi_sources": deepcopy(self.roi_sources),
                "geometry_program": self.geometry_program_snapshot(),
                "traceability": {
                    "recipe_path": self.loaded_recipe_path,
                    "recipe_hash": self.loaded_recipe_hash,
                    "input_paths": self._all_input_paths(),
                    "operation_mode": self.operation_mode,
                },
            }

        def _selection_snapshot(self, mark_id: str) -> dict:
            """Return the recipe-owned contour selection used by measurement jobs."""
            mark = self.marks.get(mark_id)
            if mark is None:
                return {"reference_label": "", "target_label": ""}
            return {
                "reference_label": mark.reference_contour_id or "",
                "target_label": mark.target_contour_id or "",
            }

        def _set_calculation_running(self, running: bool):
            self._calculation_running = bool(running)
            self.progress_bar.setVisible(True)
            self.progress_stage_label.setVisible(True)
            self.cancel_progress_btn.setVisible(running)
            self.cancel_progress_btn.setEnabled(running)
            if running:
                self.progress_bar.setValue(0)
                if hasattr(self, "status_task_dot"):
                    self.status_task_dot.setStyleSheet("color: #007AFF;")
                    self.status_task_label.setText("任务状态：正在离线计算")
            for button in (
                self.import_upper_btn, self.import_lower_btn, self.load_recipe_btn, self.analyze_all_btn, self.export_btn,
                self.analyze_roi_btn, self.auto_detect_btn, self.reset_measurement_btn,
                self.change_engineering_password_btn, self.import_images_btn, self.more_actions_btn,
                self.recent_images_btn, self.image_viewer_btn,
            ):
                button.setEnabled(not running)
            self.import_upper_action.setEnabled(not running)
            self.import_lower_action.setEnabled(not running and self._current_mode() == "Dual Image")
            self.reset_measurement_action.setEnabled(not running)
            self.fit_view_action.setEnabled(not running)
            self.operation_mode_combo.setEnabled(not running)
            self.side_tabs.setEnabled(not running)
            for table_name in ("det_table", "overlay_table", "geometry_table", "repeat_table"):
                table = getattr(self, table_name, None)
                if table is not None and hasattr(table, "set_delete_allowed"):
                    table.set_delete_allowed(self.operation_mode == "Engineering" and not running)
            self._update_roi_edit_lock()
            self._apply_operation_mode()

        def _production_preflight_errors(self) -> list[str]:
            errors: list[str] = []
            if self.operation_mode == "Production":
                if not self.loaded_recipe_path:
                    errors.append("生产模式必须加载配方")
                if self.config.recipe_validation_status != "Validated":
                    errors.append("生产模式只能使用已验证配方")
                if self.recipe_integrity_status != "Verified":
                    errors.append("生产配方尚未签章或哈希未验证")
                if not self.config.material_code.strip():
                    errors.append("物料编码不能为空")
                if not self.config.operator_name.strip():
                    errors.append("操作人员不能为空")
            if self._is_batch_mode():
                errors.extend(validate_batch_pairing(self.batch_images, self.config.mode == "Dual Image"))
            return errors

        def _recovery_payload(self) -> dict:
            return {
                "recipe_path": self.loaded_recipe_path,
                "mode": self.config.mode,
                "batch": self._is_batch_mode(),
                "mark_images": {
                    mark_id: {
                        layer: image.path if image else ""
                        for layer, image in layers.items()
                    }
                    for mark_id, layers in self.mark_images.items()
                },
                "batch_images": {
                    mark_id: {
                        layer: [image.path for image in images]
                        for layer, images in layers.items()
                    }
                    for mark_id, layers in self.batch_images.items()
                },
            }

        def _offer_recovery(self):
            pending = self.recovery_store.load()
            if not pending:
                return
            answer = QMessageBox.question(
                self,
                "恢复未完成任务",
                f"发现 {pending.get('saved_at', '')} 保存的未完成测量任务。\n是否恢复其配方和图像列表？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer != QMessageBox.Yes:
                self.recovery_store.clear()
                return
            try:
                recipe_path = pending.get("recipe_path", "")
                if recipe_path and Path(recipe_path).exists():
                    self._load_recipe_from_path(recipe_path, confirm_switch=False, show_message=False)
                self._set_mode_ui(pending.get("mode", "Single Image"))
                self._set_combo_value(self.measurement_run_mode_combo, "Batch" if pending.get("batch") else "Single")
                for mark_id, layers in pending.get("mark_images", {}).items():
                    for layer, path in layers.items():
                        if path and Path(path).exists():
                            self._set_image_for_layer(mark_id, layer, load_image(path), "single")
                for mark_id, layers in pending.get("batch_images", {}).items():
                    for layer, paths in layers.items():
                        images = [BatchImageRef.from_path(path) for path in paths if path and Path(path).exists()]
                        self._append_batch_image_data(mark_id, layer, images)
                self._sync_current_mark_images()
                self._refresh_all_widgets()
                self._append_log("已恢复上次未完成任务的配方和图像列表。")
            except Exception as exc:
                self.runtime_logger.exception("Recovery failed")
                QMessageBox.warning(self, "恢复失败", str(exc))
            finally:
                self.recovery_store.clear()

        def _all_input_paths(self) -> list[str]:
            paths: list[str] = []
            for layers in self.mark_images.values():
                paths.extend(image.path for image in layers.values() if image and image.path)
            for layers in self.batch_images.values():
                for images in layers.values():
                    paths.extend(image.path for image in images if image.path)
            return list(dict.fromkeys(paths))

        def _archive_completed_measurement(self, payload: dict):
            self.last_measurement_id = str(payload.get("measurement_id", ""))
            self.last_archive_path = str(payload.get("archive_path", ""))
            if not self.last_archive_path:
                return
            archive_path = Path(self.last_archive_path)
            # QWidget.grab() can crash the Qt offscreen plugin used by automated
            # tests. Production windows are visible and keep the trace screenshot.
            if QApplication.platformName().lower() == "offscreen" or not self.isVisible():
                self.runtime_logger.info(
                    "Measurement archived without UI screenshot: %s at %s",
                    self.last_measurement_id,
                    archive_path,
                )
                return
            self.grab().save(str(archive_path / "measurement_screen.png"))
            self.runtime_logger.info("Measurement archived: %s at %s", self.last_measurement_id, archive_path)

        def _on_calculation_timeout(self):
            if not self._calculation_running:
                return
            self._calculation_timed_out = True
            self.runtime_logger.error("Measurement timeout after %s seconds", self.params.measurement_timeout_s)
            if self._calculation_worker is not None:
                self._calculation_worker.cancel()
            self.cancel_progress_btn.setEnabled(False)
            self.progress_stage_label.setText("当前阶段：任务超时，正在停止当前算法步骤")

        def _start_measurement_job(self):
            if self._calculation_running:
                return
            self._pull_config_from_ui()
            preflight_errors = self._production_preflight_errors()
            if preflight_errors:
                QMessageBox.warning(self, "计算前检查未通过", "请先处理以下问题：\n\n" + "\n".join(f"• {item}" for item in preflight_errors))
                self.runtime_logger.warning("Measurement preflight rejected: %s", " | ".join(preflight_errors))
                return
            if not self._confirm_recipe_rois():
                self._append_log("已取消计算；配方 ROI 未确认。")
                return
            job = self._calculation_job_snapshot()
            self._active_recovery_job_id = self.recovery_store.save(self._recovery_payload())
            self._calculation_timed_out = False
            self.last_measurement_id = ""
            self.last_archive_path = ""
            self._set_calculation_running(True)
            self._calculation_timeout_timer.start(max(10, int(self.params.measurement_timeout_s)) * 1000)
            self.progress_stage_label.setText("当前阶段：正在准备后台计算")
            self._append_log(
                "计算路径：全图自动识别（不使用 ROI）"
                if self._is_auto_workflow()
                else "计算路径：手动 ROI；ROI 来源已锁定到本次任务快照"
            )
            thread = QThread(self)
            worker = MeasurementWorker(job)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            # These handlers live on a Python mixin rather than directly on the
            # QMainWindow class. Force queued delivery so every UI operation
            # remains on the GUI thread after the V1.8 module split.
            worker.progress.connect(self._on_calculation_progress, Qt.QueuedConnection)
            worker.finished.connect(self._on_calculation_completed, Qt.QueuedConnection)
            worker.failed.connect(self._on_calculation_failed, Qt.QueuedConnection)
            worker.cancelled.connect(self._on_calculation_cancelled, Qt.QueuedConnection)
            for signal in (worker.finished, worker.failed, worker.cancelled):
                signal.connect(thread.quit)
                signal.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._on_calculation_thread_finished, Qt.QueuedConnection)
            self._calculation_thread = thread
            self._calculation_worker = worker
            thread.start()

        def analyze_all_marks(self):
            return self._start_measurement_job()

        def cancel_calculation(self):
            if self._preview_worker is not None:
                self._preview_worker.cancel()
                self.cancel_progress_btn.setEnabled(False)
                self.progress_stage_label.setText("当前阶段：正在取消预览分析")
                return
            if self._calculation_worker is not None:
                self._calculation_worker.cancel()
                self.cancel_progress_btn.setEnabled(False)
                self.progress_stage_label.setText("当前阶段：正在取消，当前算法步骤完成后停止")

        def _on_calculation_progress(self, done: int, total: int, message: str):
            self.progress_bar.setValue(int(round(100.0 * done / max(1, total))))
            self.progress_stage_label.setText(f"当前阶段：{message}")
            if hasattr(self, "status_task_label"):
                self.status_task_label.setText(f"任务状态：{message}")
            self.statusBar().showMessage(message)

        def _on_calculation_completed(self, payload: dict):
            self.roi_detections = payload.get("detections", {})
            self.roi_detection_failures = payload.get("roi_failures", {"Mark1": [], "Mark2": []})
            self.detections = {}
            self._rebuild_legacy_manual_detections()
            self.overlays = payload.get("overlays", {})
            self.auto_candidates_by_mark = payload.get("auto_candidates", {"Mark1": {}, "Mark2": {}})
            self.auto_detections_by_mark = payload.get("auto_detections", {"Mark1": {}, "Mark2": {}})
            self.auto_overlays = payload.get("auto_overlays", {})
            self.auto_selections = payload.get("selections", self.auto_selections)
            for mark_id, selection in self.auto_selections.items():
                mark = self.marks.get(mark_id)
                if mark is not None:
                    mark.reference_contour_id = str(selection.get("reference_label", "") or "")
                    mark.target_contour_id = str(selection.get("target_label", "") or "")
            self.batch_overlays = payload.get("batch_overlays", {"Mark1": [], "Mark2": []})
            self.batch_run_records = payload.get("batch_records", {"Mark1": [], "Mark2": []})
            self.geometry_result = payload.get("geometry_result", GeometryRunResult())
            self.batch_geometry_results = payload.get("batch_geometry_results", [])
            self._refresh_batch_detail_selector(default_first=bool(payload.get("batch")))
            self._sync_current_mark_images()
            self._refresh_auto_selection_combos()
            self._refresh_all_widgets()
            try:
                self._archive_completed_measurement(payload)
            except Exception:
                self.runtime_logger.exception("Measurement archive failed")
                QMessageBox.warning(self, "追溯归档失败", "测量已完成，但自动追溯归档失败。请查看运行日志。")
            if payload.get("archive_error"):
                self.runtime_logger.error("Measurement archive failed: %s", payload["archive_error"])
                QMessageBox.warning(
                    self,
                    "追溯归档失败",
                    f"测量已完成，但自动追溯归档失败：\n{payload['archive_error']}\n\n请查看运行日志。",
                )
            self.recovery_store.finish(
                "completed",
                measurement_id=self.last_measurement_id,
                archive_path=self.last_archive_path,
            )
            if payload.get("batch") and hasattr(self, "result_tabs"):
                self.result_tabs.setCurrentIndex(0)
            result_map = self.auto_overlays if self._is_auto_workflow() else self.overlays
            lines = [
                f"{mark_id}: Dx={item.delta_x_um:+.3f} μm，Dy={item.delta_y_um:+.3f} μm，"
                f"Dxy={item.overlay_r_um:.3f} μm，判定={RESULT_LABELS.get(item.result, item.result)}"
                for mark_id, item in result_map.items()
            ]
            notes = list(payload.get("warnings", [])) + list(payload.get("skipped", []))
            geometry_count = len(self.geometry_result.measurements) + len(self.geometry_result.coordinate_labels)
            if lines:
                message = "计算完成：\n" + "\n".join(lines)
                if self.last_measurement_id:
                    message += f"\n\n测量编号：{self.last_measurement_id}"
                if notes:
                    message += "\n\n提示：\n" + "\n".join(notes)
                QMessageBox.information(self, "计算完成", message)
            elif geometry_count:
                invalid_count = sum(
                    item.status != "Valid" for item in self.geometry_result.measurements.values()
                ) + sum(
                    item.status != "Valid" for item in self.geometry_result.coordinate_labels.values()
                )
                message = f"测量程序完成：生成 {geometry_count} 项尺寸/坐标结果"
                if invalid_count:
                    message += f"，其中 {invalid_count} 项无效，请查看尺寸结果。"
                QMessageBox.information(self, "运行完成", message)
            else:
                QMessageBox.warning(self, "运行测量程序", "未生成任何对位或尺寸结果。\n" + "\n".join(notes))

        def _on_calculation_failed(self, message: str):
            self._calculation_timeout_timer.stop()
            self.recovery_store.finish("archived", error=message)
            self.runtime_logger.error("Measurement failed: %s", message)
            QMessageBox.critical(self, "计算失败", self._friendly_error(Exception(message)))

        def _on_calculation_cancelled(self):
            self._calculation_timeout_timer.stop()
            self.recovery_store.finish("archived", error="cancelled")
            if self._calculation_timed_out:
                QMessageBox.critical(self, "计算超时", "计算已超过设置的任务超时时间并停止。")
                self._append_log("计算超时并已停止。")
            else:
                self._append_log("计算已取消。")

        def _on_calculation_thread_finished(self):
            self._calculation_timeout_timer.stop()
            self._calculation_worker = None
            self._calculation_thread = None
            self._set_calculation_running(False)
            if self.progress_bar.value() >= 100:
                self.progress_stage_label.setText("当前阶段：计算完成")
            elif "取消" in self.progress_stage_label.text():
                self.progress_stage_label.setText("当前阶段：计算已取消")
            self._refresh_all_widgets()

        def _analyze_mark(self, mark_id: str, show_success: bool = True):
            self._pull_config_from_ui()
            mark = self.marks[mark_id]
            try:
                upper = self._detect_one(mark, "upper")
                lower = self._detect_one(mark, "lower")
                self.detections.setdefault(mark_id, {})["upper"] = upper
                self.detections.setdefault(mark_id, {})["lower"] = lower
                # 默认仍保留上层-下层计算；若用户在“基准/待测轮廓”中指定了对象，后续会用所选轮廓覆盖该结果。
                self.overlays[mark_id] = calculate_overlay(mark_id, upper, lower, self.config)
                self._refresh_auto_selection_combos()
            except Exception as exc:
                QMessageBox.critical(self, "分析失败", str(exc))
                return
            self._refresh_all_widgets()
            if show_success:
                QMessageBox.information(self, "分析完成", f"{mark_id} 分析完成。")

        def export_result_file(self):
            display_detections = self._display_detections()
            has_batch_details = self._is_batch_mode() and any(self.batch_run_records.values())
            has_geometry = bool(
                self.geometry_result.features
                or self.geometry_result.measurements
                or self.geometry_result.coordinate_labels
            )
            if not display_detections and not has_batch_details and not has_geometry:
                QMessageBox.warning(self, "无结果", "当前没有可导出的分析结果。")
                return
            self._pull_config_from_ui()
            path = self._choose_export_result_path()
            if not path:
                return
            if Path(path).suffix.lower() == ".csv":
                answer = QMessageBox.question(
                    self,
                    "CSV 导出限制",
                    "CSV 只能保存一个结果表，不包含识别明细、重复性、尺寸结果和 Mark 图片。\n\n仍要继续吗？",
                    QMessageBox.Yes | QMessageBox.Cancel,
                    QMessageBox.Cancel,
                )
                if answer != QMessageBox.Yes:
                    return
            try:
                rows = []
                if has_batch_details:
                    reference_names = []
                    target_names = []
                    for mark_id in ("Mark1", "Mark2"):
                        for record in self.batch_run_records.get(mark_id, []):
                            detected = record.get("detections", {})
                            selection = record.get("selection", {})
                            overlay = record.get("overlay")
                            if record.get("workflow") == "Auto":
                                reference = selection.get("reference_label", "")
                                target = selection.get("target_label", "")
                                if reference:
                                    reference_detection = next(iter(detected.get(reference, {}).values()), None)
                                    reference_names.append(f"{mark_id}-{candidate_display_label(reference, reference_detection)}")
                                if target:
                                    target_detection = next(iter(detected.get(target, {}).values()), None)
                                    target_names.append(f"{mark_id}-{candidate_display_label(target, target_detection)}")
                                named = {}
                                for label, layer_map in detected.items():
                                    detection = next(iter(layer_map.values()), None)
                                    named[f"{mark_id}-{candidate_display_label(label, detection)}"] = layer_map
                                overlay_key = ""
                                if target in detected:
                                    target_detection = next(iter(detected[target].values()), None)
                                    overlay_key = f"{mark_id}-{candidate_display_label(target, target_detection)}"
                                row_overlays = {overlay_key: overlay} if overlay_key and overlay else {}
                            else:
                                named = {mark_id: detected} if detected else {}
                                row_overlays = {mark_id: overlay} if overlay else {}
                            if named:
                                rows.extend(build_detection_rows(
                                    named,
                                    row_overlays,
                                    self.config,
                                    upper_file=record.get("upper_file", ""),
                                    lower_file=record.get("lower_file", ""),
                                    run_index=int(record.get("run_index", 0) or 0),
                                ))
                            for failure in record.get("roi_failures", []):
                                rows.append(build_detection_failure_row(
                                    self.config,
                                    int(record.get("run_index", 0) or 0),
                                    mark_id,
                                    record.get("upper_file", ""),
                                    record.get("lower_file", ""),
                                    failure.get("error", "ROI 识别失败"),
                                    layer=failure.get("layer", ""),
                                    roi_id=failure.get("roi_id", ""),
                                    roi_index=failure.get("roi_index"),
                                    roi_label=failure.get("roi_label", ""),
                                    status=failure.get("status", "Error"),
                                ))
                            if not named and not record.get("roi_failures"):
                                rows.append(build_detection_failure_row(
                                    self.config,
                                    int(record.get("run_index", 0) or 0),
                                    mark_id,
                                    record.get("upper_file", ""),
                                    record.get("lower_file", ""),
                                    record.get("error", "未生成识别结果"),
                                ))
                    if self._is_auto_workflow():
                        self.config.auto_reference_label = "；".join(reference_names)
                        self.config.auto_target_label = "；".join(target_names)
                elif self._is_auto_workflow():
                    reference_names = []
                    target_names = []
                    for mark_id, detected in self.auto_detections_by_mark.items():
                        selection = self.auto_selections.get(mark_id, {})
                        reference = selection.get("reference_label", "")
                        target = selection.get("target_label", "")
                        if reference:
                            reference_detection = next(iter(detected.get(reference, {}).values()), None)
                            reference_names.append(f"{mark_id}-{candidate_display_label(reference, reference_detection)}")
                        if target:
                            target_detection = next(iter(detected.get(target, {}).values()), None)
                            target_names.append(f"{mark_id}-{candidate_display_label(target, target_detection)}")
                        named = {
                            f"{mark_id}-{candidate_display_label(label, next(iter(layer_map.values()), None))}": layer_map
                            for label, layer_map in detected.items()
                        }
                        row_overlays = {}
                        if target and mark_id in self.auto_overlays:
                            target_detection = next(iter(detected.get(target, {}).values()), None)
                            row_overlays[f"{mark_id}-{candidate_display_label(target, target_detection)}"] = self.auto_overlays[mark_id]
                        upper = self._image_for_layer("upper", mark_id)
                        lower = self._image_for_layer("lower", mark_id) if self._current_mode() == "Dual Image" else None
                        rows.extend(build_detection_rows(
                            named,
                            row_overlays,
                            self.config,
                            upper_file=upper.path if upper else "",
                            lower_file=lower.path if lower else "",
                        ))
                    self.config.auto_reference_label = "；".join(reference_names)
                    self.config.auto_target_label = "；".join(target_names)
                else:
                    reference_names = []
                    target_names = []
                    for mark_id, layer_map in self.roi_detections.items():
                        selection = self.auto_selections.get(mark_id, {})
                        reference_id = selection.get("reference_label", "")
                        target_id = selection.get("target_label", "")
                        if reference_id:
                            reference_names.append(f"{mark_id}/{reference_id}")
                        if target_id:
                            target_names.append(f"{mark_id}/{target_id}")
                        upper = self._image_for_layer("upper", mark_id)
                        lower = self._image_for_layer("lower", mark_id) if self._current_mode() == "Dual Image" else None
                        rows.extend(build_detection_rows(
                            {mark_id: layer_map},
                            {mark_id: self.overlays[mark_id]} if mark_id in self.overlays else {},
                            self.config,
                            upper_file=upper.path if upper else "",
                            lower_file=lower.path if lower else "",
                        ))
                        for failure in self.roi_detection_failures.get(mark_id, []):
                            rows.append(build_detection_failure_row(
                                self.config,
                                1,
                                mark_id,
                                upper.path if upper else "",
                                lower.path if lower else "",
                                failure.get("error", "ROI 识别失败"),
                                layer=failure.get("layer", ""),
                                roi_id=failure.get("roi_id", ""),
                                roi_index=failure.get("roi_index"),
                                roi_label=failure.get("roi_label", ""),
                                status=failure.get("status", "Error"),
                            ))
                    self.config.auto_reference_label = "; ".join(reference_names)
                    self.config.auto_target_label = "; ".join(target_names)
                with TemporaryDirectory() as tmp_dir:
                    mark_images = self._build_mark_image_exports(tmp_dir)
                    geometry_rows = []
                    if self.batch_geometry_results:
                        for index, geometry_result in enumerate(self.batch_geometry_results, start=1):
                            geometry_rows.extend(build_geometry_rows(geometry_result, self.config, index))
                    else:
                        geometry_rows = build_geometry_rows(self.geometry_result, self.config)
                    export_results(
                        path,
                        rows,
                        config=self.config,
                        summary_rows=self._build_summary_rows(),
                        mark_images=mark_images,
                        repeatability_rows=self._build_repeatability_export_rows(),
                        geometry_rows=geometry_rows,
                        traceability_info={
                            "measurement_id": self.last_measurement_id,
                            "operation_mode": "生产模式" if self.operation_mode == "Production" else "工程模式",
                            "recipe_hash": self.loaded_recipe_hash,
                            "archive_path": self.last_archive_path,
                        },
                    )
                QMessageBox.information(self, "导出完成", f"结果已导出：\n{path}")
                self.dialog_state_store.remember_file("export_result", path)
            except Exception as exc:
                QMessageBox.critical(self, "导出失败", str(exc))

        def _choose_export_result_path(self) -> str:
            directory = self.dialog_state_store.directory("export_result")
            default_path = str(Path(directory) / self._default_export_filename()) if directory else self._default_export_filename()
            dialog = QFileDialog(self, "导出结果", default_path)
            dialog.setAcceptMode(QFileDialog.AcceptSave)
            dialog.setNameFilter("Excel (*.xlsx);;CSV (*.csv)")
            dialog.selectFile(default_path)
            return dialog.selectedFiles()[0] if dialog.exec() == QDialog.Accepted and dialog.selectedFiles() else ""
