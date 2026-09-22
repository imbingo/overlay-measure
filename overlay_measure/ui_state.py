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
from .batch_image_store import resolve_image
from .candidate_ordering import candidate_display_label
from .export_naming import build_export_filename
from .image_loader import SUPPORTED_EXTENSIONS, display_to_uint8, load_image
from .measurement_engine import run_measurement_job
from .geometry_models import GeometryRunResult
from .measurement_service import attach_algorithm_path, describe_algorithm_path, detect_manual_roi
from .measurement_units import ellipse_metrics_um, rotated_rect_size_um
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
from .result_exporter import build_detection_rows, export_results
from .rz_calculator import build_summary_rows
from .runtime_support import RecoveryStore, build_runtime_logger

from .ui_constants import LAYER_LABELS, RESULT_LABELS, STEP_TITLES
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
from .ui_workers import MeasurementWorker


class MainWindowStateMixin:
        def on_operation_mode_changed(self):
            requested = self.operation_mode_combo.currentData() or "Production"
            if requested == self.operation_mode:
                return
            if requested == "Engineering":
                password, accepted = QInputDialog.getText(
                    self,
                    "进入工程模式",
                    "请输入工程模式密码：",
                    QLineEdit.Password,
                )
                if not accepted or not self.access_controller.verify(password):
                    self.operation_mode_combo.blockSignals(True)
                    self._set_combo_value(self.operation_mode_combo, "Production")
                    self.operation_mode_combo.blockSignals(False)
                    self.runtime_logger.warning("Engineering mode authentication failed")
                    if accepted:
                        QMessageBox.warning(self, "密码错误", "工程模式密码不正确。")
                    return
            self.operation_mode = requested
            self.runtime_logger.info("Operation mode changed to %s", requested)
            self._apply_operation_mode()

        def change_engineering_password(self):
            if self.operation_mode != "Engineering":
                QMessageBox.warning(self, "权限不足", "请先验证密码并切换到工程模式。")
                return
            current, accepted = QInputDialog.getText(
                self, "修改工程模式密码", "请输入当前密码：", QLineEdit.Password
            )
            if not accepted:
                return
            if not self.access_controller.verify(current):
                QMessageBox.warning(self, "密码错误", "当前工程模式密码不正确。")
                return
            new_password, accepted = QInputDialog.getText(
                self, "修改工程模式密码", "请输入新密码（至少 6 个字符）：", QLineEdit.Password
            )
            if not accepted:
                return
            confirmation, accepted = QInputDialog.getText(
                self, "修改工程模式密码", "请再次输入新密码：", QLineEdit.Password
            )
            if not accepted:
                return
            if new_password != confirmation:
                QMessageBox.warning(self, "两次输入不一致", "两次输入的新密码不一致。")
                return
            try:
                self.access_controller.set_password(new_password)
            except ValueError as exc:
                QMessageBox.warning(self, "密码无效", str(exc))
                return
            self.runtime_logger.info("Engineering mode password changed")
            QMessageBox.information(self, "修改完成", "工程模式密码已更新。")

        def _set_operation_mode(self, mode: str, *, authenticated: bool = False):
            if mode == "Engineering" and not authenticated:
                raise PermissionError("切换工程模式需要身份验证")
            self.operation_mode = mode
            self.operation_mode_combo.blockSignals(True)
            self._set_combo_value(self.operation_mode_combo, mode)
            self.operation_mode_combo.blockSignals(False)
            self._apply_operation_mode()

        def _apply_operation_mode(self):
            if not hasattr(self, "side_tabs"):
                return
            engineering = self.operation_mode == "Engineering"
            self.recipe_manage_btn.setVisible(True)
            self.save_recipe_btn.setVisible(True)
            self.recipe_manage_action.setVisible(False)
            self.save_recipe_action.setVisible(False)
            self.image_mode_label.setVisible(True)
            self.mode_combo.setVisible(True)
            self.display_enhance_check.setVisible(True)
            self.analyze_roi_btn.setVisible(True)
            self.import_upper_btn.setVisible(True)
            self.import_lower_btn.setVisible(True)
            self.reset_measurement_btn.setVisible(True)
            self.side_tabs.setTabEnabled(2, engineering)
            self.side_tabs.setTabEnabled(3, engineering)
            self.side_tabs.setTabEnabled(4, engineering)
            self.save_recipe_btn.setEnabled(engineering and not self._calculation_running)
            self.upper_canvas.image_drop_enabled = not self._calculation_running
            self.lower_canvas.image_drop_enabled = not self._calculation_running
            self.recipe_manage_action.setEnabled(engineering and not self._calculation_running)
            self.save_recipe_action.setEnabled(engineering and not self._calculation_running)
            if hasattr(self, "image_diagnostics_action"):
                self.image_diagnostics_action.setEnabled(engineering and not self._calculation_running)
            self.analyze_roi_btn.setEnabled(
                not self._calculation_running and not self._is_auto_workflow()
            )
            self.diagnostic_check.setEnabled(engineering)
            if hasattr(self, "image_diagnostics_btn"):
                self.image_diagnostics_btn.setEnabled(engineering and not self._calculation_running)
            self.change_engineering_password_btn.setEnabled(engineering and not self._calculation_running)
            for widget in (
                self.material_code_edit,
                self.recipe_name_edit,
                self.recipe_version_edit,
                self.recipe_status_combo,
                self.process_name_edit,
                self.equipment_model_edit,
                self.calibration_date_edit,
            ):
                widget.setEnabled(engineering)
            if not engineering and self.side_tabs.currentIndex() in {2, 3, 4}:
                self.side_tabs.setCurrentIndex(1)
            color = "#2468B2" if engineering else "#248A3D"
            background = "#EAF3FD" if engineering else "#F1F7F3"
            self.operation_mode_combo.setStyleSheet(
                f"color: {color}; background: {background}; font-weight: 600;"
            )
            self.operation_mode_combo.setToolTip(
                "工程模式：允许修改 ROI、算法参数和配方。" if engineering
                else "生产模式：配方与算法参数已锁定。切换工程模式需要密码。"
            )
            for table_name in ("det_table", "overlay_table", "geometry_table", "repeat_table"):
                table = getattr(self, table_name, None)
                if table is not None and hasattr(table, "set_delete_allowed"):
                    table.set_delete_allowed(engineering and not self._calculation_running)
            self._update_roi_edit_lock()

        def _update_roi_edit_lock(self):
            allowed = self.operation_mode == "Engineering" and not self._calculation_running
            for canvas in (self.upper_canvas, self.lower_canvas):
                canvas.roi_editing_enabled = allowed

        def _roi_history_snapshot(self):
            return (
                deepcopy(self.marks),
                deepcopy(self.auto_selections),
                deepcopy(self.roi_sources),
            )

        def _initialize_roi_parameter_edit(self):
            self._roi_parameter_edit_active = False
            self._roi_parameter_edit_timer = QTimer(self)
            self._roi_parameter_edit_timer.setSingleShot(True)
            self._roi_parameter_edit_timer.setInterval(300)
            self._roi_parameter_edit_timer.timeout.connect(self._finish_roi_parameter_edit)

        def _push_roi_undo(self):
            if self._restoring_roi_history or self.operation_mode != "Engineering":
                return
            self._roi_undo_stack.append(self._roi_history_snapshot())
            del self._roi_undo_stack[:-20]
            self._roi_redo_stack.clear()

        def _restore_roi_history(self, snapshot):
            self._restoring_roi_history = True
            try:
                self.marks, self.auto_selections, self.roi_sources = deepcopy(snapshot)
                self.invalidate_measurement_state(
                    "ROI 历史已恢复",
                    clear_selections=False,
                    clear_batch_results=True,
                )
                self._refresh_roi_index_combo()
                self._refresh_auto_selection_combos()
                self._refresh_all_widgets()
            finally:
                self._restoring_roi_history = False

        def undo_roi_change(self):
            if self.operation_mode != "Engineering" or not self._roi_undo_stack:
                return
            self._roi_redo_stack.append(self._roi_history_snapshot())
            self._restore_roi_history(self._roi_undo_stack.pop())

        def redo_roi_change(self):
            if self.operation_mode != "Engineering" or not self._roi_redo_stack:
                return
            self._roi_undo_stack.append(self._roi_history_snapshot())
            self._restore_roi_history(self._roi_redo_stack.pop())

        def invalidate_measurement_state(
            self,
            reason: str = "",
            *,
            clear_selections: bool = True,
            clear_batch_results: bool = True,
        ):
            """Clear every result derived from image, ROI, recipe, units or algorithm state."""
            self.detections.clear()
            self.roi_detections = {"Mark1": {}, "Mark2": {}}
            self.roi_detection_failures = {"Mark1": [], "Mark2": []}
            self.overlays.clear()
            self.auto_detections_by_mark = {"Mark1": {}, "Mark2": {}}
            self.auto_candidates_by_mark = {"Mark1": {}, "Mark2": {}}
            self.auto_overlays.clear()
            self.geometry_result = GeometryRunResult()
            self.batch_geometry_results = []
            if clear_batch_results:
                self.batch_overlays = {"Mark1": [], "Mark2": []}
                self.batch_run_records = {"Mark1": [], "Mark2": []}
                self._batch_detail_run_index = 1
                self._batch_detail_last_single_index = 1
            if clear_selections:
                self.auto_selections = {
                    "Mark1": {"reference_label": "", "target_label": ""},
                    "Mark2": {"reference_label": "", "target_label": ""},
                }
                for mark in self.marks.values():
                    mark.reference_contour_id = ""
                    mark.target_contour_id = ""
            for canvas in (self.upper_canvas, self.lower_canvas):
                canvas.clear_caliper_selection(update=False)
            if reason:
                self._append_log(f"{reason}，旧测量结果已失效，请重新运行测量程序。")

        def on_display_enhancement_changed(self, checked: bool):
            self.upper_canvas.set_display_enhancement(checked)
            self.lower_canvas.set_display_enhancement(checked)
            self._append_log("已开启显示增强。" if checked else "已关闭显示增强。")

        def _warn_default_engineering_password(self):
            if not self.access_controller.should_warn_default_password():
                return
            QMessageBox.warning(
                self,
                "工程密码安全提示",
                "当前仍使用默认工程模式密码 admin123。正式投入产线前，请进入工程模式并修改密码。",
            )
            self.access_controller.acknowledge_default_password_warning()

        def _append_log(self, message: str):
            # V1.2.5：取消独立日志窗口，所有操作反馈统一显示在左下角状态栏，避免界面拥挤。
            self.statusBar().showMessage(message, 3500)

        def _friendly_error(self, exc: Exception) -> str:
            message = str(exc).strip()
            if not message:
                return "分析发生未知异常。请检查图像、ROI 和算法参数后重试。"
            if "有效边缘点不足" in message or "最少边缘点数" in message:
                return (
                    "有效边缘点不足。请确认 ROI 覆盖目标边缘，适当扩大搜索环带、"
                    "降低最小梯度，或切换到宽容质量门槛。\n"
                    f"详细信息：{message}"
                )
            if "卡尺找圆" in message and "不足" in message:
                return (
                    "卡尺圆找到的有效边缘点不足。请收窄到正确边缘、检查搜索方向和边缘极性，"
                    "或适当降低最小梯度。\n"
                    f"详细信息：{message}"
                )
            if "残差" in message:
                return (
                    "拟合残差超过当前要求。请确认 ROI 是否混入其他边缘；来料加工质量较差时，"
                    "可由工程人员评估后使用宽容质量门槛。\n"
                    f"详细信息：{message}"
                )
            if "ROI" in message:
                return f"ROI 分析失败。请确认已导入对应图像，并且 ROI 框住目标特征。\n详细信息：{message}"
            return message

        def show_algorithm_path_dialog(self):
            text = getattr(self, "algorithm_path_text", "暂无测量结果；分析 ROI 或自动识别后可查看实际算法路径。")
            QMessageBox.information(self, "算法路径", text.replace("；", "\n\n"))

        def _install_button_feedback(self):
            for button in self.findChildren(QPushButton):
                if button.property("feedback_installed"):
                    continue
                button.setProperty("feedback_installed", True)
                button.clicked.connect(lambda checked=False, b=button: self._append_log(f"点击按钮：{b.text().replace('&', '')}"))
                button.setToolTip(button.toolTip() or f"点击执行：{button.text().replace('&', '')}")

        def _pull_config_from_ui(self):
            self.config.mode = self._current_mode()
            self.config.workflow_mode = self._combo_value(self.workflow_combo)
            self.config.auto_reference_label = self.auto_reference_combo.currentData() or ""
            self.config.auto_target_label = self.auto_target_combo.currentData() or ""
            self.config.material_code = self.material_code_edit.text().strip()
            self.config.recipe_name = self.recipe_name_edit.text().strip()
            self.config.recipe_version = self.recipe_version_edit.text().strip()
            self.config.recipe_validation_status = self._combo_value(self.recipe_status_combo)
            self.config.process_name = self.process_name_edit.text().strip()
            self.config.equipment_model = self.equipment_model_edit.text().strip()
            self.config.calibration_date = self.calibration_date_edit.text().strip()
            self.config.operator_name = self.operator_name_edit.text().strip()
            self.config.pixel_size_x_um = self.pixel_x_spin.value()
            self.config.pixel_size_y_um = self.pixel_y_spin.value()
            self.config.registration_offset_x_um = self.offset_x_spin.value()
            self.config.registration_offset_y_um = self.offset_y_spin.value()
            self.config.rx_angle_urad = self.rx_angle_spin.value()
            self.config.ry_angle_urad = self.ry_angle_spin.value()
            self.config.material_thickness_mm = self.material_thickness_spin.value()
            self.config.delta_x_limit_um = self.dx_limit_spin.value()
            self.config.delta_y_limit_um = self.dy_limit_spin.value()
            self.config.overlay_r_limit_um = self.r_limit_spin.value()
            self.config.quality_profile = self._combo_value(self.quality_profile_combo)
            self.config.confidence_min = self.conf_min_spin.value()
            self.config.rz_layout = self.rz_layout_combo.currentText()
            self.config.rz_distance_l_um = self.rz_l_spin.value()
            self.config.rz_limit = self.rz_limit_spin.value()
            self.config.production_caliper_count = self.caliper_count_spin.value()
            self.config.production_caliper_width_px = self.caliper_width_spin.value()
            self.config.production_search_half_width_px = self.production_search_spin.value()
            self.config.production_min_coverage = self.production_coverage_spin.value()
            self.config.production_max_rejected_ratio = self.production_reject_spin.value()
            self.config.production_max_residual_um = self.production_residual_spin.value()
            self.config.production_max_radial_deviation_um = self.production_deviation_spin.value()

            self.params.gaussian_sigma_px = self.sigma_spin.value()
            self.params.canny_low = self.canny_low_spin.value()
            self.params.canny_high = self.canny_high_spin.value()
            self.params.min_gradient = self.min_gradient_spin.value()
            self.params.profile_half_width_px = self.profile_half_spin.value()
            self.params.profile_step_px = self.profile_step_spin.value()
            self.params.fitting_mode = self._combo_value(self.fit_mode_combo)
            self.params.upper_fitting_mode = self._combo_value(self.upper_fit_mode_combo)
            self.params.lower_fitting_mode = self._combo_value(self.lower_fit_mode_combo)
            self.params.use_ransac = self.ransac_check.isChecked()
            self.params.residual_limit_px = self.residual_limit_spin.value()
            self.params.min_edge_points = self.min_edge_points_spin.value()
            self.params.polarity = self._combo_value(self.polarity_combo)
            self.params.measurement_timeout_s = self.measurement_timeout_spin.value()
            self.params.auto_refine_roi_type = self._combo_value(self.roi_type_combo)

        def _push_config_to_ui(self):
            self._set_mode_ui(self.config.mode)
            self._set_combo_value(self.workflow_combo, getattr(self.config, "workflow_mode", "Manual"))
            self.material_code_edit.setText(getattr(self.config, "material_code", ""))
            self.recipe_name_edit.setText(getattr(self.config, "recipe_name", ""))
            self.recipe_version_edit.setText(getattr(self.config, "recipe_version", "1.0"))
            self._set_combo_value(self.recipe_status_combo, getattr(self.config, "recipe_validation_status", "Draft"))
            self.process_name_edit.setText(getattr(self.config, "process_name", ""))
            self.equipment_model_edit.setText(getattr(self.config, "equipment_model", ""))
            self.calibration_date_edit.setText(getattr(self.config, "calibration_date", ""))
            self.operator_name_edit.setText(getattr(self.config, "operator_name", ""))
            self.pixel_x_spin.setValue(self.config.pixel_size_x_um)
            self.pixel_y_spin.setValue(self.config.pixel_size_y_um)
            self.offset_x_spin.setValue(self.config.registration_offset_x_um)
            self.offset_y_spin.setValue(self.config.registration_offset_y_um)
            self.rx_angle_spin.setValue(getattr(self.config, "rx_angle_urad", 0.0))
            self.ry_angle_spin.setValue(getattr(self.config, "ry_angle_urad", 0.0))
            self.material_thickness_spin.setValue(getattr(self.config, "material_thickness_mm", 0.0))
            self.dx_limit_spin.setValue(self.config.delta_x_limit_um)
            self.dy_limit_spin.setValue(self.config.delta_y_limit_um)
            self.r_limit_spin.setValue(self.config.overlay_r_limit_um)
            self.quality_profile_combo.blockSignals(True)
            self._set_combo_value(
                self.quality_profile_combo,
                getattr(self.config, "quality_profile", "Standard"),
            )
            self.quality_profile_combo.blockSignals(False)
            self._updating_quality_controls = True
            self.conf_min_spin.setValue(self.config.confidence_min)
            self.rz_layout_combo.setCurrentText(getattr(self.config, "rz_layout", "Y向前后分布"))
            self.rz_l_spin.setValue(getattr(self.config, "rz_distance_l_um", 1.0))
            self.rz_limit_spin.setValue(getattr(self.config, "rz_limit", 999999.0))
            self.caliper_count_spin.setValue(getattr(self.config, "production_caliper_count", 64))
            self.caliper_width_spin.setValue(getattr(self.config, "production_caliper_width_px", 8.0))
            self.production_search_spin.setValue(getattr(self.config, "production_search_half_width_px", 8.0))
            self.production_coverage_spin.setValue(getattr(self.config, "production_min_coverage", 0.65))
            self.production_reject_spin.setValue(getattr(self.config, "production_max_rejected_ratio", 0.40))
            self.production_residual_spin.setValue(getattr(self.config, "production_max_residual_um", 0.30))
            self.production_deviation_spin.setValue(getattr(self.config, "production_max_radial_deviation_um", 0.60))
            self._updating_quality_controls = False
            self._refresh_quality_profile_hint()

            self.sigma_spin.setValue(self.params.gaussian_sigma_px)
            self.canny_low_spin.setValue(self.params.canny_low)
            self.canny_high_spin.setValue(self.params.canny_high)
            self.min_gradient_spin.setValue(self.params.min_gradient)
            self.profile_half_spin.setValue(self.params.profile_half_width_px)
            self.profile_step_spin.setValue(self.params.profile_step_px)
            self._set_combo_value(self.fit_mode_combo, self.params.fitting_mode)
            self._set_combo_value(self.upper_fit_mode_combo, getattr(self.params, "upper_fitting_mode", self.params.fitting_mode))
            self._set_combo_value(self.lower_fit_mode_combo, getattr(self.params, "lower_fitting_mode", self.params.fitting_mode))
            self.ransac_check.setChecked(self.params.use_ransac)
            self.residual_limit_spin.setValue(self.params.residual_limit_px)
            self.min_edge_points_spin.setValue(self.params.min_edge_points)
            self._set_combo_value(self.polarity_combo, self.params.polarity)
            self.measurement_timeout_spin.setValue(getattr(self.params, "measurement_timeout_s", 180))
            if hasattr(self, "roi_type_combo"):
                self._set_combo_value(
                    self.roi_type_combo,
                    getattr(self.params, "auto_refine_roi_type", "Caliper Circle"),
                )
            self._set_combo_value(self.auto_reference_combo, getattr(self.config, "auto_reference_label", ""))
            self._set_combo_value(self.auto_target_combo, getattr(self.config, "auto_target_label", ""))

        def _current_roi_entry(self):
            mark_id = self.mark_combo.currentText() or "Mark1"
            layer = self._current_layer()
            mark = self.marks.get(mark_id)
            if not mark:
                return None
            entries = mark.roi_entries(layer)
            if not entries:
                return None
            roi_id = self.roi_index_combo.currentData() if hasattr(self, "roi_index_combo") else ""
            return mark.roi_entry(layer, str(roi_id)) if roi_id else None

        def _current_roi(self):
            entry = self._current_roi_entry()
            return entry.roi if entry is not None else None

        def _refresh_roi_index_combo(self, preferred_id: str = ""):
            if not hasattr(self, "roi_index_combo"):
                return
            mark = self.marks.get(self._current_mark_id())
            entries = mark.roi_entries(self._current_layer()) if mark else []
            previous = preferred_id or str(self.roi_index_combo.currentData() or "")
            self.roi_index_combo.blockSignals(True)
            self.roi_index_combo.clear()
            for index, entry in enumerate(entries, start=1):
                self.roi_index_combo.addItem(f"ROI {index}", entry.roi_id)
            if entries and previous:
                target = self.roi_index_combo.findData(previous)
                self.roi_index_combo.setCurrentIndex(target)
            else:
                self.roi_index_combo.setCurrentIndex(-1)
            self.roi_index_combo.blockSignals(False)

        def on_roi_index_changed(self, *args):
            self._pending_new_roi = False
            self.on_active_roi_selection_changed()

        def select_roi_from_canvas(self, mark_id: str, layer: str, roi_id: str):
            self.mark_combo.setCurrentText(mark_id)
            self._set_combo_value(self.layer_combo, layer)
            self._refresh_roi_index_combo(roi_id)
            self.on_active_roi_selection_changed()

        def clear_roi_selection(self, mark_id: str = "", layer: str = ""):
            if mark_id and mark_id != self._current_mark_id():
                return
            if layer and layer != self._current_layer():
                return
            self._pending_new_roi = False
            self.roi_index_combo.blockSignals(True)
            self.roi_index_combo.setCurrentIndex(-1)
            self.roi_index_combo.blockSignals(False)
            self.on_active_roi_selection_changed()

        def show_canvas_interaction_message(self, message: str):
            if hasattr(self, "progress_stage_label"):
                self.progress_stage_label.setText(f"当前阶段：{message}")
            self._append_log(message)

        def handle_roi_context_action(self, mark_id: str, layer: str, roi_id: str, action: str):
            if self.operation_mode != "Engineering":
                self._append_log("生产模式不允许修改 ROI，请先进入工程模式。")
                return
            self.mark_combo.setCurrentText(mark_id)
            self._set_combo_value(self.layer_combo, layer)
            if roi_id:
                self._refresh_roi_index_combo(roi_id)
            if action == "copy":
                self.copy_current_roi()
            elif action == "delete":
                self.delete_current_roi()
            elif action == "clear_contours":
                for entry in self.marks[mark_id].roi_entries(layer):
                    self._invalidate_manual_roi(mark_id, entry.roi_id, remove_selection=False)
                self._refresh_all_widgets()
            elif action == "delete_layer":
                if QMessageBox.question(self, "删除当前层 ROI", "确定删除当前层全部 ROI 和识别结果吗？") != QMessageBox.Yes:
                    return
                self._push_roi_undo()
                for entry in list(self.marks[mark_id].roi_entries(layer)):
                    self._invalidate_manual_roi(mark_id, entry.roi_id, remove_selection=True)
                    self.marks[mark_id].remove_roi(layer, entry.roi_id)
                self.clear_roi_selection(mark_id, layer)
                self._refresh_all_widgets()

        def on_roi_parameter_edited(self, *args):
            if getattr(self, "_syncing_parameter_controls", False):
                return
            if self.operation_mode != "Engineering":
                self._refresh_all_widgets()
                return
            if self._current_roi_entry() is None:
                self._refresh_all_widgets()
                return
            if not self._roi_parameter_edit_active:
                self._push_roi_undo()
                self._roi_parameter_edit_active = True
            self.apply_roi_params_to_current(push_undo=False)
            self._roi_parameter_edit_timer.start()

        def _finish_roi_parameter_edit(self):
            self._roi_parameter_edit_active = False

        def on_algorithm_parameter_edited(self, *args):
            if getattr(self, "_syncing_parameter_controls", False):
                return
            self._pull_config_from_ui()
            self.invalidate_measurement_state("算法、标定或规格参数已修改", clear_selections=False)
            self._refresh_all_widgets()

        def begin_add_roi(self):
            if self.operation_mode != "Engineering":
                self._append_log("生产模式不允许修改 ROI，请先进入工程模式。")
                return
            self._pending_new_roi = True
            self.progress_stage_label.setText("当前阶段：请在图像上拖动创建新 ROI")
            canvas = self.upper_canvas if self._current_layer() == "upper" or self._current_mode() == "Single Image" else self.lower_canvas
            canvas.setCursor(Qt.CrossCursor)

        def copy_current_roi(self):
            if self.operation_mode != "Engineering":
                self._append_log("生产模式不允许修改 ROI，请先进入工程模式。")
                return
            entry = self._current_roi_entry()
            mark = self.marks.get(self._current_mark_id())
            if entry is None or mark is None:
                return
            self._push_roi_undo()
            copied = deepcopy(entry.roi)
            copied.x += 12.0
            copied.y += 12.0
            new_entry = mark.add_roi(self._current_layer(), copied, "manual")
            self._pending_new_roi = False
            self._refresh_roi_index_combo(new_entry.roi_id)
            self._invalidate_manual_roi(self._current_mark_id(), new_entry.roi_id)
            self._refresh_all_widgets()

        def delete_current_roi(self):
            if self.operation_mode != "Engineering":
                self._append_log("生产模式不允许修改 ROI，请先进入工程模式。")
                return
            mark = self.marks.get(self._current_mark_id())
            entry = self._current_roi_entry()
            if mark is None or entry is None:
                return
            answer = QMessageBox.question(self, "删除 ROI", "确定删除当前 ROI 及其识别结果吗？")
            if answer != QMessageBox.Yes:
                return
            self._push_roi_undo()
            removed_id = entry.roi_id
            mark.remove_roi(self._current_layer(), removed_id)
            self._invalidate_manual_roi(self._current_mark_id(), removed_id, remove_selection=True)
            self._refresh_roi_index_combo()
            self._refresh_all_widgets()

        def _invalidate_manual_roi(self, mark_id: str, roi_id: str, remove_selection: bool = False):
            removed_detection = self.roi_detections.setdefault(mark_id, {}).pop(roi_id, None)
            self._rebuild_legacy_manual_detections(mark_id)
            self.overlays.pop(mark_id, None)
            self.geometry_result = GeometryRunResult()
            selection = self.auto_selections.setdefault(mark_id, {"reference_label": "", "target_label": ""})
            selection_cleared = False
            for key in ("reference_label", "target_label"):
                if selection.get(key) == roi_id:
                    selection[key] = ""
                    selection_cleared = True
            mark = self.marks.get(mark_id)
            if mark is not None:
                if mark.reference_contour_id == roi_id:
                    mark.reference_contour_id = ""
                    selection_cleared = True
                if mark.target_contour_id == roi_id:
                    mark.target_contour_id = ""
                    selection_cleared = True
            if selection_cleared and (remove_selection or removed_detection is not None):
                self._manual_selection_requires_review.add(mark_id)
                self._append_log("所选轮廓已失效，请重新选择基准轮廓和待测轮廓。")

        def _rebuild_legacy_manual_detections(self, mark_id: str = ""):
            mark_ids = (mark_id,) if mark_id else tuple(self.roi_detections)
            for current_mark_id in mark_ids:
                projection = {}
                for detection in self.roi_detections.get(current_mark_id, {}).values():
                    projection.setdefault(detection.layer, detection)
                if projection:
                    self.detections[current_mark_id] = projection
                else:
                    self.detections.pop(current_mark_id, None)

        def _current_layer(self) -> str:
            return self.layer_combo.currentData() or "upper"

        def _current_mode(self) -> str:
            text = self.mode_combo.currentText()
            return "Dual Image" if text == "双图模式" else "Single Image"

        def _is_auto_workflow(self) -> bool:
            return self._combo_value(self.workflow_combo) == "Auto"

        def _set_mode_ui(self, mode: str):
            self.mode_combo.setCurrentText("双图模式" if mode == "Dual Image" else "单图模式")

        def _current_mark_id(self) -> str:
            return self.mark_combo.currentText() or "Mark1"

        def _ensure_mark_runtime(self, mark_id: str):
            self.mark_images.setdefault(mark_id, {"upper": None, "lower": None})
            self.mark_image_sources.setdefault(mark_id, {"upper": "none", "lower": "none"})
            self.auto_detections_by_mark.setdefault(mark_id, {})
            self.auto_candidates_by_mark.setdefault(mark_id, {})
            self.auto_selections.setdefault(mark_id, {"reference_label": "", "target_label": ""})

        def _set_image_for_layer(self, mark_id: str, layer: str, image: Optional[ImageData], source: str):
            self._ensure_mark_runtime(mark_id)
            self.mark_images[mark_id][layer] = image
            self.mark_image_sources[mark_id][layer] = source if image is not None else "none"

        def _image_source(self, mark_id: str, layer: str) -> str:
            self._ensure_mark_runtime(mark_id)
            return self.mark_image_sources.get(mark_id, {}).get(layer, "none")

        def _active_image_for_layer(self, mark_id: str, layer: str) -> Optional[ImageData]:
            self._ensure_mark_runtime(mark_id)
            if self._is_batch_mode():
                images = self.batch_images.get(mark_id, {}).get(layer, [])
                if images:
                    index = max(0, min(len(images) - 1, self._batch_detail_last_single_index - 1))
                    return resolve_image(images[index])
            image = self.mark_images[mark_id][layer]
            if image is None:
                return None
            if not self._is_batch_mode() and self._image_source(mark_id, layer) == "batch_preview":
                return None
            return image

        def _mark_images_snapshot_for_current_run(self) -> dict:
            snapshot = {}
            for mark_id in ("Mark1", "Mark2"):
                snapshot[mark_id] = {
                    layer: self._active_image_for_layer(mark_id, layer)
                    for layer in ("upper", "lower")
                }
            return snapshot

        def _switch_to_single_measurement_after_top_import(self):
            if hasattr(self, "measurement_run_mode_combo"):
                self._set_combo_value(self.measurement_run_mode_combo, "Single")
            self.batch_overlays = {"Mark1": [], "Mark2": []}
            self.batch_run_records = {"Mark1": [], "Mark2": []}
            self._batch_detail_run_index = 1
            self._batch_detail_last_single_index = 1
            self._refresh_batch_detail_selector()

        def _current_auto_detections(self):
            mark_id = self._current_mark_id()
            self._ensure_mark_runtime(mark_id)
            record = self._selected_batch_record(mark_id)
            if record and record.get("workflow") == "Auto":
                return record.get("detections", {})
            return self.auto_detections_by_mark[mark_id]

        def _selected_batch_record(self, mark_id: str, use_preview: bool = True) -> Optional[dict]:
            if not self._is_batch_mode():
                return None
            records = self.batch_run_records.get(mark_id, [])
            if not records:
                return None
            index = self._batch_detail_last_single_index if use_preview else self._batch_detail_run_index
            if isinstance(index, int) and 1 <= index <= len(records):
                return records[index - 1]
            return None

        def _current_manual_detection_map(self) -> dict:
            if not self._is_batch_mode() or not any(self.batch_run_records.values()):
                if any(self.roi_detections.values()):
                    return self.roi_detections
                return self.detections
            result = {}
            for mark_id in ("Mark1", "Mark2"):
                record = self._selected_batch_record(mark_id)
                if record and record.get("workflow") == "Manual" and record.get("detections"):
                    result[mark_id] = record["detections"]
            return result

        def _refresh_batch_detail_selector(self, default_first: bool = False):
            if not hasattr(self, "batch_detail_combo"):
                return
            maximum = max((len(records) for records in self.batch_run_records.values()), default=0)
            visible = self._is_batch_mode() and maximum > 0
            self.batch_detail_bar.setVisible(visible)
            if not visible:
                return
            previous = 1 if default_first else self.batch_detail_combo.currentData()
            self.batch_detail_combo.blockSignals(True)
            self.batch_detail_combo.clear()
            for index in range(1, maximum + 1):
                self.batch_detail_combo.addItem(f"第{index}次", index)
            self.batch_detail_combo.addItem("全部", "all")
            target_index = self.batch_detail_combo.findData(previous)
            self.batch_detail_combo.setCurrentIndex(target_index if target_index >= 0 else 0)
            self.batch_detail_combo.blockSignals(False)
            self._batch_detail_run_index = self.batch_detail_combo.currentData() or 1
            if isinstance(self._batch_detail_run_index, int):
                self._batch_detail_last_single_index = self._batch_detail_run_index
            self._update_batch_preview_label()

        def _update_batch_preview_label(self):
            if not hasattr(self, "batch_detail_preview_label"):
                return
            if self._batch_detail_run_index == "all":
                self.batch_detail_preview_label.setText(
                    f"表格：全部；图像预览：第{self._batch_detail_last_single_index}次"
                )
            else:
                self.batch_detail_preview_label.setText(f"图像预览：第{self._batch_detail_last_single_index}次")

        def _on_batch_detail_changed(self):
            if not hasattr(self, "batch_detail_combo"):
                return
            selected = self.batch_detail_combo.currentData()
            if selected is None:
                return
            self._batch_detail_run_index = selected
            if isinstance(selected, int):
                self._batch_detail_last_single_index = selected
            self._update_batch_preview_label()
            self._sync_current_mark_images()
            self._refresh_auto_selection_combos()
            self._refresh_all_widgets()

        def _sync_current_mark_images(self):
            mark_id = self._current_mark_id()
            self._ensure_mark_runtime(mark_id)
            self.upper_image = self._active_image_for_layer(mark_id, "upper")
            self.lower_image = self._active_image_for_layer(mark_id, "lower")
            self.upper_canvas.set_image(self.upper_image)
            self.lower_canvas.set_image(self.lower_image)

        def _invalidate_image_dependent_results(self, mark_id: str, layer: str):
            if self._current_mode() == "Single Image":
                self.roi_detections.pop(mark_id, None)
                self.detections.pop(mark_id, None)
            else:
                current = self.roi_detections.setdefault(mark_id, {})
                for roi_id in [key for key, detection in current.items() if detection.layer == layer]:
                    current.pop(roi_id, None)
                self._rebuild_legacy_manual_detections(mark_id)
            self.overlays.pop(mark_id, None)
            self.auto_detections_by_mark[mark_id] = {}
            self.auto_candidates_by_mark[mark_id] = {}
            self.auto_overlays.pop(mark_id, None)
            self.auto_selections[mark_id] = {"reference_label": "", "target_label": ""}

        def _combo_value(self, combo: QComboBox) -> str:
            return combo.currentData() or combo.currentText()

        def _set_combo_value(self, combo: QComboBox, value: str):
            for i in range(combo.count()):
                if combo.itemData(i) == value or combo.itemText(i) == value:
                    combo.setCurrentIndex(i)
                    return
            combo.setCurrentText(value)

        def _set_quality_threshold_controls(self):
            controls = (
                (self.conf_min_spin, self.config.confidence_min),
                (self.production_coverage_spin, self.config.production_min_coverage),
                (self.production_reject_spin, self.config.production_max_rejected_ratio),
                (self.production_residual_spin, self.config.production_max_residual_um),
                (self.production_deviation_spin, self.config.production_max_radial_deviation_um),
            )
            self._updating_quality_controls = True
            try:
                for control, value in controls:
                    control.blockSignals(True)
                    control.setValue(value)
                    control.blockSignals(False)
            finally:
                self._updating_quality_controls = False

        def _sync_quality_config_from_controls(self):
            self.config.quality_profile = self._combo_value(self.quality_profile_combo)
            self.config.confidence_min = self.conf_min_spin.value()
            self.config.production_min_coverage = self.production_coverage_spin.value()
            self.config.production_max_rejected_ratio = self.production_reject_spin.value()
            self.config.production_max_residual_um = self.production_residual_spin.value()
            self.config.production_max_radial_deviation_um = self.production_deviation_spin.value()

        def _refresh_quality_profile_hint(self):
            if not hasattr(self, "quality_profile_hint"):
                return
            profile = quality_profile_display(self.config)
            modified = quality_profile_is_modified(self.config)
            self.quality_profile_hint.setText(
                f"当前：{profile}。置信度≥{self.config.confidence_min:.2f}，"
                f"覆盖率≥{self.config.production_min_coverage:.0%}，"
                f"异常点≤{self.config.production_max_rejected_ratio:.0%}，"
                f"残差≤{self.config.production_max_residual_um:.3f} μm，"
                f"最大轮廓偏差≤{self.config.production_max_radial_deviation_um:.3f} μm。"
                + (" 详细参数已偏离该预设。" if modified else "")
            )
            self.quality_profile_hint.setToolTip(
                "宽容模式只放宽结果是否可用；识别明细仍会显示实际质量为优秀、合格、较差或无效。"
            )

        def _quality_settings_changed(self, message: str):
            for mark_map in self.roi_detections.values():
                for detection in mark_map.values():
                    annotate_detection_quality(detection, self.config)
            for detected_by_label in self.auto_detections_by_mark.values():
                for layer_map in detected_by_label.values():
                    for detection in layer_map.values():
                        annotate_detection_quality(detection, self.config)
            self.overlays = {}
            self.auto_overlays = {}
            self.batch_overlays = {"Mark1": [], "Mark2": []}
            self.batch_run_records = {"Mark1": [], "Mark2": []}
            self._refresh_quality_profile_hint()
            self._refresh_auto_selection_combos()
            self._refresh_all_widgets()
            self._append_log(message + "，请重新运行测量程序。")

        def on_quality_profile_changed(self):
            if self._updating_quality_controls:
                return
            profile = self._combo_value(self.quality_profile_combo)
            apply_quality_profile(self.config, profile)
            self._set_quality_threshold_controls()
            self._quality_settings_changed(f"识别质量门槛已切换为{QUALITY_PROFILE_LABELS[profile]}")

        def on_quality_threshold_edited(self):
            if self._updating_quality_controls:
                return
            self._sync_quality_config_from_controls()
            self._quality_settings_changed("识别质量详细参数已调整")

        def _fit_mode_for_layer(self, layer: str) -> str:
            if layer == "upper":
                mode = self._combo_value(self.upper_fit_mode_combo) if hasattr(self, "upper_fit_mode_combo") else getattr(self.params, "upper_fitting_mode", self.params.fitting_mode)
            else:
                mode = self._combo_value(self.lower_fit_mode_combo) if hasattr(self, "lower_fit_mode_combo") else getattr(self.params, "lower_fitting_mode", self.params.fitting_mode)
            if mode == "Auto":
                mode = self._combo_value(self.fit_mode_combo) if hasattr(self, "fit_mode_combo") else self.params.fitting_mode
            return mode

        def _auto_ring_roi_type(self, layer: str) -> str:
            if hasattr(self, "roi_type_combo"):
                return self._combo_value(self.roi_type_combo)
            return "Annulus"

        def _coerce_roi_to_auto_ring(self, roi: Roi, layer: str) -> Roi:
            """Return the ROI saved for this layer without overwriting its type.

            Earlier versions forced every saved ROI to the currently selected ROI Type in
            the side panel. That made mixed-shape measurements fail: for example, when
            the lower layer used 卡尺圆, the upper square ROI was also coerced to
            Caliper Circle during one-click calculation, so the square mark was measured
            as a circle.

            ROI type is now treated as a per-ROI/per-layer property. The side-panel
            settings only update the active ROI when the user draws/applies ROI
            parameters; calculation must respect each ROI's own stored type.
            """
            if roi is None:
                return roi
            # Keep the ROI's own type and geometry. Only normalize missing caliper
            # parameters for legacy recipes that do not contain these fields.
            roi_type = getattr(roi, "roi_type", "Rectangle")
            if roi_type == "Caliper Circle":
                return replace(
                    roi,
                    caliper_count=int(getattr(roi, "caliper_count", 64) or 64),
                    caliper_width_px=float(getattr(roi, "caliper_width_px", 8.0) or 8.0),
                    search_direction=getattr(roi, "search_direction", "Inner to Outer") or "Inner to Outer",
                )
            return roi

        def on_active_roi_selection_changed(self, *args):
            if hasattr(self, "three_point_circle_btn") and self.three_point_circle_btn.isChecked():
                self.three_point_circle_btn.blockSignals(True)
                self.three_point_circle_btn.setChecked(False)
                self.three_point_circle_btn.blockSignals(False)
                self.upper_canvas.set_circle_pick_mode(False)
                self.lower_canvas.set_circle_pick_mode(False)
            self._refresh_roi_index_combo()
            self._sync_current_mark_images()
            if hasattr(self, "auto_reference_combo"):
                self._push_current_auto_match_rules()
                self._refresh_auto_selection_combos()
            layer = self._current_layer()
            roi = self._current_roi()
            if roi is not None and hasattr(self, "roi_type_combo"):
                # Show current ROI parameters without triggering recursive updates.
                self.roi_type_combo.blockSignals(True)
                self.center_x_spin.blockSignals(True)
                self.center_y_spin.blockSignals(True)
                self.inner_radius_spin.blockSignals(True)
                self.outer_radius_spin.blockSignals(True)
                self.caliper_count_spin.blockSignals(True)
                self.caliper_width_spin.blockSignals(True)
                self.search_direction_combo.blockSignals(True)
                self.target_edge_combo.blockSignals(True)
                self.diameter_mode_combo.blockSignals(True)
                self.inner_ratio_spin.blockSignals(True)
                self.roi_angle_spin.blockSignals(True)
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
                self.roi_type_combo.blockSignals(False)
                self.center_x_spin.blockSignals(False)
                self.center_y_spin.blockSignals(False)
                self.inner_radius_spin.blockSignals(False)
                self.outer_radius_spin.blockSignals(False)
                self.caliper_count_spin.blockSignals(False)
                self.caliper_width_spin.blockSignals(False)
                self.search_direction_combo.blockSignals(False)
                self.target_edge_combo.blockSignals(False)
                self.diameter_mode_combo.blockSignals(False)
                self.inner_ratio_spin.blockSignals(False)
                self.roi_angle_spin.blockSignals(False)
                roi_type = getattr(roi, "roi_type", "")
                circular = roi_type in {"Circle", "Annulus", "Caliper Circle"}
                caliper = roi_type == "Caliper Circle"
                self.inner_radius_spin.setEnabled(circular)
                self.outer_radius_spin.setEnabled(circular)
                self.caliper_count_spin.setEnabled(caliper)
                self.caliper_width_spin.setEnabled(caliper)
                self.search_direction_combo.setEnabled(caliper)
                self.diameter_mode_combo.setEnabled(caliper)
                self.polarity_combo.setEnabled(roi_type in {"Caliper Circle", "Approximate Line"})
            self._refresh_all_widgets()

        def apply_roi_params_to_current(self, *, push_undo: bool = True):
            if self.operation_mode != "Engineering":
                self._append_log("生产模式不允许修改 ROI，请先进入工程模式。")
                return
            mark_id = self.mark_combo.currentText() or "Mark1"
            layer = self._current_layer()
            mark = self.marks.get(mark_id)
            if not mark:
                return
            entry = self._current_roi_entry()
            roi = entry.roi if entry is not None else None
            if roi is None:
                return
            if push_undo:
                self._push_roi_undo()
            cx = self.center_x_spin.value()
            cy = self.center_y_spin.value()
            inner = max(0.0, self.inner_radius_spin.value())
            outer = max(inner + 0.1, self.outer_radius_spin.value())
            roi.roi_type = self._auto_ring_roi_type(layer)
            if roi.roi_type in {"Circle", "Annulus", "Caliper Circle"}:
                roi.x = cx - outer
                roi.y = cy - outer
                roi.w = outer * 2.0
                roi.h = outer * 2.0
                roi.inner_ratio = float(np.clip(inner / max(outer, 1e-9), 0.0, 0.98))
            else:
                roi.x = cx - roi.w / 2.0
                roi.y = cy - roi.h / 2.0
            roi.target_edge = self._combo_value(self.target_edge_combo)
            roi.angle_deg = self.roi_angle_spin.value()
            roi.caliper_count = self.caliper_count_spin.value()
            roi.caliper_width_px = self.caliper_width_spin.value()
            roi.search_direction = self._combo_value(self.search_direction_combo)
            roi.diameter_mode = self._combo_value(self.diameter_mode_combo)
            entry.source = "manual"
            self.roi_sources.setdefault(mark_id, {})[layer] = "manual"
            self._recipe_roi_confirmation_signature = None
            self._invalidate_manual_roi(mark_id, entry.roi_id)
            # Reflect the just-drawn ROI parameters in the side panel.
            if mark_id == (self.mark_combo.currentText() or "Mark1") and layer == self._current_layer():
                self._syncing_parameter_controls = True
                try:
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
                finally:
                    self._syncing_parameter_controls = False
            self._refresh_all_widgets()

        def _refresh_all_widgets(self, *args):
            self._refresh_roi_index_combo()
            if hasattr(self, "batch_detail_bar"):
                self.batch_detail_bar.setVisible(
                    self._is_batch_mode() and any(self.batch_run_records.values())
                )
            current_mark = self.mark_combo.currentText() or "Mark1"
            current_layer = self._current_layer()
            is_dual = self._current_mode() == "Dual Image"
            self.import_upper_btn.setText("导入上层/单图")
            self.import_lower_btn.setText("导入下层图像")
            self.upper_canvas.title = "上层图像" if is_dual else "单图图像"
            self.lower_canvas.title = "下层图像"
            if hasattr(self, "upper_image_card"):
                self.upper_image_card.title_label.setText(self.upper_canvas.title)
            if hasattr(self, "lower_image_card"):
                self.lower_image_card.title_label.setText(self.lower_canvas.title)
            if self.upper_canvas.image is None:
                self.upper_canvas.setText("等待导入图像")
            if self.lower_canvas.image is None:
                self.lower_canvas.setText("等待导入图像")
            self.upper_canvas.fixed_layer = None if not is_dual else "upper"
            upper_layer = "upper" if is_dual else current_layer
            lower_layer = "lower"
            upper_roi_type = self._auto_ring_roi_type(upper_layer) if hasattr(self, "roi_type_combo") else "Annulus"
            lower_roi_type = self._auto_ring_roi_type(lower_layer) if hasattr(self, "roi_type_combo") else "Annulus"
            roi_inner_ratio = self.inner_ratio_spin.value() if hasattr(self, "inner_ratio_spin") else 0.60
            roi_target_edge = self._combo_value(self.target_edge_combo) if hasattr(self, "target_edge_combo") else "All Edges"
            roi_angle_deg = self.roi_angle_spin.value() if hasattr(self, "roi_angle_spin") else 0.0
            roi = self._current_roi()
            if roi is not None:
                roi_ring_half_width_px = max(1.0, 0.5 * (roi.outer_radius() - roi.inner_radius()))
            else:
                roi_ring_half_width_px = max(1.0, 0.5 * (self.outer_radius_spin.value() - self.inner_radius_spin.value()))
            roi_caliper_count = self.caliper_count_spin.value() if hasattr(self, "caliper_count_spin") else 64
            roi_caliper_width_px = self.caliper_width_spin.value() if hasattr(self, "caliper_width_spin") else 8.0
            roi_search_direction = self._combo_value(self.search_direction_combo) if hasattr(self, "search_direction_combo") else "Inner to Outer"
            roi_diameter_mode = self._combo_value(self.diameter_mode_combo) if hasattr(self, "diameter_mode_combo") else "Average"
            show_auto = self._is_auto_workflow()
            if hasattr(self, "workflow_explanation_label"):
                auto_roi_name = self.roi_type_combo.currentText() if hasattr(self, "roi_type_combo") else "卡尺圆"
                self.workflow_explanation_label.setText(
                    f"全图自动识别：候选精测按当前 ROI 类型“{auto_roi_name}”执行。"
                    if show_auto
                    else "手动 ROI 测量：使用各 Mark、各层当前显示的 ROI；配方 ROI 会在计算前确认。"
                )
            current_auto_detections = self._current_auto_detections()
            is_validated_recipe = self._combo_value(self.recipe_status_combo) == "Validated" if hasattr(self, "recipe_status_combo") else False
            if hasattr(self, "production_status_label"):
                self.production_status_label.setText(
                    "正式测量 / 已验证配方" if is_validated_recipe else "试测 / 未验证配方（不作正式判定）"
                )
            manual_labels = self._manual_detection_labels()
            reference_label = self.auto_reference_combo.currentData() or "" if hasattr(self, "auto_reference_combo") else ""
            target_label = self.auto_target_combo.currentData() or "" if hasattr(self, "auto_target_combo") else ""
            self.upper_canvas.set_context(
                current_mark,
                current_layer,
                self.marks,
                self.detections,
                upper_roi_type,
                roi_inner_ratio,
                roi_target_edge,
                roi_angle_deg,
                roi_ring_half_width_px,
                roi_caliper_count,
                roi_caliper_width_px,
                roi_search_direction,
                roi_diameter_mode,
                current_auto_detections,
                show_auto,
                manual_labels,
                reference_label,
                target_label,
                self.config.pixel_size_x_um,
                self.config.pixel_size_y_um,
                self.diagnostic_check.isChecked() if hasattr(self, "diagnostic_check") else False,
                self._current_manual_detection_map(),
                str(self.roi_index_combo.currentData() or ""),
            )
            self.lower_canvas.set_context(
                current_mark,
                current_layer,
                self.marks,
                self.detections,
                lower_roi_type,
                roi_inner_ratio,
                roi_target_edge,
                roi_angle_deg,
                roi_ring_half_width_px,
                roi_caliper_count,
                roi_caliper_width_px,
                roi_search_direction,
                roi_diameter_mode,
                current_auto_detections,
                show_auto,
                manual_labels,
                reference_label,
                target_label,
                self.config.pixel_size_x_um,
                self.config.pixel_size_y_um,
                self.diagnostic_check.isChecked() if hasattr(self, "diagnostic_check") else False,
                self._current_manual_detection_map(),
                str(self.roi_index_combo.currentData() or ""),
            )
            self.lower_canvas.setVisible(is_dual)
            if hasattr(self, "lower_image_card"):
                self.lower_image_card.setVisible(is_dual)
            self.import_lower_btn.setEnabled(is_dual)
            if hasattr(self, "import_upper_action"):
                self.import_upper_action.setEnabled(not self._calculation_running)
                self.import_lower_action.setEnabled(is_dual and not self._calculation_running)
            self.auto_detect_btn.setEnabled(show_auto)
            # 基准/待测轮廓选择在自动识别和手动 ROI 两种工作方式下都需要可用。
            self.auto_reference_combo.setEnabled(True)
            self.auto_target_combo.setEnabled(True)
            auto_reference = self.auto_reference_combo.currentData()
            auto_target = self.auto_target_combo.currentData()
            self.auto_calculate_btn.setEnabled(
                bool(auto_reference) and bool(auto_target) and auto_reference != auto_target
            )
            self.analyze_roi_btn.setEnabled(not show_auto and not self._calculation_running)
            self.analyze_current_btn.setEnabled(not show_auto)
            if self._is_batch_mode():
                run_errors = validate_batch_pairing(self.batch_images, is_dual)
                can_run = not run_errors
                run_reason = "；".join(run_errors) if run_errors else "批量图像配对完整，可以运行"
            else:
                upper_ready = self._active_image_for_layer(current_mark, "upper") is not None
                lower_ready = (not is_dual) or self._active_image_for_layer(current_mark, "lower") is not None
                can_run = upper_ready and lower_ready
                run_reason = "图像已就绪，可以运行" if can_run else "请先导入当前模式需要的图像"
            if self.operation_mode == "Production" and not self.loaded_recipe_path:
                can_run = False
                run_reason = "生产模式必须先加载已验证配方"
            self.analyze_all_btn.setEnabled(can_run and not self._calculation_running)
            self.analyze_all_btn.setToolTip(run_reason)
            self.three_point_circle_btn.setEnabled(not show_auto)
            self._refresh_mark_combo()
            if hasattr(self, "current_recipe_label"):
                recipe_display = self.loaded_recipe_display_name or self.recipe_name_edit.text().strip() or "未加载"
                self.current_recipe_label.setText(f"当前配方：{recipe_display}")
                self.current_recipe_label.setToolTip(f"当前配方：{recipe_display}")
            if hasattr(self, "load_recipe_btn"):
                recipe_display = self.loaded_recipe_display_name or self.recipe_name_edit.text().strip() or "未加载"
                full_text = f"当前配方：{recipe_display}  ▾"
                self.load_recipe_btn.setText(
                    self.load_recipe_btn.fontMetrics().elidedText(full_text, Qt.ElideMiddle, 290)
                )
                self.load_recipe_btn.setToolTip(
                    f"当前配方：{recipe_display}\n点击搜索、切换或从文件导入配方。"
                )
            if hasattr(self, "workflow_recipe_label"):
                recipe_display = self.loaded_recipe_display_name or self.recipe_name_edit.text().strip() or "未加载"
                self.workflow_recipe_label.setText(f"当前配方：{recipe_display}")
            if hasattr(self, "image_status_label"):
                if self.upper_image is None:
                    self.image_status_label.setText("等待导入图像")
                elif is_dual and self.lower_image is None:
                    self.image_status_label.setText("等待导入下层图像")
                elif (self.auto_overlays if show_auto else self.overlays):
                    self.image_status_label.setText("离线分析完成")
                else:
                    self.image_status_label.setText("图像已加载，可分析 ROI 或运行测量程序")
            if hasattr(self, "upper_file_label"):
                upper_img = self._image_for_layer("upper", current_mark)
                lower_img = self._image_for_layer("lower", current_mark)
                def source_name(image):
                    if image is None or not image.path:
                        return "未导入"
                    path = Path(image.path)
                    return f"{path.parent.name} / {path.name}" if path.parent.name else path.name
                self.upper_file_label.setText(source_name(upper_img))
                self.lower_file_label.setText(source_name(lower_img))
                if hasattr(self, "image_mode_tip_label"):
                    self.image_mode_tip_label.setText(self.mode_combo.currentText())
            if hasattr(self, "geometry_program"):
                self._refresh_geometry_results()
            self._refresh_tables()
            self._refresh_batch_image_table()
            self._refresh_repeatability_table()
            if hasattr(self, "_refresh_summary_cards"):
                self._refresh_summary_cards()
            if hasattr(self, "_refresh_step_status"):
                self._refresh_step_status(current_mark, show_auto)
            if hasattr(self, "_refresh_algorithm_path_panel"):
                self._refresh_algorithm_path_panel(current_mark, show_auto)
            self._update_toolbar_density()
            self._apply_operation_mode()

        def _refresh_mark_combo(self):
            self.marks = {
                mark_id: self.marks.get(mark_id, MarkRecipe(mark_id))
                for mark_id in ("Mark1", "Mark2")
            }
            current = self.mark_combo.currentText()
            self.mark_combo.blockSignals(True)
            self.mark_combo.clear()
            self.mark_combo.addItems(["Mark1", "Mark2"])
            if current in self.marks:
                self.mark_combo.setCurrentText(current)
            elif self.marks:
                self.mark_combo.setCurrentText(next(iter(self.marks)))
            self.mark_combo.blockSignals(False)

        def _mark_number(self, mark_id: str) -> str:
            digits = "".join(ch for ch in mark_id if ch.isdigit())
            return digits or mark_id

        @staticmethod
        def _alpha_label(index: int) -> str:
            label = ""
            value = index
            while True:
                value, remainder = divmod(value, 26)
                label = chr(ord("a") + remainder) + label
                if value == 0:
                    return label
                value -= 1

        def _manual_detection_labels(self):
            labels = {}
            for mark_id, roi_map in self._current_manual_detection_map().items():
                for roi_id, detection in roi_map.items():
                    labels[(mark_id, roi_id)] = str(int(detection.shape_params.get("roi_index", 1)))
            return labels

        def _display_detections(self):
            if not self._is_auto_workflow():
                return self._current_manual_detection_map()
            combined = {}
            source = self.auto_detections_by_mark
            if self._is_batch_mode() and any(self.batch_run_records.values()):
                source = {}
                for mark_id in ("Mark1", "Mark2"):
                    record = self._selected_batch_record(mark_id)
                    source[mark_id] = record.get("detections", {}) if record else {}
            for mark_id, detected in source.items():
                for label, layer_map in detected.items():
                    detection = next(iter(layer_map.values()), None)
                    combined[f"{mark_id}-{candidate_display_label(label, detection)}"] = layer_map
            return combined

        def _display_detection_entries(self) -> list[dict]:
            if self._is_batch_mode() and any(self.batch_run_records.values()):
                entries = []
                show_all = self._batch_detail_run_index == "all"
                for mark_id in ("Mark1", "Mark2"):
                    records = self.batch_run_records.get(mark_id, [])
                    selected_records = records if show_all else [
                        record for record in records
                        if record.get("run_index") == self._batch_detail_run_index
                    ]
                    for record in selected_records:
                        detections = record.get("detections", {})
                        if record.get("workflow") == "Auto":
                            for label, layer_map in detections.items():
                                detection = next(iter(layer_map.values()), None)
                                display_id = f"{mark_id}-{candidate_display_label(label, detection)}"
                                for layer, item in layer_map.items():
                                    entries.append({
                                        "run_index": record.get("run_index"), "mark_id": display_id,
                                        "source_mark_id": mark_id, "candidate_label": label,
                                        "layer": layer, "detection": item,
                                        "upper_file": record.get("upper_file", ""),
                                        "lower_file": record.get("lower_file", ""), "error": record.get("error", ""),
                                    })
                        else:
                            for roi_id, item in detections.items():
                                entries.append({
                                    "run_index": record.get("run_index"), "mark_id": mark_id,
                                    "source_mark_id": mark_id,
                                    "layer": item.layer, "roi_id": roi_id, "detection": item,
                                    "upper_file": record.get("upper_file", ""),
                                    "lower_file": record.get("lower_file", ""), "error": record.get("error", ""),
                                })
                        if not detections:
                            entries.append({
                                "run_index": record.get("run_index"), "mark_id": mark_id,
                                "source_mark_id": mark_id,
                                "layer": "", "detection": None,
                                "upper_file": record.get("upper_file", ""),
                                "lower_file": record.get("lower_file", ""),
                                "error": record.get("error", "未生成识别结果"),
                            })
                return entries
            entries = []
            if self._is_auto_workflow():
                for mark_id, detected in self.auto_detections_by_mark.items():
                    for label, layer_map in detected.items():
                        detection = next(iter(layer_map.values()), None)
                        display_id = f"{mark_id}-{candidate_display_label(label, detection)}"
                        for layer, item in layer_map.items():
                            entries.append({
                                "run_index": None, "mark_id": display_id, "source_mark_id": mark_id,
                                "candidate_label": label, "layer": layer, "detection": item,
                                "upper_file": "", "lower_file": "", "error": "",
                            })
            else:
                for mark_id, roi_map in self._current_manual_detection_map().items():
                    for roi_id, detection in roi_map.items():
                        entries.append({
                            "run_index": None, "mark_id": mark_id, "source_mark_id": mark_id,
                            "layer": detection.layer, "roi_id": roi_id,
                            "detection": detection, "upper_file": "", "lower_file": "", "error": "",
                        })
            return entries

        def _display_overlays(self):
            return self.auto_overlays if self._is_auto_workflow() else self.overlays

        def _find_auto_detection(self, mark_id: str, label: str) -> Optional[DetectionResult]:
            source = self._current_auto_detections() if mark_id == self._current_mark_id() else self.auto_detections_by_mark.get(mark_id, {})
            layer_map = source.get(label, {})
            return next(iter(layer_map.values()), None)

        def _build_summary_rows(self):
            overlay_map = self.auto_overlays if self._is_auto_workflow() else self.overlays
            is_trial = self._is_auto_workflow() and self.config.recipe_validation_status != "Validated"
            return build_summary_rows(overlay_map, self.config, is_trial=is_trial)

        def _set_result_card(self, value_label: QLabel, unit_label: QLabel, value: str, unit: str, color: str = "#1D1D1F"):
            value_label.setText(value)
            value_label.setProperty("resultColor", color)
            value_label.setStyleSheet(f"color: {color};")
            unit_label.setText(unit)
            unit_label.setToolTip(unit)

        def _refresh_summary_cards(self):
            rows = self._build_summary_rows()
            first = next((row for row in rows if row.get("项目") in {"Mark1", "Mark2"}), None)
            if first:
                dx_key = next((key for key in first if key.startswith("Dx")), "")
                dy_key = next((key for key in first if key.startswith("Dy")), "")
                dxy_key = next((key for key in first if key.startswith("Dxy")), "")
                dx = float(first.get(dx_key, 0.0))
                dy = float(first.get(dy_key, 0.0))
                dxy = float(first.get(dxy_key, 0.0))
                verdict = first.get("判定", "--")
                if verdict == "通过":
                    color = "#34C759"
                elif verdict in {"超限", "不通过", "异常"}:
                    color = "#FF3B30"
                else:
                    color = "#FFCC00"
                self._set_result_card(self.dx_value_label, self.dx_unit_label, f"{dx:+.3f}", "μm", "#007AFF")
                self._set_result_card(self.dy_value_label, self.dy_unit_label, f"{dy:+.3f}", "μm", "#FF9500")
                self._set_result_card(self.r_value_label, self.r_unit_label, f"{dxy:.3f}", "μm")
                result_note = {
                    "通过": "符合规格",
                    "超限": "对位结果超出规格",
                    "不通过": "对位结果超出规格",
                    "无效": "识别质量不满足要求",
                    "异常": "计算流程异常",
                    "试测": "未验证配方",
                }.get(verdict, "离线分析状态")
                self._set_result_card(self.result_value_label, self.result_unit_label, verdict, result_note, color)
                self.result_unit_label.setToolTip(first.get("提示", "") or result_note)
                self._update_summary_typography()
                return
            self._set_result_card(self.dx_value_label, self.dx_unit_label, "--", "μm", "#007AFF")
            self._set_result_card(self.dy_value_label, self.dy_unit_label, "--", "μm", "#FF9500")
            self._set_result_card(self.r_value_label, self.r_unit_label, "--", "μm")
            self._set_result_card(self.result_value_label, self.result_unit_label, "待分析", "离线分析状态", "#6E6E73")
            self._update_summary_typography()

        def _refresh_step_status(self, current_mark: str, show_auto: bool):
            imported = self._image_for_layer("upper", current_mark) is not None
            if self._current_mode() == "Dual Image":
                imported = imported and self._image_for_layer("lower", current_mark) is not None
            mark = self.marks.get(current_mark)
            roi_ready = bool(mark and (mark.upper_roi is not None or mark.lower_roi is not None))
            result_ready = current_mark in (self.auto_overlays if show_auto else self.overlays)
            geometry_ready = bool(self.geometry_result.features or self.geometry_result.measurements)
            states = [
                ("完成", "信息可编辑"),
                ("完成" if imported else "当前", "图像已加载" if imported else "等待导入图像"),
                ("完成" if roi_ready or show_auto or geometry_ready else ("当前" if imported else "待处理"),
                 "ROI/轮廓工具已设置" if roi_ready or geometry_ready else ("自动识别模式" if show_auto else "等待设置 ROI")),
                ("完成" if imported else "待处理", "参数已就绪" if imported else "导入图像后设置"),
                ("完成" if result_ready else "待处理", "可导出结果" if result_ready else "等待计算"),
            ]
            colors = {"待处理": "#A1A1A6", "当前": "#007AFF", "完成": "#34C759", "异常": "#FF3B30"}
            for idx, (row, dot, label, note, state_dot) in enumerate(getattr(self, "step_rows", [])):
                state, detail = states[idx]
                color = colors[state]
                dot.setStyleSheet(f"color: {color}; border: 1px solid {color}; border-radius: 13px; font-weight: 700; background: #FFFFFF;")
                label.setStyleSheet(f"font-weight: 600; color: {'#007AFF' if state == '当前' else '#1D1D1F'};")
                note.setText(detail)
                state_dot.setStyleSheet(f"color: {color};")
            if not imported:
                status_text, status_color = "等待导入图像", "#A1A1A6"
            elif result_ready:
                status_text, status_color = "离线分析完成", "#34C759"
            else:
                status_text, status_color = "图像已加载", "#34C759"
            if hasattr(self, "status_task_dot"):
                self.status_task_dot.setStyleSheet(f"color: {status_color};")
                self.status_task_label.setText(f"任务状态：{status_text}")
            if hasattr(self, "progress_stage_label") and not self._calculation_running:
                if result_ready:
                    self.progress_bar.setValue(100)
                    self.progress_stage_label.setText("当前阶段：离线分析完成")
                elif imported:
                    self.progress_bar.setValue(0)
                    self.progress_stage_label.setText("当前阶段：图像已加载，等待计算")
                else:
                    self.progress_bar.setValue(0)
                    self.progress_stage_label.setText("当前阶段：等待导入图像")
            if hasattr(self, "workflow_status_dot"):
                self.workflow_status_dot.setStyleSheet(f"color: {status_color};")
                self.workflow_status_label.setText(status_text)

        def _refresh_algorithm_path_panel(self, current_mark: str, show_auto: bool):
            if not hasattr(self, "algorithm_path_button"):
                return
            workflow = "Auto" if show_auto else "Manual"
            reference_label = self.auto_reference_combo.currentData() or "" if hasattr(self, "auto_reference_combo") else ""
            target_label = self.auto_target_combo.currentData() or "" if hasattr(self, "auto_target_combo") else ""
            if show_auto:
                reference = self._find_auto_detection(current_mark, reference_label) if reference_label else None
                target = self._find_auto_detection(current_mark, target_label) if target_label else None
            else:
                reference = self._find_manual_detection(current_mark, reference_label) if reference_label else None
                target = self._find_manual_detection(current_mark, target_label) if target_label else None

            if reference is None and target is None:
                self.algorithm_path_text = "暂无测量结果；分析 ROI 或自动识别后可查看实际算法路径。"
                self.algorithm_path_button.setToolTip(self.algorithm_path_text)
                if hasattr(self, "algorithm_path_summary_label"):
                    self.algorithm_path_summary_label.setText("算法路径：暂无测量结果")
                    self.algorithm_path_summary_label.setToolTip(self.algorithm_path_text)
                return
            parts = [f"当前 {current_mark}"]
            if reference is not None:
                reference_name = candidate_display_label(reference_label, reference) if show_auto else reference_label
                parts.append(f"基准({reference_name})：{describe_algorithm_path(reference, workflow)}")
            if target is not None:
                target_name = candidate_display_label(target_label, target) if show_auto else target_label
                parts.append(f"待测({target_name})：{describe_algorithm_path(target, workflow)}")
            self.algorithm_path_text = "；".join(parts)
            self.algorithm_path_button.setToolTip(self.algorithm_path_text)
            if hasattr(self, "algorithm_path_summary_label"):
                compact = self.algorithm_path_text.replace("当前 ", "").replace("：", " · ")
                if len(compact) > 38:
                    compact = compact[:37] + "…"
                self.algorithm_path_summary_label.setText(f"算法路径：{compact}")
                self.algorithm_path_summary_label.setToolTip(self.algorithm_path_text)

        def _refresh_tables(self):
            det_headers = [
                "次数", "输入文件", "标记", "层", "中心 X (μm)", "中心 Y (μm)",
                "尺寸/直径 (μm)", "椭圆圆度 (μm)", "参考残差 (μm)", "边缘点数", "置信度", "算法",
                "质量状态", "质量门槛", "实际质量", "质量详情", "覆盖率",
                "形状参数", "算法路径", "提示",
            ]
            det_headers.insert(4, "ROI编号")
            det_rows = []
            det_payloads = []
            manual_labels = self._manual_detection_labels()
            for entry in self._display_detection_entries():
                    mark_id = entry["mark_id"]
                    layer = entry["layer"]
                    roi_id = entry.get("roi_id", "")
                    d = entry["detection"]
                    run_text = f"第{entry['run_index']}次" if entry.get("run_index") else ""
                    upper_name = Path(entry.get("upper_file", "")).name
                    lower_name = Path(entry.get("lower_file", "")).name
                    file_text = " / ".join(value for value in (upper_name, lower_name) if value)
                    if d is None:
                        det_rows.append([
                            run_text, file_text, mark_id, "", "", "", "", "", "", "", "", "", "",
                            "异常", quality_profile_display(self.config), "未评估", "", "", "", "", entry.get("error", ""),
                        ])
                        det_payloads.append({
                            "kind": "batch_run", "run_index": entry.get("run_index"),
                        })
                        continue
                    if d.fitting_mode == "Rectangle":
                        width_um, height_um = rotated_rect_size_um(
                            d.shape_params.get("width_px", 0),
                            d.shape_params.get("height_px", 0),
                            d.shape_params.get("angle_deg", 0),
                            self.config,
                        )
                        shape_txt = (
                            f"宽={width_um:.3f}μm, "
                            f"高={height_um:.3f}μm, "
                            f"角度={d.shape_params.get('angle_deg', 0):.3f}°"
                        )
                    elif d.fitting_mode == "Ellipse":
                        metrics = ellipse_metrics_um(d.shape_params, self.config)
                        major_um = metrics.get("ellipse_major_um", 0.0)
                        minor_um = metrics.get("ellipse_minor_um", 0.0)
                        shape_txt = (
                            f"长轴={major_um:.3f}μm, "
                            f"短轴={minor_um:.3f}μm, "
                            f"角度={d.shape_params.get('angle_deg', 0):.3f}°"
                        )
                    elif d.fitting_mode == "Circle":
                        average = float(d.shape_params.get("average_diameter_um", d.diameter_um))
                        maximum = float(d.shape_params.get("maximum_diameter_um", average))
                        shape_txt = f"平均直径={average:.3f}μm, 最大直径={maximum:.3f}μm"
                    elif d.fitting_mode == "EdgeCenter":
                        shape_txt = (
                            f"边缘中心, 参考半径={d.diameter_um / 2.0:.3f}μm, "
                            f"宽={d.shape_params.get('width_px', 0) * self.config.pixel_size_x_um:.3f}μm, "
                            f"高={d.shape_params.get('height_px', 0) * self.config.pixel_size_y_um:.3f}μm"
                        )
                    elif d.fitting_mode == "CaliperCircle":
                        average = float(d.shape_params.get("average_diameter_um", d.diameter_um))
                        maximum = float(d.shape_params.get("maximum_diameter_um", average))
                        definition = "最大直径" if d.shape_params.get("diameter_mode") == "Maximum" else "平均直径"
                        shape_txt = (
                            f"卡尺圆, 平均直径={average:.3f}μm, 最大直径={maximum:.3f}μm, "
                            f"采用={definition}, "
                            f"内点={d.shape_params.get('inlier_count', 0)}, "
                            f"剔除={d.shape_params.get('rejected_count', 0)}, "
                            f"卡尺={d.shape_params.get('caliper_count', 0)}"
                        )
                    elif d.fitting_mode == "ProductionRectangle":
                        width_um, height_um = rotated_rect_size_um(
                            d.shape_params.get("width_px", 0),
                            d.shape_params.get("height_px", 0),
                            d.shape_params.get("angle_deg", 0),
                            self.config,
                        )
                        shape_txt = (
                            f"方形精测, 宽={width_um:.3f}μm, "
                            f"高={height_um:.3f}μm, "
                            f"角度={d.shape_params.get('angle_deg', 0):.3f}°"
                        )
                    elif d.fitting_mode in {"AutoCircle", "AutoRectangle", "ProductionCircle"}:
                        shape_type = "方形精测" if d.fitting_mode == "ProductionRectangle" else ("圆形精测" if d.fitting_mode == "ProductionCircle" else ("方形轮廓" if d.fitting_mode == "AutoRectangle" else "圆形轮廓"))
                        if d.fitting_mode == "ProductionCircle":
                            average = float(d.shape_params.get("average_diameter_um", d.diameter_um))
                            maximum = float(d.shape_params.get("maximum_diameter_um", average))
                            shape_txt = f"{shape_type}, 平均直径={average:.3f}μm, 最大直径={maximum:.3f}μm"
                        else:
                            shape_txt = (
                                f"{shape_type}, 参考半径={d.diameter_um / 2.0:.3f}μm, "
                                f"宽={d.shape_params.get('width_px', 0) * self.config.pixel_size_x_um:.3f}μm, "
                                f"高={d.shape_params.get('height_px', 0) * self.config.pixel_size_y_um:.3f}μm"
                            )
                    else:
                        shape_txt = ""
                    mode_txt = {
                        "Rectangle": "矩形",
                        "Ellipse": "椭圆",
                        "Circle": "圆",
                        "EdgeCenter": "边缘中心",
                        "RegionCenter": "区域中心",
                        "CaliperCircle": "卡尺找圆",
                        "AutoCircle": "自动圆轮廓",
                        "AutoRectangle": "自动方形轮廓",
                        "ProductionCircle": "正式卡尺圆拟合",
                        "ProductionRectangle": "正式四边卡尺拟合",
                    }.get(d.fitting_mode, d.fitting_mode)
                    roi_type_txt = {
                        "Annulus": "圆环",
                        "Caliper Circle": "卡尺圆",
                        "Rectangular Ring": "矩形环",
                        "Circle": "圆",
                        "Rectangle": "矩形",
                        "Auto Full Image": "全图自动识别",
                        "Auto Caliper Circle": "自动卡尺圆",
                        "Auto Four-Side Caliper": "自动四边卡尺",
                    }.get(d.shape_params.get("roi_type", ""), d.shape_params.get("roi_type", ""))
                    edge_txt = {
                        "All Edges": "全部边缘",
                        "Near Inner Boundary": "靠近内环",
                        "Near Outer Boundary": "靠近外环",
                        "Strongest Edge": "最强边缘",
                    }.get(d.shape_params.get("roi_target_edge", ""), d.shape_params.get("roi_target_edge", ""))
                    roi_txt = f"ROI={roi_type_txt}, 边缘={edge_txt}"
                    display_mark_id = mark_id
                    roi_text = f"ROI {int(d.shape_params.get('roi_index', 1))}" if roi_id else ""
                    ellipse_roundness = ""
                    if d.fitting_mode == "Ellipse":
                        ellipse_roundness = f"{ellipse_metrics_um(d.shape_params, self.config).get('ellipse_roundness_um', 0.0):.3f}"
                    det_rows.append([
                        run_text, file_text, display_mark_id, LAYER_LABELS.get(layer, layer), roi_text,
                        f"{d.center_x_um:.3f}", f"{d.center_y_um:.3f}",
                        f"{d.diameter_um:.3f}", ellipse_roundness, f"{d.residual_um:.3f}",
                        str(d.edge_point_count), f"{d.confidence:.3f}", mode_txt,
                        {"Valid": "有效", "Invalid": "无效"}.get(d.shape_params.get("quality_status", ""), ""),
                        d.shape_params.get("quality_profile_label", quality_profile_display(self.config)),
                        d.shape_params.get("quality_grade", "未评估"),
                        d.shape_params.get("quality_details", ""),
                        f"{d.shape_params.get('coverage', 0):.1%}" if "coverage" in d.shape_params else "",
                        shape_txt + "; " + roi_txt,
                        d.shape_params.get("algorithm_path", describe_algorithm_path(d, "Auto" if self._is_auto_workflow() else "Manual")),
                        d.warning or entry.get("error", ""),
                    ])
                    if entry.get("run_index"):
                        det_payloads.append({
                            "kind": "batch_run", "run_index": entry.get("run_index"),
                        })
                    elif self._is_auto_workflow():
                        det_payloads.append({
                            "kind": "auto_detection", "mark_id": entry.get("source_mark_id", mark_id),
                            "candidate_label": entry.get("candidate_label", ""), "layer": layer,
                        })
                    else:
                        det_payloads.append({
                            "kind": "manual_detection", "mark_id": entry.get("source_mark_id", mark_id),
                            "roi_id": roi_id, "layer": layer,
                        })
            self._fill_table(self.det_table, det_headers, det_rows, det_payloads)
            roundness_header = self.det_table.horizontalHeaderItem(8)
            if roundness_header is not None:
                roundness_header.setToolTip("椭圆圆度=(物理长轴-物理短轴)/2；非 ISO 最小区域圆度")

            ov_headers = ["项目", "Dx/Dy/Dxy/Rz", "数值", "判定", "质量门槛", "实际质量", "质量详情", "提示"]
            ov_rows = []
            ov_payloads = []
            display_overlays = self._display_overlays()
            for row in self._build_summary_rows():
                project = row.get("项目", "")
                verdict = row.get("判定", "")
                note = row.get("提示", "")
                overlay = display_overlays.get(project)
                profile = overlay.quality_profile if overlay else (quality_profile_display(self.config) if project != "Rz" else "—")
                grade = overlay.quality_grade if overlay else ("未评估" if project != "Rz" else "—")
                quality_summary = overlay.quality_summary if overlay else ""
                for key, value in row.items():
                    if key in {"项目", "判定", "质量门槛", "实际质量", "质量详情", "提示", "公式", "L(μm)"}:
                        continue
                    ov_rows.append([
                        project,
                        key,
                        f"{value:.3f}" if isinstance(value, float) else value,
                        verdict,
                        profile,
                        grade,
                        quality_summary,
                        note,
                    ])
                    ov_payloads.append({
                        "kind": "overlay", "mark_id": project,
                        "workflow": "Auto" if self._is_auto_workflow() else "Manual",
                        "deletable": project in {"Mark1", "Mark2"},
                    })
                if project == "Rz":
                    ov_rows[-1][7] = f"{row.get('公式', '')}；L={row.get('L(μm)', '')}；{note}".strip("；")
            self._fill_table(self.overlay_table, ov_headers, ov_rows, ov_payloads)

        def _delete_result_rows(self, payloads):
            if self.operation_mode != "Engineering":
                QMessageBox.information(self, "删除结果", "仅工程模式允许删除测量结果。")
                return
            payloads = [dict(item) for item in payloads if item and item.get("deletable", True)]
            if not payloads:
                return
            unique = []
            seen = set()
            for payload in payloads:
                identity = tuple(sorted((str(key), repr(value)) for key, value in payload.items()))
                if identity not in seen:
                    seen.add(identity)
                    unique.append(payload)
            answer = QMessageBox.question(
                self,
                "删除测量结果",
                f"确定删除选中的 {len(unique)} 项结果吗？\n"
                "相关对位、几何结果及批量统计会同步更新；原始图像和 ROI 不会删除。",
            )
            if answer != QMessageBox.Yes:
                return

            batch_indexes = {
                int(item["run_index"]) for item in unique
                if item.get("kind") == "batch_run" and item.get("run_index") not in (None, "")
            }
            if batch_indexes:
                self._delete_batch_result_runs(batch_indexes)

            geometry_payloads = [item for item in unique if item.get("kind") == "geometry"]
            if geometry_payloads:
                self.delete_geometry_results(geometry_payloads)

            for item in unique:
                kind = item.get("kind")
                if kind in {"batch_run", "geometry"}:
                    continue
                if kind == "manual_detection":
                    mark_id = str(item.get("mark_id", ""))
                    roi_id = str(item.get("roi_id", ""))
                    if mark_id and roi_id:
                        self._invalidate_manual_roi(mark_id, roi_id, remove_selection=True)
                elif kind == "auto_detection":
                    self._delete_auto_detection_result(item)
                elif kind == "overlay":
                    mark_id = str(item.get("mark_id", ""))
                    if mark_id in {"Mark1", "Mark2"}:
                        if item.get("workflow") == "Auto":
                            self.auto_overlays.pop(mark_id, None)
                        else:
                            self.overlays.pop(mark_id, None)

            self._refresh_batch_detail_selector()
            self._sync_current_mark_images()
            self._refresh_auto_selection_combos()
            self._refresh_all_widgets()
            self._append_log(f"已删除 {len(unique)} 项测量结果，并同步更新关联结果与导出数据。")

        def _delete_auto_detection_result(self, payload: dict):
            mark_id = str(payload.get("mark_id", ""))
            label = str(payload.get("candidate_label", ""))
            if not mark_id or not label:
                return
            self.auto_detections_by_mark.setdefault(mark_id, {}).pop(label, None)
            self.auto_candidates_by_mark.setdefault(mark_id, {}).pop(label, None)
            selection = self.auto_selections.setdefault(mark_id, {"reference_label": "", "target_label": ""})
            changed = False
            for key in ("reference_label", "target_label"):
                if selection.get(key) == label:
                    selection[key] = ""
                    changed = True
            mark = self.marks.get(mark_id)
            if mark is not None:
                if mark.reference_contour_id == label:
                    mark.reference_contour_id = ""
                    changed = True
                if mark.target_contour_id == label:
                    mark.target_contour_id = ""
                    changed = True
            self.auto_overlays.pop(mark_id, None)
            if changed:
                self._append_log("所选自动轮廓已删除，请重新选择基准轮廓和待测轮廓。")

        def _delete_batch_result_runs(self, run_indexes: set[int]):
            for mark_id in ("Mark1", "Mark2"):
                for layer in ("upper", "lower"):
                    images = self.batch_images.get(mark_id, {}).get(layer, [])
                    if images:
                        self.batch_images[mark_id][layer] = [
                            image for index, image in enumerate(images, start=1)
                            if index not in run_indexes
                        ]
            old_geometry = list(self.batch_geometry_results)
            self.batch_geometry_results = [
                result for index, result in enumerate(old_geometry, start=1)
                if index not in run_indexes
            ]
            for mark_id in ("Mark1", "Mark2"):
                records = list(self.batch_run_records.get(mark_id, []))
                if records:
                    records = [
                        record for record in records
                        if int(record.get("run_index", 0) or 0) not in run_indexes
                    ]
                    for new_index, record in enumerate(records, start=1):
                        record["run_index"] = new_index
                    self.batch_run_records[mark_id] = records
                    valid_overlays = [
                        record["overlay"] for record in records
                        if record.get("overlay") is not None and record["overlay"].result != "Invalid"
                    ]
                else:
                    valid_overlays = [
                        overlay for index, overlay in enumerate(self.batch_overlays.get(mark_id, []), start=1)
                        if index not in run_indexes
                    ]
                self.batch_overlays[mark_id] = valid_overlays
                self.overlays.pop(mark_id, None)
                self.auto_overlays.pop(mark_id, None)
                if valid_overlays:
                    target = self.auto_overlays if self._is_auto_workflow() else self.overlays
                    target[mark_id] = self._mean_overlay(mark_id, valid_overlays)

            maximum = max((len(records) for records in self.batch_run_records.values()), default=0)
            if maximum:
                self._batch_detail_last_single_index = min(self._batch_detail_last_single_index, maximum)
                if isinstance(self._batch_detail_run_index, int):
                    self._batch_detail_run_index = min(self._batch_detail_run_index, maximum)
            else:
                self._batch_detail_run_index = 1
                self._batch_detail_last_single_index = 1

        def _fill_table(self, table: QTableWidget, headers, rows, row_payloads=None):
            row_payloads = list(row_payloads or ({} for _ in rows))
            signature = (
                tuple(headers),
                tuple(tuple(str(value) for value in row) for row in rows),
                tuple(repr(payload) for payload in row_payloads),
            )
            if getattr(table, "_content_signature", None) == signature:
                return
            table._content_signature = signature
            sorting = table.isSortingEnabled()
            table.setSortingEnabled(False)
            table.setUpdatesEnabled(False)
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c, val in enumerate(row):
                    item = QTableWidgetItem(str(val))
                    row_text = " ".join(str(x) for x in row)
                    row_values = {str(value).strip() for value in row}
                    hard_failure = bool(
                        row_values.intersection({"失败", "超限", "无效", "异常", "Fail", "Invalid", "Error"})
                    ) or any(
                        token in row_text
                        for token in ("识别失败", "计算异常", "低于宽容门槛", "超出配方范围")
                    )
                    if hard_failure:
                        item.setBackground(QColor(255, 199, 206))
                        item.setForeground(QColor(156, 0, 6))
                    elif "较差（仅宽容可用）" in row_text:
                        item.setBackground(QColor(255, 243, 205))
                        item.setForeground(QColor(133, 77, 14))
                    elif "优秀（满足超精确）" in row_text:
                        item.setBackground(QColor(226, 244, 232))
                        item.setForeground(QColor(25, 107, 58))
                    table.setItem(r, c, item)
            if hasattr(table, "fit_result_columns"):
                table.fit_result_columns()
            else:
                table.resizeColumnsToContents()
            table.horizontalHeader().setStretchLastSection(False)
            table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
            if hasattr(table, "_sync_external_scroll_range"):
                table._sync_external_scroll_range(
                    table.horizontalScrollBar().minimum(), table.horizontalScrollBar().maximum()
                )
            if hasattr(table, "set_row_payloads"):
                table.set_row_payloads(row_payloads)
            table.setUpdatesEnabled(True)
            table.setSortingEnabled(sorting)

        def zoom_canvases(self, factor: float):
            self.upper_canvas.zoom_by(factor)
            if self.lower_canvas.isVisible():
                self.lower_canvas.zoom_by(factor)
            self._sync_zoom_level_display()

        def set_canvas_zoom_percent(self, text: str):
            if not text or not text.endswith("%"):
                return
            try:
                zoom = float(text[:-1]) / 100.0
            except ValueError:
                return
            for canvas in (self.upper_canvas, self.lower_canvas):
                canvas.user_zoom = float(np.clip(zoom, 0.05, 80.0))
                canvas.pan_x = 0.0
                canvas.pan_y = 0.0
                canvas.update()

        def reset_canvas_views(self):
            self.upper_canvas.reset_view(update=True)
            self.lower_canvas.reset_view(update=True)
            self._sync_zoom_level_display()

        def set_canvas_actual_size(self):
            """Show imported pixels at 1:1 in every visible image view."""
            canvases = [self.upper_canvas]
            if self.lower_canvas.isVisible():
                canvases.append(self.lower_canvas)
            for canvas in canvases:
                if canvas.pixmap_cache is None:
                    continue
                canvas._update_transform()
                canvas.user_zoom = float(np.clip(1.0 / max(canvas.fit_scale, 1e-9), 0.05, 80.0))
                canvas.pan_x = 0.0
                canvas.pan_y = 0.0
                canvas.update()
            self._sync_zoom_level_display()

        def _sync_zoom_level_display(self):
            if not hasattr(self, "zoom_level_combo"):
                return
            zoom_text = f"{int(round(self.upper_canvas.user_zoom * 100.0))}%"
            self.zoom_level_combo.blockSignals(True)
            if self.zoom_level_combo.findText(zoom_text) < 0:
                self.zoom_level_combo.addItem(zoom_text)
            self.zoom_level_combo.setCurrentText(zoom_text)
            self.zoom_level_combo.blockSignals(False)
