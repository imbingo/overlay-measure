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
from .export_naming import build_export_filename
from .image_loader import SUPPORTED_EXTENSIONS, display_to_uint8, load_image
from .measurement_engine import run_measurement_job
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
from .geometry_models import GeometryRunResult
from .recipe_manager import load_recipe, load_recipe_with_geometry, save_recipe
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


class MainWindowRecipeMixin:
        def _recipe_state_will_be_replaced(self) -> bool:
            has_roi = any(
                mark and (mark.upper_roi is not None or mark.lower_roi is not None)
                for mark in self.marks.values()
            )
            return bool(
                self.loaded_recipe_path
                or has_roi
                or self.detections
                or self.overlays
                or self.auto_overlays
                or any(self.batch_overlays.values())
                or any(self.batch_run_records.values())
            )

        def _confirm_recipe_switch(self, path: str) -> bool:
            if not self._recipe_state_will_be_replaced():
                return True
            if self.loaded_recipe_path and Path(self.loaded_recipe_path).resolve() == Path(path).resolve():
                return True
            answer = QMessageBox.question(
                self,
                "切换配方",
                "切换后将清除当前 ROI 和测量结果，并载入新配方中的 ROI。\n"
                "已导入的单次图像和批量图像会保留。\n\n确定继续吗？",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            return answer == QMessageBox.Yes

        def _load_recipe_from_path(
            self,
            path: str,
            *,
            confirm_switch: bool = True,
            show_message: bool = True,
        ) -> bool:
            try:
                integrity_status, recipe_hash = verify_recipe(path)
            except Exception as exc:
                QMessageBox.critical(self, "配方校验失败", str(exc))
                return False
            if integrity_status == "Mismatch":
                self.runtime_logger.error("Recipe integrity mismatch: %s", path)
                QMessageBox.critical(
                    self,
                    "配方完整性异常",
                    "配方内容与已保存的哈希不一致，已阻止加载。\n"
                    "请由工程人员核对配方文件或重新发布配方。",
                )
                return False
            if confirm_switch and not self._confirm_recipe_switch(path):
                return False
            try:
                config, params, marks, geometry_program = load_recipe_with_geometry(path)
                self.config = config
                if not getattr(self.config, "recipe_name", "").strip():
                    self.config.recipe_name = Path(path).stem
                self.params = params
                self.geometry_program = geometry_program
                self.geometry_result = GeometryRunResult()
                loaded_marks = {mark.mark_id: mark for mark in marks if mark.mark_id in {"Mark1", "Mark2"}}
                self.marks = {
                    mark_id: loaded_marks.get(mark_id, MarkRecipe(mark_id))
                    for mark_id in ("Mark1", "Mark2")
                }
                self.roi_sources = self._empty_roi_sources()
                for mark_id, mark in self.marks.items():
                    for entry in mark.upper_rois:
                        entry.source = "recipe"
                    for entry in mark.lower_rois:
                        entry.source = "recipe"
                    if mark.upper_rois:
                        self.roi_sources[mark_id]["upper"] = "recipe"
                    if mark.lower_rois:
                        self.roi_sources[mark_id]["lower"] = "recipe"
                self.loaded_recipe_path = str(Path(path).resolve())
                self.loaded_recipe_display_name = self.config.recipe_name.strip() or Path(path).stem
                self.loaded_recipe_hash = recipe_hash
                self.recipe_integrity_status = integrity_status
                self._recipe_roi_confirmation_signature = None
                # Recipe changes preserve imported images but invalidate all prior measurements.
                for runtime_mark in ("Mark1", "Mark2"):
                    self._ensure_mark_runtime(runtime_mark)
                self.detections.clear()
                self.roi_detections = {"Mark1": {}, "Mark2": {}}
                self.overlays.clear()
                self.auto_detections_by_mark = {"Mark1": {}, "Mark2": {}}
                self.auto_candidates_by_mark = {"Mark1": {}, "Mark2": {}}
                self.auto_selections = {
                    "Mark1": {
                        "reference_label": self.marks["Mark1"].reference_contour_id or getattr(self.config, "auto_reference_label", ""),
                        "target_label": self.marks["Mark1"].target_contour_id or getattr(self.config, "auto_target_label", ""),
                    },
                    "Mark2": {
                        "reference_label": self.marks["Mark2"].reference_contour_id,
                        "target_label": self.marks["Mark2"].target_contour_id,
                    },
                }
                self._manual_selection_requires_review.clear()
                self.auto_overlays.clear()
                self.batch_overlays = {"Mark1": [], "Mark2": []}
                self.batch_run_records = {"Mark1": [], "Mark2": []}
                self.recipe_library.mark_used(path)
                self._sync_current_mark_images()
                self._push_config_to_ui()
                self._push_current_auto_match_rules()
                self._refresh_auto_selection_combos()
                self._refresh_all_widgets()
                integrity_text = "哈希已验证" if integrity_status == "Verified" else "未签章"
                self._append_log(f"已加载配方：{self.loaded_recipe_display_name}（{integrity_text}）")
                if show_message:
                    QMessageBox.information(
                        self,
                        "加载完成",
                        f"配方已加载：\n{path}\n\n完整性：{integrity_text}\nSHA256：{recipe_hash}",
                    )
                return True
            except Exception as exc:
                QMessageBox.critical(self, "加载失败", str(exc))
                return False

        def show_recipe_quick_menu(self):
            if self.recipe_quick_menu is None:
                self.recipe_quick_menu = RecipeQuickMenu(self)
                self.recipe_quick_menu.recipeSelected.connect(self._load_recipe_from_path)
                self.recipe_quick_menu.importRequested.connect(self.import_recipe_file)
                self.recipe_quick_menu.managerRequested.connect(self.show_recipe_manager)
                self.recipe_quick_menu.openLibraryRequested.connect(self.open_recipe_library)
                self.recipe_quick_menu.saveRequested.connect(self.save_recipe_file)
            self.recipe_quick_menu.set_entries(self.recipe_library.scan())
            self.recipe_quick_menu.set_engineering_access(
                self.operation_mode == "Engineering" and not self._calculation_running
            )
            position = self.load_recipe_btn.mapToGlobal(QPoint(0, self.load_recipe_btn.height() + 4))
            self.recipe_quick_menu.popup(position)
            self.recipe_quick_menu.search_edit.setFocus()

        def open_recipe_library(self):
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.recipe_library.root)))

        def show_recipe_manager(self):
            dialog = RecipeLibraryDialog(
                self.recipe_library,
                self,
                engineering=self.operation_mode == "Engineering",
            )

            def import_from_manager():
                dialog.reject()
                self.import_recipe_file()

            dialog.import_btn.clicked.connect(import_from_manager)
            if dialog.exec() == QDialog.Accepted and dialog.selected_recipe_path:
                self._load_recipe_from_path(dialog.selected_recipe_path)

        def import_recipe_file(self):
            if self.operation_mode != "Engineering":
                QMessageBox.warning(self, "权限不足", "生产模式不能导入或发布配方，请先切换到工程模式。")
                return
            path, _ = QFileDialog.getOpenFileName(self, "从文件导入配方", "", "JSON (*.json)")
            if not path:
                return
            try:
                load_recipe(path)
            except Exception as exc:
                QMessageBox.critical(self, "配方无效", str(exc))
                return

            choice = QMessageBox(self)
            choice.setWindowTitle("导入配方")
            choice.setText("请选择这个配方的使用方式：")
            choice.setInformativeText(
                "加入配方库后可从顶部快捷列表中搜索和切换；仅本次加载不会复制原文件。"
            )
            add_button = choice.addButton("导入并加入配方库", QMessageBox.AcceptRole)
            once_button = choice.addButton("仅本次加载", QMessageBox.ActionRole)
            choice.addButton(QMessageBox.Cancel)
            choice.exec()
            clicked = choice.clickedButton()
            if clicked == add_button:
                try:
                    managed_path = self.recipe_library.import_recipe(path)
                except Exception as exc:
                    QMessageBox.critical(self, "导入失败", str(exc))
                    return
                self._load_recipe_from_path(str(managed_path))
            elif clicked == once_button:
                self._load_recipe_from_path(path)

        def load_recipe_file(self):
            """Backward-compatible entry point for existing integrations and tests."""
            self.import_recipe_file()

        def save_recipe_file(self):
            if self.operation_mode != "Engineering":
                QMessageBox.warning(self, "权限不足", "生产模式不能保存或修改配方，请先验证密码并切换到工程模式。")
                return
            self._pull_config_from_ui()
            name = self.config.recipe_name.strip() or "overlay_recipe"
            version = str(getattr(self.config, "recipe_version", "")).strip()
            default_name = RecipeLibrary._safe_stem(f"{name}_{version}" if version else name) + ".json"
            status_directory = RecipeLibrary._status_directory(
                str(getattr(self.config, "recipe_validation_status", ""))
            )
            default_path = self.recipe_library.root / status_directory / default_name
            path, _ = QFileDialog.getSaveFileName(self, "保存配方", str(default_path), "JSON (*.json)")
            if not path:
                return
            try:
                for mark_id, mark in self.marks.items():
                    selection = self.auto_selections.get(mark_id, {})
                    mark.reference_contour_id = str(selection.get("reference_label", ""))
                    mark.target_contour_id = str(selection.get("target_label", ""))
                save_recipe(
                    path,
                    self.config,
                    self.params,
                    list(self.marks.values()),
                    self.geometry_program,
                )
                saved_hash = seal_recipe(path)
                saved_path = Path(path).resolve()
                try:
                    saved_path.relative_to(self.recipe_library.root)
                    managed_path = saved_path
                except ValueError:
                    managed_path = self.recipe_library.import_recipe(saved_path)
                self.loaded_recipe_path = str(managed_path)
                self.loaded_recipe_display_name = self.config.recipe_name.strip() or managed_path.stem
                self.loaded_recipe_hash = saved_hash if managed_path == saved_path else seal_recipe(managed_path)
                self.recipe_integrity_status = "Verified"
                self.recipe_library.mark_used(managed_path)
                self._refresh_all_widgets()
                extra = "" if managed_path == saved_path else f"\n快捷配方库副本：\n{managed_path}"
                QMessageBox.information(self, "保存完成", f"配方已保存：\n{saved_path}{extra}")
            except Exception as exc:
                QMessageBox.critical(self, "保存失败", str(exc))
