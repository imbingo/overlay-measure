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
from .candidate_ordering import candidate_display_label
from .geometry_models import GeometryProgram, GeometryRunResult
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
from .recipe_manager import load_recipe, save_recipe
from .recipe_library import RecipeLibrary, RecipeLibraryEntry
from .recipe_integrity import seal_recipe, verify_recipe
from .result_exporter import build_detection_rows, export_results
from .rz_calculator import build_summary_rows
from .runtime_support import RecoveryStore, build_runtime_logger

from .ui_constants import LAYER_LABELS, RESULT_LABELS


class SidebarComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class SidebarDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class SidebarSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class FramelessTitleBar(QFrame):
    """Custom title bar that keeps the frameless window movable and maximizable."""

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self.window = window
        self.drag_offset: Optional[QPoint] = None
        self.setObjectName("titleBar")
        self.setFixedHeight(46)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            handle = self.window.windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
            self.drag_offset = event.globalPosition().toPoint() - self.window.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_offset is not None and event.buttons() & Qt.LeftButton and not self.window.isMaximized():
            self.window.move(event.globalPosition().toPoint() - self.drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.drag_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.window.toggle_maximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class CollapsibleSection(QWidget):
    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self.toggle_btn = QToolButton()
        self.toggle_btn.setText(title)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(expanded)
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_btn.setObjectName("sectionToggle")
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 8, 0, 0)
        self.body_layout.setSpacing(10)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toggle_btn)
        layout.addWidget(self.body)
        self.toggle_btn.toggled.connect(self._apply_state)
        self._apply_state(expanded)

    def _apply_state(self, expanded: bool):
        self.body.setVisible(expanded)
        self.toggle_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def add_widget(self, widget: QWidget):
        self.body_layout.addWidget(widget)


class GrayProfilePlot(QWidget):
    """Small dependency-free line-profile chart for engineering diagnostics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.distances = np.array([], dtype=float)
        self.values = np.array([], dtype=float)
        self.setMinimumHeight(235)

    def set_profile(self, distances, values):
        self.distances = np.asarray(distances, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))
        rect = self.rect().adjusted(50, 18, -18, -36)
        painter.setPen(QPen(QColor("#D7DCE3"), 1))
        painter.drawRect(rect)
        if self.values.size < 2:
            painter.setPen(QColor("#6B7280"))
            painter.drawText(rect, Qt.AlignCenter, "在图像上拖动一条线以查看灰度剖面")
            return
        lo, hi = float(np.min(self.values)), float(np.max(self.values))
        if hi - lo < 1e-12:
            lo -= 0.5
            hi += 0.5
        x_max = max(float(self.distances[-1]), 1e-9)
        path = QPainterPath()
        for index, (distance, value) in enumerate(zip(self.distances, self.values)):
            x = rect.left() + rect.width() * float(distance) / x_max
            y = rect.bottom() - rect.height() * (float(value) - lo) / (hi - lo)
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        pen = QPen(QColor("#007AFF"), 1.8)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawPath(path)
        painter.setPen(QColor("#4B5563"))
        painter.setFont(QFont("Microsoft YaHei UI", 8))
        painter.drawText(4, int(rect.top() + 6), f"{hi:.3f}")
        painter.drawText(4, int(rect.bottom()), f"{lo:.3f}")
        painter.drawText(int(rect.left()), self.height() - 10, "0")
        painter.drawText(int(rect.right() - 54), self.height() - 10, f"{x_max:.2f} μm")


class ImageDiagnosticsDialog(QDialog):
    """Non-destructive grayscale line inspection using the raw imported image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像诊断与边缘分析")
        self.setMinimumSize(620, 380)
        layout = QVBoxLayout(self)
        self.instruction_label = QLabel("在图像上拖动一条线；使用原始灰度，不受显示增强影响。")
        self.instruction_label.setObjectName("statusCaption")
        self.stats_label = QLabel("尚未选择剖面线")
        self.stats_label.setWordWrap(True)
        self.plot = GrayProfilePlot(self)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.plot)
        layout.addWidget(self.stats_label)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

    def set_profile(self, payload: dict):
        values = np.asarray(payload.get("values", []), dtype=float)
        distances = np.asarray(payload.get("distances_um", []), dtype=float)
        self.plot.set_profile(distances, values)
        if values.size:
            self.stats_label.setText(
                "长度={length_um:.3f} μm；灰度 最小={minimum:.3f}，最大={maximum:.3f}，"
                "均值={mean:.3f}，PV={pv:.3f}".format(
                    length_um=float(distances[-1]) if distances.size else 0.0,
                    minimum=float(np.min(values)),
                    maximum=float(np.max(values)),
                    mean=float(np.mean(values)),
                    pv=float(np.ptp(values)),
                )
            )


class ImageCanvas(QLabel):
    imageDropped = Signal(str)
    roiChanged = Signal(str, str, object)  # mark_id, layer, Roi
    roiEditCommitted = Signal(str, str, str, object)  # mark_id, layer, stable roi_id, Roi
    roiSelected = Signal(str, str, str)  # mark_id, layer, stable roi_id
    geometryClicked = Signal(str, object)  # layer, click payload
    geometryCommand = Signal(str)  # cancel / undo
    roiSelectionCleared = Signal(str, str)  # mark_id, layer
    roiContextAction = Signal(str, str, str, str)  # mark_id, layer, roi_id, action
    interactionMessage = Signal(str)
    profileMeasured = Signal(object)

    def __init__(self, title: str, fixed_layer: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.title = title
        self.setAcceptDrops(True)
        self.image_drop_enabled = True
        self.fixed_layer = fixed_layer
        self.setMinimumSize(360, 250)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.image: Optional[ImageData] = None
        self.pixmap_cache: Optional[QPixmap] = None
        self.scale = 1.0
        self.fit_scale = 1.0
        self.user_zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.active_mark_id = "Mark1"
        self.active_layer = fixed_layer or "upper"
        self.active_roi_id = ""
        self.active_roi_type = "Annulus"
        self.active_roi_inner_ratio = 0.60
        self.active_roi_target_edge = "All Edges"
        self.active_roi_angle_deg = 0.0
        self.active_ring_half_width_px = 10.0
        self.active_caliper_count = 64
        self.active_caliper_width_px = 8.0
        self.active_search_direction = "Inner to Outer"
        self.active_diameter_mode = "Average"
        self.circle_pick_mode = False
        self.circle_pick_points = []
        self.circle_preview_point = None
        self.marks: Dict[str, MarkRecipe] = {}
        self.detections: Dict[str, Dict[str, DetectionResult]] = {}
        self.roi_detections: Dict[str, Dict[str, DetectionResult]] = {}
        self.auto_detections: Dict[str, Dict[str, DetectionResult]] = {}
        self.show_auto_detections = False
        self.manual_labels = {}
        self.auto_reference_label = ""
        self.auto_target_label = ""
        self.show_diagnostics = False
        self.geometry_program = GeometryProgram()
        self.geometry_result = GeometryRunResult()
        self.geometry_interaction_active = False
        self.geometry_interaction = None
        self.geometry_hover_point = None
        self.hovered_geometry_feature_id = ""
        self.selected_geometry_feature_id = ""
        self.hovered_detection_key = ""
        self.selected_detection_key = ""
        self.highlighted_coordinate_id = ""
        self.selected_caliper_feature = None
        self.selected_caliper_detection_id = None
        self.caliper_selection_context = None
        self.display_enhancement = False
        self.roi_editing_enabled = True
        self.pixel_size_x_um = 0.1
        self.pixel_size_y_um = 0.1
        self.drag_start_img: Optional[QPoint] = None
        self.drag_current_img: Optional[QPoint] = None
        self.is_dragging = False
        self.is_adjusting_roi = False
        self.is_moving_roi = False
        self.roi_edit_preview: Optional[Roi] = None
        self.roi_edit_start: Optional[Roi] = None
        self.roi_edit_id = ""
        self.roi_edit_press_widget: Optional[QPointF] = None
        self.roi_edit_moved = False
        self.pending_caliper_hit = None
        self.adjust_roi_part = ""
        self.adjust_mark_id = ""
        self.adjust_layer = ""
        self.move_start_img = None
        self.move_start_roi = None
        self.is_panning = False
        self.space_pan_held = False
        self.profile_capture_enabled = False
        self.profile_start_img = None
        self.profile_current_img = None
        self.profile_dragging = False
        self.pan_start_pos: Optional[QPoint] = None
        self.pan_start_x = 0.0
        self.pan_start_y = 0.0
        self.setText("等待导入图像")
        self.setStyleSheet("QLabel { background: #252930; color: #F5F6F8; border: 1px solid #363C45; border-radius: 6px; }")

    def _drop_path(self, event):
        urls = event.mimeData().urls()
        if not self.image_drop_enabled or len(urls) != 1 or not urls[0].isLocalFile():
            return None
        path = Path(urls[0].toLocalFile())
        return str(path) if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS else None

    def dragEnterEvent(self, event):
        if self._drop_path(event):
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()
            self.interactionMessage.emit("请拖入一张支持的本地图像；计算期间不可导入。")

    def dragMoveEvent(self, event):
        if self._drop_path(event):
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = self._drop_path(event)
        if path:
            event.setDropAction(Qt.CopyAction)
            event.accept()
            self.imageDropped.emit(path)
        else:
            event.ignore()

    def set_image(self, image: Optional[ImageData]):
        # UI refreshes frequently when ROI selection changes. Rebinding the exact
        # same image must not discard the operator's zoom and pan position.
        if image is self.image:
            return
        self.clear_caliper_selection(update=False)
        self.image = image
        self.pixmap_cache = None
        self.reset_view(update=False)
        if image is not None:
            self.pixmap_cache = self._make_pixmap(image)
            self.setText("")
        else:
            self.setText("等待导入图像")
        self.update()

    def set_display_enhancement(self, enabled: bool):
        enabled = bool(enabled)
        if self.display_enhancement == enabled:
            return
        self.display_enhancement = enabled
        if self.image is not None:
            self.pixmap_cache = self._make_pixmap(self.image)
        self.update()

    def set_profile_capture_enabled(self, enabled: bool):
        self.profile_capture_enabled = bool(enabled)
        if not enabled:
            self.profile_dragging = False
            self.profile_start_img = None
            self.profile_current_img = None
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.update()

    def _emit_profile(self):
        if self.image is None or self.profile_start_img is None or self.profile_current_img is None:
            return
        start = np.asarray(self.profile_start_img, dtype=float)
        end = np.asarray(self.profile_current_img, dtype=float)
        delta = end - start
        length_px = float(np.linalg.norm(delta))
        if length_px < 1.0:
            self.interactionMessage.emit("剖面线过短，请拖动更长的线段。")
            return
        samples = max(2, int(np.ceil(length_px)) + 1)
        xs = np.linspace(start[0], end[0], samples)
        ys = np.linspace(start[1], end[1], samples)
        gray = np.asarray(self.image.gray, dtype=np.float64)
        x0 = np.clip(np.floor(xs).astype(int), 0, gray.shape[1] - 1)
        y0 = np.clip(np.floor(ys).astype(int), 0, gray.shape[0] - 1)
        x1 = np.clip(x0 + 1, 0, gray.shape[1] - 1)
        y1 = np.clip(y0 + 1, 0, gray.shape[0] - 1)
        fx = xs - x0
        fy = ys - y0
        values = (
            (1 - fx) * (1 - fy) * gray[y0, x0]
            + fx * (1 - fy) * gray[y0, x1]
            + (1 - fx) * fy * gray[y1, x0]
            + fx * fy * gray[y1, x1]
        )
        length_um = float(np.hypot(delta[0] * self.pixel_size_x_um, delta[1] * self.pixel_size_y_um))
        self.profileMeasured.emit({
            "start_px": tuple(start),
            "end_px": tuple(end),
            "values": values.tolist(),
            "distances_um": np.linspace(0.0, length_um, samples).tolist(),
        })

    def set_context(
        self,
        active_mark_id: str,
        active_layer: str,
        marks: Dict[str, MarkRecipe],
        detections,
        roi_type: str = "Annulus",
        roi_inner_ratio: float = 0.60,
        roi_target_edge: str = "All Edges",
        roi_angle_deg: float = 0.0,
        roi_ring_half_width_px: float = 10.0,
        roi_caliper_count: int = 64,
        roi_caliper_width_px: float = 8.0,
        roi_search_direction: str = "Inner to Outer",
        roi_diameter_mode: str = "Average",
        auto_detections=None,
        show_auto_detections: bool = False,
        manual_labels=None,
        auto_reference_label: str = "",
        auto_target_label: str = "",
        pixel_size_x_um: float = 0.1,
        pixel_size_y_um: float = 0.1,
        show_diagnostics: bool = False,
        roi_detections=None,
        active_roi_id: Optional[str] = None,
    ):
        next_layer = self.fixed_layer or active_layer
        next_context = ("auto" if show_auto_detections else "manual", active_mark_id, next_layer)
        next_roi_id = str(active_roi_id or "")
        if (
            active_mark_id != self.active_mark_id
            or next_layer != self.active_layer
            or next_roi_id != self.active_roi_id
            or bool(show_auto_detections) != self.show_auto_detections
        ):
            self._cancel_roi_edit()
            self.is_dragging = False
            self.drag_start_img = None
            self.drag_current_img = None
        if self.caliper_selection_context != next_context:
            self.clear_caliper_selection(update=False)
        self.caliper_selection_context = next_context
        self.active_mark_id = active_mark_id
        self.active_layer = next_layer
        self.active_roi_id = next_roi_id
        if active_roi_id is None:
            mark = marks.get(active_mark_id)
            entries = mark.roi_entries(next_layer) if mark is not None else []
            self.active_roi_id = entries[0].roi_id if entries else ""
        self.active_roi_type = roi_type
        self.active_roi_inner_ratio = float(roi_inner_ratio)
        self.active_roi_target_edge = roi_target_edge
        self.active_roi_angle_deg = float(roi_angle_deg)
        self.active_ring_half_width_px = float(max(1.0, roi_ring_half_width_px))
        self.active_caliper_count = int(roi_caliper_count)
        self.active_caliper_width_px = float(roi_caliper_width_px)
        self.active_search_direction = roi_search_direction
        self.active_diameter_mode = roi_diameter_mode
        self.marks = marks
        self.detections = detections
        if roi_detections is None:
            migrated = {}
            for mark_id, layer_map in (detections or {}).items():
                mark = marks.get(mark_id)
                for layer, detection in layer_map.items():
                    entries = mark.roi_entries(layer) if mark is not None else []
                    roi_id = entries[0].roi_id if entries else layer
                    migrated.setdefault(mark_id, {})[roi_id] = detection
            self.roi_detections = migrated
        else:
            self.roi_detections = roi_detections
        self.auto_detections = auto_detections or {}
        self.show_auto_detections = bool(show_auto_detections)
        self.manual_labels = manual_labels or {}
        self.auto_reference_label = auto_reference_label
        self.auto_target_label = auto_target_label
        self.pixel_size_x_um = float(pixel_size_x_um)
        self.pixel_size_y_um = float(pixel_size_y_um)
        self.show_diagnostics = bool(show_diagnostics)
        self._drop_stale_caliper_selection()
        self.update()

    def set_geometry_context(
        self,
        program: Optional[GeometryProgram],
        result: Optional[GeometryRunResult],
        interaction_active: bool = False,
        interaction=None,
    ):
        self.geometry_program = program or GeometryProgram()
        self.geometry_result = result or GeometryRunResult()
        self.geometry_interaction_active = bool(interaction_active)
        self.geometry_interaction = interaction
        if not interaction_active:
            self.geometry_hover_point = None
            self.hovered_geometry_feature_id = ""
        self.update()

    def set_geometry_interaction_active(self, active: bool):
        self.geometry_interaction_active = bool(active)
        if not active:
            self.geometry_hover_point = None
            self.hovered_geometry_feature_id = ""
            self.hovered_detection_key = ""
            self.selected_geometry_feature_id = ""
            self.selected_detection_key = ""
        self.setCursor(Qt.CrossCursor if active else Qt.ArrowCursor)
        self.update()

    def set_coordinate_highlight(self, coordinate_id: str):
        self.highlighted_coordinate_id = str(coordinate_id or "")
        self.update()

    def _geometry_hit(self, pos, tolerance_px: float = 10.0) -> str:
        best_id = ""
        best_distance = float(tolerance_px)
        for feature_id, feature in self.geometry_result.features.items():
            if feature.status != "Valid" or feature.layer != self.active_layer:
                continue
            if feature.center_px is not None:
                wx, wy = self.image_to_widget(*feature.center_px)
                distance = float(np.hypot(wx - pos.x(), wy - pos.y()))
                if feature.radius_px is not None:
                    radial = abs(distance - feature.radius_px * self.scale)
                    distance = min(distance, radial)
                if distance < best_distance:
                    best_id, best_distance = feature_id, distance
            if len(feature.points_px) >= 2 and feature.feature_type == "line":
                a = np.asarray(self.image_to_widget(*feature.points_px[0]), dtype=float)
                b = np.asarray(self.image_to_widget(*feature.points_px[1]), dtype=float)
                p = np.asarray([pos.x(), pos.y()], dtype=float)
                segment = b - a
                denom = float(np.dot(segment, segment))
                if denom > 1e-12:
                    t = float(np.clip(np.dot(p - a, segment) / denom, 0.0, 1.0))
                    distance = float(np.linalg.norm(p - (a + t * segment)))
                    if distance < best_distance:
                        best_id, best_distance = feature_id, distance
        return best_id

    def _measurement_hit(self, pos, tolerance_px: float = 10.0) -> str:
        best_id = ""
        best_distance = float(tolerance_px)
        point = np.asarray([pos.x(), pos.y()], dtype=float)
        for measurement_id, measurement in self.geometry_result.measurements.items():
            if measurement.status != "Valid" or measurement.layer != self.active_layer:
                continue
            refs = [self.geometry_result.features.get(item) for item in measurement.reference_ids]
            refs = [item for item in refs if item is not None and item.center_px is not None]
            if len(refs) < 2:
                continue
            first = np.asarray(self.image_to_widget(*refs[0].center_px), dtype=float)
            second = np.asarray(self.image_to_widget(*refs[1].center_px), dtype=float)
            segment = second - first
            denom = float(np.dot(segment, segment))
            if denom <= 1e-12:
                continue
            t = float(np.clip(np.dot(point - first, segment) / denom, 0.0, 1.0))
            distance = float(np.linalg.norm(point - (first + t * segment)))
            if distance < best_distance:
                best_id, best_distance = measurement_id, distance
        return best_id

    def _detection_key_hit(self, pos, tolerance_px: float = 10.0) -> str:
        best_key = ""
        best_distance = float(tolerance_px)
        if not self.show_auto_detections:
            for roi_id, detection in self.roi_detections.get(self.active_mark_id, {}).items():
                if detection.layer != self.active_layer:
                    continue
                distance = self._detection_hit_distance(detection, pos)
                if distance < best_distance:
                    best_key = f"{self.active_mark_id}/{roi_id}:{detection.layer}"
                    best_distance = distance
            return best_key
        maps = self.auto_detections
        for identity, layer_map in maps.items():
            for layer, detection in layer_map.items():
                if layer != self.active_layer:
                    continue
                distance = self._detection_hit_distance(detection, pos)
                if distance < best_distance:
                    prefix = f"{self.active_mark_id}/" if self.show_auto_detections else ""
                    best_key, best_distance = f"{prefix}{identity}:{layer}", distance
        return best_key

    def _detection_for_key(self, key: str) -> Optional[DetectionResult]:
        if not key:
            return None
        identity, separator, layer = key.rpartition(":")
        if not separator or layer != self.active_layer:
            return None
        prefix = f"{self.active_mark_id}/"
        local_id = identity[len(prefix):] if identity.startswith(prefix) else identity
        if self.show_auto_detections:
            return self.auto_detections.get(local_id, {}).get(layer)
        return self.roi_detections.get(self.active_mark_id, {}).get(local_id)

    @staticmethod
    def _detection_feature_type(detection: Optional[DetectionResult]) -> str:
        if detection is None:
            return ""
        mode = str(detection.fitting_mode or "")
        if mode == "Line":
            return "line"
        if mode in {"Rectangle", "ProductionRectangle"}:
            return "rectangle"
        if mode == "Ellipse":
            return "ellipse"
        if mode in {"RegionCenter", "EdgeCenter"}:
            return "region"
        return "circle"

    def _geometry_pick_payload(self, pos) -> Optional[dict]:
        point = self.widget_to_image_float(pos)
        if point is None:
            return None
        interaction = self.geometry_interaction or {}
        action = str(interaction.get("action", ""))
        pick_index = len(interaction.get("clicks", []))
        manual_geometry = action in {"feature:point", "feature:line", "feature:circle"}
        label_placement = action == "coordinate_label" and pick_index >= 1
        allow_snap = not manual_geometry and not label_placement
        feature_id = self._geometry_hit(pos, 14.0) if allow_snap else ""
        detection_key = self._detection_key_hit(pos, 14.0) if allow_snap else ""
        feature = self.geometry_result.features.get(feature_id) if feature_id else None
        detection = self._detection_for_key(detection_key)
        snapped = False
        feature_type = ""
        if feature is not None:
            feature_type = str(feature.feature_type or "")
            if feature.center_px is not None and feature_type != "line":
                point = feature.center_px
                snapped = True
        elif detection is not None:
            feature_type = self._detection_feature_type(detection)
            if feature_type != "line":
                point = (float(detection.center_x_px), float(detection.center_y_px))
                snapped = True
        return {
            "point_px": (float(point[0]), float(point[1])),
            "feature_id": feature_id,
            "detection_key": detection_key,
            "feature_type": feature_type,
            "snapped_to_center": snapped,
        }

    def clear_caliper_selection(self, update: bool = True):
        self.selected_caliper_feature = None
        self.selected_caliper_detection_id = None
        if update:
            self.update()

    @staticmethod
    def _detection_has_calipers(detection: Optional[DetectionResult]) -> bool:
        if detection is None:
            return False
        return bool(detection.shape_params.get("caliper_windows")) or detection.fitting_mode == "CaliperCircle"

    def _selected_detection(self) -> Optional[DetectionResult]:
        if not self.selected_caliper_feature:
            return None
        mode, identity, layer = self.selected_caliper_feature
        if mode == "auto":
            return self.auto_detections.get(identity, {}).get(layer)
        return self.roi_detections.get(self.active_mark_id, {}).get(identity)

    def _drop_stale_caliper_selection(self):
        if not self.selected_caliper_feature:
            return
        detection = self._selected_detection()
        if detection is None or id(detection) != self.selected_caliper_detection_id:
            self.clear_caliper_selection(update=False)

    def _select_caliper_detection(self, mode: str, identity: str, layer: str, detection: DetectionResult):
        self.selected_caliper_feature = (mode, identity, layer)
        self.selected_caliper_detection_id = id(detection)
        self.update()

    def _manual_caliper_selected(self, mark_id: str, roi_id: str, layer=None, detection: Optional[DetectionResult] = None) -> bool:
        if detection is None and isinstance(layer, DetectionResult):
            detection = layer
            layer = roi_id
            roi_id = self.active_roi_id or mark_id
        return (
            detection is not None
            and self.selected_caliper_feature in {
                ("manual", roi_id, layer),
                ("manual", mark_id, layer),
            }
            and self.selected_caliper_detection_id == id(detection)
        )

    def _auto_caliper_selected(self, label: str, layer: str, detection: Optional[DetectionResult]) -> bool:
        return (
            detection is not None
            and self.selected_caliper_feature == ("auto", label, layer)
            and self.selected_caliper_detection_id == id(detection)
        )

    def _manual_roi_visible(
        self,
        mark_id: str,
        roi_id: str,
        layer=None,
        roi: Optional[Roi] = None,
        detection: Optional[DetectionResult] = None,
    ) -> bool:
        if isinstance(layer, Roi):
            legacy_roi = layer
            legacy_detection = roi if isinstance(roi, DetectionResult) else None
            layer = roi_id
            roi_id = self.active_roi_id or mark_id
            roi = legacy_roi
            detection = legacy_detection
        if roi is None:
            return False
        if getattr(roi, "roi_type", "") != "Caliper Circle" or detection is None:
            return True
        return self._manual_caliper_selected(mark_id, roi_id, layer, detection)

    def _has_caliper_result_for_hint(self) -> bool:
        if self.show_auto_detections:
            return any(
                self._detection_has_calipers(detection)
                for layer_map in self.auto_detections.values()
                for detection in layer_map.values()
            )
        detection = self.roi_detections.get(self.active_mark_id, {}).get(self.active_roi_id)
        roi = self._active_roi()
        return (
            roi is not None
            and getattr(roi, "roi_type", "") == "Caliper Circle"
            and detection is not None
        )

    def _mean_pixel_size_um(self) -> float:
        return 0.5 * (self.pixel_size_x_um + self.pixel_size_y_um)

    def _contour_label_anchor(self, detection: DetectionResult, direction_index: Optional[int] = None):
        points = detection.shape_params.get("contour_points", detection.edge_points)
        if points:
            array = np.asarray(points, dtype=float)
            if direction_index is None:
                index = int(np.argmax(array[:, 0] - 0.75 * array[:, 1]))
            else:
                angle = -np.pi / 4.0 + direction_index * 2.3999632297
                direction = np.asarray([np.cos(angle), np.sin(angle)])
                offsets = array - np.asarray([detection.center_x_px, detection.center_y_px])
                lengths = np.maximum(np.linalg.norm(offsets, axis=1, keepdims=True), 1e-9)
                index = int(np.argmax((offsets / lengths) @ direction))
            return self.image_to_widget(float(array[index, 0]), float(array[index, 1]))
        radius = float(detection.shape_params.get("radius_px", detection.diameter_px / 2.0))
        return self.image_to_widget(
            detection.center_x_px + radius * 0.70,
            detection.center_y_px - radius * 0.70,
        )

    @staticmethod
    def _point_segment_distance(px: float, py: float, start: QPointF, end: QPointF) -> float:
        ax, ay = float(start.x()), float(start.y())
        bx, by = float(end.x()), float(end.y())
        dx, dy = bx - ax, by - ay
        denominator = dx * dx + dy * dy
        if denominator <= 1e-12:
            return float(np.hypot(px - ax, py - ay))
        t = float(np.clip(((px - ax) * dx + (py - ay) * dy) / denominator, 0.0, 1.0))
        return float(np.hypot(px - (ax + t * dx), py - (ay + t * dy)))

    def _polyline_hit_distance(self, pos, points, closed: bool = True) -> float:
        if not points or len(points) < 2:
            return float("inf")
        widget_points = [
            QPointF(*self.image_to_widget(float(point[0]), float(point[1])))
            for point in points
        ]
        pairs = list(zip(widget_points, widget_points[1:]))
        if closed and len(widget_points) > 2:
            pairs.append((widget_points[-1], widget_points[0]))
        return min(
            self._point_segment_distance(float(pos.x()), float(pos.y()), start, end)
            for start, end in pairs
        )

    def _detection_hit_distance(self, detection: DetectionResult, pos) -> float:
        """Return distance in widget pixels so hit tolerance is zoom independent."""
        cx, cy = self.image_to_widget(detection.center_x_px, detection.center_y_px)
        center_distance = float(np.hypot(float(pos.x()) - cx, float(pos.y()) - cy))
        if center_distance <= 12.0:
            return center_distance

        if detection.fitting_mode in {"Circle", "EdgeCenter", "CaliperCircle", "ProductionCircle"}:
            radius = float(detection.shape_params.get("radius_px", detection.diameter_px / 2.0))
            radial_distance = float(np.hypot(float(pos.x()) - cx, float(pos.y()) - cy))
            return abs(radial_distance - radius * self.scale)

        if detection.fitting_mode in {"Rectangle", "ProductionRectangle"}:
            width = float(detection.shape_params.get("width_px", detection.diameter_px))
            height = float(detection.shape_params.get("height_px", detection.diameter_px))
            angle = float(detection.shape_params.get("angle_deg", 0.0))
            points = self._rotated_rect_points_widget(
                detection.center_x_px,
                detection.center_y_px,
                width,
                height,
                angle,
            )
            pairs = list(zip(points, points[1:] + points[:1]))
            return min(
                self._point_segment_distance(float(pos.x()), float(pos.y()), start, end)
                for start, end in pairs
            )

        if detection.fitting_mode == "Line":
            start = detection.shape_params.get("line_start_px")
            end = detection.shape_params.get("line_end_px")
            if start is not None and end is not None:
                return self._point_segment_distance(
                    float(pos.x()),
                    float(pos.y()),
                    QPointF(*self.image_to_widget(float(start[0]), float(start[1]))),
                    QPointF(*self.image_to_widget(float(end[0]), float(end[1]))),
                )

        contour = detection.shape_params.get(
            "candidate_contour_points",
            detection.shape_params.get("contour_points", detection.edge_points),
        )
        return self._polyline_hit_distance(pos, contour)

    def _manual_caliper_hit(self, pos, tolerance_px: float = 9.0):
        mark_id = self.active_mark_id
        layer = self.active_layer
        mark = self.marks.get(mark_id)
        if mark is None:
            return None
        best = None
        best_distance = float(tolerance_px)
        for entry in mark.roi_entries(layer):
            detection = self.roi_detections.get(mark_id, {}).get(entry.roi_id)
            if (
                getattr(entry.roi, "roi_type", "") != "Caliper Circle"
                or not self._detection_has_calipers(detection)
            ):
                continue
            distance = self._detection_hit_distance(detection, pos)
            if distance <= best_distance:
                best = (entry.roi_id, layer, detection)
                best_distance = distance
        return best

    def _manual_roi_hit(self, pos, tolerance_px: float = 9.0) -> str:
        mark = self.marks.get(self.active_mark_id)
        point = self.widget_to_image_float(pos)
        if mark is None or point is None:
            return ""
        entries = mark.roi_entries(self.active_layer)
        active = mark.roi_entry(self.active_layer, self.active_roi_id) if self.active_roi_id else None
        ordered = ([active] if active is not None else []) + [
            entry for entry in reversed(entries) if active is None or entry.roi_id != active.roi_id
        ]

        # Preserve the current selection when ROIs overlap; otherwise use the
        # topmost painted ROI (the last entry in the layer list).
        for entry in ordered:
            detection = self.roi_detections.get(self.active_mark_id, {}).get(entry.roi_id)
            if detection is not None:
                distance = self._detection_hit_distance(detection, pos)
                if distance <= tolerance_px:
                    return entry.roi_id
        sample = np.array([[point[0], point[1]]], dtype=np.float64)
        for entry in ordered:
            roi = entry.roi.normalized()
            outer = roi
            if roi.roi_type in {"Annulus", "Caliper Circle"}:
                outer = replace(roi, roi_type="Circle", inner_ratio=0.0)
            elif roi.roi_type == "Rectangular Ring":
                outer = replace(roi, roi_type="Rectangle", inner_ratio=0.0)
            if bool(outer.contains_points(sample)[0]):
                return entry.roi_id
        return ""

    def _nearest_auto_caliper_hit(self, pos, tolerance_px: float = 9.0):
        best = None
        best_distance = tolerance_px
        for label, layer_map in self.auto_detections.items():
            for layer, detection in layer_map.items():
                if self.fixed_layer and layer != self.fixed_layer:
                    continue
                if not self._detection_has_calipers(detection):
                    continue
                distance = self._detection_hit_distance(detection, pos)
                if distance <= best_distance:
                    best = (label, layer, detection)
                    best_distance = distance
        return best

    def set_circle_pick_mode(self, enabled: bool):
        self.circle_pick_mode = enabled
        self.circle_pick_points = []
        self.circle_preview_point = None
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.update()

    def _make_pixmap(self, image: ImageData) -> QPixmap:
        u8 = display_to_uint8(image, self.display_enhancement)
        h, w = u8.shape
        qimg = QImage(u8.data, w, h, w, QImage.Format_Grayscale8).copy()
        return QPixmap.fromImage(qimg)

    def _base_offset_for_scale(self, scale: float):
        if self.pixmap_cache is None:
            return 0.0, 0.0
        img_w = self.pixmap_cache.width()
        img_h = self.pixmap_cache.height()
        return (self.width() - img_w * scale) / 2.0, (self.height() - img_h * scale) / 2.0

    def _update_transform(self):
        if self.pixmap_cache is None:
            return
        img_w = self.pixmap_cache.width()
        img_h = self.pixmap_cache.height()
        if img_w <= 0 or img_h <= 0:
            return
        sx = self.width() / img_w
        sy = self.height() / img_h
        self.fit_scale = min(sx, sy)
        self.scale = self.fit_scale * self.user_zoom
        base_x, base_y = self._base_offset_for_scale(self.scale)
        self.offset_x = base_x + self.pan_x
        self.offset_y = base_y + self.pan_y

    def image_to_widget(self, x: float, y: float):
        # OpenCV coordinates identify pixel centers, while QPainter draws the
        # image from its outer boundary. Account for that half-pixel difference
        # so fitted geometry is not displayed 0.5 px toward the upper-left.
        return self.offset_x + (x + 0.5) * self.scale, self.offset_y + (y + 0.5) * self.scale

    def _rotated_rect_points_widget(self, cx: float, cy: float, w: float, h: float, angle_deg: float):
        theta = np.deg2rad(angle_deg)
        ct, st = np.cos(theta), np.sin(theta)
        hw, hh = w / 2.0, h / 2.0
        local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        pts = []
        for lx, ly in local:
            ix = cx + ct * lx - st * ly
            iy = cy + st * lx + ct * ly
            wx, wy = self.image_to_widget(ix, iy)
            pts.append(QPointF(wx, wy))
        return pts

    def _draw_roi_shape(
        self,
        painter: QPainter,
        roi: Roi,
        color: QColor,
        active: bool,
        label: str = "",
        *,
        show_calipers: bool = True,
    ):
        r = roi.normalized()
        pen = QPen(color, 2.5 if active else 1.5)
        pen.setStyle(Qt.SolidLine if active else Qt.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        cx, cy = r.center()
        wcx, wcy = self.image_to_widget(cx, cy)
        typ = getattr(r, "roi_type", "Rectangle")

        if typ == "Circle":
            radius = r.outer_radius() * self.scale
            painter.drawEllipse(QRectF(wcx - radius, wcy - radius, 2 * radius, 2 * radius))
        elif typ == "Ellipse":
            painter.save()
            painter.translate(wcx, wcy)
            painter.rotate(r.angle_deg)
            painter.drawEllipse(QRectF(-r.w * self.scale / 2.0, -r.h * self.scale / 2.0, r.w * self.scale, r.h * self.scale))
            painter.restore()
        elif typ in {"Annulus", "Caliper Circle"}:
            outer = r.outer_radius() * self.scale
            inner = r.inner_radius() * self.scale
            if typ == "Caliper Circle" and show_calipers:
                ring_path = QPainterPath()
                ring_path.addEllipse(QRectF(wcx - outer, wcy - outer, 2 * outer, 2 * outer))
                inner_path = QPainterPath()
                inner_path.addEllipse(QRectF(wcx - inner, wcy - inner, 2 * inner, 2 * inner))
                ring_path = ring_path.subtracted(inner_path)
                painter.fillPath(ring_path, QColor(0, 220, 255, 32))
            painter.drawEllipse(QRectF(wcx - outer, wcy - outer, 2 * outer, 2 * outer))
            inner_pen = QPen(color, 1.8 if active else 1.2)
            inner_pen.setStyle(Qt.DotLine)
            inner_pen.setCosmetic(True)
            painter.setPen(inner_pen)
            painter.drawEllipse(QRectF(wcx - inner, wcy - inner, 2 * inner, 2 * inner))
            painter.setPen(pen)
            if typ == "Caliper Circle" and show_calipers:
                mid = 0.5 * (outer + inner)
                middle_pen = QPen(QColor(255, 220, 40), 1.3)
                middle_pen.setStyle(Qt.DashLine)
                middle_pen.setCosmetic(True)
                painter.setPen(middle_pen)
                painter.drawEllipse(QRectF(wcx - mid, wcy - mid, 2 * mid, 2 * mid))
                painter.setPen(pen)
                self._draw_calipers(painter, r, color)
        elif typ in {"Rectangular Ring", "Approximate Line"}:
            outer_poly = QPolygonF(self._rotated_rect_points_widget(cx, cy, r.w, r.h, r.angle_deg))
            painter.drawPolygon(outer_poly)
            if typ == "Rectangular Ring":
                iw, ih = r.inner_size()
                inner_poly = QPolygonF(self._rotated_rect_points_widget(cx, cy, iw, ih, r.angle_deg))
                inner_pen = QPen(color, 1.8 if active else 1.2)
                inner_pen.setStyle(Qt.DotLine)
                inner_pen.setCosmetic(True)
                painter.setPen(inner_pen)
                painter.drawPolygon(inner_poly)
                painter.setPen(pen)
            else:
                theta = np.deg2rad(r.angle_deg)
                dx, dy = np.cos(theta) * r.w / 2.0, np.sin(theta) * r.w / 2.0
                x0, y0 = self.image_to_widget(cx - dx, cy - dy)
                x1, y1 = self.image_to_widget(cx + dx, cy + dy)
                center_pen = QPen(QColor("#FFD60A"), 1.6)
                center_pen.setStyle(Qt.DashLine)
                center_pen.setCosmetic(True)
                painter.setPen(center_pen)
                painter.drawLine(int(x0), int(y0), int(x1), int(y1))
                painter.setPen(pen)
        else:
            x, y = self.image_to_widget(r.x, r.y)
            painter.drawRect(QRectF(x, y, r.w * self.scale, r.h * self.scale))

        # Center cross for advanced ROI modes so users can verify concentricity.
        if typ in {"Circle", "Annulus", "Rectangular Ring", "Caliper Circle"}:
            painter.drawLine(int(wcx - 6), int(wcy), int(wcx + 6), int(wcy))
            painter.drawLine(int(wcx), int(wcy - 6), int(wcx), int(wcy + 6))

        if label:
            # Place label near the top-left of the outer bounding box.
            x, y = self.image_to_widget(r.x, r.y)
            typ_label = {"Annulus": "圆环", "Caliper Circle": "卡尺圆", "Rectangular Ring": "矩形环", "Circle": "圆", "Ellipse": "椭圆", "Rectangle": "矩形", "Approximate Line": "近似直线", "Region Center": "区域中心", "Robust Center": "稳健中心"}.get(typ, typ)
            painter.drawText(int(x + 4), int(y + 16), f"{label} [{typ_label}]")

    def _draw_roi_handles(self, painter: QPainter, roi: Roi):
        handles = self._roi_handle_points(roi)
        if not handles:
            return
        painter.setPen(QPen(QColor("#1473E6"), 1.4))
        painter.setBrush(QColor("#FFFFFF"))
        for hx, hy in handles.values():
            painter.drawRect(QRectF(hx - 4.0, hy - 4.0, 8.0, 8.0))
        cx, cy = self.image_to_widget(*roi.center())
        painter.setPen(QPen(QColor("#1473E6"), 1.5))
        painter.setBrush(QColor("#1473E6"))
        painter.drawEllipse(QRectF(cx - 3.5, cy - 3.5, 7.0, 7.0))
        painter.setBrush(Qt.NoBrush)

    def _draw_calipers(self, painter: QPainter, roi: Roi, color: QColor):
        r = roi.normalized()
        cx, cy = r.center()
        inner = r.inner_radius()
        outer = r.outer_radius()
        mid = 0.5 * (inner + outer)
        length = outer - inner
        width = float(getattr(r, "caliper_width_px", 8.0))
        count = int(np.clip(getattr(r, "caliper_count", 64), 4, 720))
        direction = getattr(r, "search_direction", "Inner to Outer")
        caliper_pen = QPen(QColor(255, 230, 40), 1.0)
        caliper_pen.setCosmetic(True)
        arrow_pen = QPen(QColor(0, 255, 255), 1.2)
        arrow_pen.setCosmetic(True)
        for i in range(count):
            angle = 2.0 * np.pi * i / count
            radial = np.array([np.cos(angle), np.sin(angle)])
            tangent = np.array([-np.sin(angle), np.cos(angle)])
            center = np.array([cx, cy]) + radial * mid
            corners = []
            for rs, ts in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]:
                p = center + radial * (rs * length) + tangent * (ts * width)
                wx, wy = self.image_to_widget(float(p[0]), float(p[1]))
                corners.append(QPointF(wx, wy))
            painter.setPen(caliper_pen)
            painter.drawPolygon(QPolygonF(corners))
            if direction == "Outer to Inner":
                p0 = np.array([cx, cy]) + radial * (outer - 0.18 * length)
                p1 = np.array([cx, cy]) + radial * (inner + 0.18 * length)
            else:
                p0 = np.array([cx, cy]) + radial * (inner + 0.18 * length)
                p1 = np.array([cx, cy]) + radial * (outer - 0.18 * length)
            x0, y0 = self.image_to_widget(float(p0[0]), float(p0[1]))
            x1, y1 = self.image_to_widget(float(p1[0]), float(p1[1]))
            painter.setPen(arrow_pen)
            painter.drawLine(int(x0), int(y0), int(x1), int(y1))
            # arrow head
            v = np.array([x1 - x0, y1 - y0], dtype=np.float64)
            n = np.linalg.norm(v)
            if n > 1e-6:
                v /= n
                t = np.array([-v[1], v[0]])
                for sgn in (-1, 1):
                    h = np.array([x1, y1]) - v * 6 + t * sgn * 3
                    painter.drawLine(int(x1), int(y1), int(h[0]), int(h[1]))

    def widget_to_image_float(self, pos):
        if self.image is None or self.scale <= 0:
            return None
        h, w = self.image.gray.shape[:2]
        x = (pos.x() - self.offset_x) / self.scale - 0.5
        y = (pos.y() - self.offset_y) / self.scale - 0.5
        if x < -0.5 or y < -0.5 or x >= w - 0.5 or y >= h - 0.5:
            return None
        return float(x), float(y)

    def widget_to_image(self, pos) -> Optional[QPoint]:
        p = self.widget_to_image_float(pos)
        if p is None:
            return None
        x, y = p
        h, w = self.image.gray.shape[:2]
        return QPoint(int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1)))

    def _active_roi(self) -> Optional[Roi]:
        mark = self.marks.get(self.active_mark_id)
        if mark is None or not self.active_roi_id:
            return None
        entry = mark.roi_entry(self.active_layer, self.active_roi_id)
        if entry is None:
            return None
        if self.roi_edit_id == entry.roi_id and self.roi_edit_preview is not None:
            return self.roi_edit_preview
        return entry.roi

    def _roi_handle_points(self, roi: Optional[Roi] = None):
        roi = (roi or self._active_roi())
        if roi is None:
            return {}
        r = roi.normalized()
        cx, cy = r.center()
        typ = getattr(r, "roi_type", "Rectangle")
        handles = {}

        def add(name, x, y):
            handles[name] = self.image_to_widget(float(x), float(y))

        if typ in {"Circle", "Annulus", "Caliper Circle"}:
            outer = r.outer_radius()
            for name, dx, dy in (("outer_e", outer, 0), ("outer_w", -outer, 0),
                                 ("outer_n", 0, -outer), ("outer_s", 0, outer)):
                add(name, cx + dx, cy + dy)
            if typ in {"Annulus", "Caliper Circle"}:
                inner = r.inner_radius()
                for name, dx, dy in (("inner_e", inner, 0), ("inner_w", -inner, 0),
                                     ("inner_n", 0, -inner), ("inner_s", 0, inner)):
                    add(name, cx + dx, cy + dy)
            return handles

        theta = np.deg2rad(r.angle_deg if typ in {"Ellipse", "Rectangular Ring", "Approximate Line"} else 0.0)
        ct, st = np.cos(theta), np.sin(theta)

        def world(lx, ly):
            return cx + ct * lx - st * ly, cy + st * lx + ct * ly

        if typ == "Approximate Line":
            add("line_start", *world(-r.w / 2.0, 0.0))
            add("line_end", *world(r.w / 2.0, 0.0))
            add("line_width", *world(0.0, -r.h / 2.0))
            return handles

        half_w, half_h = r.w / 2.0, r.h / 2.0
        for name, lx, ly in (
            ("outer_nw", -half_w, -half_h), ("outer_n", 0, -half_h),
            ("outer_ne", half_w, -half_h), ("outer_e", half_w, 0),
            ("outer_se", half_w, half_h), ("outer_s", 0, half_h),
            ("outer_sw", -half_w, half_h), ("outer_w", -half_w, 0),
        ):
            add(name, *world(lx, ly))
        if typ == "Rectangular Ring":
            inner_w, inner_h = r.inner_size()
            for name, lx, ly in (
                ("inner_n", 0, -inner_h / 2.0), ("inner_e", inner_w / 2.0, 0),
                ("inner_s", 0, inner_h / 2.0), ("inner_w", -inner_w / 2.0, 0),
            ):
                add(name, *world(lx, ly))
        return handles

    def _roi_hit_part(self, pos) -> str:
        if self._active_roi() is None:
            return ""
        best = ""
        best_distance = 9.0
        for name, (hx, hy) in self._roi_handle_points().items():
            distance = float(np.hypot(hx - pos.x(), hy - pos.y()))
            if distance <= best_distance:
                best, best_distance = name, distance
        if best:
            return best
        return ""

    def _point_in_active_roi_band(self, pos) -> bool:
        roi = self._active_roi()
        p = self.widget_to_image_float(pos)
        if roi is None or p is None:
            return False
        r = roi.normalized()
        point = np.array([[p[0], p[1]]], dtype=np.float64)
        return bool(r.contains_points(point)[0])

    def _point_in_active_roi_outer(self, pos) -> bool:
        roi = self._active_roi()
        p = self.widget_to_image_float(pos)
        if roi is None or p is None:
            return False
        r = roi.normalized()
        x, y = p
        if r.roi_type in {"Annulus", "Caliper Circle"}:
            cx, cy = r.center()
            return float(np.hypot(x - cx, y - cy)) <= r.outer_radius()
        if r.roi_type == "Rectangular Ring":
            return bool(replace(r, roi_type="Rectangle").contains_points(np.array([[x, y]], dtype=np.float64))[0])
        if r.roi_type == "Approximate Line":
            return bool(r.contains_points(np.array([[x, y]], dtype=np.float64))[0])
        return bool(r.contains_points(np.array([[x, y]], dtype=np.float64))[0])

    def _circle_from_three_points(self, pts):
        (x1, y1), (x2, y2), (x3, y3) = pts
        d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(d) < 1e-9:
            return None
        ux = (
            (x1 * x1 + y1 * y1) * (y2 - y3)
            + (x2 * x2 + y2 * y2) * (y3 - y1)
            + (x3 * x3 + y3 * y3) * (y1 - y2)
        ) / d
        uy = (
            (x1 * x1 + y1 * y1) * (x3 - x2)
            + (x2 * x2 + y2 * y2) * (x1 - x3)
            + (x3 * x3 + y3 * y3) * (x2 - x1)
        ) / d
        radius = float(np.hypot(x1 - ux, y1 - uy))
        return ux, uy, radius

    def _caliper_roi_from_center_circle(self, circle):
        if circle is None:
            return None
        cx, cy, mid_radius = circle
        half_width = self.active_ring_half_width_px
        inner_radius = max(0.0, mid_radius - half_width)
        outer_radius = max(inner_radius + 1.0, mid_radius + half_width)
        return Roi(
            cx - outer_radius,
            cy - outer_radius,
            outer_radius * 2.0,
            outer_radius * 2.0,
            "Caliper Circle",
            inner_radius / max(outer_radius, 1e-9),
            self.active_roi_target_edge,
            self.active_roi_angle_deg,
            self.active_caliper_count,
            self.active_caliper_width_px,
            self.active_search_direction,
            self.active_diameter_mode,
        ).normalized()

    def _move_active_roi(self, pos):
        if not self.is_moving_roi or self.move_start_img is None or self.move_start_roi is None:
            return
        p = self.widget_to_image_float(pos)
        if p is None:
            return
        dx = p[0] - self.move_start_img[0]
        dy = p[1] - self.move_start_img[1]
        self.roi_edit_preview = replace(
            self.move_start_roi,
            x=self.move_start_roi.x + dx,
            y=self.move_start_roi.y + dy,
        ).normalized()

    def _adjust_active_roi(self, pos):
        p = self.widget_to_image_float(pos)
        if self.roi_edit_start is None or p is None:
            return
        r = self.roi_edit_start.normalized()
        roi = replace(r)
        x, y = p
        typ = getattr(r, "roi_type", "Annulus")
        min_outer = 5.0
        min_width = 2.0

        if typ in {"Circle", "Annulus", "Caliper Circle"}:
            cx, cy = r.center()
            dist = max(min_outer, float(np.hypot(x - cx, y - cy)))
            outer = r.outer_radius()
            inner = r.inner_radius()
            if self.adjust_roi_part.startswith("inner_") and typ != "Circle":
                new_inner = float(np.clip(dist, min_width, max(min_width, outer - min_width)))
                roi.inner_ratio = new_inner / max(outer, 1e-9)
            elif self.adjust_roi_part.startswith("outer_"):
                new_outer = max(dist, inner + min_width, min_outer)
                roi.x = cx - new_outer
                roi.y = cy - new_outer
                roi.w = new_outer * 2.0
                roi.h = new_outer * 2.0
                roi.inner_ratio = float(np.clip(inner / max(new_outer, 1e-9), 0.0, 0.98))

        elif typ == "Approximate Line":
            theta = np.deg2rad(r.angle_deg)
            axis = np.asarray([np.cos(theta), np.sin(theta)], dtype=float)
            normal = np.asarray([-axis[1], axis[0]], dtype=float)
            center = np.asarray(r.center(), dtype=float)
            if self.adjust_roi_part in {"line_start", "line_end"}:
                fixed = center + axis * (r.w / 2.0 if self.adjust_roi_part == "line_start" else -r.w / 2.0)
                dragged = np.asarray([x, y], dtype=float)
                vector = dragged - fixed
                length = max(5.0, float(np.linalg.norm(vector)))
                if length > 1e-9:
                    new_center = (fixed + dragged) / 2.0
                    roi.x = float(new_center[0] - length / 2.0)
                    roi.y = float(new_center[1] - r.h / 2.0)
                    roi.w = length
                    roi.h = r.h
                    roi.angle_deg = float(np.rad2deg(np.arctan2(vector[1], vector[0])))
                    if self.adjust_roi_part == "line_start":
                        roi.angle_deg = (roi.angle_deg + 180.0) % 360.0
            elif self.adjust_roi_part == "line_width":
                half_width = max(2.0, abs(float(np.dot(np.asarray([x, y]) - center, normal))))
                roi.h = 2.0 * half_width
        else:
            xs = np.array([x], dtype=np.float64)
            ys = np.array([y], dtype=np.float64)
            xr, yr = r._local_rotated(xs, ys)
            ax, ay = abs(float(xr[0])), abs(float(yr[0]))
            if self.adjust_roi_part.startswith("inner_") and typ == "Rectangular Ring":
                ratio = max(ax / max(r.w / 2.0, 1e-9), ay / max(r.h / 2.0, 1e-9))
                roi.inner_ratio = float(np.clip(ratio, 0.02, 0.98))
            elif self.adjust_roi_part.startswith("outer_"):
                cx, cy = r.center()
                handle = self.adjust_roi_part.split("_", 1)[1]
                new_w, new_h = r.w, r.h
                if "e" in handle or "w" in handle:
                    new_w = max(min_outer, 2.0 * ax)
                if "n" in handle or "s" in handle:
                    new_h = max(min_outer, 2.0 * ay)
                roi.x = cx - new_w / 2.0
                roi.y = cy - new_h / 2.0
                roi.w = new_w
                roi.h = new_h
        self.roi_edit_preview = roi.normalized()

    def _begin_roi_edit(self, mode: str, part: str, pos, pending_caliper=None):
        roi = self._active_roi()
        point = self.widget_to_image_float(pos)
        if roi is None or point is None or not self.active_roi_id:
            return False
        self.roi_edit_id = self.active_roi_id
        self.roi_edit_start = deepcopy(roi.normalized())
        self.roi_edit_preview = deepcopy(roi.normalized())
        self.roi_edit_press_widget = QPointF(float(pos.x()), float(pos.y()))
        self.roi_edit_moved = False
        self.pending_caliper_hit = pending_caliper
        if mode == "move":
            self.is_moving_roi = True
            self.move_start_img = point
            self.move_start_roi = deepcopy(roi.normalized())
        else:
            self.is_adjusting_roi = True
            self.adjust_roi_part = part
            self.adjust_mark_id = self.active_mark_id
            self.adjust_layer = self.active_layer
        self.setCursor(Qt.ClosedHandCursor if mode == "move" else Qt.SizeAllCursor)
        return True

    def _roi_edit_distance(self, pos) -> float:
        if self.roi_edit_press_widget is None:
            return 0.0
        return float(np.hypot(pos.x() - self.roi_edit_press_widget.x(), pos.y() - self.roi_edit_press_widget.y()))

    def _clear_roi_edit_state(self):
        self.is_moving_roi = False
        self.is_adjusting_roi = False
        self.adjust_roi_part = ""
        self.adjust_mark_id = ""
        self.adjust_layer = ""
        self.move_start_img = None
        self.move_start_roi = None
        self.roi_edit_preview = None
        self.roi_edit_start = None
        self.roi_edit_id = ""
        self.roi_edit_press_widget = None
        self.roi_edit_moved = False
        self.pending_caliper_hit = None

    def _cancel_roi_edit(self):
        if not (self.is_moving_roi or self.is_adjusting_roi):
            return False
        self._clear_roi_edit_state()
        self.setCursor(Qt.ArrowCursor)
        self.update()
        return True

    def _roi_from_drag(self, start: QPoint, end: QPoint) -> Optional[Roi]:
        x0, y0 = float(start.x()), float(start.y())
        x1, y1 = float(end.x()), float(end.y())
        if self.active_roi_type == "Approximate Line":
            length = float(np.hypot(x1 - x0, y1 - y0))
            if length < 5.0:
                return None
            band = max(12.0, 3.0 * float(self.active_caliper_width_px))
            return Roi(
                (x0 + x1 - length) / 2.0,
                (y0 + y1 - band) / 2.0,
                length,
                band,
                "Approximate Line",
                self.active_roi_inner_ratio,
                self.active_roi_target_edge,
                float(np.rad2deg(np.arctan2(y1 - y0, x1 - x0))),
                caliper_width_px=self.active_caliper_width_px,
            ).normalized()

        dx, dy = x1 - x0, y1 - y0
        circular = self.active_roi_type in {"Circle", "Annulus", "Caliper Circle"}
        if circular:
            side = min(abs(dx), abs(dy))
            if side < 5.0:
                return None
            dx = side if dx >= 0 else -side
            dy = side if dy >= 0 else -side
        elif abs(dx) < 5.0 or abs(dy) < 5.0:
            return None
        return Roi(
            x0,
            y0,
            dx,
            dy,
            self.active_roi_type,
            self.active_roi_inner_ratio,
            self.active_roi_target_edge,
            self.active_roi_angle_deg,
            self.active_caliper_count,
            self.active_caliper_width_px,
            self.active_search_direction,
            self.active_diameter_mode,
        ).normalized()

    def _update_roi_hover_cursor(self, pos):
        if not self.roi_editing_enabled:
            self.setCursor(Qt.ArrowCursor)
            return
        if self.space_pan_held:
            self.setCursor(Qt.OpenHandCursor)
            return
        if self.active_roi_id and self._roi_hit_part(pos):
            self.setCursor(Qt.SizeAllCursor)
        elif self.active_roi_id and self._point_in_active_roi_outer(pos):
            self.setCursor(Qt.OpenHandCursor)
        elif self._manual_roi_hit(pos):
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.CrossCursor)

    def reset_view(self, update: bool = True):
        self.user_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        if update:
            self.update()

    def zoom_by(self, factor: float, center_pos=None):
        if self.pixmap_cache is None:
            return
        self._update_transform()
        if center_pos is None:
            center_pos = self.rect().center()
        before = self.widget_to_image_float(center_pos)
        if before is None:
            # Zoom around widget center when the cursor is outside the image.
            before = (
                (center_pos.x() - self.offset_x) / max(self.scale, 1e-12) - 0.5,
                (center_pos.y() - self.offset_y) / max(self.scale, 1e-12) - 0.5,
            )
        self.user_zoom = float(np.clip(self.user_zoom * factor, 0.05, 80.0))
        new_scale = self.fit_scale * self.user_zoom
        base_x, base_y = self._base_offset_for_scale(new_scale)
        img_x, img_y = before
        self.pan_x = center_pos.x() - (img_x + 0.5) * new_scale - base_x
        self.pan_y = center_pos.y() - (img_y + 0.5) * new_scale - base_y
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#252930"))

        if self.pixmap_cache is None:
            painter.setPen(QColor("#F5F6F8"))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待导入图像")
            painter.end()
            return

        self._update_transform()
        target = QRectF(self.offset_x, self.offset_y, self.pixmap_cache.width() * self.scale, self.pixmap_cache.height() * self.scale)
        painter.drawPixmap(target, self.pixmap_cache, QRectF(self.pixmap_cache.rect()))

        self._draw_overlays(painter)
        self._draw_geometry_overlays(painter)
        if self.profile_capture_enabled and self.profile_start_img is not None:
            start_x, start_y = self.image_to_widget(*self.profile_start_img)
            end = self.profile_current_img or self.profile_start_img
            end_x, end_y = self.image_to_widget(*end)
            profile_pen = QPen(QColor("#00D4FF"), 2.0)
            profile_pen.setCosmetic(True)
            painter.setPen(profile_pen)
            painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
            painter.drawEllipse(QRectF(start_x - 3, start_y - 3, 6, 6))
            painter.drawEllipse(QRectF(end_x - 3, end_y - 3, 6, 6))

        header_rect = QRectF(8, 8, min(270, self.width() - 16), 48)
        painter.fillRect(header_rect, QColor(18, 21, 26, 185))
        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(QFont("Microsoft YaHei UI", 9, QFont.DemiBold))
        painter.drawText(16, 27, self.title)
        painter.setFont(QFont("Microsoft YaHei UI", 8))
        painter.setPen(QColor("#D7DCE3"))
        painter.drawText(16, 46, f"缩放 {self.user_zoom:.2f}x  ·  滚轮缩放 / 中键或空格拖动平移")
        if self.circle_pick_mode:
            hint = f"三点定圆：已选 {len(self.circle_pick_points)}/3 点；可随时中键或空格拖动平移"
            hint_rect = QRectF(12, self.height() - 76, min(360, self.width() - 24), 28)
            painter.fillRect(hint_rect, QColor(18, 21, 26, 205))
            painter.setPen(QColor("#7EE787"))
            painter.drawText(hint_rect.adjusted(8, 0, -8, 0), Qt.AlignVCenter | Qt.AlignLeft, hint)
        elif self.profile_capture_enabled:
            hint_rect = QRectF(12, self.height() - 76, min(360, self.width() - 24), 28)
            painter.fillRect(hint_rect, QColor(18, 21, 26, 205))
            painter.setPen(QColor("#00D4FF"))
            painter.drawText(hint_rect.adjusted(8, 0, -8, 0), Qt.AlignVCenter | Qt.AlignLeft, "图像诊断：拖动剖面线；Esc 退出")
        elif self._has_caliper_result_for_hint():
            hint = (
                "卡尺调整中 · 点击空白处隐藏"
                if self.selected_caliper_feature
                else "点击拟合轮廓显示卡尺，点击空白处隐藏"
            )
            hint_rect = QRectF(12, self.height() - 76, min(330, self.width() - 24), 28)
            painter.fillRect(hint_rect, QColor(18, 21, 26, 205))
            painter.setPen(QColor("#7EE787"))
            painter.drawText(hint_rect.adjusted(8, 0, -8, 0), Qt.AlignVCenter | Qt.AlignLeft, hint)
        self._draw_scale_and_axes(painter)
        painter.end()

    def _draw_geometry_overlays(self, painter: QPainter):
        result = self.geometry_result
        if result is None:
            return
        feature_pen = QPen(QColor("#34C759"), 2.0)
        feature_pen.setCosmetic(True)
        selected_pen = QPen(QColor("#FFD60A"), 2.5)
        selected_pen.setCosmetic(True)
        for feature_id, feature in result.features.items():
            if feature.status != "Valid" or feature.layer != self.active_layer:
                continue
            highlighted = feature_id in {self.selected_geometry_feature_id, self.hovered_geometry_feature_id}
            painter.setPen(selected_pen if highlighted else feature_pen)
            if feature.center_px is not None:
                cx, cy = self.image_to_widget(*feature.center_px)
                painter.drawLine(int(cx - 6), int(cy), int(cx + 6), int(cy))
                painter.drawLine(int(cx), int(cy - 6), int(cx), int(cy + 6))
                painter.drawText(int(cx + 8), int(cy - 8), feature.name)
                if feature.radius_px is not None:
                    radius = float(feature.radius_px) * self.scale
                    painter.drawEllipse(QRectF(cx - radius, cy - radius, 2.0 * radius, 2.0 * radius))
            if feature.feature_type == "line" and len(feature.points_px) >= 2:
                first = self.image_to_widget(*feature.points_px[0])
                second = self.image_to_widget(*feature.points_px[1])
                painter.drawLine(QPointF(*first), QPointF(*second))

        measurement_pen = QPen(QColor("#00C7BE"), 1.6)
        measurement_pen.setCosmetic(True)
        for measurement in result.measurements.values():
            if measurement.status != "Valid" or measurement.layer != self.active_layer:
                continue
            refs = [result.features.get(item) for item in measurement.reference_ids]
            refs = [item for item in refs if item is not None and item.center_px is not None]
            if not refs:
                continue
            painter.setPen(measurement_pen)
            anchors = [self.image_to_widget(*item.center_px) for item in refs]
            if len(anchors) >= 2:
                painter.drawLine(QPointF(*anchors[0]), QPointF(*anchors[1]))
                tx = (anchors[0][0] + anchors[1][0]) / 2.0
                ty = (anchors[0][1] + anchors[1][1]) / 2.0
            else:
                tx, ty = anchors[0][0] + 12.0, anchors[0][1] - 12.0
            text = measurement.name
            if measurement.value is not None:
                text = f"{measurement.name}: {measurement.value:.3f} {measurement.unit}"
            bounds = painter.fontMetrics().boundingRect(text).adjusted(-6, -4, 6, 4)
            bounds.moveCenter(QPoint(int(tx), int(ty - 12)))
            painter.fillRect(bounds, QColor(18, 21, 26, 210))
            painter.drawText(bounds, Qt.AlignCenter, text)

        interaction = self.geometry_interaction
        if interaction and interaction.get("layer") == self.active_layer:
            preview_pen = QPen(QColor("#FFD60A"), 1.8, Qt.DashLine)
            preview_pen.setCosmetic(True)
            painter.setPen(preview_pen)
            points = [item.get("point_px") for item in interaction.get("clicks", []) if item.get("point_px")]
            for index, point in enumerate(points, start=1):
                wx, wy = self.image_to_widget(*point)
                painter.setBrush(QColor("#FFD60A"))
                painter.drawEllipse(QRectF(wx - 5, wy - 5, 10, 10))
                painter.setBrush(Qt.NoBrush)
                badge = QRectF(wx + 7, wy - 18, 22, 18)
                painter.fillRect(badge, QColor(18, 21, 26, 225))
                painter.drawText(badge, Qt.AlignCenter, str(index))
            hover = self.geometry_hover_point
            if hover is not None:
                hover_widget = self.image_to_widget(*hover)
                painter.drawEllipse(QRectF(hover_widget[0] - 6, hover_widget[1] - 6, 12, 12))
                painter.drawLine(
                    QPointF(hover_widget[0] - 9, hover_widget[1]),
                    QPointF(hover_widget[0] + 9, hover_widget[1]),
                )
                painter.drawLine(
                    QPointF(hover_widget[0], hover_widget[1] - 9),
                    QPointF(hover_widget[0], hover_widget[1] + 9),
                )
                if self.hovered_geometry_feature_id or self.hovered_detection_key:
                    painter.drawText(int(hover_widget[0] + 11), int(hover_widget[1] - 9), "吸附中心")
            if hover is not None and points:
                action = interaction.get("action", "")
                if action == "feature:circle" and len(points) >= 2:
                    circle = self._circle_from_three_points([points[0], points[1], hover])
                    if circle is not None:
                        cx, cy, radius = circle
                        wx, wy = self.image_to_widget(cx, cy)
                        painter.drawEllipse(QRectF(wx - radius * self.scale, wy - radius * self.scale, 2 * radius * self.scale, 2 * radius * self.scale))
                else:
                    painter.drawLine(QPointF(*self.image_to_widget(*points[-1])), QPointF(*self.image_to_widget(*hover)))

        coordinate_palette = ["#00C7BE", "#FF9500", "#AF52DE", "#007AFF", "#34C759", "#FF375F"]
        coordinates = list(result.coordinate_systems.values())
        for coordinate_index, coordinate in enumerate(coordinates):
            if coordinate.status != "Valid" or coordinate.layer != self.active_layer or coordinate.origin_px is None:
                continue
            highlighted = coordinate.coordinate_id == self.highlighted_coordinate_id
            color = QColor(coordinate_palette[coordinate_index % len(coordinate_palette)])
            if self.highlighted_coordinate_id and not highlighted:
                color.setAlpha(80)
            axis_pen = QPen(color, 3.2 if highlighted else 1.8)
            axis_pen.setCosmetic(True)
            origin = self.image_to_widget(*coordinate.origin_px)
            axis_length = 72.0 if highlighted else 62.0
            x_axis = coordinate.x_axis_image or (1.0, 0.0)
            y_axis = coordinate.y_axis_image or (0.0, -1.0)
            painter.setPen(axis_pen)
            painter.drawLine(QPointF(*origin), QPointF(origin[0] + x_axis[0] * axis_length, origin[1] + x_axis[1] * axis_length))
            painter.drawLine(QPointF(*origin), QPointF(origin[0] + y_axis[0] * axis_length, origin[1] + y_axis[1] * axis_length))
            painter.drawText(int(origin[0] + x_axis[0] * axis_length + 4), int(origin[1] + x_axis[1] * axis_length), "X")
            painter.drawText(int(origin[0] + y_axis[0] * axis_length + 4), int(origin[1] + y_axis[1] * axis_length), "Y")
            layer_label = "上层" if coordinate.layer == "upper" else "下层"
            name = coordinate.name or coordinate.coordinate_id
            badge = f"{name} [{coordinate.coordinate_id}] · {layer_label}"
            badge_rect = painter.fontMetrics().boundingRect(badge).adjusted(-6, -4, 6, 4)
            badge_rect.moveTopLeft(QPoint(int(origin[0] + 8), int(origin[1] + 8)))
            painter.fillRect(badge_rect, QColor(18, 21, 26, 225 if highlighted else 185))
            painter.setPen(axis_pen)
            painter.drawText(badge_rect, Qt.AlignCenter, badge)

        label_pen = QPen(QColor("#FFFFFF"), 1.2)
        label_pen.setCosmetic(True)
        for label in result.coordinate_labels.values():
            if label.status != "Valid" or label.layer != self.active_layer or label.anchor_px is None or label.label_px is None:
                continue
            anchor = self.image_to_widget(*label.anchor_px)
            target = self.image_to_widget(*label.label_px)
            painter.setPen(label_pen)
            painter.drawLine(QPointF(*anchor), QPointF(*target))
            text = f"{label.name}  X={label.x_um:.3f} μm  Y={label.y_um:.3f} μm"
            metrics = painter.fontMetrics()
            rect = metrics.boundingRect(text).adjusted(-7, -5, 7, 5)
            rect.moveTopLeft(QPoint(int(target[0]), int(target[1] - rect.height())))
            painter.fillRect(rect, QColor(18, 21, 26, 210))
            painter.drawText(rect, Qt.AlignCenter, text)

    def _draw_scale_and_axes(self, painter: QPainter):
        if self.pixmap_cache is None or self.scale <= 0:
            return
        mean_um = max(1e-12, self._mean_pixel_size_um())
        target_um = 50.0
        for candidate in (5, 10, 20, 50, 100, 200, 500, 1000):
            if candidate / mean_um * self.scale >= 60:
                target_um = float(candidate)
                break
        bar_px_widget = target_um / mean_um * self.scale
        x0 = 24
        y0 = self.height() - 36
        painter.setPen(QPen(QColor("#FFFFFF"), 3.0))
        painter.drawLine(int(x0), int(y0), int(x0 + bar_px_widget), int(y0))
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(int(x0), int(y0 - 8), f"{target_um:g} μm")
        ax0 = self.width() - 92
        ay0 = self.height() - 44
        painter.setPen(QPen(QColor("#FFFFFF"), 2.0))
        painter.drawLine(ax0, ay0, ax0 + 42, ay0)
        painter.drawLine(ax0, ay0, ax0, ay0 - 42)
        painter.drawText(ax0 + 48, ay0 + 5, "X")
        painter.drawText(ax0 - 10, ay0 - 48, "Y")

    def _draw_secondary_detection(self, painter: QPainter, detection: DetectionResult, label: str):
        """Draw a cheap summary for non-active manual ROIs.

        Full fitted contours are still available for the selected ROI and in
        diagnostic mode.  Avoid rebuilding hundreds of QPolygonF objects on
        every pan/zoom repaint when an array contains many ROIs.
        """
        color = QColor("#34C759")
        pen = QPen(color, 1.7)
        pen.setCosmetic(True)
        painter.setPen(pen)
        cx, cy = self.image_to_widget(detection.center_x_px, detection.center_y_px)
        painter.drawLine(int(cx - 5), int(cy), int(cx + 5), int(cy))
        painter.drawLine(int(cx), int(cy - 5), int(cx), int(cy + 5))
        if not self.show_diagnostics:
            painter.drawText(int(cx + 7), int(cy - 7), label)
            return
        mode = detection.fitting_mode
        if mode in {"Circle", "EdgeCenter", "CaliperCircle"}:
            radius = float(detection.shape_params.get("radius_px", detection.diameter_px / 2.0)) * self.scale
            painter.drawEllipse(QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius))
        elif mode == "Ellipse":
            major = float(detection.shape_params.get("major_px", detection.diameter_px)) * self.scale
            minor = float(detection.shape_params.get("minor_px", detection.diameter_px)) * self.scale
            painter.drawEllipse(QRectF(cx - major / 2.0, cy - minor / 2.0, major, minor))
        elif mode == "Line":
            start = detection.shape_params.get("line_start_px")
            end = detection.shape_params.get("line_end_px")
            if start is not None and end is not None:
                x0, y0 = self.image_to_widget(float(start[0]), float(start[1]))
                x1, y1 = self.image_to_widget(float(end[0]), float(end[1]))
                painter.drawLine(int(x0), int(y0), int(x1), int(y1))
        else:
            contour = detection.shape_params.get("contour_points") or detection.shape_params.get("region_box_points")
            if contour and len(contour) >= 2:
                points = [QPointF(*self.image_to_widget(float(x), float(y))) for x, y in contour]
                painter.drawPolygon(QPolygonF(points))
        painter.drawText(int(cx + 7), int(cy - 7), label)

    def _draw_ellipse_fit(self, painter: QPainter, detection: DetectionResult, color: QColor, show_axes: bool = True):
        """Draw the fitted, rotated ellipse in image coordinates and its physical axes."""
        major = float(detection.shape_params.get("major_px", detection.diameter_px))
        minor = float(detection.shape_params.get("minor_px", detection.diameter_px))
        angle = np.deg2rad(float(detection.shape_params.get("angle_deg", 0.0)))
        ct, st = np.cos(angle), np.sin(angle)
        points = []
        for theta in np.linspace(0.0, 2.0 * np.pi, 97):
            x = detection.center_x_px + 0.5 * major * np.cos(theta) * ct - 0.5 * minor * np.sin(theta) * st
            y = detection.center_y_px + 0.5 * major * np.cos(theta) * st + 0.5 * minor * np.sin(theta) * ct
            points.append(QPointF(*self.image_to_widget(x, y)))
        pen = QPen(color, 2.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawPolyline(QPolygonF(points))
        if not show_axes:
            return
        cx, cy = self.image_to_widget(detection.center_x_px, detection.center_y_px)
        major_a = self.image_to_widget(detection.center_x_px - 0.5 * major * ct, detection.center_y_px - 0.5 * major * st)
        major_b = self.image_to_widget(detection.center_x_px + 0.5 * major * ct, detection.center_y_px + 0.5 * major * st)
        minor_a = self.image_to_widget(detection.center_x_px + 0.5 * minor * st, detection.center_y_px - 0.5 * minor * ct)
        minor_b = self.image_to_widget(detection.center_x_px - 0.5 * minor * st, detection.center_y_px + 0.5 * minor * ct)
        major_pen = QPen(QColor("#34C759"), 1.5)
        major_pen.setCosmetic(True)
        painter.setPen(major_pen)
        painter.drawLine(int(major_a[0]), int(major_a[1]), int(major_b[0]), int(major_b[1]))
        minor_pen = QPen(QColor("#00D4FF"), 1.5)
        minor_pen.setCosmetic(True)
        painter.setPen(minor_pen)
        painter.drawLine(int(minor_a[0]), int(minor_a[1]), int(minor_b[0]), int(minor_b[1]))
        major_um = detection.ellipse_major_um or major * self.pixel_size_x_um
        minor_um = detection.ellipse_minor_um or minor * self.pixel_size_y_um
        painter.setPen(QColor("#34C759"))
        painter.drawText(int(cx + 10), int(cy - 25), f"长轴 {major_um:.3f} μm")
        painter.setPen(QColor("#00D4FF"))
        painter.drawText(int(cx + 10), int(cy - 8), f"短轴 {minor_um:.3f} μm")

    def _draw_overlays(self, painter: QPainter):
        if self.image is None:
            return
        colors = {
            "upper": QColor("#007AFF"),
            "lower": QColor("#FF9500"),
        }
        for mark_id, mark in self.marks.items():
            if self.show_auto_detections:
                break
            if mark_id != self.active_mark_id:
                continue
            for layer in ("upper", "lower"):
                if self.fixed_layer and layer != self.fixed_layer:
                    continue
                if not self.fixed_layer and layer != self.active_layer:
                    continue
                entries = mark.roi_entries(layer)
                active_entry = mark.roi_entry(layer, self.active_roi_id) if self.active_roi_id else None
                roi = active_entry.roi if active_entry is not None else None
                if active_entry is not None and self.roi_edit_id == active_entry.roi_id and self.roi_edit_preview is not None:
                    roi = self.roi_edit_preview
                active_roi_id = active_entry.roi_id if active_entry is not None else ""
                det = self.roi_detections.get(mark_id, {}).get(active_roi_id)
                detection_valid = det is not None and det.shape_params.get("quality_status", "Valid") != "Invalid"
                for index, entry in enumerate(entries, start=1):
                    if entry.roi_id == active_roi_id:
                        continue
                    other_detection = self.roi_detections.get(mark_id, {}).get(entry.roi_id)
                    if self._manual_roi_visible(mark_id, entry.roi_id, layer, entry.roi, other_detection):
                        self._draw_roi_shape(
                            painter, entry.roi, QColor(colors[layer].red(), colors[layer].green(), colors[layer].blue(), 135),
                            False, f"{mark_id} {LAYER_LABELS[layer]} ROI {index}"
                        )
                    if other_detection is not None:
                        self._draw_secondary_detection(painter, other_detection, f"ROI {index}")
                is_active = active_entry is not None and mark_id == self.active_mark_id and layer == self.active_layer
                if roi is not None and (is_active or self._manual_roi_visible(mark_id, active_roi_id, layer, roi, det)):
                    active_index = entries.index(active_entry) + 1 if active_entry in entries else 1
                    calipers_selected = det is None or self._manual_caliper_selected(mark_id, active_roi_id, layer, det)
                    self._draw_roi_shape(
                        painter,
                        roi,
                        colors[layer],
                        is_active,
                        f"{mark_id} {LAYER_LABELS[layer]} ROI {active_index}",
                        show_calipers=calipers_selected,
                    )
                    if is_active and self.roi_editing_enabled:
                        self._draw_roi_handles(painter, roi)

                if det is not None:
                    show_point_diagnostics = self.show_diagnostics or not detection_valid
                    if show_point_diagnostics:
                        painter.setPen(QPen(QColor("#34C759"), 1.0))
                        pts = det.edge_points
                        if pts:
                            step = max(1, len(pts) // 1200)
                            for px, py in pts[::step]:
                                wx, wy = self.image_to_widget(px, py)
                                painter.drawEllipse(QRectF(wx - 2.0, wy - 2.0, 4.0, 4.0))
                        rejected = getattr(det, "rejected_points", [])
                        if rejected:
                            painter.setPen(QPen(QColor("#FF3B30"), 1.4))
                            for px, py in rejected:
                                wx, wy = self.image_to_widget(px, py)
                                painter.drawLine(int(wx - 3), int(wy - 3), int(wx + 3), int(wy + 3))
                                painter.drawLine(int(wx - 3), int(wy + 3), int(wx + 3), int(wy - 3))
                    cx, cy = self.image_to_widget(det.center_x_px, det.center_y_px)
                    fit_color = QColor("#34C759")
                    painter.setPen(QPen(fit_color, 2.0))
                    painter.drawLine(int(cx - 8), int(cy), int(cx + 8), int(cy))
                    painter.drawLine(int(cx), int(cy - 8), int(cx), int(cy + 8))
                    contour_label = self.manual_labels.get((mark_id, active_roi_id), "")
                    if contour_label:
                        label_x, label_y = self._contour_label_anchor(det)
                        painter.drawText(int(label_x + 6), int(label_y - 6), f"{contour_label} ({mark_id})")
                    if det.fitting_mode == "Line":
                        start = det.shape_params.get("line_start_px")
                        end = det.shape_params.get("line_end_px")
                        if start is not None and end is not None:
                            x0, y0 = self.image_to_widget(float(start[0]), float(start[1]))
                            x1, y1 = self.image_to_widget(float(end[0]), float(end[1]))
                            line_pen = QPen(fit_color, 2.5)
                            line_pen.setCosmetic(True)
                            painter.setPen(line_pen)
                            painter.drawLine(int(x0), int(y0), int(x1), int(y1))
                            painter.drawText(int(cx + 10), int(cy - 10), f"直线 角度={det.shape_params.get('line_angle_deg', 0.0):.3f}°")
                    elif det.fitting_mode in {"Circle", "EdgeCenter", "CaliperCircle"} and "radius_px" in det.shape_params:
                        rad = det.shape_params["radius_px"] * self.scale
                        painter.setPen(QPen(fit_color, 2.0))
                        painter.drawEllipse(QRectF(cx - rad, cy - rad, 2 * rad, 2 * rad))
                        if det.fitting_mode == "CaliperCircle" and self.show_diagnostics:
                            average_um = float(det.shape_params.get("average_diameter_um", det.diameter_um))
                            maximum_um = float(det.shape_params.get("maximum_diameter_um", average_um))
                            painter.setPen(fit_color)
                            painter.drawText(
                                int(cx + 12),
                                int(cy - 12),
                                f"中心=({det.center_x_um:.3f},{det.center_y_um:.3f}) μm 平均直径={average_um:.3f} μm 最大直径={maximum_um:.3f} μm",
                            )
                    elif det.fitting_mode == "RegionCenter":
                        # V1.4: region-center mode is area segmentation, not circle fitting.
                        # Show only the final selected main contour and min-area box; do not
                        # draw the equivalent-area circle because it is misleading for rounded square holes.
                        contour_points = det.shape_params.get("contour_points", [])
                        if contour_points:
                            widget_points = [QPointF(*self.image_to_widget(float(px), float(py))) for px, py in contour_points]
                            if len(widget_points) >= 3:
                                fill_path = QPainterPath()
                                fill_path.moveTo(widget_points[0])
                                for pt in widget_points[1:]:
                                    fill_path.lineTo(pt)
                                fill_path.closeSubpath()
                                painter.fillPath(fill_path, QColor(52, 199, 89, 36))
                                contour_pen = QPen(QColor("#34C759"), 2.0)
                                contour_pen.setCosmetic(True)
                                painter.setPen(contour_pen)
                                painter.drawPolygon(QPolygonF(widget_points))
                        box_points = det.shape_params.get("region_box_points", [])
                        if box_points:
                            box_widget = [QPointF(*self.image_to_widget(float(px), float(py))) for px, py in box_points]
                            if len(box_widget) >= 4:
                                fit_pen = QPen(fit_color, 2.0)
                                fit_pen.setCosmetic(True)
                                painter.setPen(fit_pen)
                                painter.drawPolygon(QPolygonF(box_widget))
                        width = det.shape_params.get("width_px", 0.0) * self.pixel_size_x_um
                        height = det.shape_params.get("height_px", 0.0) * self.pixel_size_y_um
                        area = det.shape_params.get("region_area_px2", 0.0)
                        polarity = det.shape_params.get("region_polarity", "")
                        painter.setPen(fit_color)
                        painter.drawText(
                            int(cx + 10),
                            int(cy - 10),
                            f"区域中心 W={width:.3f} μm H={height:.3f} μm 面积={area:.0f}px² {polarity}",
                        )
                    elif det.fitting_mode == "Ellipse":
                        self._draw_ellipse_fit(painter, det, fit_color, show_axes=True)
                    elif det.fitting_mode == "Rectangle":
                        # V1.0.4: draw a clearly visible rotated rectangle contour.
                        # Earlier versions calculated the rectangle center, but the outline
                        # could be too thin/ambiguous on high-resolution microscope images.
                        width = det.shape_params.get("width_px", det.diameter_px)
                        height = det.shape_params.get("height_px", det.diameter_px)
                        angle_deg = det.shape_params.get("angle_deg", 0.0)
                        angle = np.deg2rad(angle_deg)
                        hw = width / 2.0
                        hh = height / 2.0
                        local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
                        ct, st = np.cos(angle), np.sin(angle)
                        qpoints = []
                        for lx, ly in local:
                            ix = det.center_x_px + ct * lx - st * ly
                            iy = det.center_y_px + st * lx + ct * ly
                            wx, wy = self.image_to_widget(ix, iy)
                            qpoints.append((wx, wy))

                        fit_pen = QPen(fit_color, 2.8)
                        fit_pen.setCosmetic(True)
                        painter.setPen(fit_pen)
                        for i in range(4):
                            x0, y0 = qpoints[i]
                            x1, y1 = qpoints[(i + 1) % 4]
                            painter.drawLine(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))

                        # Corner handles make it obvious this is the fitted square/rectangle.
                        painter.setPen(QPen(fit_color, 1.5))
                        for wx, wy in qpoints:
                            painter.drawRect(QRectF(wx - 3.5, wy - 3.5, 7.0, 7.0))

                        # Draw the fit parameter label close to the contour.
                        label_x = int(round(min(x for x, _ in qpoints)))
                        label_y = int(round(min(y for _, y in qpoints))) - 6
                        painter.setPen(fit_color)
                        painter.drawText(
                            label_x,
                            label_y,
                            f"矩形 W={width * self.pixel_size_x_um:.3f} μm H={height * self.pixel_size_y_um:.3f} μm 角度={angle_deg:.1f}° 残差={det.residual_um:.3f} μm",
                        )

        if self.show_auto_detections:
            self._draw_auto_detection_results(painter)

        if self.is_dragging and self.drag_start_img is not None and self.drag_current_img is not None:
            preview_roi = self._roi_from_drag(self.drag_start_img, self.drag_current_img)
            if preview_roi is not None:
                self._draw_roi_shape(painter, preview_roi, QColor(120, 255, 120), True, "预览")

        if self.circle_pick_mode and self.circle_pick_points:
            painter.setPen(QPen(QColor(120, 255, 120), 2.0))
            for x, y in self.circle_pick_points:
                wx, wy = self.image_to_widget(x, y)
                painter.drawEllipse(QRectF(wx - 4, wy - 4, 8, 8))
            if len(self.circle_pick_points) == 2:
                x0, y0 = self.image_to_widget(*self.circle_pick_points[0])
                x1, y1 = self.image_to_widget(*self.circle_pick_points[1])
                painter.drawLine(int(x0), int(y0), int(x1), int(y1))
                preview_roi = self._caliper_roi_from_center_circle(
                    self._circle_from_three_points([*self.circle_pick_points, self.circle_preview_point])
                    if self.circle_preview_point is not None
                    else None
                )
                if preview_roi is not None:
                    self._draw_roi_shape(painter, preview_roi, QColor(120, 255, 120), True, "三点预览")
                    mid_radius = 0.5 * (preview_roi.inner_radius() + preview_roi.outer_radius())
                    cx, cy = self.image_to_widget(*preview_roi.center())
                    painter.setPen(QPen(QColor(120, 255, 120), 1.2))
                    painter.drawText(
                        int(cx + 10),
                        int(cy + 22),
                        f"中心半径={mid_radius:.2f} px  半宽={self.active_ring_half_width_px:.2f} px",
                    )

    def _draw_auto_detection_results(self, painter: QPainter):
        label_index = 0
        for label, layer_map in self.auto_detections.items():
            for layer, detection in layer_map.items():
                if self.fixed_layer and layer != self.fixed_layer:
                    continue
                valid = detection.shape_params.get("quality_status", "Valid") == "Valid"
                if not valid:
                    color = QColor(255, 60, 60)
                    role = "无效"
                elif label == self.auto_reference_label:
                    color = QColor(0, 220, 255)
                    role = "基准"
                elif label == self.auto_target_label:
                    color = QColor(255, 210, 0)
                    role = "待测"
                else:
                    color = QColor(0, 255, 90)
                    role = "有效"
                contour_points = detection.shape_params.get("candidate_contour_points", detection.edge_points)
                if self.show_diagnostics and contour_points:
                    widget_points = [
                        QPointF(*self.image_to_widget(float(point[0]), float(point[1])))
                        for point in contour_points
                    ]
                    pen = QPen(QColor(160, 160, 160), 1.0)
                    pen.setCosmetic(True)
                    painter.setPen(pen)
                    painter.drawPolygon(QPolygonF(widget_points))
                cx, cy = self.image_to_widget(detection.center_x_px, detection.center_y_px)
                pen = QPen(color, 2.3)
                pen.setCosmetic(True)
                painter.setPen(pen)
                if detection.fitting_mode == "ProductionCircle":
                    radius = float(detection.shape_params.get("radius_px", detection.diameter_px / 2.0)) * self.scale
                    painter.drawEllipse(QRectF(cx - radius, cy - radius, 2.0 * radius, 2.0 * radius))
                elif detection.fitting_mode == "Ellipse":
                    self._draw_ellipse_fit(painter, detection, color, show_axes=True)
                elif detection.fitting_mode == "ProductionRectangle":
                    width = float(detection.shape_params.get("width_px", detection.diameter_px))
                    height = float(detection.shape_params.get("height_px", detection.diameter_px))
                    angle = np.deg2rad(float(detection.shape_params.get("angle_deg", 0.0)))
                    ct, st = np.cos(angle), np.sin(angle)
                    points = []
                    for lx, ly in ((-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2), (-width / 2, height / 2)):
                        x = detection.center_x_px + ct * lx - st * ly
                        y = detection.center_y_px + st * lx + ct * ly
                        points.append(QPointF(*self.image_to_widget(x, y)))
                    painter.drawPolygon(QPolygonF(points))
                painter.drawLine(int(cx - 6), int(cy), int(cx + 6), int(cy))
                painter.drawLine(int(cx), int(cy - 6), int(cx), int(cy + 6))
                if self._auto_caliper_selected(label, layer, detection):
                    painter.setPen(QPen(QColor(255, 210, 0, 140), 1.0))
                    for window in detection.shape_params.get("caliper_windows", []):
                        length = float(window.get("length", 0.0)) * self.scale
                        if "angle" in window:
                            direction_x = np.cos(float(window["angle"]))
                            direction_y = np.sin(float(window["angle"]))
                        else:
                            direction_x = float(window.get("direction_x", 0.0))
                            direction_y = float(window.get("direction_y", 0.0))
                        x, y = self.image_to_widget(float(window.get("center_x", 0.0)), float(window.get("center_y", 0.0)))
                        painter.drawLine(
                            int(x - direction_x * length / 2.0),
                            int(y - direction_y * length / 2.0),
                            int(x + direction_x * length / 2.0),
                            int(y + direction_y * length / 2.0),
                        )
                if self.show_diagnostics:
                    painter.setPen(QPen(QColor("#34C759"), 1.0))
                    for px, py in detection.edge_points:
                        x, y = self.image_to_widget(px, py)
                        painter.drawEllipse(QRectF(x - 2, y - 2, 4, 4))
                    painter.setPen(QPen(QColor(255, 60, 60), 1.0))
                    for px, py in detection.rejected_points:
                        x, y = self.image_to_widget(px, py)
                        painter.drawLine(int(x - 3), int(y - 3), int(x + 3), int(y + 3))
                        painter.drawLine(int(x - 3), int(y + 3), int(x + 3), int(y - 3))
                suffix = f" {role}" if role else ""
                label_x, label_y = self._contour_label_anchor(detection, label_index)
                label_index += 1
                painter.drawText(
                    int(label_x + 5),
                    int(label_y - 5),
                    f"{candidate_display_label(label, detection)}{suffix}",
                )

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.profile_capture_enabled:
            self.set_profile_capture_enabled(False)
            self.interactionMessage.emit("已退出图像诊断剖面选取。")
            event.accept()
            return
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.space_pan_held = True
            if not self.is_panning:
                self.setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self._cancel_roi_edit():
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self.is_dragging:
            self.is_dragging = False
            self.drag_start_img = None
            self.drag_current_img = None
            self.setCursor(Qt.CrossCursor if self.roi_editing_enabled else Qt.ArrowCursor)
            self.update()
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self.geometry_interaction_active:
            self.geometryCommand.emit("cancel")
            event.accept()
            return
        if event.key() == Qt.Key_Backspace and self.geometry_interaction_active:
            self.geometryCommand.emit("undo")
            event.accept()
            return
        if event.key() == Qt.Key_Escape and (self.active_roi_id or self.selected_caliper_feature is not None):
            self.clear_caliper_selection(update=False)
            if self.active_roi_id:
                self.roiSelectionCleared.emit(self.active_mark_id, self.active_layer)
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.space_pan_held = False
            if not self.is_panning:
                self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def wheelEvent(self, event):
        if self.image is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        self.zoom_by(factor, event.position().toPoint())
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if self.image is not None:
            self.reset_view(update=True)
            event.accept()

    def mousePressEvent(self, event):
        if self.image is None:
            return
        if event.button() == Qt.RightButton:
            geometry_id = self._geometry_hit(event.position().toPoint())
            measurement_id = self._measurement_hit(event.position().toPoint())
            if geometry_id or measurement_id:
                menu = QMenu(self)
                delete_action = menu.addAction("删除此测量/几何要素")
                if menu.exec(event.globalPosition().toPoint()) == delete_action:
                    self.geometryCommand.emit(f"delete:{measurement_id or geometry_id}")
                event.accept()
                return
            if self.roi_editing_enabled and not self.circle_pick_mode and not self.show_auto_detections:
                hit_roi_id = self._manual_roi_hit(event.position().toPoint())
                menu = QMenu(self)
                if hit_roi_id:
                    if hit_roi_id != self.active_roi_id:
                        self.roiSelected.emit(self.active_mark_id, self.active_layer, hit_roi_id)
                    copy_action = menu.addAction("复制此 ROI")
                    delete_action = menu.addAction("删除此 ROI")
                    menu.addSeparator()
                    clear_selection_action = menu.addAction("取消选择")
                else:
                    clear_contours_action = menu.addAction("清除当前层识别轮廓")
                    delete_layer_action = menu.addAction("删除当前层全部 ROI")
                    menu.addSeparator()
                    clear_all_action = menu.addAction("清除所有识别与测量轮廓")
                action = menu.exec(event.globalPosition().toPoint())
                if hit_roi_id and action == copy_action:
                    self.roiContextAction.emit(self.active_mark_id, self.active_layer, hit_roi_id, "copy")
                elif hit_roi_id and action == delete_action:
                    self.roiContextAction.emit(self.active_mark_id, self.active_layer, hit_roi_id, "delete")
                elif hit_roi_id and action == clear_selection_action:
                    self.roiSelectionCleared.emit(self.active_mark_id, self.active_layer)
                elif not hit_roi_id and action == clear_contours_action:
                    self.roiContextAction.emit(self.active_mark_id, self.active_layer, "", "clear_contours")
                elif not hit_roi_id and action == delete_layer_action:
                    self.roiContextAction.emit(self.active_mark_id, self.active_layer, "", "delete_layer")
                elif not hit_roi_id and action == clear_all_action:
                    self.geometryCommand.emit("clear_all")
                event.accept()
                return
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self.space_pan_held):
            self.is_panning = True
            self.pan_start_pos = event.position().toPoint()
            self.pan_start_x = self.pan_x
            self.pan_start_y = self.pan_y
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            if self.profile_capture_enabled:
                point = self.widget_to_image_float(event.position().toPoint())
                if point is not None:
                    self.profile_start_img = point
                    self.profile_current_img = point
                    self.profile_dragging = True
                    self.update()
                event.accept()
                return
            if self.geometry_interaction_active:
                payload = self._geometry_pick_payload(event.position().toPoint())
                if payload is not None:
                    self.selected_geometry_feature_id = payload["feature_id"]
                    self.selected_detection_key = payload["detection_key"]
                    self.geometryClicked.emit(self.active_layer, payload)
                    self.update()
                event.accept()
                return
            if self.show_auto_detections:
                hit = self._nearest_auto_caliper_hit(event.position().toPoint())
                if hit is not None:
                    label, layer, detection = hit
                    self._select_caliper_detection("auto", label, layer, detection)
                elif self.selected_caliper_feature is not None:
                    self.clear_caliper_selection()
                event.accept()
                return
            if self.circle_pick_mode:
                if not self.roi_editing_enabled:
                    event.accept()
                    return
                p = self.widget_to_image_float(event.position().toPoint())
                if p is not None:
                    self.circle_pick_points.append(p)
                    if len(self.circle_pick_points) == 3:
                        roi = self._caliper_roi_from_center_circle(self._circle_from_three_points(self.circle_pick_points))
                        if roi is not None:
                            self.roiChanged.emit(self.active_mark_id, self.active_layer, roi)
                        self.set_circle_pick_mode(False)
                    self.update()
                event.accept()
                return
            manual_hit = self._manual_caliper_hit(event.position().toPoint())
            current_detection = self.roi_detections.get(self.active_mark_id, {}).get(self.active_roi_id)
            manual_selected = self._manual_caliper_selected(
                self.active_mark_id,
                self.active_roi_id,
                self.active_layer,
                current_detection,
            )
            hit_roi_id = self._manual_roi_hit(event.position().toPoint())
            if hit_roi_id and hit_roi_id != self.active_roi_id:
                self.roiSelected.emit(self.active_mark_id, self.active_layer, hit_roi_id)
                if manual_hit is not None and manual_hit[0] == hit_roi_id:
                    self._select_caliper_detection("manual", *manual_hit)
                event.accept()
                return
            if not self.roi_editing_enabled:
                event.accept()
                return
            hit_part = self._roi_hit_part(event.position().toPoint())
            if hit_part:
                self._begin_roi_edit("resize", hit_part, event.position().toPoint())
                event.accept()
                return
            if self._point_in_active_roi_outer(event.position().toPoint()):
                pending = manual_hit if manual_hit is not None and not manual_selected else None
                if self._begin_roi_edit("move", "", event.position().toPoint(), pending):
                    event.accept()
                    return
            p = self.widget_to_image(event.position().toPoint())
            if p is not None:
                self.clear_caliper_selection(update=False)
                if self.active_roi_id:
                    self.roiSelectionCleared.emit(self.active_mark_id, self.active_layer)
                self.drag_start_img = p
                self.drag_current_img = p
                self.is_dragging = True
                self.update()

    def mouseMoveEvent(self, event):
        if self.is_panning and self.pan_start_pos is not None:
            pos = event.position().toPoint()
            self.pan_x = self.pan_start_x + (pos.x() - self.pan_start_pos.x())
            self.pan_y = self.pan_start_y + (pos.y() - self.pan_start_pos.y())
            self.update()
            event.accept()
            return
        if self.circle_pick_mode and len(self.circle_pick_points) == 2:
            p = self.widget_to_image_float(event.position().toPoint())
            if p is not None:
                self.circle_preview_point = p
                self.update()
            event.accept()
            return
        if self.profile_dragging:
            point = self.widget_to_image_float(event.position().toPoint())
            if point is not None:
                self.profile_current_img = point
                self.update()
            event.accept()
            return
        if self.geometry_interaction_active:
            payload = self._geometry_pick_payload(event.position().toPoint())
            self.geometry_hover_point = payload["point_px"] if payload is not None else None
            self.hovered_geometry_feature_id = payload["feature_id"] if payload is not None else ""
            self.hovered_detection_key = payload["detection_key"] if payload is not None else ""
            snapped = bool(payload and (payload["feature_id"] or payload["detection_key"]))
            self.setCursor(Qt.PointingHandCursor if snapped else Qt.CrossCursor)
            self.update()
            event.accept()
            return
        if self.is_dragging and self.image is not None:
            p = self.widget_to_image(event.position().toPoint())
            if p is not None:
                self.drag_current_img = p
                self.update()
            return
        if self.is_adjusting_roi and self.image is not None:
            if self._roi_edit_distance(event.position()) >= 3.0:
                self.roi_edit_moved = True
                self._adjust_active_roi(event.position().toPoint())
            self.update()
            event.accept()
            return
        if self.is_moving_roi and self.image is not None:
            if self._roi_edit_distance(event.position()) >= 3.0:
                self.roi_edit_moved = True
                self._move_active_roi(event.position().toPoint())
            self.update()
            event.accept()
            return
        self._update_edge_tooltip(event.position().toPoint())
        self._update_roi_hover_cursor(event.position().toPoint())

    def _update_edge_tooltip(self, pos):
        if self.image is None:
            return
        best = None
        best_dist = 7.0
        result_maps = [self.auto_detections] if self.show_auto_detections else [self.detections]
        for result_map in result_maps:
            for mark_id, layer_map in result_map.items():
                if not self.show_auto_detections and mark_id != self.active_mark_id:
                    continue
                for layer, det in layer_map.items():
                    if self.fixed_layer and layer != self.fixed_layer:
                        continue
                    if not self.show_auto_detections and not self.fixed_layer and layer != self.active_layer:
                        continue
                    point_groups = (
                        (det.edge_points, getattr(det, "edge_gradients", []), "参与拟合"),
                        (getattr(det, "rejected_points", []), getattr(det, "rejected_gradients", []), "已剔除"),
                    )
                    for points, gradients, state in point_groups:
                        for idx, (px, py) in enumerate(points):
                            wx, wy = self.image_to_widget(px, py)
                            dist = float(np.hypot(wx - pos.x(), wy - pos.y()))
                            if dist < best_dist:
                                grad_txt = f"{gradients[idx]:.3f}" if idx < len(gradients) else "-"
                                best = (
                                    f"{mark_id} {LAYER_LABELS.get(layer, layer)} {state}\n"
                                    f"坐标=({px * self.pixel_size_x_um:.3f}, {py * self.pixel_size_y_um:.3f}) μm\n梯度={grad_txt}"
                                )
                                best_dist = dist
        self.setToolTip(best or "")

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.profile_dragging:
            point = self.widget_to_image_float(event.position().toPoint())
            if point is not None:
                self.profile_current_img = point
            self.profile_dragging = False
            self._emit_profile()
            self.update()
            event.accept()
            return
        if event.button() in (Qt.LeftButton, Qt.MiddleButton) and self.is_panning:
            self.is_panning = False
            self.pan_start_pos = None
            self.setCursor(Qt.CrossCursor if self.circle_pick_mode else Qt.ArrowCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.is_moving_roi:
            if self.roi_edit_moved:
                self._move_active_roi(event.position().toPoint())
                self.roiEditCommitted.emit(
                    self.active_mark_id, self.active_layer, self.roi_edit_id, deepcopy(self.roi_edit_preview)
                )
            elif self.pending_caliper_hit is not None:
                self._select_caliper_detection("manual", *self.pending_caliper_hit)
            self._clear_roi_edit_state()
            self.setCursor(Qt.ArrowCursor)
            self.update()
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.is_adjusting_roi:
            if self.roi_edit_moved:
                self._adjust_active_roi(event.position().toPoint())
                self.roiEditCommitted.emit(
                    self.active_mark_id, self.active_layer, self.roi_edit_id, deepcopy(self.roi_edit_preview)
                )
            self._clear_roi_edit_state()
            self.setCursor(Qt.ArrowCursor)
            self.update()
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.is_dragging and self.drag_start_img is not None:
            p = self.widget_to_image(event.position().toPoint())
            if p is None:
                p = self.drag_current_img
            self.is_dragging = False
            if p is not None:
                roi = self._roi_from_drag(self.drag_start_img, p)
                if roi is not None:
                    self.roiChanged.emit(self.active_mark_id, self.active_layer, roi)
                elif self.drag_start_img != p:
                    self.interactionMessage.emit("ROI 尺寸过小，请拖动更大的区域后重试。")
            self.drag_start_img = None
            self.drag_current_img = None
            self.update()


class RepeatabilityPlot(QWidget):
    """Lightweight repeatability trend plot without extra plotting dependencies."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.series = {}
        self.setMinimumHeight(150)
        self.setStyleSheet("QWidget { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; }")

    def set_series(self, series: dict):
        self.series = series or {}
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))
        margin_l, margin_r, margin_t, margin_b = 46, 16, 18, 30
        rect = self.rect().adjusted(margin_l, margin_t, -margin_r, -margin_b)
        painter.setPen(QPen(QColor("#DADDE3"), 1))
        painter.drawRect(rect)
        if not self.series:
            painter.setPen(QColor("#6E6E73"))
            painter.drawText(self.rect(), Qt.AlignCenter, "暂无重复性数据")
            painter.end()
            return
        values = []
        max_len = 0
        for vals in self.series.values():
            values.extend([float(v) for v in vals])
            max_len = max(max_len, len(vals))
        if not values or max_len <= 0:
            painter.end(); return
        vmin, vmax = min(values), max(values)
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1.0
            vmin = vmin - 1.0
        painter.setPen(QColor("#6E6E73"))
        painter.drawText(8, rect.top() + 10, f"{vmax:.3f}")
        painter.drawText(8, rect.bottom(), f"{vmin:.3f}")
        palette = [QColor("#007AFF"), QColor("#FF9500"), QColor("#34C759"), QColor("#AF52DE")]
        legend_x = rect.left() + 4
        for idx, (name, vals) in enumerate(self.series.items()):
            color = palette[idx % len(palette)]
            painter.setPen(QPen(color, 2.0))
            points = []
            for i, val in enumerate(vals):
                x = rect.left() + (rect.width() * i / max(1, max_len - 1))
                y = rect.bottom() - (rect.height() * (float(val) - vmin) / (vmax - vmin))
                points.append(QPointF(x, y))
            for a, b in zip(points, points[1:]):
                painter.drawLine(a, b)
            for pt in points:
                painter.drawEllipse(QRectF(pt.x() - 2.5, pt.y() - 2.5, 5, 5))
            painter.drawText(legend_x, self.rect().bottom() - 8 - 16 * idx, name)
        painter.setPen(QColor("#6E6E73"))
        painter.drawText(rect.center().x() - 40, self.rect().bottom() - 8, "测量次数")
        painter.end()
