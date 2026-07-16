from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QPoint, QPointF, QRectF, QThread, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor, QFont, QFontDatabase, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .auto_mark_detector import detect_auto_marks_with_report
from .export_naming import build_export_filename
from .image_loader import display_to_uint8, load_image
from .measurement_engine import run_measurement_job
from .measurement_service import attach_algorithm_path, describe_algorithm_path, detect_manual_roi
from .measurement_units import axis_scale_um_per_px, rotated_rect_size_um
from .models import DetectionParams, DetectionResult, ImageData, MarkRecipe, MeasurementConfig, OverlayResult, Roi
from .overlay_calculator import calculate_overlay, calculate_relative_overlay
from .production_measurement import refine_candidate
from .recipe_manager import load_recipe, save_recipe
from .result_exporter import build_detection_rows, export_results
from .rz_calculator import build_summary_rows


LAYER_LABELS = {"upper": "上层", "lower": "下层"}
STEP_TITLES = ["产品与设备信息", "图像导入", "ROI 设置", "算法参数", "结果导出"]


class SidebarComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class SidebarDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class SidebarSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()




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


class ImageCanvas(QLabel):
    roiChanged = Signal(str, str, object)  # mark_id, layer, Roi

    def __init__(self, title: str, fixed_layer: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.title = title
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
        self.active_roi_type = "Annulus"
        self.active_roi_inner_ratio = 0.60
        self.active_roi_target_edge = "All Edges"
        self.active_roi_angle_deg = 0.0
        self.active_ring_half_width_px = 10.0
        self.active_caliper_count = 64
        self.active_caliper_width_px = 8.0
        self.active_search_direction = "Inner to Outer"
        self.circle_pick_mode = False
        self.circle_pick_points = []
        self.circle_preview_point = None
        self.marks: Dict[str, MarkRecipe] = {}
        self.detections: Dict[str, Dict[str, DetectionResult]] = {}
        self.auto_detections: Dict[str, Dict[str, DetectionResult]] = {}
        self.show_auto_detections = False
        self.manual_labels = {}
        self.auto_reference_label = ""
        self.auto_target_label = ""
        self.show_diagnostics = False
        self.display_enhancement = False
        self.pixel_size_x_um = 0.1
        self.pixel_size_y_um = 0.1
        self.drag_start_img: Optional[QPoint] = None
        self.drag_current_img: Optional[QPoint] = None
        self.is_dragging = False
        self.is_adjusting_roi = False
        self.is_moving_roi = False
        self.adjust_roi_part = ""
        self.adjust_mark_id = ""
        self.adjust_layer = ""
        self.move_start_img = None
        self.move_start_roi = None
        self.is_panning = False
        self.pan_start_pos: Optional[QPoint] = None
        self.pan_start_x = 0.0
        self.pan_start_y = 0.0
        self.setText("等待导入图像")
        self.setStyleSheet("QLabel { background: #2B2F36; color: #F5F6F8; border: 1px solid #3A404A; border-radius: 12px; }")

    def set_image(self, image: Optional[ImageData]):
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
        auto_detections=None,
        show_auto_detections: bool = False,
        manual_labels=None,
        auto_reference_label: str = "",
        auto_target_label: str = "",
        pixel_size_x_um: float = 0.1,
        pixel_size_y_um: float = 0.1,
        show_diagnostics: bool = False,
    ):
        self.active_mark_id = active_mark_id
        self.active_layer = self.fixed_layer or active_layer
        self.active_roi_type = roi_type
        self.active_roi_inner_ratio = float(roi_inner_ratio)
        self.active_roi_target_edge = roi_target_edge
        self.active_roi_angle_deg = float(roi_angle_deg)
        self.active_ring_half_width_px = float(max(1.0, roi_ring_half_width_px))
        self.active_caliper_count = int(roi_caliper_count)
        self.active_caliper_width_px = float(roi_caliper_width_px)
        self.active_search_direction = roi_search_direction
        self.marks = marks
        self.detections = detections
        self.auto_detections = auto_detections or {}
        self.show_auto_detections = bool(show_auto_detections)
        self.manual_labels = manual_labels or {}
        self.auto_reference_label = auto_reference_label
        self.auto_target_label = auto_target_label
        self.pixel_size_x_um = float(pixel_size_x_um)
        self.pixel_size_y_um = float(pixel_size_y_um)
        self.show_diagnostics = bool(show_diagnostics)
        self.update()

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
        return self.offset_x + x * self.scale, self.offset_y + y * self.scale

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

    def _draw_roi_shape(self, painter: QPainter, roi: Roi, color: QColor, active: bool, label: str = ""):
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
        elif typ in {"Annulus", "Caliper Circle"}:
            outer = r.outer_radius() * self.scale
            inner = r.inner_radius() * self.scale
            if typ == "Caliper Circle":
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
            if typ == "Caliper Circle":
                mid = 0.5 * (outer + inner)
                middle_pen = QPen(QColor(255, 220, 40), 1.3)
                middle_pen.setStyle(Qt.DashLine)
                middle_pen.setCosmetic(True)
                painter.setPen(middle_pen)
                painter.drawEllipse(QRectF(wcx - mid, wcy - mid, 2 * mid, 2 * mid))
                painter.setPen(pen)
                self._draw_calipers(painter, r, color)
        elif typ == "Rectangular Ring":
            outer_poly = QPolygonF(self._rotated_rect_points_widget(cx, cy, r.w, r.h, r.angle_deg))
            iw, ih = r.inner_size()
            inner_poly = QPolygonF(self._rotated_rect_points_widget(cx, cy, iw, ih, r.angle_deg))
            painter.drawPolygon(outer_poly)
            inner_pen = QPen(color, 1.8 if active else 1.2)
            inner_pen.setStyle(Qt.DotLine)
            inner_pen.setCosmetic(True)
            painter.setPen(inner_pen)
            painter.drawPolygon(inner_poly)
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
            typ_label = {"Annulus": "圆环", "Caliper Circle": "卡尺圆", "Rectangular Ring": "矩形环", "Circle": "圆", "Rectangle": "矩形"}.get(typ, typ)
            painter.drawText(int(x + 4), int(y + 16), f"{label} [{typ_label}]")

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
        x = (pos.x() - self.offset_x) / self.scale
        y = (pos.y() - self.offset_y) / self.scale
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return float(x), float(y)

    def widget_to_image(self, pos) -> Optional[QPoint]:
        p = self.widget_to_image_float(pos)
        if p is None:
            return None
        x, y = p
        return QPoint(int(round(x)), int(round(y)))

    def _active_roi(self) -> Optional[Roi]:
        mark = self.marks.get(self.active_mark_id)
        if mark is None:
            return None
        return mark.upper_roi if self.active_layer == "upper" else mark.lower_roi

    def _roi_hit_part(self, pos) -> str:
        roi = self._active_roi()
        p = self.widget_to_image_float(pos)
        if roi is None or p is None:
            return ""
        r = roi.normalized()
        x, y = p
        tol = max(4.0 / max(self.scale, 1e-9), 2.0)
        typ = getattr(r, "roi_type", "Annulus")

        if typ in {"Annulus", "Caliper Circle"}:
            cx, cy = r.center()
            dist = float(np.hypot(x - cx, y - cy))
            inner = r.inner_radius()
            outer = r.outer_radius()
            if abs(dist - inner) <= tol:
                return "inner"
            if abs(dist - outer) <= tol:
                return "outer"
            return ""

        if typ == "Rectangular Ring":
            xs = np.array([x], dtype=np.float64)
            ys = np.array([y], dtype=np.float64)
            xr, yr = r._local_rotated(xs, ys)
            ax, ay = abs(float(xr[0])), abs(float(yr[0]))
            ow, oh = max(r.w, 1e-9), max(r.h, 1e-9)
            iw, ih = r.inner_size()
            inner_dist = min(abs(ax - iw / 2.0), abs(ay - ih / 2.0))
            outer_dist = min(abs(ax - ow / 2.0), abs(ay - oh / 2.0))
            if ax <= ow / 2.0 + tol and ay <= oh / 2.0 + tol:
                if inner_dist <= tol and ax <= iw / 2.0 + tol and ay <= ih / 2.0 + tol:
                    return "inner"
                if outer_dist <= tol:
                    return "outer"
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
        ).normalized()

    def _move_active_roi(self, pos):
        if not self.is_moving_roi or self.move_start_img is None or self.move_start_roi is None:
            return
        p = self.widget_to_image_float(pos)
        if p is None:
            return
        dx = p[0] - self.move_start_img[0]
        dy = p[1] - self.move_start_img[1]
        roi = replace(self.move_start_roi, x=self.move_start_roi.x + dx, y=self.move_start_roi.y + dy)
        self.roiChanged.emit(self.active_mark_id, self.active_layer, roi.normalized())

    def _adjust_active_roi(self, pos):
        mark = self.marks.get(self.adjust_mark_id)
        if mark is None:
            return
        roi = mark.upper_roi if self.adjust_layer == "upper" else mark.lower_roi
        p = self.widget_to_image_float(pos)
        if roi is None or p is None:
            return
        r = roi.normalized()
        x, y = p
        typ = getattr(r, "roi_type", "Annulus")
        min_outer = 5.0
        min_width = 2.0

        if typ in {"Annulus", "Caliper Circle"}:
            cx, cy = r.center()
            dist = max(min_outer, float(np.hypot(x - cx, y - cy)))
            outer = r.outer_radius()
            inner = r.inner_radius()
            if self.adjust_roi_part == "inner":
                new_inner = float(np.clip(dist, min_width, max(min_width, outer - min_width)))
                roi.inner_ratio = new_inner / max(outer, 1e-9)
            elif self.adjust_roi_part == "outer":
                new_outer = max(dist, inner + min_width, min_outer)
                roi.x = cx - new_outer
                roi.y = cy - new_outer
                roi.w = new_outer * 2.0
                roi.h = new_outer * 2.0
                roi.inner_ratio = float(np.clip(inner / max(new_outer, 1e-9), 0.0, 0.98))

        elif typ == "Rectangular Ring":
            xs = np.array([x], dtype=np.float64)
            ys = np.array([y], dtype=np.float64)
            xr, yr = r._local_rotated(xs, ys)
            ax, ay = abs(float(xr[0])), abs(float(yr[0]))
            if self.adjust_roi_part == "inner":
                ratio = max(ax / max(r.w / 2.0, 1e-9), ay / max(r.h / 2.0, 1e-9))
                roi.inner_ratio = float(np.clip(ratio, 0.02, 0.98))
            elif self.adjust_roi_part == "outer":
                cx, cy = r.center()
                scale = max(ax / max(r.w / 2.0, 1e-9), ay / max(r.h / 2.0, 1e-9), min_outer / max(min(r.w, r.h), 1e-9))
                new_w = max(min_outer, r.w * scale)
                new_h = max(min_outer, r.h * scale)
                inner_w, inner_h = r.inner_size()
                roi.x = cx - new_w / 2.0
                roi.y = cy - new_h / 2.0
                roi.w = new_w
                roi.h = new_h
                roi.inner_ratio = float(np.clip(max(inner_w / new_w, inner_h / new_h), 0.0, 0.98))

        self.roiChanged.emit(self.adjust_mark_id, self.adjust_layer, roi.normalized())

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
                (center_pos.x() - self.offset_x) / max(self.scale, 1e-12),
                (center_pos.y() - self.offset_y) / max(self.scale, 1e-12),
            )
        self.user_zoom = float(np.clip(self.user_zoom * factor, 0.05, 80.0))
        new_scale = self.fit_scale * self.user_zoom
        base_x, base_y = self._base_offset_for_scale(new_scale)
        img_x, img_y = before
        self.pan_x = center_pos.x() - img_x * new_scale - base_x
        self.pan_y = center_pos.y() - img_y * new_scale - base_y
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#2B2F36"))

        if self.pixmap_cache is None:
            painter.setPen(QColor("#F5F6F8"))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待导入图像")
            painter.end()
            return

        self._update_transform()
        target = QRectF(self.offset_x, self.offset_y, self.pixmap_cache.width() * self.scale, self.pixmap_cache.height() * self.scale)
        painter.drawPixmap(target, self.pixmap_cache, QRectF(self.pixmap_cache.rect()))

        self._draw_overlays(painter)

        painter.setPen(QColor("#F5F6F8"))
        painter.drawText(10, 20, f"缩放: {self.user_zoom:.2f}x")
        painter.drawText(10, 40, "滚轮缩放；右键/中键拖动画面；双击复位；左键拖 ROI，拖内/外环调范围")
        self._draw_scale_and_axes(painter)
        painter.end()

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
                roi = mark.upper_roi if layer == "upper" else mark.lower_roi
                if roi is not None:
                    is_active = (mark_id == self.active_mark_id and layer == self.active_layer)
                    self._draw_roi_shape(painter, roi, colors[layer], is_active, f"{mark_id} {LAYER_LABELS[layer]}")

                det = self.detections.get(mark_id, {}).get(layer)
                if det is not None:
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
                    contour_label = self.manual_labels.get((mark_id, layer), "")
                    if contour_label:
                        label_x, label_y = self._contour_label_anchor(det)
                        painter.drawText(int(label_x + 6), int(label_y - 6), f"{contour_label} ({mark_id})")
                    if det.fitting_mode in {"Circle", "EdgeCenter", "CaliperCircle"} and "radius_px" in det.shape_params:
                        rad = det.shape_params["radius_px"] * self.scale
                        painter.setPen(QPen(fit_color, 2.0))
                        painter.drawEllipse(QRectF(cx - rad, cy - rad, 2 * rad, 2 * rad))
                        if det.fitting_mode == "CaliperCircle":
                            radius_um = det.shape_params.get("radius_px", 0) * self._mean_pixel_size_um()
                            painter.setPen(fit_color)
                            painter.drawText(
                                int(cx + 12),
                                int(cy - 12),
                                f"中心=({det.center_x_um:.3f},{det.center_y_um:.3f}) μm 半径={radius_um:.3f} μm 残差={det.residual_um:.3f} μm 置信度={det.confidence:.3f}",
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
                        major = det.shape_params.get("major_px", det.diameter_px) * self.scale
                        minor = det.shape_params.get("minor_px", det.diameter_px) * self.scale
                        # For V1 display, draw axis-aligned ellipse; angle is reported numerically in table.
                        painter.setPen(QPen(fit_color, 2.0))
                        painter.drawEllipse(QRectF(cx - major / 2, cy - minor / 2, major, minor))
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
            preview_roi = Roi(
                float(self.drag_start_img.x()),
                float(self.drag_start_img.y()),
                float(self.drag_current_img.x() - self.drag_start_img.x()),
                float(self.drag_current_img.y() - self.drag_start_img.y()),
                self.active_roi_type,
                self.active_roi_inner_ratio,
                self.active_roi_target_edge,
                self.active_roi_angle_deg,
            ).normalized()
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
                if self.show_diagnostics:
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
                    f"{label}{suffix}",
                )

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
            if not self.show_auto_detections and self._point_in_active_roi_outer(event.position().toPoint()):
                menu = QMenu(self)
                delete_action = menu.addAction("删除当前 ROI")
                action = menu.exec(event.globalPosition().toPoint())
                if action == delete_action:
                    self.roiChanged.emit(self.active_mark_id, self.active_layer, None)
                event.accept()
                return
            self.is_panning = True
            self.pan_start_pos = event.position().toPoint()
            self.pan_start_x = self.pan_x
            self.pan_start_y = self.pan_y
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.pan_start_pos = event.position().toPoint()
            self.pan_start_x = self.pan_x
            self.pan_start_y = self.pan_y
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            if self.show_auto_detections:
                event.accept()
                return
            if self.circle_pick_mode:
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
            hit_part = self._roi_hit_part(event.position().toPoint())
            if hit_part:
                self.is_adjusting_roi = True
                self.adjust_roi_part = hit_part
                self.adjust_mark_id = self.active_mark_id
                self.adjust_layer = self.active_layer
                self.setCursor(Qt.SizeAllCursor)
                event.accept()
                return
            if self._point_in_active_roi_outer(event.position().toPoint()):
                roi = self._active_roi()
                p = self.widget_to_image_float(event.position().toPoint())
                if roi is not None and p is not None:
                    self.is_moving_roi = True
                    self.move_start_img = p
                    self.move_start_roi = roi.normalized()
                    self.setCursor(Qt.SizeAllCursor)
                    event.accept()
                    return
            p = self.widget_to_image(event.position().toPoint())
            if p is not None:
                self.drag_start_img = p
                self.drag_current_img = p
                self.is_dragging = True
                self.update()

    def mouseMoveEvent(self, event):
        if self.circle_pick_mode and len(self.circle_pick_points) == 2:
            p = self.widget_to_image_float(event.position().toPoint())
            if p is not None:
                self.circle_preview_point = p
                self.update()
            event.accept()
            return
        if self.is_panning and self.pan_start_pos is not None:
            pos = event.position().toPoint()
            self.pan_x = self.pan_start_x + (pos.x() - self.pan_start_pos.x())
            self.pan_y = self.pan_start_y + (pos.y() - self.pan_start_pos.y())
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
            self._adjust_active_roi(event.position().toPoint())
            self.update()
            event.accept()
            return
        if self.is_moving_roi and self.image is not None:
            self._move_active_roi(event.position().toPoint())
            self.update()
            event.accept()
            return
        self._update_edge_tooltip(event.position().toPoint())

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
        if event.button() in (Qt.RightButton, Qt.MiddleButton) and self.is_panning:
            self.is_panning = False
            self.pan_start_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.is_moving_roi:
            self._move_active_roi(event.position().toPoint())
            self.is_moving_roi = False
            self.move_start_img = None
            self.move_start_roi = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.is_adjusting_roi:
            self._adjust_active_roi(event.position().toPoint())
            self.is_adjusting_roi = False
            self.adjust_roi_part = ""
            self.adjust_mark_id = ""
            self.adjust_layer = ""
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
                x0, y0 = self.drag_start_img.x(), self.drag_start_img.y()
                x1, y1 = p.x(), p.y()
                if abs(x1 - x0) >= 5 and abs(y1 - y0) >= 5:
                    roi_type = self.active_roi_type
                    w = float(x1 - x0)
                    h = float(y1 - y0)
                    if roi_type == "Caliper Circle":
                        side = min(abs(w), abs(h))
                        w = side if w >= 0 else -side
                        h = side if h >= 0 else -side
                    roi = Roi(
                        float(x0),
                        float(y0),
                        w,
                        h,
                        roi_type,
                        self.active_roi_inner_ratio,
                        self.active_roi_target_edge,
                        self.active_roi_angle_deg,
                    ).normalized()
                    self.roiChanged.emit(self.active_mark_id, self.active_layer, roi)
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


class MeasurementWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, job: dict):
        super().__init__()
        self.job = job
        self._cancel_requested = False

    @Slot()
    def run(self):
        try:
            result = run_measurement_job(
                self.job,
                lambda done, total, text: self.progress.emit(done, total, text),
                lambda: self._cancel_requested,
            )
            if self._cancel_requested:
                self.cancelled.emit()
            else:
                self.finished.emit(result)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot()
    def cancel(self):
        self._cancel_requested = True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        for font_path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
            if font_path.exists() and QFontDatabase.addApplicationFont(str(font_path)) >= 0:
                break
        self.setFont(QFont("Microsoft YaHei UI", 9))
        self.setWindowTitle("对位偏差测量软件 V1.5.4")
        self.resize(1500, 920)

        self.config = MeasurementConfig()
        self.params = DetectionParams()
        self.upper_image: Optional[ImageData] = None
        self.lower_image: Optional[ImageData] = None
        self.marks: Dict[str, MarkRecipe] = {"Mark1": MarkRecipe("Mark1"), "Mark2": MarkRecipe("Mark2")}
        self.mark_images: Dict[str, Dict[str, Optional[ImageData]]] = {
            "Mark1": {"upper": None, "lower": None},
            "Mark2": {"upper": None, "lower": None},
        }
        self.detections: Dict[str, Dict[str, DetectionResult]] = {}
        self.overlays = {}
        self.auto_detections_by_mark: Dict[str, Dict[str, Dict[str, DetectionResult]]] = {
            "Mark1": {},
            "Mark2": {},
        }
        self.auto_candidates_by_mark: Dict[str, Dict[str, Dict[str, DetectionResult]]] = {
            "Mark1": {},
            "Mark2": {},
        }
        self.auto_selections = {
            "Mark1": {"reference_label": "", "target_label": ""},
            "Mark2": {"reference_label": "", "target_label": ""},
        }
        self.auto_overlays = {}
        # V1.3: batch measurement data. Each mark can contain multiple static repeats.
        self.batch_images: Dict[str, Dict[str, list[ImageData]]] = {
            "Mark1": {"upper": [], "lower": []},
            "Mark2": {"upper": [], "lower": []},
        }
        self.batch_overlays: Dict[str, list[OverlayResult]] = {"Mark1": [], "Mark2": []}
        self.batch_run_records: Dict[str, list[dict]] = {"Mark1": [], "Mark2": []}
        self.roi_sources = self._empty_roi_sources()
        self.loaded_recipe_path = ""
        self.loaded_recipe_display_name = ""
        self._recipe_roi_confirmation_signature = None
        self._calculation_thread: Optional[QThread] = None
        self._calculation_worker: Optional[MeasurementWorker] = None
        self._calculation_running = False

        self._apply_window_style()
        self._build_ui()
        self._connect_actions()
        self._refresh_all_widgets()

    @staticmethod
    def _empty_roi_sources() -> dict:
        return {
            "Mark1": {"upper": "none", "lower": "none"},
            "Mark2": {"upper": "none", "lower": "none"},
        }

    def _roi_source(self, mark_id: Optional[str] = None, layer: Optional[str] = None) -> str:
        mark_id = mark_id or self._current_mark_id()
        layer = layer or self._current_layer()
        return self.roi_sources.get(mark_id, {}).get(layer, "none")

    @staticmethod
    def _roi_source_text(source: str) -> str:
        return {"recipe": "配方 ROI", "manual": "本次手动 ROI", "none": "未设置"}.get(source, "未设置")


    def _apply_window_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #F5F6F8; color: #1D1D1F; }
            QLabel#titleLabel { font-size: 22px; font-weight: 700; color: #1D1D1F; }
            QLabel#versionLabel, QLabel#recipeLabel, QLabel#statusCaption, QLabel#stepNote { color: #6E6E73; }
            QLabel#imageCardTitleUpper { color: #007AFF; font-size: 15px; font-weight: 700; }
            QLabel#imageCardTitleLower { color: #FF9500; font-size: 15px; font-weight: 700; }
            QLabel#resultTitle { font-size: 12px; color: #6E6E73; }
            QLabel#resultValue { font-size: 24px; font-weight: 700; color: #1D1D1F; }
            QLabel#resultUnit { font-size: 12px; color: #6E6E73; }
            QFrame#toolbarCard, QFrame#stepCard, QFrame#resultCard, QFrame#summaryCard, QFrame#imageCard { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px; }
            QGroupBox, QTableWidget, QPlainTextEdit { background: #FFFFFF; border: 1px solid #DADDE3; border-radius: 10px; margin-top: 8px; padding-top: 8px; }
            QPlainTextEdit { padding: 8px; color: #1D1D1F; font-family: "Microsoft YaHei UI"; font-size: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #1D1D1F; font-weight: 600; }
            QPushButton { background: #FFFFFF; border: 1px solid #DADDE3; border-radius: 9px; padding: 7px 12px; min-height: 20px; }
            QPushButton:hover { background: #FAFAFB; border-color: #BFC5CF; }
            QPushButton#primaryButton { background: #007AFF; color: #FFFFFF; border-color: #007AFF; font-weight: 600; }
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox { background: #FFFFFF; border: 1px solid #DADDE3; border-radius: 8px; padding: 5px 7px; min-height: 18px; }
            QTabWidget::pane { border: 1px solid #E5E7EB; border-radius: 12px; background: #FFFFFF; }
            QTabBar::tab { background: #F5F6F8; border: 1px solid #E5E7EB; padding: 8px 12px; margin-right: 2px; border-top-left-radius: 8px; border-top-right-radius: 8px; }
            QTabBar::tab:selected { background: #FFFFFF; color: #007AFF; font-weight: 600; }
            QToolButton#sectionToggle { text-align: left; font-weight: 600; padding: 10px 12px; background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; }
        """)

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)
        self.setCentralWidget(central)

        toolbar_card = QFrame()
        toolbar_card.setObjectName("toolbarCard")
        toolbar = QHBoxLayout(toolbar_card)
        toolbar.setContentsMargins(16, 12, 16, 12)
        toolbar.setSpacing(10)
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        self.title_label = QLabel("对位偏差测量软件")
        self.title_label.setObjectName("titleLabel")
        self.version_label = QLabel("V1.5.4")
        self.version_label.setObjectName("versionLabel")
        title_row.addWidget(self.title_label)
        title_row.addWidget(self.version_label)
        title_row.addStretch(1)
        title_col.addLayout(title_row)

        self.import_upper_btn = QPushButton("📂 导入上层/单图")
        self.import_lower_btn = QPushButton("📁 导入下层图像")
        self.load_recipe_btn = QPushButton("🧾 加载配方")
        self.save_recipe_btn = QPushButton("💾 保存配方")
        self.analyze_all_btn = QPushButton("▶ 计算对位偏差")
        self.export_btn = QPushButton("📤 导出结果")
        self.analyze_all_btn.setObjectName("primaryButton")
        toolbar.addLayout(title_col, stretch=1)
        for btn in (self.import_upper_btn, self.import_lower_btn, self.load_recipe_btn, self.save_recipe_btn, self.analyze_all_btn, self.export_btn):
            toolbar.addWidget(btn)
        root.addWidget(toolbar_card)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        root.addWidget(main_splitter, stretch=3)
        main_splitter.addWidget(self._build_step_panel())

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(10)

        image_toolbar = QFrame()
        image_toolbar.setObjectName("toolbarCard")
        image_toolbar_layout = QHBoxLayout(image_toolbar)
        image_toolbar_layout.setContentsMargins(12, 10, 12, 10)
        image_toolbar_layout.setSpacing(8)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["单图模式", "双图模式"])
        self.display_enhance_check = QCheckBox("显示增强")
        self.display_enhance_check.setChecked(False)
        self.reset_measurement_btn = QPushButton("重置")
        self.zoom_in_btn = QPushButton("放大")
        self.zoom_out_btn = QPushButton("缩小")
        self.reset_view_btn = QPushButton("复位")
        self.analyze_roi_btn = QPushButton("分析 ROI 区域")
        self.analyze_current_btn = QPushButton("计算当前对位")
        self.analyze_current_btn.setVisible(False)
        self.image_status_label = QLabel("等待导入图像")
        self.image_status_label.setObjectName("statusCaption")
        image_toolbar_layout.addWidget(QLabel("图像模式"))
        image_toolbar_layout.addWidget(self.mode_combo)
        image_toolbar_layout.addWidget(self.display_enhance_check)
        image_toolbar_layout.addSpacing(8)
        for btn in (self.reset_measurement_btn, self.zoom_in_btn, self.zoom_out_btn, self.reset_view_btn):
            image_toolbar_layout.addWidget(btn)
        image_toolbar_layout.addStretch(1)
        image_toolbar_layout.addWidget(self.analyze_roi_btn)
        image_toolbar_layout.addWidget(self.image_status_label)
        center_layout.addWidget(image_toolbar)

        image_row = QHBoxLayout()
        image_row.setSpacing(12)
        self.upper_canvas = ImageCanvas("上层图像 / 单图", fixed_layer=None)
        self.lower_canvas = ImageCanvas("下层图像", fixed_layer="lower")
        self.upper_image_card = self._build_image_card("上层图像 / 单图", "upper", self.upper_canvas)
        self.lower_image_card = self._build_image_card("下层图像", "lower", self.lower_canvas)
        image_row.addWidget(self.upper_image_card, stretch=1)
        image_row.addWidget(self.lower_image_card, stretch=1)
        center_layout.addLayout(image_row, stretch=4)

        center_layout.addWidget(self._build_summary_panel(), stretch=0)

        result_card = QFrame()
        result_card.setObjectName("tableCard")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(10, 10, 10, 10)
        result_layout.setSpacing(4)
        self.result_tabs = QTabWidget()

        detail_tab = QWidget()
        detail_tab_layout = QVBoxLayout(detail_tab)
        detail_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.det_table = QTableWidget()
        self.det_table.setMinimumHeight(300)
        detail_tab_layout.addWidget(self.det_table)

        overlay_tab = QWidget()
        overlay_tab_layout = QVBoxLayout(overlay_tab)
        overlay_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.overlay_table = QTableWidget()
        self.overlay_table.setMinimumHeight(300)
        overlay_tab_layout.addWidget(self.overlay_table)

        repeat_tab = QWidget()
        repeat_layout = QVBoxLayout(repeat_tab)
        repeat_layout.setContentsMargins(0, 0, 0, 0)
        repeat_layout.setSpacing(8)
        self.repeat_table = QTableWidget()
        self.repeat_table.setMinimumHeight(300)
        repeat_layout.addWidget(self.repeat_table, stretch=1)

        self.result_tabs.addTab(detail_tab, "识别明细")
        self.result_tabs.addTab(overlay_tab, "对位结果")
        self.result_tabs.addTab(repeat_tab, "重复性分析")
        self.result_tabs.setMinimumHeight(340)
        result_layout.addWidget(self.result_tabs, stretch=1)
        center_layout.addWidget(result_card, stretch=4)
        main_splitter.addWidget(center)

        self.side_tabs = QTabWidget()
        for page, title in (
            (self._build_product_tab(), "① 产品信息"),
            (self._build_image_tab(), "② 图像导入"),
            (self._build_roi_tab(), "③ ROI 设置"),
            (self._build_algo_tab(), "④ 算法参数"),
            (self._build_spec_tab(), "⑤ 结果导出"),
        ):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.side_tabs.addTab(scroll, title)
        main_splitter.addWidget(self.side_tabs)
        main_splitter.setSizes([170, 1080, 350])
        self._install_progress_status_widgets()
        self._install_algorithm_path_status_button()

    def _install_progress_status_widgets(self):
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(180)
        self.progress_bar.setVisible(False)
        self.progress_stage_label = QLabel("")
        self.progress_stage_label.setObjectName("statusCaption")
        self.progress_stage_label.setVisible(False)
        self.cancel_progress_btn = QPushButton("取消计算")
        self.cancel_progress_btn.setVisible(False)
        self.cancel_progress_btn.clicked.connect(self.cancel_calculation)
        self.statusBar().addPermanentWidget(self.progress_stage_label)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.statusBar().addPermanentWidget(self.cancel_progress_btn)

    def _install_algorithm_path_status_button(self):
        self.algorithm_path_text = "暂无测量结果；分析 ROI 或自动识别后可查看实际算法路径。"
        self.algorithm_path_button = QToolButton()
        self.algorithm_path_button.setText("算法路径")
        self.algorithm_path_button.setAutoRaise(True)
        self.algorithm_path_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.algorithm_path_button.setToolTip(self.algorithm_path_text)
        self.statusBar().addPermanentWidget(self.algorithm_path_button)

    def _build_image_card(self, title: str, layer: str, canvas: ImageCanvas) -> QWidget:
        # V1.2：去掉图像区顶部的大标题条，减少占用空间，保留画布本身。
        # 为了兼容原有刷新逻辑，仍保留一个隐藏的 title_label 属性。
        card = QFrame()
        card.setObjectName("imageCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        hidden_title = QLabel(title)
        hidden_title.setVisible(False)
        layout.addWidget(hidden_title)
        layout.addWidget(canvas)
        card.title_label = hidden_title
        return card

    def _build_step_panel(self) -> QWidget:
        card = QFrame()
        card.setObjectName("stepCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 16, 14, 16)
        layout.setSpacing(10)
        title = QLabel("分析流程")
        title.setStyleSheet("font-weight: 700; font-size: 15px;")
        layout.addWidget(title)
        self.step_rows = []
        for index, title_text in enumerate(STEP_TITLES, start=1):
            row = QFrame()
            row.setObjectName("stepRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(2, 7, 2, 7)
            row_layout.setSpacing(8)
            dot = QLabel(str(index))
            dot.setAlignment(Qt.AlignCenter)
            dot.setFixedSize(26, 26)
            label = QLabel(title_text)
            note = QLabel("")
            note.setObjectName("stepNote")
            col = QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(1)
            col.addWidget(label)
            col.addWidget(note)
            state_dot = QLabel("●")
            state_dot.setAlignment(Qt.AlignCenter)
            row_layout.addWidget(dot)
            row_layout.addLayout(col, stretch=1)
            row_layout.addWidget(state_dot)
            layout.addWidget(row)
            self.step_rows.append((row, dot, label, note, state_dot))
        layout.addStretch(1)
        self.workflow_status_dot = QLabel("●")
        self.workflow_status_label = QLabel("等待导入图像")
        self.workflow_recipe_label = QLabel("当前配方：未加载")
        status_layout = QVBoxLayout()
        status_line = QHBoxLayout()
        status_line.addWidget(self.workflow_status_dot)
        status_line.addWidget(self.workflow_status_label)
        status_line.addStretch(1)
        status_layout.addLayout(status_line)
        status_layout.addWidget(self.workflow_recipe_label)
        layout.addLayout(status_layout)
        return card

    def _result_metric_card(self, title: str):
        card = QFrame()
        card.setObjectName("resultCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("resultTitle")
        value_label = QLabel("--")
        value_label.setObjectName("resultValue")
        unit_label = QLabel("")
        unit_label.setObjectName("resultUnit")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(unit_label)
        return card, value_label, unit_label

    def _build_summary_panel(self) -> QWidget:
        card = QFrame()
        card.setObjectName("summaryCard")
        # V1.3.1：压缩顶部结果卡片高度，把更多垂直空间让给“对位结果/重复性分析”。
        card.setMaximumHeight(108)
        card.setMinimumHeight(86)
        layout = QGridLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)
        self.dx_card, self.dx_value_label, self.dx_unit_label = self._result_metric_card("ΔX")
        self.dy_card, self.dy_value_label, self.dy_unit_label = self._result_metric_card("ΔY")
        self.r_card, self.r_value_label, self.r_unit_label = self._result_metric_card("综合偏差 R")
        self.result_card, self.result_value_label, self.result_unit_label = self._result_metric_card("判定结果")
        layout.addWidget(self.dx_card, 0, 0)
        layout.addWidget(self.dy_card, 0, 1)
        layout.addWidget(self.r_card, 0, 2)
        layout.addWidget(self.result_card, 0, 3)

        return card

    def _build_product_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        group_info = QGroupBox("产品与设备信息")
        form_info = QFormLayout(group_info)
        self.material_code_edit = QLineEdit()
        self.recipe_name_edit = QLineEdit()
        self.recipe_version_edit = QLineEdit()
        self.recipe_status_combo = SidebarComboBox()
        self.recipe_status_combo.addItem("草稿 / 未验证", "Draft")
        self.recipe_status_combo.addItem("已验证 / 正式生产", "Validated")
        self.process_name_edit = QLineEdit()
        self.equipment_model_edit = QLineEdit()
        self.calibration_date_edit = QLineEdit()
        self.operator_name_edit = QLineEdit()
        form_info.addRow("物料编码", self.material_code_edit)
        form_info.addRow("配方名称", self.recipe_name_edit)
        form_info.addRow("配方版本", self.recipe_version_edit)
        form_info.addRow("配方状态", self.recipe_status_combo)
        form_info.addRow("工序", self.process_name_edit)
        form_info.addRow("测量设备型号", self.equipment_model_edit)
        form_info.addRow("设备校准日期", self.calibration_date_edit)
        form_info.addRow("操作人员", self.operator_name_edit)
        layout.addWidget(group_info)
        hint = QLabel("提示：保存配方会保留执行测量所需的所有配置。相同物料再次测量时，只需加载配方、更新必要的产品/设备信息，然后按流程继续。")
        hint.setWordWrap(True)
        hint.setObjectName("statusCaption")
        layout.addWidget(hint)
        layout.addStretch(1)
        return w

    def _build_image_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        group = QGroupBox("图像导入")
        form = QFormLayout(group)
        self.image_mode_tip_label = QLabel("请在图像区上方选择：单图模式 / 双图模式")
        self.upper_file_label = QLabel("未导入")
        self.lower_file_label = QLabel("未导入")
        for label in (self.image_mode_tip_label, self.upper_file_label, self.lower_file_label):
            label.setObjectName("statusCaption")
            label.setWordWrap(True)
        form.addRow("图像模式", self.image_mode_tip_label)
        form.addRow("上层/单图文件", self.upper_file_label)
        form.addRow("下层图像文件", self.lower_file_label)
        form.addRow("导入说明", QLabel("请使用顶部工具栏导入上层/单图和下层图像。"))
        note = QLabel("单图模式：只导入单张图像；双图模式：分别导入上层物料图像和下层物料图像。")
        note.setWordWrap(True)
        note.setObjectName("statusCaption")
        form.addRow(note)
        layout.addWidget(group)

        batch_group = QGroupBox("批量测量导入")
        batch_layout = QVBoxLayout(batch_group)
        batch_form = QFormLayout()
        self.measurement_run_mode_combo = SidebarComboBox()
        self.measurement_run_mode_combo.addItem("单次测量", "Single")
        self.measurement_run_mode_combo.addItem("批量测量", "Batch")
        self.batch_count_spin = SidebarSpinBox()
        self.batch_count_spin.setRange(1, 99)
        self.batch_count_spin.setValue(3)
        batch_form.addRow("测量方式", self.measurement_run_mode_combo)
        batch_form.addRow("预计重复次数", self.batch_count_spin)
        batch_layout.addLayout(batch_form)
        btn_row1 = QHBoxLayout()
        self.batch_import_mark1_upper_btn = QPushButton("导入 Mark1 上层/单图")
        self.batch_import_mark1_lower_btn = QPushButton("导入 Mark1 下层")
        btn_row1.addWidget(self.batch_import_mark1_upper_btn)
        btn_row1.addWidget(self.batch_import_mark1_lower_btn)
        btn_row2 = QHBoxLayout()
        self.batch_import_mark2_upper_btn = QPushButton("导入 Mark2 上层/单图")
        self.batch_import_mark2_lower_btn = QPushButton("导入 Mark2 下层")
        btn_row2.addWidget(self.batch_import_mark2_upper_btn)
        btn_row2.addWidget(self.batch_import_mark2_lower_btn)
        self.batch_clear_btn = QPushButton("清空批量图像")
        batch_layout.addLayout(btn_row1)
        batch_layout.addLayout(btn_row2)
        batch_layout.addWidget(self.batch_clear_btn)
        self.batch_image_table = QTableWidget()
        batch_layout.addWidget(self.batch_image_table)
        batch_note = QLabel("批量测量会复用当前 Mark 的 ROI 模板和算法参数；点击顶部“计算对位偏差”后会自动计算所有已导入重复测量。")
        batch_note.setWordWrap(True)
        batch_note.setObjectName("statusCaption")
        batch_layout.addWidget(batch_note)
        layout.addWidget(batch_group)
        layout.addStretch(1)
        return w

    def _build_roi_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        group_mark = QGroupBox("当前 Mark")
        form_mark = QFormLayout(group_mark)
        self.mark_combo = SidebarComboBox()
        self.layer_combo = SidebarComboBox()
        self.layer_combo.addItem("上层", "upper")
        self.layer_combo.addItem("下层", "lower")
        form_mark.addRow("Mark编号", self.mark_combo)
        form_mark.addRow("当前层", self.layer_combo)
        self.roi_source_label = QLabel("未设置")
        self.roi_source_label.setObjectName("statusCaption")
        form_mark.addRow("当前 ROI 来源", self.roi_source_label)
        layout.addWidget(group_mark)

        group_auto = QGroupBox("ROI 设置方式")
        form_auto = QFormLayout(group_auto)
        self.workflow_combo = SidebarComboBox()
        self.workflow_combo.addItem("手动 ROI 测量", "Manual")
        self.workflow_combo.addItem("自动识别测量", "Auto")
        self.auto_detect_btn = QPushButton("自动识别当前 Mark")
        self.auto_reference_combo = SidebarComboBox()
        self.auto_target_combo = SidebarComboBox()
        self.auto_calculate_btn = QPushButton("计算所选轮廓对位偏差")
        self.auto_calculate_btn.setVisible(False)
        self.diagnostic_check = QCheckBox("显示诊断信息（原始轮廓 / 边缘点）")
        self.production_status_label = QLabel("自动模式：正式精测结果")
        self.workflow_explanation_label = QLabel("手动 ROI 测量会使用当前 Mark 各层已设置的 ROI。")
        self.workflow_explanation_label.setWordWrap(True)
        self.workflow_explanation_label.setObjectName("statusCaption")
        form_auto.addRow("工作方式", self.workflow_combo)
        form_auto.addRow(self.auto_detect_btn)
        form_auto.addRow("当前 Mark 基准轮廓", self.auto_reference_combo)
        form_auto.addRow("当前 Mark 待测轮廓", self.auto_target_combo)
        form_auto.addRow(self.diagnostic_check)
        form_auto.addRow(self.production_status_label)
        form_auto.addRow(self.workflow_explanation_label)
        layout.addWidget(group_auto)

        roi_section = CollapsibleSection("环形 ROI 参数", True)
        group_roi = QGroupBox("环形 ROI 参数")
        form_roi = QFormLayout(group_roi)
        self.roi_type_combo = SidebarComboBox()
        self.roi_type_combo.addItem("矩形区域", "Rectangle")
        self.roi_type_combo.addItem("圆形区域", "Circle")
        self.roi_type_combo.addItem("卡尺圆", "Caliper Circle")
        self.roi_type_combo.addItem("圆环区域", "Annulus")
        self.roi_type_combo.addItem("矩形环区域", "Rectangular Ring")
        self.center_x_spin = SidebarDoubleSpinBox()
        self.center_y_spin = SidebarDoubleSpinBox()
        self.inner_radius_spin = SidebarDoubleSpinBox()
        self.outer_radius_spin = SidebarDoubleSpinBox()
        for spin in (self.center_x_spin, self.center_y_spin, self.inner_radius_spin, self.outer_radius_spin):
            spin.setRange(-1000000, 1000000)
            spin.setDecimals(3)
            spin.setSingleStep(1.0)
        self.inner_radius_spin.setMinimum(0.0)
        self.outer_radius_spin.setMinimum(0.1)
        self.inner_radius_spin.setValue(80.0)
        self.outer_radius_spin.setValue(100.0)
        self.caliper_count_spin = SidebarSpinBox()
        self.caliper_count_spin.setRange(4, 720)
        self.caliper_count_spin.setValue(64)
        self.caliper_width_spin = SidebarDoubleSpinBox()
        self.caliper_width_spin.setRange(1.0, 10000.0)
        self.caliper_width_spin.setDecimals(3)
        self.caliper_width_spin.setValue(8.0)
        self.search_direction_combo = SidebarComboBox()
        self.search_direction_combo.addItem("由内向外", "Inner to Outer")
        self.search_direction_combo.addItem("由外向内", "Outer to Inner")
        self.target_edge_combo = SidebarComboBox()
        self.target_edge_combo.addItem("全部边缘", "All Edges")
        self.target_edge_combo.addItem("靠近内环", "Near Inner Boundary")
        self.target_edge_combo.addItem("靠近外环", "Near Outer Boundary")
        self.target_edge_combo.addItem("最强边缘", "Strongest Edge")
        self.inner_ratio_spin = SidebarDoubleSpinBox()
        self.inner_ratio_spin.setRange(0.0, 0.98)
        self.inner_ratio_spin.setDecimals(3)
        self.inner_ratio_spin.setSingleStep(0.05)
        self.inner_ratio_spin.setValue(0.60)
        self.roi_angle_spin = SidebarDoubleSpinBox()
        self.roi_angle_spin.setRange(-180.0, 180.0)
        self.roi_angle_spin.setDecimals(3)
        self.roi_angle_spin.setSingleStep(1.0)
        self.roi_angle_spin.setValue(0.0)
        self.three_point_circle_btn = QPushButton("三点定圆环中心")
        self.three_point_circle_btn.setCheckable(True)
        self.apply_roi_params_btn = QPushButton("应用环形范围")
        self.clear_current_roi_btn = QPushButton("清除当前层 ROI")
        self.clear_recipe_rois_btn = QPushButton("清除全部配方 ROI")
        form_roi.addRow("ROI类型", self.roi_type_combo)
        form_roi.addRow("中心 X", self.center_x_spin)
        form_roi.addRow("中心 Y", self.center_y_spin)
        form_roi.addRow("内半径", self.inner_radius_spin)
        form_roi.addRow("外半径", self.outer_radius_spin)
        form_roi.addRow("卡尺数量", self.caliper_count_spin)
        form_roi.addRow("卡尺宽度", self.caliper_width_spin)
        form_roi.addRow("搜索方向", self.search_direction_combo)
        form_roi.addRow("目标边缘", self.target_edge_combo)
        form_roi.addRow("矩形环角度", self.roi_angle_spin)
        form_roi.addRow(self.three_point_circle_btn)
        form_roi.addRow(self.apply_roi_params_btn)
        form_roi.addRow(self.clear_current_roi_btn)
        form_roi.addRow(self.clear_recipe_rois_btn)
        roi_section.add_widget(group_roi)
        layout.addWidget(roi_section)

        hint = QLabel("自动识别测量：先点击自动识别当前 Mark，再选择基准轮廓和待测轮廓。手动 ROI 测量：先框选并分析 ROI，再在下拉框中选择基准轮廓和待测轮廓。")
        hint.setWordWrap(True)
        hint.setObjectName("statusCaption")
        layout.addWidget(hint)
        layout.addStretch(1)
        return w

    def _build_algo_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        group_size = QGroupBox("像素尺寸与双图配准")
        form_size = QFormLayout(group_size)
        self.pixel_x_spin = SidebarDoubleSpinBox()
        self.pixel_y_spin = SidebarDoubleSpinBox()
        for spin in (self.pixel_x_spin, self.pixel_y_spin):
            spin.setRange(0.000001, 1000000)
            spin.setDecimals(6)
            spin.setSingleStep(0.01)
            spin.setValue(0.1)
        self.offset_x_spin = SidebarDoubleSpinBox()
        self.offset_y_spin = SidebarDoubleSpinBox()
        for spin in (self.offset_x_spin, self.offset_y_spin):
            spin.setRange(-1000000, 1000000)
            spin.setDecimals(6)
            spin.setSingleStep(0.1)
            spin.setValue(0.0)
        self.rx_angle_spin = SidebarDoubleSpinBox()
        self.ry_angle_spin = SidebarDoubleSpinBox()
        for spin in (self.rx_angle_spin, self.ry_angle_spin):
            spin.setRange(-1000000000.0, 1000000000.0)
            spin.setDecimals(6)
            spin.setSingleStep(1.0)
            spin.setValue(0.0)
        self.material_thickness_spin = SidebarDoubleSpinBox()
        self.material_thickness_spin.setRange(0.0, 1000000.0)
        self.material_thickness_spin.setDecimals(6)
        self.material_thickness_spin.setSingleStep(0.1)
        self.material_thickness_spin.setValue(0.0)
        form_size.addRow("像素尺寸 X (μm/px)", self.pixel_x_spin)
        form_size.addRow("像素尺寸 Y (μm/px)", self.pixel_y_spin)
        form_size.addRow("配准偏移 X (μm)", self.offset_x_spin)
        form_size.addRow("配准偏移 Y (μm)", self.offset_y_spin)
        form_size.addRow("Rx角度 (μrad)", self.rx_angle_spin)
        form_size.addRow("Ry角度 (μrad)", self.ry_angle_spin)
        form_size.addRow("物料厚度 (mm)", self.material_thickness_spin)
        layout.addWidget(group_size)

        auto_rule_section = CollapsibleSection("自动识别高级规则", False)
        group_auto_rule = QGroupBox("自动识别高级规则")
        form_auto_rule = QFormLayout(group_auto_rule)
        self.auto_ref_shape_combo = SidebarComboBox()
        self.auto_target_shape_combo = SidebarComboBox()
        for combo in (self.auto_ref_shape_combo, self.auto_target_shape_combo):
            combo.addItem("任意", "Any")
            combo.addItem("圆形", "Circle")
            combo.addItem("方形", "Rectangle")
        self.auto_ref_size_min_spin = SidebarDoubleSpinBox()
        self.auto_ref_size_max_spin = SidebarDoubleSpinBox()
        self.auto_target_size_min_spin = SidebarDoubleSpinBox()
        self.auto_target_size_max_spin = SidebarDoubleSpinBox()
        for spin in (self.auto_ref_size_min_spin, self.auto_ref_size_max_spin, self.auto_target_size_min_spin, self.auto_target_size_max_spin):
            spin.setRange(0.0, 1000000000.0)
            spin.setDecimals(6)
        self.auto_ref_size_max_spin.setValue(999999.0)
        self.auto_target_size_max_spin.setValue(999999.0)
        form_auto_rule.addRow("基准预期外形", self.auto_ref_shape_combo)
        form_auto_rule.addRow("基准尺寸下限 (μm)", self.auto_ref_size_min_spin)
        form_auto_rule.addRow("基准尺寸上限 (μm)", self.auto_ref_size_max_spin)
        form_auto_rule.addRow("待测预期外形", self.auto_target_shape_combo)
        form_auto_rule.addRow("待测尺寸下限 (μm)", self.auto_target_size_min_spin)
        form_auto_rule.addRow("待测尺寸上限 (μm)", self.auto_target_size_max_spin)
        auto_rule_section.add_widget(group_auto_rule)
        layout.addWidget(auto_rule_section)

        fit_section = CollapsibleSection("常用拟合设置", False)
        group_fit = QGroupBox("常用拟合设置")
        form_fit = QFormLayout(group_fit)
        self.fit_mode_combo = SidebarComboBox()
        self.fit_mode_combo.addItem("稳健中心（推荐）", "EdgeCenter")
        self.fit_mode_combo.addItem("区域中心", "RegionCenter")
        self.fit_mode_combo.addItem("自动选择拟合模型（仅限 ROI）", "Auto")
        self.fit_mode_combo.addItem("圆拟合", "Circle")
        self.fit_mode_combo.addItem("椭圆拟合", "Ellipse")
        self.fit_mode_combo.addItem("矩形拟合", "Rectangle")
        self.upper_fit_mode_combo = SidebarComboBox()
        self.lower_fit_mode_combo = SidebarComboBox()
        for combo in (self.upper_fit_mode_combo, self.lower_fit_mode_combo):
            combo.addItem("稳健中心（推荐）", "EdgeCenter")
            combo.addItem("区域中心", "RegionCenter")
            combo.addItem("自动选择拟合模型（仅限 ROI）", "Auto")
            combo.addItem("圆拟合", "Circle")
            combo.addItem("椭圆拟合", "Ellipse")
            combo.addItem("矩形拟合", "Rectangle")
        form_fit.addRow("默认识别方式", self.fit_mode_combo)
        form_fit.addRow("上层识别方式", self.upper_fit_mode_combo)
        form_fit.addRow("下层识别方式", self.lower_fit_mode_combo)
        fit_section.add_widget(group_fit)
        layout.addWidget(fit_section)

        rz_section = CollapsibleSection("Mark 分布与 Rz", False)
        group_rz = QGroupBox("Mark 分布 / Rz")
        form_rz = QFormLayout(group_rz)
        self.rz_layout_combo = SidebarComboBox()
        self.rz_layout_combo.addItems(["Y向前后分布", "X向左右分布"])
        self.rz_l_spin = SidebarDoubleSpinBox()
        self.rz_l_spin.setRange(0.000001, 1000000000)
        self.rz_l_spin.setDecimals(6)
        self.rz_l_spin.setSingleStep(100.0)
        self.rz_l_spin.setValue(1.0)
        self.rz_limit_spin = SidebarDoubleSpinBox()
        self.rz_limit_spin.setRange(0, 1000000000)
        self.rz_limit_spin.setDecimals(6)
        self.rz_limit_spin.setSingleStep(0.001)
        self.rz_limit_spin.setValue(999999.0)
        form_rz.addRow("Mark分布方向", self.rz_layout_combo)
        form_rz.addRow("Mark间距 L (μm)", self.rz_l_spin)
        form_rz.addRow("|Rz| 上限 (μrad)", self.rz_limit_spin)
        rz_section.add_widget(group_rz)
        layout.addWidget(rz_section)

        advanced_section = CollapsibleSection("亚像素与正式精测参数", False)
        group = QGroupBox("亚像素算法")
        form = QFormLayout(group)
        self.sigma_spin = SidebarDoubleSpinBox(); self.sigma_spin.setRange(0, 10); self.sigma_spin.setDecimals(3); self.sigma_spin.setSingleStep(0.1); self.sigma_spin.setValue(1.0)
        self.canny_low_spin = SidebarDoubleSpinBox(); self.canny_high_spin = SidebarDoubleSpinBox()
        for spin, val in [(self.canny_low_spin, 40), (self.canny_high_spin, 120)]:
            spin.setRange(0, 255); spin.setDecimals(1); spin.setSingleStep(5); spin.setValue(val)
        self.min_gradient_spin = SidebarDoubleSpinBox(); self.min_gradient_spin.setRange(0, 1000000); self.min_gradient_spin.setDecimals(3); self.min_gradient_spin.setSingleStep(1); self.min_gradient_spin.setValue(5.0)
        self.profile_half_spin = SidebarDoubleSpinBox(); self.profile_half_spin.setRange(0.5, 10); self.profile_half_spin.setDecimals(2); self.profile_half_spin.setSingleStep(0.25); self.profile_half_spin.setValue(2.0)
        self.profile_step_spin = SidebarDoubleSpinBox(); self.profile_step_spin.setRange(0.05, 2); self.profile_step_spin.setDecimals(3); self.profile_step_spin.setSingleStep(0.05); self.profile_step_spin.setValue(0.25)
        self.ransac_check = QCheckBox("启用 RANSAC 异常点剔除"); self.ransac_check.setChecked(True)
        self.residual_limit_spin = SidebarDoubleSpinBox(); self.residual_limit_spin.setRange(0.001, 1000); self.residual_limit_spin.setDecimals(4); self.residual_limit_spin.setSingleStep(0.05); self.residual_limit_spin.setValue(2.0)
        self.min_edge_points_spin = SidebarSpinBox(); self.min_edge_points_spin.setRange(3, 1000000); self.min_edge_points_spin.setValue(60)
        self.polarity_combo = SidebarComboBox(); self.polarity_combo.addItem("自动", "Auto"); self.polarity_combo.addItem("暗到亮", "Dark to Bright"); self.polarity_combo.addItem("亮到暗", "Bright to Dark")
        form.addRow("高斯滤波 Sigma (px)", self.sigma_spin)
        form.addRow("Canny 低阈值", self.canny_low_spin)
        form.addRow("Canny 高阈值", self.canny_high_spin)
        form.addRow("最小梯度", self.min_gradient_spin)
        form.addRow("剖面半宽 (px)", self.profile_half_spin)
        form.addRow("剖面步长 (px)", self.profile_step_spin)
        form.addRow("RANSAC", self.ransac_check)
        form.addRow("RANSAC剔除阈值 (px)", self.residual_limit_spin)
        form.addRow("最少边缘点数", self.min_edge_points_spin)
        form.addRow("边缘极性", self.polarity_combo)
        advanced_section.add_widget(group)
        production_group = QGroupBox("自动正式精测 / 质量门槛")
        production_form = QFormLayout(production_group)
        self.production_search_spin = SidebarDoubleSpinBox(); self.production_search_spin.setRange(2.0, 1000.0); self.production_search_spin.setDecimals(3); self.production_search_spin.setValue(8.0)
        self.production_coverage_spin = SidebarDoubleSpinBox(); self.production_coverage_spin.setRange(0.0, 1.0); self.production_coverage_spin.setDecimals(3); self.production_coverage_spin.setValue(0.65)
        self.production_reject_spin = SidebarDoubleSpinBox(); self.production_reject_spin.setRange(0.0, 1.0); self.production_reject_spin.setDecimals(3); self.production_reject_spin.setValue(0.40)
        self.production_residual_spin = SidebarDoubleSpinBox(); self.production_residual_spin.setRange(0.000001, 1000000.0); self.production_residual_spin.setDecimals(6); self.production_residual_spin.setValue(0.30)
        self.production_deviation_spin = SidebarDoubleSpinBox(); self.production_deviation_spin.setRange(0.000001, 1000000.0); self.production_deviation_spin.setDecimals(6); self.production_deviation_spin.setValue(0.60)
        production_form.addRow("自动搜索半宽 (px)", self.production_search_spin)
        production_form.addRow("最低覆盖率", self.production_coverage_spin)
        production_form.addRow("最大异常点比例", self.production_reject_spin)
        production_form.addRow("最大残差 (μm)", self.production_residual_spin)
        production_form.addRow("最大轮廓偏差 (μm)", self.production_deviation_spin)
        advanced_section.add_widget(production_group)
        layout.addWidget(advanced_section)
        layout.addStretch(1)
        return w

    def _build_spec_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        group = QGroupBox("判定规格")
        form = QFormLayout(group)
        self.dx_limit_spin = SidebarDoubleSpinBox(); self.dy_limit_spin = SidebarDoubleSpinBox(); self.r_limit_spin = SidebarDoubleSpinBox(); self.conf_min_spin = SidebarDoubleSpinBox()
        for spin, val in [(self.dx_limit_spin, 0.5), (self.dy_limit_spin, 0.5), (self.r_limit_spin, 0.7)]:
            spin.setRange(0, 1000000); spin.setDecimals(6); spin.setSingleStep(0.1); spin.setValue(val)
        self.conf_min_spin.setRange(0, 1); self.conf_min_spin.setDecimals(3); self.conf_min_spin.setSingleStep(0.05); self.conf_min_spin.setValue(0.7)
        form.addRow("|ΔX| 上限 (μm)", self.dx_limit_spin)
        form.addRow("|ΔY| 上限 (μm)", self.dy_limit_spin)
        form.addRow("对位 R 上限 (μm)", self.r_limit_spin)
        form.addRow("最低置信度", self.conf_min_spin)
        layout.addWidget(group)
        note = QLabel("导出的报告格式与 V1.0.5 保持一致。")
        note.setObjectName("statusCaption")
        layout.addWidget(note)
        layout.addStretch(1)
        return w

    def _connect_actions(self):
        self.import_upper_btn.clicked.connect(self.import_upper_image)
        self.import_lower_btn.clicked.connect(self.import_lower_image)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        self.display_enhance_check.toggled.connect(self.on_display_enhancement_changed)
        self.reset_measurement_btn.clicked.connect(self.reset_measurement)
        self.zoom_in_btn.clicked.connect(lambda: self.zoom_canvases(1.25))
        self.zoom_out_btn.clicked.connect(lambda: self.zoom_canvases(0.8))
        self.reset_view_btn.clicked.connect(self.reset_canvas_views)
        self.mark_combo.currentTextChanged.connect(self.on_active_roi_selection_changed)
        self.layer_combo.currentTextChanged.connect(self.on_active_roi_selection_changed)
        self.workflow_combo.currentIndexChanged.connect(self.on_workflow_mode_changed)
        self.auto_detect_btn.clicked.connect(self.auto_identify_marks)
        self.auto_reference_combo.currentIndexChanged.connect(self.on_auto_selection_changed)
        self.auto_target_combo.currentIndexChanged.connect(self.on_auto_selection_changed)
        self.auto_calculate_btn.clicked.connect(self.calculate_auto_overlay)
        self.diagnostic_check.toggled.connect(self._refresh_all_widgets)
        self.recipe_status_combo.currentIndexChanged.connect(self._refresh_all_widgets)
        for widget in (
            self.auto_ref_shape_combo,
            self.auto_target_shape_combo,
            self.auto_ref_size_min_spin,
            self.auto_ref_size_max_spin,
            self.auto_target_size_min_spin,
            self.auto_target_size_max_spin,
        ):
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.on_auto_match_rule_changed)
            else:
                widget.valueChanged.connect(self.on_auto_match_rule_changed)
        self.roi_type_combo.currentTextChanged.connect(self._refresh_all_widgets)
        self.center_x_spin.valueChanged.connect(self.apply_roi_params_to_current)
        self.center_y_spin.valueChanged.connect(self.apply_roi_params_to_current)
        self.inner_radius_spin.valueChanged.connect(self.apply_roi_params_to_current)
        self.outer_radius_spin.valueChanged.connect(self.apply_roi_params_to_current)
        self.caliper_count_spin.valueChanged.connect(self.apply_roi_params_to_current)
        self.caliper_width_spin.valueChanged.connect(self.apply_roi_params_to_current)
        self.search_direction_combo.currentTextChanged.connect(self.apply_roi_params_to_current)
        self.target_edge_combo.currentTextChanged.connect(self._refresh_all_widgets)
        self.inner_ratio_spin.valueChanged.connect(self._refresh_all_widgets)
        self.roi_angle_spin.valueChanged.connect(self._refresh_all_widgets)
        self.three_point_circle_btn.toggled.connect(self.on_three_point_circle_toggled)
        self.fit_mode_combo.currentTextChanged.connect(self._refresh_all_widgets)
        self.upper_fit_mode_combo.currentTextChanged.connect(self._refresh_all_widgets)
        self.lower_fit_mode_combo.currentTextChanged.connect(self._refresh_all_widgets)
        self.apply_roi_params_btn.clicked.connect(self.apply_roi_params_to_current)
        self.clear_current_roi_btn.clicked.connect(self.clear_current_roi)
        self.clear_recipe_rois_btn.clicked.connect(self.clear_all_recipe_rois)
        self.upper_canvas.roiChanged.connect(self.set_roi)
        self.lower_canvas.roiChanged.connect(self.set_roi)
        self.analyze_roi_btn.clicked.connect(self.analyze_roi_regions)
        self.analyze_current_btn.clicked.connect(self.analyze_current_mark)
        self.analyze_all_btn.clicked.connect(self.analyze_all_marks)
        self.export_btn.clicked.connect(self.export_result_file)
        self.save_recipe_btn.clicked.connect(self.save_recipe_file)
        self.load_recipe_btn.clicked.connect(self.load_recipe_file)
        self.batch_import_mark1_upper_btn.clicked.connect(lambda: self.import_batch_images("Mark1", "upper"))
        self.batch_import_mark1_lower_btn.clicked.connect(lambda: self.import_batch_images("Mark1", "lower"))
        self.batch_import_mark2_upper_btn.clicked.connect(lambda: self.import_batch_images("Mark2", "upper"))
        self.batch_import_mark2_lower_btn.clicked.connect(lambda: self.import_batch_images("Mark2", "lower"))
        self.batch_clear_btn.clicked.connect(self.clear_batch_images)
        self.measurement_run_mode_combo.currentIndexChanged.connect(self._refresh_all_widgets)
        if hasattr(self, "algorithm_path_button"):
            self.algorithm_path_button.clicked.connect(self.show_algorithm_path_dialog)
        self._install_button_feedback()

    def on_display_enhancement_changed(self, checked: bool):
        self.upper_canvas.set_display_enhancement(checked)
        self.lower_canvas.set_display_enhancement(checked)
        self._append_log("已开启显示增强。" if checked else "已关闭显示增强。")

    def _append_log(self, message: str):
        # V1.2.5：取消独立日志窗口，所有操作反馈统一显示在左下角状态栏，避免界面拥挤。
        self.statusBar().showMessage(message, 3500)

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
        self._set_combo_value(self.auto_reference_combo, getattr(self.config, "auto_reference_label", ""))
        self._set_combo_value(self.auto_target_combo, getattr(self.config, "auto_target_label", ""))

    def _current_roi(self):
        mark_id = self.mark_combo.currentText() or "Mark1"
        layer = self._current_layer()
        mark = self.marks.get(mark_id)
        if not mark:
            return None
        return mark.upper_roi if layer == "upper" else mark.lower_roi

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
        self.auto_detections_by_mark.setdefault(mark_id, {})
        self.auto_candidates_by_mark.setdefault(mark_id, {})
        self.auto_selections.setdefault(mark_id, {"reference_label": "", "target_label": ""})

    def _current_auto_detections(self):
        mark_id = self._current_mark_id()
        self._ensure_mark_runtime(mark_id)
        return self.auto_detections_by_mark[mark_id]

    def _sync_current_mark_images(self):
        mark_id = self._current_mark_id()
        self._ensure_mark_runtime(mark_id)
        self.upper_image = self.mark_images[mark_id]["upper"]
        self.lower_image = self.mark_images[mark_id]["lower"]
        self.upper_canvas.set_image(self.upper_image)
        self.lower_canvas.set_image(self.lower_image)

    def _invalidate_image_dependent_results(self, mark_id: str, layer: str):
        if self._current_mode() == "Single Image":
            self.detections.pop(mark_id, None)
        elif mark_id in self.detections:
            self.detections[mark_id].pop(layer, None)
            if not self.detections[mark_id]:
                self.detections.pop(mark_id, None)
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
            self.inner_ratio_spin.blockSignals(False)
            self.roi_angle_spin.blockSignals(False)
        self._refresh_all_widgets()

    def apply_roi_params_to_current(self):
        mark_id = self.mark_combo.currentText() or "Mark1"
        layer = self._current_layer()
        mark = self.marks.get(mark_id)
        if not mark:
            return
        roi = mark.upper_roi if layer == "upper" else mark.lower_roi
        if roi is None:
            return
        cx = self.center_x_spin.value()
        cy = self.center_y_spin.value()
        inner = max(0.0, self.inner_radius_spin.value())
        outer = max(inner + 0.1, self.outer_radius_spin.value())
        roi.x = cx - outer
        roi.y = cy - outer
        roi.w = outer * 2.0
        roi.h = outer * 2.0
        roi.roi_type = self._auto_ring_roi_type(layer)
        roi.inner_ratio = float(np.clip(inner / max(outer, 1e-9), 0.0, 0.98))
        roi.target_edge = self._combo_value(self.target_edge_combo)
        roi.angle_deg = self.roi_angle_spin.value()
        roi.caliper_count = self.caliper_count_spin.value()
        roi.caliper_width_px = self.caliper_width_spin.value()
        roi.search_direction = self._combo_value(self.search_direction_combo)
        self.roi_sources.setdefault(mark_id, {})[layer] = "manual"
        self._recipe_roi_confirmation_signature = None
        if mark_id in self.detections and layer in self.detections[mark_id]:
            del self.detections[mark_id][layer]
        if mark_id in self.overlays:
            del self.overlays[mark_id]
        # Reflect the just-drawn ROI parameters in the side panel.
        if mark_id == (self.mark_combo.currentText() or "Mark1") and layer == self._current_layer():
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
            self.inner_ratio_spin.setValue(float(getattr(roi, "inner_ratio", 0.60)))
            self.roi_angle_spin.setValue(float(getattr(roi, "angle_deg", 0.0)))
        self._refresh_all_widgets()

    def _refresh_all_widgets(self, *args):
        current_mark = self.mark_combo.currentText() or "Mark1"
        current_layer = self._current_layer()
        is_dual = self._current_mode() == "Dual Image"
        self.import_upper_btn.setText("📂 导入上层/单图")
        self.import_lower_btn.setText("📁 导入下层图像")
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
        show_auto = self._is_auto_workflow()
        if hasattr(self, "roi_source_label"):
            self.roi_source_label.setText(self._roi_source_text(self._roi_source(current_mark, current_layer)))
        if hasattr(self, "workflow_explanation_label"):
            self.workflow_explanation_label.setText(
                "全图自动识别：本次计算不会读取任何 ROI。"
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
            current_auto_detections,
            show_auto,
            manual_labels,
            reference_label,
            target_label,
            self.config.pixel_size_x_um,
            self.config.pixel_size_y_um,
            self.diagnostic_check.isChecked() if hasattr(self, "diagnostic_check") else False,
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
            current_auto_detections,
            show_auto,
            manual_labels,
            reference_label,
            target_label,
            self.config.pixel_size_x_um,
            self.config.pixel_size_y_um,
            self.diagnostic_check.isChecked() if hasattr(self, "diagnostic_check") else False,
        )
        self.lower_canvas.setVisible(is_dual)
        if hasattr(self, "lower_image_card"):
            self.lower_image_card.setVisible(is_dual)
        self.import_lower_btn.setEnabled(is_dual)
        self.auto_detect_btn.setEnabled(show_auto)
        # 基准/待测轮廓选择在自动识别和手动 ROI 两种工作方式下都需要可用。
        self.auto_reference_combo.setEnabled(True)
        self.auto_target_combo.setEnabled(True)
        auto_reference = self.auto_reference_combo.currentData()
        auto_target = self.auto_target_combo.currentData()
        self.auto_calculate_btn.setEnabled(
            bool(auto_reference) and bool(auto_target) and auto_reference != auto_target
        )
        self.analyze_roi_btn.setEnabled(not show_auto)
        self.analyze_current_btn.setEnabled(not show_auto)
        self.analyze_all_btn.setEnabled(True)
        self.three_point_circle_btn.setEnabled(not show_auto)
        self.apply_roi_params_btn.setEnabled(not show_auto)
        self.clear_current_roi_btn.setEnabled(not show_auto)
        self._refresh_mark_combo()
        if hasattr(self, "current_recipe_label"):
            recipe_display = self.loaded_recipe_display_name or self.recipe_name_edit.text().strip() or "未加载"
            self.current_recipe_label.setText(f"当前配方：{recipe_display}")
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
                self.image_status_label.setText("图像已加载，可分析 ROI 区域或计算对位偏差")
        if hasattr(self, "upper_file_label"):
            upper_img = self._image_for_layer("upper", current_mark)
            lower_img = self._image_for_layer("lower", current_mark)
            self.upper_file_label.setText(Path(upper_img.path).name if upper_img and upper_img.path else "未导入")
            self.lower_file_label.setText(Path(lower_img.path).name if lower_img and lower_img.path else "未导入")
            if hasattr(self, "image_mode_tip_label"):
                self.image_mode_tip_label.setText(self.mode_combo.currentText())
        self._refresh_tables()
        self._refresh_batch_image_table()
        self._refresh_repeatability_table()
        if hasattr(self, "_refresh_summary_cards"):
            self._refresh_summary_cards()
        if hasattr(self, "_refresh_step_status"):
            self._refresh_step_status(current_mark, show_auto)
        if hasattr(self, "_refresh_algorithm_path_panel"):
            self._refresh_algorithm_path_panel(current_mark, show_auto)

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
        numbered = []
        for mark_id, layer_map in self.detections.items():
            for layer, detection in layer_map.items():
                numbered.append((detection.diameter_px, mark_id, layer))
        numbered.sort(key=lambda item: (-item[0], item[1], item[2]))
        return {(mark_id, layer): self._alpha_label(index) for index, (_, mark_id, layer) in enumerate(numbered)}

    def _display_detections(self):
        if not self._is_auto_workflow():
            return self.detections
        combined = {}
        for mark_id, detected in self.auto_detections_by_mark.items():
            for label, layer_map in detected.items():
                combined[f"{mark_id}-{label}"] = layer_map
        return combined

    def _display_overlays(self):
        return self.auto_overlays if self._is_auto_workflow() else self.overlays

    def _find_auto_detection(self, mark_id: str, label: str) -> Optional[DetectionResult]:
        layer_map = self.auto_detections_by_mark.get(mark_id, {}).get(label, {})
        return next(iter(layer_map.values()), None)

    def _build_summary_rows(self):
        overlay_map = self.auto_overlays if self._is_auto_workflow() else self.overlays
        is_trial = self._is_auto_workflow() and self.config.recipe_validation_status != "Validated"
        return build_summary_rows(overlay_map, self.config, is_trial=is_trial)


    def _set_result_card(self, value_label: QLabel, unit_label: QLabel, value: str, unit: str, color: str = "#1D1D1F"):
        value_label.setText(value)
        value_label.setStyleSheet(f"color: {color};")
        unit_label.setText(unit)

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
            color = "#34C759" if verdict == "通过" else ("#FF3B30" if verdict == "不通过" else "#FFCC00")
            self._set_result_card(self.dx_value_label, self.dx_unit_label, f"{dx:+.3f}", "μm", "#007AFF")
            self._set_result_card(self.dy_value_label, self.dy_unit_label, f"{dy:+.3f}", "μm", "#FF9500")
            self._set_result_card(self.r_value_label, self.r_unit_label, f"{dxy:.3f}", "μm")
            self._set_result_card(self.result_value_label, self.result_unit_label, verdict, first.get("提示", "") or "离线分析状态", color)
            return
        self._set_result_card(self.dx_value_label, self.dx_unit_label, "--", "μm", "#007AFF")
        self._set_result_card(self.dy_value_label, self.dy_unit_label, "--", "μm", "#FF9500")
        self._set_result_card(self.r_value_label, self.r_unit_label, "--", "μm")
        self._set_result_card(self.result_value_label, self.result_unit_label, "待分析", "离线分析状态", "#6E6E73")

    def _refresh_step_status(self, current_mark: str, show_auto: bool):
        imported = self._image_for_layer("upper", current_mark) is not None
        if self._current_mode() == "Dual Image":
            imported = imported and self._image_for_layer("lower", current_mark) is not None
        mark = self.marks.get(current_mark)
        roi_ready = bool(mark and (mark.upper_roi is not None or mark.lower_roi is not None))
        result_ready = current_mark in (self.auto_overlays if show_auto else self.overlays)
        states = [
            ("完成", "信息可编辑"),
            ("完成" if imported else "当前", "图像已加载" if imported else "等待导入图像"),
            ("完成" if roi_ready or show_auto else ("当前" if imported else "待处理"), "ROI 已设置" if roi_ready else ("自动识别模式" if show_auto else "等待设置 ROI")),
            ("完成" if imported else "待处理", "参数已就绪" if imported else "导入图像后设置"),
            ("完成" if result_ready else "待处理", "可导出结果" if result_ready else "等待计算"),
        ]
        colors = {"待处理": "#A1A1A6", "当前": "#007AFF", "完成": "#34C759", "异常": "#FF3B30"}
        for idx, (row, dot, label, note, state_dot) in enumerate(self.step_rows):
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
            return
        parts = [f"当前 {current_mark}"]
        if reference is not None:
            parts.append(f"基准({reference_label})：{describe_algorithm_path(reference, workflow)}")
        if target is not None:
            parts.append(f"待测({target_label})：{describe_algorithm_path(target, workflow)}")
        self.algorithm_path_text = "；".join(parts)
        self.algorithm_path_button.setToolTip(self.algorithm_path_text)

    def _refresh_tables(self):
        det_headers = [
            "标记", "层", "中心 X (μm)", "中心 Y (μm)",
            "尺寸/直径 (μm)", "参考残差 (μm)", "边缘点数", "置信度", "算法",
            "质量状态", "覆盖率", "形状参数", "算法路径", "提示",
        ]
        det_rows = []
        display_detections = self._display_detections()
        manual_labels = self._manual_detection_labels()
        for mark_id, layer_map in display_detections.items():
            for layer, d in layer_map.items():
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
                    angle_deg = d.shape_params.get("angle_deg", 0)
                    major_um = d.shape_params.get("major_px", 0) * axis_scale_um_per_px(self.config, angle_deg)
                    minor_um = d.shape_params.get("minor_px", 0) * axis_scale_um_per_px(self.config, angle_deg + 90.0)
                    shape_txt = (
                        f"长轴={major_um:.3f}μm, "
                        f"短轴={minor_um:.3f}μm, "
                        f"角度={d.shape_params.get('angle_deg', 0):.3f}°"
                    )
                elif d.fitting_mode == "Circle":
                    shape_txt = f"半径={d.diameter_um / 2.0:.3f}μm"
                elif d.fitting_mode == "EdgeCenter":
                    shape_txt = (
                        f"边缘中心, 参考半径={d.diameter_um / 2.0:.3f}μm, "
                        f"宽={d.shape_params.get('width_px', 0) * self.config.pixel_size_x_um:.3f}μm, "
                        f"高={d.shape_params.get('height_px', 0) * self.config.pixel_size_y_um:.3f}μm"
                    )
                elif d.fitting_mode == "CaliperCircle":
                    shape_txt = (
                        f"卡尺圆, 半径={d.diameter_um / 2.0:.3f}μm, "
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
                if not self._is_auto_workflow() and (mark_id, layer) in manual_labels:
                    display_mark_id = f"{mark_id} [{manual_labels[(mark_id, layer)]}]"
                det_rows.append([
                    display_mark_id, LAYER_LABELS.get(layer, layer),
                    f"{d.center_x_um:.3f}", f"{d.center_y_um:.3f}",
                    f"{d.diameter_um:.3f}", f"{d.residual_um:.3f}",
                    str(d.edge_point_count), f"{d.confidence:.3f}", mode_txt,
                    {"Valid": "有效", "Invalid": "无效"}.get(d.shape_params.get("quality_status", ""), ""),
                    f"{d.shape_params.get('coverage', 0):.1%}" if "coverage" in d.shape_params else "",
                    shape_txt + "; " + roi_txt,
                    d.shape_params.get("algorithm_path", describe_algorithm_path(d, "Auto" if self._is_auto_workflow() else "Manual")),
                    d.warning,
                ])
        self._fill_table(self.det_table, det_headers, det_rows)

        ov_headers = ["项目", "Dx/Dy/Dxy/Rz", "数值", "判定", "提示"]
        ov_rows = []
        for row in self._build_summary_rows():
            project = row.get("项目", "")
            verdict = row.get("判定", "")
            note = row.get("提示", "")
            for key, value in row.items():
                if key in {"项目", "判定", "提示", "公式", "L(μm)"}:
                    continue
                ov_rows.append([project, key, f"{value:.3f}" if isinstance(value, float) else value, verdict, note])
            if project == "Rz":
                ov_rows[-1][4] = f"{row.get('公式', '')}；L={row.get('L(μm)', '')}；{note}".strip("；")
        self._fill_table(self.overlay_table, ov_headers, ov_rows)

    def _fill_table(self, table: QTableWidget, headers, rows):
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                row_text = " ".join(str(x) for x in row)
                if "失败" in row_text or "超限" in row_text or "Fail" in row_text:
                    item.setBackground(QColor(255, 199, 206))
                    item.setForeground(QColor(156, 0, 6))
                table.setItem(r, c, item)
        table.resizeColumnsToContents()

    def zoom_canvases(self, factor: float):
        self.upper_canvas.zoom_by(factor)
        if self.lower_canvas.isVisible():
            self.lower_canvas.zoom_by(factor)

    def reset_canvas_views(self):
        self.upper_canvas.reset_view(update=True)
        self.lower_canvas.reset_view(update=True)

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
        if self._is_auto_workflow():
            items = []
            for mark_id, detected in self.auto_detections_by_mark.items():
                for label, layer_map in detected.items():
                    for layer_key, detection in layer_map.items():
                        image = self._image_for_layer(layer_key, mark_id)
                        radius = max(6.0, float(detection.shape_params.get("radius_px", detection.diameter_px / 2.0)))
                        roi = Roi(
                            detection.center_x_px - radius,
                            detection.center_y_px - radius,
                            radius * 2.0,
                            radius * 2.0,
                        )
                        out_path = Path(tmp_dir) / f"auto_{mark_id}_{label}_{layer_key}.png"
                        if self._crop_roi_image(image, roi, out_path):
                            items.append({
                                "mark_id": f"{mark_id}-{label}",
                                "layer": LAYER_LABELS.get(layer_key, layer_key),
                                "path": str(out_path),
                                "note": "自动识别轮廓截图",
                            })
            return items
        items = []
        for mark_id, mark in self.marks.items():
            layer_pairs = [("upper", "上层", self._image_for_layer("upper", mark_id), mark.upper_roi)]
            if self._current_mode() == "Dual Image":
                layer_pairs.append(("lower", "下层", self._image_for_layer("lower", mark_id), mark.lower_roi))
            for layer_key, layer_label, image, roi in layer_pairs:
                out_path = Path(tmp_dir) / f"{mark_id}_{layer_key}.png"
                if self._crop_roi_image(image, roi, out_path):
                    items.append({
                        "mark_id": mark_id,
                        "layer": layer_label,
                        "path": str(out_path),
                        "note": "ROI区域截图",
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
            image = self.mark_images.get(mark_id, {}).get("upper")
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
                        "判定": {"Pass": "通过", "Fail": "不通过", "Trial": "试测"}.get(overlay.result, overlay.result) if overlay else "失败",
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
                        "判定": {"Pass": "通过", "Fail": "不通过", "Trial": "试测"}.get(overlay.result, overlay.result),
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
        for mark_id in ("Mark1", "Mark2"):
            self.auto_detections_by_mark[mark_id] = {}
            self.auto_candidates_by_mark[mark_id] = {}
            self.auto_selections[mark_id] = {"reference_label": "", "target_label": ""}
        self.auto_overlays.clear()
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
        self.auto_overlays.pop(mark.mark_id, None)
        self._refresh_auto_selection_combos()
        self._refresh_all_widgets()

    def _matches_auto_rule(self, detection: DetectionResult, role: str, mark: MarkRecipe) -> bool:
        shape = detection.shape_params.get("shape_type", "")
        expected = mark.reference_shape if role == "reference" else mark.target_shape
        minimum = mark.reference_size_min_um if role == "reference" else mark.target_size_min_um
        maximum = mark.reference_size_max_um if role == "reference" else mark.target_size_max_um
        return (expected == "Any" or expected == shape) and minimum <= detection.diameter_um <= maximum

    def _refresh_auto_selection_combos(self):
        """Refresh reference/target contour selectors for both Auto and Manual workflows.

        Auto workflow uses contours detected by auto_mark_detector and labeled a/b/c/d.
        Manual workflow uses the currently analyzed manual ROI results, so users can
        explicitly choose which ROI is the reference contour and which is the target contour.
        """
        if not hasattr(self, "auto_reference_combo"):
            return
        mark_id = self._current_mark_id()
        self._ensure_mark_runtime(mark_id)
        selection = self.auto_selections[mark_id]
        previous_reference = selection.get("reference_label", "")
        previous_target = selection.get("target_label", "")
        self.auto_reference_combo.blockSignals(True)
        self.auto_target_combo.blockSignals(True)
        self.auto_reference_combo.clear()
        self.auto_target_combo.clear()

        if self._is_auto_workflow():
            mark = self.marks[mark_id]
            for label, layer_map in self.auto_detections_by_mark[mark_id].items():
                detection = next(iter(layer_map.values()))
                if detection.shape_params.get("quality_status") != "Valid":
                    continue
                shape = "方" if detection.fitting_mode == "ProductionRectangle" else "圆"
                size_name = "半尺寸" if detection.fitting_mode in {"AutoRectangle", "ProductionRectangle"} else "半径"
                text = f"{mark_id}-{label} - {shape} - {size_name}={detection.diameter_um / 2.0:.3f} μm"
                if self._matches_auto_rule(detection, "reference", mark):
                    self.auto_reference_combo.addItem(text, label)
                if self._matches_auto_rule(detection, "target", mark):
                    self.auto_target_combo.addItem(text, label)
        else:
            layer_map = self.detections.get(mark_id, {})
            for layer in ("upper", "lower"):
                detection = layer_map.get(layer)
                if detection is None:
                    continue
                label = layer
                layer_name = LAYER_LABELS.get(layer, layer)
                fit_name = {
                    "Circle": "圆拟合",
                    "Ellipse": "椭圆拟合",
                    "Rectangle": "矩形拟合",
                    "EdgeCenter": "稳健中心",
                    "RegionCenter": "区域中心",
                    "CaliperCircle": "卡尺圆",
                }.get(detection.fitting_mode, detection.fitting_mode)
                text = (
                    f"{mark_id}-{layer_name}轮廓 - {fit_name} - "
                    f"中心=({detection.center_x_um:.3f}, {detection.center_y_um:.3f}) μm"
                )
                self.auto_reference_combo.addItem(text, label)
                self.auto_target_combo.addItem(text, label)

        if self.auto_reference_combo.count() > 0:
            self._set_combo_value(self.auto_reference_combo, previous_reference)
            if not self.auto_reference_combo.currentData():
                self.auto_reference_combo.setCurrentIndex(0)
        if self.auto_target_combo.count() > 0:
            self._set_combo_value(self.auto_target_combo, previous_target)
            if self.auto_target_combo.currentData() == self.auto_reference_combo.currentData() and self.auto_target_combo.count() > 1:
                self.auto_target_combo.setCurrentIndex(1)
        self.auto_reference_combo.blockSignals(False)
        self.auto_target_combo.blockSignals(False)
        self.auto_selections[mark_id] = {
            "reference_label": self.auto_reference_combo.currentData() or "",
            "target_label": self.auto_target_combo.currentData() or "",
        }

    def auto_identify_marks(self, show_message: bool = True):
        self._pull_config_from_ui()
        mark_id = self._current_mark_id()
        images = [("upper", self._image_for_layer("upper", mark_id))]
        if self._current_mode() == "Dual Image":
            images.append(("lower", self._image_for_layer("lower", mark_id)))
        if any(image is None for _, image in images):
            if show_message:
                QMessageBox.warning(self, "自动识别", "请先导入当前测量模式需要的图像。")
            return 0

        if hasattr(self, "auto_detect_btn"):
            self.auto_detect_btn.setEnabled(False)
        self.statusBar().showMessage("正在自动识别轮廓，请稍候……", 5000)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        detected = {}
        candidates = {}
        try:
            results_all = []
            report_warnings = []
            for layer, image in images:
                try:
                    report = detect_auto_marks_with_report(
                        image.gray,
                        layer,
                        self.params,
                        self.config.pixel_size_x_um,
                        self.config.pixel_size_y_um,
                    )
                    results = report.results
                    warning_text = report.warning_text()
                    if warning_text:
                        report_warnings.append(f"{LAYER_LABELS.get(layer, layer)}：{warning_text}")
                except Exception as exc:
                    self._append_log(f"自动识别 {mark_id} {LAYER_LABELS.get(layer, layer)} 失败：{self._friendly_error(exc)}")
                    results = []
                results_all.extend(results)
                QApplication.processEvents()

            results_all.sort(key=lambda result: -result.diameter_px)
            # Avoid UI stalls on noisy images by refining only the largest/relevant candidates.
            max_candidates = 32
            results_all = results_all[:max_candidates]
            for label_index, result in enumerate(results_all):
                label = self._alpha_label(label_index)
                result.mark_id = f"{mark_id}-{label}"
                candidates[label] = {result.layer: result}
                image = self._image_for_layer(result.layer, mark_id)
                try:
                    measured = refine_candidate(image.gray, result, self.params, self.config)
                except Exception as exc:
                    measured = result
                    measured.shape_params["quality_status"] = "Invalid"
                    measured.shape_params["failure_reason"] = f"精测失败：{self._friendly_error(exc)}"
                    measured.warning = measured.shape_params["failure_reason"]
                    measured = attach_algorithm_path(measured, "Auto")
                if not (self.params.diameter_min_um <= measured.diameter_um <= self.params.diameter_max_um):
                    measured.shape_params["quality_status"] = "Invalid"
                    measured.shape_params["failure_reason"] = "尺寸超出配方范围"
                    measured.warning = "尺寸超出配方范围"
                detected[label] = {result.layer: measured}
                QApplication.processEvents()

            self.auto_candidates_by_mark[mark_id] = candidates
            self.auto_detections_by_mark[mark_id] = detected
            self.auto_overlays.pop(mark_id, None)
            self.auto_selections[mark_id] = {"reference_label": "", "target_label": ""}
            self._refresh_auto_selection_combos()
            self._refresh_all_widgets()
            if report_warnings:
                self._append_log("；".join(report_warnings))
            if show_message:
                if detected:
                    valid_count = sum(
                        next(iter(layer_map.values())).shape_params.get("quality_status") == "Valid"
                        for layer_map in detected.values()
                    )
                    message = f"共发现 {len(detected)} 个候选，精测有效 {valid_count} 个。"
                    if report_warnings:
                        message += "\n\n提示：\n" + "\n".join(report_warnings)
                    QMessageBox.information(self, "自动精测完成", message)
                else:
                    message = "未找到可用的闭合 Mark 轮廓，请检查对比度、焦面、ROI/算法参数或改用手动 ROI。"
                    if report_warnings:
                        message += "\n\n提示：\n" + "\n".join(report_warnings)
                    QMessageBox.warning(self, "自动识别", message)
            status_message = f"自动识别完成：{len(detected)} 个候选"
            if report_warnings:
                status_message += "；存在截断提示"
            self.statusBar().showMessage(status_message, 5000)
            return len(detected)
        finally:
            QApplication.restoreOverrideCursor()
            if hasattr(self, "auto_detect_btn"):
                self.auto_detect_btn.setEnabled(True)

    def _find_manual_detection(self, mark_id: str, label: str) -> Optional[DetectionResult]:
        if label in {"upper", "lower"}:
            return self.detections.get(mark_id, {}).get(label)
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

        name = f"{mark_id}: {target_label} 相对 {reference_label}"
        overlay = calculate_relative_overlay(mark_id, reference, target, self.config)
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

    def import_batch_images(self, mark_id: str, layer: str):
        title = f"批量导入 {mark_id} {LAYER_LABELS.get(layer, layer)}图像"
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            title,
            "",
            "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;矩阵 (*.csv *.txt *.npy);;全部文件 (*)",
        )
        if not paths:
            self._append_log(f"取消{title}。")
            return
        try:
            self._ensure_mark_runtime(mark_id)
            images = [load_image(path) for path in paths]
            self.batch_images[mark_id][layer] = images
            self.batch_run_records[mark_id] = []
            self.batch_overlays[mark_id] = []
            # Use first repeat as preview image so ROI can be drawn/confirmed once.
            if images:
                self.mark_images[mark_id][layer] = images[0]
                self._invalidate_image_dependent_results(mark_id, layer)
            self._set_combo_value(self.measurement_run_mode_combo, "Batch")
            self._sync_current_mark_images()
            self._refresh_auto_selection_combos()
            self._refresh_all_widgets()
            self._append_log(f"{title}完成：{len(images)} 张。")
        except Exception as exc:
            self._append_log(f"{title}失败：{exc}")
            QMessageBox.critical(self, "批量导入失败", str(exc))

    def clear_batch_images(self):
        self.batch_images = {"Mark1": {"upper": [], "lower": []}, "Mark2": {"upper": [], "lower": []}}
        self.batch_overlays = {"Mark1": [], "Mark2": []}
        self.batch_run_records = {"Mark1": [], "Mark2": []}
        self._refresh_all_widgets()
        self._append_log("已清空批量图像和重复性结果。")

    def _refresh_batch_image_table(self):
        if not hasattr(self, "batch_image_table"):
            return
        headers = ["Mark", "上层/单图数量", "下层数量", "状态"]
        rows = []
        is_dual = self._current_mode() == "Dual Image"
        for mark_id in ("Mark1", "Mark2"):
            upper_count = len(self.batch_images.get(mark_id, {}).get("upper", []))
            lower_count = len(self.batch_images.get(mark_id, {}).get("lower", []))
            if upper_count == 0:
                status = "未导入"
            elif is_dual and lower_count == 0:
                status = "缺少下层"
            elif is_dual and upper_count != lower_count:
                status = "上下数量不一致"
            else:
                status = "可批量计算"
            rows.append([mark_id, str(upper_count), str(lower_count), status])
        self._fill_table(self.batch_image_table, headers, rows)

    def _refresh_repeatability_table(self):
        if not hasattr(self, "repeat_table"):
            return
        headers = ["Mark", "次数", "Dx(μm)", "Dy(μm)", "Dxy(μm)", "判定", "提示"]
        rows = []
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
                        {"Pass": "通过", "Fail": "不通过", "Trial": "试测"}.get(overlay.result, overlay.result) if overlay else "失败",
                        overlay.warning if overlay else record.get("error", ""),
                    ])
            else:
                for idx, overlay in enumerate(overlays, start=1):
                    rows.append([
                        mark_id,
                        str(idx),
                        f"{overlay.delta_x_um:+.3f}",
                        f"{overlay.delta_y_um:+.3f}",
                        f"{overlay.overlay_r_um:.3f}",
                        {"Pass": "通过", "Fail": "不通过", "Trial": "试测"}.get(overlay.result, overlay.result),
                        overlay.warning,
                    ])
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
        self._fill_table(self.repeat_table, headers, rows)

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

    def import_upper_image(self):
        mark_id = self._current_mark_id()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入上层/单张图像",
            "",
            "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;矩阵 (*.csv *.txt *.npy);;全部文件 (*)",
        )
        if not path:
            self._append_log("取消导入上层/单图。")
            return
        try:
            self._ensure_mark_runtime(mark_id)
            self.mark_images[mark_id]["upper"] = load_image(path)
            self._invalidate_image_dependent_results(mark_id, "upper")
            self._sync_current_mark_images()
            self._refresh_auto_selection_combos()
            self._append_log(f"已导入上层/单图：{Path(path).name}")
        except Exception as exc:
            self._append_log(f"导入上层/单图失败：{exc}")
            QMessageBox.critical(self, "导入失败", str(exc))
        self._refresh_all_widgets()

    def import_lower_image(self):
        mark_id = self._current_mark_id()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入下层图像",
            "",
            "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;矩阵 (*.csv *.txt *.npy);;全部文件 (*)",
        )
        if not path:
            self._append_log("取消导入下层图像。")
            return
        try:
            self._ensure_mark_runtime(mark_id)
            self.mark_images[mark_id]["lower"] = load_image(path)
            self._invalidate_image_dependent_results(mark_id, "lower")
            self._sync_current_mark_images()
            self._refresh_auto_selection_combos()
            self._append_log(f"已导入下层图像：{Path(path).name}")
        except Exception as exc:
            self._append_log(f"导入下层图像失败：{exc}")
            QMessageBox.critical(self, "导入失败", str(exc))
        self._refresh_all_widgets()

    def add_mark(self):
        self.marks = {
            mark_id: self.marks.get(mark_id, MarkRecipe(mark_id))
            for mark_id in ("Mark1", "Mark2")
        }
        QMessageBox.information(self, "标记数量", "本版本仅支持 Mark1 和 Mark2。")
        self._refresh_all_widgets()

    def reset_measurement(self):
        self.marks = {"Mark1": MarkRecipe("Mark1"), "Mark2": MarkRecipe("Mark2")}
        self.mark_images = {
            "Mark1": {"upper": None, "lower": None},
            "Mark2": {"upper": None, "lower": None},
        }
        self.detections.clear()
        self.overlays.clear()
        self.auto_detections_by_mark = {"Mark1": {}, "Mark2": {}}
        self.auto_candidates_by_mark = {"Mark1": {}, "Mark2": {}}
        self.auto_selections = {
            "Mark1": {"reference_label": "", "target_label": ""},
            "Mark2": {"reference_label": "", "target_label": ""},
        }
        self.auto_overlays.clear()
        self.batch_images = {"Mark1": {"upper": [], "lower": []}, "Mark2": {"upper": [], "lower": []}}
        self.batch_overlays = {"Mark1": [], "Mark2": []}
        self.batch_run_records = {"Mark1": [], "Mark2": []}
        self.roi_sources = self._empty_roi_sources()
        self.loaded_recipe_path = ""
        self.loaded_recipe_display_name = ""
        self._recipe_roi_confirmation_signature = None
        self.upper_canvas.set_circle_pick_mode(False)
        self.lower_canvas.set_circle_pick_mode(False)
        self.three_point_circle_btn.blockSignals(True)
        self.three_point_circle_btn.setChecked(False)
        self.three_point_circle_btn.blockSignals(False)
        self.mark_combo.setCurrentText("Mark1")
        self._sync_current_mark_images()
        self._refresh_auto_selection_combos()
        self._refresh_all_widgets()

    def set_roi(self, mark_id: str, layer: str, roi: Roi, source: str = "manual"):
        if hasattr(self, "three_point_circle_btn") and self.three_point_circle_btn.isChecked():
            self.three_point_circle_btn.blockSignals(True)
            self.three_point_circle_btn.setChecked(False)
            self.three_point_circle_btn.blockSignals(False)
        if mark_id not in {"Mark1", "Mark2"}:
            return
        if mark_id not in self.marks:
            self.marks[mark_id] = MarkRecipe(mark_id)
        mark = self.marks[mark_id]
        if roi is not None:
            roi = self._coerce_roi_to_auto_ring(roi, layer)
        if layer == "upper":
            mark.upper_roi = roi
        else:
            mark.lower_roi = roi
        self.roi_sources.setdefault(mark_id, {})[layer] = source if roi is not None else "none"
        self._recipe_roi_confirmation_signature = None
        # Clear outdated detection for that layer.
        if mark_id in self.detections and layer in self.detections[mark_id]:
            del self.detections[mark_id][layer]
        if mark_id in self.overlays:
            del self.overlays[mark_id]
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
            self.inner_ratio_spin.setValue(float(getattr(roi, "inner_ratio", 0.60)))
            self.roi_angle_spin.setValue(float(getattr(roi, "angle_deg", 0.0)))
            for widget in widgets:
                widget.blockSignals(False)
        self._refresh_all_widgets()

    def clear_current_roi(self):
        mark_id = self._current_mark_id()
        layer = self._current_layer()
        self.set_roi(mark_id, layer, None)
        self._append_log(f"已清除 {mark_id} {LAYER_LABELS.get(layer, layer)} ROI。")

    def clear_all_recipe_rois(self):
        cleared = []
        for mark_id in ("Mark1", "Mark2"):
            mark = self.marks.get(mark_id)
            if mark is None:
                continue
            for layer in ("upper", "lower"):
                if self._roi_source(mark_id, layer) != "recipe":
                    continue
                if layer == "upper":
                    mark.upper_roi = None
                else:
                    mark.lower_roi = None
                self.roi_sources.setdefault(mark_id, {})[layer] = "none"
                cleared.append(f"{mark_id}-{LAYER_LABELS.get(layer, layer)}")
        self._recipe_roi_confirmation_signature = None
        if cleared:
            self.detections.clear()
            self.overlays.clear()
            self._append_log("已清除配方 ROI：" + "、".join(cleared))
        else:
            self._append_log("当前没有配方来源的 ROI。")
        self._refresh_all_widgets()

    def _image_for_layer(self, layer: str, mark_id: Optional[str] = None) -> Optional[ImageData]:
        mark_id = mark_id or self._current_mark_id()
        self._ensure_mark_runtime(mark_id)
        if self._current_mode() == "Single Image":
            return self.mark_images[mark_id]["upper"]
        return self.mark_images[mark_id][layer]

    def _detect_one(self, mark: MarkRecipe, layer: str) -> DetectionResult:
        self._pull_config_from_ui()
        img = self._image_for_layer(layer, mark.mark_id)
        if img is None:
            raise ValueError(f"{LAYER_LABELS[layer]} 图像未导入")
        roi = mark.upper_roi if layer == "upper" else mark.lower_roi
        if roi is None:
            raise ValueError(f"{mark.mark_id} {LAYER_LABELS[layer]} ROI 未设置")
        roi = self._coerce_roi_to_auto_ring(roi, layer)
        return detect_manual_roi(mark.mark_id, layer, img, roi, self.params, self.config)

    def analyze_current_mark(self):
        mark_id = self.mark_combo.currentText()
        if not mark_id:
            return
        self._analyze_mark(mark_id)

    def analyze_current_roi(self):
        # Backward-compatible entry. V1.2.5 uses batch ROI-region analysis.
        return self.analyze_roi_regions()

    def analyze_roi_regions(self, show_message: bool = True):
        """Analyze every ROI region already set for the current Mark.

        Manual ROI measurement should not require switching between upper/lower layers and
        clicking analyze repeatedly. This method detects all available ROI regions for the
        current Mark, refreshes reference/target selectors, and leaves overlay calculation
        to the single top toolbar button.
        """
        mark_id = self.mark_combo.currentText() or self._current_mark_id()
        if not mark_id:
            return 0
        mark = self.marks[mark_id]
        layers_to_analyze = []
        for layer in ("upper", "lower"):
            roi = mark.upper_roi if layer == "upper" else mark.lower_roi
            image = self._image_for_layer(layer, mark_id)
            if roi is not None and image is not None:
                layers_to_analyze.append(layer)
        if not layers_to_analyze:
            if show_message:
                QMessageBox.warning(self, "分析 ROI 区域", "当前 Mark 没有可分析的 ROI 区域。请先导入图像并框选 ROI。")
            return 0

        analyzed = []
        errors = []
        self._pull_config_from_ui()
        for layer in layers_to_analyze:
            try:
                det = self._detect_one(mark, layer)
                self.detections.setdefault(mark_id, {})[layer] = det
                analyzed.append((layer, det))
            except Exception as exc:
                errors.append(f"{LAYER_LABELS.get(layer, layer)}：{self._friendly_error(exc)}")

        if mark_id in self.overlays:
            del self.overlays[mark_id]
        self._refresh_auto_selection_combos()
        self._refresh_all_widgets()

        if show_message:
            if analyzed:
                lines = []
                for layer, det in analyzed:
                    radius_px = det.shape_params.get("radius_px")
                    if radius_px is not None:
                        detail = f"半径={det.diameter_um / 2.0:.3f} μm"
                    else:
                        detail = f"尺寸={det.diameter_um:.3f} μm"
                    lines.append(f"{mark_id} {LAYER_LABELS.get(layer, layer)}：中心=({det.center_x_um:.3f}, {det.center_y_um:.3f}) μm，{detail}")
                if errors:
                    lines.append("\n以下 ROI 未完成：")
                    lines.extend(errors)
                QMessageBox.information(self, "ROI 区域分析完成", "\n".join(lines))
            else:
                QMessageBox.warning(self, "分析 ROI 区域", "ROI 区域分析失败：\n" + "\n".join(errors))
        return len(analyzed)

    def _recipe_roi_usage(self) -> list[str]:
        if self._is_auto_workflow():
            return []
        usage = []
        for mark_id in ("Mark1", "Mark2"):
            images = self.batch_images.get(mark_id, {}) if self._is_batch_mode() else self.mark_images.get(mark_id, {})
            has_upper = bool(images.get("upper"))
            if not has_upper:
                continue
            for layer in ("upper", "lower"):
                roi = getattr(self.marks.get(mark_id), f"{layer}_roi", None)
                if roi is not None and self._roi_source(mark_id, layer) == "recipe":
                    usage.append(f"{mark_id} {LAYER_LABELS.get(layer, layer)}")
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
            "mark_images": {mark_id: dict(self.mark_images[mark_id]) for mark_id in ("Mark1", "Mark2")},
            "batch_images": {
                mark_id: {layer: list(self.batch_images[mark_id][layer]) for layer in ("upper", "lower")}
                for mark_id in ("Mark1", "Mark2")
            },
            "selections": deepcopy(self.auto_selections),
            "batch": self._is_batch_mode(),
            "roi_sources": deepcopy(self.roi_sources),
        }

    def _set_calculation_running(self, running: bool):
        self._calculation_running = bool(running)
        self.progress_bar.setVisible(running)
        self.progress_stage_label.setVisible(running)
        self.cancel_progress_btn.setVisible(running)
        if running:
            self.progress_bar.setValue(0)
        for button in (
            self.import_upper_btn, self.import_lower_btn, self.load_recipe_btn,
            self.save_recipe_btn, self.analyze_all_btn, self.export_btn,
            self.analyze_roi_btn, self.auto_detect_btn, self.reset_measurement_btn,
        ):
            button.setEnabled(not running)
        self.side_tabs.setEnabled(not running)

    def _start_measurement_job(self):
        if self._calculation_running:
            return
        self._pull_config_from_ui()
        if not self._confirm_recipe_rois():
            self._append_log("已取消计算；配方 ROI 未确认。")
            return
        job = self._calculation_job_snapshot()
        self._set_calculation_running(True)
        self.progress_stage_label.setText("正在准备后台计算")
        self._append_log(
            "计算路径：全图自动识别（不使用 ROI）"
            if self._is_auto_workflow()
            else "计算路径：手动 ROI；ROI 来源已锁定到本次任务快照"
        )
        thread = QThread(self)
        worker = MeasurementWorker(job)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_calculation_progress)
        worker.finished.connect(self._on_calculation_completed)
        worker.failed.connect(self._on_calculation_failed)
        worker.cancelled.connect(self._on_calculation_cancelled)
        for signal in (worker.finished, worker.failed, worker.cancelled):
            signal.connect(thread.quit)
            signal.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_calculation_thread_finished)
        self._calculation_thread = thread
        self._calculation_worker = worker
        thread.start()

    def analyze_all_marks(self):
        return self._start_measurement_job()

    def cancel_calculation(self):
        if self._calculation_worker is not None:
            self._calculation_worker.cancel()
            self.cancel_progress_btn.setEnabled(False)
            self.progress_stage_label.setText("正在取消，当前算法步骤完成后停止")

    @Slot(int, int, str)
    def _on_calculation_progress(self, done: int, total: int, message: str):
        self.progress_bar.setValue(int(round(100.0 * done / max(1, total))))
        self.progress_stage_label.setText(message)
        self.statusBar().showMessage(message)

    @Slot(object)
    def _on_calculation_completed(self, payload: dict):
        self.detections = payload.get("detections", {})
        self.overlays = payload.get("overlays", {})
        self.auto_candidates_by_mark = payload.get("auto_candidates", {"Mark1": {}, "Mark2": {}})
        self.auto_detections_by_mark = payload.get("auto_detections", {"Mark1": {}, "Mark2": {}})
        self.auto_overlays = payload.get("auto_overlays", {})
        self.auto_selections = payload.get("selections", self.auto_selections)
        self.batch_overlays = payload.get("batch_overlays", {"Mark1": [], "Mark2": []})
        self.batch_run_records = payload.get("batch_records", {"Mark1": [], "Mark2": []})
        self._refresh_auto_selection_combos()
        self._refresh_all_widgets()
        if payload.get("batch") and hasattr(self, "result_tabs"):
            self.result_tabs.setCurrentIndex(2)
        result_map = self.auto_overlays if self._is_auto_workflow() else self.overlays
        lines = [
            f"{mark_id}: Dx={item.delta_x_um:+.3f} μm，Dy={item.delta_y_um:+.3f} μm，Dxy={item.overlay_r_um:.3f} μm"
            for mark_id, item in result_map.items()
        ]
        notes = list(payload.get("warnings", [])) + list(payload.get("skipped", []))
        if lines:
            message = "计算完成：\n" + "\n".join(lines)
            if notes:
                message += "\n\n提示：\n" + "\n".join(notes)
            QMessageBox.information(self, "计算完成", message)
        else:
            QMessageBox.warning(self, "计算对位偏差", "未生成任何对位结果。\n" + "\n".join(notes))

    @Slot(str)
    def _on_calculation_failed(self, message: str):
        QMessageBox.critical(self, "计算失败", self._friendly_error(Exception(message)))

    @Slot()
    def _on_calculation_cancelled(self):
        self._append_log("计算已取消。")

    @Slot()
    def _on_calculation_thread_finished(self):
        self._calculation_worker = None
        self._calculation_thread = None
        self.cancel_progress_btn.setEnabled(True)
        self._set_calculation_running(False)
        self._refresh_all_widgets()

    def closeEvent(self, event):
        thread = self._calculation_thread
        if thread is not None and thread.isRunning():
            if self._calculation_worker is not None:
                self._calculation_worker.cancel()
            if not thread.wait(5000):
                event.ignore()
                self._append_log("后台计算仍在停止，请稍候后再次关闭。")
                return
        super().closeEvent(event)

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
        if not display_detections:
            QMessageBox.warning(self, "无结果", "当前没有可导出的分析结果。")
            return
        self._pull_config_from_ui()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出结果",
            self._default_export_filename(),
            "Excel (*.xlsx);;CSV (*.csv)",
        )
        if not path:
            return
        try:
            rows = []
            if self._is_auto_workflow():
                reference_names = []
                target_names = []
                for mark_id, detected in self.auto_detections_by_mark.items():
                    selection = self.auto_selections.get(mark_id, {})
                    reference = selection.get("reference_label", "")
                    target = selection.get("target_label", "")
                    if reference:
                        reference_names.append(f"{mark_id}-{reference}")
                    if target:
                        target_names.append(f"{mark_id}-{target}")
                    named = {f"{mark_id}-{label}": layer_map for label, layer_map in detected.items()}
                    row_overlays = {}
                    if target and mark_id in self.auto_overlays:
                        row_overlays[f"{mark_id}-{target}"] = self.auto_overlays[mark_id]
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
                for mark_id, layer_map in self.detections.items():
                    upper = self._image_for_layer("upper", mark_id)
                    lower = self._image_for_layer("lower", mark_id) if self._current_mode() == "Dual Image" else None
                    rows.extend(build_detection_rows(
                        {mark_id: layer_map},
                        {mark_id: self.overlays[mark_id]} if mark_id in self.overlays else {},
                        self.config,
                        upper_file=upper.path if upper else "",
                        lower_file=lower.path if lower else "",
                    ))
            with TemporaryDirectory() as tmp_dir:
                mark_images = self._build_mark_image_exports(tmp_dir)
                export_results(
                    path,
                    rows,
                    config=self.config,
                    summary_rows=self._build_summary_rows(),
                    mark_images=mark_images,
                    repeatability_rows=self._build_repeatability_export_rows(),
                )
            QMessageBox.information(self, "导出完成", f"结果已导出：\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def save_recipe_file(self):
        self._pull_config_from_ui()
        path, _ = QFileDialog.getSaveFileName(self, "保存配方", "overlay_recipe.json", "JSON (*.json)")
        if not path:
            return
        try:
            save_recipe(path, self.config, self.params, list(self.marks.values()))
            self.loaded_recipe_path = path
            self.loaded_recipe_display_name = self.config.recipe_name.strip() or Path(path).stem
            QMessageBox.information(self, "保存完成", f"配方已保存：\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))

    def load_recipe_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "加载配方", "", "JSON (*.json)")
        if not path:
            return
        try:
            config, params, marks = load_recipe(path)
            self.config = config
            if not getattr(self.config, "recipe_name", "").strip():
                self.config.recipe_name = Path(path).stem
            self.params = params
            loaded_marks = {m.mark_id: m for m in marks if m.mark_id in {"Mark1", "Mark2"}}
            self.marks = {
                mark_id: loaded_marks.get(mark_id, MarkRecipe(mark_id))
                for mark_id in ("Mark1", "Mark2")
            }
            self.roi_sources = self._empty_roi_sources()
            for mark_id, mark in self.marks.items():
                if mark.upper_roi is not None:
                    self.roi_sources[mark_id]["upper"] = "recipe"
                if mark.lower_roi is not None:
                    self.roi_sources[mark_id]["lower"] = "recipe"
            self.loaded_recipe_path = path
            self.loaded_recipe_display_name = self.config.recipe_name.strip() or Path(path).stem
            self._recipe_roi_confirmation_signature = None
            # 加载配方只更新产品信息、ROI、算法参数和判定规格，不重置已导入图像。
            # 这样用户可以先导入图像，再加载同类型物料的配方继续测量。
            for runtime_mark in ("Mark1", "Mark2"):
                self._ensure_mark_runtime(runtime_mark)
            self.detections.clear()
            self.overlays.clear()
            self.auto_detections_by_mark = {"Mark1": {}, "Mark2": {}}
            self.auto_candidates_by_mark = {"Mark1": {}, "Mark2": {}}
            self.auto_selections = {
                "Mark1": {"reference_label": getattr(self.config, "auto_reference_label", ""), "target_label": getattr(self.config, "auto_target_label", "")},
                "Mark2": {"reference_label": "", "target_label": ""},
            }
            self.auto_overlays.clear()
            self.batch_overlays = {"Mark1": [], "Mark2": []}
            self.batch_run_records = {"Mark1": [], "Mark2": []}
            self._sync_current_mark_images()
            self._push_config_to_ui()
            self._push_current_auto_match_rules()
            self._refresh_auto_selection_combos()
            self._refresh_all_widgets()
            QMessageBox.information(self, "加载完成", f"配方已加载：\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))


def run_app():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
