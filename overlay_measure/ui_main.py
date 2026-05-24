from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QFontDatabase, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
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
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .auto_mark_detector import detect_auto_marks
from .caliper_circle_detector import detect_caliper_circle
from .circle_ellipse_fitter import fit_mark_shape
from .image_loader import load_image, normalize_to_uint8
from .models import DetectionParams, DetectionResult, ImageData, MarkRecipe, MeasurementConfig, Roi
from .overlay_calculator import calculate_overlay, calculate_relative_overlay
from .production_measurement import refine_candidate
from .recipe_manager import load_recipe, save_recipe
from .result_exporter import build_detection_rows, export_results
from .subpixel_edge_detector import detect_subpixel_edges


LAYER_LABELS = {"upper": "上层", "lower": "下层"}


class SidebarComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class SidebarDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class SidebarSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


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
        self.setText(f"{title}\n未导入图像")
        self.setStyleSheet("QLabel { background: #222; color: #eee; border: 1px solid #555; }")

    def set_image(self, image: Optional[ImageData]):
        self.image = image
        self.pixmap_cache = None
        self.reset_view(update=False)
        if image is not None:
            self.pixmap_cache = self._make_pixmap(image.gray)
            self.setText("")
        else:
            self.setText(f"{self.title}\n未导入图像")
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

    def _make_pixmap(self, gray: np.ndarray) -> QPixmap:
        u8 = normalize_to_uint8(gray)
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
        painter.fillRect(self.rect(), QColor(34, 34, 34))

        if self.pixmap_cache is None:
            painter.setPen(QColor(230, 230, 230))
            painter.drawText(self.rect(), Qt.AlignCenter, f"{self.title}\n未导入图像")
            painter.end()
            return

        self._update_transform()
        target = QRectF(self.offset_x, self.offset_y, self.pixmap_cache.width() * self.scale, self.pixmap_cache.height() * self.scale)
        painter.drawPixmap(target, self.pixmap_cache, QRectF(self.pixmap_cache.rect()))

        self._draw_overlays(painter)

        painter.setPen(QColor(240, 240, 240))
        painter.drawText(10, 20, f"{self.title}  缩放: {self.user_zoom:.2f}x")
        painter.drawText(10, 40, "滚轮缩放；右键/中键拖动画面；双击复位；左键拖 ROI，拖内/外环调范围")
        painter.end()

    def _draw_overlays(self, painter: QPainter):
        if self.image is None:
            return
        colors = {
            "upper": QColor(0, 220, 255),
            "lower": QColor(255, 180, 0),
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
                    painter.setPen(QPen(QColor(0, 255, 80), 1.0))
                    pts = det.edge_points
                    if pts:
                        step = max(1, len(pts) // 1200)
                        for px, py in pts[::step]:
                            wx, wy = self.image_to_widget(px, py)
                            painter.drawEllipse(QRectF(wx - 2.0, wy - 2.0, 4.0, 4.0))
                    rejected = getattr(det, "rejected_points", [])
                    if rejected:
                        painter.setPen(QPen(QColor(255, 60, 60), 1.4))
                        for px, py in rejected:
                            wx, wy = self.image_to_widget(px, py)
                            painter.drawLine(int(wx - 3), int(wy - 3), int(wx + 3), int(wy + 3))
                            painter.drawLine(int(wx - 3), int(wy + 3), int(wx + 3), int(wy - 3))
                    cx, cy = self.image_to_widget(det.center_x_px, det.center_y_px)
                    painter.setPen(QPen(QColor(0, 255, 80), 2.2))
                    painter.drawLine(int(cx - 8), int(cy), int(cx + 8), int(cy))
                    painter.drawLine(int(cx), int(cy - 8), int(cx), int(cy + 8))
                    contour_label = self.manual_labels.get((mark_id, layer), "")
                    if contour_label:
                        label_x, label_y = self._contour_label_anchor(det)
                        painter.drawText(int(label_x + 6), int(label_y - 6), f"{contour_label} ({mark_id})")
                    if det.fitting_mode in {"Circle", "EdgeCenter", "CaliperCircle"} and "radius_px" in det.shape_params:
                        rad = det.shape_params["radius_px"] * self.scale
                        painter.drawEllipse(QRectF(cx - rad, cy - rad, 2 * rad, 2 * rad))
                        if det.fitting_mode == "CaliperCircle":
                            radius_um = det.shape_params.get("radius_px", 0) * self._mean_pixel_size_um()
                            painter.drawText(
                                int(cx + 12),
                                int(cy - 12),
                                f"中心=({det.center_x_um:.4f},{det.center_y_um:.4f}) μm 半径={radius_um:.4f} μm 残差={det.residual_um:.4f} μm 置信度={det.confidence:.3f}",
                            )
                    elif det.fitting_mode == "Ellipse":
                        major = det.shape_params.get("major_px", det.diameter_px) * self.scale
                        minor = det.shape_params.get("minor_px", det.diameter_px) * self.scale
                        # For V1 display, draw axis-aligned ellipse; angle is reported numerically in table.
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

                        # White shadow line for contrast, then colored fit line.
                        shadow_pen = QPen(QColor(255, 255, 255), 5.0)
                        shadow_pen.setCosmetic(True)
                        fit_pen = QPen(colors[layer], 2.8)
                        fit_pen.setCosmetic(True)
                        for pen in (shadow_pen, fit_pen):
                            painter.setPen(pen)
                            for i in range(4):
                                x0, y0 = qpoints[i]
                                x1, y1 = qpoints[(i + 1) % 4]
                                painter.drawLine(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))

                        # Corner handles make it obvious this is the fitted square/rectangle.
                        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
                        for wx, wy in qpoints:
                            painter.drawRect(QRectF(wx - 3.5, wy - 3.5, 7.0, 7.0))
                        painter.setPen(QPen(colors[layer], 1.5))
                        for wx, wy in qpoints:
                            painter.drawRect(QRectF(wx - 2.5, wy - 2.5, 5.0, 5.0))

                        # Draw the fit parameter label close to the contour.
                        label_x = int(round(min(x for x, _ in qpoints)))
                        label_y = int(round(min(y for _, y in qpoints))) - 6
                        painter.setPen(QColor(255, 255, 255))
                        painter.drawText(
                            label_x,
                            label_y,
                            f"矩形 W={width * self.pixel_size_x_um:.4f} μm H={height * self.pixel_size_y_um:.4f} μm 角度={angle_deg:.1f}° 残差={det.residual_um:.4f} μm",
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
                    painter.setPen(QPen(QColor(0, 255, 80), 1.0))
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
                                    f"坐标=({px * self.pixel_size_x_um:.4f}, {py * self.pixel_size_y_um:.4f}) μm\n梯度={grad_txt}"
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        for font_path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
            if font_path.exists() and QFontDatabase.addApplicationFont(str(font_path)) >= 0:
                break
        self.setFont(QFont("Microsoft YaHei UI", 9))
        self.setWindowTitle("对位偏差测量软件 V1.0.5")
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

        self._build_ui()
        self._connect_actions()
        self._refresh_all_widgets()

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        self.setCentralWidget(central)

        toolbar = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["单图模式", "双图模式"])
        self.import_upper_btn = QPushButton("导入上层/单张图像")
        self.import_lower_btn = QPushButton("导入下层图像")
        self.reset_measurement_btn = QPushButton("重置 ROI / 结果")
        self.zoom_in_btn = QPushButton("放大")
        self.zoom_out_btn = QPushButton("缩小")
        self.reset_view_btn = QPushButton("视图复位")
        self.analyze_roi_btn = QPushButton("分析当前 ROI")
        self.analyze_current_btn = QPushButton("计算当前对位")
        self.analyze_all_btn = QPushButton("计算全部对位")
        self.save_recipe_btn = QPushButton("保存配方")
        self.load_recipe_btn = QPushButton("加载配方")
        self.export_btn = QPushButton("导出 CSV/Excel")

        toolbar.addWidget(QLabel("测量模式："))
        toolbar.addWidget(self.mode_combo)
        toolbar.addWidget(self.import_upper_btn)
        toolbar.addWidget(self.import_lower_btn)
        toolbar.addWidget(self.reset_measurement_btn)
        toolbar.addWidget(self.zoom_in_btn)
        toolbar.addWidget(self.zoom_out_btn)
        toolbar.addWidget(self.reset_view_btn)
        toolbar.addStretch(1)
        toolbar.addWidget(self.analyze_roi_btn)
        toolbar.addWidget(self.analyze_current_btn)
        toolbar.addWidget(self.analyze_all_btn)
        toolbar.addWidget(self.save_recipe_btn)
        toolbar.addWidget(self.load_recipe_btn)
        toolbar.addWidget(self.export_btn)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, stretch=3)

        image_panel = QWidget()
        image_layout = QHBoxLayout(image_panel)
        self.upper_canvas = ImageCanvas("上层 / 单张图像", fixed_layer=None)
        self.lower_canvas = ImageCanvas("下层图像", fixed_layer="lower")
        image_layout.addWidget(self.upper_canvas, stretch=1)
        image_layout.addWidget(self.lower_canvas, stretch=1)
        splitter.addWidget(image_panel)

        side_tabs = QTabWidget()
        for page, title in (
            (self._build_basic_tab(), "基础参数"),
            (self._build_algo_tab(), "亚像素算法"),
            (self._build_spec_tab(), "判定规格"),
        ):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            side_tabs.addTab(scroll, title)
        splitter.addWidget(side_tabs)
        splitter.setSizes([1050, 430])

        tables = QSplitter(Qt.Vertical)
        self.det_table = QTableWidget()
        self.overlay_table = QTableWidget()
        tables.addWidget(self.det_table)
        tables.addWidget(self.overlay_table)
        tables.setMinimumHeight(220)
        tables.setSizes([140, 110])
        root.addWidget(tables, stretch=2)

    def _build_basic_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        group = QGroupBox("标记 / ROI")
        form = QFormLayout(group)
        self.mark_combo = SidebarComboBox()
        self.layer_combo = SidebarComboBox()
        self.layer_combo.addItem("上层", "upper")
        self.layer_combo.addItem("下层", "lower")
        form.addRow("当前标记", self.mark_combo)
        form.addRow("当前层", self.layer_combo)
        layout.addWidget(group)

        group_auto = QGroupBox("识别流程 / 自动对位")
        form_auto = QFormLayout(group_auto)
        self.workflow_combo = SidebarComboBox()
        self.workflow_combo.addItem("手动 ROI 测量", "Manual")
        self.workflow_combo.addItem("自动识别测量", "Auto")
        self.auto_detect_btn = QPushButton("自动识别当前 Mark")
        self.auto_reference_combo = SidebarComboBox()
        self.auto_target_combo = SidebarComboBox()
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
        self.auto_calculate_btn = QPushButton("自动计算对位偏差")
        self.diagnostic_check = QCheckBox("显示诊断信息（原始轮廓 / 边缘点）")
        self.production_status_label = QLabel("自动模式：正式精测结果")
        form_auto.addRow("工作方式", self.workflow_combo)
        form_auto.addRow(self.auto_detect_btn)
        form_auto.addRow("当前 Mark 基准轮廓", self.auto_reference_combo)
        form_auto.addRow("当前 Mark 待测轮廓", self.auto_target_combo)
        form_auto.addRow("基准预期外形", self.auto_ref_shape_combo)
        form_auto.addRow("基准尺寸下限 (μm)", self.auto_ref_size_min_spin)
        form_auto.addRow("基准尺寸上限 (μm)", self.auto_ref_size_max_spin)
        form_auto.addRow("待测预期外形", self.auto_target_shape_combo)
        form_auto.addRow("待测尺寸下限 (μm)", self.auto_target_size_min_spin)
        form_auto.addRow("待测尺寸上限 (μm)", self.auto_target_size_max_spin)
        form_auto.addRow(self.auto_calculate_btn)
        form_auto.addRow(self.diagnostic_check)
        form_auto.addRow(self.production_status_label)
        layout.addWidget(group_auto)

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

        group_fit = QGroupBox("常用拟合设置")
        form_fit = QFormLayout(group_fit)
        self.fit_mode_combo = SidebarComboBox()
        self.fit_mode_combo.addItem("边缘中心（默认）", "EdgeCenter")
        self.fit_mode_combo.addItem("自动", "Auto")
        self.fit_mode_combo.addItem("圆", "Circle")
        self.fit_mode_combo.addItem("椭圆", "Ellipse")
        self.fit_mode_combo.addItem("矩形/方孔", "Rectangle")
        self.upper_fit_mode_combo = SidebarComboBox()
        self.upper_fit_mode_combo.addItem("边缘中心（默认）", "EdgeCenter")
        self.upper_fit_mode_combo.addItem("自动", "Auto")
        self.upper_fit_mode_combo.addItem("圆", "Circle")
        self.upper_fit_mode_combo.addItem("椭圆", "Ellipse")
        self.upper_fit_mode_combo.addItem("矩形/方孔", "Rectangle")
        self.lower_fit_mode_combo = SidebarComboBox()
        self.lower_fit_mode_combo.addItem("边缘中心（默认）", "EdgeCenter")
        self.lower_fit_mode_combo.addItem("自动", "Auto")
        self.lower_fit_mode_combo.addItem("圆", "Circle")
        self.lower_fit_mode_combo.addItem("椭圆", "Ellipse")
        self.lower_fit_mode_combo.addItem("矩形/方孔", "Rectangle")
        form_fit.addRow("默认拟合模式", self.fit_mode_combo)
        form_fit.addRow("上层拟合模式", self.upper_fit_mode_combo)
        form_fit.addRow("下层拟合模式", self.lower_fit_mode_combo)
        layout.addWidget(group_fit)

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
        form_rz.addRow("|Rz| 上限", self.rz_limit_spin)
        layout.addWidget(group_rz)

        group_roi = QGroupBox("环形 ROI")
        form_roi = QFormLayout(group_roi)
        self.roi_type_combo = SidebarComboBox()
        self.roi_type_combo.addItem("卡尺找圆 ROI", "Caliper Circle")
        self.roi_type_combo.addItem("圆环 ROI", "Annulus")
        self.roi_type_combo.addItem("矩形 ROI", "Rectangular Ring")
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
        form_roi.addRow("ROI 外形", self.roi_type_combo)
        form_roi.addRow("中心 X", self.center_x_spin)
        form_roi.addRow("中心 Y", self.center_y_spin)
        form_roi.addRow("内半径", self.inner_radius_spin)
        form_roi.addRow("外半径", self.outer_radius_spin)
        form_roi.addRow("卡尺数量", self.caliper_count_spin)
        form_roi.addRow("卡尺宽度", self.caliper_width_spin)
        form_roi.addRow("搜索方向", self.search_direction_combo)
        form_roi.addRow("边缘选择", self.target_edge_combo)
        form_roi.addRow("矩形环角度 (deg)", self.roi_angle_spin)
        form_roi.addRow(self.three_point_circle_btn)
        form_roi.addRow(self.apply_roi_params_btn)
        layout.addWidget(group_roi)

        group2 = QGroupBox("像素尺寸 / 双图配准")
        form2 = QFormLayout(group2)
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
        form2.addRow("像素尺寸 X (μm/px)", self.pixel_x_spin)
        form2.addRow("像素尺寸 Y (μm/px)", self.pixel_y_spin)
        form2.addRow("双图配准偏移 X (μm)", self.offset_x_spin)
        form2.addRow("双图配准偏移 Y (μm)", self.offset_y_spin)
        layout.addWidget(group2)

        hint = QLabel(
            "操作提示：\n"
            "1. 选择当前标记和层。\n"
            "2. 左键拖出 ROI 外边界。\n"
            "3. 卡尺找圆 ROI 会显示卡尺窗口和搜索方向箭头。\n"
            "4. “三点定圆”用于确定圆环宽度中心，内外圆按当前半宽展开。\n"
            "5. 拖内/外环可调宽度，拖 ROI 中间可整体移动，右键 ROI 可删除。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return w

    def _build_algo_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        group = QGroupBox("亚像素边缘参数")
        form = QFormLayout(group)

        self.sigma_spin = SidebarDoubleSpinBox()
        self.sigma_spin.setRange(0, 10)
        self.sigma_spin.setDecimals(3)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(1.0)

        self.canny_low_spin = SidebarDoubleSpinBox()
        self.canny_high_spin = SidebarDoubleSpinBox()
        for spin, val in [(self.canny_low_spin, 40), (self.canny_high_spin, 120)]:
            spin.setRange(0, 255)
            spin.setDecimals(1)
            spin.setSingleStep(5)
            spin.setValue(val)

        self.min_gradient_spin = SidebarDoubleSpinBox()
        self.min_gradient_spin.setRange(0, 1000000)
        self.min_gradient_spin.setDecimals(3)
        self.min_gradient_spin.setSingleStep(1)
        self.min_gradient_spin.setValue(5.0)

        self.profile_half_spin = SidebarDoubleSpinBox()
        self.profile_half_spin.setRange(0.5, 10)
        self.profile_half_spin.setDecimals(2)
        self.profile_half_spin.setSingleStep(0.25)
        self.profile_half_spin.setValue(2.0)

        self.profile_step_spin = SidebarDoubleSpinBox()
        self.profile_step_spin.setRange(0.05, 2)
        self.profile_step_spin.setDecimals(3)
        self.profile_step_spin.setSingleStep(0.05)
        self.profile_step_spin.setValue(0.25)

        self.ransac_check = QCheckBox("启用 RANSAC 异常点剔除")
        self.ransac_check.setChecked(True)

        self.residual_limit_spin = SidebarDoubleSpinBox()
        self.residual_limit_spin.setRange(0.001, 1000)
        self.residual_limit_spin.setDecimals(4)
        self.residual_limit_spin.setSingleStep(0.05)
        self.residual_limit_spin.setValue(2.0)

        self.min_edge_points_spin = SidebarSpinBox()
        self.min_edge_points_spin.setRange(3, 1000000)
        self.min_edge_points_spin.setValue(60)

        self.polarity_combo = SidebarComboBox()
        self.polarity_combo.addItem("自动", "Auto")
        self.polarity_combo.addItem("暗到亮", "Dark to Bright")
        self.polarity_combo.addItem("亮到暗", "Bright to Dark")

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
        layout.addWidget(group)

        production_group = QGroupBox("自动正式精测 / 质量门槛")
        production_form = QFormLayout(production_group)
        self.production_search_spin = SidebarDoubleSpinBox()
        self.production_search_spin.setRange(2.0, 1000.0)
        self.production_search_spin.setDecimals(3)
        self.production_search_spin.setValue(8.0)
        self.production_coverage_spin = SidebarDoubleSpinBox()
        self.production_coverage_spin.setRange(0.0, 1.0)
        self.production_coverage_spin.setDecimals(3)
        self.production_coverage_spin.setValue(0.65)
        self.production_reject_spin = SidebarDoubleSpinBox()
        self.production_reject_spin.setRange(0.0, 1.0)
        self.production_reject_spin.setDecimals(3)
        self.production_reject_spin.setValue(0.40)
        self.production_residual_spin = SidebarDoubleSpinBox()
        self.production_residual_spin.setRange(0.000001, 1000000.0)
        self.production_residual_spin.setDecimals(6)
        self.production_residual_spin.setValue(0.30)
        self.production_deviation_spin = SidebarDoubleSpinBox()
        self.production_deviation_spin.setRange(0.000001, 1000000.0)
        self.production_deviation_spin.setDecimals(6)
        self.production_deviation_spin.setValue(0.60)
        production_form.addRow("自动搜索半宽 (px)", self.production_search_spin)
        production_form.addRow("最低覆盖率", self.production_coverage_spin)
        production_form.addRow("最大异常点比例", self.production_reject_spin)
        production_form.addRow("最大残差 (μm)", self.production_residual_spin)
        production_form.addRow("最大轮廓偏差 (μm)", self.production_deviation_spin)
        layout.addWidget(production_group)
        layout.addStretch(1)
        return w

    def _build_spec_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        group = QGroupBox("判定规格")
        form = QFormLayout(group)
        self.dx_limit_spin = SidebarDoubleSpinBox()
        self.dy_limit_spin = SidebarDoubleSpinBox()
        self.r_limit_spin = SidebarDoubleSpinBox()
        self.conf_min_spin = SidebarDoubleSpinBox()
        for spin, val in [(self.dx_limit_spin, 0.5), (self.dy_limit_spin, 0.5), (self.r_limit_spin, 0.7)]:
            spin.setRange(0, 1000000)
            spin.setDecimals(6)
            spin.setSingleStep(0.1)
            spin.setValue(val)
        self.conf_min_spin.setRange(0, 1)
        self.conf_min_spin.setDecimals(3)
        self.conf_min_spin.setSingleStep(0.05)
        self.conf_min_spin.setValue(0.7)
        form.addRow("|ΔX| 上限 (μm)", self.dx_limit_spin)
        form.addRow("|ΔY| 上限 (μm)", self.dy_limit_spin)
        form.addRow("对位 R 上限 (μm)", self.r_limit_spin)
        form.addRow("最低置信度", self.conf_min_spin)
        layout.addWidget(group)
        layout.addStretch(1)
        return w

    def _connect_actions(self):
        self.import_upper_btn.clicked.connect(self.import_upper_image)
        self.import_lower_btn.clicked.connect(self.import_lower_image)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
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
        self.upper_canvas.roiChanged.connect(self.set_roi)
        self.lower_canvas.roiChanged.connect(self.set_roi)
        self.analyze_roi_btn.clicked.connect(self.analyze_current_roi)
        self.analyze_current_btn.clicked.connect(self.analyze_current_mark)
        self.analyze_all_btn.clicked.connect(self.analyze_all_marks)
        self.export_btn.clicked.connect(self.export_result_file)
        self.save_recipe_btn.clicked.connect(self.save_recipe_file)
        self.load_recipe_btn.clicked.connect(self.load_recipe_file)

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
        return replace(
            roi,
            roi_type=self._auto_ring_roi_type(layer),
            caliper_count=self.caliper_count_spin.value() if hasattr(self, "caliper_count_spin") else getattr(roi, "caliper_count", 64),
            caliper_width_px=self.caliper_width_spin.value() if hasattr(self, "caliper_width_spin") else getattr(roi, "caliper_width_px", 8.0),
            search_direction=self._combo_value(self.search_direction_combo) if hasattr(self, "search_direction_combo") else getattr(roi, "search_direction", "Inner to Outer"),
        )

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
        self.import_upper_btn.setText(f"导入 {current_mark} 上层/单张图像")
        self.import_lower_btn.setText(f"导入 {current_mark} 下层图像")
        self.upper_canvas.title = f"{current_mark} 上层 / 单张图像"
        self.lower_canvas.title = f"{current_mark} 下层图像"
        if self.upper_canvas.image is None:
            self.upper_canvas.setText(f"{self.upper_canvas.title}\n未导入图像")
        if self.lower_canvas.image is None:
            self.lower_canvas.setText(f"{self.lower_canvas.title}\n未导入图像")
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
        self.import_lower_btn.setEnabled(is_dual)
        self.auto_detect_btn.setEnabled(show_auto)
        self.auto_reference_combo.setEnabled(show_auto)
        self.auto_target_combo.setEnabled(show_auto)
        auto_reference = self.auto_reference_combo.currentData()
        auto_target = self.auto_target_combo.currentData()
        self.auto_calculate_btn.setEnabled(
            show_auto and bool(auto_reference) and bool(auto_target) and auto_reference != auto_target
        )
        self.analyze_roi_btn.setEnabled(not show_auto)
        self.analyze_current_btn.setEnabled(not show_auto)
        self.analyze_all_btn.setEnabled(not show_auto)
        self.three_point_circle_btn.setEnabled(not show_auto)
        self.apply_roi_params_btn.setEnabled(not show_auto)
        self._refresh_mark_combo()
        self._refresh_tables()

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
        rows = []
        deltas = {}
        overlay_map = self.auto_overlays if self._is_auto_workflow() else self.overlays
        for mark_id in sorted(overlay_map.keys()):
            o = overlay_map[mark_id]
            idx = self._mark_number(mark_id)
            dx = o.delta_x_um
            dy = o.delta_y_um
            dxy = float(np.hypot(dx, dy))
            deltas[mark_id] = (dx, dy)
            warnings = []
            is_trial = self._is_auto_workflow() and (
                self.config.recipe_validation_status != "Validated" or o.result == "Trial"
            )
            if not is_trial and abs(dx) > self.config.delta_x_limit_um:
                warnings.append(f"Dx{idx}超限")
            if not is_trial and abs(dy) > self.config.delta_y_limit_um:
                warnings.append(f"Dy{idx}超限")
            if not is_trial and dxy > self.config.overlay_r_limit_um:
                warnings.append(f"Dxy{idx}超限")
            if o.warning and not is_trial:
                warnings.append(o.warning)
            rows.append({
                "项目": mark_id,
                f"Dx{idx}(μm)": dx,
                f"Dy{idx}(μm)": dy,
                f"Dxy{idx}(μm)": dxy,
                "判定": "试测" if is_trial else ("失败" if warnings else "通过"),
                "提示": "未验证配方，不作正式判定" if is_trial else "；".join(warnings),
            })

        if "Mark1" in deltas and "Mark2" in deltas:
            dx1, dy1 = deltas["Mark1"]
            dx2, dy2 = deltas["Mark2"]
            l_value = max(self.config.rz_distance_l_um, 1e-12)
            if self.config.rz_layout == "Y向前后分布":
                rz = (dx2 - dx1) / l_value
                formula = "Rz=(Dx2-Dx1)/L"
            else:
                rz = (dy2 - dy1) / l_value
                formula = "Rz=(Dy2-Dy1)/L"
            rows.append({
                "项目": "Rz",
                "Rz": rz,
                "公式": formula,
                "L(μm)": l_value,
                "判定": "试测" if self._is_auto_workflow() and self.config.recipe_validation_status != "Validated" else ("失败" if abs(rz) > self.config.rz_limit else "通过"),
                "提示": "未验证配方，不作正式判定" if self._is_auto_workflow() and self.config.recipe_validation_status != "Validated" else ("Rz超限" if abs(rz) > self.config.rz_limit else ""),
            })
        return rows

    def _refresh_tables(self):
        det_headers = [
            "标记", "层", "中心 X (μm)", "中心 Y (μm)",
            "尺寸/直径 (μm)", "参考残差 (μm)", "边缘点数", "置信度", "算法",
            "质量状态", "覆盖率", "形状参数", "提示",
        ]
        det_rows = []
        display_detections = self._display_detections()
        manual_labels = self._manual_detection_labels()
        mean_scale = 0.5 * (self.config.pixel_size_x_um + self.config.pixel_size_y_um)
        for mark_id, layer_map in display_detections.items():
            for layer, d in layer_map.items():
                if d.fitting_mode == "Rectangle":
                    shape_txt = (
                        f"宽={d.shape_params.get('width_px', 0) * self.config.pixel_size_x_um:.6f}μm, "
                        f"高={d.shape_params.get('height_px', 0) * self.config.pixel_size_y_um:.6f}μm, "
                        f"角度={d.shape_params.get('angle_deg', 0):.3f}°"
                    )
                elif d.fitting_mode == "Ellipse":
                    shape_txt = (
                        f"长轴={d.shape_params.get('major_px', 0) * mean_scale:.6f}μm, "
                        f"短轴={d.shape_params.get('minor_px', 0) * mean_scale:.6f}μm, "
                        f"角度={d.shape_params.get('angle_deg', 0):.3f}°"
                    )
                elif d.fitting_mode == "Circle":
                    shape_txt = f"半径={d.shape_params.get('radius_px', 0) * mean_scale:.6f}μm"
                elif d.fitting_mode == "EdgeCenter":
                    shape_txt = (
                        f"边缘中心, 参考半径={d.shape_params.get('radius_px', 0) * mean_scale:.6f}μm, "
                        f"宽={d.shape_params.get('width_px', 0) * self.config.pixel_size_x_um:.6f}μm, "
                        f"高={d.shape_params.get('height_px', 0) * self.config.pixel_size_y_um:.6f}μm"
                    )
                elif d.fitting_mode == "CaliperCircle":
                    shape_txt = (
                        f"卡尺圆, 半径={d.shape_params.get('radius_px', 0) * mean_scale:.6f}μm, "
                        f"内点={d.shape_params.get('inlier_count', 0)}, "
                        f"剔除={d.shape_params.get('rejected_count', 0)}, "
                        f"卡尺={d.shape_params.get('caliper_count', 0)}"
                    )
                elif d.fitting_mode == "ProductionRectangle":
                    shape_txt = (
                        f"方形精测, 宽={d.shape_params.get('width_px', 0) * self.config.pixel_size_x_um:.6f}μm, "
                        f"高={d.shape_params.get('height_px', 0) * self.config.pixel_size_y_um:.6f}μm, "
                        f"角度={d.shape_params.get('angle_deg', 0):.3f}°"
                    )
                elif d.fitting_mode in {"AutoCircle", "AutoRectangle", "ProductionCircle"}:
                    shape_type = "方形精测" if d.fitting_mode == "ProductionRectangle" else ("圆形精测" if d.fitting_mode == "ProductionCircle" else ("方形轮廓" if d.fitting_mode == "AutoRectangle" else "圆形轮廓"))
                    shape_txt = (
                        f"{shape_type}, 参考半径={d.shape_params.get('radius_px', 0) * mean_scale:.6f}μm, "
                        f"宽={d.shape_params.get('width_px', 0) * self.config.pixel_size_x_um:.6f}μm, "
                        f"高={d.shape_params.get('height_px', 0) * self.config.pixel_size_y_um:.6f}μm"
                    )
                else:
                    shape_txt = ""
                mode_txt = {
                    "Rectangle": "矩形",
                    "Ellipse": "椭圆",
                    "Circle": "圆",
                    "EdgeCenter": "边缘中心",
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
                    f"{d.center_x_um:.6f}", f"{d.center_y_um:.6f}",
                    f"{d.diameter_um:.6f}", f"{d.residual_um:.6f}",
                    str(d.edge_point_count), f"{d.confidence:.3f}", mode_txt,
                    {"Valid": "有效", "Invalid": "无效"}.get(d.shape_params.get("quality_status", ""), ""),
                    f"{d.shape_params.get('coverage', 0):.1%}" if "coverage" in d.shape_params else "",
                    shape_txt + "; " + roi_txt, d.warning,
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
                ov_rows.append([project, key, f"{value:.9f}" if isinstance(value, float) else value, verdict, note])
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
        crop = normalize_to_uint8(image.gray[y0:y1, x0:x1])
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
        mark = self.marks[mark_id]
        for label, layer_map in self.auto_detections_by_mark[mark_id].items():
            detection = next(iter(layer_map.values()))
            if detection.shape_params.get("quality_status") != "Valid":
                continue
            shape = "方" if detection.fitting_mode == "ProductionRectangle" else "圆"
            radius = detection.shape_params.get("radius_px", detection.diameter_px / 2.0)
            size_name = "半尺寸" if detection.fitting_mode in {"AutoRectangle", "ProductionRectangle"} else "半径"
            mean_scale = 0.5 * (self.config.pixel_size_x_um + self.config.pixel_size_y_um)
            text = f"{mark_id}-{label} - {shape} - {size_name}={radius * mean_scale:.6f} μm"
            if self._matches_auto_rule(detection, "reference", mark):
                self.auto_reference_combo.addItem(text, label)
            if self._matches_auto_rule(detection, "target", mark):
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
        results_all = []
        for layer, image in images:
            results = detect_auto_marks(
                image.gray,
                layer,
                self.params,
                self.config.pixel_size_x_um,
                self.config.pixel_size_y_um,
            )
            results_all.extend(results)
        results_all.sort(key=lambda result: -result.diameter_px)
        detected = {}
        candidates = {}
        for label_index, result in enumerate(results_all):
            label = self._alpha_label(label_index)
            result.mark_id = f"{mark_id}-{label}"
            candidates[label] = {result.layer: result}
            image = self._image_for_layer(result.layer, mark_id)
            measured = refine_candidate(image.gray, result, self.params, self.config)
            if not (self.params.diameter_min_um <= measured.diameter_um <= self.params.diameter_max_um):
                measured.shape_params["quality_status"] = "Invalid"
                measured.shape_params["failure_reason"] = "尺寸超出配方范围"
                measured.warning = "尺寸超出配方范围"
            detected[label] = {result.layer: measured}
        self.auto_candidates_by_mark[mark_id] = candidates
        self.auto_detections_by_mark[mark_id] = detected
        self.auto_overlays.pop(mark_id, None)
        self.auto_selections[mark_id] = {"reference_label": "", "target_label": ""}
        self._refresh_auto_selection_combos()
        self._refresh_all_widgets()
        if show_message:
            if detected:
                valid_count = sum(
                    next(iter(layer_map.values())).shape_params.get("quality_status") == "Valid"
                    for layer_map in detected.values()
                )
                QMessageBox.information(self, "自动精测完成", f"共发现 {len(detected)} 个候选，精测有效 {valid_count} 个。")
            else:
                QMessageBox.warning(self, "自动识别", "未找到可用的闭合 Mark 轮廓，请检查对比度或算法参数。")
        return len(detected)

    def calculate_auto_overlay(self, show_message: bool = True):
        mark_id = self._current_mark_id()
        reference_label = self.auto_reference_combo.currentData() or ""
        target_label = self.auto_target_combo.currentData() or ""
        if not reference_label or not target_label or reference_label == target_label:
            if show_message:
                QMessageBox.warning(self, "自动计算", "请选择不同的基准 Mark 和待测 Mark。")
            return None
        reference = self._find_auto_detection(mark_id, reference_label)
        target = self._find_auto_detection(mark_id, target_label)
        if reference is None or target is None:
            if show_message:
                QMessageBox.warning(self, "自动计算", "所选轮廓不存在，请重新执行自动识别。")
            return None
        invalid = [
            label
            for label, detection in ((reference_label, reference), (target_label, target))
            if detection.shape_params.get("quality_status") != "Valid"
        ]
        if invalid:
            if show_message:
                QMessageBox.warning(self, "自动计算", "所选轮廓未通过精测质量门槛，不能用于对位判定。")
            return None
        self._pull_config_from_ui()
        name = f"{mark_id}: {target_label} 相对 {reference_label}"
        overlay = calculate_relative_overlay(mark_id, reference, target, self.config)
        if self.config.recipe_validation_status != "Validated":
            overlay.result = "Trial"
            overlay.warning = "试测/未验证配方，不作正式判定"
        self.auto_overlays[mark_id] = overlay
        self.auto_selections[mark_id] = {
            "reference_label": reference_label,
            "target_label": target_label,
        }
        self._refresh_all_widgets()
        if show_message:
            QMessageBox.information(
                self,
                "自动计算完成",
                f"{name}：Dx={overlay.delta_x_um:.6f} μm，Dy={overlay.delta_y_um:.6f} μm，Dxy={overlay.overlay_r_um:.6f} μm",
            )
        return overlay

    def on_three_point_circle_toggled(self, checked: bool):
        if checked:
            self._set_combo_value(self.roi_type_combo, "Caliper Circle")
        self.upper_canvas.set_circle_pick_mode(checked)
        self.lower_canvas.set_circle_pick_mode(checked)

    def import_upper_image(self):
        mark_id = self._current_mark_id()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入上层/单张图像",
            "",
            "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;矩阵 (*.csv *.txt *.npy);;全部文件 (*)",
        )
        if not path:
            return
        try:
            self._ensure_mark_runtime(mark_id)
            self.mark_images[mark_id]["upper"] = load_image(path)
            self._invalidate_image_dependent_results(mark_id, "upper")
            self._sync_current_mark_images()
            self._refresh_auto_selection_combos()
        except Exception as exc:
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
            return
        try:
            self._ensure_mark_runtime(mark_id)
            self.mark_images[mark_id]["lower"] = load_image(path)
            self._invalidate_image_dependent_results(mark_id, "lower")
            self._sync_current_mark_images()
            self._refresh_auto_selection_combos()
        except Exception as exc:
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
        self.upper_canvas.set_circle_pick_mode(False)
        self.lower_canvas.set_circle_pick_mode(False)
        self.three_point_circle_btn.blockSignals(True)
        self.three_point_circle_btn.setChecked(False)
        self.three_point_circle_btn.blockSignals(False)
        self.mark_combo.setCurrentText("Mark1")
        self._sync_current_mark_images()
        self._refresh_auto_selection_combos()
        self._refresh_all_widgets()

    def set_roi(self, mark_id: str, layer: str, roi: Roi):
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

        if getattr(roi, "roi_type", "") == "Caliper Circle":
            cal = detect_caliper_circle(img.gray, roi, self.params)
            mean_px = 0.5 * (self.config.pixel_size_x_um + self.config.pixel_size_y_um)
            return DetectionResult(
                mark_id=mark.mark_id,
                layer=layer,
                center_x_px=cal.center_x_px,
                center_y_px=cal.center_y_px,
                center_x_um=cal.center_x_px * self.config.pixel_size_x_um,
                center_y_um=cal.center_y_px * self.config.pixel_size_y_um,
                diameter_px=2.0 * cal.radius_px,
                diameter_um=2.0 * cal.radius_px * mean_px,
                residual_px=cal.residual_px,
                residual_um=cal.residual_px * mean_px,
                edge_point_count=len(cal.edge_points),
                confidence=cal.confidence,
                fitting_mode="CaliperCircle",
                warning="",
                edge_points=[(float(x), float(y)) for x, y in cal.edge_points],
                rejected_points=[(float(x), float(y)) for x, y in cal.rejected_points],
                edge_gradients=[float(g) for g in cal.gradients],
                rejected_gradients=[float(g) for g in cal.rejected_gradients],
                shape_params={
                    "radius_px": cal.radius_px,
                    "inlier_count": int(len(cal.edge_points)),
                    "rejected_count": int(len(cal.rejected_points)),
                    "caliper_count": int(getattr(roi, "caliper_count", 64)),
                    "caliper_width_px": float(getattr(roi, "caliper_width_px", 8.0)),
                    "search_direction": getattr(roi, "search_direction", "Inner to Outer"),
                    "roi_type": getattr(roi, "roi_type", "Caliper Circle"),
                    "roi_inner_ratio": float(getattr(roi, "inner_ratio", 0.0)),
                    "roi_inner_radius_px": float(roi.inner_radius()),
                    "roi_outer_radius_px": float(roi.outer_radius()),
                    "roi_target_edge": getattr(roi, "target_edge", "All Edges"),
                    "roi_angle_deg": float(getattr(roi, "angle_deg", 0.0)),
                    "caliper_windows": cal.caliper_windows,
                },
            )

        layer_fit_mode = (
            getattr(self.params, "upper_fitting_mode", self.params.fitting_mode)
            if layer == "upper"
            else getattr(self.params, "lower_fitting_mode", self.params.fitting_mode)
        )
        detect_params = replace(self.params, fitting_mode=layer_fit_mode)

        edges = detect_subpixel_edges(img.gray, roi, detect_params)
        if len(edges.points_xy) < detect_params.min_edge_points:
            raise ValueError(
                f"{mark.mark_id} {LAYER_LABELS[layer]} 有效边缘点不足：{len(edges.points_xy)} < {detect_params.min_edge_points}. "
                f"可以尝试放大 ROI、降低 Canny/最小梯度，或检查焦面和对比度。"
            )
        fit = fit_mark_shape(edges.points_xy, detect_params)
        used_points = edges.points_xy
        if fit.inlier_mask is not None and len(fit.inlier_mask) == len(edges.points_xy):
            used_points = edges.points_xy[fit.inlier_mask]

        px_x = self.config.pixel_size_x_um
        px_y = self.config.pixel_size_y_um
        mean_px = 0.5 * (px_x + px_y)
        det = DetectionResult(
            mark_id=mark.mark_id,
            layer=layer,
            center_x_px=fit.center_x_px,
            center_y_px=fit.center_y_px,
            center_x_um=fit.center_x_px * px_x,
            center_y_um=fit.center_y_px * px_y,
            diameter_px=fit.diameter_px,
            diameter_um=fit.diameter_px * mean_px,
            residual_px=fit.residual_px,
            residual_um=fit.residual_px * mean_px,
            edge_point_count=len(used_points),
            confidence=fit.confidence,
            fitting_mode=fit.mode,
            warning=fit.warning or edges.warning,
            edge_points=[(float(x), float(y)) for x, y in used_points],
            shape_params={
                **fit.shape_params,
                "roi_type": getattr(roi, "roi_type", "Rectangle"),
                "roi_inner_ratio": float(getattr(roi, "inner_ratio", 0.0)),
                "roi_target_edge": getattr(roi, "target_edge", "All Edges"),
                "roi_angle_deg": float(getattr(roi, "angle_deg", 0.0)),
            },
        )
        return det

    def analyze_current_mark(self):
        mark_id = self.mark_combo.currentText()
        if not mark_id:
            return
        self._analyze_mark(mark_id)

    def analyze_current_roi(self):
        mark_id = self.mark_combo.currentText()
        if not mark_id:
            return
        layer = self._current_layer()
        mark = self.marks[mark_id]
        try:
            det = self._detect_one(mark, layer)
            self.detections.setdefault(mark_id, {})[layer] = det
            if mark_id in self.overlays:
                del self.overlays[mark_id]
        except Exception as exc:
            QMessageBox.critical(self, "分析失败", str(exc))
            return
        self._refresh_all_widgets()
        radius_px = det.shape_params.get("radius_px")
        mean_scale = 0.5 * (self.config.pixel_size_x_um + self.config.pixel_size_y_um)
        if radius_px is not None:
            detail = f"中心=({det.center_x_um:.6f}, {det.center_y_um:.6f}) μm，半径={radius_px * mean_scale:.6f} μm"
        else:
            detail = f"中心=({det.center_x_um:.6f}, {det.center_y_um:.6f}) μm，尺寸={det.diameter_um:.6f} μm"
        QMessageBox.information(self, "ROI 分析完成", f"{mark_id} {LAYER_LABELS[layer]}：\n{detail}")

    def analyze_all_marks(self):
        for mark_id in list(self.marks.keys()):
            try:
                self._analyze_mark(mark_id, show_success=False)
            except Exception as exc:
                QMessageBox.warning(self, "分析中断", f"{mark_id} 分析失败：\n{exc}")
                break
        self._refresh_all_widgets()

    def _analyze_mark(self, mark_id: str, show_success: bool = True):
        self._pull_config_from_ui()
        mark = self.marks[mark_id]
        try:
            upper = self._detect_one(mark, "upper")
            lower = self._detect_one(mark, "lower")
            self.detections.setdefault(mark_id, {})["upper"] = upper
            self.detections.setdefault(mark_id, {})["lower"] = lower
            self.overlays[mark_id] = calculate_overlay(mark_id, upper, lower, self.config)
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
            "overlay_results.xlsx",
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
            self.params = params
            loaded_marks = {m.mark_id: m for m in marks if m.mark_id in {"Mark1", "Mark2"}}
            self.marks = {
                mark_id: loaded_marks.get(mark_id, MarkRecipe(mark_id))
                for mark_id in ("Mark1", "Mark2")
            }
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
