from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np
import cv2
from PySide6.QtCore import Qt, QPointF, QRectF, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QFileDialog,
    QMessageBox,
    QButtonGroup,
    QScrollArea,
    QFrame,
)
from .. import __version__
from ..models import Roi
from ..ui_components import ImageCanvas, CollapsibleSection
from .models import Calibration, DiagnosticImage, DiagnosticsSession, transform_points
from .registration import rigid_from_points, translation_from_region
from .diagnostics_service import analyze, center_align
from .profile_comparison import compare_profiles
from .export import export_session

CYAN = "#39D5E8"
ORANGE = "#FFAB55"


class AnalysisWorker(QThread):
    completed = Signal(object, str)

    def __init__(self, session, parent, task=analyze):
        super().__init__(parent)
        self.session = session
        self.task = task

    def run(self):
        try:
            self.completed.emit(self.task(self.session), "")
        except Exception as exc:
            self.completed.emit(None, str(exc))


class DiagnosticCanvas(ImageCanvas):
    """Reuse production image display/zoom/pan; all editing belongs to A/B session."""

    def __init__(self, owner, kind):
        super().__init__(
            "Image " + kind if kind != "overlay" else "Registered Edge Overlay"
        )
        self.owner = owner
        self.kind = kind
        self.press = None
        self.roi_editing_enabled = False
        self.image_drop_enabled = False
        self.setMinimumSize(220, 140 if kind != "overlay" else 220)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.owner.busy:
            point = self.widget_to_image_float(event.position())
            if point is not None:
                self.press = point
                if self.kind == "overlay":
                    self.owner.select_point(point)
                elif self.owner.tool.currentIndex() == 3:
                    self.owner.add_point(self.kind, point)
                self.update()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        point = self.widget_to_image_float(event.position())
        if point is not None:
            self.setToolTip(f"x={point[0]:.4f} px, y={point[1]:.4f} px")
        if self.press is not None:
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.press is not None:
            end = self.widget_to_image_float(event.position())
            if end is not None and self.kind != "overlay" and not self.owner.busy:
                self.owner.drag_region(self.kind, self.press, end)
            self.press = None
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.image is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        def point(xy):
            return QPointF(*self.image_to_widget(*xy))

        session = self.owner.session
        if self.kind != "overlay":
            roi = session.roi if self.kind == "A" else session.seed_b_roi
            mapped = (
                self.kind == "B"
                and session.registration.reliable
                and session.roi is not None
            )
            if mapped:
                # Display the actual mapped reference ROI, including anisotropic ellipses.
                roi = session.roi
                matrix = session.sampling_b()
                t = np.linspace(0, 2 * np.pi, 129)
                p.setPen(QPen(QColor(ORANGE), 1))
                p.setBrush(Qt.NoBrush)
                for radius in (roi.outer_radius(), roi.inner_radius()):
                    coords = np.asarray(roi.center()) + radius * np.column_stack(
                        [np.cos(t), np.sin(t)]
                    )
                    coords = transform_points(coords, matrix)
                    path = QPainterPath(point(coords[0]))
                    for xy in coords[1:]:
                        path.lineTo(point(xy))
                    p.drawPath(path)
            elif roi is not None:
                p.setPen(QPen(QColor(CYAN if self.kind == "A" else ORANGE), 1))
                p.setBrush(Qt.NoBrush)
                for radius in (roi.outer_radius(), roi.inner_radius()):
                    c = point(roi.center())
                    p.drawEllipse(c, radius * self.scale, radius * self.scale)
            points = self.owner.points_a if self.kind == "A" else self.owner.points_b
            for i, xy in enumerate(points):
                pt = point(xy)
                p.setPen(QPen(QColor("#FFFFFF"), 1.5))
                p.drawEllipse(pt, 5, 5)
                p.drawText(pt + QPointF(8, -6), str(i + 1))
            if self.kind == "A" and self.owner.feature_region is not None:
                x, y, w, h = self.owner.feature_region
                p.setPen(QPen(QColor("#BA9CE8"), 1.5, Qt.DashLine))
                a = point((x, y))
                b = point((x + w, y + h))
                p.drawRect(QRectF(a, b))
            p.end()
            return
        result = session.result
        if result is None:
            p.end()
            return
        comp = self.owner.comparison()
        cal = session.a.oriented_calibration
        shift = (
            (result.center_a_um - result.center_b_um) / [cal.x, cal.y]
            if self.owner.centered.isChecked()
            else np.zeros(2)
        )
        for detection, color, visible, delta in [
            (result.a, CYAN, self.owner.show_a.isChecked(), np.zeros(2)),
            (result.b, ORANGE, self.owner.show_b.isChecked(), shift),
        ]:
            if visible:
                p.setPen(QPen(QColor(color), 1))
                p.setBrush(QColor(color))
                for xy in detection.edge_points + delta:
                    p.drawEllipse(point(xy), 2, 2)
            if self.owner.show_rejected.isChecked():
                p.setPen(QPen(QColor("#EC7985"), 1.2))
                for xy in detection.rejected_points + delta:
                    pt = point(xy)
                    p.drawLine(pt - QPointF(3, 3), pt + QPointF(3, 3))
                    p.drawLine(pt - QPointF(3, -3), pt + QPointF(3, -3))
        if self.owner.show_calipers.isChecked():
            p.setPen(QPen(QColor("#81909F"), 0.7))
            for trace in result.trace_a:
                p.drawLine(
                    point(trace["start"]),
                    point(
                        trace["start"]
                        + trace["search_vector"] * trace["radial_samples"][-1]
                    ),
                )
        if self.owner.show_fits.isChecked():
            p.setBrush(Qt.NoBrush)
            for center, diameter, color in [
                (result.center_a_um, result.circle_metrics["diameter_a_um"], CYAN),
                (
                    result.center_b_um + shift * [cal.x, cal.y],
                    result.circle_metrics["diameter_b_um"],
                    ORANGE,
                ),
            ]:
                p.setPen(QPen(QColor(color), 1, Qt.DashLine))
                p.drawEllipse(
                    point(center / [cal.x, cal.y]),
                    diameter / 2 / cal.x * self.scale,
                    diameter / 2 / cal.y * self.scale,
                )
        if self.owner.mode == 1:
            p.setPen(QPen(QColor("#EBD1B0"), 1))
            for i in range(0, len(comp.valid), 3):
                if not comp.valid[i]:
                    continue
                aa = comp.points_a_um[i] / [cal.x, cal.y]
                bb = comp.points_b_um[i] / [cal.x, cal.y]
                p.drawLine(
                    point(aa), point(aa + (bb - aa) * self.owner.amplification.value())
                )
            p.setPen(QColor("#FFFFFF"))
            p.drawText(
                16,
                79,
                f"位移线段 ×{self.owner.amplification.value():g}；边缘点保持真实位置",
            )
        i = self.owner.selected_index()
        if comp.valid[i]:
            xy = comp.points_a_um[i] / [cal.x, cal.y]
            if self.owner.mode == 2 and self.owner.profile is not None:
                if 'edge_a_frame_px' in self.owner.profile:
                    xy=self.owner.profile['edge_a_frame_px']
                else:
                    edge = result.trace_a[self.owner.profile["caliper_id"]]["selected"]
                    if edge is not None:
                        xy = [edge["x"], edge["y"]]
            selected = point(xy)
            p.setPen(QPen(QColor("#FFFFFF"), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(selected, 10, 10)
            if self.owner.mode == 2:
                if self.owner.profile is not None and 'sampling_line_frame_px' in self.owner.profile:
                    start,end=self.owner.profile['sampling_line_frame_px']
                    p.setPen(QPen(QColor('#FFFFFF'),1,Qt.DashLine));p.drawLine(point(start),point(end))
                    p.end();return
                trace = (
                    result.trace_a[self.owner.profile["caliper_id"]]
                    if self.owner.profile is not None
                    else None
                )
                if trace is None:
                    p.end()
                    return
                p.setPen(QPen(QColor("#FFFFFF"), 1, Qt.DashLine))
                p.drawLine(
                    point(trace["start"]),
                    point(
                        trace["start"]
                        + trace["search_vector"] * trace["radial_samples"][-1]
                    ),
                )
        p.end()


class ComparisonChart(QWidget):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.setMinimumHeight(220)
        self.setMouseTracking(True)
        self.angle_range = (0.0, 360.0)
        self.hit_rect = None

    def draw_series(self, p, x, y, r, color, limits, mask=None):
        xmin, xmax, ymin, ymax = limits
        path = QPainterPath()
        active = False
        for i, (xx, yy) in enumerate(zip(x, y)):
            if (
                not np.isfinite([xx, yy]).all()
                or (mask is not None and not mask[i])
                or not xmin <= xx <= xmax
            ):
                active = False
                continue
            pt = QPointF(
                r.left() + (xx - xmin) / (xmax - xmin) * r.width(),
                r.bottom() - (yy - ymin) / (ymax - ymin) * r.height(),
            )
            if active:
                path.lineTo(pt)
            else:
                path.moveTo(pt)
                active = True
        p.setPen(QPen(QColor(color), 1.8))
        p.drawPath(path)

    def axes(self, p, r, limits, title):
        xmin, xmax, low, high = limits
        p.setFont(QFont("Microsoft YaHei UI", 8))
        for value in np.linspace(low, high, 5):
            yy = r.bottom() - (value - low) / (high - low) * r.height()
            p.setPen(QColor("#E1E7EF"))
            p.drawLine(QPointF(r.left(), yy), QPointF(r.right(), yy))
            p.setPen(QColor("#536071"))
            p.drawText(QPointF(r.left() - 60, yy + 4), f"{value:.4g}")
        for x in np.linspace(xmin, xmax, 7):
            p.drawText(
                QPointF(
                    r.left() + (x - xmin) / (xmax - xmin) * r.width() - 12,
                    r.bottom() + 20,
                ),
                f"{x:.1f}",
            )
        p.setPen(QColor("#273544"))
        p.setFont(QFont("Microsoft YaHei UI", 9))
        p.drawText(QPointF(r.left(), r.top() - 13), title)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("white"))
        o = self.owner
        result = o.session.result
        if result is None:
            p.setPen(QColor("#687587"))
            p.drawText(self.rect(), Qt.AlignCenter, "完成配准与分析后显示曲线")
            p.end()
            return
        if o.mode == 2:
            data = o.profile
            if data is None:
                p.end()
                return
            x = data["distance_um"]
            for k in range(2):
                r = QRectF(
                    67 + k * self.width() / 2,
                    35,
                    self.width() / 2 - 92,
                    self.height() - 73,
                )
                key = "filtered" if k == 0 else "gradient"
                ys = [np.asarray(data[n][key]) for n in ("a", "b")]
                finite = np.concatenate([y[np.isfinite(y)] for y in ys])
                hi = max(float(finite.max()), 1.0) if len(finite) else 1.0
                lo = min(float(finite.min()), 0.0) if len(finite) else 0.0
                limits = (float(x[0]), float(x[-1]), lo, hi * 1.1)
                self.axes(
                    p,
                    r,
                    limits,
                    "原灰度 / 平滑灰度 DN" if k == 0 else getattr(o,'profile_marker_title',"梯度 DN/μm · 虚线为算法选峰"),
                )
                p.save()
                p.setClipRect(r)
                for name, color, y in zip(("a", "b"), (CYAN, ORANGE), ys):
                    if len(y) == len(x):
                        self.draw_series(p, x, y, r, color, limits)
                    raw = data[name]["raw"]
                    if k == 0 and len(raw) == len(x):
                        self.draw_series(
                            p, x, raw, r, QColor(color).lighter(150).name(), limits
                        )
                    selected = data[name]["selected_um"]
                    if selected is not None:
                        xx = r.left() + (selected - x[0]) / (x[-1] - x[0]) * r.width()
                        p.setPen(QPen(QColor(color), 1.5, Qt.DashLine))
                        p.drawLine(QPointF(xx, r.top()), QPointF(xx, r.bottom()))
                    if k == 1:
                        for peak in data[name]["all_candidates_um"]:
                            xx = r.left() + (peak - x[0]) / (x[-1] - x[0]) * r.width()
                            p.setPen(QPen(QColor(color), 0.7, Qt.DotLine))
                            p.drawLine(QPointF(xx, r.top()), QPointF(xx, r.bottom()))
                p.restore()
        else:
            comp = o.comparison()
            kind = o.curve.currentIndex()
            signed = comp.radius_b_um - comp.radius_a_um
            y = signed if kind == 0 else abs(signed) if kind == 1 else comp.radius_b_um
            finite = y[comp.valid]
            low = min(0, float(finite.min())) if len(finite) else -1.0
            high = max(0, float(finite.max())) if len(finite) else 1.0
            if kind == 2:
                independent = np.r_[
                    comp.radius_a_um[np.isfinite(comp.radius_a_um)],
                    comp.radius_b_um[np.isfinite(comp.radius_b_um)],
                ]
                if len(independent):
                    low = float(independent.min())
                    high = float(independent.max())
            pad = max((high - low) * 0.15, 1e-5)
            limits = (*self.angle_range, low - pad, high + pad)
            r = QRectF(67, 35, self.width() - 92, self.height() - 73)
            self.hit_rect = r
            self.axes(
                p,
                r,
                limits,
                (
                    "边缘位移 μm / 角度 °"
                    if kind < 2
                    else "相对 A 物理圆心的径向位置 μm / 角度 °"
                ),
            )
            p.save()
            p.setClipRect(r)
            if kind == 2:
                self.draw_series(
                    p,
                    comp.angles_deg,
                    comp.radius_a_um,
                    r,
                    CYAN,
                    limits,
                    np.isfinite(comp.radius_a_um),
                )
            self.draw_series(
                p,
                comp.angles_deg,
                y,
                r,
                ORANGE,
                limits,
                np.isfinite(comp.radius_b_um) if kind == 2 else comp.valid,
            )
            xx = r.left() + (o.angle - limits[0]) / (limits[1] - limits[0]) * r.width()
            p.setPen(QPen(QColor("#087EF4"), 1, Qt.DashLine))
            p.drawLine(QPointF(xx, r.top()), QPointF(xx, r.bottom()))
            if kind < 2:
                yy = r.bottom() - (0 - limits[2]) / (limits[3] - limits[2]) * r.height()
                p.setPen(QPen(QColor("#7B899A"), 1, Qt.DashLine))
                p.drawLine(QPointF(r.left(), yy), QPointF(r.right(), yy))
            p.restore()
        p.end()

    def select(self, pos):
        if self.owner.mode == 2 or self.hit_rect is None:
            return
        r = self.hit_rect
        if r.contains(pos):
            self.owner.angle = float(
                np.clip(
                    self.angle_range[0]
                    + (pos.x() - r.left())
                    / r.width()
                    * (self.angle_range[1] - self.angle_range[0]),
                    0,
                    359.999,
                )
            )
            self.owner.refresh_selection()

    def mouseMoveEvent(self, event):
        if self.owner.mode == 2 and self.owner.profile is not None:
            data = self.owner.profile
            x = data["distance_um"]
            panel = 0 if event.position().x() < self.width() / 2 else 1
            fraction = np.clip(
                (event.position().x() - panel * self.width() / 2 - 67)
                / (self.width() / 2 - 92),
                0,
                1,
            )
            i = min(int(fraction * (len(x) - 1)), len(x) - 1)
            key = "raw" if panel == 0 else "gradient"
            unit = "DN" if panel == 0 else "DN/μm"
            readings = []
            for name in ("a", "b"):
                v = data[name][key]
                readings.append(
                    name.upper()
                    + (
                        " 缺失"
                        if i >= len(v) or not np.isfinite(v[i])
                        else f" {v[i]:.6f} {unit}"
                    )
                )
            self.setToolTip(f"距离 {x[i]:.6f} μm；" + "；".join(readings))
        else:
            self.select(event.position())

    def mousePressEvent(self, event):
        self.select(event.position())

    def wheelEvent(self, event):
        if self.owner.mode == 2:
            return
        half = max(
            3,
            (self.angle_range[1] - self.angle_range[0])
            / 2
            * (0.7 if event.angleDelta().y() > 0 else 1 / 0.7),
        )
        self.angle_range = (
            max(0, self.owner.angle - half),
            min(360, self.owner.angle + half),
        )
        self.update()


class DiagnosticsDialog(QDialog):
    def __init__(self, params, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"SOMA Diagnostics · 成像诊断 · V{__version__}")
        self.resize(1500, 1020)
        self.setMinimumSize(1120, 800)
        self.session = DiagnosticsSession.from_parameters(params, config)
        self.default_calibration = Calibration(
            config.pixel_size_x_um, config.pixel_size_y_um
        )
        self.mode = 0
        self.angle = 0.0
        self.points_a = []
        self.points_b = []
        self.feature_region = None
        self.busy = False
        self.worker = None
        self.profile = None
        self._profile_key = None
        self.setStyleSheet(
            "QDialog{background:#F5F7FA;} QLabel{color:#283544;} QPushButton,QComboBox,QSpinBox,QDoubleSpinBox{border:1px solid #D6DCE4;border-radius:5px;background:white;padding:5px;} QPushButton:checked{background:#EAF3FF;color:#087EF4;}"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 12, 18, 12)
        root.addWidget(
            QLabel(
                f"<b>SOMA Vision Metrology / 成像诊断</b>　V{__version__}　独立 A/B 会话"
            )
        )
        self.mutations = QWidget()
        toolbar = QHBoxLayout(self.mutations)
        toolbar.setContentsMargins(0, 0, 0, 0)
        for title, key in [("导入 Image A", "A"), ("导入 Image B", "B")]:
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, k=key: self.import_image(k))
            toolbar.addWidget(button)
        self.rotation = QComboBox()
        self.rotation.addItems(
            ["B 原始方向", "B 顺时针 90°", "B 180°", "B 顺时针 270°"]
        )
        self.rotation.currentIndexChanged.connect(self.rotate_b)
        toolbar.addWidget(self.rotation)
        self.registration_method = QComboBox()
        self.registration_method.addItems(
            ["手工独立对应点 / 刚性", "独立特征区 / 平移", "圆心平移 / 仅形状"]
        )
        toolbar.addWidget(self.registration_method)
        register = QPushButton("配准")
        register.clicked.connect(self.register)
        toolbar.addWidget(register)
        self.run_button = QPushButton("开始分析")
        self.run_button.setStyleSheet("background:#087EF4;color:white;font-weight:600;")
        self.run_button.clicked.connect(self.start_analysis)
        toolbar.addWidget(self.run_button)
        self.export_button = QPushButton("导出结果")
        self.export_button.clicked.connect(self.export)
        toolbar.addWidget(self.export_button)
        toolbar.addStretch()
        root.addWidget(self.mutations)
        self.tool = QComboBox()
        self.tool.addItems(
            [
                "在 A 拖动参考环带",
                "在 B 拖动初始环带（圆心对齐用）",
                "在 A 拖动独立配准矩形",
                "A/B 交替点击对应点",
                "浏览 / 平移",
            ]
        )
        tools = QHBoxLayout()
        tools.addWidget(self.tool)
        clear = QPushButton("清除基准点")
        clear.clicked.connect(self.clear_points)
        tools.addWidget(clear)
        self.point_status = QLabel("基准点 A 0 / B 0")
        tools.addWidget(self.point_status)
        tools.addStretch()
        root.addLayout(tools)
        body = QHBoxLayout()
        left = QVBoxLayout()
        body.addLayout(left, 1)
        sources = QHBoxLayout()
        self.source_a = DiagnosticCanvas(self, "A")
        self.source_b = DiagnosticCanvas(self, "B")
        self.source_a.setMaximumHeight(185)
        self.source_b.setMaximumHeight(185)
        sources.addWidget(self.source_a)
        sources.addWidget(self.source_b)
        left.addLayout(sources)
        tabs = QHBoxLayout()
        self.views = QButtonGroup(self)
        for i, text in enumerate(
            [
                "Edge Overlay / 边缘叠加",
                "Edge Displacement / 局部位移",
                "Edge Profile / 灰度与梯度",
            ]
        ):
            b = QPushButton(text)
            b.setCheckable(True)
            self.views.addButton(b, i)
            tabs.addWidget(b)
        self.views.button(0).setChecked(True)
        self.views.idClicked.connect(self.set_mode)
        left.addLayout(tabs)
        visibility = QHBoxLayout()
        for attr, title, checked in [
            ("show_a", "A 青色", True),
            ("show_b", "B 橙色", True),
            ("show_fits", "拟合圆", True),
            ("show_rejected", "剔除点", True),
            ("show_calipers", "卡尺", False),
        ]:
            c = QCheckBox(title)
            c.setChecked(checked)
            c.toggled.connect(self.refresh_selection)
            setattr(self, attr, c)
            visibility.addWidget(c)
        self.amplification = QDoubleSpinBox()
        self.amplification.setRange(1, 100)
        self.amplification.setValue(4)
        self.amplification.setPrefix("位移线 ×")
        self.amplification.valueChanged.connect(self.refresh_selection)
        visibility.addWidget(self.amplification)
        for text in ["适应窗口", "1:1", "局部放大"]:
            b = QPushButton(text)
            b.clicked.connect(lambda checked=False, t=text: self.zoom(t))
            visibility.addWidget(b)
        left.addLayout(visibility)
        self.overlay = DiagnosticCanvas(self, "overlay")
        left.addWidget(self.overlay, 1)
        curves = QHBoxLayout()
        self.curve = QComboBox()
        self.curve.addItems(
            ["带符号 B − A", "绝对差异 |B − A|", "A/B 统一参考径向位置"]
        )
        self.curve.currentIndexChanged.connect(self.refresh_selection)
        curves.addWidget(self.curve)
        self.centered = QCheckBox("各自圆心对齐 / 消除平移")
        self.centered.toggled.connect(self.refresh_selection)
        curves.addWidget(self.centered)
        self.background = QComboBox()
        self.background.addItems(
            ["背景 A 原灰度", "背景 B 配准预览（仅显示）", "无灰度背景"]
        )
        self.background.currentIndexChanged.connect(self.update_background)
        curves.addWidget(self.background)
        reset = QPushButton("角度 0–360°")
        reset.clicked.connect(self.reset_angles)
        curves.addWidget(reset)
        curves.addStretch()
        left.addLayout(curves)
        self.chart = ComparisonChart(self)
        left.addWidget(self.chart)
        side = QWidget()
        side.setFixedWidth(285)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(10, 0, 0, 0)
        self.summary = QLabel("请导入 A/B，指定参考 ROI 和独立配准基准。")
        self.summary.setWordWrap(True)
        side_layout.addWidget(self.summary)
        self.selected_label = QLabel("")
        self.selected_label.setWordWrap(True)
        self.selected_label.setStyleSheet("color:#087EF4;")
        side_layout.addWidget(self.selected_label)
        self.profile_label = QLabel("")
        self.profile_label.setWordWrap(True)
        side_layout.addWidget(self.profile_label)
        settings = CollapsibleSection("检测参数与像素标定", False)
        form = QFormLayout()
        settings.body_layout.addLayout(form)
        self.editors = {}
        for key, title, value, low, high, decimals in [
            ("sigma", "Gaussian σ px", params.gaussian_sigma_px, 0, 20, 2),
            ("gradient", "梯度阈值 DN/px", params.min_gradient, 0, 65535, 2),
            ("step", "径向步长 px", params.profile_step_px, 0.05, 2, 2),
            ("width", "卡尺宽度 px", 8, 1, 64, 1),
            ("inner", "内半径比", 0.6, 0.05, 0.98, 2),
            ("residual", "RANSAC 残差 px", params.residual_limit_px, 0.1, 30, 3),
            ("ax", "A X μm/px", config.pixel_size_x_um, 0.000001, 1000, 6),
            ("ay", "A Y μm/px", config.pixel_size_y_um, 0.000001, 1000, 6),
            ("bx", "B X μm/px", config.pixel_size_x_um, 0.000001, 1000, 6),
            ("by", "B Y μm/px", config.pixel_size_y_um, 0.000001, 1000, 6),
        ]:
            editor = QDoubleSpinBox()
            editor.setDecimals(decimals)
            editor.setRange(low, high)
            editor.setValue(value)
            self.editors[key] = editor
            form.addRow(title, editor)
            editor.valueChanged.connect(lambda value, k=key: self.parameters_changed(k))
        self.count = QSpinBox()
        self.count.setRange(16, 720)
        self.count.setValue(64)
        self.count.valueChanged.connect(lambda _: self.parameters_changed("count"))
        form.addRow("卡尺数量", self.count)
        self.polarity = QComboBox()
        self.polarity.addItems(["Auto", "Dark to Bright", "Bright to Dark"])
        self.polarity.setCurrentText(params.polarity)
        self.polarity.currentIndexChanged.connect(
            lambda _: self.parameters_changed("polarity")
        )
        form.addRow("边缘极性", self.polarity)
        self.target = QComboBox()
        self.target.addItems(
            ["Strongest Edge", "Near Inner Boundary", "Near Outer Boundary"]
        )
        self.target.currentIndexChanged.connect(
            lambda _: self.parameters_changed("target")
        )
        form.addRow("目标边缘", self.target)
        self.direction = QComboBox()
        self.direction.addItems(["Inner to Outer", "Outer to Inner"])
        self.direction.currentIndexChanged.connect(
            lambda _: self.parameters_changed("direction")
        )
        form.addRow("搜索方向", self.direction)
        self.ransac = QCheckBox("RANSAC")
        self.ransac.setChecked(params.use_ransac)
        self.ransac.toggled.connect(lambda _: self.parameters_changed("ransac"))
        form.addRow(self.ransac)
        self.max_residual = QDoubleSpinBox()
        self.max_residual.setRange(0.000001, 100)
        self.max_residual.setDecimals(6)
        self.max_residual.setValue(0.2)
        self.max_residual.valueChanged.connect(lambda _: self.invalidate(True))
        form.addRow("配准残差门槛 μm", self.max_residual)
        settings_container = QWidget()
        settings_layout = QVBoxLayout(settings_container)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.addWidget(settings)
        settings_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(settings_container)
        scroll.setMinimumHeight(70)
        side_layout.addWidget(scroll, 1)
        body.addWidget(side)
        root.addLayout(body, 1)
        self.status = QLabel(
            "先统一已知扫描方向；独立特征优先。单圆不能确定旋转，圆心对齐不能用来判断真实圆心漂移。"
        )
        self.status.setWordWrap(True)
        root.addWidget(self.status)

    def value(self, key):
        return self.editors[key].value()

    def parameters_changed(self, key):
        if self.busy:
            return
        p = self.session.params
        p.gaussian_sigma_px = self.value("sigma")
        p.min_gradient = self.value("gradient")
        p.profile_step_px = self.value("step")
        p.residual_limit_px = self.value("residual")
        p.polarity = self.polarity.currentText()
        p.use_ransac = self.ransac.isChecked()
        for roi in (self.session.roi, self.session.seed_b_roi):
            if roi is not None:
                roi.caliper_count = self.count.value()
                roi.caliper_width_px = self.value("width")
                roi.inner_ratio = self.value("inner")
                roi.target_edge = self.target.currentText()
                roi.search_direction = self.direction.currentText()
        if self.session.a:
            self.session.a.calibration = Calibration(self.value("ax"), self.value("ay"))
        if self.session.b:
            self.session.b.calibration = Calibration(self.value("bx"), self.value("by"))
        self.sync_canvas_calibration()
        reset = (
            key in ("ax", "ay", "bx", "by")
            or self.session.registration.method == "circle_center"
        )
        self.invalidate(reset)

    def invalidate(self, registration=False):
        self.session.invalidate(registration)
        self.profile = None
        self._profile_key = None
        self.status.setText(
            "配置已变化，请" + ("重新配准并" if registration else "") + "重新分析"
        )
        self.refresh_selection()
        if registration:
            self.update_background()

    def sync_canvas_calibration(self):
        for slot, canvas in [
            (self.session.a, self.source_a),
            (self.session.b, self.source_b),
            (self.session.a, self.overlay),
        ]:
            if slot is not None:
                canvas.pixel_size_x_um = slot.oriented_calibration.x
                canvas.pixel_size_y_um = slot.oriented_calibration.y

    def update_background(self, *args):
        s = self.session
        if s.a is None:
            return
        image = s.a.oriented
        kind = self.background.currentIndex()
        if kind == 1 and s.b is not None and s.registration.reliable:
            # Display only. The detector always receives s.b.oriented.gray.
            warp = np.linalg.inv(s.sampling_b())
            gray = cv2.warpAffine(
                s.b.oriented.gray,
                warp[:2],
                (image.gray.shape[1], image.gray.shape[0]),
                flags=cv2.INTER_LINEAR,
            )
            image = replace(image, gray=gray, source_dtype=s.b.original.source_dtype)
        elif kind == 2:
            image = replace(image, gray=np.zeros_like(image.gray))
        zoom, px, py = self.overlay.user_zoom, self.overlay.pan_x, self.overlay.pan_y
        self.overlay.set_image(image)
        self.overlay.user_zoom = zoom
        self.overlay.pan_x = px
        self.overlay.pan_y = py
        self.overlay.update()

    def import_image(self, kind):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入 Image " + kind,
            "",
            "Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg *.npy *.csv *.txt)",
        )
        if path:
            try:
                self.load_path(kind, path)
            except Exception as exc:
                QMessageBox.warning(self, "导入失败", str(exc))

    def load_path(self, kind, path):
        if self.busy:
            raise ValueError("分析进行中")
        slot = DiagnosticImage.load(
            path,
            Calibration(
                self.value("ax" if kind == "A" else "bx"),
                self.value("ay" if kind == "A" else "by"),
            ),
        )
        if kind == "B":
            slot.turns = self.rotation.currentIndex()
        setattr(self.session, kind.lower(), slot)
        image = slot.oriented
        h, w = image.gray.shape
        size = min(w, h) * 0.8
        roi = Roi(
            (w - size) / 2,
            (h - size) / 2,
            size,
            size,
            "Caliper Circle",
            self.value("inner"),
            self.target.currentText(),
            caliper_count=self.count.value(),
            caliper_width_px=self.value("width"),
            search_direction=self.direction.currentText(),
        )
        if kind == "A":
            self.session.roi = roi
            self.source_a.set_image(image)
            self.overlay.set_image(image)
        else:
            self.session.seed_b_roi = roi
            self.source_b.set_image(image)
        self.clear_points()
        self.feature_region = None
        self.invalidate(True)
        self.sync_canvas_calibration()

    def rotate_b(self, turns):
        if self.session.b is None:
            return
        self.session.b.turns = turns
        self.source_b.set_image(self.session.b.oriented)
        self.session.seed_b_roi = None
        self.sync_canvas_calibration()
        self.clear_points()
        self.invalidate(True)

    def clear_points(self):
        if self.busy:
            return
        self.points_a = []
        self.points_b = []
        self.point_status.setText("基准点 A 0 / B 0")
        if self.session.registration.method == "manual_rigid":
            self.invalidate(True)
        self.source_a.update()
        self.source_b.update()

    def add_point(self, kind, point):
        if self.busy:
            return
        points = self.points_a if kind == "A" else self.points_b
        points.append(tuple(point))
        self.point_status.setText(
            f"基准点 A {len(self.points_a)} / B {len(self.points_b)}；按编号对应"
        )
        self.invalidate(True)
        self.source_a.update()
        self.source_b.update()

    def drag_region(self, kind, start, end):
        index = self.tool.currentIndex()
        x, y = min(start[0], end[0]), min(start[1], end[1])
        w, h = abs(end[0] - start[0]), abs(end[1] - start[1])
        if min(w, h) < 4:
            return
        if index == 2 and kind == "A":
            self.feature_region = (int(x), int(y), int(w), int(h))
            self.invalidate(True)
        elif (index == 0 and kind == "A") or (index == 1 and kind == "B"):
            size = min(w, h)
            roi = Roi(
                x + (w - size) / 2,
                y + (h - size) / 2,
                size,
                size,
                "Caliper Circle",
                self.value("inner"),
                self.target.currentText(),
                caliper_count=self.count.value(),
                caliper_width_px=self.value("width"),
                search_direction=self.direction.currentText(),
            )
            if kind == "A":
                self.session.roi = roi
            else:
                self.session.seed_b_roi = roi
            self.invalidate(self.session.registration.method == "circle_center")
        self.source_a.update()
        self.source_b.update()

    def register(self):
        try:
            self.session.require_pair()
            self.session.invalidate(True)
            s = self.session
            a = s.a
            b = s.b
            if self.registration_method.currentIndex() == 0:
                reg = rigid_from_points(
                    self.points_a,
                    self.points_b,
                    a.oriented_calibration,
                    b.oriented_calibration,
                    max_residual_um=self.max_residual.value(),
                )
            elif self.registration_method.currentIndex() == 1:
                if self.feature_region is None:
                    raise ValueError("请先在 A 拖动独立于被测孔的配准基准区")
                if s.roi is not None:
                    x, y, w, h = self.feature_region
                    rx, ry = s.roi.center()
                    r = s.roi.outer_radius()
                    if x < rx + r and x + w > rx - r and y < ry + r and y + h > ry - r:
                        raise ValueError(
                            "配准基准区与被测孔 ROI 相交，请选择独立稳定特征或使用圆心对齐口径"
                        )
                reg = translation_from_region(
                    a.oriented.gray,
                    b.oriented.gray,
                    a.oriented_calibration,
                    b.oriented_calibration,
                    self.feature_region,
                )
            else:
                self.set_busy(True)
                self.status.setText("圆心平移配准中：共用卡尺迭代，不改变原始图像…")
                snapshot = DiagnosticsSession(
                    a=replace(s.a),
                    b=replace(s.b),
                    params=deepcopy(s.params),
                    roi=deepcopy(s.roi),
                    seed_b_roi=deepcopy(s.seed_b_roi),
                )
                self.worker = AnalysisWorker(snapshot, self, task=center_align)
                self.worker.completed.connect(self.registration_done)
                self.worker.finished.connect(self.worker_finished)
                self.worker.start()
                return
            s.registration = reg
            self.profile = None
            self.refresh_selection()
            self.update_background()
            self.status.setText(
                f"{reg.method}：" + (reg.warning or "配准通过，开始分析可计算边缘差异")
            )
        except Exception as exc:
            self.invalidate(True)
            QMessageBox.warning(self, "配准失败", str(exc))

    def registration_done(self, registration, error):
        if error:
            self.session.invalidate(True)
            self.status.setText("配准失败：" + error)
        else:
            self.session.registration = registration
            self.status.setText(registration.warning or "配准通过，可以开始分析")
        self.profile = None
        self._profile_key = None
        self.refresh_selection()
        self.update_background()

    def start_analysis(self):
        if self.busy:
            return
        if not self.session.registration.reliable:
            QMessageBox.information(self, "尚未配准", "请先建立可靠的配准基准")
            return
        self.session.result = None
        self.profile = None
        self.refresh_selection()
        # Snapshot settings; read-only source arrays are retained without doubling large TIFF memory.
        s = self.session
        snapshot = DiagnosticsSession(
            a=replace(s.a),
            b=replace(s.b),
            params=deepcopy(s.params),
            roi=deepcopy(s.roi),
            seed_b_roi=deepcopy(s.seed_b_roi),
            registration=deepcopy(s.registration),
        )
        self.set_busy(True)
        self.worker = AnalysisWorker(snapshot, self)
        self.worker.completed.connect(self.analysis_done)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    def set_busy(self, value):
        self.busy = value
        self.mutations.setEnabled(not value)
        self.tool.setEnabled(not value)
        for editor in [
            *self.editors.values(),
            self.count,
            self.polarity,
            self.target,
            self.direction,
            self.ransac,
            self.max_residual,
        ]:
            editor.setEnabled(not value)
        self.status.setText(
            "分析中：共用生产卡尺算法，原始图像不重采样…"
            if value
            else "分析完成；可查看边缘、局部位移与采样剖面。"
        )

    def analysis_done(self, result, error):
        self.session.result = result
        self._profile_key = None
        if error:
            self.status.setText("分析失败：" + error)
        elif result is not None:
            self.status.setText("；".join(result.warnings))
        self.refresh_selection()

    def worker_finished(self):
        text = self.status.text()
        self.set_busy(False)
        self.status.setText(text)
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def comparison(self):
        return (
            self.session.result.shape
            if self.centered.isChecked()
            else self.session.result.registered
        )

    def selected_index(self):
        return int(round(self.angle)) % 360

    def select_point(self, point):
        if self.session.result is None:
            return
        physical = (
            self.session.a.oriented_calibration.physical([point])[0]
            - self.session.result.center_a_um
        )
        self.angle = float(np.degrees(np.arctan2(physical[1], physical[0])) % 360)
        self.refresh_selection()

    def set_mode(self, mode):
        self.mode = mode
        self.views.button(mode).setChecked(True)
        self.refresh_selection()

    def refresh_selection(self, *args):
        result = self.session.result
        self.profile_label.clear()
        reg = self.session.registration
        if result is None:
            self.summary.setText(
                f"配准：{reg.method}\n"
                + ("通过" if reg.reliable else "未通过 / 尚未配准")
                + "\n"
                + reg.warning
            )
            self.selected_label.clear()
            self.overlay.update()
            self.chart.update()
            return
        comp = self.comparison()
        m = comp.metrics
        circle = result.circle_metrics

        def f(value):
            return "—" if value is None else f"{value:.6f}"

        confidence = "可比较" if result.reliable else "不可靠 / 仅供排查，禁止定量结论"
        translation = reg.matrix[:2, 2]
        self.summary.setText(
            f"<b>诊断摘要 / {confidence}</b><br>配准：{reg.method}<br>{reg.features}<br>"
            f"B→A 平移 {translation[0]:+.5f} / {translation[1]:+.5f} μm<br>"
            f"旋转 {reg.rotation_deg:+.5f}°<br>残差 {f(reg.residual_um)} μm / 响应 {f(reg.response)}<hr>"
            f'RMS <b style="font-size:17pt">{f(m["rms_um"])} μm</b><br>'
            f'P-V <b style="font-size:17pt">{f(m["pv_um"])} μm</b><br>'
            f'Max |Δ| {f(m["max_abs_um"])} μm<br>P95 |Δ| {f(m["p95_abs_um"])} μm<br>'
            f'Mean {f(m["mean_um"])} μm<br>有效覆盖率 {100*m["coverage"]:.2f}%<hr>'
            f'物理轮廓圆心 / 原图 px<br>A {np.round(circle["center_a_native_px"],5)}<br>'
            f'B {np.round(circle["center_b_native_px"],5)}<br>统一坐标 ΔC μm<br>'
            f'{np.round(circle["center_delta_registered_um"],6)}<br>直径 A {circle["diameter_a_um"]:.6f} μm<br>'
            f'B {circle["diameter_b_um"]:.6f} μm<br>ΔD {circle["diameter_delta_um"]:+.6f} μm'
        )
        i = self.selected_index()
        text = f"{comp.angles_deg[i]:.1f}° / " + (
            f"Δn {comp.radius_b_um[i]-comp.radius_a_um[i]:+.6f} μm"
            if comp.valid[i]
            else "无有效对应，不插补"
        )
        if comp.valid[i]:
            calibration = [
                self.session.a.oriented_calibration.x,
                self.session.a.oriented_calibration.y,
            ]
            signed_px = np.sign(
                comp.radius_b_um[i] - comp.radius_a_um[i]
            ) * np.linalg.norm(
                (comp.points_b_um[i] - comp.points_a_um[i]) / calibration
            )
            text += f" / {signed_px:+.5f} 参考 A px"
            text += f"\nA 角插值位置 px {np.round(comp.points_a_um[i]/calibration,5)}"
        text += "\n口径：" + (
            "各自圆心对齐，平移被消除"
            if self.centered.isChecked()
            else (
                "圆心平移后的相对形状"
                if reg.method == "circle_center"
                else "独立基准配准后几何差异"
            )
        )
        if reg.method == "circle_center":
            text += "\n圆心对齐已消除平移，旋转未知"
        self.selected_label.setText(text)
        if self.mode == 2:
            key = (id(result), i)
            try:
                if self._profile_key != key:
                    self.profile = compare_profiles(self.session, comp.angles_deg[i])
                    self._profile_key = key
                d = self.profile
                descriptions = []
                for name in ("a", "b"):
                    item = d[name]
                    width = (
                        f'{item["width_um"]:.6f} μm'
                        if item["width_um"] is not None
                        else "不报告：" + item["width_reason"]
                    )
                    descriptions.append(
                        f'{name.upper()} 峰数 {len(item["candidates_um"])}/{len(item["all_candidates_um"])}（过阈值/全部）\n峰位置 {f(item["selected_um"])} μm\n梯度峰幅值 {f(item["selected_gradient_dn_per_um"])} DN/μm\n10–90% 宽度 {width}\n状态 {item["status"]}'
                    )
                kind = (
                    "局部有效位移"
                    if d["local_shift_valid"]
                    else "候选峰位置差（非有效测量）"
                )
                direction = (
                    "向孔外"
                    if self.session.roi.search_direction == "Inner to Outer"
                    else "向孔内"
                )
                self.profile_label.setText(
                    "\n".join(descriptions)
                    + f'\n{kind} {f(d["local_shift_um"])} μm\n实际卡尺角 {d["selected_physical_angle_deg"]:.3f}°\n采样正方向：{direction}\n步长 {d["step_um"]:.6f} μm / 双线性\n原灰度细线；Gaussian 后粗线\n剖面保持配准物理坐标，不随形状对齐移动'
                )
            except ValueError as exc:
                self.profile_label.setText(str(exc))
                self.profile = None
        self.overlay.update()
        self.chart.update()
        self.source_a.update()
        self.source_b.update()

    def zoom(self, action):
        if action == "适应窗口":
            self.overlay.reset_view()
        elif action == "1:1":
            self.overlay._update_transform()
            self.overlay.user_zoom = 1 / max(self.overlay.fit_scale, 1e-12)
            self.overlay.pan_x = 0
            self.overlay.pan_y = 0
            self.overlay.update()
        elif self.session.result is not None:
            comp = self.comparison()
            i = self.selected_index()
            if comp.valid[i]:
                xy = comp.points_a_um[i] / [
                    self.session.a.oriented_calibration.x,
                    self.session.a.oriented_calibration.y,
                ]
                self.overlay._update_transform()
                self.overlay.user_zoom = 12 / max(self.overlay.fit_scale, 1e-12)
                self.overlay.user_zoom = min(self.overlay.user_zoom, 80)
                scale = self.overlay.fit_scale * self.overlay.user_zoom
                bx, by = self.overlay._base_offset_for_scale(scale)
                self.overlay.pan_x = (
                    self.overlay.width() / 2 - (xy[0] + 0.5) * scale - bx
                )
                self.overlay.pan_y = (
                    self.overlay.height() / 2 - (xy[1] + 0.5) * scale - by
                )
                self.overlay.update()

    def reset_angles(self):
        self.chart.angle_range = (0.0, 360.0)
        self.chart.update()

    def export(self):
        if self.session.result is None:
            QMessageBox.information(self, "导出", "请先分析")
            return
        directory = QFileDialog.getExistingDirectory(
            self, "选择导出目录（将新建独立诊断子目录）"
        )
        if not directory:
            return
        try:
            from datetime import datetime

            output = Path(directory) / (
                "SOMA_Diagnostics_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
            export_session(self.session, output, excel=True)
            export_session(self.session, output, excel=False)
            self.overlay.grab().save(str(output / "edge_overlay.png"))
            self.chart.grab().save(str(output / "current_view.png"))
            if self.profile is not None:
                from ..result_exporter import export_data_tables

                d = self.profile
                rows = []
                for i, x in enumerate(d["distance_um"]):
                    row = {"distance_um": x}
                    for name in ("a", "b"):
                        for key in ("raw", "filtered", "gradient"):
                            v = d[name][key]
                            row[name + "_" + key] = (
                                float(v[i])
                                if i < len(v) and np.isfinite(v[i])
                                else None
                            )
                    rows.append(row)
                export_data_tables(str(output / "csv"), {"selected_profile": rows})
            self.status.setText("已导出：" + str(output))
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            event.ignore()
            self.status.setText("分析尚未结束，请稍后关闭窗口")
            return
        super().closeEvent(event)
