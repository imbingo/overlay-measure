"""Interactive, isolated parameter comparison for one production caliper ROI."""
from __future__ import annotations

from dataclasses import replace
from html import escape
import time
import numpy as np

from PySide6.QtCore import QPointF, QRectF, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QPushButton, QScrollArea, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from . import __version__
from .parameter_comparison import (
    ComparisonSource, TUNABLE_FIELDS, available_rois, calculate_preview,
    point_explanation, raw_profile,
)
from .theme import ui_color
from .ui_components import ImageCanvas

CYAN, AMBER, RED = "#67c5d8", "#e8b45a", "#ff6b6b"
STATUS = {"valid": "保留", "fit_rejected": "剔除", "not_fitted": "未拟合",
          "outside_image": "采样越界", "no_peak": "无梯度峰",
          "below_gradient_threshold": "低于梯度门槛", "pending_fit": "待拟合"}


class PreviewWorker(QThread):
    completed = Signal(int, object, object)

    def __init__(self, source, params, revision, needs_baseline, parent):
        super().__init__(parent)
        self.source, self.params = source, replace(params)
        self.revision, self.needs_baseline = revision, needs_baseline

    def run(self):
        deadline = time.monotonic() + max(1, self.params.measurement_timeout_s)
        def cancelled():
            if self.isInterruptionRequested():
                return True
            if time.monotonic() >= deadline:
                raise TimeoutError("预览计算超时，请缩小 ROI 或减少卡尺采样范围")
            return False
        try:
            baseline = calculate_preview(self.source, self.source.params, cancelled) if self.needs_baseline else None
            if self.isInterruptionRequested():
                return
            current = baseline if baseline is not None and self.params == baseline.params else calculate_preview(self.source, self.params, cancelled)
            if not self.isInterruptionRequested():
                self.completed.emit(self.revision, baseline, current)
        except InterruptedError:
            pass
        except Exception as exc:
            from .parameter_comparison import ComparisonResult
            self.completed.emit(self.revision, None, ComparisonResult(self.params, [], None, str(exc)))


class ComparisonCanvas(ImageCanvas):
    caliperSelected = Signal(int)
    viewChanged = Signal()

    def __init__(self):
        super().__init__("基准 / 当前边缘叠加")
        self.roi_editing_enabled = False
        self.image_drop_enabled = False
        self.setMinimumSize(250, 180)
        self.source = self.baseline = self.current = None
        self.selected = 0
        self.kind = "overlay"

    def zoom_by(self, factor, center_pos=None):
        super().zoom_by(factor, center_pos)
        self.viewChanged.emit()

    def mouseMoveEvent(self, event):
        was_panning = self.is_panning
        super().mouseMoveEvent(event)
        if was_panning:
            self.viewChanged.emit()

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        self.viewChanged.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.space_pan_held and self.source:
            point = self.widget_to_image_float(event.position().toPoint())
            if point is not None:
                cx, cy = self.source.roi.normalized().center()
                angle = np.arctan2(point[1] - cy, point[0] - cx) % (2 * np.pi)
                count = int(np.clip(self.source.roi.caliper_count, 4, 720))
                self.caliperSelected.emit(int(np.floor(angle / (2 * np.pi) * count + 0.5)) % count)
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.source or self.pixmap_cache is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        def points(values, color, cross=False):
            painter.setPen(QPen(QColor(color), 1.6))
            for x, y in values:
                wx, wy = self.image_to_widget(float(x), float(y))
                if cross:
                    painter.drawLine(QPointF(wx - 3, wy - 3), QPointF(wx + 3, wy + 3))
                    painter.drawLine(QPointF(wx - 3, wy + 3), QPointF(wx + 3, wy - 3))
                else:
                    painter.drawEllipse(QPointF(wx, wy), 2, 2)

        for result, color, show in ((self.baseline, CYAN, self.kind != "current"),
                                     (self.current, AMBER, self.kind != "baseline")):
            if not result or not show:
                continue
            if result.circle is not None:
                points(result.circle.edge_points, color)
                if result is self.current:
                    points(result.circle.rejected_points, RED, True)
            else:
                candidates = [r["selected"] for r in result.records if r.get("selected")]
                points([(c["x"], c["y"]) for c in candidates], "#9a9a9f")
        result = self.baseline if self.kind == "baseline" else self.current
        if result and self.selected < len(result.records):
            for r in result.records:
                if r.get("selected") is None:
                    xy = r["start"] + r["search_vector"] * float(r["radial_samples"][-1]) * 0.5
                    wx, wy = self.image_to_widget(*xy)
                    painter.setPen(QPen(QColor("#9a9a9f"), 1))
                    painter.drawEllipse(QPointF(wx, wy), 3, 3)
            record = result.records[self.selected]
            half_width = float(self.source.roi.caliper_width_px) / 2
            start, end = record["start"], record["start"] + record["search_vector"] * record["radial_samples"][-1]
            corners = [start - record["tangent"] * half_width, end - record["tangent"] * half_width,
                       end + record["tangent"] * half_width, start + record["tangent"] * half_width]
            painter.setPen(QPen(QColor("white"), 1.2))
            painter.drawPolygon(QPolygonF([QPointF(*self.image_to_widget(*p)) for p in corners]))
            wx, wy = self.image_to_widget(*end)
            label = f"卡尺 {self.selected + 1}"
            label_width = painter.fontMetrics().horizontalAdvance(label) + 12
            self._draw_overlay_label(painter, int(np.clip(wx + 6, 5, max(5, self.width() - label_width))),
                                     int(np.clip(wy, 20, max(20, self.height() - 8))), label)
            selected = record.get("selected")
            if selected and self.width() >= 500 and self.height() >= 280:
                frame = QRectF(self.width() - 194, self.height() - 180, 182, 168)
                content = frame.adjusted(1, 25, -1, -1)
                crop = QRectF(selected["x"] - 16, selected["y"] - 16, 32, 32)
                painter.save()
                painter.setPen(QPen(QColor(ui_color("#D8DEE6", "border")), 1))
                painter.setBrush(QColor(ui_color("#FFFFFF", "background")))
                painter.drawRect(frame)
                painter.setPen(QColor(ui_color("#20242B")))
                painter.drawText(frame.adjusted(7, 2, -2, -140), f"局部放大 · 卡尺 {self.selected + 1}")
                painter.setClipRect(content)
                # Square source pixels stay square in the magnifier.
                side = min(content.width(), content.height())
                target = QRectF(content.center().x() - side / 2, content.top(), side, side)
                painter.drawPixmap(target, self.pixmap_cache, crop)
                for preview, color, show in ((self.baseline, CYAN, self.kind != "current"),
                                             (self.current, AMBER, self.kind != "baseline")):
                    if preview is None or preview.circle is None or not show:
                        continue
                    for values, rejected in ((preview.circle.edge_points, False), (preview.circle.rejected_points, True)):
                        if rejected and preview is not self.current:
                            continue
                        painter.setPen(QPen(QColor(RED if rejected else color), 1.5))
                        for x, y in values:
                            pos = QPointF(target.left() + (x - crop.left()) / 32 * side,
                                          target.top() + (y - crop.top()) / 32 * side)
                            if not target.contains(pos):
                                continue
                            if rejected:
                                painter.drawLine(pos + QPointF(-3, -3), pos + QPointF(3, 3))
                                painter.drawLine(pos + QPointF(-3, 3), pos + QPointF(3, -3))
                            else:
                                painter.drawEllipse(pos, 2.5, 2.5)
                painter.restore()
        painter.end()


class ProfileComparisonPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(180)
        self.source = self.baseline = self.current = None
        self.selected = 0
        self.raw = np.empty(0)

    def set_data(self, source, baseline, current, selected):
        self.source, self.baseline, self.current, self.selected = source, baseline, current, selected
        reference = current or baseline
        self.raw = raw_profile(source, reference.records[selected]) if reference and selected < len(reference.records) else np.empty(0)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(ui_color("#FFFFFF", "background")))
        for column in range(2):
            bounds = QRectF(column * self.width() / 2 + 57, 34, self.width() / 2 - 84, self.height() - 77)
            title = f"卡尺 {self.selected + 1} · " + ("灰度剖面（原始 / 基准 / 当前）" if column == 0 else "梯度与选峰（基准 / 当前）")
            p.setPen(QColor(ui_color("#20242B")))
            p.drawText(QPointF(bounds.left(), 20), title)
            series, thresholds, markers = [], [], []
            reference = self.current or self.baseline
            if column == 0 and reference and self.selected < len(reference.records):
                r = reference.records[self.selected]
                series.append((np.asarray(r["radial_samples"])[:len(self.raw)], self.raw, "#9a9a9f"))
            for result, color in ((self.baseline, CYAN), (self.current, AMBER)):
                if result is None or self.selected >= len(result.records):
                    continue
                r = result.records[self.selected]
                y = np.asarray(r["profile"] if column == 0 else r["gradient"])
                if column:
                    y = y if result.params.polarity == "Dark to Bright" else -y if result.params.polarity == "Bright to Dark" else np.abs(y)
                    thresholds.append((result.params.min_gradient, color))
                x = np.asarray(r["radial_samples"])[:len(y)]
                series.append((x, y, color))
                if column and r.get("selected"):
                    markers.append((r["selected_distance_px"], r["selected"]["gradient"], color))
            finite = [y[np.isfinite(y)] for _, y, _ in series if len(y)]
            lo = min((float(y.min()) for y in finite if len(y)), default=0)
            hi = max([float(y.max()) for y in finite if len(y)] + [v for v, _ in thresholds] + [lo + 1])
            if column:
                lo = 0
            hi += (hi - lo) * 0.08
            xmax = max((float(x[-1]) for x, _, _ in series if len(x)), default=1)
            if column:
                legend_x = bounds.right() - 230
                for label, result, color in (("基准门槛", self.baseline, CYAN), ("当前门槛", self.current, AMBER)):
                    if result is not None:
                        text = f"{label} {result.params.min_gradient:g}"
                        p.setPen(QColor(color))
                        p.drawText(QPointF(legend_x, 20), text)
                        legend_x += 115
            def xy(x, y):
                return QPointF(bounds.left() + x / max(xmax, 1e-12) * bounds.width(),
                               bounds.bottom() - (y - lo) / max(hi - lo, 1e-12) * bounds.height())
            p.setPen(QPen(QColor(ui_color("#D8DEE6", "border")), 1))
            p.drawRect(bounds)
            for i in range(5):
                v = lo + (hi - lo) * i / 4
                p.setPen(QColor(ui_color("#68717D")))
                p.drawText(QRectF(bounds.left() - 54, xy(0, v).y() - 8, 48, 16), Qt.AlignRight, f"{v:.1f}")
                x = xmax * i / 4
                p.drawText(QRectF(xy(x, lo).x() - 20, bounds.bottom() + 3, 40, 16), Qt.AlignCenter, f"{x:.1f}")
            p.drawText(QRectF(bounds.left(), bounds.bottom() + 22, bounds.width(), 20), Qt.AlignCenter,
                       "搜索行程 (px)" + (" · 灰度 DN" if column == 0 else " · 梯度 DN/px"))
            p.save()
            p.setClipRect(bounds.adjusted(-1, -1, 1, 1))
            for value, color in thresholds:
                p.setPen(QPen(QColor(color), 1, Qt.DashLine))
                p.drawLine(xy(0, value), xy(xmax, value))
            for x, y, color in series:
                path, started = QPainterPath(), False
                for xx, yy in zip(x, y):
                    if not np.isfinite(yy):
                        started = False
                    elif started:
                        path.lineTo(xy(float(xx), float(yy)))
                    else:
                        path.moveTo(xy(float(xx), float(yy)))
                        started = True
                p.setPen(QPen(QColor(color), 1.5))
                p.drawPath(path)
            for x, y, color in markers:
                p.setPen(QPen(QColor(color), 1, Qt.DotLine))
                p.drawLine(xy(x, lo), xy(x, hi))
                p.setBrush(QColor(color))
                p.drawEllipse(xy(x, y), 3, 3)
            p.restore()
        p.end()


class ParameterComparisonDialog(QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle(f"SOMA 调参对比 · V{__version__}")
        self.resize(1500, 950)
        self.setMinimumSize(1080, 720)
        self.source = self.baseline = self.current = None
        self.worker = None
        self._pending = None
        self._revision = 0
        self._dirty = True
        self._closing = False
        self._initial_focus_pending = True
        self._building = True
        self.choices = available_rois(main)
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        row = QHBoxLayout()
        row.addWidget(QLabel("<b>调参对比</b>　同一图像 / 同一 ROI · 预览不修改正式结果"))
        row.addStretch()
        self.rebase_button = QPushButton("固定当前预览为基准")
        self.rebase_button.clicked.connect(self.rebase)
        row.addWidget(self.rebase_button)
        root.addLayout(row)
        row = QHBoxLayout()
        self.mark_choice, self.layer_choice, self.roi_choice = QComboBox(), QComboBox(), QComboBox()
        for box in (self.mark_choice, self.layer_choice, self.roi_choice):
            row.addWidget(box)
        self.file_label = QLabel()
        self.file_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        row.addWidget(self.file_label, 1)
        self.sync_button = QPushButton("重新同步主页面")
        self.sync_button.clicked.connect(self.resync)
        row.addWidget(self.sync_button)
        root.addLayout(row)
        self.vertical = QSplitter(Qt.Vertical)
        middle = QSplitter(Qt.Horizontal)
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        legend = QLabel(f'<span style="color:{CYAN}">● 基准</span>　<span style="color:{AMBER}">● 当前</span>　<span style="color:{RED}">× 拟合剔除</span>　○ 未找到')
        bar.addWidget(legend, 1)
        self.view_mode = QComboBox()
        self.view_mode.addItems(["叠加", "并排"])
        self.view_mode.currentIndexChanged.connect(self.set_view_mode)
        bar.addWidget(self.view_mode)
        for text, action in (("适应窗口", self.fit_views), ("局部放大", self.zoom_selected)):
            b = QPushButton(text)
            b.clicked.connect(action)
            bar.addWidget(b)
        layout.addLayout(bar)
        self.canvases = QSplitter(Qt.Horizontal)
        self.canvas, self.second_canvas = ComparisonCanvas(), ComparisonCanvas()
        for canvas in (self.canvas, self.second_canvas):
            self.canvases.addWidget(canvas)
            canvas.caliperSelected.connect(self.select_caliper)
        self.canvas.viewChanged.connect(lambda: self.sync_view(self.canvas, self.second_canvas))
        self.second_canvas.viewChanged.connect(lambda: self.sync_view(self.second_canvas, self.canvas))
        self.second_canvas.hide()
        layout.addWidget(self.canvases, 1)
        layout.addWidget(QLabel("点击卡尺查看灰度、梯度和取点原因；滚轮缩放，中键或空格拖动平移。"))
        self.metrics = QTableWidget(4, 4)
        self.metrics.setObjectName("resultTable")
        self.metrics.setHorizontalHeaderLabels(["指标", "基准", "当前", "变化"])
        self.metrics.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics.setFixedHeight(124)
        self.metrics.verticalHeader().setMinimumSectionSize(18)
        self.metrics.verticalHeader().setDefaultSectionSize(22)
        self.metrics.verticalHeader().hide()
        self.metrics.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.metrics)
        middle.addWidget(left)
        panel = QWidget()
        inspector = QVBoxLayout(panel)
        inspector.setContentsMargins(12, 0, 0, 0)
        inspector.addWidget(QLabel("<b>参数预览</b>　基准 → 当前"))
        form = QFormLayout()
        self.controls, self.baseline_labels = {}, {}
        for field, title, maximum, decimals in (
            ("gaussian_sigma_px", "高斯 Sigma (px)", 10, 3),
            ("min_gradient", "最小梯度", 1000000, 3),
            ("residual_limit_px", "RANSAC 阈值 (px)", 1000, 4),
        ):
            box = QDoubleSpinBox()
            box.setRange(0.001 if field == "residual_limit_px" else 0, maximum)
            box.setDecimals(decimals)
            box.setSingleStep(0.05 if field != "min_gradient" else 1)
            baseline = QLabel("—")
            line = QHBoxLayout()
            line.addWidget(baseline)
            line.addWidget(QLabel("→"))
            line.addWidget(box)
            form.addRow(title, line)
            self.controls[field], self.baseline_labels[field] = box, baseline
            box.valueChanged.connect(self.mark_dirty)
        inspector.addLayout(form)
        self.ransac = QCheckBox("启用 RANSAC 异常点剔除")
        self.ransac.toggled.connect(self.mark_dirty)
        inspector.addWidget(self.ransac)
        inspector.addWidget(QLabel("RANSAC 实际剔除阈值下限为 0.10 px"))
        inspector.addWidget(QLabel("Canny 低 / 高阈值：不参与卡尺圆路径"))
        self.update_button = QPushButton("更新预览")
        self.update_button.setObjectName("primaryButton")
        self.update_button.clicked.connect(self.request_preview)
        inspector.addWidget(self.update_button)
        restore = QPushButton("恢复基准参数")
        restore.clicked.connect(self.restore_baseline)
        inspector.addWidget(restore)
        self.selected_label = QLabel("卡尺 1")
        inspector.addWidget(self.selected_label)
        self.point_info = QLabel("正在准备基准…")
        self.point_info.setWordWrap(True)
        inspector.addWidget(self.point_info)
        inspector.addWidget(QLabel("未找到边缘与拟合剔除分别统计。\n残差降低不代表边界一定正确。\n应用会更新全局算法参数，使旧结果失效。"))
        inspector.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(305)
        middle.addWidget(scroll)
        middle.setStretchFactor(0, 3)
        middle.setStretchFactor(1, 1)
        middle.setSizes([1080, 380])
        self.vertical.addWidget(middle)
        self.plot = ProfileComparisonPlot()
        self.vertical.addWidget(self.plot)
        self.vertical.setStretchFactor(0, 3)
        self.vertical.setStretchFactor(1, 1)
        self.vertical.setSizes([650, 230])
        root.addWidget(self.vertical, 1)
        row = QHBoxLayout()
        self.status = QLabel("预览未修改正式结果与配方")
        self.status.setWordWrap(True)
        row.addWidget(self.status, 1)
        self.apply_button = QPushButton("应用到正式参数")
        self.apply_button.setToolTip("更新主页面全局算法参数，影响其他 ROI；使旧结果失效，不自动保存配方")
        self.apply_button.clicked.connect(self.apply_parameters)
        row.addWidget(self.apply_button)
        close = QPushButton("关闭")
        close.clicked.connect(self.close)
        row.addWidget(close)
        root.addLayout(row)
        self.mark_choice.currentIndexChanged.connect(self.populate_layers)
        self.layer_choice.currentIndexChanged.connect(self.populate_rois)
        self.roi_choice.currentIndexChanged.connect(self.load_source)
        self._building = False
        self.populate_marks()
        self.freshness_timer = QTimer(self)
        self.freshness_timer.setInterval(800)
        self.freshness_timer.timeout.connect(self.check_freshness)
        self.freshness_timer.start()

    def populate_marks(self):
        self.mark_choice.blockSignals(True)
        self.mark_choice.clear()
        self.mark_choice.addItems(list(dict.fromkeys(c[0] for c in self.choices)))
        self.mark_choice.setCurrentText(self.main._current_mark_id())
        self.mark_choice.blockSignals(False)
        self.populate_layers()

    def populate_layers(self, *_):
        self.layer_choice.blockSignals(True)
        self.layer_choice.clear()
        for layer, label in (("upper", "上层图像"), ("lower", "下层图像")):
            if any(c[0] == self.mark_choice.currentText() and c[1] == layer for c in self.choices):
                self.layer_choice.addItem(label, layer)
        index = self.layer_choice.findData(self.main._current_layer())
        if index >= 0:
            self.layer_choice.setCurrentIndex(index)
        self.layer_choice.blockSignals(False)
        self.populate_rois()

    def populate_rois(self, *_):
        self.roi_choice.blockSignals(True)
        self.roi_choice.clear()
        for mark, layer, identity, label in self.choices:
            if mark == self.mark_choice.currentText() and layer == self.layer_choice.currentData():
                self.roi_choice.addItem(label, identity)
        index = self.roi_choice.findData(self.main.roi_index_combo.currentData())
        if index >= 0:
            self.roi_choice.setCurrentIndex(index)
        self.roi_choice.blockSignals(False)
        self.load_source()

    def resync(self):
        self.choices = available_rois(self.main)
        self.populate_marks()

    def load_source(self, *_):
        if self._building:
            return
        self._revision += 1
        self._pending = None
        if self.worker is not None:
            self.worker.requestInterruption()
        self.baseline = self.current = self.source = None
        self._initial_focus_pending = True
        for canvas in (self.canvas, self.second_canvas):
            canvas.source = None
            canvas.set_image(None)
        self.plot.set_data(None, None, None, 0)
        try:
            self.source = ComparisonSource.capture(self.main, self.mark_choice.currentText(),
                                                   self.layer_choice.currentData(), self.roi_choice.currentData())
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.status.setText("无法预览：" + str(exc))
            self.refresh_results()
            return
        self.file_label.setText(self.source.image.display_name + (f" · 第 {self.source.batch_index} 张" if self.source.batch_index else ""))
        self.file_label.setToolTip(self.source.image.path)
        self.set_controls(self.source.params)
        for canvas in (self.canvas, self.second_canvas):
            canvas.source = self.source
            canvas.set_image(self.source.image)
            canvas.selected = 0
        self.request_preview()

    def set_controls(self, params):
        for field, control in self.controls.items():
            control.blockSignals(True)
            control.setValue(getattr(params, field))
            control.blockSignals(False)
        self.ransac.blockSignals(True)
        self.ransac.setChecked(params.use_ransac)
        self.ransac.blockSignals(False)

    def preview_params(self):
        return replace(self.source.params, **{field: box.value() for field, box in self.controls.items()},
                       use_ransac=self.ransac.isChecked())

    def mark_dirty(self, *_):
        self._dirty = True
        self._revision += 1
        self._pending = None
        if self.worker is not None:
            self.worker.requestInterruption()
        self.status.setText("参数已修改，图中仍是上次预览；请点击“更新预览”。")
        self.apply_button.setEnabled(False)
        self.rebase_button.setEnabled(False)

    def request_preview(self):
        if self.source is None:
            return
        self._revision += 1
        self._dirty = True
        self._pending = (self.source, self.preview_params(), self._revision, self.baseline is None)
        self.apply_button.setEnabled(False)
        self.rebase_button.setEnabled(False)
        self.status.setText("正在计算预览… 正式结果与配方未修改")
        if self.worker is not None:
            self.worker.requestInterruption()
        else:
            self.start_pending()

    def start_pending(self):
        if self._pending is None or self._closing:
            return
        source, params, revision, needs_baseline = self._pending
        self._pending = None
        self.worker = PreviewWorker(source, params, revision, needs_baseline, self)
        worker = self.worker
        worker.completed.connect(self.preview_completed)
        worker.finished.connect(lambda w=worker: self.worker_finished(w))
        worker.start()

    def worker_finished(self, worker):
        if self.worker is worker:
            self.worker = None
        worker.deleteLater()
        if not self._closing:
            self.start_pending()

    def preview_completed(self, revision, baseline, current):
        if revision != self._revision or self._closing:
            return
        if self.baseline is None:
            self.baseline = baseline
        self.current = current
        self._dirty = False
        self.status.setText("预览失败：" + current.error if current.error else "预览完成 · 正式结果与配方未修改")
        self.refresh_results()
        if self._initial_focus_pending and current.circle is not None:
            self.focus_circle(current.circle)
            self._initial_focus_pending = False

    def refresh_results(self):
        for field, label in self.baseline_labels.items():
            label.setText(f"{getattr(self.baseline.params, field):g}" if self.baseline else "—")
        if self.baseline:
            self.ransac.setToolTip("基准：RANSAC " + ("开启" if self.baseline.params.use_ransac else "关闭"))
        for canvas in (self.canvas, self.second_canvas):
            canvas.baseline, canvas.current = self.baseline, self.current
            canvas.update()
        def values(result):
            if result is None:
                return [None] * 4
            return [result.valid_count, None if result.coverage is None else 100 * result.coverage,
                    result.residual_um, result.diameter_um]
        a, b = values(self.baseline), values(self.current)
        for row, title in enumerate(("有效点 / 卡尺数", "覆盖率 (%)", "参考残差 (μm)", "直径 (μm)")):
            cells = [title]
            for result, vals in ((self.baseline, a), (self.current, b)):
                value = vals[row]
                planned = int(np.clip(self.source.roi.caliper_count, 4, 720)) if self.source else 0
                cells.append("—" if value is None else f"{value} / {planned}" if row == 0 else f"{value:.4f}")
            cells.append("—" if a[row] is None or b[row] is None else f"{b[row] - a[row]:+d}" if row == 0 else f"{b[row] - a[row]:+.4f}" + (" 个百分点" if row == 1 else ""))
            for column, text in enumerate(cells):
                self.metrics.setItem(row, column, QTableWidgetItem(text))
        self.select_caliper(self.canvas.selected)
        self.check_freshness()

    def select_caliper(self, index):
        if self.source is None:
            self.point_info.setText("请选择有图像的卡尺圆 ROI")
            return
        self.canvas.selected = self.second_canvas.selected = index
        count = int(np.clip(self.source.roi.caliper_count, 4, 720))
        self.selected_label.setText(f"卡尺 {index + 1} · {index * 360 / count:.1f}°")
        lines = []
        for title, result in (("基准", self.baseline), ("当前", self.current)):
            if result and index < len(result.records):
                status = result.records[index]["status"]
                color = CYAN if title == "基准" else RED if status == "fit_rejected" else AMBER
                lines.append(f'<span style="color:{color}">{title}：{STATUS.get(status, "未完成")}</span>')
        if self.current:
            lines.append(escape(point_explanation(self.current, index)).replace("\n", "<br>"))
        self.point_info.setText("<br>".join(lines))
        self.plot.set_data(self.source, self.baseline, self.current, index)
        self.canvas.update()
        self.second_canvas.update()

    def rebase(self):
        if self.current and self.current.circle is not None and not self._dirty:
            self.baseline = self.current
            self.refresh_results()

    def restore_baseline(self):
        if self.baseline:
            self.set_controls(self.baseline.params)
            self.request_preview()

    def check_freshness(self):
        fresh = self.source is not None and self.source.is_current(self.main)
        usable = self.current is not None and self.current.circle is not None and not self._dirty
        different = bool(self.source and self.current and any(
            getattr(self.source.params, field) != getattr(self.current.params, field) for field in TUNABLE_FIELDS))
        self.apply_button.setEnabled(bool(fresh and usable and different))
        self.rebase_button.setEnabled(bool(usable))
        if self.source is not None and not fresh:
            self.status.setText("主页面图像、ROI、参数或工作状态已改变；此处保留旧快照，请重新同步后应用。")

    def apply_parameters(self):
        if self._dirty or not self.current or self.current.circle is None or not self.source or not self.source.is_current(self.main):
            self.check_freshness()
            return False
        changed = any(getattr(self.main.params, field) != getattr(self.current.params, field) for field in TUNABLE_FIELDS)
        if not changed:
            self.status.setText("预览参数与正式参数相同，无需应用。")
            return True
        for field in TUNABLE_FIELDS:
            setattr(self.main.params, field, getattr(self.current.params, field))
        for field, name in (("gaussian_sigma_px", "sigma_spin"), ("min_gradient", "min_gradient_spin"),
                            ("residual_limit_px", "residual_limit_spin"), ("use_ransac", "ransac_check")):
            widget = getattr(self.main, name)
            widget.blockSignals(True)
            if field == "use_ransac":
                widget.setChecked(self.current.params.use_ransac)
            else:
                widget.setValue(getattr(self.current.params, field))
            widget.blockSignals(False)
        self.main.invalidate_measurement_state("调参对比参数已应用", clear_selections=False)
        self.main._refresh_all_widgets()
        self.source.params = replace(self.main.params)
        self.status.setText("已应用到正式算法参数；旧结果已失效，请重新测量。配方文件尚未保存。")
        self.apply_button.setEnabled(False)
        return True

    def sync_view(self, source, target):
        source._update_transform()
        target._update_transform()
        target.user_zoom = source.scale / max(target.fit_scale, 1e-12)
        target.pan_x, target.pan_y = source.pan_x, source.pan_y
        target.update()

    def set_view_mode(self, index):
        self.canvas._update_transform()
        scale = self.canvas.scale
        self.second_canvas.setVisible(bool(index))
        self.canvas.kind = "baseline" if index else "overlay"
        self.second_canvas.kind = "current"
        self.canvas.title = "基准边缘" if index else "基准 / 当前边缘叠加"
        self.second_canvas.title = "当前边缘"
        if index:
            self.canvases.setSizes([max(1, self.canvases.width() // 2)] * 2)
        def restore_scale():
            self.canvas._update_transform()
            self.canvas.user_zoom = scale / max(self.canvas.fit_scale, 1e-12)
            self.sync_view(self.canvas, self.second_canvas)
            self.canvas.update()
        QTimer.singleShot(0, self, restore_scale)

    def fit_views(self):
        for canvas in (self.canvas, self.second_canvas):
            canvas.reset_view()

    def focus_circle(self, circle):
        canvas = self.canvas
        canvas._update_transform()
        scale = 0.72 * min(canvas.width(), canvas.height()) / max(2 * circle.radius_px, 1)
        canvas.user_zoom = scale / max(canvas.fit_scale, 1e-12)
        canvas._update_transform()
        x, y = canvas.image_to_widget(circle.center_x_px, circle.center_y_px)
        canvas.pan_x += canvas.width() / 2 - x
        canvas.pan_y += canvas.height() / 2 - y
        self.sync_view(canvas, self.second_canvas)
        canvas.update()

    def zoom_selected(self):
        if not self.current or self.canvas.selected >= len(self.current.records):
            return
        r = self.current.records[self.canvas.selected]
        point = r.get("selected")
        if point:
            position = QPointF(*self.canvas.image_to_widget(point["x"], point["y"])).toPoint()
            self.canvas.zoom_by(2, position)

    def stop_worker(self):
        self._pending = None
        self._revision += 1
        if self.worker is not None:
            self.worker.requestInterruption()
            return self.worker.wait(5000)
        return True

    def prepare_close(self):
        if self._closing:
            return True
        self._closing = True
        if not self.stop_worker():
            self._closing = False
            self.status.setText("正在停止预览，请稍后关闭。")
            return False
        self.freshness_timer.stop()
        return True

    def done(self, result):
        if self.prepare_close():
            super().done(result)

    def reject(self):
        self.done(QDialog.Rejected)

    def accept(self):
        self.done(QDialog.Accepted)

    def closeEvent(self, event):
        if not self.prepare_close():
            event.ignore()
            return
        super().closeEvent(event)
