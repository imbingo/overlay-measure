"""Small main-page companion: no image imports, ROI editors or point-picking."""

from datetime import datetime
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QButtonGroup,
    QSplitter,
    QFileDialog,
    QMessageBox,
)
from .. import __version__
from .models import DiagnosticsSession
from .measurement_bridge import snapshot_main, align_measurement, export_measurement
from .measurement_profile import measurement_profile
from .ui_diagnostics import DiagnosticCanvas, ComparisonChart, DiagnosticsDialog


class MeasurementDiagnosticsDialog(QDialog):
    profile_marker_title = "梯度 DN/μm · 虚线为主页面边缘位置"

    def __init__(self, params, config, parent=None, source_provider=None):
        super().__init__(parent)
        self.setWindowTitle(f"SOMA 成像诊断 · V{__version__}")
        self.resize(1400, 920)
        self.setMinimumSize(1000, 700)
        self.provider = source_provider or (lambda: snapshot_main(parent))
        self.snapshot = None
        self.session = DiagnosticsSession.from_parameters(params, config)
        self.mode = 0
        self.angle = 0.0
        self.busy = False
        self.worker = None
        self.profile = None
        self._profile_key = None
        self.filtered_cache = {}
        self.setStyleSheet(
            "QDialog{background:#F5F7FA;} QPushButton,QComboBox{background:white;border:1px solid #D6DCE4;border-radius:5px;padding:7px 12px;} QPushButton:checked{background:#EAF3FF;color:#087EF4;} QLabel{color:#354152;}"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)
        row = QHBoxLayout()
        row.addWidget(QLabel("<b>成像诊断</b>　直接使用主页面上下层图像与识别结果"))
        row.addStretch()
        sync = QPushButton("同步主页面")
        sync.clicked.connect(self.sync_main)
        row.addWidget(sync)
        self.rotation = QComboBox()
        self.rotation.addItems(
            ["方向不变", "下层旋转 90°", "下层旋转 180°", "下层旋转 270°"]
        )
        self.rotation.currentIndexChanged.connect(self.invalidate)
        row.addWidget(self.rotation)
        self.align_button = QPushButton("自动配准")
        self.align_button.setStyleSheet(
            "background:#087EF4;color:white;font-weight:600;"
        )
        self.align_button.clicked.connect(self.auto_align)
        row.addWidget(self.align_button)
        self.export_button = QPushButton("导出")
        self.export_button.clicked.connect(self.export)
        row.addWidget(self.export_button)
        root.addLayout(row)
        self.source_label = QLabel("请在主页面导入上下层图像并运行识别。")
        self.source_label.setWordWrap(True)
        root.addWidget(self.source_label)
        self.pairs = QWidget()
        pairs = QHBoxLayout(self.pairs)
        pairs.setContentsMargins(0, 0, 0, 0)
        self.upper_choice = QComboBox()
        self.lower_choice = QComboBox()
        pairs.addWidget(QLabel("比较轮廓"))
        pairs.addWidget(self.upper_choice)
        pairs.addWidget(QLabel("与"))
        pairs.addWidget(self.lower_choice)
        pairs.addStretch()
        for box in (self.upper_choice, self.lower_choice):
            box.currentIndexChanged.connect(self.invalidate)
        root.addWidget(self.pairs)
        tabs = QHBoxLayout()
        self.views = QButtonGroup(self)
        for i, text in enumerate(["边缘叠加", "局部位移", "灰度 / 梯度"]):
            button = QPushButton(text)
            button.setCheckable(True)
            self.views.addButton(button, i)
            tabs.addWidget(button)
        self.views.button(0).setChecked(True)
        self.views.idClicked.connect(self.set_mode)
        root.addLayout(tabs)
        row = QHBoxLayout()
        for attr, text, checked in [
            ("show_a", "上层边缘（青色）", True),
            ("show_b", "下层边缘（橙色）", True),
            ("show_fits", "拟合圆", False),
            ("show_rejected", "剔除点", True),
        ]:
            check = QCheckBox(text)
            check.setChecked(checked)
            check.toggled.connect(self.refresh_selection)
            setattr(self, attr, check)
            row.addWidget(check)
        self.show_calipers = QCheckBox(self)
        self.show_calipers.hide()
        self.centered = QCheckBox(self)
        self.centered.hide()
        self.amplification = QDoubleSpinBox()
        self.amplification.setRange(1, 100)
        self.amplification.setValue(4)
        self.amplification.setPrefix("位移线 ×")
        self.amplification.valueChanged.connect(self.refresh_selection)
        self.amplification.hide()
        row.addWidget(self.amplification)
        row.addStretch()
        for title in ["适应窗口", "1:1", "局部放大"]:
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, t=title: self.zoom(t))
            row.addWidget(button)
        root.addLayout(row)
        self.splitter = QSplitter(Qt.Vertical)
        self.overlay = DiagnosticCanvas(self, "overlay")
        self.overlay.title = "主页面轮廓 · 点击“自动配准”"
        self.overlay.setMinimumHeight(330)
        self.splitter.addWidget(self.overlay)
        lower = QWidget()
        chart_layout = QVBoxLayout(lower)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.curve = QComboBox()
        self.curve.addItems(["带符号位移", "绝对位移", "上下层径向位置"])
        self.curve.currentIndexChanged.connect(self.refresh_selection)
        row.addWidget(self.curve)
        reset = QPushButton("全部角度")
        reset.clicked.connect(self.reset_angles)
        row.addWidget(reset)
        self.selected_label = QLabel()
        row.addWidget(self.selected_label)
        row.addStretch()
        chart_layout.addLayout(row)
        self.chart = ComparisonChart(self)
        self.chart.setMinimumHeight(170)
        chart_layout.addWidget(self.chart)
        self.splitter.addWidget(lower)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([540, 220])
        root.addWidget(self.splitter, 1)
        self.metrics = QLabel("")
        self.metrics.setWordWrap(True)
        root.addWidget(self.metrics)
        self.status = QLabel("在主页面完成识别后，只需点击“自动配准”。")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.sync_main()

    def sync_main(self):
        try:
            snapshot = self.provider()
        except Exception as exc:
            self.status.setText("暂时无法同步主页面：" + str(exc))
            return
        self.snapshot = snapshot
        self.session = snapshot.session
        self.filtered_cache.clear()
        for box, entries, preferred in [
            (self.upper_choice, snapshot.upper, snapshot.selected_upper),
            (self.lower_choice, snapshot.lower, snapshot.selected_lower),
        ]:
            box.blockSignals(True)
            box.clear()
            for key, detection in entries.items():
                box.addItem(
                    detection.shape_params.get("roi_label")
                    or detection.shape_params.get("candidate_display_label")
                    or str(key),
                    key,
                )
            index = box.findData(preferred)
            box.setCurrentIndex(max(0, index))
            box.blockSignals(False)
        self.pairs.setVisible(len(snapshot.upper) > 1 or len(snapshot.lower) > 1)
        self.overlay.set_image(
            self.session.a.oriented if self.session.a is not None else None
        )
        if self.session.a:
            self.overlay.pixel_size_x_um = self.session.a.calibration.x
            self.overlay.pixel_size_y_um = self.session.a.calibration.y
        ready = (
            self.session.a is not None
            and self.session.b is not None
            and bool(snapshot.upper)
            and bool(snapshot.lower)
        )
        self.align_button.setEnabled(ready)
        self.export_button.setEnabled(False)
        if ready:
            self.source_label.setText(
                f"{snapshot.label}　上层：{self.session.a.original.display_name}　 /　 下层：{self.session.b.original.display_name}"
            )
            self.status.setText(
                "主页面轮廓已就绪，点击“自动配准”即可。无需重新导图、画 ROI 或选择对应点。"
            )
        else:
            self.source_label.setText("尚无完整的上下层圆孔结果")
            self.status.setText(
                "请在主页面切换双图模式、导入上下层图像并运行识别，然后点击“同步主页面”。"
            )
        self.session.result = None
        self.profile = None
        self._profile_key = None
        self.metrics.clear()
        self.refresh_selection()

    def invalidate(self, *args):
        self.session.result = None
        self.profile = None
        self._profile_key = None
        self.filtered_cache.clear()
        self.export_button.setEnabled(False)
        self.metrics.clear()
        self.status.setText("比较对象或方向已变化，点击“自动配准”更新。")
        self.refresh_selection()

    def auto_align(self):
        if self.snapshot is None:
            return
        try:
            align_measurement(
                self.snapshot,
                self.upper_choice.currentData(),
                self.lower_choice.currentData(),
                self.rotation.currentIndex(),
            )
            self.overlay.title = "上下层边缘 · 主页面原始识别点"
            self.status.setText("；".join(self.session.result.warnings))
            self.export_button.setEnabled(True)
            self._profile_key = None
            self.refresh_selection()
            self.fit_contour()
        except Exception as exc:
            self.invalidate()
            self.status.setText(str(exc))

    def comparison(self):
        return self.session.result.registered

    def selected_index(self):
        return int(round(self.angle)) % 360

    def select_point(self, point):
        if self.session.result is None:
            return
        delta = (
            self.session.a.calibration.physical([point])[0]
            - self.session.result.center_a_um
        )
        self.angle = float(np.degrees(np.arctan2(delta[1], delta[0])) % 360)
        self.refresh_selection()

    def set_mode(self, mode):
        self.mode = mode
        self.views.button(mode).setChecked(True)
        self.amplification.setVisible(mode == 1)
        self.curve.setVisible(mode != 2)
        self.refresh_selection()
        if mode == 1 and self.session.result is not None:
            self.zoom("局部放大")

    def refresh_selection(self, *args):
        r = self.session.result
        if r is None:
            self.selected_label.clear()
            self.overlay.update()
            self.chart.update()
            return
        c = self.comparison()
        m = c.metrics
        i = self.selected_index()

        def f(value):
            return "—" if value is None else f"{value:.4f}"

        self.metrics.setText(
            f'边缘 RMS　{f(m["rms_um"])} μm　　 P-V　{f(m["pv_um"])} μm　　 最大偏移　{f(m["max_abs_um"])} μm　　 覆盖率　{m["coverage"]:.1%}　　 直径差　{r.circle_metrics["diameter_delta_um"]:+.4f} μm'
            + ("　【仅供查看：覆盖不足】" if not r.reliable else "")
        )
        text = f"{i}°　" + (
            f"位移 {c.radius_b_um[i]-c.radius_a_um[i]:+.5f} μm"
            if c.valid[i]
            else "缺少有效对应"
        )
        if self.mode == 2:
            try:
                key = (id(r), i)
                if key != self._profile_key:
                    self.profile = measurement_profile(
                        self.snapshot, i, self.filtered_cache
                    )
                    self._profile_key = key
                text += "　｜　虚线：主页面边缘位置；点线：剖面候选峰"
            except ValueError as exc:
                self.profile = None
                text = str(exc)
        self.selected_label.setText(text)
        self.overlay.update()
        self.chart.update()

    def fit_contour(self):
        result = self.session.result
        if result is None:
            self.overlay.reset_view()
            return
        points = np.vstack([result.a.edge_points, result.b.edge_points])
        low = points.min(0)
        high = points.max(0)
        center = (low + high) * 0.5
        self.overlay._update_transform()
        scale = min(
            self.overlay.width() / max((high - low)[0] * 1.18, 1),
            self.overlay.height() / max((high - low)[1] * 1.18, 1),
        )
        self.overlay.user_zoom = min(80, scale / max(self.overlay.fit_scale, 1e-12))
        scale = self.overlay.fit_scale * self.overlay.user_zoom
        bx, by = self.overlay._base_offset_for_scale(scale)
        self.overlay.pan_x = self.overlay.width() / 2 - (center[0] + 0.5) * scale - bx
        self.overlay.pan_y = self.overlay.height() / 2 - (center[1] + 0.5) * scale - by
        self.overlay.update()

    def zoom(self, action):
        if action == "适应窗口":
            self.fit_contour()
        else:
            DiagnosticsDialog.zoom(self, action)

    def reset_angles(self):
        self.chart.angle_range = (0.0, 360.0)
        self.chart.update()

    def export(self):
        if self.session.result is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not directory:
            return
        try:
            out = export_measurement(
                self.snapshot,
                self.upper_choice.currentData(),
                self.lower_choice.currentData(),
                Path(directory)
                / ("SOMA_Diagnostics_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")),
            )
            self.overlay.grab().save(str(out / "edge_overlay.png"))
            self.chart.grab().save(str(out / "current_curve.png"))
            if self.profile is not None:
                from ..result_exporter import export_data_tables
                from .export import json_value
                import json

                data = self.profile
                rows = []
                for i, x in enumerate(data["distance_um"]):
                    row = {"distance_um": float(x)}
                    for name in ("a", "b"):
                        for key in ("raw", "filtered", "gradient"):
                            value = data[name][key][i]
                            row[name + "_" + key] = (
                                float(value) if np.isfinite(value) else None
                            )
                    rows.append(row)
                export_data_tables(str(out / "csv"), {"selected_profile": rows})
                (out / "selected_profile.json").write_text(
                    json.dumps(data, default=json_value, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            self.status.setText("已导出：" + str(out))
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
