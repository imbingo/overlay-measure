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
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QFontDatabase, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
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

from . import __version__
from .auto_mark_detector import detect_auto_marks_with_report
from .access_control import AccessController
from .batch_pairing import validate_batch_pairing
from .export_naming import build_export_filename
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
from .recipe_manager import load_recipe, load_recipe_with_geometry, save_recipe
from .recipe_library import RecipeLibrary, RecipeLibraryEntry
from .recent_image_store import RecentImageStore
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
from .ui_workers import MeasurementWorker, PreviewWorker
from .ui_frameless import FramelessWindowMixin
from .ui_builders import MainWindowBuilderMixin
from .ui_state import MainWindowStateMixin
from .ui_workflows import MainWindowWorkflowMixin
from .ui_recipe_actions import MainWindowRecipeMixin
from .ui_geometry import MainWindowGeometryMixin


def application_icon_path() -> Path:
    runtime_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return runtime_root / "assets" / "overlay_measure_icon.png"


class MainWindow(
    FramelessWindowMixin,
    QMainWindow,
    MainWindowGeometryMixin,
    MainWindowBuilderMixin,
    MainWindowStateMixin,
    MainWindowWorkflowMixin,
    MainWindowRecipeMixin,
):
    def __init__(self):
        super().__init__()
        for font_path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
            if font_path.exists() and QFontDatabase.addApplicationFont(str(font_path)) >= 0:
                break
        self.setFont(QFont("Microsoft YaHei UI", 9))
        self.setWindowTitle(f"SOMA Vision Metrology V{__version__}")
        icon_path = application_icon_path()
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setMinimumSize(1120, 720)
        self.resize(1500, 920)

        self.config = MeasurementConfig()
        self.params = DetectionParams()
        self._updating_quality_controls = False
        self.upper_image: Optional[ImageData] = None
        self.lower_image: Optional[ImageData] = None
        self.marks: Dict[str, MarkRecipe] = {"Mark1": MarkRecipe("Mark1"), "Mark2": MarkRecipe("Mark2")}
        self.mark_images: Dict[str, Dict[str, Optional[ImageData]]] = {
            "Mark1": {"upper": None, "lower": None},
            "Mark2": {"upper": None, "lower": None},
        }
        self.mark_image_sources = self._empty_image_sources()
        self.detections: Dict[str, Dict[str, DetectionResult]] = {}
        # Full manual ROI results keyed by stable ROI id. ``detections`` remains
        # the V2.0 compatibility projection (first result on each layer).
        self.roi_detections: Dict[str, Dict[str, DetectionResult]] = {
            "Mark1": {},
            "Mark2": {},
        }
        self.roi_detection_failures = {"Mark1": [], "Mark2": []}
        self._pending_new_roi = False
        self._roi_undo_stack = []
        self._roi_redo_stack = []
        self._restoring_roi_history = False
        self._initialize_roi_parameter_edit()
        self._manual_selection_requires_review = set()
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
        self._batch_detail_run_index = 1
        self._batch_detail_last_single_index = 1
        self.roi_sources = self._empty_roi_sources()
        self.loaded_recipe_path = ""
        self.loaded_recipe_display_name = ""
        self.loaded_recipe_hash = ""
        self.recipe_integrity_status = "Unsealed"
        self.recipe_library = RecipeLibrary()
        self.recent_image_store = RecentImageStore()
        self.recipe_quick_menu: Optional[RecipeQuickMenu] = None
        self.access_controller = AccessController()
        self.operation_mode = "Production"
        self.runtime_logger = build_runtime_logger()
        self.recovery_store = RecoveryStore()
        self.last_measurement_id = ""
        self.last_archive_path = ""
        self._calculation_timed_out = False
        self._calculation_timeout_timer = QTimer(self)
        self._calculation_timeout_timer.setSingleShot(True)
        self._calculation_timeout_timer.timeout.connect(self._on_calculation_timeout)
        self._recipe_roi_confirmation_signature = None
        self._calculation_thread: Optional[QThread] = None
        self._calculation_worker: Optional[MeasurementWorker] = None
        self._calculation_running = False
        self._preview_thread: Optional[QThread] = None
        self._preview_worker: Optional[PreviewWorker] = None
        self._preview_show_message = True
        self.step_rows = []
        self._initialize_geometry_state()

        self._apply_window_style()
        self._build_ui()
        self._connect_actions()
        self._roi_undo_action = QAction("撤销 ROI", self)
        self._roi_undo_action.setShortcut("Ctrl+Z")
        self._roi_undo_action.triggered.connect(self.undo_roi_change)
        self.addAction(self._roi_undo_action)
        self._roi_redo_action = QAction("重做 ROI", self)
        self._roi_redo_action.setShortcut("Ctrl+Y")
        self._roi_redo_action.triggered.connect(self.redo_roi_change)
        self.addAction(self._roi_redo_action)
        self._refresh_all_widgets()
        self._apply_operation_mode()
        if QApplication.platformName().lower() != "offscreen":
            QTimer.singleShot(0, self._offer_recovery)
            QTimer.singleShot(200, self._warn_default_engineering_password)

    @staticmethod
    def _empty_roi_sources() -> dict:
        return {
            "Mark1": {"upper": "none", "lower": "none"},
            "Mark2": {"upper": "none", "lower": "none"},
        }

    @staticmethod
    def _empty_image_sources() -> dict:
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







































































































































    def toggle_maximized(self):
        if self.isMaximized():
            self.showNormal()
            self.maximize_btn.setText("□")
            self.maximize_btn.setToolTip("最大化")
        else:
            self.showMaximized()
            self.maximize_btn.setText("❐")
            self.maximize_btn.setToolTip("还原")

    def _update_summary_typography(self):
        if not hasattr(self, "summary_panel"):
            return
        cell_width = max(1, self.summary_panel.width() // 4)
        if cell_width >= 235:
            point_size = 27
        elif cell_width >= 185:
            point_size = 23
        elif cell_width >= 145:
            point_size = 19
        else:
            point_size = 16
        for label in (
            self.dx_value_label,
            self.dy_value_label,
            self.r_value_label,
            self.result_value_label,
        ):
            font = label.font()
            font.setPointSize(point_size)
            font.setWeight(QFont.Bold)
            label.setFont(font)
            color = label.property("resultColor") or "#1D1D1F"
            label.setStyleSheet(f"color: {color}; font-size: {point_size}px; font-weight: 700;")

    def _update_toolbar_density(self):
        if not hasattr(self, "import_upper_btn"):
            return
        compact = self.width() < 1280
        labels = {
            self.import_upper_btn: "导入上层" if compact else "导入上层/单图",
            self.import_lower_btn: "导入下层" if compact else "导入下层图像",
            self.analyze_all_btn: "运行" if compact else "运行测量程序",
            self.export_btn: "导出" if compact else "导出结果",
        }
        for button, text in labels.items():
            button.setText(text)
        self.reset_measurement_btn.setText("重置")
        self.analyze_roi_btn.setText("分析 ROI")
        self.display_enhance_check.setText("增强" if compact else "显示增强")
        self.image_status_label.setVisible(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_summary_typography()
        self._update_toolbar_density()

    def closeEvent(self, event):
        for thread, worker in (
            (self._preview_thread, self._preview_worker),
            (self._calculation_thread, self._calculation_worker),
        ):
            if thread is not None and thread.isRunning():
                if worker is not None:
                    worker.cancel()
                if not thread.wait(5000):
                    event.ignore()
                    QMessageBox.information(
                        self,
                        "正在停止后台任务",
                        "后台算法正在安全退出。请稍候，再次关闭窗口。",
                    )
                    return
        super().closeEvent(event)













def run_app():
    smoke_test = "--smoke-test" in sys.argv
    if smoke_test:
        # The packaged release builder uses the offscreen platform to verify
        # that Qt plugins and all modular UI imports are present.
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication(sys.argv)
    icon_path = application_icon_path()
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    win = MainWindow()
    win.show()
    if smoke_test:
        QTimer.singleShot(300, app.quit)
    return_code = app.exec()
    if smoke_test:
        return return_code
    sys.exit(return_code)
