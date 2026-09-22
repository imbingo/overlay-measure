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

from . import __version__
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
from .result_table import ResultTableWidget
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


class MainWindowBuilderMixin:
        def _apply_window_style(self):
            self.setStyleSheet("""
                QMainWindow, QWidget { background: #F5F7FA; color: #20242B; }
                QMainWindow { border: 1px solid #C9CED6; }
                QLabel { background: transparent; }
                QFrame#titleBar { background: #FFFFFF; border: none; border-bottom: 1px solid #E3E7EC; }
                QFrame#commandBar { background: #FFFFFF; border: none; border-bottom: 1px solid #E3E7EC; }
                QLabel#brandDot { color: #2878D0; font-size: 15px; }
                QLabel#titleLabel { font-size: 16px; font-weight: 600; color: #1D2530; }
                QLabel#versionLabel { color: #2468B2; background: #EAF3FD; border: 1px solid #D6E8FA; border-radius: 6px; padding: 4px 9px; }
                QLabel#recipeMenuTitle { color: #1D2530; font-size: 14px; font-weight: 700; }
                QLabel#recipeMenuHint { color: #7A8491; }
                QLabel#recipeLabel, QLabel#statusCaption, QLabel#stepNote { color: #68717D; }
                QLabel#imageCardTitleUpper { color: #007AFF; font-size: 15px; font-weight: 700; }
                QLabel#imageCardTitleLower { color: #FF9500; font-size: 15px; font-weight: 700; }
                QLabel#resultTitle { font-size: 12px; color: #68717D; }
                QLabel#resultValue { font-size: 27px; font-weight: 700; color: #1D2530; }
                QLabel#resultUnit { font-size: 12px; color: #68717D; }
                QFrame#summaryCard, QFrame#imageCard, QFrame#tableCard { background: #FFFFFF; border: 1px solid #DEE3E9; border-radius: 7px; }
                QWidget#metricCell { background: transparent; border: none; }
                QGroupBox, QTableWidget, QPlainTextEdit { background: #FFFFFF; border: 1px solid #D8DEE6; border-radius: 7px; margin-top: 8px; padding-top: 8px; }
                QTableWidget#resultTable { margin-top: 0px; padding-top: 0px; border-radius: 5px; }
                QPlainTextEdit { padding: 8px; color: #20242B; font-family: "Microsoft YaHei UI"; font-size: 12px; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #20242B; font-weight: 600; }
                QPushButton { background: #FFFFFF; border: 1px solid #D6DCE4; border-radius: 6px; padding: 7px 12px; min-height: 20px; }
                QPushButton:hover { background: #F7F9FB; border-color: #B9C2CE; }
                QPushButton:pressed { background: #EEF2F6; }
                QPushButton#primaryButton { background: #087EF4; color: #FFFFFF; border-color: #087EF4; font-weight: 600; padding-left: 18px; padding-right: 18px; }
                QPushButton#primaryButton:hover { background: #006DDB; border-color: #006DDB; }
                QPushButton#commandAction, QToolButton#commandAction { background: #FFFFFF; border: 1px solid #D6DCE4; border-radius: 6px; padding: 6px 10px; font-weight: 500; }
                QPushButton#commandAction:hover, QToolButton#commandAction:hover { background: #F7F9FB; border-color: #B9C2CE; }
                QToolButton#commandMore { background: #FFFFFF; border: 1px solid #D6DCE4; border-radius: 6px; padding: 6px 9px; }
                QToolButton#commandMore:hover { background: #F7F9FB; border-color: #B9C2CE; }
                QPushButton#statusCancelButton { background: #FFFFFF; border: 1px solid #D6DCE4; border-radius: 5px; padding: 2px 9px; min-height: 0; font-size: 11px; }
                QPushButton#titleAction { border: none; background: transparent; padding: 5px 10px; min-height: 22px; }
                QPushButton#titleAction:hover { background: #F2F5F8; }
                QPushButton#recipeSwitcher { background: #F7F9FB; border: 1px solid #DCE2E9; border-radius: 7px; padding: 6px 12px; text-align: left; min-width: 190px; }
                QPushButton#recipeSwitcher:hover { background: #EEF5FC; border-color: #B9D2EB; }
                QComboBox#accessMode { background: #F1F7F3; color: #248A3D; border: 1px solid #CDE6D4; font-weight: 600; min-width: 94px; }
                QPushButton#zoomButton { min-width: 34px; max-width: 34px; padding: 6px 0; font-size: 16px; }
                QPushButton#windowButton { border: none; border-radius: 0; min-width: 36px; padding: 4px; background: transparent; font-size: 15px; }
                QPushButton#windowButton:hover { background: #EEF1F4; }
                QPushButton#closeButton { border: none; border-radius: 0; min-width: 38px; padding: 4px; background: transparent; font-size: 17px; }
                QPushButton#closeButton:hover { background: #E81123; color: #FFFFFF; }
                QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox { background: #FFFFFF; border: 1px solid #D6DCE4; border-radius: 6px; padding: 5px 7px; min-height: 18px; }
                QTabWidget::pane { border: 1px solid #E0E4E9; border-radius: 7px; background: #FFFFFF; }
                QTabBar::tab { background: #F5F7FA; border: 1px solid #E0E4E9; padding: 8px 12px; margin-right: 1px; border-top-left-radius: 5px; border-top-right-radius: 5px; }
                QTabBar::tab:selected { background: #FFFFFF; color: #087EF4; font-weight: 600; border-bottom-color: #FFFFFF; }
                QTabWidget#sideTabs QTabBar::tab { padding: 9px 7px; font-size: 11px; }
                QToolButton#sectionToggle { text-align: left; font-weight: 600; padding: 9px 10px; background: #FFFFFF; border: 1px solid #E0E4E9; border-radius: 7px; }
                QScrollArea { border: none; background: #FFFFFF; }
                QStatusBar { background: #FFFFFF; border-top: 1px solid #DEE3E9; color: #4B5563; min-height: 36px; }
                QStatusBar::item { border: none; }
                QProgressBar { border: none; border-radius: 5px; background: #EDF1F5; text-align: center; color: #68717D; }
                QProgressBar::chunk { background: #087EF4; border-radius: 4px; }
                QSplitter::handle { background: #EEF1F4; width: 7px; }
                QMenu#recipeQuickMenu { background: #FFFFFF; border: 1px solid #C9D1DB; border-radius: 8px; padding: 0; }
                QMenu#recipeQuickMenu QTreeWidget { border: 1px solid #E0E4E9; border-radius: 6px; background: #FFFFFF; alternate-background-color: #F8FAFC; }
                QMenu#recipeQuickMenu QHeaderView::section { background: #F3F5F7; color: #68717D; border: none; border-bottom: 1px solid #E0E4E9; padding: 6px; }
            """)

        def _build_ui(self):
            central = QWidget()
            root = QVBoxLayout(central)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)
            self.setCentralWidget(central)

            toolbar_card = FramelessTitleBar(self)
            self.title_bar = toolbar_card
            toolbar = QHBoxLayout(toolbar_card)
            toolbar.setContentsMargins(18, 5, 0, 5)
            toolbar.setSpacing(6)
            title_row = QHBoxLayout()
            title_row.setSpacing(9)
            self.brand_icon_label = QLabel()
            self.brand_icon_label.setFixedSize(36, 36)
            self.brand_icon_label.setPixmap(self.windowIcon().pixmap(32, 32))
            brand_text = QVBoxLayout()
            brand_text.setSpacing(0)
            self.title_label = QLabel("SOMA Vision Metrology")
            self.title_label.setObjectName("titleLabel")
            self.title_label.setMinimumWidth(178)
            self.brand_subtitle_label = QLabel("See Once, Measure All  ·  视觉轮廓与对位量测平台")
            self.brand_subtitle_label.setObjectName("statusCaption")
            self.version_label = QLabel(f"V{__version__}")
            self.version_label.setObjectName("versionLabel")
            self.operation_mode_combo = QComboBox()
            self.operation_mode_combo.setObjectName("accessMode")
            self.operation_mode_combo.addItem("生产模式", "Production")
            self.operation_mode_combo.addItem("工程模式", "Engineering")
            self.operation_mode_combo.setFixedWidth(112)
            brand_text.addWidget(self.title_label)
            brand_text.addWidget(self.brand_subtitle_label)
            title_row.addWidget(self.brand_icon_label)
            title_row.addLayout(brand_text)
            title_row.addWidget(self.version_label)
            title_row.addWidget(self.operation_mode_combo)
            title_row.addStretch(1)

            self.import_upper_btn = QPushButton("导入上层/单图")
            self.import_lower_btn = QPushButton("导入下层图像")
            self.load_recipe_btn = QPushButton("当前配方：未加载  ▾")
            self.recipe_manage_btn = QPushButton("配方管理")
            self.save_recipe_btn = QPushButton("保存配方")
            self.analyze_all_btn = QPushButton("运行测量程序")
            self.export_btn = QPushButton("导出结果")
            # Keep command buttons text-first. Native platform icons made this
            # compact engineering toolbar visually heavier than the V2.2 style.
            self.analyze_all_btn.setObjectName("primaryButton")
            self.load_recipe_btn.setObjectName("recipeSwitcher")
            self.recipe_manage_btn.setObjectName("titleAction")
            self.save_recipe_btn.setObjectName("titleAction")
            self.load_recipe_btn.setMinimumWidth(225)
            self.load_recipe_btn.setMaximumWidth(320)
            toolbar.addLayout(title_row, stretch=1)
            for btn in (self.load_recipe_btn, self.recipe_manage_btn, self.save_recipe_btn):
                toolbar.addWidget(btn)
            toolbar.addSpacing(10)
            self.minimize_btn = QPushButton("—")
            self.maximize_btn = QPushButton("□")
            self.close_btn = QPushButton("×")
            for button in (self.minimize_btn, self.maximize_btn):
                button.setObjectName("windowButton")
                button.setFixedSize(42, 45)
            self.close_btn.setObjectName("closeButton")
            self.close_btn.setFixedSize(46, 45)
            self.minimize_btn.setToolTip("最小化")
            self.maximize_btn.setToolTip("最大化")
            self.close_btn.setToolTip("关闭")
            self.minimize_btn.clicked.connect(self.showMinimized)
            self.maximize_btn.clicked.connect(self.toggle_maximized)
            self.close_btn.clicked.connect(self.close)
            toolbar.addWidget(self.minimize_btn)
            toolbar.addWidget(self.maximize_btn)
            toolbar.addWidget(self.close_btn)
            root.addWidget(toolbar_card)

            command_bar = QFrame()
            command_bar.setObjectName("commandBar")
            command_layout = QHBoxLayout(command_bar)
            command_layout.setContentsMargins(18, 9, 18, 9)
            command_layout.setSpacing(9)
            self.command_bar = command_bar

            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["单图模式", "双图模式"])
            self.mode_combo.setMinimumWidth(108)
            self.image_mode_label = QLabel("图像模式")
            self.display_enhance_check = QCheckBox("显示增强")
            self.display_enhance_check.setChecked(False)
            self.reset_measurement_btn = QPushButton("重置")
            self.zoom_out_btn = QPushButton("−")
            self.zoom_out_btn.setObjectName("zoomButton")
            self.zoom_level_combo = QComboBox()
            self.zoom_level_combo.addItems(["50%", "75%", "100%", "125%", "150%", "200%"])
            self.zoom_level_combo.setCurrentText("100%")
            self.zoom_level_combo.setMinimumWidth(78)
            self.zoom_in_btn = QPushButton("+")
            self.zoom_in_btn.setObjectName("zoomButton")
            self.reset_view_btn = QPushButton("□")
            self.reset_view_btn.setToolTip("适应窗口")
            self.magnifier_btn = QPushButton("放大镜")
            self.magnifier_btn.setCheckable(True)
            self.magnifier_btn.setToolTip("在图像上移动鼠标查看局部 3 倍放大")
            self.analyze_roi_btn = QPushButton("分析 ROI")
            self.analyze_current_btn = QPushButton("计算当前对位")
            self.analyze_current_btn.setVisible(False)
            self.image_status_label = QLabel("等待导入图像")
            self.image_status_label.setObjectName("statusCaption")
            self.image_status_label.setVisible(False)

            self.import_images_btn = QToolButton()
            self.import_images_btn.setText("导入图像")
            self.import_images_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
            self.import_images_btn.setPopupMode(QToolButton.InstantPopup)
            self.import_images_btn.setToolTip("导入单图、上层图像或下层图像")
            self.import_images_menu = QMenu(self.import_images_btn)
            self.import_upper_action = self.import_images_menu.addAction("导入上层/单图")
            self.import_lower_action = self.import_images_menu.addAction("导入下层图像")
            self.import_images_btn.setMenu(self.import_images_menu)

            self.recent_images_btn = QToolButton()
            self.recent_images_btn.setText("最近文件")
            self.recent_images_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
            self.recent_images_btn.setPopupMode(QToolButton.InstantPopup)
            self.recent_images_btn.setToolTip("快速重新导入最近使用的上层、下层或单图文件")
            self.recent_images_menu = QMenu(self.recent_images_btn)
            self.recent_images_menu.aboutToShow.connect(self.refresh_recent_images_menu)
            self.recent_images_btn.setMenu(self.recent_images_menu)

            self.more_actions_btn = QToolButton()
            self.more_actions_btn.setText("更多")
            self.more_actions_btn.setPopupMode(QToolButton.InstantPopup)
            self.more_actions_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            self.more_actions_btn.setToolTip("显示与测量会话操作")
            self.more_actions_menu = QMenu(self.more_actions_btn)
            self.recipe_manage_action = self.more_actions_menu.addAction("配方管理")
            self.save_recipe_action = self.more_actions_menu.addAction("保存当前配方")
            self.more_actions_menu.addSeparator()
            self.display_enhance_action = self.more_actions_menu.addAction("显示增强")
            self.display_enhance_action.setCheckable(True)
            self.fit_view_action = self.more_actions_menu.addAction("适应窗口")
            self.image_diagnostics_action = self.more_actions_menu.addAction("图像诊断与边缘分析")
            self.more_actions_menu.addSeparator()
            self.reset_measurement_action = self.more_actions_menu.addAction("重置测量")
            self.more_actions_btn.setMenu(self.more_actions_menu)

            command_layout.addWidget(self.image_mode_label)
            command_layout.addWidget(self.mode_combo)
            command_layout.addWidget(self.display_enhance_check)
            command_layout.addWidget(self.import_upper_btn)
            command_layout.addWidget(self.import_lower_btn)
            command_layout.addWidget(self.recent_images_btn)
            command_layout.addWidget(self.reset_measurement_btn)
            command_layout.addSpacing(10)
            command_layout.addWidget(self.zoom_out_btn)
            command_layout.addWidget(self.zoom_level_combo)
            command_layout.addWidget(self.zoom_in_btn)
            command_layout.addWidget(self.reset_view_btn)
            command_layout.addWidget(self.magnifier_btn)
            command_layout.addStretch(1)
            command_layout.addWidget(self.analyze_roi_btn)
            command_layout.addWidget(self.analyze_all_btn)
            command_layout.addWidget(self.export_btn)
            root.addWidget(command_bar)

            workspace = QWidget()
            workspace_layout = QVBoxLayout(workspace)
            workspace_layout.setContentsMargins(10, 8, 10, 8)
            workspace_layout.setSpacing(0)

            main_splitter = QSplitter(Qt.Horizontal)
            main_splitter.setChildrenCollapsible(False)
            workspace_layout.addWidget(main_splitter)
            root.addWidget(workspace, stretch=1)

            center = QWidget()
            center_layout = QVBoxLayout(center)
            center_layout.setContentsMargins(0, 0, 0, 0)
            center_layout.setSpacing(10)

            image_row = QHBoxLayout()
            image_row.setSpacing(8)
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
            self.batch_detail_bar = QWidget()
            batch_detail_layout = QHBoxLayout(self.batch_detail_bar)
            batch_detail_layout.setContentsMargins(0, 0, 0, 4)
            batch_detail_layout.setSpacing(8)
            batch_detail_layout.addWidget(QLabel("批次明细"))
            self.batch_detail_combo = QComboBox()
            self.batch_detail_combo.setMinimumWidth(130)
            self.batch_detail_combo.currentIndexChanged.connect(self._on_batch_detail_changed)
            batch_detail_layout.addWidget(self.batch_detail_combo)
            self.batch_detail_preview_label = QLabel("图像预览：第1次")
            self.batch_detail_preview_label.setObjectName("statusCaption")
            batch_detail_layout.addWidget(self.batch_detail_preview_label)
            batch_detail_layout.addStretch(1)
            self.batch_detail_bar.setVisible(False)
            detail_tab_layout.addWidget(self.batch_detail_bar)
            self.det_table = ResultTableWidget()
            self.det_table.setMinimumHeight(300)
            detail_tab_layout.addWidget(self.det_table)
            detail_tab_layout.addWidget(self.det_table.external_horizontal_bar)

            overlay_tab = QWidget()
            overlay_tab_layout = QVBoxLayout(overlay_tab)
            overlay_tab_layout.setContentsMargins(0, 0, 0, 0)
            self.overlay_table = ResultTableWidget()
            self.overlay_table.setMinimumHeight(300)
            overlay_tab_layout.addWidget(self.overlay_table)
            overlay_tab_layout.addWidget(self.overlay_table.external_horizontal_bar)

            repeat_tab = QWidget()
            repeat_layout = QVBoxLayout(repeat_tab)
            repeat_layout.setContentsMargins(0, 0, 0, 0)
            repeat_layout.setSpacing(8)
            self.repeat_table = ResultTableWidget()
            self.repeat_table.setMinimumHeight(300)
            repeat_layout.addWidget(self.repeat_table, stretch=1)
            repeat_layout.addWidget(self.repeat_table.external_horizontal_bar)

            geometry_tab = QWidget()
            geometry_layout = QVBoxLayout(geometry_tab)
            geometry_layout.setContentsMargins(0, 0, 0, 0)
            self.geometry_table = ResultTableWidget()
            self.geometry_table.setMinimumHeight(300)
            geometry_layout.addWidget(self.geometry_table)
            geometry_layout.addWidget(self.geometry_table.external_horizontal_bar)

            for table in (self.det_table, self.overlay_table, self.geometry_table, self.repeat_table):
                table.deleteRequested.connect(self._delete_result_rows)

            self.result_tabs.addTab(detail_tab, "识别明细")
            self.result_tabs.addTab(overlay_tab, "对位结果")
            self.result_tabs.addTab(geometry_tab, "尺寸结果")
            self.result_tabs.addTab(repeat_tab, "重复性分析")
            self.result_tabs.setMinimumHeight(255)
            result_layout.addWidget(self.result_tabs, stretch=1)
            center_layout.addWidget(result_card, stretch=4)
            main_splitter.addWidget(center)

            self.side_tabs = QTabWidget()
            self.side_tabs.setObjectName("sideTabs")
            for page, title in (
                (self._build_product_tab(), "① 产品信息"),
                (self._build_image_tab(), "② 图像导入"),
                (self._build_roi_tab(), "③ ROI 与轮廓"),
                (self._build_algo_tab(), "④ 算法参数"),
                (self._build_spec_tab(), "⑤ 结果导出"),
            ):
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setWidget(page)
                self.side_tabs.addTab(scroll, title)
            main_splitter.addWidget(self.side_tabs)
            self.side_tabs.setMinimumWidth(340)
            self.side_tabs.setMaximumWidth(480)
            main_splitter.setSizes([1030, 450])
            self.main_splitter = main_splitter
            self._install_progress_status_widgets()
            self._install_algorithm_path_status_button()

        def _install_progress_status_widgets(self):
            self.statusBar().setSizeGripEnabled(False)
            self.statusBar().setMinimumHeight(34)
            self.status_shell = QWidget()
            self.status_shell.setMinimumHeight(30)
            self.status_shell_layout = QHBoxLayout(self.status_shell)
            self.status_shell_layout.setContentsMargins(10, 2, 10, 2)
            self.status_shell_layout.setSpacing(10)
            self.status_task_dot = QLabel("●")
            self.status_task_dot.setStyleSheet("color: #A1A1A6;")
            self.status_task_label = QLabel("任务状态：等待导入图像")
            self.status_shell_layout.addWidget(self.status_task_dot)
            self.status_shell_layout.addWidget(self.status_task_label)

            self.current_recipe_label = QLabel("当前配方：未加载")
            self.current_recipe_label.setObjectName("recipeLabel")
            self.current_recipe_label.setMinimumWidth(180)
            self.current_recipe_label.setMaximumWidth(280)
            self.status_shell_layout.addWidget(self.current_recipe_label)

            progress_caption = QLabel("进度：")
            progress_caption.setObjectName("statusCaption")
            self.status_shell_layout.addWidget(progress_caption)

            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")
            self.progress_bar.setFixedSize(190, 17)
            self.progress_bar.setVisible(True)
            self.status_shell_layout.addWidget(self.progress_bar)

            self.cancel_progress_btn = QPushButton("取消计算")
            self.cancel_progress_btn.setObjectName("statusCancelButton")
            self.cancel_progress_btn.setFixedSize(76, 20)
            self.cancel_progress_btn.setVisible(False)
            self.cancel_progress_btn.setEnabled(False)
            self.cancel_progress_btn.clicked.connect(self.cancel_calculation)
            self.status_shell_layout.addWidget(self.cancel_progress_btn, 0, Qt.AlignVCenter)

            self.progress_stage_label = QLabel("当前阶段：等待导入图像")
            self.progress_stage_label.setObjectName("statusCaption")
            self.progress_stage_label.setMinimumWidth(170)
            self.progress_stage_label.setVisible(True)
            self.status_shell_layout.addWidget(self.progress_stage_label)
            self.status_shell_layout.addStretch(1)
            self.statusBar().addPermanentWidget(self.status_shell, 1)

        def _install_algorithm_path_status_button(self):
            self.algorithm_path_text = "暂无测量结果；分析 ROI 或自动识别后可查看实际算法路径。"
            self.algorithm_path_summary_label = QLabel("算法路径：暂无测量结果")
            self.algorithm_path_summary_label.setObjectName("statusCaption")
            self.algorithm_path_summary_label.setMinimumWidth(170)
            self.algorithm_path_summary_label.setMaximumWidth(330)
            self.algorithm_path_button = QToolButton()
            self.algorithm_path_button.setText("查看")
            self.algorithm_path_button.setAutoRaise(True)
            self.algorithm_path_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            self.algorithm_path_button.setToolTip(self.algorithm_path_text)
            self.status_shell_layout.addWidget(self.algorithm_path_summary_label)
            self.status_shell_layout.addWidget(self.algorithm_path_button)

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
            card = QWidget()
            card.setObjectName("metricCell")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(18, 8, 18, 8)
            layout.setSpacing(2)
            title_label = QLabel(title)
            title_label.setObjectName("resultTitle")
            value_label = QLabel("--")
            value_label.setObjectName("resultValue")
            value_label.setMinimumWidth(0)
            unit_label = QLabel("")
            unit_label.setObjectName("resultUnit")
            unit_label.setMinimumWidth(0)
            layout.addWidget(title_label)
            layout.addWidget(value_label)
            layout.addWidget(unit_label)
            return card, value_label, unit_label

        def _build_summary_panel(self) -> QWidget:
            card = QFrame()
            card.setObjectName("summaryCard")
            # V1.3.1：压缩顶部结果卡片高度，把更多垂直空间让给“对位结果/重复性分析”。
            card.setMaximumHeight(112)
            card.setMinimumHeight(96)
            layout = QHBoxLayout(card)
            layout.setContentsMargins(6, 4, 6, 4)
            layout.setSpacing(0)
            self.dx_card, self.dx_value_label, self.dx_unit_label = self._result_metric_card("ΔX")
            self.dy_card, self.dy_value_label, self.dy_unit_label = self._result_metric_card("ΔY")
            self.r_card, self.r_value_label, self.r_unit_label = self._result_metric_card("综合偏差 R")
            self.result_card, self.result_value_label, self.result_unit_label = self._result_metric_card("判定结果")
            for index, metric in enumerate((self.dx_card, self.dy_card, self.r_card, self.result_card)):
                if index:
                    separator = QFrame()
                    separator.setFrameShape(QFrame.VLine)
                    separator.setStyleSheet("color: #E0E4E9;")
                    layout.addWidget(separator)
                layout.addWidget(metric, stretch=1)

            self.summary_panel = card

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
            self.change_engineering_password_btn = QPushButton("修改工程模式密码")
            form_info.addRow("物料编码", self.material_code_edit)
            form_info.addRow("配方名称", self.recipe_name_edit)
            form_info.addRow("配方版本", self.recipe_version_edit)
            form_info.addRow("配方状态", self.recipe_status_combo)
            form_info.addRow("工序", self.process_name_edit)
            form_info.addRow("测量设备型号", self.equipment_model_edit)
            form_info.addRow("设备校准日期", self.calibration_date_edit)
            form_info.addRow("操作人员", self.operator_name_edit)
            form_info.addRow(self.change_engineering_password_btn)
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
            batch_form.addRow("测量方式", self.measurement_run_mode_combo)
            self.batch_import_source_combo = SidebarComboBox()
            self.batch_import_source_combo.addItem("选择图片", "Files")
            self.batch_import_source_combo.addItem("选择文件夹", "Folder")
            self.batch_recursive_check = QCheckBox("含子目录")
            self.batch_recursive_check.setChecked(False)
            self.batch_recursive_check.setEnabled(False)
            source_row = QHBoxLayout()
            source_row.addWidget(self.batch_import_source_combo, stretch=1)
            source_row.addWidget(self.batch_recursive_check)
            batch_form.addRow("导入来源", source_row)
            batch_layout.addLayout(batch_form)
            btn_row1 = QHBoxLayout()
            self.batch_import_mark1_upper_btn = QPushButton("追加 M1 上层/单图")
            self.batch_import_mark1_lower_btn = QPushButton("追加 M1 下层")
            btn_row1.addWidget(self.batch_import_mark1_upper_btn)
            btn_row1.addWidget(self.batch_import_mark1_lower_btn)
            btn_row2 = QHBoxLayout()
            self.batch_import_mark2_upper_btn = QPushButton("追加 M2 上层/单图")
            self.batch_import_mark2_lower_btn = QPushButton("追加 M2 下层")
            btn_row2.addWidget(self.batch_import_mark2_upper_btn)
            btn_row2.addWidget(self.batch_import_mark2_lower_btn)
            self.batch_clear_btn = QPushButton("清空批量图像")
            batch_layout.addLayout(btn_row1)
            batch_layout.addLayout(btn_row2)
            batch_layout.addWidget(self.batch_clear_btn)
            self.batch_image_table = QTableWidget()
            batch_layout.addWidget(self.batch_image_table)
            batch_note = QLabel(
                "可多选图片，也可选择文件夹自动导入；勾选“包含子文件夹”可扫描多级测量目录。重复文件会自动跳过。"
                "双图模式下，上下层按追加后的列表顺序一一配对；"
                "如需重新选择，请先清空批量图像。计算时会复用当前 Mark 的 ROI 模板和算法参数。"
            )
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
            self.roi_index_combo = SidebarComboBox()
            form_mark.addRow("ROI编号", self.roi_index_combo)
            roi_selection_hint = QLabel("空白处拖动新建；点击 ROI 编辑；空白单击或 Esc 取消选择；右键管理。")
            roi_selection_hint.setWordWrap(True)
            roi_selection_hint.setObjectName("statusCaption")
            form_mark.addRow(roi_selection_hint)
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
            self.roi_type_combo.addItem("椭圆区域（计算圆度）", "Ellipse")
            self.roi_type_combo.addItem("卡尺圆", "Caliper Circle")
            self.roi_type_combo.addItem("圆环区域", "Annulus")
            self.roi_type_combo.addItem("矩形环区域", "Rectangular Ring")
            self.roi_type_combo.addItem("近似直线", "Approximate Line")
            self.roi_type_combo.addItem("区域中心区域", "Region Center")
            self.roi_type_combo.addItem("稳健轮廓中心区域", "Robust Center")
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
            self.diameter_mode_combo = SidebarComboBox()
            self.diameter_mode_combo.addItem("平均直径（推荐）", "Average")
            self.diameter_mode_combo.addItem("最大直径", "Maximum")
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
            form_roi.addRow("ROI类型", self.roi_type_combo)
            form_roi.addRow("中心 X", self.center_x_spin)
            form_roi.addRow("中心 Y", self.center_y_spin)
            form_roi.addRow("内半径", self.inner_radius_spin)
            form_roi.addRow("外半径", self.outer_radius_spin)
            form_roi.addRow("卡尺数量", self.caliper_count_spin)
            form_roi.addRow("卡尺宽度", self.caliper_width_spin)
            form_roi.addRow("搜索方向", self.search_direction_combo)
            form_roi.addRow("目标边缘", self.target_edge_combo)
            form_roi.addRow("圆直径定义", self.diameter_mode_combo)
            form_roi.addRow("矩形环角度", self.roi_angle_spin)
            form_roi.addRow(self.three_point_circle_btn)
            roi_section.add_widget(group_roi)
            layout.addWidget(roi_section)

            contour_section = CollapsibleSection("轮廓测量", True)
            contour_section.add_widget(self._build_geometry_tab())
            layout.addWidget(contour_section)

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

            # Legacy fitting controls are kept as hidden compatibility state for
            # old recipes. Manual fitting is now selected by each ROI type.
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

            quality_section = CollapsibleSection("识别质量门槛", True)
            quality_group = QGroupBox("质量预设")
            quality_layout = QVBoxLayout(quality_group)
            quality_form = QFormLayout()
            self.quality_profile_combo = SidebarComboBox()
            self.quality_profile_combo.addItem("标准（默认）", "Standard")
            self.quality_profile_combo.addItem("宽容（来料质量较差）", "Tolerant")
            self.quality_profile_combo.addItem("超精确（高质量来料）", "UltraPrecise")
            quality_form.addRow("识别质量门槛", self.quality_profile_combo)
            quality_layout.addLayout(quality_form)
            self.quality_profile_hint = QLabel()
            self.quality_profile_hint.setWordWrap(True)
            self.quality_profile_hint.setObjectName("statusCaption")
            quality_layout.addWidget(self.quality_profile_hint)
            quality_section.add_widget(quality_group)
            layout.addWidget(quality_section)

            diagnostics_section = CollapsibleSection("图像诊断与边缘分析", False)
            diagnostics_group = QGroupBox("原始灰度诊断")
            diagnostics_layout = QVBoxLayout(diagnostics_group)
            diagnostics_hint = QLabel("在当前图像上拖动一条线，查看原始灰度剖面与统计；不修改原图或测量结果。")
            diagnostics_hint.setWordWrap(True)
            diagnostics_hint.setObjectName("statusCaption")
            self.image_diagnostics_btn = QPushButton("打开灰度剖面")
            diagnostics_layout.addWidget(diagnostics_hint)
            diagnostics_layout.addWidget(self.image_diagnostics_btn)
            diagnostics_section.add_widget(diagnostics_group)
            layout.addWidget(diagnostics_section)

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
            self.measurement_timeout_spin = SidebarSpinBox(); self.measurement_timeout_spin.setRange(10, 3600); self.measurement_timeout_spin.setValue(180)
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
            form.addRow("任务超时 (s)", self.measurement_timeout_spin)
            advanced_section.add_widget(group)
            production_group = QGroupBox("自动正式精测")
            production_form = QFormLayout(production_group)
            self.production_search_spin = SidebarDoubleSpinBox(); self.production_search_spin.setRange(2.0, 1000.0); self.production_search_spin.setDecimals(3); self.production_search_spin.setValue(8.0)
            production_form.addRow("自动搜索半宽 (px)", self.production_search_spin)
            advanced_section.add_widget(production_group)

            quality_detail_group = QGroupBox("质量门槛详细参数（特殊需求）")
            quality_detail_form = QFormLayout(quality_detail_group)
            self.conf_min_spin = SidebarDoubleSpinBox(); self.conf_min_spin.setRange(0, 1); self.conf_min_spin.setDecimals(3); self.conf_min_spin.setSingleStep(0.05); self.conf_min_spin.setValue(0.7)
            self.production_coverage_spin = SidebarDoubleSpinBox(); self.production_coverage_spin.setRange(0.0, 1.0); self.production_coverage_spin.setDecimals(3); self.production_coverage_spin.setValue(0.65)
            self.production_reject_spin = SidebarDoubleSpinBox(); self.production_reject_spin.setRange(0.0, 1.0); self.production_reject_spin.setDecimals(3); self.production_reject_spin.setValue(0.40)
            self.production_residual_spin = SidebarDoubleSpinBox(); self.production_residual_spin.setRange(0.000001, 1000000.0); self.production_residual_spin.setDecimals(6); self.production_residual_spin.setValue(0.30)
            self.production_deviation_spin = SidebarDoubleSpinBox(); self.production_deviation_spin.setRange(0.000001, 1000000.0); self.production_deviation_spin.setDecimals(6); self.production_deviation_spin.setValue(0.60)
            quality_detail_form.addRow("最低置信度", self.conf_min_spin)
            quality_detail_form.addRow("最低覆盖率", self.production_coverage_spin)
            quality_detail_form.addRow("最大异常点比例", self.production_reject_spin)
            quality_detail_form.addRow("最大残差 (μm)", self.production_residual_spin)
            quality_detail_form.addRow("最大轮廓偏差 (μm)", self.production_deviation_spin)
            advanced_section.add_widget(quality_detail_group)
            layout.addWidget(advanced_section)
            self._refresh_quality_profile_hint()
            layout.addStretch(1)
            return w

        def _build_spec_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)
            group = QGroupBox("判定规格")
            form = QFormLayout(group)
            self.dx_limit_spin = SidebarDoubleSpinBox(); self.dy_limit_spin = SidebarDoubleSpinBox(); self.r_limit_spin = SidebarDoubleSpinBox()
            for spin, val in [(self.dx_limit_spin, 0.5), (self.dy_limit_spin, 0.5), (self.r_limit_spin, 0.7)]:
                spin.setRange(0, 1000000); spin.setDecimals(6); spin.setSingleStep(0.1); spin.setValue(val)
            form.addRow("|ΔX| 上限 (μm)", self.dx_limit_spin)
            form.addRow("|ΔY| 上限 (μm)", self.dy_limit_spin)
            form.addRow("对位 R 上限 (μm)", self.r_limit_spin)
            layout.addWidget(group)
            note = QLabel("Excel 同时导出识别明细、对位结果、重复性与尺寸结果。")
            note.setObjectName("statusCaption")
            layout.addWidget(note)
            layout.addStretch(1)
            return w

        def _connect_actions(self):
            self.operation_mode_combo.currentIndexChanged.connect(self.on_operation_mode_changed)
            self.quality_profile_combo.currentIndexChanged.connect(self.on_quality_profile_changed)
            for control in (
                self.pixel_x_spin, self.pixel_y_spin, self.offset_x_spin, self.offset_y_spin,
                self.rx_angle_spin, self.ry_angle_spin, self.material_thickness_spin,
                self.sigma_spin, self.canny_low_spin, self.canny_high_spin, self.min_gradient_spin,
                self.profile_half_spin, self.profile_step_spin, self.residual_limit_spin,
                self.min_edge_points_spin, self.measurement_timeout_spin, self.production_search_spin,
                self.rz_l_spin, self.rz_limit_spin,
                self.dx_limit_spin, self.dy_limit_spin, self.r_limit_spin,
            ):
                control.editingFinished.connect(self.on_algorithm_parameter_edited)
            for control in (self.ransac_check,):
                control.toggled.connect(self.on_algorithm_parameter_edited)
            for control in (self.polarity_combo, self.fit_mode_combo, self.upper_fit_mode_combo, self.lower_fit_mode_combo, self.rz_layout_combo):
                control.currentIndexChanged.connect(self.on_algorithm_parameter_edited)
            for quality_control in (
                self.conf_min_spin,
                self.production_coverage_spin,
                self.production_reject_spin,
                self.production_residual_spin,
                self.production_deviation_spin,
            ):
                quality_control.valueChanged.connect(self.on_quality_threshold_edited)
            self.change_engineering_password_btn.clicked.connect(self.change_engineering_password)
            self.image_diagnostics_btn.clicked.connect(self.start_image_diagnostics)
            self.import_upper_btn.clicked.connect(self.import_upper_image)
            self.import_lower_btn.clicked.connect(self.import_lower_image)
            self.import_upper_action.triggered.connect(lambda checked=False: self.import_upper_image())
            self.import_lower_action.triggered.connect(lambda checked=False: self.import_lower_image())
            self.recipe_manage_action.triggered.connect(lambda checked=False: self.show_recipe_manager())
            self.save_recipe_action.triggered.connect(lambda checked=False: self.save_recipe_file())
            self.image_diagnostics_action.triggered.connect(lambda checked=False: self.start_image_diagnostics())
            self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
            self.display_enhance_check.toggled.connect(self.on_display_enhancement_changed)
            self.display_enhance_check.toggled.connect(self.display_enhance_action.setChecked)
            self.display_enhance_action.toggled.connect(self.display_enhance_check.setChecked)
            self.reset_measurement_btn.clicked.connect(self.reset_measurement)
            self.reset_measurement_action.triggered.connect(lambda checked=False: self.reset_measurement())
            self.fit_view_action.triggered.connect(lambda checked=False: self.reset_canvas_views())
            self.zoom_in_btn.clicked.connect(lambda: self.zoom_canvases(1.25))
            self.zoom_out_btn.clicked.connect(lambda: self.zoom_canvases(0.8))
            self.zoom_level_combo.currentTextChanged.connect(self.set_canvas_zoom_percent)
            self.reset_view_btn.clicked.connect(self.reset_canvas_views)
            self.magnifier_btn.toggled.connect(self.on_magnifier_toggled)
            self.mark_combo.currentTextChanged.connect(self.on_active_roi_selection_changed)
            self.layer_combo.currentTextChanged.connect(self.on_active_roi_selection_changed)
            self.roi_index_combo.currentIndexChanged.connect(self.on_roi_index_changed)
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
            for widget in (self.roi_type_combo, self.target_edge_combo, self.diameter_mode_combo, self.search_direction_combo):
                widget.currentIndexChanged.connect(self.on_roi_parameter_edited)
            for widget in (
                self.center_x_spin, self.center_y_spin, self.inner_radius_spin, self.outer_radius_spin,
                self.caliper_count_spin, self.caliper_width_spin, self.inner_ratio_spin, self.roi_angle_spin,
            ):
                widget.valueChanged.connect(self.on_roi_parameter_edited)
            self.three_point_circle_btn.toggled.connect(self.on_three_point_circle_toggled)
            self.upper_canvas.roiChanged.connect(self.set_roi)
            self.upper_canvas.imageDropped.connect(lambda path: self.import_dropped_image(path, "upper"))
            self.lower_canvas.imageDropped.connect(lambda path: self.import_dropped_image(path, "lower"))
            self.lower_canvas.roiChanged.connect(self.set_roi)
            self.upper_canvas.roiEditCommitted.connect(self.commit_canvas_roi_edit)
            self.lower_canvas.roiEditCommitted.connect(self.commit_canvas_roi_edit)
            self.upper_canvas.roiSelected.connect(self.select_roi_from_canvas)
            self.lower_canvas.roiSelected.connect(self.select_roi_from_canvas)
            self.upper_canvas.roiSelectionCleared.connect(self.clear_roi_selection)
            self.lower_canvas.roiSelectionCleared.connect(self.clear_roi_selection)
            self.upper_canvas.roiContextAction.connect(self.handle_roi_context_action)
            self.lower_canvas.roiContextAction.connect(self.handle_roi_context_action)
            self.upper_canvas.interactionMessage.connect(self.show_canvas_interaction_message)
            self.lower_canvas.interactionMessage.connect(self.show_canvas_interaction_message)
            self.upper_canvas.profileMeasured.connect(self.receive_image_profile)
            self.lower_canvas.profileMeasured.connect(self.receive_image_profile)
            self.upper_canvas.geometryClicked.connect(self._on_geometry_canvas_clicked)
            self.lower_canvas.geometryClicked.connect(self._on_geometry_canvas_clicked)
            self.upper_canvas.geometryCommand.connect(self.handle_geometry_command)
            self.lower_canvas.geometryCommand.connect(self.handle_geometry_command)
            # QPushButton.clicked emits a checked boolean. Passing that signal
            # directly used to overwrite show_message=False and silently suppress
            # every error dialog for a normal (unchecked) button click.
            self.analyze_roi_btn.clicked.connect(
                lambda checked=False: self.analyze_roi_regions(show_message=True)
            )
            self.analyze_current_btn.clicked.connect(self.analyze_current_mark)
            self.analyze_all_btn.clicked.connect(self.analyze_all_marks)
            self.export_btn.clicked.connect(self.export_result_file)
            self.save_recipe_btn.clicked.connect(self.save_recipe_file)
            self.load_recipe_btn.clicked.connect(self.show_recipe_quick_menu)
            self.recipe_manage_btn.clicked.connect(self.show_recipe_manager)
            self.batch_import_mark1_upper_btn.clicked.connect(lambda: self.import_batch_images("Mark1", "upper"))
            self.batch_import_mark1_lower_btn.clicked.connect(lambda: self.import_batch_images("Mark1", "lower"))
            self.batch_import_mark2_upper_btn.clicked.connect(lambda: self.import_batch_images("Mark2", "upper"))
            self.batch_import_mark2_lower_btn.clicked.connect(lambda: self.import_batch_images("Mark2", "lower"))
            self.batch_clear_btn.clicked.connect(self.clear_batch_images)
            self.measurement_run_mode_combo.currentIndexChanged.connect(self._refresh_all_widgets)
            self.batch_import_source_combo.currentIndexChanged.connect(
                lambda: self.batch_recursive_check.setEnabled(self._combo_value(self.batch_import_source_combo) == "Folder")
            )
            if hasattr(self, "algorithm_path_button"):
                self.algorithm_path_button.clicked.connect(self.show_algorithm_path_dialog)
            self._install_button_feedback()
