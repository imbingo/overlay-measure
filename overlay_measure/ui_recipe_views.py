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
from .recipe_manager import load_recipe, save_recipe
from .recipe_library import RecipeLibrary, RecipeLibraryEntry
from .recipe_integrity import seal_recipe, verify_recipe
from .result_exporter import build_detection_rows, export_results
from .rz_calculator import build_summary_rows
from .runtime_support import RecoveryStore, build_runtime_logger


class RecipeQuickMenu(QMenu):
    recipeSelected = Signal(str)
    importRequested = Signal()
    managerRequested = Signal()
    openLibraryRequested = Signal()
    saveRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("recipeQuickMenu")
        self.setMinimumWidth(540)
        self._entries: list[RecipeLibraryEntry] = []

        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(12, 12, 12, 10)
        layout.setSpacing(9)

        heading = QHBoxLayout()
        title = QLabel("快速切换配方")
        title.setObjectName("recipeMenuTitle")
        hint = QLabel("双击即可加载")
        hint.setObjectName("recipeMenuHint")
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(hint)
        layout.addLayout(heading)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索配方名称、物料编码或版本")
        self.search_edit.setClearButtonEnabled(True)
        layout.addWidget(self.search_edit)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["配方名称", "物料编码", "版本", "状态", "来源"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setMinimumHeight(285)
        self.tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for column, width in ((1, 100), (2, 64), (3, 76), (4, 56)):
            self.tree.header().setSectionResizeMode(column, QHeaderView.Fixed)
            self.tree.setColumnWidth(column, width)
        layout.addWidget(self.tree)

        action_row = QHBoxLayout()
        self.import_btn = QPushButton("从文件导入…")
        self.manager_btn = QPushButton("配方管理…")
        self.save_btn = QPushButton("保存当前配方")
        self.open_library_btn = QPushButton("打开配方库")
        action_row.addWidget(self.import_btn)
        action_row.addWidget(self.manager_btn)
        action_row.addWidget(self.save_btn)
        action_row.addStretch(1)
        action_row.addWidget(self.open_library_btn)
        layout.addLayout(action_row)

        action = QWidgetAction(self)
        action.setDefaultWidget(host)
        self.addAction(action)
        self.search_edit.textChanged.connect(self._populate)
        self.tree.itemDoubleClicked.connect(self._activate_item)
        self.tree.itemActivated.connect(self._activate_item)
        self.import_btn.clicked.connect(self._request_import)
        self.manager_btn.clicked.connect(self._request_manager)
        self.open_library_btn.clicked.connect(self._request_open_library)
        self.save_btn.clicked.connect(self._request_save)

    def set_entries(self, entries: list[RecipeLibraryEntry]) -> None:
        self._entries = list(entries)
        self.search_edit.clear()
        self._populate()

    def set_engineering_access(self, enabled: bool) -> None:
        self.import_btn.setEnabled(enabled)
        self.import_btn.setToolTip("" if enabled else "生产模式不能导入或发布配方")
        self.save_btn.setEnabled(enabled)
        self.save_btn.setToolTip("" if enabled else "生产模式不能修改或保存配方")

    @staticmethod
    def _matches(entry: RecipeLibraryEntry, query: str) -> bool:
        haystack = " ".join((entry.name, entry.material_code, entry.version, entry.status, entry.source)).lower()
        return query in haystack

    @staticmethod
    def _status_text(status: str) -> str:
        mapping = {
            "validated": "已验证",
            "approved": "已批准",
            "released": "已发布",
            "draft": "草稿",
            "archived": "已归档",
            "obsolete": "已停用",
        }
        return mapping.get(status.strip().lower(), status or "未验证")

    def _add_entry(self, entry: RecipeLibraryEntry, prefix: str = "") -> None:
        label = f"{prefix}{entry.name}"
        status_text = self._status_text(entry.status)
        item = QTreeWidgetItem([label, entry.material_code, entry.version, status_text, entry.source])
        item.setData(0, Qt.UserRole, str(entry.path))
        item.setToolTip(0, str(entry.path))
        if entry.favorite:
            item.setForeground(0, QColor("#B77900"))
        if "验证" in status_text or "批准" in status_text or "发布" in status_text:
            item.setForeground(3, QColor("#248A3D"))
        self.tree.addTopLevelItem(item)

    def _add_section(self, title: str) -> None:
        item = QTreeWidgetItem([title, "", "", "", ""])
        item.setFlags(Qt.ItemIsEnabled)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        item.setForeground(0, QColor("#68717D"))
        item.setBackground(0, QColor("#F4F6F8"))
        self.tree.addTopLevelItem(item)
        self.tree.setFirstColumnSpanned(self.tree.indexOfTopLevelItem(item), self.tree.rootIndex(), True)

    def _populate(self) -> None:
        query = self.search_edit.text().strip().lower()
        self.tree.clear()
        if query:
            for entry in self._entries:
                if self._matches(entry, query):
                    self._add_entry(entry, "★ " if entry.favorite else "")
        else:
            favorites = [entry for entry in self._entries if entry.favorite]
            recent = sorted(
                (entry for entry in self._entries if entry.last_used and not entry.favorite),
                key=lambda entry: entry.last_used,
                reverse=True,
            )[:8]
            remaining = [entry for entry in self._entries if entry not in favorites and entry not in recent]
            if favorites:
                self._add_section("已固定")
                for entry in favorites[:8]:
                    self._add_entry(entry, "★ ")
            if recent:
                self._add_section("最近使用")
                for entry in recent:
                    self._add_entry(entry)
            if remaining or not self._entries:
                self._add_section("全部配方")
                for entry in remaining:
                    self._add_entry(entry)
        if not self._entries:
            empty = QTreeWidgetItem(["配方库暂无配方，可从文件导入", "", "", "", ""])
            empty.setFlags(Qt.ItemIsEnabled)
            empty.setForeground(0, QColor("#8A939F"))
            self.tree.addTopLevelItem(empty)

    def _activate_item(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        path = item.data(0, Qt.UserRole)
        if path:
            self.hide()
            self.recipeSelected.emit(str(path))

    def _request_import(self) -> None:
        self.hide()
        self.importRequested.emit()

    def _request_manager(self) -> None:
        self.hide()
        self.managerRequested.emit()

    def _request_open_library(self) -> None:
        self.hide()
        self.openLibraryRequested.emit()

    def _request_save(self) -> None:
        self.hide()
        self.saveRequested.emit()


class RecipeLibraryDialog(QDialog):
    def __init__(self, library: RecipeLibrary, parent=None, engineering: bool = True):
        super().__init__(parent)
        self.library = library
        self.engineering = bool(engineering)
        self.selected_recipe_path = ""
        self.setWindowTitle("配方管理")
        self.resize(920, 560)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 14)
        root.setSpacing(10)

        top = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索配方名称、物料编码、版本或状态")
        self.search_edit.setClearButtonEnabled(True)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["全部来源", "本机", "共享"])
        top.addWidget(self.search_edit, 1)
        top.addWidget(self.source_combo)
        root.addLayout(top)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["收藏", "配方名称", "物料编码", "版本", "状态", "来源", "文件路径"])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(6, QHeaderView.Stretch)
        for column, width in ((0, 52), (2, 120), (3, 70), (4, 90), (5, 62)):
            self.tree.header().setSectionResizeMode(column, QHeaderView.Fixed)
            self.tree.setColumnWidth(column, width)
        root.addWidget(self.tree, 1)

        self.local_label = QLabel()
        self.local_label.setObjectName("recipeMenuHint")
        self.local_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.local_label.setMinimumWidth(0)
        local_row = QHBoxLayout()
        local_row.addWidget(self.local_label, 1)
        self.change_local_btn = QPushButton("修改本机目录…")
        self.restore_local_btn = QPushButton("恢复默认目录")
        self.open_btn = QPushButton("打开本机目录")
        local_row.addWidget(self.change_local_btn)
        local_row.addWidget(self.restore_local_btn)
        local_row.addWidget(self.open_btn)
        root.addLayout(local_row)

        self.shared_label = QLabel()
        self.shared_label.setObjectName("recipeMenuHint")
        self.shared_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.shared_label.setMinimumWidth(0)
        shared_row = QHBoxLayout()
        shared_row.addWidget(self.shared_label, 1)
        self.shared_btn = QPushButton("设置共享目录…")
        self.clear_shared_btn = QPushButton("清除共享目录")
        shared_row.addWidget(self.shared_btn)
        shared_row.addWidget(self.clear_shared_btn)
        root.addLayout(shared_row)

        actions = QHBoxLayout()
        self.import_btn = QPushButton("从文件导入…")
        self.favorite_btn = QPushButton("切换收藏")
        self.load_btn = QPushButton("加载所选配方")
        self.load_btn.setObjectName("primaryButton")
        self.close_btn = QPushButton("关闭")
        for button in (self.import_btn, self.favorite_btn):
            actions.addWidget(button)
        actions.addStretch(1)
        actions.addWidget(self.load_btn)
        actions.addWidget(self.close_btn)
        root.addLayout(actions)

        self.search_edit.textChanged.connect(self.refresh)
        self.source_combo.currentTextChanged.connect(self.refresh)
        self.favorite_btn.clicked.connect(self._toggle_favorite)
        self.change_local_btn.clicked.connect(self._choose_local_directory)
        self.restore_local_btn.clicked.connect(self._restore_default_directory)
        self.shared_btn.clicked.connect(self._choose_shared_directory)
        self.clear_shared_btn.clicked.connect(self._clear_shared_directory)
        self.open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.library.root))))
        self.load_btn.clicked.connect(self._accept_selected)
        self.tree.itemDoubleClicked.connect(lambda *_: self._accept_selected())
        self.close_btn.clicked.connect(self.reject)
        for button in (
            self.import_btn,
            self.change_local_btn,
            self.restore_local_btn,
            self.shared_btn,
            self.clear_shared_btn,
        ):
            button.setEnabled(self.engineering)
            if not self.engineering:
                button.setToolTip("生产模式下目录配置和配方导入已锁定")
        self.refresh()

    def refresh(self) -> None:
        query = self.search_edit.text().strip().lower()
        source = self.source_combo.currentText()
        self.tree.clear()
        for entry in self.library.scan():
            if source != "全部来源" and entry.source != source:
                continue
            if query and not RecipeQuickMenu._matches(entry, query):
                continue
            item = QTreeWidgetItem([
                "★" if entry.favorite else "",
                entry.name,
                entry.material_code,
                entry.version,
                RecipeQuickMenu._status_text(entry.status),
                entry.source,
                str(entry.path),
            ])
            item.setData(0, Qt.UserRole, str(entry.path))
            self.tree.addTopLevelItem(item)
        environment_note = "（环境变量覆盖）" if self.library.environment_override else ""
        self.local_label.setText(f"本机配方库：{self.library.root}{environment_note}")
        self.shared_label.setText(f"共享配方库：{self.library.shared_library or '未配置'}")
        self.local_label.setToolTip(str(self.library.root))
        self.shared_label.setToolTip(str(self.library.shared_library or "未配置"))

    def _change_local_directory(self, path: str) -> None:
        target = Path(path).expanduser().resolve()
        if target == self.library.root:
            QMessageBox.information(self, "目录未变化", "所选目录已经是当前本机配方库。")
            return
        if self.library.environment_override:
            QMessageBox.warning(
                self,
                "环境变量覆盖",
                "当前设置了 OVERLAY_MEASURE_RECIPE_LIBRARY。界面修改本次运行会生效，"
                "但下次启动仍可能被该环境变量覆盖。",
            )
        choice = QMessageBox(self)
        choice.setWindowTitle("切换本机配方库")
        choice.setText(f"新目录：\n{target}")
        choice.setInformativeText(
            "“复制迁移并切换”会复制配方、SHA256、收藏和最近使用记录，且保留原目录；"
            "“仅切换”不会复制原目录中的配方。"
        )
        migrate_btn = choice.addButton("复制迁移并切换", QMessageBox.AcceptRole)
        switch_btn = choice.addButton("仅切换目录", QMessageBox.ActionRole)
        choice.addButton(QMessageBox.Cancel)
        choice.exec()
        clicked = choice.clickedButton()
        if clicked != migrate_btn and clicked != switch_btn:
            return
        try:
            report = self.library.change_local_library(target, migrate=clicked == migrate_btn)
        except Exception as exc:
            QMessageBox.critical(self, "切换失败", f"本机配方库没有切换：\n{exc}")
            return
        self.refresh()
        if report.migrated:
            detail = (
                f"已复制 {report.copied} 个配方，复用 {report.reused} 个相同配方，"
                f"重命名 {report.renamed} 个冲突配方。"
            )
        else:
            detail = "未复制原配方。"
        QMessageBox.information(
            self,
            "本机配方库已切换",
            f"当前目录：\n{report.new_root}\n\n{detail}\n原目录仍保留：\n{report.old_root}",
        )

    def _choose_local_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择新的本机配方库目录", str(self.library.root))
        if path:
            self._change_local_directory(path)

    def _restore_default_directory(self) -> None:
        self._change_local_directory(str(self.library.default_root()))

    def _selected_path(self) -> str:
        item = self.tree.currentItem()
        return str(item.data(0, Qt.UserRole)) if item and item.data(0, Qt.UserRole) else ""

    def _toggle_favorite(self) -> None:
        path = self._selected_path()
        if path:
            self.library.toggle_favorite(path)
            self.refresh()

    def _choose_shared_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择共享配方库目录", str(self.library.shared_library or ""))
        if path:
            self.library.set_shared_library(path)
            self.refresh()

    def _clear_shared_directory(self) -> None:
        self.library.set_shared_library(None)
        self.refresh()

    def _accept_selected(self) -> None:
        path = self._selected_path()
        if not path:
            QMessageBox.information(self, "未选择配方", "请先选择一个配方。")
            return
        self.selected_recipe_path = path
        self.accept()
