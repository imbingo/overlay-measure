"""Application chrome themes, matching Surface Analyzer's neutral dark palette.

Only widget presentation is themed. Image pixels, overlays, measurement models
and exported reports retain their original colors and values.
"""
from __future__ import annotations

import re
import weakref

from PySide6.QtCore import QEvent, QObject, QSettings, Qt, Signal
from PySide6.QtGui import QActionGroup, QColor, QIcon, QPalette, QPixmap, QPainter, QPen
from PySide6.QtWidgets import QApplication, QMenu, QStyledItemDelegate, QTableWidget, QToolButton, QTreeWidget, QWidget

from .runtime_support import app_data_root

WORKSPACE = "#0e0e0f"
PANEL = "#141415"
CARD = "#19191b"
FIELD = "#111112"
BORDER = "#2c2c2f"
DIVIDER = "#232326"
TEXT = "#ededed"
MUTED = "#9a9a9f"
DISABLED = "#5c5c61"
SELECTED = "#26262a"
HOVER = "#303035"
PASS = "#5cc98f"
FAIL = "#ff6b6b"
WARN = "#e8b45a"
INFO = "#8ab4f8"
LABELS = {"light": "浅色", "dark": "深色", "system": "跟随系统"}
_DARK = {WORKSPACE, PANEL, CARD, FIELD, BORDER, DIVIDER, TEXT, MUTED,
         DISABLED, SELECTED, HOVER, PASS, FAIL, WARN, INFO}
_COLOR = re.compile(r"#[0-9a-fA-F]{6}\b|\b(?:white|black)\b")
_DECL = re.compile(r"([\w-]+)\s*:\s*([^;{}]+)")
_BLOCK = re.compile(r"(?<=\{)[^{}]+(?=\})")


def dark_color(prop: str, source: str) -> str:
    key = source.lower()
    if key in _DARK:
        return source
    color = QColor(key)
    r, g, b, _ = color.getRgb()
    value, sat = (r + g + b) / 3, max(r, g, b) - min(r, g, b)
    if prop in ("color", "selection-color"):
        if sat < 40:
            return TEXT if value < 70 or value > 245 else MUTED
        if r >= g * 0.95 and g > b * 1.45:
            return WARN
        if r > 1.27 * g and r > 1.23 * b:
            return WARN if g > 1.45 * b else FAIL
        if g > 1.19 * r and g > 1.12 * b:
            return PASS
        return INFO if b > r else TEXT
    if prop.startswith("border") or prop == "outline":
        return DIVIDER if value > 220 else BORDER
    if prop.startswith("background") or prop == "alternate-background-color":
        if key in {"#f5f7fa", "#eceff3", "#f2f1f0", "#252930"}:
            return WORKSPACE
        if key in {"#f7f9fb", "#fafafa", "#f4f6f8"}:
            return PANEL
        if prop == "alternate-background-color":
            return CARD
        return CARD if value > 245 else SELECTED
    if prop == "selection-background-color":
        return SELECTED
    if prop == "gridline-color":
        return DIVIDER
    return source


def dark_qss(css: str) -> str:
    def declaration(match):
        prop, rhs = match.groups()
        return f"{prop}: {_COLOR.sub(lambda c: dark_color(prop.lower(), c.group()), rhs)}"
    return (_BLOCK.sub(lambda block: _DECL.sub(declaration, block.group()), css)
            if "{" in css else _DECL.sub(declaration, css))


_OVERRIDES = f"""
QWidget {{ background: {WORKSPACE}; color: {TEXT}; }}
QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
QFrame#titleBar, QFrame#commandBar, QDialog, QMenu {{ background: {PANEL}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{ background: {FIELD}; color: {TEXT}; border-color: {BORDER}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: {DISABLED}; background: {PANEL}; }}
QPushButton:disabled, QToolButton:disabled, QLabel:disabled {{ color: {DISABLED}; }}
QPushButton#primaryButton {{ background: {TEXT}; color: {FIELD}; border-color: {TEXT}; }}
QPushButton#primaryButton:hover {{ background: #ffffff; color: {FIELD}; }}
QPushButton#primaryButton:disabled {{ background: {HOVER}; color: {DISABLED}; border-color: {BORDER}; }}
QComboBox#accessMode, QToolButton#themeButton {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 8px; }}
QToolButton#themeButton:hover {{ background: {HOVER}; }}
QToolButton#themeButton::menu-indicator {{ image: none; }}
QTabBar::tab:selected {{ background: {SELECTED}; color: {TEXT}; }}
QTableView, QTreeView, QListView {{ background: {FIELD}; color: {TEXT}; gridline-color: {DIVIDER}; selection-background-color: {SELECTED}; selection-color: {TEXT}; }}
QHeaderView::section {{ background: {PANEL}; color: {MUTED}; border-color: {BORDER}; }}
QMenu, QComboBox QAbstractItemView, QToolTip {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER}; selection-background-color: {SELECTED}; selection-color: {TEXT}; }}
QMenu::item:selected {{ background: {SELECTED}; }}
QScrollBar {{ background: {PANEL}; }}
QScrollBar::handle {{ background: {BORDER}; }}
QCheckBox::indicator:checked {{ background: {MUTED}; border-color: {MUTED}; }}
QCheckBox::indicator:disabled {{ background: {PANEL}; border-color: {BORDER}; }}
QProgressBar {{ background: {FIELD}; color: {TEXT}; border-color: {BORDER}; }}
QProgressBar::chunk {{ background: {INFO}; }}
"""


class ThemeManager(QObject):
    changed = Signal(str)

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.settings = QSettings(str(app_data_root() / "ui_preferences.ini"), QSettings.IniFormat)
        preference = str(self.settings.value("theme", "light"))
        self.preference = preference if preference in LABELS else "light"
        self.mode = "light"
        self.windows = weakref.WeakSet()
        self._closed_windows = weakref.WeakSet()
        self._palette = QPalette(app.palette())
        self._busy = False
        self._initialized = False
        app.installEventFilter(self)
        app.styleHints().colorSchemeChanged.connect(self._system_changed)

    @staticmethod
    def system_mode():
        return "dark" if QApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark else "light"

    def _system_changed(self, *_):
        if self.preference == "system":
            self.set_preference("system", persist=False)

    def register(self, window):
        was_closed = window in self._closed_windows
        self._closed_windows.discard(window)
        self.windows.add(window)
        if not self._initialized:
            self.set_preference(self.preference, persist=False)
        elif self.mode == "dark" or was_closed:
            for widget in (window, *window.findChildren(QWidget)):
                self._sync(widget)

    def unregister(self, window):
        self.windows.discard(window)
        self._closed_windows.add(window)

    def set_preference(self, preference, *, persist=True):
        self._initialized = True
        self.preference = preference if preference in LABELS else "light"
        self.mode = self.system_mode() if self.preference == "system" else self.preference
        if persist:
            self.settings.setValue("theme", self.preference)
            self.settings.sync()
        palette = QPalette(self._palette)
        if self.mode == "dark":
            for role, color in {"Window": PANEL, "WindowText": TEXT, "Base": FIELD,
                                "AlternateBase": CARD, "Text": TEXT, "Button": CARD,
                                "ButtonText": TEXT, "Highlight": SELECTED,
                                "HighlightedText": TEXT, "ToolTipBase": CARD,
                                "ToolTipText": TEXT, "PlaceholderText": MUTED,
                                "Light": BORDER, "Mid": DIVIDER, "Dark": WORKSPACE}.items():
                palette.setColor(getattr(QPalette.ColorRole, role), QColor(color))
            for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
                palette.setColor(QPalette.Disabled, role, QColor(DISABLED))
        self.app.setPalette(palette)
        for widget in self.app.allWidgets():
            if QWidget.window(widget) in self._closed_windows:
                continue
            self._sync(widget)
            QWidget.update(widget)
        self.changed.emit(self.mode)

    def _sync(self, widget):
        if isinstance(widget, (QTreeWidget, QTableWidget)) and type(widget.itemDelegate()) is QStyledItemDelegate:
            widget.setItemDelegate(ThemeItemDelegate(widget))
        css = widget.styleSheet()
        applied = widget.property("_somaAppliedQss")
        light = widget.property("_somaLightQss") if applied is not None and applied == css else css
        light = light or ""
        if not light and widget not in self.windows:
            return
        target = dark_qss(light) if self.mode == "dark" else light
        if self.mode == "dark" and widget in self.windows:
            target += _OVERRIDES
        if self.mode == "dark" and widget.objectName() == "accessMode":
            target += f"color: {TEXT}; background: {CARD};"
        widget.setProperty("_somaLightQss", light)
        widget.setProperty("_somaAppliedQss", target)
        if css != target:
            self._busy = True
            try:
                widget.setStyleSheet(target)
            finally:
                self._busy = False

    def eventFilter(self, obj, event):
        if isinstance(obj, QWidget) and QWidget.window(obj) in self._closed_windows:
            if event.type() == QEvent.Show and obj in self._closed_windows:
                self.register(obj)
            return False
        if (not self._busy and self.mode == "dark" and isinstance(obj, QWidget)
                and event.type() in (QEvent.Polish, QEvent.StyleChange)):
            self._sync(obj)
        return False


def manager():
    app = QApplication.instance()
    if not hasattr(app, "_soma_theme_manager"):
        app._soma_theme_manager = ThemeManager(app)
    return app._soma_theme_manager


def ui_color(source, prop="color", widget=None):
    if widget is not None and widget.property("_somaLightSnapshot"):
        return source
    app = QApplication.instance()
    theme = getattr(app, "_soma_theme_manager", None)
    return dark_color(prop, source) if theme and theme.mode == "dark" else source


def save_light_snapshot(widget, path):
    previous = widget.property("_somaLightSnapshot")
    widget.setProperty("_somaLightSnapshot", True)
    try:
        return widget.grab().save(str(path))
    finally:
        widget.setProperty("_somaLightSnapshot", previous)
        widget.update()


class ThemeItemDelegate(QStyledItemDelegate):
    """Theme explicit status brushes without changing stored/exported item data."""
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if manager().mode != "dark":
            return
        brush = index.data(Qt.ForegroundRole)
        if brush is not None and brush.style() != Qt.NoBrush:
            option.palette.setColor(QPalette.Text, QColor(dark_color("color", brush.color().name())))
        brush = index.data(Qt.BackgroundRole)
        if brush is not None and brush.style() != Qt.NoBrush:
            r, g, b, _ = brush.color().getRgb()
            if max(r, g, b) - min(r, g, b) < 12:
                background = CARD
            else:
                background = "#18271f" if g > r else "#2a2418" if g > b * 1.1 else "#2a1d1e"
            option.backgroundBrush = QColor(background)


def create_theme_button(parent):
    theme = manager()
    button = QToolButton(parent)
    button.setObjectName("themeButton")
    button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    button.setPopupMode(QToolButton.InstantPopup)
    button.setToolTip("外观：浅色 / 深色 / 跟随系统")
    menu = QMenu(button)
    group = QActionGroup(menu)
    group.setExclusive(True)
    for key, label in LABELS.items():
        action = menu.addAction(label)
        action.setCheckable(True)
        group.addAction(action)
        action.triggered.connect(lambda checked=False, value=key: theme.set_preference(value))
    button.setMenu(menu)
    button_ref = weakref.ref(button)

    def refresh(*_):
        button = button_ref()
        if button is None:
            return
        button.setText(LABELS[theme.preference])
        button.menu().actions()[list(LABELS).index(theme.preference)].setChecked(True)
        pixmap = QPixmap(18, 18)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(TEXT if theme.mode == "dark" else "#68717d"), 1.5))
        if theme.mode == "dark":
            painter.drawArc(3, 2, 12, 12, 70 * 16, 280 * 16)
            painter.drawArc(7, 0, 9, 10, 150 * 16, 150 * 16)
        else:
            painter.drawEllipse(5, 5, 8, 8)
            for x1, y1, x2, y2 in ((9, 1, 9, 3), (9, 15, 9, 17), (1, 9, 3, 9), (15, 9, 17, 9)):
                painter.drawLine(x1, y1, x2, y2)
        painter.end()
        button.setIcon(QIcon(pixmap))
    theme.changed.connect(refresh)
    button.destroyed.connect(lambda *_: theme.changed.disconnect(refresh))
    refresh()
    return button
