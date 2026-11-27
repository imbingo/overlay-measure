from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QStyleOptionViewItem, QTableWidgetItem

from overlay_measure.models import ImageData
from overlay_measure.theme import CARD, FIELD, PASS, TEXT, ThemeManager, manager, save_light_snapshot
from overlay_measure.ui_main import MainWindow, RepeatabilityPlot


@pytest.fixture
def theme(tmp_path):
    app = QApplication.instance() or QApplication([])
    controller = manager()
    previous_settings, previous_preference = controller.settings, controller.preference
    controller.settings = QSettings(str(tmp_path / "theme.ini"), QSettings.IniFormat)
    controller.set_preference("light", persist=False)
    yield app, controller
    controller.set_preference(previous_preference, persist=False)
    controller.settings = previous_settings


def test_theme_control_next_to_operation_mode_and_menu_switches(theme):
    app, controller = theme
    window = MainWindow()
    try:
        window.show()
        for preference in ("light", "dark", "system"):
            controller.set_preference(preference)
            for width, height in ((1120, 720), (1500, 920)):
                window.resize(width, height)
                app.processEvents()
                assert window.width() == width
                assert window.theme_button.x() > window.version_label.geometry().right()
                assert window.theme_button.geometry().right() < window.recent_images_btn.x()
        actions = window.theme_button.menu().actions()
        assert [action.text() for action in actions] == ["浅色", "深色", "跟随系统"]
        actions[1].trigger()
        assert controller.mode == "dark"
        assert window.theme_button.text() == "深色"
        assert actions[1].isChecked()
        assert window.analyze_all_btn.palette().color(QPalette.ButtonText).isValid()
    finally:
        window.close()


def test_theme_preference_is_persisted_and_reloaded(theme):
    app, controller = theme
    controller.set_preference("dark")
    settings = QSettings(controller.settings.fileName(), QSettings.IniFormat)
    assert settings.value("theme") == "dark"
    # Use an isolated manager to exercise startup preference loading.
    from unittest.mock import patch
    with patch("overlay_measure.theme.app_data_root", return_value=Path(settings.fileName()).parent):
        settings.setValue("unused", True)
        target = Path(settings.fileName()).parent / "ui_preferences.ini"
        target.write_bytes(Path(settings.fileName()).read_bytes())
        restored = ThemeManager(app)
        assert restored.preference == "dark"
        app.removeEventFilter(restored)
        restored.deleteLater()


def test_styles_roundtrip_and_late_dialog_updates(theme):
    app, controller = theme
    window = MainWindow()
    original = window.styleSheet()
    try:
        controller.set_preference("dark")
        window.show()
        dialog = QDialog(window)
        button = QPushButton("Late dialog", dialog)
        button.setStyleSheet("background: #FFFFFF; color: #20242B;")
        dialog.show()
        app.processEvents()
        assert "#FFFFFF" not in button.styleSheet()
        # Runtime status updates must also be converted, not just initial CSS.
        button.setStyleSheet("color: #248A3D; background: #FFFFFF;")
        app.processEvents()
        assert PASS in button.styleSheet()
        controller.set_preference("light")
        assert window.styleSheet() == original
        assert button.styleSheet() == "color: #248A3D; background: #FFFFFF;"
        dialog.close()
    finally:
        window.close()


def test_follow_system_reacts_to_scheme_change(theme, monkeypatch):
    _, controller = theme
    monkeypatch.setattr(controller, "system_mode", lambda: "dark")
    controller.set_preference("system")
    assert controller.mode == "dark"
    monkeypatch.setattr(controller, "system_mode", lambda: "light")
    controller._system_changed()
    assert controller.preference == "system" and controller.mode == "light"


def test_reopening_closed_window_uses_current_theme(theme):
    app, controller = theme
    window = MainWindow()
    original = window.styleSheet()
    try:
        window.show()
        controller.set_preference("dark")
        window.close()
        controller.set_preference("light")
        window.show()
        app.processEvents()
        assert window.styleSheet() == original
        assert window.theme_button.text() == "浅色"
    finally:
        window.close()


def test_theme_preserves_pixels_zoom_results_and_item_status_roles(theme):
    app, controller = theme
    window = MainWindow()
    try:
        image = ImageData("synthetic", np.arange(256, dtype=np.float32).reshape(16, 16), "synthetic")
        window.upper_canvas.set_image(image)
        window.upper_canvas.user_zoom = 4.0
        window.upper_canvas.pan_x = 23.0
        before = window.upper_canvas.pixmap_cache.toImage()
        table = window.det_table
        table.setRowCount(1)
        item = QTableWidgetItem("优秀（满足超精确）")
        item.setBackground(QColor(226, 244, 232))
        item.setForeground(QColor(25, 107, 58))
        table.setItem(0, 0, item)
        original_background = item.background()
        original_foreground = item.foreground()
        controller.set_preference("dark")
        app.processEvents()
        option = QStyleOptionViewItem()
        table.itemDelegate().initStyleOption(option, table.model().index(0, 0))
        assert option.palette.color(QPalette.Text).name() == PASS
        assert option.backgroundBrush.color().name() == "#18271f"
        assert item.background() == original_background
        assert item.foreground() == original_foreground
        assert window.upper_canvas.pixmap_cache.toImage() == before
        assert window.upper_canvas.user_zoom == 4.0
        assert window.upper_canvas.pan_x == 23.0
        assert window.upper_canvas.image is image
    finally:
        window.close()


def test_repeatability_and_diagnostic_chart_dark_but_snapshot_white(theme, tmp_path):
    app, controller = theme
    from types import SimpleNamespace
    from overlay_measure.diagnostics.ui_diagnostics import ComparisonChart
    owner = SimpleNamespace(session=SimpleNamespace(result=None))
    chart = ComparisonChart(owner)
    plot = RepeatabilityPlot()
    try:
        controller.set_preference("dark")
        for widget in (chart, plot):
            widget.resize(600, 240)
            widget.show()
        app.processEvents()
        assert chart.grab().toImage().pixelColor(0, 0).name() == CARD
        assert plot.grab().toImage().pixelColor(0, 0).name() == CARD
        target = tmp_path / "curve.png"
        assert save_light_snapshot(chart, target)
        from PySide6.QtGui import QImage
        assert QImage(str(target)).pixelColor(0, 0).name() == "#ffffff"
        assert chart.grab().toImage().pixelColor(0, 0).name() == CARD
    finally:
        chart.close()
        plot.close()
