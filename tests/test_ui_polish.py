from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QTableWidgetItem, QVBoxLayout, QWidget

from overlay_measure.result_table import ResultTableWidget
from overlay_measure.ui_components import CollapsibleSection, ui_asset_url
from overlay_measure.ui_main import MainWindow


def _app():
    return QApplication.instance() or QApplication([])


def _visible_command_widgets(window):
    layout = window.command_bar.layout()
    widgets = [layout.itemAt(index).widget() for index in range(layout.count())]
    return [widget for widget in widgets if widget is not None and widget.isVisible()]


@pytest.mark.parametrize("name", ["chevron-down", "chevron-up", "chevron-right", "check"])
def test_stylesheet_glyph_assets_are_shipped(name):
    assert Path(ui_asset_url(f"{name}.png")).is_file()
    assert Path(ui_asset_url(f"{name}@2x.png")).is_file()


@pytest.mark.parametrize("mode", ["Production", "Engineering"])
@pytest.mark.parametrize("batch_preview", [False, True])
@pytest.mark.parametrize("width", [1120, 1280, 1366, 1500])
def test_command_bar_controls_never_overlap(mode, batch_preview, width):
    app = _app()
    window = MainWindow()
    window.show()
    window._set_operation_mode(mode, authenticated=True)
    window.batch_preview_nav.setVisible(batch_preview)
    window.resize(width, 720)
    app.processEvents()
    window._update_toolbar_density()
    app.processEvents()

    widgets = _visible_command_widgets(window)
    for left, right in zip(widgets, widgets[1:]):
        assert left.geometry().right() < right.geometry().left()
    assert widgets[-1].geometry().right() <= window.command_bar.width()
    # Every density keeps the image-mode selector and a route to display enhancement.
    assert window.mode_combo.isVisible()
    assert window.display_enhance_check.isVisible() or window.more_actions_btn.isVisible()
    window.close()
    app.processEvents()


def test_command_bar_uses_full_labels_when_there_is_room():
    app = _app()
    window = MainWindow()
    window.show()
    window._set_operation_mode("Engineering", authenticated=True)
    window.resize(1920, 1000)
    app.processEvents()

    assert window.import_upper_btn.text() == "导入上层/单图"
    assert window.analyze_all_btn.text() == "运行测量程序"
    assert window.image_mode_label.isVisible()
    assert window.display_enhance_check.isVisible()
    window.close()
    app.processEvents()


def test_side_tabs_fit_without_scroll_arrows_at_minimum_window_size():
    app = _app()
    window = MainWindow()
    window.resize(1120, 720)
    window.show()
    app.processEvents()

    assert window.side_tabs.tabBar().sizeHint().width() <= window.side_tabs.width()
    window.close()
    app.processEvents()


def test_result_scroll_bar_only_shows_when_columns_overflow():
    app = _app()
    host = QWidget()
    layout = QVBoxLayout(host)
    table = ResultTableWidget()
    layout.addWidget(table)
    layout.addWidget(table.external_horizontal_bar)
    host.resize(900, 300)
    table.setColumnCount(2)
    table.setRowCount(1)
    host.show()
    app.processEvents()
    assert not table.external_horizontal_bar.isVisible()

    table.setColumnCount(12)
    for column in range(12):
        table.setItem(0, column, QTableWidgetItem("content-" * 6))
    table.fit_result_columns()
    app.processEvents()
    assert table.external_horizontal_bar.isVisible()
    assert table.external_horizontal_bar.maximum() > 0
    host.close()
    app.processEvents()


def test_collapsible_section_header_spans_width_and_toggles_body():
    app = _app()
    section = CollapsibleSection("环形 ROI 参数", True)
    section.add_widget(QWidget())
    section.resize(400, 200)
    section.show()
    app.processEvents()

    assert section.toggle_btn.width() == section.width()
    assert not section.toggle_btn.icon().isNull()
    assert section.body.isVisible()
    section.toggle_btn.click()
    app.processEvents()
    assert not section.body.isVisible()
    section.close()
    app.processEvents()
