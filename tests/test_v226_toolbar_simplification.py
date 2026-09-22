from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from overlay_measure.ui_main import MainWindow


def _app():
    return QApplication.instance() or QApplication([])


def test_v224_toolbar_layout_is_preserved_in_production_mode():
    app = _app()
    window = MainWindow()
    window.show()
    app.processEvents()

    assert window.operation_mode == "Production"
    assert window.load_recipe_btn.isVisible()
    assert window.analyze_all_btn.isVisible()
    assert window.export_btn.isVisible()
    assert window.recent_images_btn.isVisible()
    assert not hasattr(window, "recipe_manage_btn")
    assert not hasattr(window, "save_recipe_btn")
    assert window.image_mode_label.isVisible()
    assert window.mode_combo.isVisible()
    assert window.display_enhance_check.isVisible()
    assert window.analyze_roi_btn.isVisible()
    assert window.import_upper_btn.isVisible()
    assert window.import_lower_btn.isVisible()
    assert window.reset_measurement_btn.isVisible()
    assert not window.import_images_btn.isVisible()
    assert not window.more_actions_btn.isVisible()
    assert window.cancel_progress_btn.isHidden()
    window.close()
    app.processEvents()


def test_engineering_mode_restores_configuration_and_roi_actions():
    app = _app()
    window = MainWindow()
    window.show()
    window._set_operation_mode("Engineering", authenticated=True)
    window.side_tabs.setCurrentIndex(2)
    app.processEvents()

    assert window.recent_images_btn.isVisible()
    assert not hasattr(window, "recipe_manage_action")
    assert not hasattr(window, "save_recipe_action")
    assert window.image_mode_label.isVisible()
    assert window.mode_combo.isVisible()
    assert window.display_enhance_check.isVisible()
    assert window.analyze_roi_btn.isVisible()

    window.mode_combo.setCurrentText("单图模式")
    app.processEvents()
    assert not window.import_lower_action.isEnabled()
    window.mode_combo.setCurrentText("双图模式")
    app.processEvents()
    assert window.import_lower_action.isEnabled()

    window.display_enhance_action.setChecked(True)
    assert window.display_enhance_check.isChecked()
    window._set_calculation_running(True)
    app.processEvents()
    assert window.cancel_progress_btn.isVisible()
    assert (window.cancel_progress_btn.width(), window.cancel_progress_btn.height()) == (76, 20)
    window._set_calculation_running(False)
    assert window.cancel_progress_btn.isHidden()
    window.close()
    app.processEvents()
