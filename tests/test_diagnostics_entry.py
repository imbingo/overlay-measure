import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from overlay_measure.ui_main import MainWindow


@pytest.mark.parametrize('size',[(1120,720),(1366,768),(1500,920)])
def test_engineering_diagnostics_entry_is_visible_and_opens_dialog(size):
    app=QApplication.instance() or QApplication([])
    window=MainWindow()
    try:
        window._set_operation_mode('Engineering',authenticated=True)
        window.resize(*size);window.show();app.processEvents()
        button=window.more_actions_btn
        assert window.command_bar.layout().indexOf(button)>=0
        assert button.isVisible()
        assert window.command_bar.rect().contains(button.geometry())
        assert window.soma_diagnostics_action.isEnabled()
        menu=window.more_actions_menu
        menu.popup(button.mapToGlobal(QPoint(0,button.height())))
        app.processEvents()
        QTest.mouseClick(menu,Qt.LeftButton,pos=menu.actionGeometry(window.soma_diagnostics_action).center())
        app.processEvents()
        assert window._soma_diagnostics_dialog.isVisible()
        assert window._soma_diagnostics_dialog.session.a is None
        window._soma_diagnostics_dialog.close()
        window._set_calculation_running(True)
        assert not window.soma_diagnostics_action.isEnabled()
        window._set_calculation_running(False)
        window._set_operation_mode('Production')
        assert not button.isVisible()
        assert not window.soma_diagnostics_action.isEnabled()
    finally:
        window.close();app.processEvents()
