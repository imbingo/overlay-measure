from __future__ import annotations

from PySide6.QtWidgets import QApplication

from overlay_measure.models import Roi
from overlay_measure.ui_main import MainWindow


def _app():
    return QApplication.instance() or QApplication([])


def _three_point_roi(cx: float, cy: float) -> Roi:
    return Roi(cx - 20.0, cy - 20.0, 40.0, 40.0, "Caliper Circle", 0.6)


def test_three_point_circle_adds_new_roi_without_touching_selected_one():
    app = _app()
    window = MainWindow()
    window.operation_mode = "Engineering"
    first = window.marks["Mark1"].add_roi("upper", Roi(10, 10, 30, 30, "Rectangle"), "manual")
    window._refresh_roi_index_combo(first.roi_id)
    window.on_active_roi_selection_changed()
    app.processEvents()

    for cx in (80.0, 140.0):
        window.three_point_circle_btn.setChecked(True)
        app.processEvents()
        assert window.upper_canvas.circle_pick_mode
        window.upper_canvas.roiChanged.emit("Mark1", "upper", _three_point_roi(cx, 60.0))
        app.processEvents()

    entries = window.marks["Mark1"].roi_entries("upper")
    assert len(entries) == 3
    assert entries[0].roi_id == first.roi_id
    assert entries[0].roi.roi_type == "Rectangle"
    assert (entries[0].roi.x, entries[0].roi.w) == (10, 30)
    assert [entry.roi.roi_type for entry in entries[1:]] == ["Caliper Circle", "Caliper Circle"]
    assert window.roi_index_combo.currentData() == entries[2].roi_id
    assert not window.three_point_circle_btn.isChecked()
    window.close()
