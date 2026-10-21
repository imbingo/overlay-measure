"""Real main-page detection -> one-click companion view, synthetic QA data."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QFont
from overlay_measure.models import Roi, MarkRecipe, DetectionParams
from overlay_measure.image_loader import load_image
from overlay_measure.measurement_service import detect_manual_roi
from overlay_measure.ui_main import MainWindow
from overlay_measure.diagnostics.measurement_bridge import export_measurement


def main():
    out = ROOT / "artifacts/diagnostics-v260"
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    for font in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc"]:
        QFontDatabase.addApplicationFont(font)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = MainWindow()
    window._set_operation_mode("Engineering", authenticated=True)
    window._set_mode_ui("Dual Image")
    window.params = DetectionParams(use_ransac=False, profile_step_px=0.25)
    mark = MarkRecipe("Mark1")
    window.marks["Mark1"] = mark
    y, x = np.mgrid[:512, :600]
    for layer, center in [("upper", (300, 256)), ("lower", (309, 248))]:
        xx = x - center[0]
        yy = y - center[1]
        theta = np.arctan2(yy, xx)
        radius = np.hypot(xx, yy)
        edge = (
            150
            if layer == "upper"
            else 150
            + 1.2 * np.cos(2 * theta)
            + 1.8 * np.exp(-(((theta - np.deg2rad(135)) / 0.25) ** 2))
        )
        gray = 1000 + 48000 / (1 + np.exp(np.clip(-(radius - edge) / 1.1, -60, 60)))
        if layer == "lower":
            gray += 6000 / (1 + np.exp(np.clip(-(radius - edge - 5) / 0.6, -60, 60)))
        path = out / ("DEMO_" + layer + ".tif")
        _, encoded = cv2.imencode(".tif", gray.astype(np.uint16))
        encoded.tofile(str(path))
        image = load_image(str(path))
        window._set_image_for_layer("Mark1", layer, image, "single")
        roi = Roi(
            center[0] - 175,
            center[1] - 175,
            350,
            350,
            "Caliper Circle",
            0.7,
            "Strongest Edge",
            caliper_count=96,
            caliper_width_px=5,
        )
        entry = mark.add_roi(layer, roi, roi_id=layer + "-1")
        window.roi_detections["Mark1"][entry.roi_id] = detect_manual_roi(
            "Mark1", layer, image, roi, window.params, window.config
        )
    mark.reference_contour_id = "upper-1"
    mark.target_contour_id = "lower-1"
    window._rebuild_legacy_manual_detections("Mark1")
    window._refresh_all_widgets()
    window.show()
    app.processEvents()
    window.start_soma_diagnostics()
    dialog = window._soma_diagnostics_dialog
    dialog.align_button.click()
    app.processEvents()
    assert dialog.session.result is not None
    dialog.angle = 135
    for mode in range(3):
        dialog.set_mode(mode)
        if mode == 0:
            dialog.zoom("适应窗口")
        else:
            dialog.zoom("局部放大")
        dialog.source_label.setText(
            "DEMO / 模拟数据　｜　主页面已识别的上层 / 下层圆孔（不重新检测）"
        )
        app.processEvents()
        dialog.grab().save(str(out / f"0{mode+1}-simple-ui.png"))
    export_measurement(
        dialog.snapshot,
        dialog.upper_choice.currentData(),
        dialog.lower_choice.currentData(),
        out / ("export-" + datetime.now().strftime("%H%M%S_%f")),
    )
    print("PASS main-page snapshots / one click / 3 rendered views / export:", out)
    print(dialog.session.result.registered.metrics)
    dialog.close()
    window.close()
    app.processEvents()


if __name__ == "__main__":
    main()
