"""Render real Diagnostics UI through actual loading/registration/worker analysis.

All input images generated here are explicitly synthetic QA fixtures.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QFont
from overlay_measure.diagnostics.ui_diagnostics import DiagnosticsDialog
from overlay_measure.models import DetectionParams, MeasurementConfig, Roi


def main():
    out = ROOT / "artifacts/diagnostics-v250"
    out.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:256, :288]
    for name, dx, dy in [("A", 0, 0), ("B", 5, -3)]:
        xx = x - 140 - dx
        yy = y - 128 - dy
        angle = np.arctan2(yy, xx)
        radius = np.hypot(xx, yy)
        edge = (
            70
            if name == "A"
            else 70
            + 0.65 * np.cos(2 * angle)
            + 1.3 * np.exp(-(((angle - np.deg2rad(135)) / 0.3) ** 2))
        )
        data = 1000 + 46000 / (1 + np.exp(np.clip(-(radius - edge) / 1.1, -60, 60)))
        if name == "B":
            # An extra weaker transition tests the actual candidate-peak diagnostics.
            data += 7000 / (1 + np.exp(np.clip(-(radius - edge - 6) / 0.6, -60, 60)))
        data = np.clip(data, 0, 65535).astype(np.uint16)
        for px, py in [(20, 20), (260, 20), (20, 230)]:
            cv2.rectangle(
                data, (px + dx - 3, py + dy - 3), (px + dx + 3, py + dy + 3), 6000, -1
            )
        _, encoded = cv2.imencode(".tif", data)
        encoded.tofile(str(out / f"DEMO_{name}.tif"))
    app = QApplication.instance() or QApplication([])
    for path in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc"]:
        if Path(path).exists():
            QFontDatabase.addApplicationFont(path)
    app.setFont(QFont("Microsoft YaHei", 9))
    dialog = DiagnosticsDialog(DetectionParams(), MeasurementConfig())
    dialog.setWindowTitle(dialog.windowTitle() + " / DEMO 模拟数据")
    dialog.load_path("A", str(out / "DEMO_A.tif"))
    dialog.load_path("B", str(out / "DEMO_B.tif"))
    dialog.count.setValue(96)
    dialog.editors["width"].setValue(5)
    dialog.session.roi = Roi(
        52,
        40,
        176,
        176,
        "Caliper Circle",
        0.65,
        "Strongest Edge",
        caliper_count=96,
        caliper_width_px=5,
    )
    dialog.points_a = [(20, 20), (260, 20), (20, 230)]
    dialog.points_b = [(25, 17), (265, 17), (25, 227)]
    dialog.register()
    dialog.show()
    app.processEvents()
    dialog.start_analysis()
    deadline = time.monotonic() + 90
    while dialog.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    if dialog.busy:
        raise RuntimeError("worker timed out")
    if dialog.session.result is None:
        raise RuntimeError(dialog.status.text())
    dialog.angle = 135
    dialog.refresh_selection()
    for mode in range(3):
        dialog.set_mode(mode)
        if mode == 1:
            dialog.zoom("局部放大")
        if mode == 2:
            dialog.overlay.reset_view()
            dialog.zoom("局部放大")
        dialog.status.setText(
            "DEMO / 模拟数据 · 实际导入 TIFF → 独立点刚性配准 → 共用卡尺检测 → 真实曲线与导出流程"
        )
        app.processEvents()
        dialog.grab().save(str(out / f"0{mode+1}-actual-ui.png"))
    from overlay_measure.diagnostics.export import export_session

    export_session(dialog.session, out / "export", excel=True)
    export_session(dialog.session, out / "export", excel=False)
    print("PASS real worker/UI pipeline; outputs:", out)
    print(dialog.session.result.registered.metrics)
    dialog.close()
    app.processEvents()


if __name__ == "__main__":
    main()
