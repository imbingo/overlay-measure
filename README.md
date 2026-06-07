# Overlay Measure

对位偏差测量软件，用于识别 Upper / Lower mark 的中心位置，并计算对位偏差 `Dx`、`Dy`、`Dxy` 和 `Rz` 等结果。

## 当前版本

- 当前代码版本：V1.4.2
- 来源文件：`overlay_mark_measure_v1_4_2.zip`
- 主入口：`main.py`
- 主界面代码：`overlay_measure/ui_main.py`

压缩包原始 README 中包含 V1.4.2.2 的修复说明；应用窗口和当前源码版本标签为 V1.4.2。本次仓库更新按压缩包文件名和应用版本统一记录为 V1.4.2。

## 运行

```powershell
python -m pip install -r requirements.txt
python .\main.py
```

## 主要功能

- 支持 Single Image / Dual Image 测量模式。
- 支持手动 ROI 与自动识别测量。
- 支持圆、椭圆、矩形、环形 ROI 等 mark 拟合流程。
- 支持批量测量与重复性分析。
- 支持配方 JSON 保存/加载。
- 支持 Excel 结果导出。
- 支持样例图片和样例 recipe 快速验证流程。

## 目录说明

- `overlay_measure/`: 主程序模块。
- `sample_data/`: 示例图片和示例 recipe。
- `main.py`: GUI 启动入口。
- `requirements.txt`: Python 依赖。
- `legacy/v1.0.5/`: 更新前仓库版本归档，用于回看旧版本文件。
- `CHANGELOG.md`: 版本更替记录。

## 历史版本

仓库根目录始终放当前推荐版本。更新前版本已移动到 `legacy/v1.0.5/`，同时 Git 历史也保留了完整提交记录。
