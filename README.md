# Overlay Measure

对位偏差测量软件，用于识别 Upper / Lower mark 的中心位置，并计算对位偏差 `Dx`、`Dy`、`Dxy` 和 `Rz` 等结果。

## 当前版本

- 当前代码版本：V1.5
- 来源文件：`overlay_mark_measure_v1_4_2.zip`
- Windows 启动文件：`start_overlay_measure.bat`
- Python 主入口：`main.py`
- 主界面代码：`overlay_measure/ui_main.py`

V1.5 增加了可测试的测量服务层、黄金样例测试、算法路径显示、自动识别截断提示，并改进非等比例像素下的物理单位换算。

## 运行

```powershell
python -m pip install -r requirements.txt
python .\main.py
```

Windows 下也可以直接双击仓库根目录的 `start_overlay_measure.bat` 启动软件；如果缺少依赖，先执行上面的安装命令。

如需生产环境完全复现当前依赖，可使用：

```powershell
python -m pip install -r requirements.lock.txt
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
- `start_overlay_measure.bat`: Windows 双击启动文件。
- `requirements.txt`: Python 依赖。
- `legacy/v1.0.5/`: 更新前仓库版本归档，用于回看旧版本文件。
- `CHANGELOG.md`: 版本更替记录。

## 数据文件规范

- 公开仓库只保留 `sample_data/` 下的演示图片和示例 recipe。
- 程序运行生成的 Excel 结果、根目录配方 JSON、真实料号 recipe 不再提交到仓库。
- 如需共享真实测量数据，请使用私有仓库、Release 附件或单独的数据存储位置。

## 历史版本

仓库根目录始终放当前推荐版本。更新前版本已移动到 `legacy/v1.0.5/`，同时 Git 历史也保留了完整提交记录。
