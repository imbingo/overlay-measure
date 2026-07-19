# Overlay Measure

对位偏差测量软件，用于识别 Upper / Lower mark 的中心位置，并计算对位偏差 `Dx`、`Dy`、`Dxy` 和 `Rz` 等结果。

## 当前版本

- 当前代码版本：V1.6.1
- 来源文件：`overlay_mark_measure_v1_4_2.zip`
- Windows 启动文件：`start_overlay_measure.bat`
- Python 主入口：`main.py`
- 主界面代码：`overlay_measure/ui_main.py`

V1.6.1 在 V1.6.0 生产权限与追溯基础上增加可配置的本机配方库目录。工程师可在配方管理中迁移并切换目录、仅切换目录或恢复默认目录；迁移会复制配方、SHA256、收藏及最近使用记录并保留原目录。生产模式下目录配置和配方导入均被锁定。

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
- 支持配方 JSON 保存/加载、本机配方库、共享配方目录、收藏和最近使用快捷切换。
- 支持工程模式修改本机配方库位置，并安全迁移配方签章与使用状态。
- 支持生产/工程权限隔离、配方 SHA256 完整性校验和生产测量前置检查。
- 支持通过、超限、无效、异常四态结果，以及输入文件和参数快照追溯。
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

## 生产模式

- 软件启动后默认处于生产模式。
- 初始工程模式密码为 `admin123`；进入工程模式后可在“产品信息”页修改密码，正式部署后应在受控流程中修改并限制知悉范围。
- 生产模式只接受“已验证”且 `.sha256` 签章匹配的配方。
- 运行日志、未完成任务和测量追溯档案默认保存在 `%LOCALAPPDATA%\OverlayMeasure`。

## 历史版本

仓库根目录始终放当前推荐版本。更新前版本已移动到 `legacy/v1.0.5/`，同时 Git 历史也保留了完整提交记录。
