# V2.5.0 Diagnostics 验证记录

验证日期：2026-10-21（Asia/Hong_Kong）。代码起点：v2.4.8，`d4f7a3e70999f0f99a6207a40f8d047bf7b8eb9e`。

## 原始基线

- 修改正式代码前执行全量 pytest：**180 passed in 47.39s**。
- 同期生成 `tests/data/diagnostics_v248_baseline.json`：记录输入 hash、ROI/参数/配置、全部卡尺有效/剔除点及梯度、圆参数、正式手动与生产细化结果、圆度、DX/DY 和导出行。
- `tools/capture_diagnostics_baseline.py` 在文件存在时拒绝覆盖。回归比较仅排除导出行的墙钟 timestamp，不舍入或修改任何数值。

## 完成后的检查

- 全量 `python -m pytest -q`：**209 passed in 57.23s**（原有 180 项 + 新增 29 项用例）。
- `python main.py --smoke-test`：退出码 0。
- `python scripts/capture_diagnostics_qa.py`：实际 TIFF 导入、独立点刚性配准、QThread 检测、三视图渲染、完整 Excel/CSV/JSON/PNG 导出均完成。
- `git diff --check`：通过。
- 版本一致性：`overlay_measure/__init__.py`、`pyproject.toml`、`installer/OverlayMeasure.iss` 均为 2.5.0。
- 原有量测测试期望值未调整。仅将安装器元数据测试中的显式版本断言从 2.4.8 更新为 2.5.0，这是本次用户要求的版本更新。

新增测试涵盖：8/16-bit TIFF/PNG 灰度保真、中文路径和 hash；四种整数直角方向的像素中心映射与标定轴交换；已知平移与刚性变换；不同原生像素标定下的真实检测；同输入与 collector 开关边缘一致；半径/椭圆化差异；不同点数；缺角不补与覆盖率；最终 RANSAC 剔除和候选峰分离；坐标逆变换；不可靠配准阻止分析；多峰/非单调宽度抑制；完整精度导出；正式配方/状态隔离；异步圆心配准及分析线程生命周期。

## 默认卡尺路径性能抽查

在同一运行环境中，通过 `git show d4f7a3e:overlay_measure/caliper_circle_detector.py` 加载未修改的原检测器，与当前检测器交替运行。使用基线合成图、48 卡尺、相同参数，预热后各测 7 次：

| 路径 | 中位耗时 |
|---|---:|
| v2.4.8 原默认路径 | 0.441719 s |
| v2.5.0 默认路径（采集关闭） | 0.437894 s |
| v2.5.0 诊断采集开启 | 0.442306 s |

此样本中默认路径没有明显增加开销，差异处于运行波动范围。这是单数据集抽查，不是对所有 LSM 图像、尺寸和卡尺数的性能保证。

## 实际 UI QA 文件

生成于 `artifacts/diagnostics-v250/`（仓库忽略的本地输出目录）：

- `01-actual-ui.png`：实际分析总览。
- `02-actual-ui.png`：实际局部位移。
- `03-actual-ui.png`：实际灰度/梯度与原检测器多峰。
- `DEMO_A.tif` / `DEMO_B.tif`：明确标注的合成 QA 输入。
- `export/`：完整可检查的 Excel、CSV、session.json 和曲线 PNG。

QA 合成数据的注册口径结果：有效覆盖率 100%，Mean 0.01645230068 μm，RMS 0.05543092413 μm，P-V 0.20254831203 μm，Max |Δ| 0.13042515697 μm。它们来自真实共用检测器输出，不是 UI 硬编码指标，也不是实际设备测量结果。

本次没有提供真实 ZEISS LSM A/B 原图，尚未进行实际设备数据验收或测量不确定度评定。没有构建/发布 Windows 安装包，也没有推送远端仓库。
