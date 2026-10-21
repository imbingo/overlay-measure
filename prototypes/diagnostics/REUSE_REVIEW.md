# SOMA Diagnostics：复用审查与最小改动方案

状态：仅 Step 1–6，等待 UI 确认。未实现 Diagnostics 正式功能。

## 1. 核查基线

- 仓库：https://github.com/imbingo/overlay-measure ，默认分支 `main`。
- 2026-10-21 本地核查：HEAD / origin/main / 标签 v2.4.8 均指向 `d4f7a3e70999f0f99a6207a40f8d047bf7b8eb9e`，提交日期 2026-10-09，标题 `release: v2.4.8 unified ellipse roundness and major-axis angle`。
- `pyproject.toml` 版本 2.4.8，Python >=3.10,<3.14；PySide6 / NumPy / OpenCV / pandas / openpyxl / Pillow 桌面项目。
- 本次新克隆仓库；未发现适用的 AGENTS.md。不是在旧目录结构假设上工作。
- 本次只在 `prototypes/diagnostics/` 新增文件，没有修改 `overlay_measure/`、`tests/` 或正式配方。

## 2. 具体复用清单

| 文件 / 真实接口 | 已有能力 | Diagnostics 建议 |
|---|---|---|
| `image_loader.py:load_image`, `_load_raster_image_unicode_safe` | `np.fromfile` + `cv2.imdecode(IMREAD_UNCHANGED)`，中文路径、TIFF/PNG、原始深度；输出 float32 灰度 | 直接调用两次，独立保存 ImageData A/B。每个源文件另记 SHA256、大小、时间和源 dtype。彩色转灰不是原始单通道保真场景，须区分。 |
| `image_loader.py:display_to_uint8`, `normalize_to_uint8` | 显示映射与分析灰度分离 | 只用于背景，不写回 gray。 |
| `models.py:ImageData` | path / gray / display_name / source_dtype / source_min/max | 保留；会话另存标定和变换。gray 只读或隔离副本，不绑定 Upper/Lower。 |
| `models.py:Roi.normalized/center/contains_points/edge_target_mask` | ROI 语义、环带、卡尺数/宽度/方向/目标边缘 | 复用模型与参数副本；不能直接复制 A 的像素矩形到 B。 |
| `models.py:DetectionParams` | Gaussian、梯度门槛、剖面步长、极性、RANSAC、残差限制等 | 单个不可变参数快照驱动两次调用；卡尺数、宽度、搜索方向实际属于 Roi，不能错误地只复制 DetectionParams。 |
| `caliper_circle_detector.py:detect_caliper_circle` | 径向/切向采样、候选峰、一致性选峰、拟合、剔除、直径统计 | A/B 都调用这一入口，绝不另写检测器。建议可选 keyword-only collector，默认 None。 |
| 同文件 `_profile_edge_candidates`, `_select_consistent_strongest`, `_initial_candidate` | 二次峰插值、目标边缘选择、半径层一致性与迭代重分配 | 保持执行顺序、dtype、阈值、随机性及结果不变；只旁路记录。 |
| `subpixel_edge_detector.py:detect_subpixel_edges/refine_contour_edges`, `_bilinear_sample`, `_quadratic_peak_offset` | 通用亚像素边缘；卡尺已共用双线性采样与峰插值 | 圆孔优先卡尺路径，一般轮廓未来经现有路径；不复制另一套亚像素实现。私有采样函数若需公共化，仅增加兼容包装。 |
| `circle_ellipse_fitter.py:fit_circle_least_squares/fit_circle_geometric_robust/fit_circle_ransac/fit_ellipse/circle_diameter_statistics/fit_mark_shape` | 现有圆/椭圆拟合与统计 | 检测参数与拟合口径保持一致。诊断的形状差异是在点集上分析，不代替正式圆度定义。 |
| `measurement_service.py:detect_manual_roi`, `fitting_mode_for_roi` | ROI 语义分发、统一 DetectionResult、物理指标和质量标注 | 建议让 service 透传可选 collector，Diagnostics 可调用这个公共检测入口。其 `layer` 元数据需兼容 A/B；不能借用主窗口 upper/lower 状态。圆卡尺分支不依赖 upper/lower fitting mode；其他分支需单独审计。 |
| `production_measurement.py:refine_circle_candidate` | 自动模式先构建卡尺 Roi，再调用同一个 detect_caliper_circle | 保持原调用不变。自动候选细化参数由生产配置生成，不能声称任意手动 ROI 与自动模式必然相同；一致性前提是实际算法、ROI、参数完全一致。 |
| `measurement_units.py:points_px_to_um/point_um_to_px/axis_scale_um_per_px/points_to_um_distances/robust_circle_um/radial_diameter_statistics_um` | X/Y 独立标定，float64 物理坐标，Y 向下 | 必须共用；新增变换组合管理，不用 `scalar_px_to_um` 的平均像素尺寸计算任意方向位移。 |
| `ui_components.py:ImageCanvas.set_image/image_to_widget/widget_to_image_float/reset_view/zoom_by` | 原始灰度显示、缩放平移、浮点坐标、ROI 绘制/编辑 | 隔离实例或提取显示/坐标映射适配层；复用交互逻辑，不继承主窗口量测状态。 |
| `ui_components.py:CollapsibleSection` | 轻量折叠面板 | 用于详细配准、参数和追溯信息。 |
| `ui_components.py:ImageDiagnosticsDialog.set_profile`, `GrayProfilePlot.set_profile` | 单图原灰度线剖面与统计 | 保留原入口；图表控件可最小扩展到多曲线，但现有图没有梯度峰数据和双图联动。不能把它当成完整诊断引擎。 |
| `ui_components.py:ImageCanvas._emit_profile` | 双线性采样和各向异性物理长度 | 当前采样实现内嵌 UI，边界裁剪行为也与 `_bilinear_sample` 需核对。提取公共采样包装时必须回归；选峰解释直接用卡尺 collector 的原剖面，不在 UI 重新选峰。 |
| `recipe_manager.py:load_recipe/save_recipe`, `_roi_to_dict/_roi_from_dict`; `models.py:dataclass_to_dict` | dataclass 配置序列化、旧版 ROI 迁移 | 复用字段和序列化机制。现有文件格式绑定 MarkRecipe upper/lower，不直接拿 save_recipe 写 A/B；新 session schema 引用参数快照，无双向写回。 |
| `result_exporter.py:export_results`, `_style_sheet`, `resize_dimensions_preserving_aspect` | UTF-8 BOM CSV、openpyxl 多表与图像、表格样式、追溯信息 | 复用写入和样式基础设施，但需新增诊断数据表模式。现函数 `.round(3)`、生产字段改名和 summary 优先 CSV 不适合原始点/逐角度数据；新增可选精度/表集合接口，默认原行为。 |
| `export_naming.py:build_export_filename` / `export_visualization.py` | 命名与现有图像导出相关基础 | 使用诊断会话类型，避免与生产量测文件覆盖；PNG 图表单独输出。 |

## 3. 卡尺可选诊断输出：只记录，不重跑

建议接口（未实现）：`detect_caliper_circle(gray, roi, params, *, diagnostics=None)`；CaliperCircleResult 的现有字段及默认调用保持兼容。

当前结果已含 `edge_points/rejected_points/gradients/rejected_gradients/inlier_mask/caliper_windows`、圆参数、覆盖率和直径统计。窗口里已有 angle（弧度）、center、length、width、gradient、candidate_count、accepted、selected_radius，但不等于完整采样轨迹。

在原循环内按 caliper id 采集：start、radial/search_vector、tangent、radial_samples、tangent_offsets、有效采样 mask、原生 profile、候选 sub_index 与 gradient、阈值前/后候选、最终 selected candidate id、剔除原因、拟合 inlier 状态。完整二维采样坐标可由上述轴和偏移重建，或仅诊断开启时物化。梯度用原 profile、原 step、原 dtype、同一 `np.gradient` 口径；保存与 Gaussian 平滑后的采样对应关系，不冒称是未经滤波的原灰度。原始灰度另行显示并标注。

关键细节：现有 window `accepted=True` 在拟合前赋值，不能用于解释 RANSAC 最终 inlier。必须保存 selected→window 的稳定索引以及最终 mask；无峰、低梯度、越界、未选中候选和拟合剔除分开记录。不得把“非选中峰”混成“拟合剔除点”。

默认 None 不分配 profile 副本或候选快照；开启时也不改变候选对象、排序、拟合输入、循环和随机状态。验证 collector 开关的数组/掩码/圆参数一致性，并单测默认路径耗时分布。RANSAC 有随机性时在回归两端固定相同输入和随机状态，不能靠重写期望值过关。

## 4. 确实需要新增的模块

建议结构（全部留待确认后开发）：

```
overlay_measure/diagnostics/
  models.py                # A/B Session、图像/物理变换、对应点/有效性、追溯
  registration.py          # 90°无损索引映射、平移/刚性、手工独立对应点、可靠性
  edge_comparison.py       # 物理角度径向对应、缺失区、统计、中心对齐口径
  profile_comparison.py    # 同物理采样轴映射、采集数据解释、宽度适用性
  diagnostics_service.py   # 独立会话、只读快照、既有检测入口编排
  ui_diagnostics.py        # 单工作区，三种视图，多曲线、双向选择
```

调用：独立会话 → load_image(A/B) → 参数/ROI/标定快照 → 建立物理坐标刚性变换 → 分别映射原图 ROI → detect_manual_roi(透传 collector) → 现有 detect_caliper_circle → 现有拟合 → 点集转换与对应 → 指标/曲线/剖面 → 诊断导出适配。禁止在变换插值后的显示图上重新做正式边缘检测。

配准只允许确定旋转、平移或刚性，记录原坐标→方向统一→物理标定→统一参考坐标的完整链和逆变换。先独立基准，只有单孔时允许明确标注的圆心平移，旋转未知；不能因此推断真实圆心漂移。原图不覆盖。可靠性不达门槛时不输出有效定量结论。

**ROI 支持边界是实质性问题：** 当前圆卡尺固定 `angle=2π*index/count`，Roi.angle_deg 不参与其采样；映射到旋转后的 B 不能自动保持相同卡尺物理角度。非等比例 X/Y 标定或不同轴比例还会使物理圆环在像素中变成椭圆，现有 Caliper Circle Roi 无法精确表示。不能用外接圆静默代替。正式实现需在同一采样核心最小增加默认不变的采样坐标框架（角相位/物理到像素基向量），或首版对不可表达的组合显式阻止定量比较；这需要开发时做专项验证。普通同标定平移情况可以直接映射现有圆 ROI。90°交换图像尺寸、像素轴标定和坐标原点偏移；不能只交换宽高。

对应关系在统一物理坐标建立，约定 X 向右、Y 向下、角度顺时针、向孔外位移为正。按物理角度射线与参考轮廓或局部法线匹配；设最大角间隙，缺失点不跨越插值，不按数组索引配对。覆盖率定义为计划角度集合上有效 A/B 对应占比，另记有效角区间；不能照搬单图 `_angular_statistics` 覆盖率。

Mean / RMS / P-V / P95(abs) / Max(abs) 只对有效对应计算，并记录角采样权重。并列保存原图圆心、配准后圆心差、各自中心对齐形状差；px 位移必须注明是哪张参考像素轴的单位。各向异性时优先 μm，不能把 px 标量乘平均像素尺寸。

## 5. 耦合与生产隔离

- `ui_workflows.py:start_image_diagnostics` 使用当前层画布、工程模式与 `_image_diagnostics_dialog`；新会话不能复用这些状态引用。
- `ImageCanvas.set_context` 涉及 active_mark_id/active_layer、auto_detections、geometry、ROI 编辑信号，直接挂到主窗口可能污染配方。独立 adapter 只连接独立 session 信号，必要时提取窄接口。
- `ui_builders.py` 更多菜单目前有“图像诊断与边缘分析”，右侧还有原始灰度诊断按钮；`ui_state.py` 控制工程模式可用性。建议新增“SOMA Diagnostics”并保留“单图灰度剖面（原功能）”可访问，沿用工程模式权限。
- DetectionParams、Roi、MeasurementConfig 用 deepcopy/replace 独立快照；禁止写回 MainWindow.config/params/marks、生产 recipe、DX/DY 或批量测量状态。检测结果也不共享可变数组。
- 校准共享的是类型和换算函数，各图可有独立标定。Gaussian px 参数在不同标定下不代表同一物理模糊宽度，应记录物理等效值，必要时阻止不具可比性的组合。
- float32 可精确表示 0–65535 的每个整数，因此现有单通道 uint8/uint16 转换本身无灰度量化损失；Gaussian 和插值随后有浮点舍入，不应改动成熟算法 dtype 以追求表面精度。Diagnostics 的坐标变换、匹配与统计使用 float64。保真结论需通过 TIFF/PNG 实测，不能扩展到所有 float/彩色格式。

## 6. 确认后的测试与原基准策略

当前阶段没有修改检测器，也未宣称生产 pytest 全部通过。确认后第一步在此原始提交上保存基线：样例图 SHA256、recipe/ROI/params/config、随机状态、有效/剔除点及梯度、circle/ellipse/roundness、批量结果、DX/DY、导出表值及版本环境。以数值/数组内容为准，不以 XLSX ZIP 时间戳比较。

现有回归重点：`test_caliper_roundness.py`、`test_roundness_axis.py`、`test_measurement_service.py`、`test_v160_production.py`、`test_v227_halo_edges.py`、`test_v246_batch_pairing_summary.py`、`test_units_and_reports.py`、`test_v220_safety_and_units.py`、`test_v240_auto_semantic_and_diagnostics.py`、`test_ui_smoke.py`。

新增 pytest 覆盖用户列出的 12 项：8/16-bit/中文路径保真；90°索引与标定轴交换；已知平移/刚性和错误基准拒绝；已知半径与椭圆化位移；缺角覆盖和不同点数；各向异性坐标/逆变换；相同输入边缘一致；collector 开关一致和性能；A/B 状态不污染；原量测与导出回归。多峰/错物理边缘、低覆盖、单圆旋转不可辨识及非均匀标定 ROI 边界需负例。不能修改原测试期望掩盖变化。

导出拟采用 metrics.csv/xlsx、edges_A/B.csv、displacement.csv、曲线 PNG、session.json；包括有效 mask/原因、角度区间、配准可靠性、ROI、参数、标定、坐标约定和文件 hash。原始点与计算值不走三位小数的生产展示舍入路径。
