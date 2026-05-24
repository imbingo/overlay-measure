# 对位偏差测量软件 V1.0.5

用于识别上下贴合结构中 Upper Mark 与 Lower Mark 的中心，并计算 Upper 相对 Lower 的对位偏差 ΔX / ΔY / Overlay R。

## V1.0.5 更新说明

本版重点新增 **Advanced ROI**，用于同心圆、同心方孔、多层边缘 mark 的选择性拟合。

新增 ROI 类型：

- `Rectangle`：普通矩形 ROI。
- `Circle`：圆形 ROI，适合单个圆孔。
- `Annulus`：同心圆环 ROI，适合多个同心圆时只选其中一圈。
- `Rectangular Ring`：矩形环 ROI，适合多个同心方孔/方框时只选其中一圈。

新增 ROI 参数：

- `Target Edge`
  - `All Edges`：使用环带内全部边缘点。
  - `Near Inner Boundary`：优先使用靠近内边界的边缘点。
  - `Near Outer Boundary`：优先使用靠近外边界的边缘点。
  - `Strongest Edge`：保留当前环带内强边缘，几何上等同 All Edges，主要为后续增强预留。
- `Inner Ratio`：内边界尺寸占外边界尺寸的比例。
- `Ring Angle`：矩形环 ROI 的旋转角。

使用建议：

- 多个同心圆：选择 `Annulus`，拖出外圆边界，再通过 `Inner Ratio` 调整内圆边界。
- 多个同心方框：选择 `Rectangular Ring`，拖出外方框边界，再通过 `Inner Ratio` 和 `Ring Angle` 调整内方框与旋转角。
- 如果先画了 ROI 后发现类型不对，可以切换 ROI Type，然后点击 `应用到当前 ROI`。
- 图像上会显示外边界、内边界和中心十字，方便确认算法实际使用的有效区域。

## 原有核心特性

- 支持 Single Image Mode：Upper/Lower mark 在同一张图中。
- 支持 Dual Image Mode：Upper/Lower mark 来自两张不同图像。
- 支持 Upper / Lower 独立拟合模式，可做方孔对圆孔测量。
- 主算法为亚像素边缘点提取 + 圆/椭圆/方孔矩形拟合。
- 支持 registration offset，用于双图坐标补偿。
- 输出中心坐标、直径/尺寸、拟合残差、边缘点数量、置信度。
- 支持 CSV / Excel 导出。
- 支持 JSON recipe 保存与加载。
- 支持图像滚轮缩放、右键/中键拖动画面、双击视图复位。

## 安装

```bash
cd overlay_mark_measure_v1_0_5
pip install -r requirements.txt
python main.py
```

## PyCharm 运行

1. PyCharm 打开 `overlay_mark_measure_v1_0_5` 文件夹。
2. 配置 Python Interpreter。
3. 在 Terminal 中执行：

```bash
pip install -r requirements.txt
```

4. 右键运行 `main.py`。

## 使用流程

### 单图模式

1. Measurement Mode 选择 `Single Image`。
2. 点击 `导入 Upper/Single 图像`。
3. 选择当前 Mark 和 Layer。
4. 在 Advanced ROI 区选择 ROI Type。
5. 在图像上左键拖出 ROI。
6. 分别画 Upper ROI 与 Lower ROI。
7. 点击 `分析当前 Mark` 或 `分析全部 Mark`。
8. 查看 ΔX / ΔY / R。

### 双图模式

1. Measurement Mode 选择 `Dual Image`。
2. 分别导入 Upper Image 和 Lower Image。
3. 在左侧 Upper 图像画 Upper ROI，在右侧 Lower 图像画 Lower ROI。
4. 如两张图存在坐标偏移，输入 Registration Offset X/Y。
5. 点击分析。

## 同心圆/同心方框 ROI 示例

### 同心圆

1. `ROI Type = Annulus`。
2. `Fitting Mode = Circle` 或 Upper/Lower 分别设为 Circle。
3. 拖动 ROI，让外圆边界包住所需环带的外侧。
4. 调整 `Inner Ratio`，让内虚线边界避开不想参与拟合的内圈边缘。
5. 点击 `应用到当前 ROI`。
6. 点击分析。

### 同心方框

1. `ROI Type = Rectangular Ring`。
2. 对应层的 Fitting Mode 设为 `Rectangle`。
3. 拖动 ROI，让外矩形包住所需方框区域。
4. 调整 `Inner Ratio` 和 `Ring Angle`。
5. 点击 `应用到当前 ROI`。
6. 点击分析。

## 坐标与偏差定义

默认偏差定义：

```text
ΔX = X_upper - X_lower_corrected
ΔY = Y_upper - Y_lower_corrected
```

其中：

```text
X_lower_corrected = X_lower + registration_offset_x
Y_lower_corrected = Y_lower + registration_offset_y
```

界面中 pixel 和 μm 两套结果都会显示。

## 支持的图像格式

- png / jpg / jpeg / bmp / tif / tiff
- npy：二维数组
- csv / txt：二维数值矩阵

`.sur` 文件格式由不同设备厂商实现可能不同，V1.0.5 没有内置稳定通用解析器。如果你的设备只能导出 `.sur`，建议先从设备软件导出 tif/csv 高度图，或者后续提供 `.sur` 样例文件后再定制解析器。

## 算法说明

亚像素边缘流程：

1. ROI 外包围框内灰度归一化。
2. Gaussian 去噪。
3. Canny/Sobel 粗边缘提取。
4. 对每个粗边缘点计算梯度方向。
5. 沿梯度法线方向采样灰度剖面。
6. 找梯度峰值，并通过二次曲线插值得到亚像素边缘点。
7. 按 ROI Type 过滤有效边缘点。
8. 根据 Target Edge 进一步筛选边缘点。
9. 根据 Fitting Mode 使用圆拟合、椭圆拟合或方孔矩形拟合。
10. 输出中心、直径/尺寸、残差、边缘点数和置信度。

## 注意

- 环形 ROI 的核心作用是限制“算法看哪一圈”，不是自动判断哪一圈是正确边缘。
- 如果同一环带内还有多条真实边缘，应减小环带宽度或调整 Target Edge。
- 真实重复性不仅取决于算法，还受到焦面稳定性、照明、噪声、振动、mark 边缘质量影响。
- 如果上下图像不是同一 stage 坐标基准，必须输入 registration offset，否则计算出的对位偏差没有物理意义。

## 示例数据

`sample_data/` 中提供了测试文件：

- `sample_upper.png`
- `sample_lower.png`
- `sample_single.png`
- `demo_recipe.json`
- `sample_square_upper.png`
- `sample_square_lower.png`
- `sample_square_single.png`
- `demo_square_recipe.json`
- `sample_concentric_single.png`
- `sample_concentric_square_single.png`
- `demo_annulus_recipe.json`
- `demo_rect_ring_recipe.json`

测试圆环 ROI：

1. 选择 `Single Image`。
2. 导入 `sample_concentric_single.png`。
3. 加载 `demo_annulus_recipe.json`。
4. 点击 `分析当前 Mark`。

测试矩形环 ROI：

1. 选择 `Single Image`。
2. 导入 `sample_concentric_square_single.png`。
3. 加载 `demo_rect_ring_recipe.json`。
4. 点击 `分析当前 Mark`。
