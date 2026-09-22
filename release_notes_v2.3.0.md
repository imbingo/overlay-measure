# SOMA Vision Metrology V2.3.0

## 性能改动 1

- 手动多 ROI 的通用亚像素边缘点按轮廓均匀限采样，默认最多 2,000 点。
- 边缘扫描和 RANSAC 圆拟合可在内部循环响应取消。
- 非激活 ROI 默认只显示中心与编号；诊断模式保留完整拟合轮廓。

## 识别改动 2

- 自动孔阵列识别同时使用全局 Otsu 与局部自适应阈值，以改善不均匀照明下的漏孔。
- 自动候选、轮廓和时间预算成为 Recipe 参数，默认值为 128、256、12 秒。
- 移除双图候选汇总后固定只精测 32 个的截断；每层候选均可进入精测。

## Compatibility

Existing recipes remain compatible.  New performance and auto-array settings use
safe defaults when absent from an older recipe.
