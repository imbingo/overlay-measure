# SOMA Vision Metrology V2.6.2

本次为缺陷修复版本，不改变测量算法、配方格式与导出内容。

**修复**

- 近似直线 ROI 点击“分析 ROI”时报错 `Both input arrays must be (arrays of) 3-dimensional vectors, but they are 2 and 2 dimensional instead.`。原因是安装包内置的 NumPy 2.5 移除了二维向量叉积，V2.4.7 – V2.6.1 均受影响。
- 几何工具中的“直线交点”和“点到直线距离”存在同样问题，已一并修复。

测量结果、判定逻辑与 V2.6.1 保持一致。
