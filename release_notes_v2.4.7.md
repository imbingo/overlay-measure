# SOMA Vision Metrology V2.4.7

## V2.4.7 卡尺圆轮廓圆度

- 卡尺圆新增独立圆度：最终有效轮廓点经 X/Y 分别标定后，计算相对拟合中心的 Rmax - Rmin，单位 μm，不乘 2。
- 结果表、批量结果、Excel 识别明细及明细 CSV 贯通 roundness_um；不足 3 个有效点返回空值。
- 保留原有直径 PV、平均/最大直径、椭圆圆度、RANSAC 和配方兼容性。
- 新增 13 项测试，全量 155 项测试通过。

计算说明：RANSAC有效轮廓点 → 标定物理半径 → 最大半径 - 最小半径。

椭圆圆度仍为 (ellipse_major_um - ellipse_minor_um) / 2，与卡尺圆轮廓圆度独立。
