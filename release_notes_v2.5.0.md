# SOMA Vision Metrology V2.5.0

新增 SOMA Diagnostics（成像诊断），用于参考/当前两图的轻量 LSM 成像变化排查。

- 入口：工程模式 → 更多 → SOMA Diagnostics / 成像诊断。
- A/B 会话独立于正式 Upper/Lower、Recipe 与 DX/DY；原单图灰度诊断入口仍可使用。
- 共用卡尺亚像素边缘算法，叠加边缘点、剔除点、圆、卡尺及局部位移线段。
- 支持方向统一、物理坐标刚性/平移配准、径向差异统计及原灰度/滤波灰度/梯度选峰解释。
- 导出 Excel、CSV、PNG 和包含源文件 SHA256 的 session.json；原始点数据保留完整精度。

刚性配准优先使用独立稳定特征；圆心平移只能判断相对形状，不能据此判断真实圆心漂移。数值小数位数不是测量不确定度。生产量测算法基线与测试口径保持不变。

本版本通过 209 项 pytest，包括正式量测回归基线、灰度保真、配准、边缘差异、导出和会话隔离测试。

发布附件包含 Windows x64 安装包、SHA256 校验文件及 update_manifest.json。

参见 [使用说明](https://github.com/imbingo/overlay-measure/blob/v2.5.0/docs/diagnostics.md) 和 [验证记录](https://github.com/imbingo/overlay-measure/blob/v2.5.0/docs/diagnostics_validation.md)。
