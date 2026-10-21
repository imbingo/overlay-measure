# SOMA Vision Metrology V2.5.1

修复 V2.5.0 的 SOMA Diagnostics 入口不可见问题。此前“更多”菜单和功能对象已经创建，但按钮未加入可见工具栏，导致无法从主界面进入。

使用位置：顶部切换到 **工程模式** → 图像工具栏右侧 **更多**（在“分析 ROI”左侧）→ **SOMA Diagnostics / 成像诊断**。

- 增加三个窗口尺寸下的可见性、真实菜单点击和诊断窗口打开回归测试。
- 保留原单图灰度诊断入口、生产模式布局及全部正式量测算法。
- 全量 212 项测试通过，包括先复现入口缺失再验证修复的回归测试。
- 附带完整 Windows x64 安装包、SHA256 校验文件及更新清单，可覆盖升级 V2.5.0。

[Diagnostics 使用说明](https://github.com/imbingo/overlay-measure/blob/v2.5.1/docs/diagnostics.md)
