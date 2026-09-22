# Windows 安装与升级

V2.3.0 使用两层打包：

1. PyInstaller `onedir` 构建 `dist/OverlayMeasure/`。
2. Inno Setup 将整个运行目录封装成一个 `SOMA_Vision_Metrology_Vx.y.z_Setup.exe`。

用户只需要运行 Setup。安装程序默认安装到：

```text
C:\Program Files\Overlay Measure
```

安装向导使用 Inno Setup 官方内置语言，软件名称、快捷方式和应用界面保持中文，避免依赖未随编译器发布的第三方语言文件。

配方、日志、恢复状态和测量导出不写入该目录，因此覆盖升级不会删除用户数据。

## 构建环境

- Windows 10/11 x64
- Python 3.10-3.13
- `requirements-dev.txt`
- Inno Setup 6

## 一键构建

在仓库根目录运行：

```powershell
.\scripts\build_release.bat -Python "D:\python\python.exe"
```

脚本依次执行：

1. 校验版本号。
2. 运行完整 pytest。
3. 生成 Windows EXE 版本资源。
4. 使用 PyInstaller 构建文件夹式应用。
5. 以 `--smoke-test` 启动打包后的 EXE。
6. 使用 Inno Setup 构建安装程序。
7. 生成 `update_manifest.json` 和 `.sha256`。

## 升级规则

所有 V1.8.1 及后续版本必须保持 `OverlayMeasure.iss` 中的 `AppId` 不变。新版完整安装程序会识别原安装目录并覆盖升级。

当前不提供二进制差分补丁。对于量测软件，完整安装包更容易验证和回退。将来可以由受控更新器读取 `update_manifest.json`，下载并校验完整安装程序后交给工程人员确认升级。
