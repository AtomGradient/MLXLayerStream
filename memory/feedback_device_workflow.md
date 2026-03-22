---
name: device_workflow
description: Device testing requires copying model files to app sandbox before launching; app needs increased-memory-limit entitlement
type: feedback
---

设备测试流程必须：
1. 先通过 `devicectl device copy to` 将模型文件拷贝到 app 的 Documents/model/ 目录
2. 拷贝完成后再启动/重启 app（app 只在启动时读取模型）
3. App 必须开启 "Increased Memory Limit" entitlement (com.apple.developer.kernel.increased-memory-limit)

**Why:** iPad 上 app 显示 "No Model found" 是因为模型没有被拷贝到 app sandbox。App 在 `.task` 中自动加载模型，如果模型在 app 启动后才拷贝进去，需要重启 app。
**How to apply:** run_device_benchmark.sh 必须按顺序：install app → copy model → terminate app → relaunch app。
