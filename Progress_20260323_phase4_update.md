# Progress — Phase 4 Update: 设备端验证 + Swift Streaming 调试

**日期**: 2026-03-23
**设备**: iPad Air M3 (8GB), iPhone 15 Pro Max (8GB)

## 设备 Baseline 结果 (确认)

| 模型 | iPad TPS | iPhone TPS | Peak Mem | 带宽比 |
|------|---------|-----------|----------|--------|
| 0.8B-8bit | 103.2 | 53.5/54.0 | 792-799 MB | 1.93x (理论 2.0x) |
| 4B-4bit | 35.5 | 18.4 | 2317-2330 MB | 1.93x |
| **9B-6bit** | **OOM** | — | — | — |

## 9B-6bit OOM 分析
- 权重 6.9 GB < 8 GB 物理内存
- iOS 系统开销 ~2-3 GB → 实际可用 ~5-6 GB
- 9B-6bit 无法在 8GB 设备上全驻留 → **Layer-Streaming 是刚需**

## Swift Layer-Streaming 实现状态

### 已完成
1. `split_weights.py` — 按层拆分 safetensors（排除 vision_tower，排除 index 文件）
2. `streamingForward()` 添加到 Qwen35.swift — per-layer load/unload callbacks
3. `StreamingEngine.swift` — layer load/unload/stats
4. `StreamBenchmarkApp.swift` — streaming 模式检测 + benchmark
5. `streamLog()` — 设备端调试日志

### 阻塞问题
**`LLMModelFactory.loadContainer()` 无法用于 streaming 加载**

原因：
1. `loadContainer` 创建完整模型架构（32 层）
2. 每层的 Linear 模块分配全精度随机权重（float32）
3. `loadWeights` 用 `verify: .all` 要求所有参数都有对应权重
4. 量化依赖 weights 字典中的 `.scales` 键 — non_layer 没有层的 `.scales`
5. 层不被量化 → 保持 Linear → 32 层 × ~830 MB = ~26 GB 随机权重 → OOM

### 解决方案（下次会话实现）

**方案 A: 自定义模型创建流程**
```swift
// 1. 从 config.json 创建模型架构
let model = typeRegistry.createModel(config)

// 2. 直接量化所有可量化模块（不依赖 weights 字典）
quantize(model) { _, module in
    return (groupSize: 64, bits: 6)  // 从 config 读取
}
// 量化后 Linear → QuantizedLinear（weight 变小）

// 3. 只加载 non_layer 权重
let weights = loadArrays(url: nonLayerURL)
let sanitized = model.sanitize(weights: weights)
model.update(parameters: sanitized, verify: .none)
eval(sanitized.values)

// 4. 层权重在 streamingForward 中按需加载
```

**方案 B: 修改 MLX Swift 的 QuantizedLinear.init**
- 添加一个不执行实际量化的 init（只设置 shape/dtype）
- "placeholder" 初始化，权重稍后从文件加载

## 其他发现

1. **设备 container 隔离**: 每次 install 创建新 container，旧数据丢失
2. **文件残留问题**: `streaming_info.json` 在切换模型时残留 → 误触发 streaming 模式
3. **需要在推模型前清理设备上的旧文件**
4. **设备存储空间充足**（用户确认）

## 文件

- `split_weights.py` — 按层拆分工具
- `StreamBenchmarkApp/` — iOS benchmark app
- `run_device_benchmark.sh` — 设备自动化脚本
- `Qwen35.swift` (H2OAttnScore 仓库) — 添加了 streamingForward

## 下一步

1. 实现方案 A：自定义模型创建 + 量化 + 部分权重加载
2. 在设备上验证 9B-6bit streaming（预估 TPS 和实际 TPS 对比）
3. 如果成功，扩展到 9B-8bit 和 27B-4bit
