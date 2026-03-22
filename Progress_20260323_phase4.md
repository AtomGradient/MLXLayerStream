# Progress — Phase 4: 设备端验证（进行中）

**日期**: 2026-03-23
**设备**: iPad Air M3 (8GB), iPhone 15 Pro Max (8GB)

## 设备 Baseline 结果

### Qwen3.5-0.8B-8bit (988 MB)
| 设备 | Decode TPS | Prompt TPS | Peak Mem | Load Time |
|------|-----------|------------|----------|-----------|
| Mac Studio | 302.7 | - | 810 MB | - |
| **iPad Air M3** | **103.2** | 183.3 | 799 MB | 0.8s |
| **iPhone 15 Pro Max** | **53.5** | 116.5 | 792 MB | 1.0s |

### Qwen3.5-4B-4bit (2.9 GB)
| 设备 | Decode TPS | Prompt TPS | Peak Mem | Load Time |
|------|-----------|------------|----------|-----------|
| Mac Studio | 148.1 | - | 2340 MB | - |
| **iPad Air M3** | **35.5** | 56.8 | 2330 MB | 1.4s |
| **iPhone 15 Pro Max** | **18.4** | 41.9 | 2317 MB | 1.5s |

### Qwen3.5-9B-6bit (6.9 GB) — OOM!
| 设备 | 结果 |
|------|------|
| iPad Air M3 | **OOM — 被系统杀死** (24.7s 后) |

## 关键分析

### 1. 带宽缩放验证
| 模型 | iPad/iPhone TPS 比 | 理论带宽比 (100/50) |
|------|-------------------|-------------------|
| 0.8B-8bit | 103.2/53.5 = **1.93x** | 2.0x |
| 4B-4bit | 35.5/18.4 = **1.93x** | 2.0x |

**完美匹配！** 设备间 TPS 差异完全由内存带宽决定。

### 2. Mac vs iPad 缩放
| 模型 | Mac/iPad TPS 比 | 理论带宽比 (800/100) |
|------|----------------|-------------------|
| 0.8B-8bit | 302.7/103.2 = **2.93x** | 8.0x |
| 4B-4bit | 148.1/35.5 = **4.17x** | 8.0x |

Mac vs iPad 比不完全是 8x — 小模型在 Mac Studio 上不完全 bandwidth-bound（compute overhead 占比更大）。4B 模型更接近理论值。

### 3. 9B-6bit OOM — Layer-Streaming 的必要性证明
- 9B-6bit 权重 6.9 GB < iPad 物理内存 8 GB
- **但 iOS 系统开销 ~2-3 GB → 实际可用内存 ~5-6 GB**
- 6.9 GB 模型无法在 8GB iPad 上全驻留加载
- **这直接证明了 Layer-Streaming 对 8GB 设备的必要性**
- 即使权重 < 设备物理内存，也可能需要 streaming

### 4. 修正后的 8GB 设备可用内存估计
- iOS 系统开销: ~2-3 GB
- 实际可用 GPU 内存: **~5-6 GB**（含 increased-memory-limit entitlement）
- 最大全驻留模型: ~5 GB 权重 → **4B-4bit (2.9 GB) 或 4B-6bit (3.8 GB)**
- 9B 系列全部需要 streaming！

## 下一步
1. 在设备上测试 9B-6bit with streaming（需要 Swift 实现）
2. 继续测试更多 baseline 模型
3. 实现 Swift 版 Layer-Streaming
