# Progress — Phase 1-2: MLX 分析 + Layer-Streaming 实现

**日期**: 2026-03-22
**设备**: Mac Studio M2 Ultra (192 GB, ~800 GB/s, SSD ~7.4 GB/s)

## 关键发现

### Phase 1: MLX 不使用 mmap 加载权重
- MLX 用标准 `read()`/`lseek()` 文件 I/O，权重被完整加载到 GPU 内存
- `mx.set_memory_limit()` 只是指导性限制，超出则抛异常
- **Layer-Streaming 必须在应用层实现**（推翻了 PLAN 中 "MLX mmap 自动做 streaming" 的假设）

### Phase 2: Layer-Streaming 实现与验证

**核心机制**:
1. 模型架构加载，但层权重不驻留
2. 每层 forward 前: 从 safetensors 加载权重 (mx.load → lazy → mx.eval)
3. 每层 forward 后: 替换为占位符 + mx.clear_cache() 释放 GPU 内存
4. embed_tokens, norm, lm_head 始终驻留（小，边界需要）

**正确性**: Logits 完全一致 (max diff = 0.0)

## Streaming Benchmark 结果 (Mac Studio)

### 内存减少

| 模型 | 正常内存 | 流式内存 | 减少 |
|------|---------|---------|------|
| 0.8B-8bit | 0.74 GB | 0.29 GB | 60% |
| 9B-8bit | 8.86 GB | 2.29 GB | 74% |
| **27B-4bit** | **14.09 GB** | **1.70 GB** | **88%** |

### TPS vs 驻留比例

#### Qwen3.5-0.8B-8bit (24 layers, baseline 309.5 TPS)
| 模式 | TPS | 内存 | I/O% | 驻留层 |
|------|-----|------|------|--------|
| full_resident | 309.5 | 0.81 GB | 0% | all |
| stream_0pct | 8.5 | 0.31 GB | 74% | 0 |
| hybrid_50pct | 15.1 | 0.55 GB | 67% | 12 |
| hybrid_75pct | 25.6 | 0.68 GB | 56% | 18 |

#### Qwen3.5-4B-4bit (32 layers, baseline 150.9 TPS)
| 模式 | TPS | 内存 | I/O% | 驻留层 |
|------|-----|------|------|--------|
| full_resident | 150.9 | 2.34 GB | 0% | all |
| stream_0pct | 3.8 | 0.47 GB | 79% | 0 |
| hybrid_50pct | 6.9 | 1.40 GB | 73% | 16 |
| hybrid_75pct | 12.2 | 1.87 GB | 64% | 24 |

#### Qwen3.5-9B-8bit (32 layers, baseline 68.3 TPS, **主要目标**)
| 模式 | TPS | 内存 | I/O% | 驻留层 |
|------|-----|------|------|--------|
| full_resident | 68.3 | 8.97 GB | 0% | all |
| stream_0pct | 1.7 | 2.31 GB | 81% | 0 |
| hybrid_50pct | 3.2 | 5.73 GB | 77% | 16 |
| hybrid_75pct | 5.9 | 7.44 GB | 70% | 24 |

#### Qwen3.5-27B-4bit (64 layers, baseline 35.6 TPS, **hero 目标**)
| 模式 | TPS | 内存 | I/O% | 驻留层 |
|------|-----|------|------|--------|
| full_resident | 35.6 | 14.34 GB | 0% | all |
| stream_0pct | 0.9 | 1.73 GB | 80% | 0 |
| hybrid_25pct | 1.1 | 4.91 GB | 80% | 16 |
| hybrid_50pct | 1.7 | 8.10 GB | 77% | 32 |
| hybrid_75pct | 3.1 | 11.29 GB | 71% | 48 |

## 关键分析

### 1. I/O 是绝对瓶颈
- 全流式模式 I/O 占总时间 74-81%
- 每层加载约 3-14ms（取决于层大小），卸载 0.4-0.8ms
- Mac Studio 的 SSD 读取被 OS 文件缓存加速

### 2. 驻留比例与 TPS 的权衡
- 75% 驻留可恢复 baseline TPS 的 ~8-9%
- 每增加 25% 驻留，TPS 约提升 30-60%
- **但内存占用线性增加**

### 3. 8GB 设备可行性分析
- **9B-8bit + 75% 驻留**: 7.44 GB 内存, 5.9 TPS → 8GB 设备临界可行
- **27B-4bit + 25% 驻留**: 4.91 GB 内存, 1.1 TPS → 8GB 设备可运行但慢
- **27B-4bit 全流式**: 1.73 GB 内存, 0.9 TPS → 在任何设备上都能跑

### 4. 设备端预估（需实测验证）
Mac Studio 数据受 OS 文件缓存影响，设备端 SSD 直读会更慢：
- iPad SSD ~2 GB/s, Mac Studio SSD ~7.4 GB/s
- 27B 单层 ~218 MB: 设备上加载 ~109ms vs Mac Studio ~29ms
- **设备端全流式 27B 预估: 0.2-0.3 TPS**
- **设备端 25% 驻留 27B 预估: 0.4-0.6 TPS**

## 优化方向

1. **Prefetching**: 计算当前层时预加载下一层（需要异步 I/O）
2. **权重预分割**: 将 safetensors 按层拆分为独立文件，减少 mx.load() 开销
3. **更激进量化**: 3bit/2bit 减小每层大小
4. **选择性驻留**: 基于 profiling 选择驻留哪些层（而非简单 first_n）

## 文件

- `layer_streaming.py` — 核心 streaming 实现
- `test_streaming.py` — 正确性和性能测试
- `benchmark_streaming.py` — 完整 benchmark 脚本
- `results/streaming_benchmark.json` — 所有结果数据

## 下一步

1. **优化 I/O**: prefetching + 权重预分割
2. **设备端测试**: iPad/iPhone 实测
3. **层级 profiling**: 确定最优驻留策略
