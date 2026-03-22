# Progress — Phase 3: 层驻留策略研究

**日期**: 2026-03-22
**设备**: Mac Studio M2 Ultra (192 GB)

## 关键发现

### 1. 驻留策略对 TPS 无影响

对 9B-8bit 和 27B-4bit 在 50% 驻留比例下测试三种策略：

| 模型 | 策略 | TPS | I/O% |
|------|------|-----|------|
| 9B-8bit | first_n | 3.2 | 73% |
| 9B-8bit | last_n | 3.1 | 74% |
| 9B-8bit | interleaved | 3.1 | 76% |
| 27B-4bit | first_n | 1.6 | 73% |
| 27B-4bit | last_n | 1.6 | 73% |
| 27B-4bit | interleaved | 1.6 | 76% |

**原因**: Qwen3.5 的层完全均匀（~200-205 MB, ~0.06ms compute）。策略选择只影响**哪些**层被流式加载，不影响**多少**数据被加载。`first_n` 略优（可能因为文件系统顺序访问更优）。

### 2. 权重分布分析

27B-4bit 层内权重分布：
- **Attention（含 GatedDeltaNet）**: 30% (~60 MB/层)
- **MLP**: 70% (~143 MB/层)
- 子层流式（只流 MLP）收益有限：I/O 量几乎相同（7.44 GB vs 7.34 GB per token）

### 3. 模拟 8GB 设备最优配置

| 模型 | 权重总量 | 非层开销 | 驻留层 | 流式层 | Mac TPS | iPad 预估 |
|------|---------|---------|--------|--------|---------|-----------|
| 4B-bf16 | 8.5 GB | 1.18 GB | 27/32 (84%) | 5 | 9.0 | ~3.3 |
| **9B-6bit** | **6.9 GB** | **1.60 GB** | **32/32 (100%)** | **0** | **全速** | **~10** |
| 9B-8bit | 9.7 GB | 2.07 GB | 23/32 (72%) | 9 | 5.2 | ~1.8 |
| 27B-4bit | 14.3 GB | 1.39 GB | 28/64 (44%) | 36 | 1.4 | ~0.5 |

### 4. 8GB 设备推荐配置

**首选**: **Qwen3.5-9B-6bit** — 完全驻留 8GB 设备，无需流式，PPL=4.485
- vs 4B-4bit（全驻留）: PPL 4.485 vs 4.959 = **9.6% PPL 改善**
- vs 4B-bf16（84% 驻留）: 9B-6bit 更优（全速 + 更好 PPL）

**进阶**: **Qwen3.5-9B-8bit** — 72% 驻留, iPad ~1.8 TPS, PPL=4.455
- 比 9B-6bit 多 1.7GB 但 PPL 几乎无改善 (4.485 vs 4.455)
- 流式带来的 TPS 损失不值得 0.7% PPL 改善

**极限**: **Qwen3.5-27B-4bit** — 44% 驻留, iPad ~0.5 TPS, PPL=4.010
- 比 9B-6bit PPL 改善 10.6%，但 TPS 降到 ~0.5（非交互场景）

### 5. 核心公式

```
TPS = max_tokens / (num_streamed_layers × load_time_per_layer + total_compute_time)

where:
  load_time_per_layer ≈ per_layer_bytes / ssd_bandwidth
  total_compute_time ≈ total_weight_bytes / memory_bandwidth

For 27B-4bit on iPad:
  load_time = 204 MB / 2 GB/s = 102 ms/layer
  compute = 14.3 GB / 100 GB/s = 143 ms/pass = 2.2 ms/layer
  Full stream TPS = 1000 / (64 × 102 + 143) = 0.15
  44% resident TPS = 1000 / (36 × 102 + 143) = 0.27
```

## 结论

1. **9B-6bit 是 8GB 设备的甜蜜点** — 全驻留、全速、PPL 优于 4B
2. **层驻留策略无关紧要**（均匀架构），只有驻留数量重要
3. **I/O 是绝对瓶颈** — 计算/加载比为 1:1660
4. **27B 在 8GB 设备上可运行但缓慢** — 适合非交互式批量推理
5. **Sub-layer streaming 收益有限** — 在均匀架构上与全层策略差异 <2%

## 下一步

- Phase 4: iPad/iPhone 实测验证预估值
- Phase 5: H2O + Layer-Streaming 组合实验
