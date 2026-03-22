# MLX Layer-Streaming Offloading — Research Plan

## 北极星

> 通过按层流式加载模型权重（Layer-Streaming），让超出设备物理内存的大模型（7B、14B）在 8GB 边缘设备（iPhone、iPad）上实现可用推理。

---

## 1. 动机与背景

### 1.1 来自 H2OAttnScore 的关键洞见

在 H2OAttnScore V2 项目中，我们建立了以下已验证的事实：

1. **模型规模收益 >> weight 精度收益**
   - PPL-内存效率前沿：4B-6bit → 4B-bf16 花 4.62 GB 只换 0.2% PPL（0.04%/GB）
   - 对比：0.8B-8bit → 2B-6bit 花 0.69 GB 换 12.7% PPL（18.4%/GB）
   - **核心原则：在有限内存下，优先选更大模型 + 更激进量化**

2. **移动端推理是 bandwidth-bound**
   - iPad/iPhone TPS 比 (~2×) 完美匹配带宽比 (~100/~50 GB/s)
   - 自回归 decode 每步需要加载全部模型权重 → 带宽是瓶颈

3. **KV cache 已经通过 H2O 解决**
   - H2O + 8bit KV 在混合架构上 <1% PPL，KV cache 不再是内存瓶颈
   - 瓶颈转移到**模型权重本身**

### 1.2 核心假设

如果模型权重不需要全部驻留 GPU 内存，而是从 NVMe SSD 按需流式加载：

- **iPhone NVMe SSD**: >2 GB/s 读取
- **4B-4bit 单层权重**: ~80-100 MB（2.83 GB / 32 layers）
- **7B-4bit 单层权重**: ~130 MB
- **14B-4bit 单层权重**: ~250 MB
- **单层加载延迟**: 250 MB / 2 GB/s = ~125 ms

对于 **decode**（每步生成 1 个 token，遍历所有层一次）：
- 如果所有层都从 SSD 加载: 32 层 × 125 ms = 4 秒/token — 太慢
- 如果热门层驻留内存，冷门层按需加载: 可以大幅缩短

关键洞见：**不需要流式加载所有层**。策略是：
- 在 8 GB 内存中尽可能多地驻留层（比如 60-70%）
- 只有溢出的部分从 SSD 流式加载
- 利用 MLX 的 memory-mapped 机制和虚拟内存系统

### 1.3 MLX 已有的相关机制

MLX 在 Apple Silicon 上利用统一内存（UMA）架构：
- `mx.load()` 使用 memory-mapped I/O — 权重文件不需要全部复制到 RAM
- 操作系统的虚拟内存系统会自动管理页面的换入换出
- Apple Silicon 的 SSD 控制器和 UMA 设计让这个过程比传统 GPU offloading 更高效

**关键问题**：MLX 的 mmap 是否已经自动做了 layer-streaming？如果是，我们需要做的可能不是"实现" layer-streaming，而是"测量和优化"它的性能特征。

---

## 2. 研究问题

### Q1: MLX mmap 的基线性能

> 当模型权重超过物理内存时，MLX 的 mmap 机制下 decode TPS 是多少？随着模型/内存比增加，TPS 如何退化？

### Q2: 显式 Layer-Streaming 是否优于 mmap

> 显式按层加载/释放（手动 prefetch + evict）是否比操作系统的 mmap 页面管理更高效？

### Q3: 层驻留策略

> 如果只能驻留部分层，哪些层应该驻留？Attention 层 vs FFN 层？前几层 vs 后几层？是否可以用 profiling 数据决定？

### Q4: Prefill vs Decode 的不同策略

> Prefill（处理整个 prompt）需要顺序遍历所有层，是流式加载的最坏情况。Decode（逐 token）也需要遍历但每层计算量更小。两者是否需要不同的 offloading 策略？

### Q5: 与 H2O 的组合效果

> Layer-Streaming（控制权重内存）+ H2O（控制 KV 内存）组合时，总内存占用和 TPS 表现如何？

---

## 3. 实验设计

### 3.1 模型矩阵

| 模型 | 大小 | 适合设备 | 研究目的 |
|------|------|---------|---------|
| Qwen3.5-4B-4bit | 2.9 GB | 全部 | 基线（完全驻留内存） |
| Qwen3.5-4B-bf16 | 8.7 GB | Desktop only | 测试超 8GB 内存的 mmap 行为 |
| Qwen2.5-7B-4bit | ~4.5 GB | iPad/iPhone tight | 首个 streaming 候选 |
| Qwen2.5-14B-4bit | ~8.5 GB | Desktop only（设备需 streaming） | 主要研究目标 |
| Qwen2.5-14B-3bit | ~6 GB | iPad/iPhone（需部分 streaming） | 激进量化 + streaming |

注：需要先下载/量化这些模型到 `/Users/alex/Documents/mlx-community/`

### 3.2 评估指标

| 指标 | 方法 | 工具 |
|------|------|------|
| Decode TPS | stream_generate, 200 tokens | benchmark 脚本 |
| Prefill TPS | 512 token prompt 处理速度 | benchmark 脚本 |
| Peak Memory | mx.get_peak_memory() | 内置 |
| Resident Memory | 实际物理内存占用（非 mmap 映射） | `memory_pressure`, Activity Monitor |
| Page Faults | mmap 导致的页面错误次数 | `vm_stat`, Instruments |
| SSD Read | 实际从 SSD 读取的数据量 | `iostat` |
| PPL | 10 samples × 512 tokens | eval 脚本 |

### 3.3 测试策略

```
每个模型 × 4 个策略:
1. Full-Resident    — 权重全部加载到内存（基线）
2. MLX-mmap-Default — 使用 MLX 默认 mmap 加载，不做干预
3. Manual-Stream    — 显式按层 prefetch/evict
4. Hybrid-Resident  — 驻留 N 层，其余 mmap/stream
```

---

## 4. 执行计划

### Phase 0: 环境搭建 + 基线测量 (Day 1)

```bash
# 安装 mlx 和 mlx-lm
cd /Users/alex/Documents/Codes/MLXLayerStream
source ~/Documents/mlx-community/3-11-mlx-community-env/bin/activate
cd mlx && pip install -e ".[dev]" && cd ..
cd mlx-lm && pip install -e . && cd ..

# 下载/量化模型
# 需要: Qwen2.5-7B-4bit, Qwen2.5-14B-4bit, Qwen2.5-14B-3bit
```

**基线测量**：
1. 在 Mac Studio（192 GB）上跑 4B/7B/14B 的 decode TPS 和 PPL — 这是"完全驻留内存"的理论上限
2. 记录每个模型的 peak memory
3. 确认 PPL 正确（排除加载错误）

### Phase 1: MLX mmap 行为研究 (Day 2-3)

**核心实验**：在 Mac Studio 上限制可用内存（`memory_pressure` 或 cgroups），模拟 8 GB 设备环境。

```python
# 伪代码：测量 mmap 行为
import mlx.core as mx
from mlx_lm import load, generate

# 正常加载（MLX 默认使用 mmap）
model, tokenizer = load("path/to/14B-4bit")

# 测量：实际物理内存占用 vs 虚拟内存映射
# 测量：decode 过程中的页面错误率
# 测量：SSD 读取量 (iostat)
```

**关键问题要回答**：
- MLX 的 mmap 是否已经让 14B 在 8GB "能跑"？
- 如果能跑，TPS 是多少？
- 页面错误集中在哪些操作上？

### Phase 2: 显式 Layer-Streaming 实现 (Day 4-6)

如果 Phase 1 发现 mmap 不够高效，实现显式控制：

**方案 A: 修改 mlx-lm 的 Model.forward()**
```python
class StreamingModel(nn.Module):
    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        for i, layer in enumerate(self.layers):
            # 确保当前层权重在内存中
            self._ensure_resident(i)
            h = layer(h, mask=mask, cache=cache[i])
            # 释放不再需要的层
            self._maybe_evict(i)
        return self.lm_head(self.norm(h))

    def _ensure_resident(self, layer_idx):
        """Prefetch layer weights into GPU memory"""
        # 选项 1: mx.eval() 强制 materialize
        # 选项 2: 手动 load + 替换
        pass

    def _maybe_evict(self, layer_idx):
        """Release layer weights back to mmap"""
        # 选项 1: 将参数替换为 mmap-backed lazy array
        # 选项 2: 手动 del + gc
        pass
```

**方案 B: 修改 MLX core 的内存管理**
- 在 C++ 层面添加 memory budget API
- 让 MLX 感知内存限制并自动管理
- 更优雅但改动更大

### Phase 3: 层驻留策略研究 (Day 7-8)

用 profiling 确定最优驻留策略：
1. 测量每层的计算时间
2. 测量每层从 SSD 加载的时间
3. 找到 "加载时间 > 计算时间" 的层 — 这些是 streaming 的瓶颈
4. 尝试几种驻留策略：
   - First-N: 前 N 层驻留
   - Last-N: 后 N 层驻留
   - Interleaved: 每隔 K 层驻留一层
   - Profiled: 基于 profiling 数据选择

### Phase 4: 设备端验证 (Day 9-10)

在真实 iPad/iPhone 上验证：
1. 用 iOS app（可从 H2OAttnScore 的 H2OBenchmarkApp 修改）
2. 测试 7B-4bit 在 8GB iPad 上的 decode TPS
3. 测试 14B-3bit/4bit 在 8GB iPad 上是否能跑
4. 测量实际电池消耗（SSD 读取耗电）

### Phase 5: H2O 组合实验 (Day 11)

将 H2O KV cache 与 Layer-Streaming 组合：
- Layer-Streaming 控制权重内存（模型大小不受物理内存限制）
- H2O 控制 KV cache 内存（序列长度不受物理内存限制）
- 两者叠加 = 在 8GB 设备上运行 14B 模型 + 长上下文

### Phase 6: 论文 + 网站 (Day 12-14)

结构类似 H2OAttnScore V2：
- LaTeX 论文 (docs/paper.tex)
- 双语网站 (docs/index.html)
- 数据审计
- 发布

---

## 5. 预期结果

### 乐观预期
- 14B-4bit 在 iPad 8GB 上以 3-5 TPS decode（足够用于对话）
- Layer-Streaming 的 TPS 损失相对 full-resident < 50%
- 与 H2O 组合后，14B 模型能处理 1K+ token 上下文

### 保守预期
- 7B-4bit 在 iPad 上可用（5-10 TPS）
- 14B 需要更激进的量化（3bit/2bit）才能达到可用 TPS
- SSD 读取的功耗可能是个问题

### 最坏情况
- MLX 的 mmap 已经足够好，显式 streaming 没有明显改善
- 这其实是好消息 — 意味着 MLX 已经解决了这个问题，论文变成"验证 MLX 的 mmap 在边缘设备上的表现"

---

## 6. 从 H2OAttnScore 学到的经验

### 实验流程
1. **先跑基线**：在无限资源（Mac Studio）上确认模型正确性
2. **再跑受限环境**：模拟或在真实设备上测试
3. **设备冷却**：每个模型之间 60-120 秒冷却
4. **增量保存**：每跑完一个模型就保存结果 JSON
5. **跳过 build**：用 `SKIP_BUILD=1` 避免重复编译

### 数据管理
1. 所有结果存 JSON（`results/` 目录）
2. 论文/网站的每个数字必须可追溯到 JSON
3. 数据审计是必须步骤
4. iPhone 带宽用 ~50 GB/s（实测值），不用公开规格

### 设备测试
1. `run_device_benchmark.sh` 脚本已验证可用
2. 设备端 benchmark app 通过 Xcode 部署
3. 超时设为 900 秒（大模型需要更长时间）
4. 结果通过 `devicectl copy from` 拉取

---

## 7. 目录结构

```
MLXLayerStream/
├── CLAUDE.md              # 项目规则
├── PLAN.md                # 本文件
├── mlx/                   # MLX core（可能需要修改内存管理）
├── mlx-lm/                # mlx-lm（修改 model loading + forward）
├── mlx-swift/             # mlx-swift（设备端）
├── mlx-swift-lm/          # mlx-swift-examples（设备 app）
├── benchmark_stream.py    # 桌面端 benchmark 脚本
├── profile_layers.py      # 层级 profiling 脚本
├── run_device_benchmark.sh # 设备自动化（从 H2OAttnScore 复制）
├── results/               # 所有实验结果 JSON
├── docs/
│   ├── paper.tex          # 论文
│   ├── paper.pdf
│   └── index.html         # 双语网站
└── Progress_*.md          # 进度快照
```

---

## 8. 恢复上下文检查清单

新会话开始时，读取以下文件：
1. `PLAN.md` — 本文件
2. `CLAUDE.md` — 项目规则
3. 最新的 `Progress_*.md` — 进度快照
4. `results/` 目录下的最新 JSON — 实验数据
