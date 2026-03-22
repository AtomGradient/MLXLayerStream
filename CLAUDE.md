# CLAUDE.md

## 战略目标（北极星）

> **通过 Layer-Streaming Offloading，让超出设备内存的大模型在边缘设备上可用推理**

## 协作规则

### 思维原则

- **北极星高于一切**：任何能让更大模型在更小设备上跑的工作都在范围内
- **先深入思考，再写代码**：实现之前必须画清数据流 — 权重从哪里来、何时加载、何时释放
- **如果现有接口不满足需求，扩展接口**：不要将就用一个错误的路径

### 执行纪律

- **实测驱动**：纸上分析不算数，必须在真实设备上跑通才算完成
- **每次改代码必须跑回归测试**：确认无回归再提交
- **出了 bug 先加测试再修**
- **每个阶段结束写 Progress_timestamp.md**：保留进度快照
- **不硬编码本地路径**

### 沟通方式

- **用户给方向和决策，Claude 执行和实现**
- **用户实测发现问题后，Claude 分析根因而非打补丁**
- **不回避问题**：如果架构有根本性缺陷，直接说清楚
- **不问"要不要做"**：如果明确是应该做的事，直接开始

### BenchMark

- **任何优化必须有 BenchMark 数据支撑**

### 环境

- env: `source ~/Documents/mlx-community/3-11-mlx-community-env/bin/activate`
- 模型目录: `/Users/alex/Documents/mlx-community`
- MLX 安装: `cd mlx && pip install -e ".[dev]"` + `cd mlx-lm && pip install -e .`

### 硬件

| 设备 | 芯片 | GPU Cores | 内存 | 带宽 | NVMe |
|------|------|-----------|------|------|------|
| Mac Studio | M2 Ultra | 76 | 192 GB | ~800 GB/s | ~7.4 GB/s |
| iPad Air M3 | M3 | 10 | 8 GB | ~100 GB/s | ~2 GB/s |
| iPhone 15 Pro Max | A17 Pro | 6 | 8 GB | ~50 GB/s | ~2 GB/s |
