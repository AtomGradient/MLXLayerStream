---
name: mlx_no_mmap
description: MLX does NOT use mmap for weight loading - uses standard file I/O, all weights must be fully resident
type: project
---

MLX 不使用 mmap 加载权重，用的是标准 read()/lseek() 文件 I/O。
权重被完整加载到 GPU 内存，没有内建 lazy loading 机制。

**Why:** 这推翻了 PLAN.md 中 "MLX mmap 是否已经自动做了 layer-streaming" 的假设。
**How to apply:** Layer-Streaming 必须在应用层实现（显式加载/释放层权重），PLAN Phase 2 方案 A 是正确路径。Phase 1 的 "mmap 行为研究" 可以跳过，直接进入显式 streaming 实现。
