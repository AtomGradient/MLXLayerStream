# Layer-Streaming Offloading for Edge LLM Inference

> **Work in Progress** — This research is ongoing. Results and conclusions may be updated as additional optimizations are implemented and validated.

*By [AtomGradient](https://github.com/AtomGradient)*

Running 9B+ language models on 8GB Apple Silicon edge devices by streaming layer weights from NVMe storage during inference.

## Key Results

| Metric | Value |
|--------|-------|
| Memory reduction | 63–81% on real devices |
| 9B-6bit on 8GB iPad | **0.3 TPS at 1.78 GB** (was OOM) |
| iPad/iPhone TPS ratio | 1.90–1.93x baseline (matches 2x bandwidth ratio) |
| 27B-4bit peak memory | 14.09 GB → 1.70 GB (88% reduction, Mac Studio) |

### Device Streaming Results (Measured)

| Model | iPad M3 Baseline | iPad M3 Stream | iPhone Baseline | iPhone Stream | Mem Savings |
|-------|:---:|:---:|:---:|:---:|:---:|
| 0.8B-8bit | 103.2 TPS / 799 MB | 5.8 TPS / 293 MB | 53.5 TPS / 792 MB | 4.1 TPS / 293 MB | 63% |
| 4B-4bit | 35.5 TPS / 2330 MB | 1.9 TPS / 436 MB | 18.4 TPS / 2317 MB | 0.7 TPS / 436 MB | 81% |
| **9B-6bit** | **OOM** | **0.3 TPS / 1780 MB** | — | **0.3 TPS / 1780 MB** | — |
| **9B-6bit hybrid** | **OOM** | **0.4 TPS / 3629 MB** | — | **0.4 TPS / 3629 MB** | +43% |

### Device Baselines (Fully Resident)

| Model | iPad Air M3 | iPhone 15 Pro Max | Ratio |
|-------|------------|-------------------|-------|
| 0.8B-8bit (0.80 GB) | 103.2 TPS | 53.5 TPS | 1.93x |
| 2B-8bit (1.93 GB) | 43.8 TPS | 23.0 TPS | 1.90x |
| 4B-4bit (2.34 GB) | 35.5 TPS | 18.4 TPS | 1.93x |
| 4B-6bit (3.32 GB) | 25.2 TPS | 13.2 TPS | 1.91x |
| **9B-6bit (6.90 GB)** | **OOM** | — | — |

### Streaming TPS (Mac Studio, Qwen3.5-9B-8bit)

| Resident Layers | TPS | Memory | I/O Overhead |
|----------------|-----|--------|-------------|
| 32/32 (100%) | 68.3 | 8.97 GB | 0% |
| 24/32 (75%) | 5.9 | 7.44 GB | 70% |
| 16/32 (50%) | 3.2 | 5.73 GB | 77% |
| 0/32 (0%) | 1.7 | 2.31 GB | 81% |

## Publication

- **Paper**: [Download PDF](https://atomgradient.github.io/MLXLayerStream/paper.pdf)
- **Website**: [https://atomgradient.github.io/MLXLayerStream/](https://atomgradient.github.io/MLXLayerStream/)

## Repository Structure

```
MLXLayerStream/
├── layer_streaming.py       # Core streaming implementation (Python/MLX)
├── benchmark_stream.py      # Baseline benchmark script
├── benchmark_streaming.py   # Streaming benchmark with residency levels
├── profile_layers.py        # Per-layer profiling
├── split_weights.py         # Pre-split weights by layer for device streaming
├── test_streaming.py        # Correctness and performance tests
├── run_device_benchmark.sh  # Device automation (iPad/iPhone)
├── StreamBenchmarkApp/      # iOS benchmark app
├── results/                 # All experiment results (JSON)
├── docs/                    # Paper (LaTeX/PDF) + website
├── mlx/                     # MLX core (submodule)
├── mlx-lm/                  # mlx-lm (submodule, modified)
└── Progress_*.md            # Phase progress snapshots
```

## How It Works

1. **Pre-split** model weights into per-layer safetensors files
2. **Load** only non-layer weights (embeddings, norm, LM head) — always resident
3. **For each decode step**, iterate layers:
   - Load layer weights from NVMe → GPU memory
   - Execute layer forward pass
   - Replace weights with placeholders → free GPU memory
4. Peak memory = non-layer weights + 1 layer ≈ 10–15% of full model

## Hardware

| Device | Chip | Memory | Bandwidth | NVMe |
|--------|------|--------|-----------|------|
| Mac Studio | M2 Ultra | 192 GB | ~800 GB/s | ~7.4 GB/s |
| iPad Air M3 | M3 | 8 GB | ~100 GB/s | ~2 GB/s |
| iPhone 15 Pro Max | A17 Pro | 8 GB | ~50 GB/s | ~2 GB/s |

## Citation

```bibtex
@misc{atomgradient2026layerstream,
  title={Layer-Streaming Offloading: Running 9B+ Language Models on 8GB Edge Devices with MLX},
  author={AtomGradient},
  year={2026},
  url={https://github.com/AtomGradient/MLXLayerStream}
}
```
