#!/usr/bin/env python3
"""MLX Layer-Streaming Benchmark — Decode TPS + Memory Profiling

Measures decode performance and memory behavior for models that
may exceed physical memory, testing MLX's mmap-based loading.

Usage:
    python benchmark_stream.py --model Qwen3.5-4B-4bit
    python benchmark_stream.py --model Qwen2.5-14B-4bit --memory-limit 8
"""

import argparse
import json
import math
import os
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mlx-lm"))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm import load
from mlx_lm.generate import stream_generate

# ── Config ──────────────────────────────────────────────────────────

MODEL_DIR = "/Users/alex/Documents/mlx-community"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

NUM_PPL_SAMPLES = 10
SEQ_LEN = 512
PREFILL = 32

NUM_SPEED_RUNS = 3
MAX_GEN_TOKENS = 200
SPEED_PROMPT = (
    "Write a detailed explanation of how neural networks work, "
    "covering backpropagation, gradient descent, and activation functions."
)


# ── Data Loading ────────────────────────────────────────────────────

def load_eval_data(tokenizer, num_samples, seq_len, seed=42):
    from mlx_lm.tuner.datasets import load_dataset

    np.random.seed(seed)
    args = types.SimpleNamespace(
        hf_dataset={
            "path": "allenai/tulu-3-sft-mixture",
            "train_split": "train",
            "valid_split": "train[:1]",
        },
        train=True,
        test=False,
    )
    dataset = load_dataset(args, tokenizer)[0]
    perm = np.random.permutation(len(dataset)).tolist()
    data = []
    i = 0
    while len(data) < seq_len * num_samples and i < len(perm):
        tokens, _ = dataset.process(dataset[perm[i]])
        i += 1
        data.extend(tokens)
    data = mx.array(data[: (len(data) // seq_len) * seq_len]).reshape(-1, seq_len)
    return data[:num_samples]


# ── PPL Evaluation ──────────────────────────────────────────────────

def eval_ppl(model, data, prefill_len=32):
    total_loss = 0.0
    total_tokens = 0
    for si in range(data.shape[0]):
        tokens = data[si]
        cache = None
        if hasattr(model, "make_cache"):
            cache = model.make_cache()
        else:
            num_layers = len(model.layers)
            from mlx_lm.models.cache import KVCache
            cache = [KVCache() for _ in range(num_layers)]

        model(tokens[:prefill_len][None], cache=cache)
        mx.eval([c.state for c in cache if not c.empty()])

        for t in range(prefill_len, tokens.shape[0] - 1):
            logits = model(tokens[t : t + 1][None], cache=cache)
            loss = nn.losses.cross_entropy(
                logits[:, -1, :].astype(mx.float32),
                tokens[t + 1 : t + 2],
                reduction="sum",
            )
            mx.eval(loss)
            total_loss += loss.item()
            total_tokens += 1
        mx.clear_cache()
    return math.exp(total_loss / total_tokens)


# ── Speed Evaluation ────────────────────────────────────────────────

def eval_speed(model, tokenizer, prompt, max_tokens):
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()

    last = None
    for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        last = r

    mx.synchronize()
    return {
        "gen_tps": last.generation_tps,
        "prompt_tps": last.prompt_tps,
        "peak_mem_gb": mx.get_peak_memory() / (1024**3),
    }


# ── Memory Stats ────────────────────────────────────────────────────

def get_memory_stats():
    """Get current memory usage stats (macOS specific)."""
    import subprocess

    try:
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split("\n")
        stats = {}
        for line in lines[1:]:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip().rstrip(".")
                try:
                    stats[key] = int(val)
                except ValueError:
                    pass
        return stats
    except Exception:
        return {}


# ── Main ────────────────────────────────────────────────────────────

def run_benchmark(model_path):
    model_name = os.path.basename(model_path)
    print(f"\n{'=' * 80}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 80}")

    # Memory stats before loading
    mem_before = get_memory_stats()
    page_faults_before = mem_before.get("Pageins", 0)

    model, tokenizer = load(model_path)

    # Count layers
    if hasattr(model, "language_model"):
        layers = model.language_model.layers
    elif hasattr(model, "model"):
        layers = model.model.layers
    else:
        layers = model.layers
    num_layers = len(layers)
    print(f"Layers: {num_layers}")

    # Memory stats after loading
    mem_after = get_memory_stats()
    page_faults_after = mem_after.get("Pageins", 0)
    print(f"Page-ins during load: {page_faults_after - page_faults_before}")

    # ── PPL ──
    print(f"\n--- PPL ({NUM_PPL_SAMPLES} samples x {SEQ_LEN} tokens) ---")
    data = load_eval_data(tokenizer, NUM_PPL_SAMPLES, SEQ_LEN)
    ppl = eval_ppl(model, data, PREFILL)
    print(f"  PPL = {ppl:.3f}")

    # ── Speed ──
    print(f"\n--- Speed ({NUM_SPEED_RUNS} runs x {MAX_GEN_TOKENS} tokens) ---")
    runs = []
    for r in range(NUM_SPEED_RUNS):
        mem_pre_gen = get_memory_stats()
        pf_pre = mem_pre_gen.get("Pageins", 0)

        result = eval_speed(model, tokenizer, SPEED_PROMPT, MAX_GEN_TOKENS)

        mem_post_gen = get_memory_stats()
        pf_post = mem_post_gen.get("Pageins", 0)
        result["page_faults"] = pf_post - pf_pre

        runs.append(result)
        print(
            f"  Run {r+1}: TPS={result['gen_tps']:.1f}, "
            f"Mem={result['peak_mem_gb']:.2f} GB, "
            f"PageFaults={result['page_faults']}"
        )

    avg_tps = sum(r["gen_tps"] for r in runs) / len(runs)
    avg_mem = sum(r["peak_mem_gb"] for r in runs) / len(runs)
    avg_pf = sum(r["page_faults"] for r in runs) / len(runs)

    print(f"\n  Avg: TPS={avg_tps:.1f}, Mem={avg_mem:.2f} GB, PageFaults={avg_pf:.0f}")

    del model
    mx.clear_cache()

    return {
        "model": model_name,
        "num_layers": num_layers,
        "ppl": ppl,
        "speed": {
            "avg_tps": avg_tps,
            "avg_mem": avg_mem,
            "avg_page_faults": avg_pf,
            "runs": runs,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="MLX Layer-Streaming Benchmark")
    parser.add_argument(
        "--model",
        action="append",
        help="Model name(s) to benchmark",
    )
    parser.add_argument(
        "--model-dir",
        default=MODEL_DIR,
        help=f"Base directory for models (default: {MODEL_DIR})",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(RESULTS_DIR, "stream_results.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    if not args.model:
        print("ERROR: specify at least one --model")
        sys.exit(1)

    model_paths = [os.path.join(args.model_dir, m) for m in args.model]

    for p in model_paths:
        if not os.path.isdir(p):
            print(f"ERROR: Model not found: {p}")
            sys.exit(1)

    all_results = []
    for model_path in model_paths:
        result = run_benchmark(model_path)
        all_results.append(result)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [saved to {args.output}]")

    print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
    main()
