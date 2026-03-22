#!/usr/bin/env python3
"""Benchmark layer-streaming with various residency strategies.

Tests: full-resident baseline, full-streaming, hybrid with N resident layers.
Reports TPS, memory, and streaming overhead for each configuration.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mlx-lm"))

import mlx.core as mx
from mlx_lm import load

from layer_streaming import (
    LayerStreamer,
    compute_resident_layers,
    streaming_forward,
)

MODEL_DIR = "/Users/alex/Documents/mlx-community"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

PROMPT = (
    "Write a detailed explanation of how neural networks work, "
    "covering backpropagation, gradient descent, and activation functions."
)
MAX_TOKENS = 50


def measure_baseline(model_path, tokenizer, model, max_tokens):
    """Measure full-resident baseline TPS."""
    from mlx_lm.generate import stream_generate

    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()

    last = None
    for r in stream_generate(model, tokenizer, PROMPT, max_tokens=max_tokens):
        last = r

    mx.synchronize()
    return {
        "mode": "full_resident",
        "resident_layers": "all",
        "gen_tps": last.generation_tps,
        "prompt_tps": last.prompt_tps,
        "peak_mem_gb": mx.get_peak_memory() / (1024**3),
        "streaming_overhead_pct": 0.0,
    }


def measure_streaming(model_path, tokenizer, model, max_tokens,
                       resident_layers=None, mode_name="streaming"):
    """Measure streaming TPS with optional resident layers."""
    streamer = LayerStreamer(model, model_path, resident_layers=resident_layers)
    streamer.setup_streaming()

    tokens = tokenizer.encode(PROMPT)
    input_ids = mx.array([tokens])

    # Prefill
    mx.reset_peak_memory()
    cache = model.make_cache()
    logits = streaming_forward(model, streamer, input_ids, cache=cache)
    mx.eval(logits)
    next_token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(next_token)

    # Decode
    streamer.reset_stats()
    mx.synchronize()
    t_start = time.perf_counter()

    for step in range(max_tokens):
        logits = streaming_forward(model, streamer, next_token.reshape(1, 1), cache=cache)
        mx.eval(logits)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)

    mx.synchronize()
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    tps = max_tokens / elapsed
    peak_mem = mx.get_peak_memory() / (1024**3)

    stats = streamer.get_stats_summary()
    io_time = stats["total_load_time_s"] + stats["total_unload_time_s"]

    num_resident = len(resident_layers) if resident_layers else 0
    num_layers = streamer.num_layers

    result = {
        "mode": mode_name,
        "resident_layers": num_resident,
        "resident_pct": num_resident / num_layers * 100,
        "streamed_layers": num_layers - num_resident,
        "gen_tps": tps,
        "peak_mem_gb": peak_mem,
        "streaming_overhead_pct": io_time / elapsed * 100,
        "avg_load_ms": stats["avg_load_ms"],
        "avg_unload_ms": stats["avg_unload_ms"],
        "total_io_s": io_time,
        "total_decode_s": elapsed,
    }

    del streamer
    mx.clear_cache()
    return result


def run_benchmark(model_name, max_tokens=MAX_TOKENS):
    """Run full benchmark suite for a model."""
    model_path = os.path.join(MODEL_DIR, model_name)
    print(f"\n{'='*80}")
    print(f"STREAMING BENCHMARK: {model_name}")
    print(f"{'='*80}")

    results = {"model": model_name, "max_tokens": max_tokens, "configs": []}

    # 1. Full-resident baseline
    print("\n--- Full-Resident Baseline ---")
    model, tokenizer = load(model_path)
    num_layers = len(model.layers)
    baseline = measure_baseline(model_path, tokenizer, model, max_tokens)
    print(f"  TPS={baseline['gen_tps']:.1f}, Mem={baseline['peak_mem_gb']:.2f} GB")
    results["configs"].append(baseline)
    results["num_layers"] = num_layers
    del model
    mx.clear_cache()

    # Get total weight size for budget calculations
    total_weight_gb = baseline["peak_mem_gb"] * 0.9  # approximate

    # 2. Full streaming (0% resident)
    print("\n--- Full Streaming (0% resident) ---")
    model, tokenizer = load(model_path, lazy=True)
    res = measure_streaming(model_path, tokenizer, model, max_tokens,
                           resident_layers=set(), mode_name="stream_0pct")
    print(f"  TPS={res['gen_tps']:.1f}, Mem={res['peak_mem_gb']:.2f} GB, "
          f"I/O={res['streaming_overhead_pct']:.0f}%")
    results["configs"].append(res)
    del model
    mx.clear_cache()

    # 3. Hybrid: 25%, 50%, 75% resident (first_n strategy)
    for pct in [25, 50, 75]:
        n_resident = int(num_layers * pct / 100)
        resident_set = set(range(n_resident))

        print(f"\n--- Hybrid: {pct}% resident ({n_resident}/{num_layers} layers) ---")
        model, tokenizer = load(model_path, lazy=True)
        res = measure_streaming(model_path, tokenizer, model, max_tokens,
                               resident_layers=resident_set,
                               mode_name=f"hybrid_{pct}pct_first_n")
        print(f"  TPS={res['gen_tps']:.1f}, Mem={res['peak_mem_gb']:.2f} GB, "
              f"I/O={res['streaming_overhead_pct']:.0f}%")
        results["configs"].append(res)
        del model
        mx.clear_cache()

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*80}")
    print(f"{'Mode':<30} {'TPS':>8} {'Mem(GB)':>8} {'I/O%':>6} {'Resident':>10}")
    print("-" * 65)
    for c in results["configs"]:
        print(f"{c['mode']:<30} {c['gen_tps']:>8.1f} {c['peak_mem_gb']:>8.2f} "
              f"{c.get('streaming_overhead_pct', 0):>5.0f}% "
              f"{str(c['resident_layers']):>10}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", help="Model name(s)")
    parser.add_argument("--tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--output", default=os.path.join(RESULTS_DIR, "streaming_benchmark.json"))
    args = parser.parse_args()

    if not args.model:
        # Default: test key models from the plan
        args.model = [
            "Qwen3.5-0.8B-8bit",   # Tier 1 baseline
            "Qwen3.5-4B-4bit",     # Tier 1 key baseline
            "Qwen3.5-9B-8bit",     # Tier 3 main target
            "Qwen3.5-27B-4bit",    # Tier 4 hero
        ]

    all_results = []
    for model_name in args.model:
        result = run_benchmark(model_name, max_tokens=args.tokens)
        all_results.append(result)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [saved to {args.output}]")


if __name__ == "__main__":
    main()
