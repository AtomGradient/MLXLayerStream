#!/usr/bin/env python3
"""Layer-Level Profiling — Measure per-layer compute time and weight size.

Determines which layers are compute-heavy vs memory-heavy,
informing the optimal layer-residency strategy for streaming.

Usage:
    python profile_layers.py --model Qwen3.5-4B-4bit
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mlx-lm"))

import mlx.core as mx
import mlx.nn as nn

from mlx_lm import load

MODEL_DIR = "/Users/alex/Documents/mlx-community"


def count_params(module):
    """Count total parameters in a module."""
    total = 0
    for name, param in module.parameters().items():
        if isinstance(param, mx.array):
            total += param.size
        elif isinstance(param, dict):
            for v in param.values():
                if isinstance(v, mx.array):
                    total += v.size
    return total


def estimate_layer_bytes(layer):
    """Estimate memory footprint of a layer's weights."""
    total_bytes = 0
    for name, w in nn.utils.tree_flatten(layer.parameters()):
        if isinstance(w, mx.array):
            total_bytes += w.nbytes
    return total_bytes


def profile_model(model_path):
    model_name = os.path.basename(model_path)
    print(f"\n{'=' * 70}")
    print(f"LAYER PROFILE: {model_name}")
    print(f"{'=' * 70}")

    model, tokenizer = load(model_path)

    # Get layers
    if hasattr(model, "language_model"):
        layers = model.language_model.layers
        embed = model.language_model.embed_tokens
        norm = model.language_model.norm
    elif hasattr(model, "model"):
        layers = model.model.layers
        embed = model.model.embed_tokens
        norm = model.model.norm
    else:
        layers = model.layers
        embed = model.embed_tokens
        norm = model.norm

    print(f"Total layers: {len(layers)}")
    print()

    # Profile each layer
    print(f"{'Layer':>6} {'Type':<20} {'Params':>12} {'Size (MB)':>10} {'Fwd (ms)':>10}")
    print("-" * 62)

    # Create a dummy input
    dummy_input = mx.zeros((1, 1, layers[0].input_layernorm.weight.shape[0]))

    total_bytes = 0
    layer_info = []

    for i, layer in enumerate(layers):
        layer_bytes = estimate_layer_bytes(layer)
        total_bytes += layer_bytes

        # Detect layer type
        if hasattr(layer, "is_linear"):
            ltype = "GatedDeltaNet" if layer.is_linear else "SoftmaxAttn"
        else:
            ltype = "Standard"

        # Time forward pass
        cache_entry = None
        if hasattr(model, "make_cache"):
            cache = model.make_cache()
            cache_entry = cache[i]

        # Warm up
        try:
            if cache_entry is not None:
                layer(dummy_input, cache=cache_entry)
            else:
                layer(dummy_input)
            mx.eval(dummy_input)
        except Exception:
            pass

        # Measure
        mx.synchronize()
        t0 = time.perf_counter()
        n_iters = 10
        for _ in range(n_iters):
            try:
                if cache_entry is not None:
                    layer(dummy_input, cache=cache_entry)
                else:
                    layer(dummy_input)
                mx.synchronize()
            except Exception:
                break
        t1 = time.perf_counter()
        fwd_ms = (t1 - t0) / n_iters * 1000

        info = {
            "layer_idx": i,
            "type": ltype,
            "size_mb": layer_bytes / (1024**2),
            "fwd_ms": fwd_ms,
        }
        layer_info.append(info)

        print(f"{i:>6} {ltype:<20} {'-':>12} {layer_bytes / (1024**2):>9.1f} {fwd_ms:>9.2f}")

    print("-" * 62)
    print(f"{'Total':>6} {'':<20} {'-':>12} {total_bytes / (1024**2):>9.1f}")

    # Analysis
    print(f"\n--- Streaming Analysis ---")
    avg_size = total_bytes / len(layers) / (1024**2)
    print(f"Avg layer size: {avg_size:.1f} MB")
    print(f"Total weight size: {total_bytes / (1024**3):.2f} GB")

    ssd_speed_gbps = 2.0  # iPhone NVMe ~2 GB/s
    avg_load_ms = avg_size / (ssd_speed_gbps * 1024) * 1000
    print(f"Avg layer load time (@ {ssd_speed_gbps} GB/s): {avg_load_ms:.1f} ms")

    avg_fwd = sum(l["fwd_ms"] for l in layer_info) / len(layer_info)
    print(f"Avg layer forward time: {avg_fwd:.2f} ms")

    if avg_load_ms > avg_fwd:
        print(f"=> STREAMING BOTTLENECK: load ({avg_load_ms:.1f}ms) > compute ({avg_fwd:.2f}ms)")
        print(f"   Streaming decode TPS upper bound: {1000 / (len(layers) * avg_load_ms):.1f}")
    else:
        print(f"=> COMPUTE BOTTLENECK: compute ({avg_fwd:.2f}ms) > load ({avg_load_ms:.1f}ms)")
        print(f"   Streaming overhead is manageable")

    del model
    mx.clear_cache()

    return {"model": model_name, "layers": layer_info, "total_bytes": total_bytes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, args.model)
    if not os.path.isdir(model_path):
        print(f"ERROR: {model_path} not found")
        sys.exit(1)

    profile_model(model_path)


if __name__ == "__main__":
    main()
