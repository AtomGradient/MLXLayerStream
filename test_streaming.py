#!/usr/bin/env python3
"""Test layer-streaming correctness and performance.

Verifies that streaming forward produces identical results to normal forward,
then measures streaming overhead.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mlx-lm"))

import mlx.core as mx
import mlx.nn as nn

from mlx_lm import load
from mlx_lm.generate import stream_generate

from layer_streaming import (
    LayerStreamer,
    streaming_forward,
)

MODEL_DIR = "/Users/alex/Documents/mlx-community"


def test_correctness(model_name="Qwen3.5-0.8B-8bit"):
    """Verify streaming forward produces identical logits to normal forward."""
    print(f"\n{'='*70}")
    print(f"CORRECTNESS TEST: {model_name}")
    print(f"{'='*70}")

    model_path = os.path.join(MODEL_DIR, model_name)
    model, tokenizer = load(model_path)

    # Create test input
    prompt = "Hello, how are you?"
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Normal forward
    cache_normal = model.make_cache()
    logits_normal = model(input_ids, cache=cache_normal)
    mx.eval(logits_normal)

    # Get reference output token
    ref_token = mx.argmax(logits_normal[:, -1, :], axis=-1)
    print(f"Normal forward - next token: {tokenizer.decode(ref_token.tolist())!r}")

    # Reset model for streaming test
    del cache_normal
    mx.clear_cache()

    # Streaming forward
    streamer = LayerStreamer(model, model_path)
    streamer.setup_streaming()

    cache_stream = model.make_cache()
    logits_stream = streaming_forward(model, streamer, input_ids, cache=cache_stream)
    mx.eval(logits_stream)

    stream_token = mx.argmax(logits_stream[:, -1, :], axis=-1)
    print(f"Streaming forward - next token: {tokenizer.decode(stream_token.tolist())!r}")

    # Compare
    match = mx.array_equal(ref_token, stream_token).item()
    max_diff = mx.max(mx.abs(logits_normal - logits_stream)).item()
    print(f"Tokens match: {match}")
    print(f"Max logit diff: {max_diff:.6e}")

    if match and max_diff < 1e-3:
        print("PASS: Streaming forward is correct!")
    else:
        print("FAIL: Streaming forward differs from normal!")

    # Print streaming stats
    stats = streamer.get_stats_summary()
    print(f"\nStreaming stats:")
    print(f"  Avg load time: {stats['avg_load_ms']:.1f} ms")
    print(f"  Avg unload time: {stats['avg_unload_ms']:.1f} ms")
    print(f"  Total loads: {stats['total_loads']}")
    print(f"  Total load time: {stats['total_load_time_s']:.2f} s")

    del model, streamer
    mx.clear_cache()
    return match


def test_memory_savings(model_name="Qwen3.5-0.8B-8bit"):
    """Measure memory usage: normal vs streaming."""
    print(f"\n{'='*70}")
    print(f"MEMORY TEST: {model_name}")
    print(f"{'='*70}")

    model_path = os.path.join(MODEL_DIR, model_name)

    # Normal mode
    mx.clear_cache()
    mx.reset_peak_memory()

    model, tokenizer = load(model_path)
    mem_normal = mx.get_peak_memory() / (1024**3)
    print(f"Normal load peak memory: {mem_normal:.2f} GB")

    del model
    mx.clear_cache()

    # Streaming mode
    mx.reset_peak_memory()

    model, tokenizer = load(model_path, lazy=True)
    streamer = LayerStreamer(model, model_path)
    streamer.setup_streaming()
    mem_streaming = mx.get_peak_memory() / (1024**3)
    print(f"Streaming setup peak memory: {mem_streaming:.2f} GB")

    # Run one forward pass
    prompt = "Hello"
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    mx.reset_peak_memory()
    cache = model.make_cache()
    logits = streaming_forward(model, streamer, input_ids, cache=cache)
    mx.eval(logits)
    mem_forward = mx.get_peak_memory() / (1024**3)
    print(f"Streaming forward peak memory: {mem_forward:.2f} GB")

    print(f"\nMemory savings: {mem_normal:.2f} GB → {max(mem_streaming, mem_forward):.2f} GB "
          f"({(1 - max(mem_streaming, mem_forward)/mem_normal)*100:.0f}% reduction)")

    del model, streamer
    mx.clear_cache()


def test_streaming_tps(model_name="Qwen3.5-0.8B-8bit", max_tokens=20):
    """Measure decode TPS with streaming."""
    print(f"\n{'='*70}")
    print(f"STREAMING TPS TEST: {model_name} ({max_tokens} tokens)")
    print(f"{'='*70}")

    model_path = os.path.join(MODEL_DIR, model_name)
    model, tokenizer = load(model_path, lazy=True)

    streamer = LayerStreamer(model, model_path)
    streamer.setup_streaming()

    prompt = "Explain neural networks in detail:"
    tokens = tokenizer.encode(prompt)

    # Prefill
    print("Prefill...")
    cache = model.make_cache()
    input_ids = mx.array([tokens])
    logits = streaming_forward(model, streamer, input_ids, cache=cache)
    mx.eval(logits)
    next_token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(next_token)

    # Decode
    print(f"Decoding {max_tokens} tokens...")
    streamer.reset_stats()
    mx.synchronize()
    t_start = time.perf_counter()

    generated = []
    for step in range(max_tokens):
        logits = streaming_forward(model, streamer, next_token.reshape(1, 1), cache=cache)
        mx.eval(logits)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)
        generated.append(next_token.item())

    mx.synchronize()
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    tps = max_tokens / elapsed
    print(f"Generated: {tokenizer.decode(generated)!r}")
    print(f"Time: {elapsed:.2f}s, TPS: {tps:.1f}")

    stats = streamer.get_stats_summary()
    print(f"\nStreaming overhead:")
    print(f"  Avg load: {stats['avg_load_ms']:.1f} ms/layer")
    print(f"  Avg unload: {stats['avg_unload_ms']:.1f} ms/layer")
    print(f"  Total I/O: {stats['total_load_time_s'] + stats['total_unload_time_s']:.2f}s "
          f"of {elapsed:.2f}s ({(stats['total_load_time_s'] + stats['total_unload_time_s'])/elapsed*100:.0f}%)")

    del model, streamer
    mx.clear_cache()
    return tps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3.5-0.8B-8bit")
    parser.add_argument("--test", choices=["correctness", "memory", "tps", "all"], default="all")
    parser.add_argument("--tokens", type=int, default=20)
    args = parser.parse_args()

    if args.test in ("correctness", "all"):
        test_correctness(args.model)
    if args.test in ("memory", "all"):
        test_memory_savings(args.model)
    if args.test in ("tps", "all"):
        test_streaming_tps(args.model, args.tokens)
