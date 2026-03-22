#!/usr/bin/env python3
"""Split model weights into per-layer safetensors files for device streaming.

Creates a directory structure optimized for layer-streaming on iOS:
  output_dir/
    non_layer.safetensors    # embed_tokens, norm, lm_head
    layer_0000.safetensors   # layer 0 weights
    layer_0001.safetensors   # layer 1 weights
    ...
    config.json              # copied from source
    tokenizer.json           # copied from source
    tokenizer_config.json    # copied from source
    streaming_info.json      # layer count, sizes, metadata

Usage:
    python split_weights.py --model Qwen3.5-9B-6bit
    python split_weights.py --model Qwen3.5-27B-4bit --output /tmp/split_27B
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mlx-lm"))

import mlx.core as mx

MODEL_DIR = "/Users/alex/Documents/mlx-community"


def split_model(model_path, output_dir):
    model_name = os.path.basename(model_path)
    print(f"Splitting: {model_name}")
    print(f"Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Load all weight files lazily
    weight_files = sorted(glob.glob(os.path.join(model_path, "model*.safetensors")))
    print(f"Weight files: {len(weight_files)}")

    all_weights = {}
    for wf in weight_files:
        all_weights.update(mx.load(wf))
    print(f"Total tensors: {len(all_weights)}")

    # Group by layer, exclude vision_tower weights
    layer_weights = defaultdict(dict)
    non_layer_weights = {}
    skipped = 0

    for k, v in all_weights.items():
        # Skip vision tower weights (multimodal models)
        if k.startswith("vision_tower") or k.startswith("model.visual"):
            skipped += 1
            continue
        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            layer_weights[int(m.group(1))][k] = v
        else:
            non_layer_weights[k] = v

    if skipped:
        print(f"Skipped {skipped} vision_tower tensors")

    num_layers = max(layer_weights.keys()) + 1 if layer_weights else 0
    print(f"Layers: {num_layers}")
    print(f"Non-layer tensors: {len(non_layer_weights)}")

    # Save non-layer weights
    print("Saving non_layer.safetensors...")
    non_layer_path = os.path.join(output_dir, "non_layer.safetensors")
    mx.save_safetensors(non_layer_path, non_layer_weights)
    non_layer_size = os.path.getsize(non_layer_path)
    print(f"  Size: {non_layer_size / (1024**2):.0f} MB")

    # Save per-layer weights
    layer_sizes = {}
    for i in range(num_layers):
        fname = f"layer_{i:04d}.safetensors"
        fpath = os.path.join(output_dir, fname)
        mx.save_safetensors(fpath, layer_weights[i])
        size = os.path.getsize(fpath)
        layer_sizes[i] = size
        if i % 8 == 0 or i == num_layers - 1:
            print(f"  Layer {i}: {size / (1024**2):.0f} MB ({len(layer_weights[i])} tensors)")

    # Copy config and tokenizer files (exclude index files that reference original weights)
    for pattern in ["config.json", "configuration.json", "tokenizer*.json",
                     "preprocessor_config.json", "processor_config.json",
                     "vocab.json", "merges.txt", "special_tokens_map.json",
                     "generation_config.json"]:
        for f in glob.glob(os.path.join(model_path, pattern)):
            dst = os.path.join(output_dir, os.path.basename(f))
            shutil.copy2(f, dst)

    # Write streaming metadata
    info = {
        "model_name": model_name,
        "num_layers": num_layers,
        "non_layer_size_bytes": non_layer_size,
        "layer_sizes_bytes": {str(i): s for i, s in layer_sizes.items()},
        "total_weight_bytes": non_layer_size + sum(layer_sizes.values()),
        "avg_layer_bytes": sum(layer_sizes.values()) // num_layers if num_layers else 0,
    }
    with open(os.path.join(output_dir, "streaming_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    total_mb = info["total_weight_bytes"] / (1024**2)
    avg_mb = info["avg_layer_bytes"] / (1024**2)
    print(f"\nDone! Total: {total_mb:.0f} MB, Avg layer: {avg_mb:.0f} MB")
    print(f"Files: {num_layers + 1} safetensors + configs")

    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, args.model)
    if not os.path.isdir(model_path):
        print(f"ERROR: {model_path} not found")
        sys.exit(1)

    output_dir = args.output or os.path.join(args.model_dir, f"{args.model}-streaming")
    split_model(model_path, output_dir)


if __name__ == "__main__":
    main()
