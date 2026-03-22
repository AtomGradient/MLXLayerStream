#!/usr/bin/env python3
"""Layer-Streaming Offloading for MLX models.

Enables running models larger than device memory by loading/unloading
layer weights on demand from NVMe SSD during inference.

Core mechanism:
1. Model architecture loaded, but layer weights NOT resident
2. Before each layer forward: load weights from safetensors (lazy → eval)
3. After each layer forward: replace weights with placeholders + clear cache
4. embed_tokens, norm, lm_head always resident (small, needed at boundaries)
"""

import glob
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map


class LayerStreamer:
    """Manages layer-level weight streaming for memory-constrained inference."""

    def __init__(
        self,
        model: nn.Module,
        model_path: str,
        resident_layers: Optional[Set[int]] = None,
    ):
        self.model = model
        self.model_path = Path(model_path)
        self.resident_layers = resident_layers or set()

        # Get the actual layer list
        if hasattr(model, "language_model"):
            self._layers = model.language_model.model.layers
        elif hasattr(model, "model"):
            self._layers = model.model.layers
        else:
            self._layers = model.layers
        self.num_layers = len(self._layers)

        # Find weight files
        self.weight_files = sorted(
            glob.glob(str(self.model_path / "model*.safetensors"))
        )

        # Build index: file → {layer_idx → [keys]}
        self._build_index()

        # Stats
        self.stats = {
            "load_times_ms": [],
            "unload_times_ms": [],
            "loads_count": 0,
            "unloads_count": 0,
        }

    def _build_index(self):
        """Map each layer to its weight keys and source files."""
        # layer_idx → [(file_path, [keys_in_that_file])]
        self.layer_file_keys: Dict[int, List[Tuple[str, List[str]]]] = defaultdict(list)
        self.non_layer_file_keys: List[Tuple[str, List[str]]] = []

        for wf in self.weight_files:
            lazy_data = mx.load(wf)
            keys_by_layer: Dict[int, List[str]] = defaultdict(list)
            non_layer_keys: List[str] = []

            for k in lazy_data.keys():
                m = re.search(r"layers\.(\d+)\.", k)
                if m:
                    keys_by_layer[int(m.group(1))].append(k)
                else:
                    non_layer_keys.append(k)

            for layer_idx, keys in keys_by_layer.items():
                self.layer_file_keys[layer_idx].append((wf, keys))

            if non_layer_keys:
                self.non_layer_file_keys.append((wf, non_layer_keys))

    def load_layer(self, layer_idx: int):
        """Load a single layer's weights from disk into GPU memory."""
        if layer_idx in self.resident_layers:
            return

        t0 = time.perf_counter()

        raw_weights = {}
        for wf, keys in self.layer_file_keys[layer_idx]:
            file_data = mx.load(wf)
            for k in keys:
                raw_weights[k] = file_data[k]

        # Apply model sanitization (key remapping, conv1d transpose, etc.)
        if hasattr(self.model, "sanitize"):
            raw_weights = self.model.sanitize(raw_weights)

        self.model.load_weights(list(raw_weights.items()), strict=False)
        mx.eval(self._layers[layer_idx].parameters())

        t1 = time.perf_counter()
        self.stats["load_times_ms"].append((t1 - t0) * 1000)
        self.stats["loads_count"] += 1

    def unload_layer(self, layer_idx: int):
        """Release a layer's weights from GPU memory."""
        if layer_idx in self.resident_layers:
            return

        t0 = time.perf_counter()

        layer = self._layers[layer_idx]

        # Replace all parameter arrays with tiny (1,) placeholders
        def _placeholder(x):
            if isinstance(x, mx.array):
                return mx.zeros((1,), dtype=x.dtype)
            return x

        placeholders = tree_map(_placeholder, layer.parameters())
        layer.update(placeholders)
        mx.eval(layer.parameters())  # eval placeholders so old refs are freed
        mx.clear_cache()

        t1 = time.perf_counter()
        self.stats["unload_times_ms"].append((t1 - t0) * 1000)
        self.stats["unloads_count"] += 1

    def setup_streaming(self):
        """Initialize streaming: eval resident layers, unload the rest."""
        # Ensure non-layer params are resident (embed_tokens, norm, lm_head)
        for wf, keys in self.non_layer_file_keys:
            file_data = mx.load(wf)
            non_layer_weights = {k: file_data[k] for k in keys}
            if hasattr(self.model, "sanitize"):
                non_layer_weights = self.model.sanitize(non_layer_weights)
            self.model.load_weights(list(non_layer_weights.items()), strict=False)

        # Eval non-layer parameters
        if hasattr(self.model, "language_model"):
            base = self.model.language_model.model
        elif hasattr(self.model, "model"):
            base = self.model.model
        else:
            base = self.model

        mx.eval(base.embed_tokens.parameters())
        mx.eval(base.norm.parameters())
        if hasattr(self.model, "language_model"):
            if hasattr(self.model.language_model, "lm_head"):
                mx.eval(self.model.language_model.lm_head.parameters())
        elif hasattr(self.model, "lm_head"):
            mx.eval(self.model.lm_head.parameters())

        # Load resident layers, unload the rest
        for i in range(self.num_layers):
            if i in self.resident_layers:
                self.load_layer_direct(i)
            else:
                self.unload_layer(i)

        mx.clear_cache()

    def load_layer_direct(self, layer_idx: int):
        """Load layer weights without stats tracking (for resident layers)."""
        raw_weights = {}
        for wf, keys in self.layer_file_keys[layer_idx]:
            file_data = mx.load(wf)
            for k in keys:
                raw_weights[k] = file_data[k]

        if hasattr(self.model, "sanitize"):
            raw_weights = self.model.sanitize(raw_weights)

        self.model.load_weights(list(raw_weights.items()), strict=False)
        mx.eval(self._layers[layer_idx].parameters())

    def get_stats_summary(self) -> dict:
        """Return streaming performance statistics."""
        load_times = self.stats["load_times_ms"]
        unload_times = self.stats["unload_times_ms"]
        return {
            "avg_load_ms": sum(load_times) / len(load_times) if load_times else 0,
            "avg_unload_ms": sum(unload_times) / len(unload_times) if unload_times else 0,
            "total_loads": self.stats["loads_count"],
            "total_unloads": self.stats["unloads_count"],
            "total_load_time_s": sum(load_times) / 1000,
            "total_unload_time_s": sum(unload_times) / 1000,
        }

    def reset_stats(self):
        self.stats = {
            "load_times_ms": [],
            "unload_times_ms": [],
            "loads_count": 0,
            "unloads_count": 0,
        }

    # ── Prefetch API ──────────────────────────────────────────────

    def prefetch_layer(self, layer_idx: int):
        """Start preparing layer weights in a background thread.

        Does Python-level work (mx.load header + sanitize) in background.
        GPU eval is deferred to apply_prefetched() on main thread.
        """
        if layer_idx in self.resident_layers:
            return None

        def _do_prefetch():
            raw_weights = {}
            for wf, keys in self.layer_file_keys[layer_idx]:
                file_data = mx.load(wf)
                for k in keys:
                    raw_weights[k] = file_data[k]

            if hasattr(self.model, "sanitize"):
                raw_weights = self.model.sanitize(raw_weights)

            self._prefetched[layer_idx] = raw_weights

        if not hasattr(self, "_prefetched"):
            self._prefetched = {}
            self._prefetch_threads = {}

        t = threading.Thread(target=_do_prefetch)
        t.start()
        self._prefetch_threads[layer_idx] = t
        return t

    def apply_prefetched(self, layer_idx: int):
        """Apply previously prefetched weights to the model.

        Waits for the prefetch thread, then does GPU eval on main thread.
        """
        if layer_idx in self.resident_layers:
            return

        t0 = time.perf_counter()

        # Wait for prefetch to complete
        thread = self._prefetch_threads.pop(layer_idx, None)
        if thread is not None:
            thread.join()

        raw_weights = self._prefetched.pop(layer_idx, None)
        if raw_weights is None:
            # Fallback: synchronous load
            self.load_layer(layer_idx)
            return

        self.model.load_weights(list(raw_weights.items()), strict=False)
        mx.eval(self._layers[layer_idx].parameters())

        t1 = time.perf_counter()
        self.stats["load_times_ms"].append((t1 - t0) * 1000)
        self.stats["loads_count"] += 1


def streaming_forward(model, streamer, inputs, cache=None, input_embeddings=None):
    """Forward pass with layer-streaming: load/unload weights per layer.

    Replaces the model's normal forward to enable streaming.
    """
    # Get the inner text model
    if hasattr(model, "language_model"):
        text_model = model.language_model.model
        lm_head_fn = lambda out: model.language_model.lm_head(out)
        if model.language_model.args.tie_word_embeddings:
            lm_head_fn = lambda out: text_model.embed_tokens.as_linear(out)
    elif hasattr(model, "model"):
        text_model = model.model
        if hasattr(model, "lm_head"):
            lm_head_fn = lambda out: model.lm_head(out)
        else:
            lm_head_fn = lambda out: text_model.embed_tokens.as_linear(out)
    else:
        text_model = model
        lm_head_fn = lambda out: model.lm_head(out)

    # Embedding
    if input_embeddings is not None:
        hidden_states = input_embeddings
    else:
        hidden_states = text_model.embed_tokens(inputs)

    if cache is None:
        cache = [None] * len(text_model.layers)

    # Import mask creation
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    fa_mask = create_attention_mask(hidden_states, cache[text_model.fa_idx])
    ssm_mask = create_ssm_mask(hidden_states, cache[text_model.ssm_idx])

    # Layer-by-layer streaming forward
    for i, (layer, c) in enumerate(zip(text_model.layers, cache)):
        # Load layer weights from disk
        streamer.load_layer(i)

        # Forward
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden_states = layer(hidden_states, mask=mask, cache=c)

        # Force eval of outputs before unloading weights
        mx.eval(hidden_states)
        if c is not None:
            if hasattr(c, "state"):
                mx.eval(c.state)

        # Unload layer weights
        streamer.unload_layer(i)

    # Final norm + lm_head
    hidden_states = text_model.norm(hidden_states)
    logits = lm_head_fn(hidden_states)

    return logits


def streaming_forward_prefetch(model, streamer, inputs, cache=None, input_embeddings=None):
    """Forward pass with prefetching: load layer i+1 while computing layer i."""
    # Get the inner text model
    if hasattr(model, "language_model"):
        text_model = model.language_model.model
        lm_head_fn = lambda out: model.language_model.lm_head(out)
        if model.language_model.args.tie_word_embeddings:
            lm_head_fn = lambda out: text_model.embed_tokens.as_linear(out)
    elif hasattr(model, "model"):
        text_model = model.model
        if hasattr(model, "lm_head"):
            lm_head_fn = lambda out: model.lm_head(out)
        else:
            lm_head_fn = lambda out: text_model.embed_tokens.as_linear(out)
    else:
        text_model = model
        lm_head_fn = lambda out: model.lm_head(out)

    if input_embeddings is not None:
        hidden_states = input_embeddings
    else:
        hidden_states = text_model.embed_tokens(inputs)

    if cache is None:
        cache = [None] * len(text_model.layers)

    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    fa_mask = create_attention_mask(hidden_states, cache[text_model.fa_idx])
    ssm_mask = create_ssm_mask(hidden_states, cache[text_model.ssm_idx])

    num_layers = len(text_model.layers)

    # Start prefetch for layer 0
    streamer.prefetch_layer(0)

    for i, (layer, c) in enumerate(zip(text_model.layers, cache)):
        # Apply prefetched weights (waits if still loading)
        streamer.apply_prefetched(i)

        # Start prefetch for next layer (overlaps with current compute)
        if i + 1 < num_layers:
            streamer.prefetch_layer(i + 1)

        # Forward
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden_states = layer(hidden_states, mask=mask, cache=c)

        # Force eval before unloading
        mx.eval(hidden_states)
        if c is not None and hasattr(c, "state"):
            mx.eval(c.state)

        # Unload current layer
        streamer.unload_layer(i)

    hidden_states = text_model.norm(hidden_states)
    logits = lm_head_fn(hidden_states)
    return logits


def make_streaming_generate_step(model, streamer):
    """Create a generate_step compatible function that uses streaming forward."""

    def _streaming_model_call(inputs, cache=None, **kwargs):
        return streaming_forward(model, streamer, inputs, cache=cache, **kwargs)

    return _streaming_model_call


def compute_resident_layers(
    num_layers: int,
    memory_budget_gb: float,
    total_weight_gb: float,
    strategy: str = "first_n",
) -> Set[int]:
    """Determine which layers to keep resident given a memory budget.

    Args:
        num_layers: Total number of layers in the model.
        memory_budget_gb: Available memory for layer weights (excluding embed/norm/cache).
        total_weight_gb: Total weight size of all layers.
        strategy: Residency strategy - 'first_n', 'last_n', 'interleaved', 'none'.

    Returns:
        Set of layer indices to keep resident.
    """
    if strategy == "none":
        return set()

    per_layer_gb = total_weight_gb / num_layers
    max_resident = int(memory_budget_gb / per_layer_gb)
    max_resident = min(max_resident, num_layers)

    if max_resident <= 0:
        return set()

    if strategy == "first_n":
        return set(range(max_resident))
    elif strategy == "last_n":
        return set(range(num_layers - max_resident, num_layers))
    elif strategy == "interleaved":
        step = max(1, num_layers // max_resident)
        return set(range(0, num_layers, step))[:max_resident]
    else:
        return set()
