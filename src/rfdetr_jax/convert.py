# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""Map RF-DETR PyTorch checkpoints onto this port's Flax parameter tree.

Every parameter in :mod:`rfdetr_jax.models.layers` is stored under its PyTorch name and in the
PyTorch memory layout, so conversion needs no transposes at all. The only systematic difference is
that Flax module names cannot contain ``.``, so an integer path component such as
``decoder.layers.2`` is written ``decoder.layers_2``; :func:`torch_key_to_jax_path` performs that
rename and nothing else.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

# Lookahead keeps the trailing separator unconsumed so runs of consecutive indices
# (``stages_sampling.0.1.0``) all rename in a single pass.
_INDEX_PATTERN = re.compile(r"\.(\d+)(?=\.)")

#: Torch modules whose ``.weight`` becomes a bare Flax parameter array.
_BARE_PARAMS = ("refpoint_embed", "query_feat")

#: Torch buffers and bookkeeping entries with no Flax counterpart.
_SKIP_SUFFIXES = ("num_batches_tracked",)

#: Torch buffers that encode configuration rather than learned state; this port derives them
#: from the keypoint schema instead of storing them.
_SKIP_KEYS = frozenset({"_kp_active_mask"})
_SKIP_SUFFIX_NAMES = ("keypoint_class_mask",)


def torch_key_to_jax_path(key: str) -> tuple[str, ...] | None:
    """Convert a PyTorch ``state_dict`` key to the matching Flax parameter path.

    Args:
        key: Dotted PyTorch parameter name, e.g. ``backbone.0.encoder.encoder.encoder.layer.3.norm1.weight``.

    Returns:
        The Flax path as a tuple of names, or ``None`` when the entry has no counterpart
        (``num_batches_tracked`` and other bookkeeping buffers).
    """
    if key.endswith(_SKIP_SUFFIXES) or key.endswith(_SKIP_SUFFIX_NAMES) or key in _SKIP_KEYS:
        return None
    # With ``lite_refpoint_refine=False`` upstream aliases ``transformer.decoder.bbox_embed`` to the
    # shared ``bbox_embed`` module, so the state dict lists the same tensors twice.
    if key.startswith("transformer.decoder.bbox_embed."):
        return None

    # `backbone` is an nn.Sequential(Backbone, PositionEmbedding); the position embedding is
    # parameter-free, so only slot 0 survives, flattened to `backbone_0`.
    name = _INDEX_PATTERN.sub(r"_\1", key)

    # The upstream Backbone wraps a DinoV2 adapter that itself holds the HF backbone under
    # `.encoder`; this port drops that intermediate level.
    name = name.replace("backbone_0.encoder.encoder.", "backbone_0.encoder.")

    for bare in _BARE_PARAMS:
        if name == f"{bare}.weight":
            return (bare,)

    return tuple(name.split("."))


def flatten_params(params: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], np.ndarray]:
    """Flatten a nested Flax parameter dict into ``{path_tuple: array}``."""
    flat: dict[tuple[str, ...], np.ndarray] = {}
    for key, value in params.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            flat.update(flatten_params(value, path))
        else:
            flat[path] = value
    return flat


def unflatten_params(flat: dict[tuple[str, ...], Any]) -> dict:
    """Rebuild a nested Flax parameter dict from ``{path_tuple: array}``."""
    nested: dict = {}
    for path, value in flat.items():
        node = nested
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return nested


def convert_state_dict(
    state_dict: dict[str, np.ndarray],
    template: dict | None = None,
    strict: bool = True,
) -> tuple[dict, list[str], list[str]]:
    """Convert a numpy-valued torch ``state_dict`` into a Flax parameter tree.

    Args:
        state_dict: Torch parameter names mapped to numpy arrays.
        template: Optional initialised Flax parameter tree used to validate paths and shapes.
        strict: When ``True`` and a template is given, raise on any shape mismatch.

    Returns:
        A ``(params, missing, unexpected)`` triple. ``missing`` lists template paths that the
        checkpoint did not supply; ``unexpected`` lists checkpoint keys with no template slot.

    Raises:
        ValueError: If ``strict`` is set and a converted array's shape does not match the template.
    """
    template_flat = flatten_params(template) if template is not None else None
    converted: dict[tuple[str, ...], np.ndarray] = {}
    unexpected: list[str] = []

    for key, value in state_dict.items():
        path = torch_key_to_jax_path(key)
        if path is None:
            continue
        array = np.asarray(value)
        if template_flat is not None:
            if path not in template_flat:
                unexpected.append(key)
                continue
            expected = tuple(template_flat[path].shape)
            if tuple(array.shape) != expected:
                message = f"shape mismatch for {key} -> {'.'.join(path)}: {array.shape} vs {expected}"
                if strict:
                    raise ValueError(message)
                unexpected.append(f"{key} ({message})")
                continue
        converted[path] = array

    missing: list[str] = []
    if template_flat is not None:
        missing = [".".join(path) for path in template_flat if path not in converted]

    return unflatten_params(converted), missing, unexpected


def save_params_npz(path: str | Path, params: dict) -> None:
    """Write a Flax parameter tree to an ``.npz`` keyed by dotted path."""
    flat = {".".join(key): np.asarray(value) for key, value in flatten_params(params).items()}
    np.savez(path, **flat)


def load_params_npz(path: str | Path) -> dict:
    """Read a Flax parameter tree written by :func:`save_params_npz`."""
    with np.load(path) as data:
        flat = {tuple(key.split(".")): data[key] for key in data.files}
    return unflatten_params(flat)
