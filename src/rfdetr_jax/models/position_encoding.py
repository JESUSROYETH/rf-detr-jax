# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# Source ancestry: LW-DETR / Conditional DETR / DETR / Deformable DETR.
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
"""Sine positional encoding, ported from ``rfdetr.models.position_encoding``."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from rfdetr_jax.models.ops import precise_divide

Array = jax.Array


def position_embedding_sine(
    mask: Array,
    num_pos_feats: int = 64,
    temperature: int = 10000,
    normalize: bool = True,
    scale: float | None = None,
) -> Array:
    """Build the DETR sine position embedding for a padding mask.

    Args:
        mask: Boolean padding mask ``[B, H, W]`` where ``True`` marks a padded pixel.
        num_pos_feats: Number of features per spatial axis; the output has ``2 * num_pos_feats`` channels.
        temperature: Frequency base.
        normalize: Whether to normalize the cumulative coordinates to ``[0, scale]``.
        scale: Normalization scale; defaults to ``2 * pi``.

    Returns:
        Position embedding ``[B, 2 * num_pos_feats, H, W]`` (the ``align_dim_orders=False`` layout
        used by ``Joiner``).
    """
    if scale is None:
        scale = 2 * math.pi
    not_mask = (~mask).astype(jnp.float32)
    y_embed = jnp.cumsum(not_mask, axis=1)
    x_embed = jnp.cumsum(not_mask, axis=2)
    if normalize:
        eps = 1e-6
        y_embed = precise_divide(y_embed, y_embed[:, -1:, :] + eps) * scale
        x_embed = precise_divide(x_embed, x_embed[:, :, -1:] + eps) * scale

    dim_t = jnp.arange(num_pos_feats, dtype=jnp.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = precise_divide(x_embed[:, :, :, None], dim_t)
    pos_y = precise_divide(y_embed[:, :, :, None], dim_t)
    pos_x = jnp.stack([jnp.sin(pos_x[..., 0::2]), jnp.cos(pos_x[..., 1::2])], axis=4).reshape(*pos_x.shape[:3], -1)
    pos_y = jnp.stack([jnp.sin(pos_y[..., 0::2]), jnp.cos(pos_y[..., 1::2])], axis=4).reshape(*pos_y.shape[:3], -1)
    pos = jnp.concatenate([pos_y, pos_x], axis=3)
    return jnp.transpose(pos, (0, 3, 1, 2))


def gen_sineembed_for_position(pos_tensor: Array, dim: int = 128) -> Array:
    """Build the decoder's reference-point sine embedding.

    Args:
        pos_tensor: Reference points ``[B, N, 2]`` or ``[B, N, 4]`` in ``[0, 1]``.
        dim: Number of features per coordinate.

    Returns:
        Embedding ``[B, N, dim * C]`` where ``C`` is the number of input coordinates.

    Raises:
        ValueError: If the trailing axis is neither 2 nor 4.
    """
    scale = 2 * math.pi
    dim_t = jnp.arange(dim, dtype=pos_tensor.dtype)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)

    def embed(coord: Array) -> Array:
        """Interleave sin/cos over the frequency axis for one coordinate."""
        value = precise_divide(coord[:, :, None] * scale, dim_t)
        return jnp.stack([jnp.sin(value[:, :, 0::2]), jnp.cos(value[:, :, 1::2])], axis=3).reshape(
            *value.shape[:2], -1
        )

    pos_x = embed(pos_tensor[:, :, 0])
    pos_y = embed(pos_tensor[:, :, 1])
    if pos_tensor.shape[-1] == 2:
        return jnp.concatenate([pos_y, pos_x], axis=2)
    if pos_tensor.shape[-1] == 4:
        pos_w = embed(pos_tensor[:, :, 2])
        pos_h = embed(pos_tensor[:, :, 3])
        return jnp.concatenate([pos_y, pos_x, pos_w, pos_h], axis=2)
    raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.shape[-1]}")
