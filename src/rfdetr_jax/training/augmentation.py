# Modified JAX adaptation of RF-DETR training resize contracts.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
"""Upstream's deterministic, optimizer-step-local multi-scale image resize."""
from __future__ import annotations

import random
from functools import partial

import jax
import jax.numpy as jnp


def compute_multi_scale_scales(resolution, expanded_scales=False, patch_size=16, num_windows=4):
    unit = patch_size * num_windows
    if resolution <= 0 or unit <= 0:
        raise ValueError("resolution, patch_size and num_windows must be positive")
    offsets = range(-5, 6) if expanded_scales else range(-3, 5)
    return [(resolution // unit + offset) * unit for offset in offsets if resolution // unit + offset >= 2]


def scale_for_step(scales, optimizer_step):
    """Share a scale across accumulation microbatches and restore it after resume."""
    return random.Random(int(optimizer_step)).choice(scales)


def resize_training_batch(batch, scale):
    """Resize pixels only; normalized geometry and target masks retain their frame.

    Upstream resizes its image padding mask as well. Our square batches contain no
    image padding, and target masks are resized by the criterion when sampled.
    """
    return {**batch, "pixel_values": _resize_pixels(batch["pixel_values"], scale)}


@partial(jax.jit, static_argnames=("scale",))
def _resize_pixels(pixels, scale):
    # ATen's non-antialiased bilinear path forms half-pixel coordinates in
    # float32 with fused multiply-add. Using the general separable resampler's
    # float64 host coefficients changes some coordinates by several ULPs.
    # Keep this operation fused on device, then accumulate in float32 as ATen does.
    def axis(length):
        source = jnp.maximum(jnp.float32(length / scale) * (jnp.arange(scale, dtype=jnp.float32) + .5) - .5, 0)
        lower = jnp.floor(source).astype(jnp.int32)
        upper = jnp.minimum(lower + 1, length - 1)
        return lower, upper, source - lower

    y0, y1, wy = axis(pixels.shape[-2])
    x0, x1, wx = axis(pixels.shape[-1])
    values = pixels.astype(jnp.float32)
    top = (1 - wx) * values[..., y0[:, None], x0] + wx * values[..., y0[:, None], x1]
    bottom = (1 - wx) * values[..., y1[:, None], x0] + wx * values[..., y1[:, None], x1]
    return ((1 - wy)[None, None, :, None] * top + wy[None, None, :, None] * bottom).astype(pixels.dtype)
