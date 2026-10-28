# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""Segmentation head and PointRend-style point sampling, ported from
``rfdetr.models.heads.segmentation``.

The upstream head wraps its depthwise convolution in a custom autograd ``Function`` purely to
disable cuDNN in the backward pass on some CUDA stacks; that workaround has no counterpart here, so
this port calls the convolution directly. Mixed-precision gradient differences are bounded
against the published implementation by the precision reference tests.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from rfdetr_jax.models.layers import Conv2d, LayerNorm, Linear
from rfdetr_jax.models.ops import bilinear_grid_sample
from rfdetr_jax.utilities.interpolate import resize_2d

Array = jax.Array


class DepthwiseConvBlock(nn.Module):
    """A simplified ConvNeXt block: depthwise conv, LayerNorm, pointwise linear, GELU, residual."""

    dim: int
    layer_scale_init_value: float = 0.0
    compute_dtype: str | None = None

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Apply the block to an NCHW feature map."""
        residual = x
        conv_input = x if self.compute_dtype is None else x.astype(self.compute_dtype)
        y = Conv2d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim, name="dwconv")(conv_input)
        y = jnp.transpose(y, (0, 2, 3, 1))
        y = LayerNorm(self.dim, eps=1e-6, name="norm")(y)
        y = Linear(self.dim, self.dim, name="pwconv1")(y)
        y = jax.nn.gelu(y, approximate=False)
        if self.layer_scale_init_value > 0:
            gamma = self.param(
                "gamma",
                lambda key, shape: jnp.full(shape, self.layer_scale_init_value, jnp.float32),
                (self.dim,),
            )
            y = gamma * y
        return jnp.transpose(y, (0, 3, 1, 2)) + residual


class MLPBlock(nn.Module):
    """Pre-norm two-layer GELU MLP with a residual connection, over ``[B, N, C]`` query features."""

    dim: int
    layer_scale_init_value: float = 0.0
    compute_dtype: str | None = None

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Apply the block to query features."""
        residual = x
        y = LayerNorm(self.dim, name="norm_in")(x)
        if self.compute_dtype is not None:
            y = y.astype(self.compute_dtype)
        y = Linear(self.dim, self.dim * 4, name="layers_0")(y)
        y = jax.nn.gelu(y, approximate=False)
        y = Linear(self.dim * 4, self.dim, name="layers_2")(y)
        if self.layer_scale_init_value > 0:
            gamma = self.param(
                "gamma",
                lambda key, shape: jnp.full(shape, self.layer_scale_init_value, jnp.float32),
                (self.dim,),
            )
            y = gamma * y
        return y + residual


class SegmentationHead(nn.Module):
    """Mask head: refine the finest feature map, then take a dot product with the query features."""

    in_dim: int
    num_blocks: int
    bottleneck_ratio: int = 1
    downsample_ratio: int = 4
    compute_dtype: str | None = None

    @nn.compact
    def __call__(
        self,
        spatial_features: Array,
        query_features: list[Array],
        image_size: tuple[int, int],
        skip_blocks: bool = False,
    ) -> list[Array]:
        """Return one mask-logit map per decoder layer.

        Args:
            spatial_features: The finest backbone feature map ``[B, C, H, W]``.
            query_features: One ``[B, N, C]`` query tensor per decoder layer.
            image_size: The input image ``(height, width)``; masks are produced at
                ``image_size // downsample_ratio``.
            skip_blocks: Run only the projection, used for the two-stage encoder output.

        Returns:
            A list of ``[B, N, Hm, Wm]`` mask logits.

        Raises:
            ValueError: If ``skip_blocks`` is set with more than one query tensor.
        """
        interaction_dim = self.in_dim // self.bottleneck_ratio
        target = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
        features = resize_2d(spatial_features, target, mode="bilinear", align_corners=False)

        spatial_proj = Conv2d(self.in_dim, interaction_dim, kernel_size=1, name="spatial_features_proj")
        query_block = MLPBlock(self.in_dim, compute_dtype=self.compute_dtype, name="query_features_block")
        query_proj = Linear(self.in_dim, interaction_dim, name="query_features_proj")
        bias = self.param("bias", nn.initializers.zeros, (1,))

        blocks = [DepthwiseConvBlock(self.in_dim, compute_dtype=self.compute_dtype, name=f"blocks_{index}")
                  for index in range(self.num_blocks)]

        def cast_activation(value):
            return value if self.compute_dtype is None else value.astype(self.compute_dtype)

        mask_logits = []
        if skip_blocks:
            if len(query_features) != 1:
                raise ValueError("skip_blocks is only supported for a single query-feature tensor")
            projected = spatial_proj(cast_activation(features))
            queries = query_proj(cast_activation(query_block(query_features[0])))
            mask_logits.append(jnp.einsum("bchw,bnc->bnhw", projected, queries) + bias)
        else:
            for block, queries in zip(blocks, query_features, strict=False):
                features = block(features)
                projected = spatial_proj(cast_activation(features))
                queries = query_proj(cast_activation(query_block(queries)))
                mask_logits.append(jnp.einsum("bchw,bnc->bnhw", projected, queries) + bias)
        return mask_logits


def point_sample(inputs: Array, point_coords: Array, mode: str = "bilinear") -> Array:
    """Sample ``inputs`` at ``[0, 1]``-normalized points.

    Args:
        inputs: Feature map ``[N, C, H, W]``.
        point_coords: Points ``[N, P, 2]`` or ``[N, Hg, Wg, 2]`` in ``[0, 1]``.
        mode: ``"bilinear"`` for mask logits, ``"nearest"`` for binary ground-truth masks —
            the two modes RF-DETR uses.

    Returns:
        Sampled features ``[N, C, P]`` or ``[N, C, Hg, Wg]``.

    Raises:
        ValueError: If a mode other than ``"bilinear"`` or ``"nearest"`` is requested.
    """
    if mode not in ("bilinear", "nearest"):
        raise ValueError(f"point_sample supports bilinear and nearest sampling, got {mode!r}")
    squeeze = point_coords.ndim == 3
    if squeeze:
        point_coords = point_coords[:, :, None, :]
    grid = 2.0 * point_coords - 1.0
    output = (_nearest_grid_sample(inputs, grid) if mode == "nearest"
              else bilinear_grid_sample(inputs, grid, padding_mode="border"))
    return output[..., 0] if squeeze else output


def _nearest_grid_sample(inputs: Array, grid: Array) -> Array:
    """Nearest-neighbour ``grid_sample`` with ``align_corners=False`` and border padding."""
    n, c, h, w = inputs.shape
    gh, gw = grid.shape[1], grid.shape[2]
    x = ((grid[..., 0] + 1) * w - 1) / 2
    y = ((grid[..., 1] + 1) * h - 1) / 2
    xi = jnp.clip(jnp.round(x).astype(jnp.int32), 0, w - 1)
    yi = jnp.clip(jnp.round(y).astype(jnp.int32), 0, h - 1)
    # Gather whole rows of a channels-last view, as in rfdetr_jax.models.ops.
    rows = inputs.reshape(n, c, h * w).swapaxes(1, 2)
    index = (yi * w + xi).reshape(n, gh * gw, 1)
    taken = jnp.take_along_axis(rows, index, axis=1, mode="promise_in_bounds")
    return taken.reshape(n, gh, gw, c).transpose(0, 3, 1, 2)


def calculate_uncertainty(logits: Array) -> Array:
    """Return ``-|logit|``: points nearest the decision boundary are the most uncertain."""
    return -jnp.abs(logits)


def get_uncertain_point_coords_with_randomness(
    coarse_logits: Array,
    num_points: int,
    key: Array | None,
    oversample_ratio: int = 3,
    importance_sample_ratio: float = 0.75,
    draws: tuple[Array, Array] | None = None,
) -> Array:
    """Sample point coordinates biased toward uncertain regions, as in PointRend.

    Args:
        coarse_logits: Mask logits ``[N, 1, H, W]``.
        num_points: Number of points to return per mask.
        key: PRNG key.
        oversample_ratio: How many candidate points to draw before selection.
        importance_sample_ratio: Fraction of returned points taken from the most uncertain candidates.
        draws: Optional ``(candidates [N, S, 2], randoms [N, R, 2])`` uniform draws replacing ``key``.

    Returns:
        Point coordinates ``[N, num_points, 2]`` in ``[0, 1]``.
    """
    num_masks = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    if draws is None:
        if key is None:
            raise ValueError("point sampling needs a PRNG key or recorded draws")
        key, candidate_key, random_key = jax.random.split(key, 3)
        candidates = jax.random.uniform(candidate_key, (num_masks, num_sampled, 2))
    else:
        candidates, randoms = draws
        if candidates.shape != (num_masks, num_sampled, 2):
            raise ValueError(f"recorded candidates must have shape {(num_masks, num_sampled, 2)}")
    uncertainties = calculate_uncertainty(point_sample(coarse_logits, candidates))

    num_uncertain = int(importance_sample_ratio * num_points)
    _, indices = jax.lax.top_k(uncertainties[:, 0, :], num_uncertain)
    selected = jnp.take_along_axis(candidates, indices[..., None], axis=1)

    num_random = num_points - num_uncertain
    if num_random > 0:
        extra = (jax.random.uniform(random_key, (num_masks, num_random, 2)) if draws is None
                 else randoms[:, :num_random])
        selected = jnp.concatenate([selected, extra], axis=1)
    return selected
