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
"""Multi-scale deformable attention, ported from ``rfdetr.models.ops``.

The PyTorch original evaluates its sampling grid with ``F.grid_sample`` (bilinear, ``padding_mode=
"zeros"``, ``align_corners=False``). :func:`bilinear_grid_sample` reimplements exactly that
convention with gather arithmetic so the two agree to floating-point noise.

The sampling itself runs channels-last on a flattened ``[N, H * W, C]`` feature map and pulls all
four bilinear corners in a single gather. Gathering whole rows along a major axis is the layout an
accelerator can vectorize; the equivalent NCHW form indexes the minormost axis once per corner, and
its transpose in the backward pass is a scatter-add over single elements. The public NCHW entry
points below keep their signatures and transpose into that layout.
"""

from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp

from rfdetr_jax.models.layers import Linear

Array = jax.Array

# The matmul sampler materialises an [N, L, H*W] interpolation kernel. Its memory
# cost grows with batch size, sample count and feature-map area. The auto policy
# uses feature-map area to select a sampler; choose "gather" explicitly when the
# interpolation kernel would consume too much memory. The threshold is a tuning
# heuristic, not a guarantee that matmul is faster on every device or shape.
_MATMUL_SAMPLER_MAX_LEVEL = 4096



def precise_divide(numerator: Array, denominator: Array) -> Array:
    """Divide elementwise with a single IEEE rounding, as PyTorch does.

    XLA can rewrite ``x / broadcast(y)`` into ``x * broadcast(1 / y)``, which rounds twice.
    Those rounding differences can propagate through the decoder's sine embeddings. Wherever
    upstream divides by a broadcast quantity on that path the divisor is materialised behind an
    optimization barrier and the division is performed as written. Operands promote as ``/`` does.
    The guarantee does not survive ``jax.vmap`` over the numerator alone: the batching rule
    re-broadcasts an unbatched divisor after the barrier. Batch the divisor as well, or avoid vmap.
    """
    numerator = jnp.asarray(numerator)
    dtype = jnp.result_type(numerator, denominator)
    numerator = numerator.astype(dtype)
    denominator = jnp.asarray(denominator).astype(dtype)
    shape = jnp.broadcast_shapes(numerator.shape, denominator.shape)
    divisor = jax.lax.optimization_barrier(jnp.broadcast_to(denominator, shape))
    return numerator / divisor

def _sample_rows(values: Array, grid: Array, height: int, width: int, padding_mode: str = "zeros") -> Array:
    """Bilinearly sample a flattened feature map at ``grid``, channels last.

    Args:
        values: Feature rows ``[N, H * W, C]``.
        grid: Sampling positions ``[N, L, 2]`` holding ``(x, y)`` in ``[-1, 1]``.
        height: Source height ``H``.
        width: Source width ``W``.

    Returns:
        Sampled rows ``[N, L, C]``; out-of-bounds samples contribute zero.
    """
    num, length = grid.shape[0], grid.shape[1]

    # Sampling positions are computed in at least float32 even when the activations are narrower: a
    # bf16 coordinate quantises the sub-pixel offset the interpolation is built from.
    grid = grid.astype(jnp.promote_types(grid.dtype, jnp.float32))
    # Written as torch's grid_sampler_unnormalize with align_corners=False, ((coord + 1) * size - 1) / 2;
    # the division by two is exact, so this equals (coord + 1) * size / 2 - 0.5 bit for bit.
    ix = ((grid[..., 0] + 1) * width - 1) / 2
    iy = ((grid[..., 1] + 1) * height - 1) / 2
    ix0 = jnp.floor(ix)
    iy0 = jnp.floor(iy)

    wx1 = (ix - ix0).astype(values.dtype)[..., None]
    wy1 = (iy - iy0).astype(values.dtype)[..., None]
    wx0 = 1 - wx1
    wy0 = 1 - wy1

    ix0i = ix0.astype(jnp.int32)
    iy0i = iy0.astype(jnp.int32)
    ix1i = ix0i + 1
    iy1i = iy0i + 1

    in_x0 = (ix0i >= 0) & (ix0i < width)
    in_x1 = (ix1i >= 0) & (ix1i < width)
    in_y0 = (iy0i >= 0) & (iy0i < height)
    in_y1 = (iy1i >= 0) & (iy1i < height)

    ix0c = jnp.clip(ix0i, 0, width - 1)
    iy0c = jnp.clip(iy0i, 0, height - 1)
    ix1c = jnp.clip(ix1i, 0, width - 1)
    iy1c = jnp.clip(iy1i, 0, height - 1)

    # One gather for all four corners: four separate ones cost four passes over the feature map and
    # four scatter-adds in the backward pass.
    corners = jnp.concatenate(
        [
            iy0c * width + ix0c,
            iy0c * width + ix1c,
            iy1c * width + ix0c,
            iy1c * width + ix1c,
        ],
        axis=1,
    )
    # The indices are clipped above, so the bounds check XLA would otherwise emit is dead code.
    taken = jnp.take_along_axis(values, corners[:, :, None], axis=1, mode="promise_in_bounds")
    v00, v10, v01, v11 = jnp.split(taken.reshape(num, 4, length, values.shape[-1]), 4, axis=1)

    weight = jnp.stack([wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1], axis=1)
    if padding_mode == "zeros":
        inside = jnp.stack(
            [in_x0 & in_y0, in_x1 & in_y0, in_x0 & in_y1, in_x1 & in_y1], axis=1
        ).astype(values.dtype)[..., None]
        weight = weight * inside
    elif padding_mode != "border":
        raise ValueError(f"padding_mode must be 'zeros' or 'border', got {padding_mode!r}")
    # With border padding the clipped corner indices already replicate the edge pixel, so the
    # unmasked weights (which sum to one) reproduce ``grid_sample(padding_mode="border")``.
    stacked = jnp.concatenate([v00, v10, v01, v11], axis=1)
    return (stacked * weight).sum(1)


def _sample_rows_f32(values: Array, grid: Array, height: int, width: int) -> Array:
    """:func:`_sample_rows` on float32 features, cast back to the input dtype.

    The gather's backward pass scatter-adds sample gradients into the feature map. Promoting
    the features makes that accumulation use float32, which can change CUDA atomic lowering
    at the cost of wider feature reads. Sampler performance depends on the backend and shape.
    """
    return _sample_rows(values.astype(jnp.float32), grid, height, width).astype(values.dtype)


def _sample_rows_matmul(values: Array, grid: Array, height: int, width: int) -> Array:
    """Bilinearly sample by contracting a separable interpolation kernel with the feature map.

    Same result as :func:`_sample_rows`, expressed as a dense matmul instead of a gather. The
    kernel is one row per sample with four non-zeros, built from two axis kernels; positions off
    the edge simply match no column, which reproduces ``padding_mode="zeros"`` without any masking.
    It trades ``L * H * W`` of arithmetic for the indexed memory traffic a gather costs, and its
    backward pass is another matmul rather than a scatter-add.

    Args:
        values: Feature rows ``[N, H * W, C]``.
        grid: Sampling positions ``[N, L, 2]`` holding ``(x, y)`` in ``[-1, 1]``.
        height: Source height ``H``.
        width: Source width ``W``.

    Returns:
        Sampled rows ``[N, L, C]``.
    """
    grid = grid.astype(jnp.promote_types(grid.dtype, jnp.float32))
    # Written as torch's grid_sampler_unnormalize with align_corners=False, ((coord + 1) * size - 1) / 2;
    # the division by two is exact, so this equals (coord + 1) * size / 2 - 0.5 bit for bit.
    ix = ((grid[..., 0] + 1) * width - 1) / 2
    iy = ((grid[..., 1] + 1) * height - 1) / 2
    ix0 = jnp.floor(ix)
    iy0 = jnp.floor(iy)
    wx1 = (ix - ix0)[..., None]
    wy1 = (iy - iy0)[..., None]

    columns = jnp.arange(width, dtype=ix.dtype)
    rows = jnp.arange(height, dtype=iy.dtype)
    kx = jnp.where(columns == ix0[..., None], 1 - wx1, 0.0) + jnp.where(columns == ix0[..., None] + 1, wx1, 0.0)
    ky = jnp.where(rows == iy0[..., None], 1 - wy1, 0.0) + jnp.where(rows == iy0[..., None] + 1, wy1, 0.0)

    # The axis kernels are formed in at least float32 so the position comparisons are exact, then narrowed
    # before the outer product: the `[N, L, H*W]` result is the tensor that costs memory, and in
    # bf16 it costs half as much and doubles the batch this sampler fits.
    kernel = (ky.astype(values.dtype)[..., :, None] * kx.astype(values.dtype)[..., None, :]).reshape(
        *grid.shape[:2], height * width
    )
    return jnp.einsum("nlj,njd->nld", kernel, values)


def bilinear_grid_sample(inputs: Array, grid: Array, padding_mode: str = "zeros") -> Array:
    """Sample ``inputs`` at ``grid`` with ``F.grid_sample(..., align_corners=False)`` semantics.

    Args:
        inputs: Feature map ``[N, C, H, W]``.
        grid: Sampling grid ``[N, Hg, Wg, 2]`` holding ``(x, y)`` in ``[-1, 1]``.
        padding_mode: ``"zeros"`` (out-of-bounds samples contribute zero, as deformable attention
            needs) or ``"border"`` (edge pixels are replicated, as PointRend point sampling needs).

    Returns:
        Sampled values ``[N, C, Hg, Wg]``.
    """
    n, c, h, w = inputs.shape
    gh, gw = grid.shape[1], grid.shape[2]
    rows = inputs.reshape(n, c, h * w).swapaxes(1, 2)
    sampled = _sample_rows(rows, grid.reshape(n, gh * gw, 2), h, w, padding_mode)
    return sampled.reshape(n, gh, gw, c).transpose(0, 3, 1, 2)


def ms_deform_attn_core_rows(
    value: Array,
    spatial_shapes: tuple[tuple[int, int], ...],
    sampling_locations: Array,
    attention_weights: Array,
    sampler: str = "gather",
) -> Array:
    """Sample and weight multi-scale features, channels last.

    Args:
        value: Projected features ``[B * n_heads, sum(H*W), head_dim]``.
        spatial_shapes: Per-level ``(H, W)`` pairs, as Python ints.
        sampling_locations: Sample positions ``[B, Lq, n_heads, n_levels, n_points, 2]`` in ``[0, 1]``.
        attention_weights: Softmaxed weights ``[B, Lq, n_heads, n_levels * n_points]``.
        sampler: ``"gather"`` for the indexed read, ``"gather-f32"`` for the same read on float32
            features (see :func:`_sample_rows_f32`), ``"matmul"`` for the dense kernel contraction,
            ``"auto"`` to pick per level from :data:`_MATMUL_SAMPLER_MAX_LEVEL`, or ``"pallas"`` for
            the fused TPU kernel in :mod:`rfdetr_jax.models.deform_pallas`, which builds the dense
            kernel tile by tile in VMEM instead of in HBM.

    Returns:
        Attention output ``[B, Lq, n_heads * head_dim]``.

    Raises:
        ValueError: If ``sampler`` is not one of ``"gather"``, ``"gather-f32"``, ``"matmul"``, ``"auto"``
            or ``"pallas"``.
    """
    if sampler not in ("gather", "gather-f32", "matmul", "auto", "pallas"):
        raise ValueError(
            f"unknown sampler {sampler!r}; expected 'gather', 'gather-f32', 'matmul', 'auto' or 'pallas'"
        )
    batch, len_query, n_heads = sampling_locations.shape[:3]
    num_points = sampling_locations.shape[4]
    head_dim = value.shape[-1]

    if sampler == "pallas":
        from rfdetr_jax.models.deform_pallas import sample_weighted

        n_levels = len(spatial_shapes)
        weights = attention_weights.reshape(batch, len_query, n_heads, n_levels, num_points)
        total = None
        start = 0
        for level, (height, width) in enumerate(spatial_shapes):
            locations = jnp.transpose(sampling_locations[:, :, :, level], (0, 2, 1, 3, 4)).reshape(
                batch * n_heads, len_query, num_points, 2
            )
            level_weights = jnp.transpose(weights[:, :, :, level], (0, 2, 1, 3)).reshape(
                batch * n_heads, len_query, num_points
            )
            part = sample_weighted(value[:, start : start + height * width, :], locations, level_weights, height, width)
            total = part if total is None else total + part
            start += height * width
        output = total.astype(jnp.result_type(value.dtype, attention_weights.dtype))
        return output.reshape(batch, n_heads, len_query, head_dim).transpose(0, 2, 1, 3).reshape(
            batch, len_query, n_heads * head_dim
        )

    offsets = [0]
    for height, width in spatial_shapes:
        offsets.append(offsets[-1] + height * width)

    # At least float32 for the sampling positions; a float64 reference run stays float64.
    grids = 2 * sampling_locations.astype(jnp.promote_types(sampling_locations.dtype, jnp.float32)) - 1
    sampled = []
    for level, (height, width) in enumerate(spatial_shapes):
        value_l = value[:, offsets[level] : offsets[level + 1], :]
        grid_l = jnp.transpose(grids[:, :, :, level], (0, 2, 1, 3, 4)).reshape(
            batch * n_heads, len_query * num_points, 2
        )
        chosen = sampler
        if chosen == "auto":
            chosen = "matmul" if height * width <= _MATMUL_SAMPLER_MAX_LEVEL else "gather"
        sample = {"gather": _sample_rows, "gather-f32": _sample_rows_f32}.get(chosen, _sample_rows_matmul)
        sampled.append(sample(value_l, grid_l, height, width).reshape(
            batch * n_heads, len_query, num_points, head_dim
        ))

    values = sampled[0] if len(sampled) == 1 else jnp.concatenate(sampled, axis=2)
    weights = jnp.transpose(attention_weights, (0, 2, 1, 3)).reshape(
        batch * n_heads, len_query, values.shape[2]
    )
    output = jnp.einsum("nqp,nqpd->nqd", weights, values)
    return output.reshape(batch, n_heads, len_query, head_dim).transpose(0, 2, 1, 3).reshape(
        batch, len_query, n_heads * head_dim
    )


def ms_deform_attn_core(
    value: Array,
    spatial_shapes: tuple[tuple[int, int], ...],
    sampling_locations: Array,
    attention_weights: Array,
) -> Array:
    """Sample and weight multi-scale features.

    Args:
        value: Projected features ``[B, n_heads, head_dim, sum(H*W)]``.
        spatial_shapes: Per-level ``(H, W)`` pairs, as Python ints.
        sampling_locations: Sample positions ``[B, Lq, n_heads, n_levels, n_points, 2]`` in ``[0, 1]``.
        attention_weights: Softmaxed weights ``[B, Lq, n_heads, n_levels * n_points]``.

    Returns:
        Attention output ``[B, Lq, n_heads * head_dim]``.
    """
    batch, n_heads, head_dim, length = value.shape
    rows = value.transpose(0, 1, 3, 2).reshape(batch * n_heads, length, head_dim)
    return ms_deform_attn_core_rows(rows, spatial_shapes, sampling_locations, attention_weights)


def sampling_offsets_bias_init(n_heads: int, n_levels: int, n_points: int) -> Array:
    """Return the grid-shaped bias initializer used by ``MSDeformAttn._reset_parameters``."""
    thetas = jnp.arange(n_heads, dtype=jnp.float32) * (2.0 * math.pi / n_heads)
    grid_init = jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=-1)
    grid_init = grid_init / jnp.abs(grid_init).max(-1, keepdims=True)
    grid_init = jnp.broadcast_to(grid_init.reshape(n_heads, 1, 1, 2), (n_heads, n_levels, n_points, 2))
    scale = jnp.arange(1, n_points + 1, dtype=jnp.float32).reshape(1, 1, n_points, 1)
    return (grid_init * scale).reshape(-1)


class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention, matching ``rfdetr.models.ops.modules.MSDeformAttn``."""

    d_model: int = 256
    n_levels: int = 4
    n_heads: int = 8
    n_points: int = 4
    param_dtype: jnp.dtype = jnp.float32
    sampler: str = "gather"

    @nn.compact
    def __call__(
        self,
        query: Array,
        reference_points: Array,
        input_flatten: Array,
        spatial_shapes: tuple[tuple[int, int], ...],
        input_padding_mask: Array | None = None,
    ) -> Array:
        """Attend from ``query`` to multi-scale features around ``reference_points``.

        Args:
            query: Query features ``[B, Lq, C]``.
            reference_points: ``[B, Lq, n_levels, 2]`` centres or ``[B, Lq, n_levels, 4]`` boxes in ``[0, 1]``.
            input_flatten: Flattened multi-scale features ``[B, sum(H*W), C]``.
            spatial_shapes: Per-level ``(H, W)`` pairs, as Python ints.
            input_padding_mask: Optional ``[B, sum(H*W)]`` mask, ``True`` on padded positions.

        Returns:
            Output features ``[B, Lq, C]``.

        Raises:
            ValueError: If ``reference_points`` has a trailing dimension other than 2 or 4.
        """
        batch, len_query, _ = query.shape
        len_input = input_flatten.shape[1]

        sampling_offsets_layer = Linear(
            self.d_model,
            self.n_heads * self.n_levels * self.n_points * 2,
            param_dtype=self.param_dtype,
            name="sampling_offsets",
        )
        attention_weights_layer = Linear(
            self.d_model, self.n_heads * self.n_levels * self.n_points, param_dtype=self.param_dtype,
            name="attention_weights",
        )
        value_proj = Linear(self.d_model, self.d_model, param_dtype=self.param_dtype, name="value_proj")
        output_proj = Linear(self.d_model, self.d_model, param_dtype=self.param_dtype, name="output_proj")

        value = value_proj(input_flatten)
        if input_padding_mask is not None:
            value = jnp.where(input_padding_mask[..., None], 0.0, value)

        attention_weights = attention_weights_layer(query).reshape(
            batch, len_query, self.n_heads, self.n_levels * self.n_points
        )
        sampling_offsets = sampling_offsets_layer(query).reshape(
            batch, len_query, self.n_heads, self.n_levels, self.n_points, 2
        )

        if reference_points.shape[-1] == 2:
            shapes = jnp.asarray(spatial_shapes, dtype=jnp.float32)
            offset_normalizer = jnp.stack([shapes[:, 1], shapes[:, 0]], axis=-1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + precise_divide(sampling_offsets, offset_normalizer[None, None, None, :, None, :])
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + precise_divide(sampling_offsets, self.n_points) * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]}.")

        softmax_dtype = jnp.promote_types(attention_weights.dtype, jnp.float32)
        attention_weights = jax.nn.softmax(attention_weights.astype(softmax_dtype), axis=-1).astype(query.dtype)
        head_dim = self.d_model // self.n_heads
        value = value.reshape(batch, len_input, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(batch * self.n_heads, len_input, head_dim)
        output = ms_deform_attn_core_rows(
            value, spatial_shapes, sampling_locations, attention_weights, self.sampler
        )
        return output_proj(output)
