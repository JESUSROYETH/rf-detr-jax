# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""Deformable-attention sampling as a fused Pallas TPU kernel.

The ``"matmul"`` sampler in :mod:`rfdetr_jax.models.ops` contracts a dense bilinear kernel
``[N, L, H * W]`` with the feature map. The contraction is cheap, but XLA writes that kernel to HBM
and reads it back, then does the same with its gradient in the backward pass: at Nano geometry and
eight images per chip that is about 1.15 GB per decoder layer, and the sampler becomes a memory
stream rather than arithmetic. Here each tile of the kernel is built in VMEM from iotas and consumed
at once, so HBM only ever carries the feature map, the sample positions, the weights and the output.

Layout: samples run along the lane axis and feature positions along sublanes, so a tile of sample
coordinates broadcasts against a column of positions without any transpose. A sample off the map
matches no position and contributes zero, which is ``grid_sample``'s ``padding_mode="zeros"``.

The attention weight of each sample is folded in, so the kernel returns ``a * sample`` and the
caller only sums over points.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

Array = jax.Array

#: Largest number of samples per kernel tile along the lane axis.
MAX_TILE = 4096
#: VMEM the backward pass's three ``[H*W, tile]`` float32 kernel tiles may take together.
_TILE_BUDGET_BYTES = 32 * 1024 * 1024


def _tile_for(positions: int, samples: int) -> int:
    """Samples per tile: the largest power of two whose backward tiles fit the VMEM budget.

    Wider tiles mean fewer grid steps over the same feature map. A larger map gets a narrower
    tile so the three kernel tiles still fit, and a call with few samples gets one tile no wider
    than it needs.
    """
    tile = MAX_TILE
    while tile > 128 and 3 * positions * tile * 4 > _TILE_BUDGET_BYTES:
        tile //= 2
    needed = 128
    while needed < samples:
        needed *= 2
    return min(tile, needed)


def _axis_weights(position: Array, size: int):
    """Linear-interpolation weights along one axis and their derivative: ``[size, T]`` each."""
    index = jax.lax.broadcasted_iota(jnp.int32, (size, 1), 0).astype(jnp.float32)
    lower = jnp.floor(position)
    frac = position - lower
    at_lower = (index == lower).astype(jnp.float32)
    at_upper = (index == lower + 1).astype(jnp.float32)
    return at_lower * (1 - frac) + at_upper * frac, at_upper - at_lower


def _bilinear_tile(ix: Array, iy: Array, height: int, width: int):
    """Build the ``[H*W, T]`` bilinear kernel tile and its x/y derivatives for one tile of samples.

    The kernel is separable, ``K = k_y (x) k_x``, so only the two small axis kernels are computed
    elementwise and each ``[H*W, T]`` tile costs one broadcast multiply.

    Args:
        ix, iy: ``[1, T]`` sample positions in pixel units (``loc * size - 0.5``).

    Returns:
        ``(kernel, d_kernel_dx, d_kernel_dy)``, each ``[H*W, T]`` float32.
    """
    kx, dkx = _axis_weights(ix, width)
    ky, dky = _axis_weights(iy, height)
    tile = ix.shape[-1]

    def outer(along_y, along_x):
        return (along_y[:, None, :] * along_x[None, :, :]).reshape(height * width, tile)

    return outer(ky, kx), outer(ky, dkx), outer(dky, kx)


def _forward_kernel(vt_ref, ix_ref, iy_ref, a_ref, out_ref, *, height: int, width: int):
    """``out[:, tile] = a * (V^T @ K)`` for one batch-head and one tile of samples."""
    kernel, _, _ = _bilinear_tile(ix_ref[0], iy_ref[0], height, width)
    # Interpolation weights stay float32: in VMEM there is no memory reason to narrow them.
    vt = vt_ref[0].astype(jnp.float32)
    sampled = jnp.dot(vt, kernel, preferred_element_type=jnp.float32)
    out_ref[0] = sampled * a_ref[0]


def _backward_kernel(
    vt_ref, ix_ref, iy_ref, a_ref, g_ref, dvt_ref, dix_ref, diy_ref, da_ref, *, height: int, width: int
):
    """Gradients for one batch-head and tile; ``dV^T`` accumulates across the tile axis."""
    kernel, dkx, dky = _bilinear_tile(ix_ref[0], iy_ref[0], height, width)
    vt = vt_ref[0].astype(jnp.float32)
    g = g_ref[0]
    a = a_ref[0]
    sampled = jnp.dot(vt, kernel, preferred_element_type=jnp.float32)
    along_x = jnp.dot(vt, dkx, preferred_element_type=jnp.float32)
    along_y = jnp.dot(vt, dky, preferred_element_type=jnp.float32)
    da_ref[0] = jnp.sum(g * sampled, axis=0, keepdims=True)
    dix_ref[0] = a * jnp.sum(g * along_x, axis=0, keepdims=True)
    diy_ref[0] = a * jnp.sum(g * along_y, axis=0, keepdims=True)

    contribution = jax.lax.dot_general(
        g * a, kernel, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )

    @pl.when(pl.program_id(1) == 0)
    def _():
        dvt_ref[0] = jnp.zeros_like(dvt_ref[0])

    dvt_ref[0] += contribution


def _specs(n: int, dim: int, positions: int, samples: int, tile: int):
    """Block specs shared by the forward and backward calls."""
    grid = (n, samples // tile)
    per_head = pl.BlockSpec((1, dim, positions), lambda i, t: (i, 0, 0))
    per_tile = pl.BlockSpec((1, 1, tile), lambda i, t: (i, 0, t))
    per_tile_dim = pl.BlockSpec((1, dim, tile), lambda i, t: (i, 0, t))
    return grid, per_head, per_tile, per_tile_dim


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _sample(
    vt: Array, ix: Array, iy: Array, a: Array, height: int, width: int, interpret: bool, tile: int
) -> Array:
    return _forward(vt, ix, iy, a, height, width, interpret, tile)


def _forward(vt, ix, iy, a, height, width, interpret, tile):
    n, dim, positions = vt.shape
    samples = ix.shape[-1]
    grid, per_head, per_tile, per_tile_dim = _specs(n, dim, positions, samples, tile)
    return pl.pallas_call(
        functools.partial(_forward_kernel, height=height, width=width),
        grid=grid,
        in_specs=[per_head, per_tile, per_tile, per_tile],
        out_specs=per_tile_dim,
        out_shape=jax.ShapeDtypeStruct((n, dim, samples), jnp.float32),
        interpret=interpret,
    )(vt, ix, iy, a)


def _sample_fwd(vt, ix, iy, a, height, width, interpret, tile):
    return _forward(vt, ix, iy, a, height, width, interpret, tile), (vt, ix, iy, a)


def _sample_bwd(height, width, interpret, tile, residuals, g):
    vt, ix, iy, a = residuals
    n, dim, positions = vt.shape
    samples = ix.shape[-1]
    grid, per_head, per_tile, per_tile_dim = _specs(n, dim, positions, samples, tile)
    dvt, dix, diy, da = pl.pallas_call(
        functools.partial(_backward_kernel, height=height, width=width),
        grid=grid,
        in_specs=[per_head, per_tile, per_tile, per_tile, per_tile_dim],
        out_specs=[per_head, per_tile, per_tile, per_tile],
        out_shape=[
            jax.ShapeDtypeStruct((n, dim, positions), jnp.float32),
            jax.ShapeDtypeStruct(ix.shape, jnp.float32),
            jax.ShapeDtypeStruct(iy.shape, jnp.float32),
            jax.ShapeDtypeStruct(a.shape, jnp.float32),
        ],
        interpret=interpret,
    )(vt, ix, iy, a, g.astype(jnp.float32))
    return dvt.astype(vt.dtype), dix, diy, da.astype(a.dtype)


_sample.defvjp(_sample_fwd, _sample_bwd)


def _targets_tpu() -> bool:
    """Whether the computation being traced will run on a TPU.

    A ``jax.default_device(cpu)`` context — how ``--host-init`` initialises parameters on a TPU host —
    overrides the process's default backend, and the TPU kernel cannot lower there.
    """
    device = jax.config.jax_default_device
    if device is not None:
        return (device if isinstance(device, str) else device.platform) == "tpu"
    return jax.default_backend() == "tpu"


def sample_weighted(
    values: Array, locations: Array, weights: Array, height: int, width: int, interpret: bool | None = None
) -> Array:
    """Weighted bilinear samples of one level, summed over each query's points.

    Args:
        values: Feature rows ``[N, H * W, D]`` (``N`` = batch x heads).
        locations: Sample positions ``[N, Q, P, 2]`` as ``(x, y)`` in ``[0, 1]``.
        weights: Attention weights ``[N, Q, P]``.
        height, width: The level's size.
        interpret: Run the kernel in the Pallas interpreter. ``None`` does so on any backend but
            TPU, which keeps the sampler usable (and testable) off a TPU, if slowly.

    Returns:
        ``[N, Q, D]`` float32: ``sum_p weights * grid_sample(values, locations)``.
    """
    if interpret is None:
        interpret = not _targets_tpu()
    n, queries, points, _ = locations.shape
    samples = queries * points
    tile = _tile_for(height * width, samples)
    padded = -(-samples // tile) * tile
    locations = locations.astype(jnp.float32).reshape(n, samples, 2)
    ix = locations[..., 0] * width - 0.5
    iy = locations[..., 1] * height - 0.5
    a = weights.astype(jnp.float32).reshape(n, samples)
    if padded != samples:
        # Padding samples sit far off the map with zero weight, so they read nothing and add nothing.
        pad = ((0, 0), (0, padded - samples))
        ix = jnp.pad(ix, pad, constant_values=-8.0)
        iy = jnp.pad(iy, pad, constant_values=-8.0)
        a = jnp.pad(a, pad)
    vt = jnp.swapaxes(values, 1, 2)
    out = _sample(vt, ix[:, None, :], iy[:, None, :], a[:, None, :], height, width, interpret, tile)
    out = out[:, :, :samples].reshape(n, -1, queries, points).sum(-1)
    return jnp.swapaxes(out, 1, 2)
