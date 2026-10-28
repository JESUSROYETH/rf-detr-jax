# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""JAX ``torch.nn.functional.interpolate`` equivalents.

The resampling weights come from :mod:`rfdetr_jax.utilities.resample`, which builds the exact
separable matrix ATen would apply; this module only contracts with it on device.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rfdetr_jax.utilities.resample import weights_for_axis

Array = jax.Array


def resize_2d(
    x: Array,
    size: tuple[int, int],
    mode: str = "bilinear",
    align_corners: bool = False,
    antialias: bool = False,
) -> Array:
    """Resize an NCHW array, matching ``torch.nn.functional.interpolate``.

    Args:
        x: Input array ``[N, C, H, W]``.
        size: Target ``(height, width)``.
        mode: ``"nearest"``, ``"bilinear"`` or ``"bicubic"``.
        align_corners: PyTorch's ``align_corners`` flag; ignored for ``"nearest"``.
        antialias: PyTorch's ``antialias`` flag; only meaningful when downsampling.

    Returns:
        The resized array ``[N, C, size[0], size[1]]``.

    Raises:
        ValueError: If ``mode`` is not one of the supported modes.
    """
    out_h, out_w = size
    in_h, in_w = x.shape[-2], x.shape[-1]
    if (in_h, in_w) == (out_h, out_w):
        return x

    if mode == "nearest":
        rows = jnp.minimum((jnp.arange(out_h) * (in_h / out_h)).astype(jnp.int32), in_h - 1)
        cols = jnp.minimum((jnp.arange(out_w) * (in_w / out_w)).astype(jnp.int32), in_w - 1)
        return x[..., rows, :][..., :, cols]

    if mode not in ("bilinear", "bicubic"):
        raise ValueError(f"Unsupported interpolate mode: {mode!r}")

    compute = jnp.promote_types(x.dtype, jnp.float32)
    arithmetic = "float64" if compute == jnp.float64 else "float32"
    wh = jnp.asarray(weights_for_axis(in_h, out_h, mode, align_corners, antialias, arithmetic), compute)
    ww = jnp.asarray(weights_for_axis(in_w, out_w, mode, align_corners, antialias, arithmetic), compute)
    # Width first, then height: ATen's separable kernels nest the sums in this order.
    y = jnp.einsum("pw,nchw->nchp", ww, x.astype(compute), precision="highest")
    y = jnp.einsum("oh,nchp->ncop", wh, y, precision="highest")
    return y.astype(x.dtype)
