# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""NumPy resampling weights matching ``torch.nn.functional.interpolate``.

Kept free of any JAX import so the data pipeline can be shared with a PyTorch process, which is
what makes an identical-batch comparison between the two backends possible.

``jax.image.resize`` is not a drop-in substitute: its cubic kernel uses Keys ``a = -0.5`` where
PyTorch's plain bicubic path uses ``a = -0.75``, and its antialias normalisation differs from
ATen's. Since every call site has statically known sizes, this module builds the exact separable
weight matrix ATen would apply and contracts with it.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import sparse


def cubic_convolution1(x: np.ndarray, a: np.floating) -> np.ndarray:
    """ATen's ``cubic_convolution1``: the Keys kernel on ``|x| < 1``, in ATen's operation order."""
    return ((a + 2) * x - (a + 3)) * x * x + 1


def cubic_convolution2(x: np.ndarray, a: np.floating) -> np.ndarray:
    """ATen's ``cubic_convolution2``: the Keys kernel on ``1 <= |x| < 2``, in ATen's operation order."""
    return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a


def _aa_filter(x: np.ndarray, mode: str, dtype: type) -> np.ndarray:
    """Evaluate ATen's antialias filter (``HelperInterpLinear/Cubic::aa_filter``) in ``dtype``."""
    x = np.abs(x)
    if mode == "bilinear":
        return np.where(x < 1, dtype(1) - x, dtype(0)).astype(dtype)
    a = dtype(-0.5)
    inner = ((a + 2) * x - (a + 3)) * x * x + 1
    outer = (((x - 5) * x + 8) * x - 4) * a
    return np.where(x < 1, inner, np.where(x < 2, outer, dtype(0))).astype(dtype)


def _aten_scale(in_size: int, out_size: int, align_corners: bool, dtype: type) -> np.floating:
    """``area_pixel_compute_scale`` for a size-driven resize, rounded once in ``dtype``."""
    if align_corners:
        return dtype(in_size - 1) / dtype(out_size - 1) if out_size > 1 else dtype(0)
    return dtype(in_size) / dtype(out_size)


def _source_index(scale: np.floating, out_size: int, align_corners: bool, cubic: bool, dtype: type) -> np.ndarray:
    """``area_pixel_compute_source_index`` for every output position, in ``dtype`` arithmetic."""
    index = np.arange(out_size, dtype=dtype)
    if align_corners:
        return scale * index
    real = scale * (index + dtype(0.5)) - dtype(0.5)
    return real if cubic else np.maximum(real, dtype(0))


def antialias_weights(
    in_size: int, out_size: int, mode: str, align_corners: bool, dtype: type = np.float32
) -> np.ndarray:
    """Build the antialiased ``[out_size, in_size]`` resampling matrix ATen uses with ``antialias=True``.

    Follows ``HelperInterpBase::_compute_indices_weights_aa``: a triangle or Pillow ``a = -0.5``
    cubic filter evaluated at ``(j - center + 0.5) * invscale`` over the support window and
    renormalised over the in-bounds taps, all in the input dtype's arithmetic.
    """
    del align_corners  # ATen's antialiased path has no align_corners variant.
    interp_size = 4 if mode == "bicubic" else 2
    scale = _aten_scale(in_size, out_size, False, dtype)
    support = (interp_size * dtype(0.5)) * scale if scale >= 1 else interp_size * dtype(0.5)
    invscale = dtype(1) / scale if scale >= 1 else dtype(1)
    weights = np.zeros((out_size, in_size), dtype=dtype)
    for i in range(out_size):
        center = scale * (dtype(i) + dtype(0.5))
        xmin = max(int(center - support + dtype(0.5)), 0)
        xsize = min(int(center + support + dtype(0.5)), in_size) - xmin
        taps = np.arange(xsize, dtype=dtype)
        row = _aa_filter((taps + dtype(xmin) - center + dtype(0.5)) * invscale, mode, dtype)
        total = dtype(row.sum(dtype=dtype))
        if total != 0:
            row = row / total
        weights[i, xmin : xmin + xsize] = row
    return weights


def border_weights(
    in_size: int, out_size: int, mode: str, align_corners: bool, dtype: type = np.float32
) -> np.ndarray:
    """Build the plain weight matrix, replicating ATen's index clamping and operation order."""
    scale = _aten_scale(in_size, out_size, align_corners, dtype)
    weights = np.zeros((out_size, in_size), dtype=dtype)
    rows = np.arange(out_size)
    if mode == "bilinear":
        real = _source_index(scale, out_size, align_corners, False, dtype)
        index0 = real.astype(np.int64)
        index1 = index0 + (index0 < in_size - 1)
        lambda1 = np.clip(real - index0.astype(dtype), dtype(0), dtype(1)).astype(dtype)
        lambda0 = dtype(1) - lambda1
        np.add.at(weights, (rows, index0), lambda0)
        np.add.at(weights, (rows, index1), lambda1)
        return weights
    real = _source_index(scale, out_size, align_corners, True, dtype)
    base = np.floor(real).astype(np.int64)
    t = (real - base.astype(dtype)).astype(dtype)
    a = dtype(-0.75)
    one_minus = dtype(1) - t
    coefficients = (
        cubic_convolution2(t + dtype(1), a),
        cubic_convolution1(t, a),
        cubic_convolution1(one_minus, a),
        cubic_convolution2(one_minus + dtype(1), a),
    )
    for offset, coefficient in zip(range(-1, 3), coefficients, strict=True):
        np.add.at(weights, (rows, np.clip(base + offset, 0, in_size - 1)), coefficient.astype(dtype))
    return weights


@lru_cache(maxsize=256)
def weights_for_axis(
    in_size: int, out_size: int, mode: str, align_corners: bool, antialias: bool, dtype: str = "float32"
) -> np.ndarray:
    """Pick the antialiased or border-clamped weight matrix for one axis.

    ``dtype`` names the arithmetic ATen would use for an input of that dtype (``float32`` or
    ``float64``), so the weights carry the same rounding as the reference. Cached because a data
    pipeline resizes thousands of images between the same handful of sizes, and building a weight
    matrix costs more than applying one.
    """
    if antialias and align_corners:
        raise NotImplementedError("antialias=True with align_corners=True is not implemented")
    kind = np.dtype(dtype).type
    if kind not in (np.float32, np.float64):
        raise ValueError(f"Unsupported resampling arithmetic dtype: {dtype!r}")
    if antialias:
        matrix = antialias_weights(in_size, out_size, mode, align_corners, kind)
    else:
        matrix = border_weights(in_size, out_size, mode, align_corners, kind)
    matrix = np.ascontiguousarray(matrix, dtype=kind)
    matrix.flags.writeable = False
    return matrix


@lru_cache(maxsize=256)
def sparse_weights_for_axis(
    in_size: int, out_size: int, mode: str, align_corners: bool, antialias: bool, dtype: str = "float32"
) -> sparse.csr_matrix:
    """Return :func:`weights_for_axis` as a CSR matrix holding only its non-zero entries."""
    return sparse.csr_matrix(weights_for_axis(in_size, out_size, mode, align_corners, antialias, dtype))


def resize_2d_numpy(
    x: np.ndarray,
    size: tuple[int, int],
    mode: str = "bilinear",
    align_corners: bool = False,
    antialias: bool = False,
) -> np.ndarray:
    """Resize an NCHW numpy array, matching ``torch.nn.functional.interpolate``.

    Raises:
        ValueError: If ``mode`` is not ``"nearest"``, ``"bilinear"`` or ``"bicubic"``.
    """
    out_h, out_w = size
    in_h, in_w = x.shape[-2], x.shape[-1]
    if (in_h, in_w) == (out_h, out_w):
        return x

    if mode == "nearest":
        rows = np.minimum((np.arange(out_h) * (in_h / out_h)).astype(np.int64), in_h - 1)
        cols = np.minimum((np.arange(out_w) * (in_w / out_w)).astype(np.int64), in_w - 1)
        return x[..., rows, :][..., :, cols]

    if mode not in ("bilinear", "bicubic"):
        raise ValueError(f"Unsupported interpolate mode: {mode!r}")

    kind = np.float64 if x.dtype == np.float64 else np.float32
    wh = sparse_weights_for_axis(in_h, out_h, mode, align_corners, antialias, kind.__name__)
    ww = sparse_weights_for_axis(in_w, out_w, mode, align_corners, antialias, kind.__name__)
    # Each weight row has only a few non-zeros — the kernel's support — so a sparse product does the
    # same sums as the dense GEMM without the multiplications by zero. It does not call BLAS,
    # avoiding nested BLAS thread pools inside a threaded loader.
    leading = x.shape[:-2]
    planes = np.ascontiguousarray(x, dtype=kind).reshape(-1, in_h, in_w)
    out = np.empty((planes.shape[0], out_h, out_w), dtype=kind)
    for index, plane in enumerate(planes):
        # Width first, then height: ATen's separable kernels nest the sums in this order.
        out[index] = wh @ (ww @ plane.T).T
    return out.reshape(*leading, out_h, out_w).astype(x.dtype, copy=False)
