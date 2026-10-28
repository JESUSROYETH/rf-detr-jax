# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""Flax Linen layers that mirror ``torch.nn`` parameter names, shapes and numerics.

Every module here stores its parameters under the *PyTorch* names and in the *PyTorch* memory
layout (``weight`` as ``[out, in]`` for a linear layer, ``[O, I, kH, kW]`` for a convolution, and so
on). That makes checkpoint conversion a pure key rename with no transposes, and it lets a parity
test compare a JAX parameter against its ``state_dict`` counterpart element-by-element.

Tensors keep the upstream NCHW layout for the same reason: the port can then follow the original
line by line instead of re-deriving every axis permutation.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array
_ATTENTION = os.environ.get("RFDETR_JAX_ATTENTION", "xla")


def set_attention_implementation(name: str) -> None:
    """Select ``"xla"`` or ``"cudnn"`` for :func:`scaled_dot_product_attention` in later traces."""
    global _ATTENTION
    if name not in ("xla", "cudnn"):
        raise ValueError(f"unknown attention implementation {name!r}; expected 'xla' or 'cudnn'")
    _ATTENTION = name


def _on_gpu() -> bool:
    """Select the platform policy from ``jax.default_device`` or the default backend.

    Low-level callers targeting a non-default backend must set the matching
    default-device context while tracing; an explicit ``jit(backend=...)`` or
    input sharding alone does not expose its target to this Python helper.
    """
    device = jax.config.jax_default_device
    if isinstance(device, str):
        return device == "gpu"  # JAX validates strings: "cpu", "gpu", "tpu".
    platform = device.platform if device is not None else jax.default_backend()
    return platform == "gpu"


def scaled_dot_product_attention(q: Array, k: Array, v: Array, mask: Array | None = None) -> Array:
    """Softmax attention over ``[B, H, L, D]`` heads, with torch's boolean-mask convention.

    The reference path is two einsums around a softmax promoted to at least float32. On a GPU with bf16 or fp16
    activations, ``RFDETR_JAX_ATTENTION=cudnn`` routes the same computation to cuDNN's fused
    attention, which never writes the ``[B, H, Lq, Lk]`` score tensor to memory.

    Args:
        q: Queries ``[B, H, Lq, D]``.
        k: Keys ``[B, H, Lk, D]``.
        v: Values ``[B, H, Lk, D]``.
        mask: Optional boolean mask broadcastable to ``[Lq, Lk]``; ``True`` blocks a position.

    Returns:
        Attention output ``[B, H, Lq, D]``.
    """
    head_dim = q.shape[-1]
    if _ATTENTION == "cudnn" and q.dtype in (jnp.bfloat16, jnp.float16) and _on_gpu():
        allowed = None if mask is None else jnp.broadcast_to(~mask, (q.shape[0], q.shape[1], q.shape[2], k.shape[2]))
        q_len, kv_len = q.shape[2], k.shape[2]
        lengths = {}
        if allowed is None and (q_len % 2 or kv_len % 2):
            # cuDNN's training kernels reject odd sequence lengths; pad one row and tell it the real
            # lengths, so the pad never takes part in the softmax and its output row is dropped.
            pad = lambda t: jnp.pad(t, ((0, 0), (0, 0), (0, t.shape[2] % 2), (0, 0)))  # noqa: E731
            q, k, v = pad(q), pad(k), pad(v)
            lengths = {
                "query_seq_lengths": jnp.full((q.shape[0],), q_len, jnp.int32),
                "key_value_seq_lengths": jnp.full((q.shape[0],), kv_len, jnp.int32),
            }
        out = jax.nn.dot_product_attention(
            q.transpose(0, 2, 1, 3),
            k.transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
            mask=allowed,
            implementation="cudnn",
            **lengths,
        )
        return out.transpose(0, 2, 1, 3)[:, :, :q_len]
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(head_dim)
    if mask is not None:
        scores = jnp.where(mask, jnp.finfo(scores.dtype).min, scores)
    softmax_dtype = jnp.promote_types(scores.dtype, jnp.float32)
    weights = jax.nn.softmax(scores.astype(softmax_dtype), axis=-1).astype(scores.dtype)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    """Return an int or 2-sequence as a ``(height, width)`` pair."""
    if isinstance(value, Sequence):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def kaiming_uniform_fan_in(shape: tuple[int, ...], fan_in: int, dtype=jnp.float32) -> Callable:
    """Return torch's default ``nn.Linear``/``nn.Conv2d`` weight initializer (Kaiming uniform, a=sqrt(5))."""
    # torch uses kaiming_uniform_(a=sqrt(5)), whose bound simplifies to exactly 1/sqrt(fan_in).
    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0

    def init(key: Array) -> Array:
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init


def _uniform_bias(fan_in: int) -> Callable:
    """Return torch's default bias initializer: ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``."""
    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0

    def init(key: Array, shape: tuple[int, ...], dtype=jnp.float32) -> Array:
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init


class ScopeCache:
    """Hands out child scopes by path, creating each one only once.

    Flax refuses a second ``push`` of the same name, so a caller that wants to read several
    parameters out of the same submodule scope has to remember it. This keeps that bookkeeping in
    one place.
    """

    def __init__(self, root):
        """Wrap a ``flax.core`` scope, typically ``self.scope`` inside a compact method."""
        self._root = root
        self._children: dict[str, object] = {}

    def child(self, path: str):
        """Return the scope at a ``/``-separated path below the root, creating it if needed.

        Every prefix along the way is cached too, so two paths that share a parent — the layers of
        one MLP, say — push that parent once between them.
        """
        scope = self._root
        prefix = ""
        for part in path.split("/"):
            prefix = f"{prefix}/{part}" if prefix else part
            if prefix not in self._children:
                self._children[prefix] = scope.push(part)
            scope = self._children[prefix]
        return scope


def stacked_linear(
    cache: ScopeCache, paths: Sequence[str], in_features: int, out_features: int, param_dtype=jnp.float32
) -> tuple[Array, Array]:
    """Return one group of identically shaped :class:`Linear` parameters stacked on a leading axis.

    Flax gives every submodule its own scope, so a group of same-shaped heads applied to the same
    input becomes a group of same-shaped small matmuls — the pattern an accelerator handles worst.
    Reading those parameters back through their own scopes and stacking them lets a single batched
    einsum replace the group while the parameter tree a checkpoint sees stays exactly as it was.

    Args:
        cache: A :class:`ScopeCache` over the parent scope.
        paths: Submodule paths, in the order the stacked axis should follow.
        in_features: Input width of each linear.
        out_features: Output width of each linear.
        param_dtype: Parameter dtype.

    Returns:
        A ``(weight, bias)`` pair shaped ``[G, out_features, in_features]`` and ``[G, out_features]``.
    """
    bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 0.0

    def weight_init(key: Array) -> Array:
        """Reproduce ``Linear``'s torch-matching weight initializer."""
        return jax.random.uniform(key, (out_features, in_features), param_dtype, minval=-bound, maxval=bound)

    weights, biases = [], []
    for path in paths:
        scope = cache.child(path)
        weights.append(scope.param("weight", weight_init))
        biases.append(scope.param("bias", _uniform_bias(in_features), (out_features,), param_dtype))
    return jnp.stack(weights), jnp.stack(biases)


def stacked_layer_norm(
    cache: ScopeCache, paths: Sequence[str], features: int, param_dtype=jnp.float32
) -> tuple[Array, Array]:
    """Return a group of :class:`LayerNorm` parameters stacked on a leading axis.

    The counterpart of :func:`stacked_linear`; see it for why the group is fused.
    """
    weights, biases = [], []
    for path in paths:
        scope = cache.child(path)
        weights.append(scope.param("weight", nn.initializers.ones, (features,), param_dtype))
        biases.append(scope.param("bias", nn.initializers.zeros, (features,), param_dtype))
    return jnp.stack(weights), jnp.stack(biases)


def grouped_linear(x: Array, weight: Array, bias: Array, grouped_input: bool = True) -> Array:
    """Apply one linear per group in a single batched einsum.

    Args:
        x: ``[G, ..., in]`` when ``grouped_input``, otherwise a shared ``[..., in]`` broadcast
            against every group.
        weight: ``[G, out, in]``, the torch layout stacked over groups.
        bias: ``[G, out]``.
        grouped_input: Whether ``x`` already carries the group axis.

    Returns:
        ``[G, ..., out]``.
    """
    subscript = "g...i,goi->g...o" if grouped_input else "...i,goi->g...o"
    y = jnp.einsum(subscript, x, weight.astype(x.dtype))
    return y + bias.reshape((bias.shape[0],) + (1,) * (y.ndim - 2) + (bias.shape[-1],)).astype(y.dtype)


def select_rows(values: Array, indices: Array) -> Array:
    """Gather rows of ``values`` at ``indices``, with every leading batch axis collapsed first.

    Args:
        values: ``[..., N, C]``, the rows to select from.
        indices: ``[..., K]``, row indices, which must be in bounds.

    Returns:
        ``[..., K, C]``.

    Collapse leading axes to express a single batched row gather, then restore
    the original batch shape. This exposes whole-row indexing to the compiler.
    """
    leading = indices.shape[:-1]
    picks, width = indices.shape[-1], values.shape[-1]
    flat = values.reshape(-1, values.shape[-2], width)
    taken = jnp.take_along_axis(
        flat, indices.reshape(-1, picks)[..., None], axis=1, mode="promise_in_bounds"
    )
    return taken.reshape(*leading, picks, width)


def grouped_layer_norm(x: Array, weight: Array, bias: Array, eps: float = 1e-5) -> Array:
    """Apply :class:`LayerNorm` over the trailing axis of ``[G, ...]`` with per-group parameters."""
    compute = jnp.promote_types(x.dtype, jnp.float32)
    x32 = x.astype(compute)
    mean = x32.mean(-1, keepdims=True)
    var = x32.var(-1, keepdims=True)
    y = (x32 - mean) * jax.lax.rsqrt(var + eps)
    shape = (weight.shape[0],) + (1,) * (x.ndim - 2) + (weight.shape[-1],)
    return (y * weight.reshape(shape).astype(compute) + bias.reshape(shape).astype(compute)).astype(x.dtype)


class Linear(nn.Module):
    """``torch.nn.Linear``: stores ``weight`` as ``[out_features, in_features]`` and ``bias`` as ``[out_features]``."""

    in_features: int
    out_features: int
    use_bias: bool = True
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Apply ``x @ weight.T + bias`` over the trailing axis."""
        bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
        weight = self.param(
            "weight",
            lambda key: jax.random.uniform(
                key, (self.out_features, self.in_features), self.param_dtype, minval=-bound, maxval=bound
            ),
        )
        y = jnp.einsum("...i,oi->...o", x, weight.astype(x.dtype))
        if self.use_bias:
            bias = self.param("bias", _uniform_bias(self.in_features), (self.out_features,), self.param_dtype)
            y = y + bias.astype(y.dtype)
        return y


class LayerNorm(nn.Module):
    """``torch.nn.LayerNorm`` over the trailing axis, with torch's ``weight``/``bias`` names."""

    features: int
    eps: float = 1e-5
    use_affine: bool = True
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Normalize the trailing axis, then scale and shift."""
        compute = jnp.promote_types(x.dtype, jnp.float32)
        x32 = x.astype(compute)
        mean = x32.mean(-1, keepdims=True)
        var = x32.var(-1, keepdims=True)
        y = (x32 - mean) * jax.lax.rsqrt(var + self.eps)
        if self.use_affine:
            weight = self.param("weight", nn.initializers.ones, (self.features,), self.param_dtype)
            bias = self.param("bias", nn.initializers.zeros, (self.features,), self.param_dtype)
            y = y * weight.astype(compute) + bias.astype(compute)
        return y.astype(x.dtype)


class RMSNorm(nn.Module):
    """``torch.nn.RMSNorm`` over the trailing axis."""

    features: int
    eps: float = 1e-6
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Scale the trailing axis by the inverse root-mean-square, then apply ``weight``."""
        compute = jnp.promote_types(x.dtype, jnp.float32)
        x32 = x.astype(compute)
        y = x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), -1, keepdims=True) + self.eps)
        weight = self.param("weight", nn.initializers.ones, (self.features,), self.param_dtype)
        return (y * weight.astype(compute)).astype(x.dtype)


class ChannelLayerNorm(nn.Module):
    """The projector's ``LayerNorm``: normalizes the channel axis of an NCHW tensor.

    Mirrors ``rfdetr.models.backbone.projector.LayerNorm``, which permutes to NHWC, applies
    ``F.layer_norm`` over the channel axis, and permutes back.
    """

    features: int
    eps: float = 1e-6
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Normalize ``x`` (NCHW) over its channel axis, keeping torch's flat ``weight``/``bias`` names."""
        weight = self.param("weight", nn.initializers.ones, (self.features,), self.param_dtype)
        bias = self.param("bias", nn.initializers.zeros, (self.features,), self.param_dtype)
        y = jnp.transpose(x, (0, 2, 3, 1))
        compute = jnp.promote_types(y.dtype, jnp.float32)
        y32 = y.astype(compute)
        mean = y32.mean(-1, keepdims=True)
        var = y32.var(-1, keepdims=True)
        y32 = (y32 - mean) * jax.lax.rsqrt(var + self.eps)
        y32 = y32 * weight.astype(compute) + bias.astype(compute)
        return jnp.transpose(y32.astype(x.dtype), (0, 3, 1, 2))


class Conv2d(nn.Module):
    """``torch.nn.Conv2d`` on NCHW inputs, storing ``weight`` as ``[O, I // groups, kH, kW]``."""

    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int, int]
    stride: int | tuple[int, int] = 1
    padding: int | tuple[int, int] = 0
    groups: int = 1
    dilation: int | tuple[int, int] = 1
    use_bias: bool = True
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Run a stride/padding/dilation/group convolution matching torch's semantics."""
        kh, kw = _pair(self.kernel_size)
        sh, sw = _pair(self.stride)
        ph, pw = _pair(self.padding)
        dh, dw = _pair(self.dilation)
        fan_in = self.in_channels // self.groups * kh * kw
        weight = self.param(
            "weight",
            kaiming_uniform_fan_in(
                (self.out_channels, self.in_channels // self.groups, kh, kw), fan_in, self.param_dtype
            ),
        )
        y = jax.lax.conv_general_dilated(
            x,
            weight.astype(x.dtype),
            window_strides=(sh, sw),
            padding=((ph, ph), (pw, pw)),
            rhs_dilation=(dh, dw),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=self.groups,
        )
        if self.use_bias:
            bias = self.param("bias", _uniform_bias(fan_in), (self.out_channels,), self.param_dtype)
            y = y + bias.astype(y.dtype)[None, :, None, None]
        return y


class ConvTranspose2d(nn.Module):
    """``torch.nn.ConvTranspose2d`` on NCHW inputs, storing ``weight`` as ``[I, O // groups, kH, kW]``."""

    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int, int]
    stride: int | tuple[int, int] = 1
    padding: int | tuple[int, int] = 0
    groups: int = 1
    use_bias: bool = True
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Run a transposed convolution matching torch's output size and weight layout."""
        kh, kw = _pair(self.kernel_size)
        sh, sw = _pair(self.stride)
        ph, pw = _pair(self.padding)
        fan_in = self.out_channels // self.groups * kh * kw
        weight = self.param(
            "weight",
            kaiming_uniform_fan_in(
                (self.in_channels, self.out_channels // self.groups, kh, kw), fan_in, self.param_dtype
            ),
        )
        # torch stores [I, O, kH, kW]; lax.conv_transpose with transpose_kernel=True wants [I, O, kH, kW]
        # flipped spatially, which `conv_general_dilated` reproduces via explicit input dilation.
        kernel = jnp.flip(weight.astype(x.dtype), axis=(2, 3))
        kernel = jnp.swapaxes(kernel, 0, 1)  # -> [O, I, kH, kW]
        pad_h = (kh - 1 - ph, kh - 1 - ph)
        pad_w = (kw - 1 - pw, kw - 1 - pw)
        y = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding=(pad_h, pad_w),
            lhs_dilation=(sh, sw),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=self.groups,
        )
        if self.use_bias:
            bias = self.param("bias", _uniform_bias(fan_in), (self.out_channels,), self.param_dtype)
            y = y + bias.astype(y.dtype)[None, :, None, None]
        return y


class BatchNorm2d(nn.Module):
    """``torch.nn.BatchNorm2d`` on NCHW inputs, with torch's parameter and buffer names."""

    features: int
    eps: float = 1e-5
    momentum: float = 0.1
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, *, train: bool = False) -> Array:
        """Normalize per channel, updating running statistics when ``train`` is set."""
        weight = self.param("weight", nn.initializers.ones, (self.features,), self.param_dtype)
        bias = self.param("bias", nn.initializers.zeros, (self.features,), self.param_dtype)
        running_mean = self.variable("batch_stats", "running_mean", jnp.zeros, (self.features,), self.param_dtype)
        running_var = self.variable("batch_stats", "running_var", jnp.ones, (self.features,), self.param_dtype)

        if train:
            mean = x.mean(axis=(0, 2, 3))
            var = x.var(axis=(0, 2, 3))
            if not self.is_initializing():
                count = x.shape[0] * x.shape[2] * x.shape[3]
                unbiased = var * count / max(count - 1, 1)
                running_mean.value = (1 - self.momentum) * running_mean.value + self.momentum * mean
                running_var.value = (1 - self.momentum) * running_var.value + self.momentum * unbiased
        else:
            mean = running_mean.value
            var = running_var.value

        y = (x - mean[None, :, None, None]) * jax.lax.rsqrt(var[None, :, None, None] + self.eps)
        return y * weight.astype(y.dtype)[None, :, None, None] + bias.astype(y.dtype)[None, :, None, None]


class MLP(nn.Module):
    """``rfdetr.models.math.MLP``: ``num_layers`` linear layers with ReLU between them."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Apply the stack, with ReLU after every layer but the last."""
        hidden = [self.hidden_dim] * (self.num_layers - 1)
        dims_in = [self.input_dim, *hidden]
        dims_out = [*hidden, self.output_dim]
        for index, (n, k) in enumerate(zip(dims_in, dims_out, strict=True)):
            x = Linear(n, k, param_dtype=self.param_dtype, name=f"layers_{index}")(x)
            if index < self.num_layers - 1:
                x = jax.nn.relu(x)
        return x


class MultiheadAttention(nn.Module):
    """``torch.nn.MultiheadAttention`` with ``batch_first=True`` and fused ``in_proj_weight``."""

    embed_dim: int
    num_heads: int
    use_bias: bool = True
    param_dtype: jnp.dtype = jnp.float32
    dropout: float = 0.0

    @nn.compact
    def __call__(self, query: Array, key: Array, value: Array, attn_mask: Array | None = None,
                 train: bool = False) -> Array:
        """Return the attention output for ``[B, L, D]`` inputs.

        Args:
            query: Query tensor ``[B, Lq, D]``.
            key: Key tensor ``[B, Lk, D]``.
            value: Value tensor ``[B, Lk, D]``.
            attn_mask: Optional boolean mask ``[Lq, Lk]`` (or broadcastable); ``True`` blocks a position,
                matching torch's boolean-mask convention.
            train: Apply attention-probability dropout (``torch.nn.MultiheadAttention(dropout=...)``)
                from the ``dropout`` RNG stream when ``dropout > 0``.

        Returns:
            Attention output ``[B, Lq, D]``.
        """
        d = self.embed_dim
        head_dim = d // self.num_heads
        bound = 1.0 / math.sqrt(d)
        in_proj_weight = self.param(
            "in_proj_weight",
            lambda key_: jax.random.uniform(key_, (3 * d, d), self.param_dtype, minval=-bound, maxval=bound),
        )
        q = jnp.einsum("...i,oi->...o", query, in_proj_weight[:d].astype(query.dtype))
        k = jnp.einsum("...i,oi->...o", key, in_proj_weight[d : 2 * d].astype(key.dtype))
        v = jnp.einsum("...i,oi->...o", value, in_proj_weight[2 * d :].astype(value.dtype))
        if self.use_bias:
            in_proj_bias = self.param("in_proj_bias", nn.initializers.zeros, (3 * d,), self.param_dtype)
            q = q + in_proj_bias[:d].astype(q.dtype)
            k = k + in_proj_bias[d : 2 * d].astype(k.dtype)
            v = v + in_proj_bias[2 * d :].astype(v.dtype)

        def split(t: Array) -> Array:
            return t.reshape(*t.shape[:2], self.num_heads, head_dim).transpose(0, 2, 1, 3)

        q, k, v = split(q), split(k), split(v)
        if self.dropout > 0.0 and train:
            scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(head_dim)
            if attn_mask is not None:
                scores = jnp.where(attn_mask, jnp.finfo(scores.dtype).min, scores)
            softmax_dtype = jnp.promote_types(scores.dtype, jnp.float32)
            weights = jax.nn.softmax(scores.astype(softmax_dtype), axis=-1).astype(scores.dtype)
            weights = nn.Dropout(rate=self.dropout, deterministic=False, name="attn_dropout")(weights)
            out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        else:
            out = scaled_dot_product_attention(q, k, v, attn_mask)
        out = out.transpose(0, 2, 1, 3).reshape(*query.shape[:2], d)
        return Linear(d, d, use_bias=self.use_bias, param_dtype=self.param_dtype, name="out_proj")(out)
