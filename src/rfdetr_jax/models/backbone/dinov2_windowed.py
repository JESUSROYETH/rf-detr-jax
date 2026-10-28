# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# Modified from the RF-DETR/HuggingFace windowed DINOv2 implementation.
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 Meta Inc. and the HuggingFace Inc. team. All rights reserved.
# ------------------------------------------------------------------------
"""Windowed DINOv2(-with-registers) backbone, ported from
``rfdetr.models.backbone.dinov2_with_windowed_attn``.

Module and parameter names mirror the HuggingFace module tree the upstream checkpoints were saved
from (``encoder.layer.<i>.attention.attention.query.weight`` and friends), with ``.`` replaced by
``_`` where Flax forbids it. :mod:`rfdetr_jax.convert` performs that one rename and nothing else.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp

from rfdetr_jax.models.layers import Conv2d, LayerNorm, Linear, scaled_dot_product_attention
from rfdetr_jax.utilities.interpolate import resize_2d

Array = jax.Array

SIZE_TO_WIDTH = {"tiny": 192, "small": 384, "base": 768, "large": 1024}


@dataclass(frozen=True)
class WindowedDinov2Config:
    """Architecture settings for the windowed DINOv2 backbone.

    Mirrors ``WindowedDinov2WithRegistersConfig``; only the fields RF-DETR actually reads are kept.
    """

    hidden_size: int = 384
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    mlp_ratio: int = 4
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    image_size: int = 518
    patch_size: int = 14
    num_channels: int = 3
    qkv_bias: bool = True
    layerscale_value: float = 1.0
    drop_path_rate: float = 0.0
    use_swiglu_ffn: bool = False
    num_register_tokens: int = 4
    out_features: tuple[str, ...] = ("stage12",)
    apply_layernorm: bool = True
    reshape_hidden_states: bool = True
    num_windows: int = 1
    window_block_indexes: tuple[int, ...] = ()
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    gradient_checkpointing: bool = False

    @property
    def stage_names(self) -> list[str]:
        """Return HuggingFace stage names, ``stem`` followed by one per hidden layer."""
        return ["stem", *[f"stage{index}" for index in range(1, self.num_hidden_layers + 1)]]


def compute_window_block_indexes(
    out_feature_indexes: list[int], window_block_indexes: list[int] | None = None
) -> list[int]:
    """Return the 0-indexed blocks that run windowed attention.

    When ``window_block_indexes`` is given it is returned unchanged; otherwise the windowed set is
    the complement of ``out_feature_indexes`` within ``range(out_feature_indexes[-1] + 1)``, which
    is the derivation every released RF-DETR checkpoint was trained with.
    """
    if window_block_indexes is not None:
        return list(window_block_indexes)
    remaining = set(range(out_feature_indexes[-1] + 1))
    remaining.difference_update(out_feature_indexes)
    return sorted(remaining)


def load_dinov2_json(size: str, use_registers: bool) -> dict:
    """Load the upstream DINOv2 JSON config for ``size``.

    Args:
        size: One of ``"small"``, ``"base"``, ``"large"``.
        use_registers: Whether to load the with-registers variant.

    Returns:
        The parsed JSON dictionary.
    """
    stem = f"dinov2_with_registers_{size}.json" if use_registers else f"dinov2_{size}.json"
    candidate = Path(__file__).resolve().parent / "dinov2_configs" / stem
    if candidate.exists():
        return json.loads(candidate.read_text())
    raise FileNotFoundError(f"Could not find packaged DINOv2 config {stem!r}")


def drop_path(x: Array, drop_prob: float, deterministic: bool, rng: Array | None) -> Array:
    """Apply per-sample stochastic depth, matching ``Dinov2WithRegistersDropPath``."""
    if drop_prob == 0.0 or deterministic or rng is None:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = jnp.floor(keep_prob + jax.random.uniform(rng, shape, dtype=x.dtype))
    return x / keep_prob * mask


class PatchEmbeddings(nn.Module):
    """Patch projection: a stride-``patch_size`` convolution flattened to a token sequence."""

    config: WindowedDinov2Config

    @nn.compact
    def __call__(self, pixel_values: Array) -> Array:
        """Return ``[B, num_patches, hidden_size]`` patch tokens for an NCHW image batch."""
        cfg = self.config
        y = Conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            name="projection",
        )(pixel_values)
        return y.reshape(y.shape[0], y.shape[1], -1).transpose(0, 2, 1)


class Embeddings(nn.Module):
    """CLS token, register tokens, interpolated position embeddings and window partitioning."""

    config: WindowedDinov2Config

    @nn.compact
    def __call__(self, pixel_values: Array) -> Array:
        """Embed an NCHW image batch into (optionally windowed) token sequences.

        Args:
            pixel_values: Image batch ``[B, C, H, W]``; ``H`` and ``W`` must be divisible by
                ``patch_size * num_windows``.

        Returns:
            Token embeddings ``[B * num_windows ** 2, tokens_per_window, hidden_size]``.

        Raises:
            ValueError: If the spatial dimensions are not divisible by ``patch_size * num_windows``.
        """
        cfg = self.config
        batch_size, _, height, width = pixel_values.shape
        divisor = cfg.patch_size * cfg.num_windows
        if height % divisor != 0 or width % divisor != 0:
            raise ValueError(
                f"Input spatial dimensions must be divisible by patch_size * num_windows "
                f"({cfg.patch_size} * {cfg.num_windows} = {divisor}), but got height={height}, width={width}."
            )

        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        cls_token = self.param("cls_token", nn.initializers.normal(1.0), (1, 1, cfg.hidden_size))
        self.param("mask_token", nn.initializers.zeros, (1, cfg.hidden_size))
        position_embeddings = self.param(
            "position_embeddings", nn.initializers.normal(1.0), (1, num_patches + 1, cfg.hidden_size)
        )

        embeddings = PatchEmbeddings(cfg, name="patch_embeddings")(pixel_values)
        embeddings = jnp.concatenate(
            [jnp.broadcast_to(cls_token, (batch_size, 1, cfg.hidden_size)).astype(embeddings.dtype), embeddings],
            axis=1,
        )
        embeddings = embeddings + self._interpolate_pos_encoding(
            position_embeddings, embeddings.shape[1] - 1, height, width
        ).astype(embeddings.dtype)

        if cfg.num_windows > 1:
            embeddings = self._to_windows(embeddings, batch_size, height, width)

        if cfg.num_register_tokens > 0:
            register_tokens = self.param(
                "register_tokens", nn.initializers.zeros, (1, cfg.num_register_tokens, cfg.hidden_size)
            )
            expanded = jnp.broadcast_to(
                register_tokens, (embeddings.shape[0], cfg.num_register_tokens, cfg.hidden_size)
            ).astype(embeddings.dtype)
            embeddings = jnp.concatenate([embeddings[:, :1], expanded, embeddings[:, 1:]], axis=1)
        return embeddings

    def _interpolate_pos_encoding(
        self, position_embeddings: Array, num_patches: int, height: int, width: int
    ) -> Array:
        """Resize the pretrained position table to the current patch grid (bicubic, antialiased)."""
        cfg = self.config
        num_positions = position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return position_embeddings

        class_pos_embed = position_embeddings[:, 0]
        patch_pos_embed = position_embeddings[:, 1:]
        dim = position_embeddings.shape[-1]
        grid_h = height // cfg.patch_size
        grid_w = width // cfg.patch_size
        side = int(round(math.sqrt(num_positions)))

        patch_pos_embed = patch_pos_embed.reshape(1, side, side, dim).transpose(0, 3, 1, 2)
        patch_pos_embed = resize_2d(
            patch_pos_embed.astype(jnp.float32), (grid_h, grid_w), mode="bicubic", align_corners=False, antialias=True
        )
        patch_pos_embed = patch_pos_embed.transpose(0, 2, 3, 1).reshape(1, -1, dim)
        return jnp.concatenate([class_pos_embed[None], patch_pos_embed], axis=1)

    def _to_windows(self, embeddings: Array, batch_size: int, height: int, width: int) -> Array:
        """Split the patch grid into ``num_windows ** 2`` windows, replicating the CLS token."""
        cfg = self.config
        num_h = height // cfg.patch_size
        num_w = width // cfg.patch_size
        windows = cfg.num_windows
        cls_part = embeddings[:, :1]
        pixel_part = embeddings[:, 1:].reshape(batch_size, num_h, num_w, -1)
        per_h = num_h // windows
        per_w = num_w // windows
        pixel_part = pixel_part.reshape(batch_size * windows, per_h, windows, per_w, -1)
        pixel_part = pixel_part.transpose(0, 2, 1, 3, 4)
        pixel_part = pixel_part.reshape(batch_size * windows**2, per_h * per_w, -1)
        # The CLS row is identical across the batch (same parameter, same broadcast position
        # embedding), so torch's `.repeat` tiling order is numerically equivalent to this one.
        cls_part = jnp.tile(cls_part, (windows**2, 1, 1))
        return jnp.concatenate([cls_part, pixel_part], axis=1)


class SelfAttention(nn.Module):
    """Scaled dot-product self-attention with separate query/key/value projections."""

    config: WindowedDinov2Config

    @nn.compact
    def __call__(self, hidden_states: Array) -> Array:
        """Return the attention context for ``[B, L, C]`` tokens."""
        cfg = self.config
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        heads = cfg.num_attention_heads

        def project(name: str) -> Array:
            """Apply one of the q/k/v projections and split it into heads."""
            y = Linear(cfg.hidden_size, cfg.hidden_size, use_bias=cfg.qkv_bias, name=name)(hidden_states)
            return y.reshape(*y.shape[:2], heads, head_dim).transpose(0, 2, 1, 3)

        query, key, value = project("query"), project("key"), project("value")
        context = scaled_dot_product_attention(query, key, value)
        return context.transpose(0, 2, 1, 3).reshape(*hidden_states.shape[:2], cfg.hidden_size)


class Attention(nn.Module):
    """Self-attention followed by the output projection (``attention.attention`` + ``attention.output``)."""

    config: WindowedDinov2Config

    @nn.compact
    def __call__(self, hidden_states: Array) -> Array:
        """Return the projected attention output."""
        context = SelfAttention(self.config, name="attention")(hidden_states)
        return SelfOutput(self.config, name="output")(context)


class SelfOutput(nn.Module):
    """The post-attention dense projection; the residual add lives in :class:`Layer`."""

    config: WindowedDinov2Config

    @nn.compact
    def __call__(self, hidden_states: Array) -> Array:
        """Apply the output dense layer."""
        return Linear(self.config.hidden_size, self.config.hidden_size, name="dense")(hidden_states)


class LayerScale(nn.Module):
    """Per-channel learned scale (``lambda1``) applied to a residual branch."""

    config: WindowedDinov2Config

    @nn.compact
    def __call__(self, hidden_state: Array) -> Array:
        """Multiply by the learned ``lambda1`` vector."""
        lambda1 = self.param(
            "lambda1",
            lambda key, shape: jnp.full(shape, self.config.layerscale_value, jnp.float32),
            (self.config.hidden_size,),
        )
        return hidden_state * lambda1.astype(hidden_state.dtype)


class Mlp(nn.Module):
    """The standard two-layer GELU feed-forward block."""

    config: WindowedDinov2Config

    @nn.compact
    def __call__(self, hidden_state: Array) -> Array:
        """Apply ``fc2(act(fc1(x)))``."""
        cfg = self.config
        hidden_features = int(cfg.hidden_size * cfg.mlp_ratio)
        y = Linear(cfg.hidden_size, hidden_features, name="fc1")(hidden_state)
        y = _activation(cfg.hidden_act)(y)
        return Linear(hidden_features, cfg.hidden_size, name="fc2")(y)


class SwiGLUFFN(nn.Module):
    """The SwiGLU feed-forward variant used by some DINOv2 sizes."""

    config: WindowedDinov2Config

    @nn.compact
    def __call__(self, hidden_state: Array) -> Array:
        """Apply ``weights_out(silu(x1) * x2)`` after a fused input projection."""
        cfg = self.config
        hidden_features = int(cfg.hidden_size * cfg.mlp_ratio)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        y = Linear(cfg.hidden_size, 2 * hidden_features, name="weights_in")(hidden_state)
        x1, x2 = jnp.split(y, 2, axis=-1)
        return Linear(hidden_features, cfg.hidden_size, name="weights_out")(jax.nn.silu(x1) * x2)


def _activation(name: str):
    """Return the activation callable for a HuggingFace activation name."""
    if name in ("gelu", "gelu_new"):
        return lambda x: jax.nn.gelu(x, approximate=False)
    if name == "relu":
        return jax.nn.relu
    if name == "silu":
        return jax.nn.silu
    raise ValueError(f"Unsupported hidden_act: {name!r}")


class Layer(nn.Module):
    """One pre-norm transformer block with LayerScale, drop-path and optional window merging."""

    config: WindowedDinov2Config
    drop_path_rate: float = 0.0

    @nn.compact
    def __call__(self, hidden_states: Array, run_full_attention: bool, deterministic: bool) -> Array:
        """Run the block, merging windows first when ``run_full_attention`` is set."""
        cfg = self.config
        shortcut = hidden_states
        if run_full_attention:
            batch_windows, tokens_per_window, channels = hidden_states.shape
            squared = cfg.num_windows**2
            hidden_states = hidden_states.reshape(batch_windows // squared, squared * tokens_per_window, channels)

        attention_output = Attention(cfg, name="attention")(
            LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps, name="norm1")(hidden_states)
        )

        if run_full_attention:
            batch_windows, tokens_per_window, channels = hidden_states.shape
            squared = cfg.num_windows**2
            attention_output = attention_output.reshape(
                batch_windows * squared, tokens_per_window // squared, channels
            )

        attention_output = LayerScale(cfg, name="layer_scale1")(attention_output)
        hidden_states = self._drop_path(attention_output, deterministic) + shortcut

        layer_output = LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps, name="norm2")(hidden_states)
        ffn = SwiGLUFFN if cfg.use_swiglu_ffn else Mlp
        layer_output = ffn(cfg, name="mlp")(layer_output)
        layer_output = LayerScale(cfg, name="layer_scale2")(layer_output)
        return self._drop_path(layer_output, deterministic) + hidden_states

    def _drop_path(self, x: Array, deterministic: bool) -> Array:
        """Apply stochastic depth using the ``dropout`` RNG stream when training."""
        if self.drop_path_rate == 0.0 or deterministic:
            return x
        return drop_path(x, self.drop_path_rate, deterministic, self.make_rng("dropout"))


class Encoder(nn.Module):
    """The stack of transformer blocks, returning every stage's hidden state."""

    config: WindowedDinov2Config
    drop_path_rates: tuple[float, ...] = ()

    @nn.compact
    def __call__(self, hidden_states: Array, deterministic: bool) -> list[Array]:
        """Return ``[embedding_output, after_layer_0, ...]`` up to the last requested stage."""
        cfg = self.config
        last_feature = cfg.out_features[-1]
        early_stop_layer = int(last_feature[5:]) if last_feature.startswith("stage") else None

        # Checkpointing recomputes each block in the backward pass instead of holding its
        # activations, which is what makes a 384px batch fit alongside `group_detr` query groups
        # on a small GPU. `static_argnums` keeps the two Python bools out of the remat trace.
        # `nn.remat` threads the same RNG through the recomputation, so a non-zero
        # `drop_path_rate` still draws one mask per block rather than two different ones.
        block = nn.remat(Layer, static_argnums=(2, 3)) if cfg.gradient_checkpointing else Layer

        all_hidden_states = [hidden_states]
        for index in range(cfg.num_hidden_layers):
            if early_stop_layer is not None and index > early_stop_layer:
                break
            rate = self.drop_path_rates[index] if index < len(self.drop_path_rates) else cfg.drop_path_rate
            hidden_states = block(cfg, drop_path_rate=rate, name=f"layer_{index}")(
                hidden_states, index not in cfg.window_block_indexes, deterministic
            )
            all_hidden_states.append(hidden_states)
        return all_hidden_states


class WindowedDinov2Backbone(nn.Module):
    """The DINOv2 backbone: embeddings, encoder, and per-stage feature-map reshaping."""

    config: WindowedDinov2Config
    drop_path_rates: tuple[float, ...] = ()

    @nn.compact
    def __call__(self, pixel_values: Array, deterministic: bool = True) -> list[Array]:
        """Return one NCHW feature map per entry of ``config.out_features``."""
        cfg = self.config
        embedding_output = Embeddings(cfg, name="embeddings")(pixel_values)
        hidden_states = Encoder(cfg, self.drop_path_rates, name="encoder")(embedding_output, deterministic)
        layernorm = LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps, name="layernorm")

        batch_size, _, height, width = pixel_values.shape
        num_h = height // cfg.patch_size
        num_w = width // cfg.patch_size

        feature_maps = []
        for stage, hidden_state in zip(cfg.stage_names, hidden_states, strict=False):
            if stage not in cfg.out_features:
                continue
            if cfg.apply_layernorm:
                hidden_state = layernorm(hidden_state)
            if cfg.reshape_hidden_states:
                hidden_state = hidden_state[:, cfg.num_register_tokens + 1 :]
                if cfg.num_windows > 1:
                    hidden_state = self._undo_windows(hidden_state, num_h, num_w)
                hidden_state = hidden_state.reshape(batch_size, num_h, num_w, -1).transpose(0, 3, 1, 2)
            feature_maps.append(hidden_state)
        return feature_maps

    def _undo_windows(self, hidden_state: Array, num_h: int, num_w: int) -> Array:
        """Reassemble windowed tokens back into a single row-major patch sequence."""
        windows = self.config.num_windows
        squared = windows**2
        batch_windows, tokens_per_window, channels = hidden_state.shape
        per_h = num_h // windows
        per_w = num_w // windows
        hidden_state = hidden_state.reshape(batch_windows // squared, squared * tokens_per_window, channels)
        hidden_state = hidden_state.reshape(
            (batch_windows // squared) * windows, windows, per_h, per_w, channels
        )
        return hidden_state.transpose(0, 2, 1, 3, 4)


def build_dinov2_config(
    size: str,
    out_feature_indexes: list[int],
    use_registers: bool,
    patch_size: int,
    num_windows: int,
    positional_encoding_size: int,
    drop_path_rate: float = 0.0,
    window_block_indexes: list[int] | None = None,
    gradient_checkpointing: bool = False,
) -> WindowedDinov2Config:
    """Assemble a :class:`WindowedDinov2Config` the way ``rfdetr.models.backbone.dinov2.DinoV2`` does."""
    raw = load_dinov2_json(size, use_registers)
    resolved_windows = compute_window_block_indexes(out_feature_indexes, window_block_indexes)
    image_size = positional_encoding_size * patch_size
    return WindowedDinov2Config(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        mlp_ratio=raw["mlp_ratio"],
        hidden_act=raw["hidden_act"],
        layer_norm_eps=raw["layer_norm_eps"],
        image_size=image_size,
        patch_size=patch_size,
        num_channels=raw["num_channels"],
        qkv_bias=raw["qkv_bias"],
        layerscale_value=raw["layerscale_value"],
        drop_path_rate=drop_path_rate,
        use_swiglu_ffn=raw["use_swiglu_ffn"],
        num_register_tokens=raw.get("num_register_tokens", 0) if use_registers else 0,
        out_features=tuple(f"stage{index}" for index in out_feature_indexes),
        apply_layernorm=raw.get("apply_layernorm", True),
        reshape_hidden_states=raw.get("reshape_hidden_states", True),
        num_windows=num_windows,
        window_block_indexes=tuple(resolved_windows),
        gradient_checkpointing=gradient_checkpointing,
    )
