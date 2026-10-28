# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""Backbone + Joiner, ported from ``rfdetr.models.backbone``.

``Backbone`` runs the windowed DINOv2 encoder and the multi-scale projector; ``Joiner`` adds the
sine position embedding. Together they occupy the ``backbone.0`` / ``backbone.1`` slots of the
upstream ``nn.Sequential``, which this port flattens to the single name ``backbone_0`` (the
position embedding is parameter-free).
"""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from rfdetr_jax.models.backbone.dinov2_windowed import (
    SIZE_TO_WIDTH,
    WindowedDinov2Backbone,
    WindowedDinov2Config,
    build_dinov2_config,
)
from rfdetr_jax.models.backbone.projector import MultiScaleProjector, scale_factors_for
from rfdetr_jax.models.position_encoding import position_embedding_sine
from rfdetr_jax.utilities.interpolate import resize_2d

Array = jax.Array


class Backbone(nn.Module):
    """The DINOv2 encoder plus the multi-scale projector (and an optional second projector)."""

    dino_config: WindowedDinov2Config
    encoder_channels: Sequence[int]
    out_channels: int
    scale_factors: Sequence[float]
    layer_norm: bool = True
    rms_norm: bool = False
    dual_projector: bool = False
    drop_path_rates: tuple[float, ...] = ()

    @nn.compact
    def __call__(self, pixel_values: Array, train: bool = False) -> tuple[list[Array], list[Array] | None]:
        """Return projected feature maps, and cross-attention feature maps when dual-projected."""
        raw_features = WindowedDinov2Backbone(self.dino_config, self.drop_path_rates, name="encoder")(
            pixel_values, deterministic=not train
        )
        features = MultiScaleProjector(
            self.encoder_channels,
            self.out_channels,
            self.scale_factors,
            layer_norm=self.layer_norm,
            rms_norm=self.rms_norm,
            name="projector",
        )(raw_features, train)
        cross_attn_features = None
        if self.dual_projector:
            cross_attn_features = MultiScaleProjector(
                self.encoder_channels,
                self.out_channels,
                self.scale_factors,
                layer_norm=self.layer_norm,
                rms_norm=self.rms_norm,
                name="cross_attn_projector",
            )(raw_features, train)
        return features, cross_attn_features


def downsample_mask(mask: Array, size: tuple[int, int]) -> Array:
    """Nearest-downsample a ``[B, H, W]`` boolean padding mask onto a feature-map grid."""
    resized = resize_2d(mask[:, None].astype(jnp.float32), size, mode="nearest")
    return resized[:, 0].astype(bool)


def build_position_embeddings(masks: Sequence[Array], hidden_dim: int) -> list[Array]:
    """Return one sine position embedding per feature level, in NCHW layout."""
    return [position_embedding_sine(mask, hidden_dim // 2, normalize=True) for mask in masks]


def resolve_encoder_name(name: str) -> tuple[str, bool, bool]:
    """Split an encoder name such as ``dinov2_windowed_small`` into ``(size, registers, windowed)``.

    Raises:
        AssertionError: If the name does not start with ``dinov2`` or has an unexpected shape.
    """
    parts = name.split("_")
    assert parts[0] == "dinov2", "encoder name must start with 'dinov2'"
    use_registers = "registers" in parts
    if use_registers:
        parts.remove("registers")
    use_windowed = "windowed" in parts
    if use_windowed:
        parts.remove("windowed")
    assert len(parts) == 2, "name should be dinov2, then registers/windowed markers, then the size"
    return parts[-1], use_registers, use_windowed


def build_backbone(
    encoder: str,
    out_feature_indexes: list[int],
    projector_scale: list[str],
    hidden_dim: int,
    patch_size: int,
    num_windows: int,
    positional_encoding_size: int,
    layer_norm: bool = True,
    rms_norm: bool = False,
    dual_projector: bool = False,
    drop_path: float = 0.0,
    window_block_indexes: list[int] | None = None,
    gradient_checkpointing: bool = False,
) -> Backbone:
    """Construct the :class:`Backbone` module described by an RF-DETR model config."""
    size, use_registers, use_windowed = resolve_encoder_name(encoder)
    if not use_windowed:
        raise NotImplementedError("Only the windowed DINOv2 encoder is ported; use a 'dinov2_windowed_*' encoder.")
    config = build_dinov2_config(
        size,
        out_feature_indexes,
        use_registers,
        patch_size,
        num_windows,
        positional_encoding_size,
        drop_path_rate=drop_path,
        window_block_indexes=window_block_indexes,
        gradient_checkpointing=gradient_checkpointing,
    )
    scales = scale_factors_for(projector_scale)
    assert sorted(projector_scale) == list(projector_scale), "projector scales must be in ascending order"
    channels = [SIZE_TO_WIDTH[size]] * len(out_feature_indexes)
    rates = tuple(
        float(value) for value in np.linspace(0, drop_path, config.num_hidden_layers, dtype=np.float32)
    ) if drop_path > 0 else ()
    return Backbone(
        dino_config=config,
        encoder_channels=channels,
        out_channels=hidden_dim,
        scale_factors=scales,
        layer_norm=layer_norm,
        rms_norm=rms_norm,
        dual_projector=dual_projector,
        drop_path_rates=rates,
    )
