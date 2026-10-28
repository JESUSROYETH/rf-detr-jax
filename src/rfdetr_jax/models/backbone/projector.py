# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# Modified from LW-DETR and ViTDet.
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Multi-scale projector, ported from ``rfdetr.models.backbone.projector``.

Submodule names follow the PyTorch ``nn.Sequential`` / ``nn.ModuleList`` indices (``stages_sampling
.0.1.0.conv``) with ``.`` replaced by ``_``, so checkpoint conversion stays a mechanical rename.
"""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from rfdetr_jax.models.layers import BatchNorm2d, ChannelLayerNorm, Conv2d, ConvTranspose2d, RMSNorm

Array = jax.Array

_LEVEL_TO_SCALE = {"P3": 2.0, "P4": 1.0, "P5": 0.5, "P6": 0.25}


def get_activation(name: str | None):
    """Return the activation callable used by ``ConvX``/``Bottleneck``."""
    if name == "silu":
        return jax.nn.silu
    if name == "relu":
        return jax.nn.relu
    if name in ("LeakyReLU", "leakyrelu", "lrelu"):
        return lambda x: jax.nn.leaky_relu(x, 0.1)
    if name is None:
        return lambda x: x
    raise AttributeError(f"Unsupported act type: {name}")


class ConvX(nn.Module):
    """Conv + norm + activation, matching ``rfdetr.models.backbone.projector.ConvX``."""

    in_planes: int
    out_planes: int
    kernel: int | tuple[int, int] = 3
    stride: int = 1
    groups: int = 1
    dilation: int = 1
    act: str = "relu"
    layer_norm: bool = False
    rms_norm: bool = False

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> Array:
        """Apply ``act(norm(conv(x)))`` on an NCHW input."""
        kernel = self.kernel if isinstance(self.kernel, tuple) else (self.kernel, self.kernel)
        padding = (kernel[0] // 2, kernel[1] // 2)
        y = Conv2d(
            self.in_planes,
            self.out_planes,
            kernel_size=kernel,
            stride=self.stride,
            padding=padding,
            groups=self.groups,
            dilation=self.dilation,
            use_bias=False,
            name="conv",
        )(x)
        if self.rms_norm:
            y = jnp.transpose(y, (0, 2, 3, 1))
            y = RMSNorm(self.out_planes, name="bn")(y)
            y = jnp.transpose(y, (0, 3, 1, 2))
        elif self.layer_norm:
            y = ChannelLayerNorm(self.out_planes, name="bn")(y)
        else:
            y = BatchNorm2d(self.out_planes, name="bn")(y, train=train)
        return get_activation(self.act)(y)


class Bottleneck(nn.Module):
    """Standard YOLO bottleneck used inside :class:`C2f`."""

    c1: int
    c2: int
    shortcut: bool = True
    groups: int = 1
    kernel: tuple[int, int] = (3, 3)
    expansion: float = 0.5
    act: str = "silu"
    layer_norm: bool = False
    rms_norm: bool = False

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> Array:
        """Apply the two convolutions, adding the residual when shapes allow."""
        hidden = int(self.c2 * self.expansion)
        y = ConvX(
            self.c1, hidden, self.kernel[0], 1, act=self.act, layer_norm=self.layer_norm,
            rms_norm=self.rms_norm, name="cv1",
        )(x, train)
        y = ConvX(
            hidden, self.c2, self.kernel[1], 1, groups=self.groups, act=self.act,
            layer_norm=self.layer_norm, rms_norm=self.rms_norm, name="cv2",
        )(y, train)
        return x + y if (self.shortcut and self.c1 == self.c2) else y


class C2f(nn.Module):
    """CSP bottleneck with two convolutions, matching the projector's ``C2f``."""

    c1: int
    c2: int
    n: int = 1
    shortcut: bool = False
    groups: int = 1
    expansion: float = 0.5
    act: str = "silu"
    layer_norm: bool = False
    rms_norm: bool = False

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> Array:
        """Split, run ``n`` bottlenecks, and fuse with the output convolution."""
        c = int(self.c2 * self.expansion)
        y = ConvX(
            self.c1, 2 * c, 1, 1, act=self.act, layer_norm=self.layer_norm, rms_norm=self.rms_norm, name="cv1"
        )(x, train)
        parts = [y[:, :c], y[:, c : 2 * c]]
        for index in range(self.n):
            parts.append(
                Bottleneck(
                    c, c, self.shortcut, self.groups, (3, 3), 1.0, self.act, self.layer_norm, self.rms_norm,
                    name=f"m_{index}",
                )(parts[-1], train)
            )
        fused = jnp.concatenate(parts, axis=1)
        return ConvX(
            (2 + self.n) * c, self.c2, 1, 1, act=self.act, layer_norm=self.layer_norm,
            rms_norm=self.rms_norm, name="cv2",
        )(fused, train)


class MultiScaleProjector(nn.Module):
    """Build pyramid features from the backbone's per-stage feature maps."""

    in_channels: Sequence[int]
    out_channels: int
    scale_factors: Sequence[float]
    num_blocks: int = 3
    layer_norm: bool = False
    rms_norm: bool = False

    @nn.compact
    def __call__(self, features: list[Array], train: bool = False) -> list[Array]:
        """Resample every input level to each requested scale, then fuse per scale.

        Args:
            features: One NCHW feature map per backbone output stage.
            train: Whether BatchNorm (used only when neither norm flag is set) updates its statistics.

        Returns:
            One fused NCHW feature map per entry of ``scale_factors``, plus an extra pooled level
            when a ``0.25`` scale is requested.

        Raises:
            NotImplementedError: If a scale factor other than 4, 2, 1, 0.5 or 0.25 is requested.
        """
        use_extra_pool = False
        results = []
        for stage_index, scale in enumerate(self.scale_factors):
            fused_parts = []
            for level_index, in_dim in enumerate(self.in_channels):
                y = features[level_index]
                prefix = f"stages_sampling_{stage_index}_{level_index}"
                if scale == 4.0:
                    y = ConvTranspose2d(in_dim, in_dim // 2, 2, 2, name=f"{prefix}_0")(y)
                    y = ChannelLayerNorm(in_dim // 2, name=f"{prefix}_1")(y)
                    y = jax.nn.gelu(y, approximate=False)
                    y = ConvTranspose2d(in_dim // 2, in_dim // 4, 2, 2, name=f"{prefix}_3")(y)
                elif scale == 2.0:
                    y = ConvTranspose2d(in_dim, in_dim // 2, 2, 2, name=f"{prefix}_0")(y)
                elif scale == 1.0:
                    pass
                elif scale == 0.5:
                    y = ConvX(in_dim, in_dim, 3, 2, layer_norm=self.layer_norm, name=f"{prefix}_0")(y, train)
                elif scale == 0.25:
                    use_extra_pool = True
                    continue
                else:
                    raise NotImplementedError(f"Unsupported scale_factor:{scale}")
                fused_parts.append(y)

            fused = fused_parts[0] if len(fused_parts) == 1 else jnp.concatenate(fused_parts, axis=1)
            in_dim = int(sum(channel // max(1, scale) for channel in self.in_channels))
            # Upstream passes only ``layer_norm`` here; ``rms_norm`` applies to the other stages.
            fused = C2f(
                in_dim, self.out_channels, self.num_blocks, layer_norm=self.layer_norm,
                name=f"stages_{stage_index}_0",
            )(fused, train)
            fused = ChannelLayerNorm(self.out_channels, name=f"stages_{stage_index}_1")(fused)
            results.append(fused)

        if use_extra_pool:
            results.append(_max_pool_stride2(results[-1]))
        return results


def _max_pool_stride2(x: Array) -> Array:
    """Apply ``F.max_pool2d(x, kernel_size=1, stride=2)``: a plain 2x spatial decimation."""
    return x[:, :, ::2, ::2]


def scale_factors_for(projector_scale: Sequence[str]) -> list[float]:
    """Map ``["P3", "P4", ...]`` level names to their numeric scale factors."""
    return [_LEVEL_TO_SCALE[level] for level in projector_scale]
