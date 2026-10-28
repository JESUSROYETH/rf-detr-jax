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
"""RF-DETR transformer, ported from ``rfdetr.models.transformer``.

There is no encoder stack: the ViT backbone plays that role. This module holds the two-stage
query-selection head and the deformable decoder, with parameter names mirroring the PyTorch tree
(``decoder.layers.0.self_attn.in_proj_weight`` becomes ``decoder.layers_0.self_attn.in_proj_weight``).
"""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from rfdetr_jax.models.heads.keypoints import ConditionalQueryInitializer
from rfdetr_jax.models.layers import (
    MLP,
    LayerNorm,
    Linear,
    MultiheadAttention,
    ScopeCache,
    _on_gpu,
    grouped_layer_norm,
    grouped_linear,
    select_rows,
    stacked_layer_norm,
    stacked_linear,
)
from rfdetr_jax.models.ops import MSDeformAttn, precise_divide
from rfdetr_jax.models.position_encoding import gen_sineembed_for_position

Array = jax.Array


def gen_encoder_output_proposals(
    memory: Array,
    memory_padding_mask: Array | None,
    spatial_shapes: tuple[tuple[int, int], ...],
    unsigmoid: bool = True,
) -> tuple[Array, Array]:
    """Build per-position anchor proposals and mask out invalid encoder positions.

    Args:
        memory: Flattened multi-scale features ``[B, sum(H*W), C]``.
        memory_padding_mask: Optional ``[B, sum(H*W)]`` mask, ``True`` on padded positions.
        spatial_shapes: Per-level ``(H, W)`` pairs.
        unsigmoid: Whether to return proposals in logit space (``True``) or in ``[0, 1]``.

    Returns:
        A ``(output_memory, output_proposals)`` pair with shapes ``[B, sum(H*W), C]`` and
        ``[B, sum(H*W), 4]``.
    """
    batch_size = memory.shape[0]
    proposals = []
    cursor = 0
    for level, (height, width) in enumerate(spatial_shapes):
        if memory_padding_mask is not None:
            mask_level = memory_padding_mask[:, cursor : cursor + height * width].reshape(
                batch_size, height, width
            )
            valid_height = jnp.sum(~mask_level[:, :, 0], axis=1)
            valid_width = jnp.sum(~mask_level[:, 0, :], axis=1)
        else:
            valid_height = jnp.full((batch_size,), height, dtype=jnp.int32)
            valid_width = jnp.full((batch_size,), width, dtype=jnp.int32)

        # torch.linspace(0, n - 1, n) is exactly the integers; jnp.linspace is not (its step is
        # formed as a reciprocal product, 47 * (1 / 47) < 1 in float32), so index directly.
        grid_y, grid_x = jnp.meshgrid(
            jnp.arange(height, dtype=jnp.float32),
            jnp.arange(width, dtype=jnp.float32),
            indexing="ij",
        )
        grid = jnp.stack([grid_x, grid_y], axis=-1)
        scale = jnp.stack([valid_width, valid_height], axis=-1).reshape(batch_size, 1, 1, 2).astype(jnp.float32)
        proposals_grid = precise_divide(grid[None] + 0.5, scale)
        wh = jnp.full_like(proposals_grid, 0.05 * (2.0**level))
        proposals.append(jnp.concatenate([proposals_grid, wh], axis=-1).reshape(batch_size, -1, 4))
        cursor += height * width

    output_proposals = jnp.concatenate(proposals, axis=1)
    valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdims=True)

    if unsigmoid:
        output_proposals = jnp.log(output_proposals / (1 - output_proposals))
        if memory_padding_mask is not None:
            output_proposals = jnp.where(memory_padding_mask[..., None], jnp.inf, output_proposals)
        output_proposals = jnp.where(valid, output_proposals, jnp.inf)
    else:
        if memory_padding_mask is not None:
            output_proposals = jnp.where(memory_padding_mask[..., None], 0.0, output_proposals)
        output_proposals = jnp.where(valid, output_proposals, 0.0)

    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = jnp.where(memory_padding_mask[..., None], 0.0, output_memory)
    output_memory = jnp.where(valid, output_memory, 0.0)
    return output_memory.astype(memory.dtype), output_proposals.astype(memory.dtype)


def keypoint_class_mask(num_keypoints_per_class: Sequence[int]) -> Array:
    """Build the attention mask that blocks keypoints of one class attending to another's.

    The mask covers ``1 + total_keypoints`` tokens: the instance token followed by every class's
    keypoint tokens concatenated. ``True`` blocks a position, matching torch's boolean convention.
    """
    if not num_keypoints_per_class:
        return jnp.zeros((1, 1), dtype=bool)
    total = sum(num_keypoints_per_class)
    owner = np.zeros(1 + total, dtype=np.int64)
    owner[0] = -1
    cursor = 1
    for class_index, count in enumerate(num_keypoints_per_class):
        owner[cursor : cursor + count] = class_index
        cursor += count
    blocked = (owner[:, None] != owner[None, :]) & (owner[:, None] >= 0) & (owner[None, :] >= 0)
    return jnp.asarray(blocked)


def get_valid_ratio(mask: Array) -> Array:
    """Return the ``[B, 2]`` valid width/height fractions of a ``[B, H, W]`` padding mask."""
    _, height, width = mask.shape
    valid_height = jnp.sum(~mask[:, :, 0], axis=1)
    valid_width = jnp.sum(~mask[:, 0, :], axis=1)
    ratio_w = precise_divide(valid_width.astype(jnp.float32), width)
    ratio_h = precise_divide(valid_height.astype(jnp.float32), height)
    return jnp.stack([ratio_w, ratio_h], axis=-1)


class TransformerDecoderLayer(nn.Module):
    """One decoder layer: grouped self-attention, deformable cross-attention, and an FFN."""

    d_model: int
    sa_nhead: int
    ca_nhead: int
    dim_feedforward: int = 2048
    dropout: float = 0.0
    activation: str = "relu"
    group_detr: int = 1
    num_feature_levels: int = 4
    dec_n_points: int = 4
    deform_sampler: str = "gather"
    enable_keypoint_processing: bool = False
    grouppose_keypoint_dim_downscale: int = 1
    keypoint_cross_attn: bool = True
    inter_instance_kp_attn: bool = False

    @nn.compact
    def __call__(
        self,
        tgt: Array,
        memory: Array,
        query_pos: Array,
        reference_points: Array,
        spatial_shapes: tuple[tuple[int, int], ...],
        memory_key_padding_mask: Array | None = None,
        train: bool = False,
        keypoint_tgt: Array | None = None,
        keypoint_pos: Array | None = None,
        kp_class_mask: Array | None = None,
        kp_cross_attn_memory: Array | None = None,
    ) -> Array | tuple[Array, Array]:
        """Update ``tgt`` for one decoder layer.

        Args:
            tgt: Query features ``[B, Q, C]``.
            memory: Flattened backbone features ``[B, sum(H*W), C]``.
            query_pos: Query positional encoding ``[B, Q, C]``.
            reference_points: Deformable references ``[B, Q, n_levels, 2 or 4]``.
            spatial_shapes: Per-level ``(H, W)`` pairs.
            memory_key_padding_mask: Optional ``[B, sum(H*W)]`` padding mask.
            train: When ``True``, self-attention is split into ``group_detr`` independent groups.
            keypoint_tgt: GroupPose keypoint tokens ``[B, Q, K, kp_dim]``.
            keypoint_pos: Keypoint positional embedding, same shape as ``keypoint_tgt``.
            kp_class_mask: ``[1 + K, 1 + K]`` mask blocking cross-class keypoint attention.
            kp_cross_attn_memory: Optional second-projector memory for keypoint cross-attention.

        Returns:
            Updated query features ``[B, Q, C]``, or a ``(queries, keypoints)`` pair when the
            keypoint stream is enabled.

        Raises:
            ValueError: If the keypoint stream is enabled without its tokens or positions.
        """
        batch, num_queries, _ = tgt.shape
        query = key = tgt + query_pos
        value = tgt
        if train and self.group_detr > 1:
            per_group = num_queries // self.group_detr
            query = _split_groups(query, per_group)
            key = query
            value = _split_groups(value, per_group)

        attn = MultiheadAttention(self.d_model, self.sa_nhead, dropout=self.dropout, name="self_attn")(
            query, key, value, train=train)
        if train and self.group_detr > 1:
            attn = _merge_groups(attn, batch)
        tgt = LayerNorm(self.d_model, name="norm1")(tgt + _dropout(attn, self.dropout, train, self, "dropout1"))

        cross = MSDeformAttn(
            self.d_model,
            n_levels=self.num_feature_levels,
            n_heads=self.ca_nhead,
            n_points=self.dec_n_points,
            sampler=self.deform_sampler,
            name="cross_attn",
        )(tgt + query_pos, reference_points, memory, spatial_shapes, memory_key_padding_mask)
        tgt = LayerNorm(self.d_model, name="norm2")(tgt + _dropout(cross, self.dropout, train, self, "dropout2"))

        hidden = Linear(self.d_model, self.dim_feedforward, name="linear1")(tgt)
        hidden = _activation(self.activation)(hidden)
        hidden = _dropout(hidden, self.dropout, train, self, "dropout")
        hidden = Linear(self.dim_feedforward, self.d_model, name="linear2")(hidden)
        tgt = LayerNorm(self.d_model, name="norm3")(tgt + _dropout(hidden, self.dropout, train, self, "dropout3"))

        if not self.enable_keypoint_processing:
            return tgt
        if keypoint_tgt is None or keypoint_pos is None:
            raise ValueError("Keypoint processing is enabled but keypoint_tgt/keypoint_pos are missing")
        return self._keypoint_stream(
            tgt,
            memory,
            query_pos,
            reference_points,
            spatial_shapes,
            memory_key_padding_mask,
            train,
            keypoint_tgt,
            keypoint_pos,
            kp_class_mask,
            kp_cross_attn_memory,
        )

    def _keypoint_stream(
        self,
        tgt: Array,
        memory: Array,
        query_pos: Array,
        reference_points: Array,
        spatial_shapes: tuple[tuple[int, int], ...],
        memory_key_padding_mask: Array | None,
        train: bool,
        keypoint_tgt: Array,
        keypoint_pos: Array,
        kp_class_mask: Array | None,
        kp_cross_attn_memory: Array | None,
    ) -> tuple[Array, Array]:
        """Run the GroupPose keypoint branch and return the updated instance and keypoint tokens."""
        batch, num_queries, _ = tgt.shape
        downscale = self.grouppose_keypoint_dim_downscale
        kp_dim = self.d_model // downscale

        def project(x: Array, name: str, in_dim: int, out_dim: int) -> Array:
            """Apply an optional projection; identity when the keypoint dim is not downscaled."""
            return x if downscale == 1 else Linear(in_dim, out_dim, name=name)(x)

        tgt_for_kp = project(tgt, "inst_in_proj", self.d_model, kp_dim)
        tgt_for_kp_pos = project(query_pos, "inst_pos_in_proj", self.d_model, kp_dim)

        num_kp = keypoint_tgt.shape[2]
        combined_features = jnp.concatenate([tgt_for_kp[:, :, None], keypoint_tgt], axis=2)
        combined_positions = jnp.concatenate(
            [jnp.zeros_like(tgt_for_kp)[:, :, None], keypoint_pos], axis=2
        )
        combined_features = combined_features.reshape(batch * num_queries, 1 + num_kp, kp_dim)
        combined_positions = combined_positions.reshape(batch * num_queries, 1 + num_kp, kp_dim)

        attended = MultiheadAttention(
            kp_dim, max(1, self.sa_nhead // downscale), dropout=self.dropout, name="kp_inst_self_attn"
        )(
            combined_features + combined_positions,
            combined_features + combined_positions,
            combined_features,
            attn_mask=kp_class_mask,
            train=train,
        ).reshape(batch, num_queries, 1 + num_kp, kp_dim)

        layer_scale = self.param("instance_kp_layer_scale", nn.initializers.constant(1e-6), (1,))
        instance_update = project(attended[:, :, 0], "inst_out_proj", kp_dim, self.d_model)
        tgt = tgt + _dropout(instance_update, self.dropout, train, self, "kp_inst_dropout") * layer_scale
        tgt = LayerNorm(self.d_model, name="kp_inst_norm")(tgt)
        keypoint_tgt = keypoint_tgt + _dropout(attended[:, :, 1:], self.dropout, train, self, "kp_dropout")
        keypoint_tgt = LayerNorm(kp_dim, name="kp_norm")(keypoint_tgt)

        if self.inter_instance_kp_attn:
            swapped = keypoint_tgt.transpose(0, 2, 1, 3).reshape(batch * num_kp, num_queries, kp_dim)
            swapped_pos = jnp.broadcast_to(
                tgt_for_kp_pos[:, None], (batch, num_kp, num_queries, kp_dim)
            ).reshape(batch * num_kp, num_queries, kp_dim)
            query = swapped + swapped_pos
            attended_across = MultiheadAttention(
                kp_dim, max(1, self.ca_nhead // downscale), dropout=self.dropout, name="inter_inst_kp_attn"
            )(query, query, swapped, train=train)
            attended_across = attended_across.reshape(batch, num_kp, num_queries, kp_dim).transpose(0, 2, 1, 3)
            keypoint_tgt = keypoint_tgt + _dropout(
                attended_across, self.dropout, train, self, "inter_inst_kp_dropout"
            )
            keypoint_tgt = LayerNorm(kp_dim, name="inter_inst_kp_norm")(keypoint_tgt)

        if self.keypoint_cross_attn:
            keypoint_query = keypoint_tgt + jnp.broadcast_to(
                tgt_for_kp_pos[:, :, None], (batch, num_queries, num_kp, kp_dim)
            )
            keypoint_query = keypoint_query.reshape(batch, num_queries * num_kp, kp_dim)
            levels, coords = reference_points.shape[2], reference_points.shape[3]
            box_reference = jnp.broadcast_to(
                reference_points[:, :, None], (batch, num_queries, num_kp, levels, coords)
            ).reshape(batch, num_queries * num_kp, levels, coords)
            kp_memory = kp_cross_attn_memory if kp_cross_attn_memory is not None else memory
            sampled = MSDeformAttn(
                kp_dim,
                n_levels=self.num_feature_levels,
                n_heads=max(1, self.ca_nhead // downscale),
                n_points=self.dec_n_points,
                sampler=self.deform_sampler,
                name="kp_cross_attn",
            )(
                keypoint_query,
                box_reference,
                project(kp_memory, "memory_in_proj", self.d_model, kp_dim),
                spatial_shapes,
                memory_key_padding_mask,
            ).reshape(batch, num_queries, num_kp, kp_dim)
            keypoint_tgt = keypoint_tgt + _dropout(
                sampled, self.dropout, train, self, "kp_cross_attn_dropout"
            )
            keypoint_tgt = LayerNorm(kp_dim, name="kp_cross_attn_norm")(keypoint_tgt)

        hidden = Linear(kp_dim, self.d_model * 4 // downscale, name="kp_linear1")(keypoint_tgt)
        hidden = _activation(self.activation)(hidden)
        hidden = _dropout(hidden, self.dropout, train, self, "kp_dropout2")
        hidden = Linear(self.d_model * 4 // downscale, kp_dim, name="kp_linear3")(hidden)
        keypoint_tgt = keypoint_tgt + _dropout(hidden, self.dropout, train, self, "kp_dropout4")
        return tgt, LayerNorm(kp_dim, name="kp_norm5")(keypoint_tgt)


def _split_groups(x: Array, per_group: int) -> Array:
    """Move query groups onto the batch axis: ``[B, G * P, C] -> [G * B, P, C]``."""
    batch, total, channels = x.shape
    groups = total // per_group
    return x.reshape(batch, groups, per_group, channels).transpose(1, 0, 2, 3).reshape(groups * batch, per_group, -1)


def _merge_groups(x: Array, batch: int) -> Array:
    """Invert :func:`_split_groups`: ``[G * B, P, C] -> [B, G * P, C]``."""
    total, per_group, channels = x.shape
    groups = total // batch
    return x.reshape(groups, batch, per_group, channels).transpose(1, 0, 2, 3).reshape(batch, groups * per_group, -1)


def _dropout(x: Array, rate: float, train: bool, module: nn.Module, name: str) -> Array:
    """Apply dropout with the given rate, drawing from the ``dropout`` RNG stream when training."""
    if rate == 0.0:
        return x
    return nn.Dropout(rate=rate, deterministic=not train, name=name)(x)


def _activation(name: str):
    """Return the decoder FFN activation."""
    if name == "relu":
        return jax.nn.relu
    if name == "gelu":
        return lambda x: jax.nn.gelu(x, approximate=False)
    raise RuntimeError(f"activation should be relu/gelu, not {name}.")


class TransformerDecoder(nn.Module):
    """The decoder stack with reference-point refinement."""

    num_layers: int
    d_model: int
    sa_nhead: int
    ca_nhead: int
    dim_feedforward: int
    dropout: float
    activation: str
    group_detr: int
    num_feature_levels: int
    dec_n_points: int
    lite_refpoint_refine: bool
    bbox_reparam: bool
    deform_sampler: str = "gather"
    decoder_norm_type: str = "LN"
    gradient_checkpointing: bool = False
    enable_keypoint_processing: bool = False
    num_keypoints_per_class: tuple[int, ...] = ()
    grouppose_keypoint_dim_downscale: int = 1
    keypoint_cross_attn: bool = True
    inter_instance_kp_attn: bool = False

    @nn.compact
    def __call__(
        self,
        tgt: Array,
        memory: Array,
        refpoints_unsigmoid: Array,
        spatial_shapes: tuple[tuple[int, int], ...],
        valid_ratios: Array | None,
        memory_key_padding_mask: Array | None = None,
        bbox_embed: nn.Module | None = None,
        train: bool = False,
        keypoint_tgt: Array | None = None,
        kp_cross_attn_memory: Array | None = None,
    ) -> tuple[Array, Array] | tuple[Array, Array, Array]:
        """Run every decoder layer and return the stacked outputs and reference points.

        Args:
            tgt: Initial query features ``[B, Q, C]``.
            memory: Flattened backbone features ``[B, sum(H*W), C]``.
            refpoints_unsigmoid: Initial references ``[B, Q, 4]``.
            spatial_shapes: Per-level ``(H, W)`` pairs.
            valid_ratios: ``[B, n_levels, 2]`` valid fractions, or ``None`` when unpadded.
            memory_key_padding_mask: Optional ``[B, sum(H*W)]`` padding mask.
            bbox_embed: Shared box MLP, used only when ``lite_refpoint_refine`` is ``False``.
            train: Training flag, forwarded to dropout and grouped attention.
            keypoint_tgt: Initial GroupPose keypoint tokens ``[B, Q, K, kp_dim]``.
            kp_cross_attn_memory: Optional second-projector memory for keypoint cross-attention.

        Returns:
            ``(intermediate, references)`` stacked over layers, with shapes ``[L, B, Q, C]`` and
            ``[L, B, Q, 4]`` (``[1, B, Q, 4]`` when references are not refined per layer). With the
            keypoint stream enabled, the stacked keypoint tokens are appended as a third element.

        Raises:
            ValueError: If the keypoint stream is enabled without initial keypoint tokens.
        """
        norm = LayerNorm(self.d_model, name="norm") if self.decoder_norm_type == "LN" else None
        ref_point_head = MLP(2 * self.d_model, self.d_model, self.d_model, 2, name="ref_point_head")

        def get_reference(refpoints: Array) -> tuple[Array, Array]:
            """Return per-level reference points and the query positional encoding."""
            obj_center = refpoints[..., :4]
            if valid_ratios is None:
                refpoints_input = jnp.broadcast_to(
                    obj_center[:, :, None], (*obj_center.shape[:2], self.num_feature_levels, 4)
                )
            else:
                ratios = jnp.concatenate([valid_ratios, valid_ratios], axis=-1)[:, None]
                refpoints_input = obj_center[:, :, None] * ratios
            query_sine_embed = gen_sineembed_for_position(refpoints_input[:, :, 0, :], self.d_model // 2)
            return refpoints_input, ref_point_head(query_sine_embed.astype(tgt.dtype))

        # Grouped self-attention materialises `group_detr * batch` attention matrices per layer,
        # which dominates decoder memory; recomputing them in the backward pass trades a second
        # forward for that peak.
        # `static_argnums` counts `self` as index 0, so 5 is `spatial_shapes` (a tuple of Python
        # ints used for reshapes) and 7 is `train`. Both must stay static or remat traces them.
        layer_class = (
            nn.remat(TransformerDecoderLayer, static_argnums=(5, 7))
            if self.gradient_checkpointing
            else TransformerDecoderLayer
        )

        kp_query_pos = None
        kp_mask = None
        intermediate_keypoints: list[Array] = []
        if self.enable_keypoint_processing:
            if keypoint_tgt is None:
                raise ValueError("Keypoint processing is enabled but keypoint_tgt was not provided")
            kp_dim = self.d_model // self.grouppose_keypoint_dim_downscale
            total_keypoints = sum(self.num_keypoints_per_class)
            keypoint_pos_embed = self.param(
                "keypoint_pos_embed", nn.initializers.normal(1.0), (total_keypoints, kp_dim)
            )
            kp_query_pos = jnp.broadcast_to(
                keypoint_pos_embed[None, None], (*keypoint_tgt.shape[:2], total_keypoints, kp_dim)
            )
            kp_mask = keypoint_class_mask(self.num_keypoints_per_class)

        output = tgt
        intermediate = []
        hs_refpoints = [refpoints_unsigmoid]

        if self.lite_refpoint_refine:
            source = refpoints_unsigmoid if self.bbox_reparam else jax.nn.sigmoid(refpoints_unsigmoid)
            refpoints_input, query_pos = get_reference(source)

        for layer_id in range(self.num_layers):
            if not self.lite_refpoint_refine:
                source = refpoints_unsigmoid if self.bbox_reparam else jax.nn.sigmoid(refpoints_unsigmoid)
                refpoints_input, query_pos = get_reference(source)

            layer_output = layer_class(
                d_model=self.d_model,
                sa_nhead=self.sa_nhead,
                ca_nhead=self.ca_nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation=self.activation,
                group_detr=self.group_detr,
                num_feature_levels=self.num_feature_levels,
                dec_n_points=self.dec_n_points,
                deform_sampler=self.deform_sampler,
                enable_keypoint_processing=self.enable_keypoint_processing,
                grouppose_keypoint_dim_downscale=self.grouppose_keypoint_dim_downscale,
                keypoint_cross_attn=self.keypoint_cross_attn,
                inter_instance_kp_attn=self.inter_instance_kp_attn,
                name=f"layers_{layer_id}",
            )(
                output,
                memory,
                query_pos,
                refpoints_input,
                spatial_shapes,
                memory_key_padding_mask,
                train,
                keypoint_tgt,
                kp_query_pos,
                kp_mask,
                kp_cross_attn_memory,
            )
            if self.enable_keypoint_processing:
                output, keypoint_tgt = layer_output
                intermediate_keypoints.append(keypoint_tgt)
            else:
                output = layer_output

            if not self.lite_refpoint_refine:
                assert bbox_embed is not None, "iterative refinement requires a bbox_embed"
                delta = bbox_embed(output)
                refined = refpoints_refine(refpoints_unsigmoid, delta, self.bbox_reparam)
                if layer_id != self.num_layers - 1:
                    hs_refpoints.append(refined)
                refpoints_unsigmoid = jax.lax.stop_gradient(refined)

            intermediate.append(norm(output) if norm is not None else output)

        if norm is not None:
            output = norm(output)
            intermediate[-1] = output

        stacked = jnp.stack(intermediate)
        references = jnp.stack(hs_refpoints) if not self.lite_refpoint_refine else refpoints_unsigmoid[None]
        if self.enable_keypoint_processing:
            return stacked, references, jnp.stack(intermediate_keypoints)
        return stacked, references


def refpoints_refine(refpoints_unsigmoid: Array, delta: Array, bbox_reparam: bool) -> Array:
    """Apply one refinement step to reference points."""
    if bbox_reparam:
        if _on_gpu():
            delta = delta.astype(jnp.float32)
        cxcy = delta[..., :2] * refpoints_unsigmoid[..., 2:] + refpoints_unsigmoid[..., :2]
        wh = jnp.exp(delta[..., 2:]) * refpoints_unsigmoid[..., 2:]
        return jnp.concatenate([cxcy, wh], axis=-1)
    return refpoints_unsigmoid + delta


class Transformer(nn.Module):
    """Two-stage query selection followed by the deformable decoder."""

    d_model: int = 256
    sa_nhead: int = 8
    ca_nhead: int = 8
    num_queries: int = 300
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.0
    activation: str = "relu"
    group_detr: int = 1
    two_stage: bool = True
    num_feature_levels: int = 1
    dec_n_points: int = 4
    lite_refpoint_refine: bool = True
    decoder_norm_type: str = "LN"
    bbox_reparam: bool = True
    num_classes: int = 91
    gradient_checkpointing: bool = False
    deform_sampler: str = "gather"
    use_grouppose_keypoints: bool = False
    num_keypoints_per_class: tuple[int, ...] = ()
    grouppose_keypoint_dim_downscale: int = 1
    keypoint_cross_attn: bool = True
    inter_instance_kp_attn: bool = False
    num_registers: int = 0
    dual_projector_kp_only: bool = False

    @nn.compact
    def __call__(
        self,
        srcs: list[Array],
        masks: list[Array] | None,
        pos_embeds: list[Array],
        refpoint_embed: Array,
        query_feat: Array,
        bbox_embed: nn.Module | None = None,
        train: bool = False,
        cross_attn_srcs: list[Array] | None = None,
        proposal_indices: Array | None = None,
    ) -> dict[str, Array | None]:
        """Run query selection and the decoder.

        ``proposal_indices`` (``[groups, batch, num_queries]``) replaces the two-stage top-k
        selection with a given one. Parity checks pass upstream's selection so that framework
        specific ordering of tied proposal scores cannot mask a genuine difference. The selection
        actually used and the per-position scores it ranks are returned as ``proposal_indices``
        and ``proposal_scores``.

        Args:
            srcs: Per-level NCHW feature maps.
            masks: Per-level ``[B, H, W]`` padding masks, or ``None`` when unpadded.
            pos_embeds: Per-level NCHW sine position embeddings.
            refpoint_embed: Learned reference embedding ``[Q_total, 4]``.
            query_feat: Learned query features ``[Q_total, C]``.
            bbox_embed: The shared box head, passed in only for non-lite iterative refinement.
            train: Training flag; controls group-DETR expansion and dropout.
            cross_attn_srcs: Optional second-projector feature maps for keypoint cross-attention.

        Returns:
            Dict with ``hs`` ``[L, B, Q, C]``, ``references`` ``[L or 1, B, Q, 4]``, and the
            two-stage ``hs_enc`` / ``ref_enc`` / ``enc_class_logits`` tensors (``None`` when
            ``two_stage`` is off).
        """
        spatial_shapes = tuple((int(src.shape[2]), int(src.shape[3])) for src in srcs)
        memory = jnp.concatenate([src.reshape(*src.shape[:2], -1).transpose(0, 2, 1) for src in srcs], axis=1)
        # The level position embeddings are computed and threaded through upstream but never read:
        # `TransformerDecoder.forward` accepts `pos` and never uses it (the decoder's positional
        # signal is `query_pos`, derived from the reference points). Dropped here rather than
        # computed and discarded.
        del pos_embeds

        cross_attn_memory = None
        if cross_attn_srcs is not None:
            cross_attn_memory = jnp.concatenate(
                [src.reshape(*src.shape[:2], -1).transpose(0, 2, 1) for src in cross_attn_srcs], axis=1
            )

        mask_flatten = None
        valid_ratios = None
        if masks is not None:
            mask_flatten = jnp.concatenate([mask.reshape(mask.shape[0], -1) for mask in masks], axis=1)
            valid_ratios = jnp.stack([get_valid_ratio(mask) for mask in masks], axis=1)

        group_detr = self.group_detr if train else 1
        batch = memory.shape[0]

        hs_enc = None
        ref_enc = None
        enc_class_logits = None
        proposal_scores = None
        refpoint_embed_ts = None
        if self.two_stage:
            # CUDA autocast promotes the upstream projector's final LayerNorm to
            # FP32, so its coordinate proposals are FP32 too. Keep the eligible
            # encoder head matmuls at the activation dtype independently of geometry.
            proposal_memory = memory.astype(jnp.float32) if _on_gpu() else memory
            output_memory, output_proposals = gen_encoder_output_proposals(
                proposal_memory, mask_flatten, spatial_shapes, unsigmoid=not self.bbox_reparam
            )
            output_memory = output_memory.astype(memory.dtype)
            # Every group runs the same shapes over the same encoder memory, so the whole
            # query-selection head is one batched einsum per stage rather than `group_detr` small
            # ones. The parameters are still read through their own per-group scopes, so the tree a
            # checkpoint sees is unchanged.
            cache = ScopeCache(self.scope)
            groups = range(group_detr)
            project_w, project_b = stacked_linear(
                cache, [f"enc_output_{g}" for g in groups], self.d_model, self.d_model
            )
            norm_w, norm_b = stacked_layer_norm(
                cache, [f"enc_output_norm_{g}" for g in groups], self.d_model
            )
            class_w, class_b = stacked_linear(
                cache, [f"enc_out_class_embed_{g}" for g in groups], self.d_model, self.num_classes
            )
            bbox_layers = [
                stacked_linear(
                    cache,
                    [f"enc_out_bbox_embed_{g}/layers_{index}" for g in groups],
                    self.d_model,
                    self.d_model if index < 2 else 4,
                )
                for index in range(3)
            ]

            projected = grouped_linear(output_memory, project_w, project_b, grouped_input=False)
            projected = grouped_layer_norm(projected, norm_w, norm_b)
            logits = grouped_linear(projected, class_w, class_b)

            topk = min(self.num_queries, logits.shape[-2])
            proposal_scores = logits.max(-1)
            _, topk_indices = jax.lax.top_k(proposal_scores, topk)
            if proposal_indices is not None:
                if proposal_indices.shape != topk_indices.shape:
                    raise ValueError(f"proposal_indices must have shape {topk_indices.shape}")
                topk_indices = proposal_indices.astype(topk_indices.dtype)
            proposal_indices = topk_indices
            selected_memory = select_rows(projected, topk_indices)
            selected_proposals = select_rows(
                jnp.broadcast_to(output_proposals[None], (group_detr, *output_proposals.shape)),
                topk_indices,
            )
            # The same per-group head scores the selected queries, matching how LWDETR reuses
            # transformer.enc_out_class_embed[g] on memory_ts after the decoder runs.
            selected_logits = grouped_linear(selected_memory, class_w, class_b)

            delta = selected_memory
            for index, (weight, bias) in enumerate(bbox_layers):
                delta = grouped_linear(delta, weight, bias)
                if index < 2:
                    delta = jax.nn.relu(delta)

            if self.bbox_reparam:
                # exp is an FP32 autocast operation on CUDA, unlike CPU autocast.
                if _on_gpu():
                    delta = delta.astype(jnp.float32)
                cxcy = delta[..., :2] * selected_proposals[..., 2:] + selected_proposals[..., :2]
                wh = jnp.exp(delta[..., 2:]) * selected_proposals[..., 2:]
                boxes = jnp.concatenate([cxcy, wh], axis=-1)
            else:
                boxes = delta + selected_proposals

            def ungroup(x: Array) -> Array:
                """Fold the group axis back onto the query axis, in `concatenate` order."""
                return x.transpose(1, 0, 2, 3).reshape(x.shape[1], group_detr * x.shape[2], x.shape[3])

            refpoint_embed_ts = jax.lax.stop_gradient(ungroup(boxes))
            hs_enc = ungroup(selected_memory)
            boxes_ts = ungroup(boxes)
            enc_class_logits = ungroup(selected_logits)
            ref_enc = boxes_ts if self.bbox_reparam else jax.nn.sigmoid(boxes_ts)

        total_keypoints = sum(self.num_keypoints_per_class)
        enc_keypoints = None
        init_kp_ref_xy = None
        if self.use_grouppose_keypoints and self.two_stage and total_keypoints > 0:
            keypoint_memory = ConditionalQueryInitializer(
                self.d_model,
                total_keypoints,
                out_dim=self.d_model // self.grouppose_keypoint_dim_downscale,
                name="keypoint_query_initializer_enc",
            )(hs_enc)
            boxes_reference = ref_enc
            keypoint_parts = []
            for group_index in range(group_detr):
                width = keypoint_memory.shape[1] // group_detr
                window = slice(group_index * width, (group_index + 1) * width)
                # `Transformer.__init__` builds these as `MLP(kp_dim, d_model, kp_dim, 2)`, but
                # `LWDETR.__init__` then overwrites them with deep copies of its own
                # `keypoint_embed`, so the shape a checkpoint actually carries is this one.
                delta = MLP(
                    self.d_model // self.grouppose_keypoint_dim_downscale,
                    self.d_model // self.grouppose_keypoint_dim_downscale,
                    8,
                    3,
                    name=f"enc_out_keypoint_embed_{group_index}",
                )(keypoint_memory[:, window])
                delta = jnp.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                reference = boxes_reference[:, window]
                keypoint_xy = delta[..., :2] * reference[..., None, 2:] + reference[..., None, :2]
                keypoint_parts.append(jnp.concatenate([keypoint_xy, delta[..., 2:]], axis=-1))
            enc_keypoints = jnp.concatenate(keypoint_parts, axis=1)
            init_kp_ref_xy = jax.lax.stop_gradient(enc_keypoints[..., :2])

        tgt = jnp.broadcast_to(query_feat[None], (batch, *query_feat.shape))
        refpoints = jnp.broadcast_to(refpoint_embed[None], (batch, *refpoint_embed.shape))

        if self.two_stage:
            ts_len = refpoint_embed_ts.shape[-2]
            learned = refpoints[:, :ts_len]
            trailing = refpoints[:, ts_len:]
            if self.bbox_reparam:
                cxcy = learned[..., :2] * refpoint_embed_ts[..., 2:] + refpoint_embed_ts[..., :2]
                wh = jnp.exp(learned[..., 2:]) * refpoint_embed_ts[..., 2:]
                learned = jnp.concatenate([cxcy, wh], axis=-1)
            else:
                learned = learned + refpoint_embed_ts
            refpoints = jnp.concatenate([learned, trailing], axis=-2)

        keypoint_tgt = None
        if self.use_grouppose_keypoints and total_keypoints > 0:
            keypoint_tgt = ConditionalQueryInitializer(
                self.d_model,
                total_keypoints,
                out_dim=self.d_model // self.grouppose_keypoint_dim_downscale,
                name="keypoint_query_initializer",
            )(tgt)

        decoder_memory = memory
        kp_cross_attn_memory = None
        if cross_attn_memory is not None:
            if self.dual_projector_kp_only:
                kp_cross_attn_memory = cross_attn_memory
            else:
                decoder_memory = cross_attn_memory

        decoder_outputs = TransformerDecoder(
            num_layers=self.num_decoder_layers,
            d_model=self.d_model,
            sa_nhead=self.sa_nhead,
            ca_nhead=self.ca_nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            group_detr=self.group_detr,
            num_feature_levels=self.num_feature_levels,
            dec_n_points=self.dec_n_points,
            lite_refpoint_refine=self.lite_refpoint_refine,
            bbox_reparam=self.bbox_reparam,
            deform_sampler=self.deform_sampler,
            decoder_norm_type=self.decoder_norm_type,
            gradient_checkpointing=self.gradient_checkpointing,
            enable_keypoint_processing=self.use_grouppose_keypoints,
            num_keypoints_per_class=self.num_keypoints_per_class,
            grouppose_keypoint_dim_downscale=self.grouppose_keypoint_dim_downscale,
            keypoint_cross_attn=self.keypoint_cross_attn,
            inter_instance_kp_attn=self.inter_instance_kp_attn,
            name="decoder",
        )(
            tgt,
            decoder_memory,
            refpoints,
            spatial_shapes,
            valid_ratios,
            mask_flatten,
            bbox_embed=None if self.lite_refpoint_refine else bbox_embed,
            train=train,
            keypoint_tgt=keypoint_tgt,
            kp_cross_attn_memory=kp_cross_attn_memory,
        )
        if self.use_grouppose_keypoints:
            hs, references, keypoint_hs = decoder_outputs
        else:
            hs, references = decoder_outputs
            keypoint_hs = None
        del init_kp_ref_xy  # upstream threads this only into the keypoint reference bookkeeping
        return {
            "hs": hs,
            "references": references,
            "hs_enc": hs_enc,
            "ref_enc": ref_enc,
            "enc_class_logits": enc_class_logits,
            "proposal_indices": proposal_indices if self.two_stage else None,
            "proposal_scores": proposal_scores,
            "keypoint_hs": keypoint_hs,
            "enc_keypoints": enc_keypoints,
        }
