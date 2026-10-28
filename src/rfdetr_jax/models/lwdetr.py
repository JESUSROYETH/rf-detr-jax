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
"""The RF-DETR (LW-DETR) detector, ported from ``rfdetr.models.lwdetr``."""

from __future__ import annotations

import math
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from rfdetr_jax.models.backbone.backbone import build_backbone, downsample_mask
from rfdetr_jax.models.heads.keypoints import active_keypoint_mask
from rfdetr_jax.models.heads.segmentation import SegmentationHead
from rfdetr_jax.models.layers import MLP, Linear
from rfdetr_jax.models.transformer import Transformer

Array = jax.Array


@dataclass(frozen=True)
class LWDETRSpec:
    """Everything :func:`build_model` needs to instantiate a detector.

    Field names mirror the RF-DETR ``ModelConfig`` fields they come from, so a variant config maps
    across one-to-one.
    """

    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 16
    num_windows: int = 2
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: tuple[str, ...] = ("P4",)
    out_feature_indexes: tuple[int, ...] = (3, 6, 9, 12)
    resolution: int = 512
    positional_encoding_size: int = 32
    num_classes: int = 90
    group_detr: int = 13
    two_stage: bool = True
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    rms_norm: bool = False
    dim_feedforward: int = 2048
    dropout: float = 0.0
    drop_path: float = 0.0
    decoder_norm: str = "LN"
    aux_loss: bool = True
    dual_projector: bool = False
    segmentation_head: bool = False
    mask_downsample_ratio: int = 4
    gradient_checkpointing: bool = False
    deform_sampler: str = "gather"
    compute_dtype: str = "float32"
    # "activation" runs detection/keypoint head matmuls and segmentation convolutions/matmuls
    # in compute_dtype while retaining FP32 coordinate and residual boundaries.
    # "float32" preserves the initial port's full-precision decoder-head policy.
    head_dtype: str = "float32"
    use_grouppose_keypoints: bool = False
    num_keypoints_per_class: tuple[int, ...] = ()
    grouppose_keypoint_dim_downscale: int = 1
    keypoint_cross_attn: bool = True
    inter_instance_kp_attn: bool = False
    dual_projector_kp_only: bool = False

    @property
    def activation_dtype(self) -> jnp.dtype:
        """The dtype activations flow in.

        Parameters stay float32 — every layer casts its weights to the activation dtype — so this
        is mixed precision, not a low-precision checkpoint. Normalisations and softmaxes already
        promote to float32 internally, and the detector's outputs are cast back before the loss, so
        only the wide intermediate tensors change.
        """
        return jnp.dtype(self.compute_dtype)

    @property
    def num_feature_levels(self) -> int:
        """Number of projector output levels consumed by the decoder."""
        return len(self.projector_scale)

    @property
    def keypoint_dim(self) -> int:
        """Width of the keypoint token stream."""
        return self.hidden_dim // self.grouppose_keypoint_dim_downscale

    @property
    def detector_classes(self) -> int:
        """``num_classes + 1``: RF-DETR reserves one extra slot, as the upstream builder does."""
        return self.num_classes + 1


class LWDETR(nn.Module):
    """RF-DETR detector: DINOv2 backbone, multi-scale projector, and a deformable decoder."""

    spec: LWDETRSpec

    def setup(self) -> None:
        """Create the backbone, transformer, heads and learned query parameters."""
        spec = self.spec
        self.backbone_0 = build_backbone(
            encoder=spec.encoder,
            out_feature_indexes=list(spec.out_feature_indexes),
            projector_scale=list(spec.projector_scale),
            hidden_dim=spec.hidden_dim,
            patch_size=spec.patch_size,
            num_windows=spec.num_windows,
            positional_encoding_size=spec.positional_encoding_size,
            layer_norm=spec.layer_norm,
            rms_norm=spec.rms_norm,
            dual_projector=spec.dual_projector,
            drop_path=spec.drop_path,
            gradient_checkpointing=spec.gradient_checkpointing,
        )
        self.transformer = Transformer(
            d_model=spec.hidden_dim,
            sa_nhead=spec.sa_nheads,
            ca_nhead=spec.ca_nheads,
            num_queries=spec.num_queries,
            num_decoder_layers=spec.dec_layers,
            dim_feedforward=spec.dim_feedforward,
            dropout=spec.dropout,
            group_detr=spec.group_detr,
            two_stage=spec.two_stage,
            num_feature_levels=spec.num_feature_levels,
            dec_n_points=spec.dec_n_points,
            lite_refpoint_refine=spec.lite_refpoint_refine,
            decoder_norm_type=spec.decoder_norm,
            bbox_reparam=spec.bbox_reparam,
            num_classes=spec.detector_classes,
            gradient_checkpointing=spec.gradient_checkpointing,
            deform_sampler=spec.deform_sampler,
            use_grouppose_keypoints=spec.use_grouppose_keypoints,
            num_keypoints_per_class=spec.num_keypoints_per_class,
            grouppose_keypoint_dim_downscale=spec.grouppose_keypoint_dim_downscale,
            keypoint_cross_attn=spec.keypoint_cross_attn,
            inter_instance_kp_attn=spec.inter_instance_kp_attn,
            dual_projector_kp_only=spec.dual_projector_kp_only,
        )
        self.segmentation_head = (
            SegmentationHead(
                spec.hidden_dim,
                spec.dec_layers,
                downsample_ratio=spec.mask_downsample_ratio,
                compute_dtype=spec.compute_dtype if spec.head_dtype == "activation" else None,
                name="segmentation_head",
            )
            if spec.segmentation_head
            else None
        )
        self.class_embed = Linear(spec.hidden_dim, spec.detector_classes, name="class_embed")
        self.bbox_embed = MLP(spec.hidden_dim, spec.hidden_dim, 4, 3, name="bbox_embed")
        self.keypoint_embed = (
            MLP(spec.keypoint_dim, spec.keypoint_dim, 8, 3, name="keypoint_embed")
            if spec.use_grouppose_keypoints
            else None
        )
        total_queries = spec.num_queries * spec.group_detr
        self.refpoint_embed = self.param("refpoint_embed", nn.initializers.zeros, (total_queries, 4))
        self.query_feat = self.param(
            "query_feat", nn.initializers.normal(1.0), (total_queries, spec.hidden_dim)
        )

    def __call__(
        self,
        pixel_values: Array,
        pixel_mask: Array | None = None,
        train: bool = False,
        proposal_indices: Array | None = None,
    ) -> dict:
        """Run the detector.

        Args:
            pixel_values: Normalized image batch ``[B, 3, H, W]``.
            pixel_mask: Optional ``[B, H, W]`` padding mask, ``True`` on padded pixels. ``None``
                means no padding, which skips the valid-ratio rescaling exactly as the upstream
                export path does.
            train: Training flag; enables group-DETR self-attention, dropout and drop-path.
            proposal_indices: Optional two-stage selection override ``[groups, B, num_queries]``
                for parity checks; see :meth:`Transformer.__call__`.

        Returns:
            Dict with ``pred_logits`` ``[B, Q, C]``, ``pred_boxes`` ``[B, Q, 4]``, and (when
            configured) ``aux_outputs`` and ``enc_outputs``.
        """
        spec = self.spec
        image_size = (pixel_values.shape[2], pixel_values.shape[3])
        pixel_values = pixel_values.astype(spec.activation_dtype)
        features, cross_attn_features = self.backbone_0(pixel_values, train=train)

        masks = None
        if pixel_mask is not None:
            masks = [downsample_mask(pixel_mask, (feat.shape[2], feat.shape[3])) for feat in features]

        group_detr = spec.group_detr if train else 1
        active_queries = spec.num_queries * group_detr
        # Reference embeddings participate in geometry, not an autocast-eligible matmul.
        refpoint_embed = self.refpoint_embed[:active_queries]
        query_feat = self.query_feat[:active_queries].astype(spec.activation_dtype)

        outputs = self.transformer(
            features,
            masks,
            [],
            refpoint_embed,
            query_feat,
            bbox_embed=None if spec.lite_refpoint_refine else self.bbox_embed,
            train=train,
            cross_attn_srcs=cross_attn_features,
            proposal_indices=proposal_indices,
        )
        selection = {key: outputs.pop(key) for key in ("proposal_indices", "proposal_scores")}

        # Back to float32 at the transformer's boundary: the heads below turn these into box
        # coordinates through an exponential, and bf16 there costs more than a pixel at 384 px.
        activation_outputs = outputs
        hs_activation = outputs["hs"].astype(spec.activation_dtype)
        outputs = {
            name: None if value is None else jnp.asarray(value).astype(
                jnp.promote_types(jnp.asarray(value).dtype, jnp.float32))
            for name, value in outputs.items()
        }
        hs = outputs["hs"]
        # The transformer's outputs go back to float32 because the box heads end in exp/sigmoid.
        # ``head_dtype="activation"`` keeps the heads' matmuls in the activation dtype, as torch
        # autocast does, and promotes only their outputs.
        if spec.head_dtype not in ("float32", "activation"):
            raise ValueError(f"unknown head_dtype {spec.head_dtype!r}; expected 'float32' or 'activation'")
        head_input = hs_activation if spec.head_dtype == "activation" else hs
        references = outputs["references"]
        head_output_dtype = jnp.promote_types(references.dtype, jnp.float32)
        if spec.bbox_reparam:
            delta = self.bbox_embed(head_input).astype(head_output_dtype)
            coord_cxcy = delta[..., :2] * references[..., 2:] + references[..., :2]
            coord_wh = jnp.exp(delta[..., 2:]) * references[..., 2:]
            outputs_coord = jnp.concatenate([coord_cxcy, coord_wh], axis=-1)
        else:
            outputs_coord = jax.nn.sigmoid(self.bbox_embed(head_input).astype(head_output_dtype) + references)
        outputs_class = self.class_embed(head_input).astype(head_output_dtype)

        outputs_keypoints = None
        if self.keypoint_embed is not None:
            keypoint_hs = (activation_outputs["keypoint_hs"].astype(spec.activation_dtype)
                           if spec.head_dtype == "activation" else outputs["keypoint_hs"])
            outputs_keypoints = self._keypoint_predictions(keypoint_hs, references)
            outputs_class = outputs_class + self._keypoint_class_boost(outputs_keypoints)

        outputs_masks = None
        if self.segmentation_head is not None:
            outputs_masks = self.segmentation_head(features[0], list(hs), image_size)

        result: dict = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if selection["proposal_indices"] is not None:
            result["proposal_indices"] = selection["proposal_indices"]
            result["proposal_scores"] = selection["proposal_scores"]
        if outputs_masks is not None:
            result["pred_masks"] = outputs_masks[-1]
        if outputs_keypoints is not None:
            result["pred_keypoints"] = outputs_keypoints[-1]
        if spec.aux_loss:
            result["aux_outputs"] = [
                {
                    "pred_logits": outputs_class[index],
                    "pred_boxes": outputs_coord[index],
                    **({"pred_masks": outputs_masks[index]} if outputs_masks is not None else {}),
                    **(
                        {"pred_keypoints": outputs_keypoints[index]}
                        if outputs_keypoints is not None
                        else {}
                    ),
                }
                for index in range(outputs_class.shape[0] - 1)
            ]
        if spec.two_stage:
            enc: dict = {
                "pred_logits": outputs["enc_class_logits"],
                "pred_boxes": outputs["ref_enc"],
            }
            if self.segmentation_head is not None:
                enc["pred_masks"] = self.segmentation_head(
                    features[0], [outputs["hs_enc"]], image_size, skip_blocks=True
                )[0]
            if outputs["enc_keypoints"] is not None:
                enc_keypoints = self._pad_keypoints(outputs["enc_keypoints"])
                enc["pred_keypoints"] = enc_keypoints
                enc["pred_logits"] = enc["pred_logits"] + self._keypoint_class_boost(enc_keypoints)
            result["enc_outputs"] = enc
        return result

    def _keypoint_predictions(self, keypoint_hs: Array, references: Array) -> Array:
        """Turn decoder keypoint tokens into class-padded keypoint predictions.

        Args:
            keypoint_hs: Per-layer keypoint tokens ``[L, B, Q, K, kp_dim]``.
            references: Reference boxes ``[L or 1, B, Q, 4]`` the offsets are relative to.

        Returns:
            Keypoints ``[L, B, Q, num_classes * max_keypoints, 8]``.
        """
        delta = self.keypoint_embed(keypoint_hs)
        # The box head shares `references`; a non-finite delta would poison its gradient through
        # this multiply even if a downstream guard zeroed the value, so sanitise at the source.
        delta = jnp.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        reference_wh = references[..., None, 2:]
        reference_xy = references[..., None, :2]
        keypoints_xy = delta[..., :2] * reference_wh + reference_xy
        compact = jnp.concatenate([keypoints_xy, delta[..., 2:]], axis=-1)
        return self._pad_keypoints(compact)

    def _pad_keypoints(self, compact: Array) -> Array:
        """Expand compact per-class keypoints into the per-class padded layout."""
        schema = self.spec.num_keypoints_per_class
        if not schema:
            return compact
        max_keypoints = max(schema)
        if compact.shape[-2] == len(schema) * max_keypoints:
            return compact
        blocks = []
        cursor = 0
        for count in schema:
            block = compact[..., cursor : cursor + count, :]
            padding = [(0, 0)] * (compact.ndim - 2) + [(0, max_keypoints - count), (0, 0)]
            blocks.append(jnp.pad(block, padding))
            cursor += count
        return jnp.concatenate(blocks, axis=-2)

    def _keypoint_class_boost(self, keypoints: Array) -> Array:
        """Aggregate per-keypoint class logits into a detection-class logit boost."""
        schema = self.spec.num_keypoints_per_class
        if not schema:
            return jnp.zeros((*keypoints.shape[:-2], self.spec.detector_classes), keypoints.dtype)
        max_keypoints = max(schema)
        contributions = keypoints[..., 7].reshape(*keypoints.shape[:-2], len(schema), max_keypoints)
        active = active_keypoint_mask(schema).astype(contributions.dtype)
        boost = (contributions * active).sum(-1)
        pad = self.spec.detector_classes - boost.shape[-1]
        if pad > 0:
            boost = jnp.pad(boost, [(0, 0)] * (boost.ndim - 1) + [(0, pad)])
        elif pad < 0:
            boost = boost[..., : self.spec.detector_classes]
        return boost


def build_model(spec: LWDETRSpec) -> LWDETR:
    """Instantiate an :class:`LWDETR` from a spec."""
    return LWDETR(spec=spec)


def initial_class_bias(num_classes: int, prior_prob: float = 0.01) -> float:
    """Return the focal-loss prior bias RF-DETR initialises the classification head with."""
    return -math.log((1 - prior_prob) / prior_prob)
