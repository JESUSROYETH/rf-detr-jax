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
"""Set-prediction criterion, ported from ``rfdetr.models.criterion``.

Targets arrive padded to a fixed ``max_targets`` with a validity mask, so every loss is a masked
reduction over static shapes and the whole criterion traces cleanly under ``jit``. The scatter
writes that PyTorch performs with advanced indexing become ``.at[...].set(..., mode="drop")`` with
invalid rows steered out of bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from rfdetr_jax.models.matcher import auction_match, build_cost_matrix, hungarian_match
from rfdetr_jax.utilities.box_ops import (
    box_cxcywh_to_xyxy,
    elementwise_box_iou,
    elementwise_generalized_box_iou,
)

Array = jax.Array


@dataclass(frozen=True)
class CriterionConfig:
    """Loss weights and switches, mirroring ``SetCriterion.__init__`` plus ``build_matcher``."""

    num_classes: int = 91
    focal_alpha: float = 0.25
    group_detr: int = 13
    sum_group_losses: bool = False
    ia_bce_loss: bool = True
    use_varifocal_loss: bool = False
    use_position_supervised_loss: bool = False
    aux_loss: bool = True
    two_stage: bool = True
    cls_loss_coef: float = 1.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    segmentation_head: bool = False
    mask_ce_loss_coef: float = 5.0
    mask_dice_loss_coef: float = 5.0
    mask_point_sample_ratio: int = 16
    use_grouppose_keypoints: bool = False
    num_keypoints_per_class: tuple[int, ...] = ()
    keypoint_l1_loss_coef: float = 0.0
    keypoint_findable_loss_coef: float = 0.0
    keypoint_visible_loss_coef: float = 0.0
    keypoint_nll_loss_coef: float = 0.0
    set_cost_class: float = 2.0
    set_cost_bbox: float = 5.0
    set_cost_giou: float = 2.0
    matcher: str = "hungarian"
    #: Mesh axis the batch is split over when the loss runs per device (``shard_map``). The loss
    #: denominator is then summed over that axis, so every device divides by the global target
    #: count, exactly as the unsplit batch does.
    axis_name: str | None = None

    def __post_init__(self) -> None:
        if self.use_varifocal_loss or self.use_position_supervised_loss:
            raise NotImplementedError(
                "use_varifocal_loss and use_position_supervised_loss are not implemented; "
                "the port supports upstream's default ia_bce_loss and the plain focal loss"
            )

    def solve(self, cost: Array, sizes: Array, group_detr: int) -> tuple[Array, Array, Array]:
        """Run the configured assignment solver on a ``[..., B, Q, T]`` cost matrix.

        ``"hungarian"`` is the exact SciPy solve upstream uses, reached through a host callback.
        ``"auction"`` solves the same problem on device: it removes the round trip that otherwise
        stalls an accelerator mid-step and is what lets the step be sharded across chips.
        """
        if self.matcher == "auction":
            return auction_match(cost, sizes, group_detr)
        if self.matcher == "hungarian":
            return hungarian_match(cost, sizes, group_detr)
        raise ValueError(f"unknown matcher {self.matcher!r}; expected 'hungarian' or 'auction'")

    def weight_dict(self, dec_layers: int) -> dict[str, float]:
        """Build the loss-name to weight map, including aux and encoder suffixes."""
        base = {
            "loss_ce": self.cls_loss_coef,
            "loss_bbox": self.bbox_loss_coef,
            "loss_giou": self.giou_loss_coef,
        }
        if self.segmentation_head:
            base["loss_mask_ce"] = self.mask_ce_loss_coef
            base["loss_mask_dice"] = self.mask_dice_loss_coef
        if self.use_grouppose_keypoints:
            base["loss_keypoints_l1"] = self.keypoint_l1_loss_coef
            base["loss_keypoints_findable"] = self.keypoint_findable_loss_coef
            base["loss_keypoints_visible"] = self.keypoint_visible_loss_coef
            base["loss_keypoints_nll"] = self.keypoint_nll_loss_coef
        weights = dict(base)
        if self.aux_loss:
            for index in range(dec_layers - 1):
                weights.update({f"{key}_{index}": value for key, value in base.items()})
            if self.two_stage:
                weights.update({f"{key}_enc": value for key, value in base.items()})
        return weights


@jax.tree_util.register_dataclass
@dataclass
class Targets:
    """A padded batch of ground-truth boxes and labels.

    Attributes:
        labels: Class indices ``[B, T]``; padded slots may hold any in-range value.
        boxes: Normalized ``cxcywh`` boxes ``[B, T, 4]``.
        mask: ``[B, T]``, ``True`` where the slot holds a real target.
        masks: Optional segmentation masks ``[B, T, Hg, Wg]`` as float 0/1.
        keypoints: Optional keypoints ``[B, T, K, 3]`` as normalized ``(x, y, visibility)``.
    """

    labels: Array
    boxes: Array
    mask: Array
    masks: Array | None = None
    keypoints: Array | None = None

    @property
    def sizes(self) -> Array:
        """Per-image count of real targets, as ``[B]``."""
        return self.mask.sum(-1)


def num_boxes_for_targets(
    targets: Targets, group_detr: int, sum_group_losses: bool, axis_name: str | None = None
) -> Array:
    """Return the loss denominator: total targets times the active group count, floored at one.

    With ``axis_name`` the count is summed over that mesh axis, so a per-device shard of the batch
    divides by the whole batch's targets.
    """
    total = targets.mask.sum().astype(jnp.float32)
    if axis_name is not None:
        total = jax.lax.psum(total, axis_name)
    if not sum_group_losses:
        total = total * group_detr
    return jnp.clip(total, 1.0, None)


def loss_labels(
    pred_logits: Array,
    pred_boxes: Array,
    targets: Targets,
    match: tuple[Array, Array, Array],
    num_boxes: Array,
    config: CriterionConfig,
) -> Array:
    """Classification loss.

    Implements the IA-BCE variant RF-DETR enables by default, and the plain sigmoid focal loss
    otherwise.

    Args:
        pred_logits: Logits ``[B, Q, C]``.
        pred_boxes: Predicted boxes ``[B, Q, 4]``.
        targets: Padded ground truth.
        match: ``(rows, cols, valid)`` from the matcher.
        num_boxes: Loss denominator.
        config: Loss configuration.

    Returns:
        Scalar classification loss.
    """
    rows, cols, valid = match
    batch, num_queries, num_classes = pred_logits.shape
    batch_index = jnp.broadcast_to(jnp.arange(batch)[:, None], rows.shape)

    target_classes = jnp.take_along_axis(targets.labels, cols, axis=1)
    # Steer invalid matches out of bounds so `mode="drop"` discards their scatter writes.
    safe_rows = jnp.where(valid, rows, num_queries)

    if config.ia_bce_loss:
        src_boxes = pred_boxes[batch_index, jnp.clip(rows, 0, num_queries - 1)]
        target_boxes = jnp.take_along_axis(targets.boxes, cols[..., None], axis=1)
        iou, _ = elementwise_box_iou(
            box_cxcywh_to_xyxy(jax.lax.stop_gradient(src_boxes)), box_cxcywh_to_xyxy(target_boxes)
        )
        probabilities = jax.nn.sigmoid(pred_logits)
        matched_prob = probabilities[batch_index, jnp.clip(rows, 0, num_queries - 1), target_classes]
        weight = jnp.power(matched_prob, config.focal_alpha) * jnp.power(iou, 1 - config.focal_alpha)
        weight = jax.lax.stop_gradient(jnp.clip(weight, 0.01, None))
        weight = jnp.where(valid, weight, 0.0)

        pos_weights = jnp.zeros_like(pred_logits).at[batch_index, safe_rows, target_classes].set(
            weight, mode="drop"
        )
        neg_weights = jnp.power(probabilities, 2.0).at[batch_index, safe_rows, target_classes].set(
            1 - weight, mode="drop"
        )
        loss = neg_weights * pred_logits - jax.nn.log_sigmoid(pred_logits) * (pos_weights + neg_weights)
        return loss.sum() / num_boxes

    onehot = jnp.zeros_like(pred_logits).at[batch_index, safe_rows, target_classes].set(1.0, mode="drop")
    loss = sigmoid_focal_loss(pred_logits, onehot, num_boxes, config.focal_alpha, 2.0)
    return loss * num_queries


def sigmoid_focal_loss(inputs: Array, targets: Array, num_boxes: Array, alpha: float, gamma: float) -> Array:
    """Sigmoid focal loss, averaged over queries and summed over the batch."""
    probabilities = jax.nn.sigmoid(inputs)
    ce = _bce_with_logits(inputs, targets)
    p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


def _bce_with_logits(logits: Array, targets: Array) -> Array:
    """Numerically stable elementwise ``binary_cross_entropy_with_logits``."""
    max_val = jnp.clip(-logits, 0.0, None)
    return (1 - targets) * logits + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp(-logits - max_val))


def loss_boxes(
    pred_boxes: Array,
    targets: Targets,
    match: tuple[Array, Array, Array],
    num_boxes: Array,
) -> tuple[Array, Array]:
    """Return the ``(L1, GIoU)`` box losses over matched pairs."""
    rows, cols, valid = match
    batch = pred_boxes.shape[0]
    batch_index = jnp.broadcast_to(jnp.arange(batch)[:, None], rows.shape)
    src_boxes = pred_boxes[batch_index, jnp.clip(rows, 0, pred_boxes.shape[1] - 1)]
    target_boxes = jnp.take_along_axis(targets.boxes, cols[..., None], axis=1)

    weight = valid.astype(pred_boxes.dtype)
    l1 = (jnp.abs(src_boxes - target_boxes).sum(-1) * weight).sum() / num_boxes
    giou = elementwise_generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
    giou_loss = ((1 - giou) * weight).sum() / num_boxes
    return l1, giou_loss


def layer_cost(
    layer: dict,
    targets: Targets,
    config: CriterionConfig,
    key: Array | None = None,
    mask_points: Array | None = None,
) -> Array:
    """Build one output layer's ``[B, Q, T]`` matching cost.

    Segmentation layers cost masks on ``Hm * Wm // mask_point_sample_ratio`` uniform points drawn
    from ``key`` (one draw per layer, shared across the batch, as upstream does); ``mask_points``
    supplies the draw explicitly for parity checks.
    """
    pred_masks = layer.get("pred_masks") if targets.masks is not None else None
    if pred_masks is not None and mask_points is None:
        if key is None:
            raise ValueError("mask matching costs need a PRNG key for point sampling")
        num_points = pred_masks.shape[-2] * pred_masks.shape[-1] // config.mask_point_sample_ratio
        mask_points = jax.random.uniform(key, (num_points, 2), dtype=jnp.float32)
    return build_cost_matrix(
        layer["pred_logits"],
        layer["pred_boxes"],
        targets.labels,
        targets.boxes,
        targets.mask,
        config.set_cost_class,
        config.set_cost_bbox,
        config.set_cost_giou,
        config.focal_alpha,
        pred_keypoints=layer.get("pred_keypoints"),
        target_keypoints=targets.keypoints,
        num_keypoints_per_class=config.num_keypoints_per_class,
        keypoint_costs=(
            config.keypoint_l1_loss_coef,
            config.keypoint_findable_loss_coef,
            config.keypoint_visible_loss_coef,
            config.keypoint_nll_loss_coef,
        ),
        pred_masks=pred_masks,
        target_masks=targets.masks if pred_masks is not None else None,
        mask_costs=(config.mask_ce_loss_coef, config.mask_dice_loss_coef),
        mask_points=mask_points,
    )


def layer_losses(
    layer: dict,
    targets: Targets,
    num_boxes: Array,
    config: CriterionConfig,
    group_detr: int,
    key: Array | None = None,
    match: tuple[Array, Array, Array] | None = None,
    mask_draws: tuple[Array, Array] | None = None,
    cost_points: Array | None = None,
) -> dict[str, Array]:
    """Match one output layer against the targets and return its losses.

    Args:
        layer: Layer outputs with ``pred_logits``/``pred_boxes`` and optionally ``pred_masks``.
        targets: Padded ground truth.
        num_boxes: Loss denominator.
        config: Loss configuration.
        group_detr: Active number of query groups.
        key: PRNG key for mask point sampling; required when mask losses are computed.
        match: A pre-computed ``(rows, cols, valid)`` assignment; solved here when omitted.
        mask_draws: Optional recorded ``(candidates, randoms)`` point draws replacing the key.
        cost_points: Optional recorded matcher points, used when the match is solved here.

    Returns:
        Dict of this layer's losses.
    """
    pred_logits = layer["pred_logits"]
    pred_boxes = layer["pred_boxes"]
    if match is None:
        cost_key = None
        if key is not None:
            cost_key, key = jax.random.split(key)
        match = config.solve(layer_cost(layer, targets, config, cost_key, cost_points), targets.sizes, group_detr)
    l1, giou = loss_boxes(pred_boxes, targets, match, num_boxes)
    losses = {
        "loss_ce": loss_labels(pred_logits, pred_boxes, targets, match, num_boxes, config),
        "loss_bbox": l1,
        "loss_giou": giou,
    }
    if "pred_keypoints" in layer and targets.keypoints is not None:
        losses.update(
            loss_keypoints(
                layer["pred_keypoints"], targets, match, num_boxes, config.num_keypoints_per_class
            )
        )
    if "pred_masks" in layer and targets.masks is not None:
        if key is None and mask_draws is None:
            raise ValueError("mask losses need a PRNG key for point sampling")
        mask_ce, mask_dice = loss_masks(
            layer["pred_masks"], targets.masks, match, num_boxes, key, config.mask_point_sample_ratio,
            draws=mask_draws,
        )
        losses["loss_mask_ce"] = mask_ce
        losses["loss_mask_dice"] = mask_dice
    return losses


def compute_losses(
    outputs: dict,
    targets: Targets,
    config: CriterionConfig,
    train: bool = True,
    key: Array | None = None,
    num_boxes: Array | None = None,
    sampling_overrides: dict[str, dict] | None = None,
) -> dict[str, Array]:
    """Compute every configured loss for one ``(outputs, targets)`` pair.

    Args:
        outputs: Model outputs with ``pred_logits``/``pred_boxes`` and optional ``aux_outputs``
            and ``enc_outputs``.
        targets: Padded ground truth for the batch.
        config: Loss configuration.
        train: Whether the group-DETR multiplier applies, matching ``SetCriterion.training``.
        key: PRNG key for mask point sampling; only needed for segmentation models.
        num_boxes: Optional denominator override; one returns raw loss numerators for manual accumulation.
        sampling_overrides: Parity-check hook keyed by layer suffix (``""``, ``"_0"``, ``"_enc"``):
            ``{"cost_points": [P, 2], "loss_draws": (candidates [S, 2], randoms [R, 2])}`` replaces
            the random mask point draws with recorded upstream values.

    Returns:
        Dict of scalar losses keyed exactly as upstream (``loss_ce``, ``loss_bbox_0``,
        ``loss_giou_enc``, ...).
    """
    group_detr = config.group_detr if train else 1
    if num_boxes is None:
        num_boxes = num_boxes_for_targets(targets, group_detr, config.sum_group_losses, config.axis_name)

    losses: dict[str, Array] = {}
    layers = [("", outputs)]
    layers.extend((f"_{index}", aux) for index, aux in enumerate(outputs.get("aux_outputs", [])))
    if "enc_outputs" in outputs:
        layers.append(("_enc", outputs["enc_outputs"]))

    # One key per layer for the matcher's mask-point draw and one for the mask loss's sampling,
    # mirroring upstream's separate ``torch.rand`` calls.
    if key is not None:
        split = jax.random.split(key, 2 * len(layers))
        cost_keys, loss_keys = list(split[: len(layers)]), list(split[len(layers):])
    else:
        cost_keys = loss_keys = [None] * len(layers)
    overrides = sampling_overrides or {}

    # Every output layer poses the same assignment problem against the same targets, so they are
    # solved together. With the SciPy matcher that turns one device-to-host round trip per layer
    # into one per step, which is what the round trip actually costs on an accelerator; with the
    # on-device matcher it is one larger batched solve instead of several small ones.
    matches: list[tuple[Array, Array, Array] | None] = [None] * len(layers)
    if len(layers) > 1 and len({layer["pred_logits"].shape[:2] for _, layer in layers}) == 1:
        stacked = jnp.stack([
            layer_cost(layer, targets, config, cost_key, overrides.get(suffix, {}).get("cost_points"))
            for (suffix, layer), cost_key in zip(layers, cost_keys, strict=True)
        ])
        rows, cols, valid = config.solve(stacked, targets.sizes, group_detr)
        matches = [(rows[index], cols[index], valid[index]) for index in range(len(layers))]

    for (suffix, layer), layer_key, match in zip(layers, loss_keys, matches, strict=True):
        computed = layer_losses(layer, targets, num_boxes, config, group_detr, layer_key, match,
                                mask_draws=overrides.get(suffix, {}).get("loss_draws"),
                                cost_points=overrides.get(suffix, {}).get("cost_points"))
        losses.update({f"{name}{suffix}": value for name, value in computed.items()})
    return losses


def weighted_total(losses: dict[str, Array], weights: dict[str, float]) -> Array:
    """Sum the losses that carry a weight, exactly as the upstream training step does."""
    return sum(value * weights[key] for key, value in losses.items() if key in weights)


def sigmoid_ce_mask_loss(logits: Array, targets: Array, weight: Array, num_masks: Array) -> Array:
    """Mean-over-points sigmoid cross-entropy, summed over matched masks."""
    loss = _bce_with_logits(logits, targets).mean(-1)
    return (loss * weight).sum() / num_masks


def dice_mask_loss(logits: Array, targets: Array, weight: Array, num_masks: Array) -> Array:
    """Dice loss over sampled points, summed over matched masks."""
    probabilities = jax.nn.sigmoid(logits)
    numerator = 2 * (probabilities * targets).sum(-1)
    denominator = probabilities.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return (loss * weight).sum() / num_masks


def loss_masks(
    pred_masks: Array,
    target_masks: Array,
    match: tuple[Array, Array, Array],
    num_boxes: Array,
    key: Array | None,
    mask_point_sample_ratio: int = 16,
    draws: tuple[Array, Array] | None = None,
) -> tuple[Array, Array]:
    """Return the ``(cross-entropy, Dice)`` mask losses on matched pairs.

    Args:
        pred_masks: Predicted mask logits ``[B, Q, Hm, Wm]``.
        target_masks: Padded ground-truth masks ``[B, T, Hg, Wg]`` as float 0/1.
        match: ``(rows, cols, valid)`` from the matcher.
        num_boxes: Loss denominator.
        key: PRNG key for the PointRend-style point sampling.
        mask_point_sample_ratio: Divisor setting how many points are sampled per mask.
        draws: Optional ``(candidates [S, 2], randoms [R, 2])`` uniform draws shared by every
            mask, replacing ``key`` for parity checks against recorded upstream values.

    Returns:
        A ``(loss_mask_ce, loss_mask_dice)`` pair.
    """
    from rfdetr_jax.models.heads.segmentation import (
        get_uncertain_point_coords_with_randomness,
        point_sample,
    )

    batch, _, mask_h, mask_w = pred_masks.shape
    rows, cols, valid = match
    batch_index = jnp.broadcast_to(jnp.arange(batch)[:, None], rows.shape)
    matches = rows.shape[1]

    src = pred_masks[batch_index, jnp.clip(rows, 0, pred_masks.shape[1] - 1)]
    tgt = target_masks[batch_index, jnp.clip(cols, 0, target_masks.shape[1] - 1)]
    src = src.reshape(batch * matches, 1, mask_h, mask_w)
    tgt = tgt.reshape(batch * matches, 1, target_masks.shape[2], target_masks.shape[3])

    num_points = max(mask_h, mask_h * mask_w // mask_point_sample_ratio)
    uniform = None
    if draws is not None:
        uniform = tuple(jnp.broadcast_to(draw[None], (src.shape[0], *draw.shape)) for draw in draws)
    coords = get_uncertain_point_coords_with_randomness(jax.lax.stop_gradient(src), num_points, key, draws=uniform)
    point_logits = point_sample(src, coords)[:, 0]
    point_labels = jax.lax.stop_gradient(point_sample(tgt, coords, mode="nearest")[:, 0])

    weight = valid.reshape(-1).astype(point_logits.dtype)
    return (
        sigmoid_ce_mask_loss(point_logits, point_labels, weight, num_boxes),
        dice_mask_loss(point_logits, point_labels, weight, num_boxes),
    )


def loss_keypoints(
    pred_keypoints: Array,
    targets: Targets,
    match: tuple[Array, Array, Array],
    num_boxes: Array,
    num_keypoints_per_class: tuple[int, ...],
) -> dict[str, Array]:
    """Return the four GroupPose keypoint losses over matched pairs.

    Args:
        pred_keypoints: Predictions ``[B, Q, K_total, 8]`` in the class-padded layout.
        targets: Padded ground truth carrying ``keypoints``.
        match: ``(rows, cols, valid)`` from the matcher.
        num_boxes: Loss denominator.
        num_keypoints_per_class: Keypoint count for each class.

    Returns:
        A dict with ``loss_keypoints_l1``, ``_findable``, ``_visible`` and ``_nll``.

    Raises:
        ValueError: If the targets carry no keypoints.
    """
    from rfdetr_jax.models.heads.keypoints import compute_l1_keypoint_loss

    if targets.keypoints is None:
        raise ValueError("keypoint losses need targets.keypoints")

    rows, cols, valid = match
    batch, num_queries = pred_keypoints.shape[:2]
    batch_index = jnp.broadcast_to(jnp.arange(batch)[:, None], rows.shape)

    matched_predictions = pred_keypoints[batch_index, jnp.clip(rows, 0, num_queries - 1)]
    matched_keypoints = jnp.take_along_axis(targets.keypoints, cols[..., None, None], axis=1)
    matched_labels = jnp.take_along_axis(targets.labels, cols, axis=1)
    matched_boxes = jnp.take_along_axis(targets.boxes, cols[..., None], axis=1)
    areas = matched_boxes[..., 2] * matched_boxes[..., 3]

    # `compute_l1_keypoint_loss` works on a flat list of matched targets, as upstream's does.
    flat = (batch * rows.shape[1],)
    location, findable, visible, nll = compute_l1_keypoint_loss(
        matched_predictions.reshape(*flat, *matched_predictions.shape[2:]),
        matched_keypoints.reshape(*flat, *matched_keypoints.shape[2:]),
        matched_labels.reshape(flat),
        areas.reshape(flat),
        num_keypoints_per_class,
    )
    weight = valid.reshape(flat).astype(location.dtype)
    return {
        "loss_keypoints_l1": (location * weight).sum() / num_boxes,
        "loss_keypoints_findable": (findable * weight).sum() / num_boxes,
        "loss_keypoints_visible": (visible * weight).sum() / num_boxes,
        "loss_keypoints_nll": (nll * weight).sum() / num_boxes,
    }
