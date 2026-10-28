# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""GroupPose keypoint head, losses and matching costs, ported from
``rfdetr.models.heads.keypoints``.

Much of the upstream code is finiteness guarding for mixed-precision training: a single non-finite
channel must not poison a whole target's loss, and `torch.where` is used to keep an unsafe value
out of an op's *local* backward as well as its forward. The same guards are reproduced here, both
because they change which keypoints contribute and because JAX has the identical
``0.0 * nan == nan`` hazard.
"""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from rfdetr_jax.models.layers import Linear

Array = jax.Array

#: Predicted keypoint channels: x, y, findable logit, visible logit, log_l11, l21, log_l22.
KEYPOINT_CHANNELS = 7


def modulate(features: Array, scale: Array, shift: Array) -> Array:
    """Apply AdaLN modulation: ``(scale + 1) * features + shift``."""
    return (scale + 1.0) * features + shift


def active_keypoint_mask(num_keypoints_per_class: Sequence[int]) -> Array:
    """Return the ``[num_classes, max_keypoints]`` mask of keypoints each class actually owns."""
    if not num_keypoints_per_class:
        return jnp.zeros((0, 0), dtype=bool)
    max_keypoints = max(num_keypoints_per_class)
    columns = jnp.arange(max_keypoints)[None, :]
    counts = jnp.asarray(list(num_keypoints_per_class))[:, None]
    return columns < counts


class ConditionalQueryInitializer(nn.Module):
    """Initialise keypoint query tokens by AdaLN-modulating a learned query bank."""

    dim: int
    num_queries: int
    out_dim: int | None = None

    @nn.compact
    def __call__(self, query_features: Array) -> Array:
        """Return ``[..., num_queries, out_dim]`` keypoint queries conditioned on detection features.

        Args:
            query_features: Conditioning features ``[..., dim]``.

        Returns:
            Modulated keypoint queries with one extra trailing query axis.
        """
        out_dim = self.out_dim or self.dim
        queries = self.param("queries", nn.initializers.normal(1.0), (self.num_queries, out_dim))

        compute = jnp.promote_types(queries.dtype, jnp.float32)
        centred = queries.astype(compute)
        normed = (centred - centred.mean(-1, keepdims=True)) * jax.lax.rsqrt(
            centred.var(-1, keepdims=True) + 1e-5
        )
        normed = normed.astype(query_features.dtype)

        hidden = Linear(self.dim, self.dim, name="adaLN_modulation_0")(query_features[..., None, :])
        hidden = jax.nn.gelu(hidden, approximate=False)
        modulation = Linear(self.dim, out_dim * 3, name="adaLN_modulation_2")(hidden)
        scale, shift, gate = jnp.split(modulation, 3, axis=-1)

        modulated = Linear(out_dim, out_dim, name="out_proj")(modulate(normed, scale, shift))
        return modulated * gate + queries.astype(modulated.dtype)


def _bce_with_logits(logits: Array, targets: Array) -> Array:
    """Numerically stable elementwise ``binary_cross_entropy_with_logits``."""
    max_val = jnp.clip(-logits, 0.0, None)
    return (1 - targets) * logits + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp(-logits - max_val))


def compute_l1_keypoint_loss(
    all_pred_keypoints: Array,
    target_keypoints: Array,
    target_classes: Array,
    target_areas: Array,
    num_keypoints_per_class: Sequence[int],
) -> tuple[Array, Array, Array, Array]:
    """Return the per-target ``(location, findable, visible, Gaussian NLL)`` keypoint losses.

    Args:
        all_pred_keypoints: Predictions ``[N, K_total, >= 7]`` in the class-padded layout.
        target_keypoints: Ground truth ``[N, K_max, 3]`` as ``(x, y, visibility)``.
        target_classes: Class id per target ``[N]``.
        target_areas: Normalized box area per target ``[N]``.
        num_keypoints_per_class: Keypoint count for each class.

    Returns:
        Four ``[N]`` arrays, one per loss term.

    Raises:
        ValueError: If the schema is empty or does not divide the padded keypoint count.
    """
    num_targets, total_padded, pred_dim = all_pred_keypoints.shape
    if pred_dim < KEYPOINT_CHANNELS:
        raise ValueError(f"expected at least {KEYPOINT_CHANNELS} keypoint channels, got {pred_dim}")
    num_classes = len(num_keypoints_per_class)
    if num_classes == 0:
        raise ValueError("num_keypoints_per_class must be non-empty when computing keypoint losses.")
    if total_padded % num_classes != 0:
        raise ValueError(f"total padded keypoints ({total_padded}) must divide by num_classes ({num_classes})")

    per_class = total_padded // num_classes
    split = all_pred_keypoints.reshape(num_targets, num_classes, per_class, pred_dim)
    in_schema = (target_classes >= 0) & (target_classes < num_classes)
    safe_classes = jnp.clip(target_classes, 0, num_classes - 1)
    selected = split[jnp.arange(num_targets), safe_classes]

    # Compact upstream schemas may leave detector classes with boxes but no keypoint slot.
    # JAX gathers clamp out-of-range indices, so mask these classes explicitly before losses.
    schema_mask = active_keypoint_mask(num_keypoints_per_class)[safe_classes] & in_schema[:, None]
    area = target_areas.astype(jnp.float32)
    area_eps = jnp.finfo(jnp.float32).eps
    valid_area = jnp.isfinite(area) & (area > area_eps)

    valid_xy = jnp.isfinite(selected[:, :, :2]).all(-1) & jnp.isfinite(target_keypoints[:, :, :2]).all(-1)
    labelled = jnp.isfinite(target_keypoints[:, :, 2]) & (target_keypoints[:, :, 2] > 0)
    location_mask = schema_mask & labelled & valid_xy & valid_area[:, None]
    location_count = jnp.clip(location_mask.sum(-1).astype(selected.dtype), 1.0, None)

    findable_mask = schema_mask & jnp.isfinite(selected[:, :, 2])
    findable_count = jnp.clip(findable_mask.sum(-1).astype(selected.dtype), 1.0, None)
    visible_mask = schema_mask & jnp.isfinite(selected[:, :, 3])
    visible_count = jnp.clip(visible_mask.sum(-1).astype(selected.dtype), 1.0, None)

    safe_area = jnp.where(valid_area, area, 1.0)
    safe_area_sqrt = jnp.sqrt(jnp.clip(safe_area, area_eps, None))

    zeros_xy = jnp.zeros_like(selected[:, :, :2])
    safe_pred_xy = jnp.where(location_mask[..., None], selected[:, :, :2], zeros_xy)
    safe_target_xy = jnp.where(location_mask[..., None], target_keypoints[:, :, :2], zeros_xy)
    l1 = jnp.abs(safe_pred_xy - safe_target_xy).sum(-1)
    location_loss = jnp.where(location_mask, l1, 0.0).sum(-1) / safe_area_sqrt / location_count

    safe_findable = jnp.where(findable_mask, selected[:, :, 2], 0.0)
    findable_bce = _bce_with_logits(safe_findable, (target_keypoints[:, :, 2] > 0).astype(selected.dtype))
    findable_loss = jnp.where(findable_mask, findable_bce, 0.0).sum(-1) / findable_count

    safe_visible = jnp.where(visible_mask, selected[:, :, 3], 0.0)
    visible_bce = _bce_with_logits(safe_visible, (target_keypoints[:, :, 2] > 1).astype(selected.dtype))
    visible_loss = jnp.where(visible_mask, visible_bce, 0.0).sum(-1) / visible_count

    nll_loss = _gaussian_nll(selected, target_keypoints, location_mask, safe_area, area_eps)
    return location_loss, findable_loss, visible_loss, nll_loss


def _gaussian_nll(
    selected: Array,
    target_keypoints: Array,
    location_mask: Array,
    safe_area: Array,
    area_eps: float,
) -> Array:
    """Return the per-target Gaussian NLL over the predicted precision Cholesky parameters."""
    raw_log_l11 = selected[:, :, 4].astype(jnp.float32)
    raw_l21 = selected[:, :, 5].astype(jnp.float32)
    raw_log_l22 = selected[:, :, 6].astype(jnp.float32)
    finite = jnp.isfinite(raw_log_l11) & jnp.isfinite(raw_l21) & jnp.isfinite(raw_log_l22)
    mask = location_mask & finite

    zeros_xy = jnp.zeros_like(selected[:, :, :2])
    delta = (
        jnp.where(mask[..., None], selected[:, :, :2], zeros_xy)
        - jnp.where(mask[..., None], target_keypoints[:, :, :2], zeros_xy)
    ).astype(jnp.float32)

    def chain(active: Array) -> tuple[Array, Array, Array, Array, Array]:
        """Evaluate the Mahalanobis chain with every input masked before it is combined."""
        log_l11 = jnp.where(active, raw_log_l11, 0.0)
        l21 = jnp.where(active, raw_l21, 0.0)
        log_l22 = jnp.where(active, raw_log_l22, 0.0)
        dx = jnp.where(active, delta[:, :, 0], 0.0)
        dy = jnp.where(active, delta[:, :, 1], 0.0)
        u0 = jnp.exp(log_l11) * dx + l21 * dy
        u1 = jnp.exp(log_l22) * dy
        return log_l11, log_l22, u0, u1, u0 * u0 + u1 * u1

    _, _, u0, u1, maha2 = chain(mask)
    # A finite raw log-precision can still overflow once exponentiated, which the pre-exp
    # finiteness check cannot see. Refine the mask and redo the chain so the unsafe value never
    # reaches an op whose local backward would then produce `0.0 * inf == nan`.
    mask = mask & jnp.isfinite(u0) & jnp.isfinite(u1) & jnp.isfinite(maha2)
    log_l11, log_l22, _, _, maha2 = chain(mask)

    count = mask.sum(-1).astype(selected.dtype)
    nll = 0.5 * (maha2 / jnp.clip(safe_area, area_eps, None)[:, None]) - (log_l11 + log_l22)
    finfo = jnp.finfo(jnp.float32)
    nll = jnp.nan_to_num(nll, nan=0.0, posinf=finfo.max / 2, neginf=finfo.min)
    total = jnp.where(mask, nll, 0.0).sum(-1) / jnp.clip(count, 1.0, None)
    return jnp.where(count <= 0, 0.0, total)


def _cdist_bce_with_logits(logits: Array, targets: Array) -> Array:
    """Pairwise BCE-with-logits between ``[..., K]`` logits and ``[N, K]`` targets, summed over ``K``."""
    return jax.nn.softplus(logits).sum(-1)[..., None] - jnp.einsum(
        "...k,nk->...n", logits, targets.astype(logits.dtype)
    )


def compute_keypoint_matching_cost(
    all_pred_keypoints: Array,
    target_keypoints: Array,
    target_classes: Array,
    target_areas: Array,
    num_keypoints_per_class: Sequence[int],
) -> tuple[Array, Array, Array, Array]:
    """Return the ``(l1, findable, visible, nll)`` matching costs, each ``[B, Q, N]``.

    Each target is scored only against the keypoint slots of its own class. Upstream expresses that
    with a Python loop over classes and a dynamic ``nonzero`` gather; here the per-class cost is
    computed for every target and then selected with ``jnp.where``, which keeps shapes static and
    the whole function traceable.

    Args:
        all_pred_keypoints: Predictions ``[B, Q, K_total, >= 7]`` in the class-padded layout.
        target_keypoints: Ground truth ``[N, K_max, 3]``.
        target_classes: Class id per target ``[N]``.
        target_areas: Normalized box area per target ``[N]``.
        num_keypoints_per_class: Keypoint count for each class.

    Returns:
        Four ``[B, Q, N]`` cost arrays.

    Raises:
        ValueError: If the keypoint schema is empty.
    """
    batch, num_queries, total_keypoints, pred_dim = all_pred_keypoints.shape
    num_classes = len(num_keypoints_per_class)
    if num_classes == 0:
        raise ValueError("num_keypoints_per_class must be non-empty when computing keypoint matching costs.")
    num_targets = target_keypoints.shape[0]
    if num_targets == 0:
        zeros = jnp.zeros((batch, num_queries, 0), dtype=all_pred_keypoints.dtype)
        return zeros, zeros, zeros, zeros

    per_class = total_keypoints // num_classes
    predictions = all_pred_keypoints.reshape(batch, num_queries, num_classes, per_class, pred_dim)

    cost_l1 = jnp.zeros((batch, num_queries, num_targets), all_pred_keypoints.dtype)
    cost_findable = jnp.zeros_like(cost_l1)
    cost_visible = jnp.zeros_like(cost_l1)
    cost_nll = jnp.zeros_like(cost_l1)

    area_eps = float(jnp.finfo(jnp.float32).eps)
    areas = target_areas.astype(jnp.float32)
    valid_area = jnp.isfinite(areas) & (areas > area_eps)
    safe_areas = jnp.clip(areas, area_eps, None)

    for class_index in range(num_classes):
        num_kpts = num_keypoints_per_class[class_index]
        if num_kpts == 0:
            continue
        belongs = (target_classes == class_index)[None, None, :]

        pred = predictions[:, :, class_index, :num_kpts, :]  # [B, Q, k, D]
        target = target_keypoints[:, :num_kpts, :]  # [N, k, 3]
        target_xy = target[:, :, :2]

        visible = (
            jnp.isfinite(target[:, :, 2])
            & (target[:, :, 2] > 0)
            & jnp.isfinite(target_xy).all(-1)
            & valid_area[:, None]
        )
        valid_per_target = visible.sum(-1).astype(jnp.float32)
        denominator = jnp.clip(valid_per_target, 1.0, None)

        difference = pred[:, :, None, :, :2].astype(jnp.float32) - target_xy[None, None]
        per_keypoint_l1 = jnp.abs(difference).sum(-1)
        location = jnp.where(visible[None, None], per_keypoint_l1, 0.0).sum(-1)
        location = location / denominator[None, None] / jnp.sqrt(safe_areas)[None, None]
        location = jnp.nan_to_num(location, nan=0.0, posinf=0.0, neginf=0.0)

        nll = _matching_nll(pred, difference, visible, safe_areas, valid_per_target)

        findable = _cdist_bce_with_logits(pred[..., 2], (target[:, :, 2] > 0)) / float(num_kpts)
        visible_cost = _cdist_bce_with_logits(pred[..., 3], (target[:, :, 2] > 1)) / float(num_kpts)

        cost_l1 = jnp.where(belongs, location.astype(cost_l1.dtype), cost_l1)
        cost_nll = jnp.where(belongs, nll.astype(cost_nll.dtype), cost_nll)
        cost_findable = jnp.where(belongs, findable.astype(cost_findable.dtype), cost_findable)
        cost_visible = jnp.where(belongs, visible_cost.astype(cost_visible.dtype), cost_visible)

    return cost_l1, cost_findable, cost_visible, cost_nll


def _matching_nll(
    pred: Array, difference: Array, visible: Array, safe_areas: Array, valid_per_target: Array
) -> Array:
    """Return the ``[B, Q, N]`` Gaussian NLL matching cost for one class's keypoint slots."""
    log_l11 = pred[..., 4].astype(jnp.float32)
    l21 = pred[..., 5].astype(jnp.float32)
    log_l22 = pred[..., 6].astype(jnp.float32)
    finite_pred = (
        jnp.isfinite(pred[..., :2]).all(-1)
        & jnp.isfinite(log_l11)
        & jnp.isfinite(l21)
        & jnp.isfinite(log_l22)
    )

    dx = difference[..., 0]
    dy = difference[..., 1]
    u0 = jnp.exp(log_l11)[:, :, None] * dx + l21[:, :, None] * dy
    u1 = jnp.exp(log_l22)[:, :, None] * dy
    maha2 = u0 * u0 + u1 * u1

    mask = (
        visible[None, None]
        & finite_pred[:, :, None]
        & jnp.isfinite(u0)
        & jnp.isfinite(u1)
        & jnp.isfinite(maha2)
    )
    nll = 0.5 * (maha2 / safe_areas[None, None, :, None]) - (log_l11 + log_l22)[:, :, None]
    finfo = jnp.finfo(jnp.float32)
    nll = jnp.nan_to_num(nll, nan=0.0, posinf=finfo.max / 2, neginf=finfo.min)
    total = jnp.where(mask, nll, 0.0).sum(-1) / jnp.clip(valid_per_target, 1.0, None)[None, None]
    return jnp.where((valid_per_target > 0)[None, None], total, 0.0)
