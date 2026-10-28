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
"""Hungarian matcher, ported from ``rfdetr.models.matcher``.

The assignment itself is a host-side SciPy solve, exactly as upstream. It runs through
``jax.pure_callback`` so the whole training step still lives inside one ``jit``: matching is
gradient-free (``@torch.no_grad`` upstream), so a callback costs nothing in correctness.

Targets are padded to a fixed ``max_targets`` per batch and carried with a validity mask, which is
what lets the cost matrix keep a static shape.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

from rfdetr_jax.utilities.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

Array = jax.Array

_FOCAL_LOSS_GAMMA = 2.0


def focal_classification_cost(logits: Array, alpha: float) -> Array:
    """Focal-loss classification cost for logits already gathered at their target classes."""
    probabilities = jax.nn.sigmoid(logits)
    negative = (1 - alpha) * (probabilities**_FOCAL_LOSS_GAMMA) * (-jax.nn.log_sigmoid(-logits))
    positive = alpha * ((1 - probabilities) ** _FOCAL_LOSS_GAMMA) * (-jax.nn.log_sigmoid(logits))
    return positive - negative


def build_cost_matrix(
    pred_logits: Array,
    pred_boxes: Array,
    target_labels: Array,
    target_boxes: Array,
    target_mask: Array,
    cost_class: float,
    cost_bbox: float,
    cost_giou: float,
    focal_alpha: float,
    pred_keypoints: Array | None = None,
    target_keypoints: Array | None = None,
    num_keypoints_per_class: tuple[int, ...] = (),
    keypoint_costs: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    pred_masks: Array | None = None,
    target_masks: Array | None = None,
    mask_costs: tuple[float, float] = (0.0, 0.0),
    mask_points: Array | None = None,
) -> Array:
    """Build the padded ``[B, Q, T]`` matching cost.

    Args:
        pred_logits: Classification logits ``[B, Q, C]``.
        pred_boxes: Predicted boxes ``[B, Q, 4]`` in normalized ``cxcywh``.
        target_labels: Padded target classes ``[B, T]``.
        target_boxes: Padded target boxes ``[B, T, 4]`` in normalized ``cxcywh``.
        target_mask: ``[B, T]``, ``True`` where the padded slot holds a real target.
        cost_class: Classification cost weight.
        cost_bbox: L1 box cost weight.
        cost_giou: GIoU cost weight.
        focal_alpha: Focal-loss alpha for the classification cost.
        pred_keypoints: Optional predicted keypoints ``[B, Q, K_total, 8]``.
        target_keypoints: Optional padded target keypoints ``[B, T, K, 3]``.
        num_keypoints_per_class: Keypoint count for each class.
        keypoint_costs: ``(l1, findable, visible, nll)`` cost weights.
        pred_masks: Optional mask logits ``[B, Q, Hm, Wm]``.
        target_masks: Optional padded ground-truth masks ``[B, T, Hg, Wg]`` as float 0/1.
        mask_costs: ``(cross-entropy, dice)`` mask cost weights.
        mask_points: ``[P, 2]`` normalized sampling points shared by every mask, as upstream's
            ``torch.rand(1, P, 2)`` draw; required when masks are costed.

    Returns:
        Cost matrix ``[B, Q, T]``; padded columns are finite but never selected, because the solver
        only looks at the first ``size`` columns of each image.
    """
    # Reading each target's own class out of the logits is a gather along the minormost axis, the
    # slowest indexing an accelerator does. With so few classes it is cheaper as a contraction
    # against a one-hot: `[B, Q, C] x [B, T, C] -> [B, Q, T]` is a few megaflops.
    labels_onehot = jax.nn.one_hot(target_labels, pred_logits.shape[-1], dtype=pred_logits.dtype)
    gathered = jnp.einsum("bqc,btc->bqt", pred_logits, labels_onehot)
    class_cost = focal_classification_cost(gathered, focal_alpha)

    bbox_cost = jnp.abs(pred_boxes[:, :, None, :] - target_boxes[:, None, :, :]).sum(-1)

    giou = jax.vmap(lambda p, t: generalized_box_iou(box_cxcywh_to_xyxy(p), box_cxcywh_to_xyxy(t)))(
        pred_boxes, target_boxes
    )
    giou_cost = -giou

    cost = cost_bbox * bbox_cost + cost_class * class_cost + cost_giou * giou_cost

    if pred_keypoints is not None and target_keypoints is not None and num_keypoints_per_class:
        from rfdetr_jax.models.heads.keypoints import compute_keypoint_matching_cost

        areas = target_boxes[..., 2] * target_boxes[..., 3]
        # The keypoint cost is defined per image against that image's own targets, so map over
        # the batch rather than flattening: a query must never be costed against another image.
        keypoint_cost = jax.vmap(
            lambda predictions, keypoints, labels, area: sum(
                weight * term
                for weight, term in zip(
                    keypoint_costs,
                    compute_keypoint_matching_cost(
                        predictions[None], keypoints, labels, area, num_keypoints_per_class
                    ),
                    strict=True,
                )
            )[0]
        )(pred_keypoints, target_keypoints, target_labels, areas)
        cost = cost + keypoint_cost

    if pred_masks is not None and target_masks is not None:
        if mask_points is None:
            raise ValueError("mask matching costs need sampling points (mask_points)")
        from rfdetr_jax.models.heads.segmentation import point_sample
        from rfdetr_jax.utilities.box_ops import batch_dice_loss, batch_sigmoid_ce_loss

        # Upstream samples ``Hm * Wm // ratio`` uniform points once per layer, bilinear (border
        # padded) on the logits and nearest on the ground truth, then forms pairwise BCE/Dice costs.
        points = jnp.broadcast_to(mask_points[None], (pred_masks.shape[0], *mask_points.shape))
        point_logits = point_sample(pred_masks, points)  # [B, Q, P]
        point_labels = point_sample(target_masks.astype(point_logits.dtype), points, mode="nearest")  # [B, T, P]
        mask_ce = jax.vmap(batch_sigmoid_ce_loss)(point_logits, point_labels)
        mask_dice = jax.vmap(batch_dice_loss)(point_logits, point_labels)
        cost = cost + mask_costs[0] * mask_ce + mask_costs[1] * mask_dice

    # Padded columns must not be picked even if a caller passes an oversized `size`.
    return jnp.where(target_mask[:, None, :], cost, jnp.float32(1e4))


def _solve_batch(cost: np.ndarray, sizes: np.ndarray, group_detr: int) -> tuple[np.ndarray, ...]:
    """Solve every image/group assignment on the host with SciPy.

    Args:
        cost: Padded cost ``[B, Q, T]`` where ``Q`` spans all groups.
        sizes: Real target count per image ``[B]``.
        group_detr: Number of query groups.

    Returns:
        ``(rows, cols, valid)`` each shaped ``[B, group_detr * T]``; ``rows`` are global query
        indices, ``cols`` are target indices, ``valid`` marks the used slots.
    """
    batch, num_queries, max_targets = cost.shape
    per_group = num_queries // group_detr
    rows = np.zeros((batch, group_detr * max_targets), dtype=np.int32)
    cols = np.zeros((batch, group_detr * max_targets), dtype=np.int32)
    valid = np.zeros((batch, group_detr * max_targets), dtype=bool)

    for image in range(batch):
        size = int(sizes[image])
        if size == 0:
            continue
        for group in range(group_detr):
            start = group * per_group
            block = cost[image, start : start + per_group, :size]
            if not np.isfinite(block).all():
                finite = block[np.isfinite(block)]
                sentinel = (finite.max() + 1.0) if finite.size else 1.0
                block = np.nan_to_num(block, nan=sentinel, posinf=sentinel, neginf=sentinel)
            row, col = linear_sum_assignment(block)
            offset = group * max_targets
            count = row.shape[0]
            rows[image, offset : offset + count] = row + start
            cols[image, offset : offset + count] = col
            valid[image, offset : offset + count] = True
    return rows, cols, valid


def hungarian_match(cost: Array, sizes: Array, group_detr: int) -> tuple[Array, Array, Array]:
    """Solve the padded assignment problem, returning ``(rows, cols, valid)``.

    ``rows`` holds global query indices, ``cols`` target indices, and ``valid`` marks which of the
    ``group_detr * T`` slots carry a real match. Shapes are static, so this is safe under ``jit``.

    ``cost`` may carry leading axes in front of the batch axis (``[..., B, Q, T]``). They are
    folded into one host call so a decoder stack pays a single device-to-host round trip instead of
    one per output layer, which on TPU is the difference between one pipeline stall and several.
    """
    leading = cost.shape[:-3]
    batch, _, max_targets = cost.shape[-3:]
    folded = cost.reshape(-1, *cost.shape[-2:])
    folded_sizes = jnp.broadcast_to(sizes.astype(jnp.int32), (*leading, batch)).reshape(-1)

    slots = group_detr * max_targets
    index_shape = jax.ShapeDtypeStruct((folded.shape[0], slots), jnp.int32)
    result_shape = (index_shape, index_shape, jax.ShapeDtypeStruct((folded.shape[0], slots), jnp.bool_))
    rows, cols, valid = jax.pure_callback(
        lambda c, s: _solve_batch(np.asarray(c), np.asarray(s), group_detr),
        result_shape,
        jax.lax.stop_gradient(folded).astype(jnp.float32),
        folded_sizes,
        vmap_method="sequential",
    )
    shape = (*leading, batch, slots)
    return rows.reshape(shape), cols.reshape(shape), valid.reshape(shape)


def _auction_solve(
    cost: Array,
    valid_col: Array,
    epsilon: Array,
    max_rounds: int,
    phases: int = 1,
    complete: bool = True,
) -> Array:
    """Assign every valid column to a distinct row with Bertsekas' auction algorithm.

    Near-tied columns are the hard case: they can trade rows for many rounds before prices
    separate them. A greedy completion therefore fills any column still unassigned when the round
    cap is reached, so the result is always a full assignment even on a degenerate cost matrix.

    Args:
        cost: ``[N, R, T]`` cost of assigning row ``r`` to column ``t``, for ``N`` independent
            problems with ``R >= T``.
        valid_col: ``[N, T]``, ``True`` for columns that must be assigned.
        epsilon: ``[N, 1]`` bid increment for the final pass. The assignment is within
            ``T * epsilon`` of the optimum, so this is the knob that trades rounds for exactness.
        max_rounds: Hard cap on bidding rounds per pass, so the loop is bounded.
        phases: Number of epsilon-scaling passes; each earlier pass keeps its prices and
            restarts the assignment with a four-fold larger increment.
        complete: Whether to run the greedy completion. Turning it off leaves a column unassigned
            when the round cap is reached, and exists to price the completion on its own.

    Returns:
        ``[N, T]`` row index per column, ``-1`` only where the column is invalid.
    """
    num_problems, num_rows, num_cols = cost.shape
    value = -jnp.swapaxes(cost, 1, 2)  # [N, T, R]: auction maximizes
    neutral = jnp.finfo(value.dtype).min / 4
    rows_axis = jnp.arange(num_rows)
    cols_axis = jnp.arange(num_cols)

    def round_step(carry, epsilon):
        """Run one Jacobi bidding round: every unassigned column bids on its best row."""
        prices, col_row, row_col, count = carry
        bidding = valid_col & (col_row < 0)

        margin = value - prices[:, None, :]
        margin = jnp.where(bidding[:, :, None], margin, neutral)
        best_row = jnp.argmax(margin, axis=-1)
        best = jnp.max(margin, axis=-1)
        # Masking the best entry rather than scattering into it: a select over the whole tensor
        # beats a three-index scatter on an accelerator, and this runs once per bidding round.
        without_best = jnp.where(rows_axis[None, None, :] == best_row[:, :, None], neutral, margin)
        second = jnp.max(without_best, axis=-1)
        bid = jnp.take_along_axis(prices, best_row, axis=1) + (best - second) + epsilon

        # Each row keeps only its highest bidder; a column bids for exactly one row, so no column
        # can win twice in a round.
        targeted = (best_row[:, :, None] == rows_axis[None, None, :]) & bidding[:, :, None]
        offers = jnp.where(targeted, bid[:, :, None], neutral)  # [N, T, R]
        highest = jnp.max(offers, axis=1)
        winner = jnp.argmax(offers, axis=1)
        taken = highest > neutral

        previous_owner = row_col
        prices = jnp.where(taken, highest, prices)
        row_col = jnp.where(taken, winner, row_col)

        won = taken[:, None, :] & (winner[:, None, :] == cols_axis[None, :, None])
        won_any = won.any(-1)
        won_row = jnp.argmax(won, axis=-1)
        displaced = (
            taken[:, None, :]
            & (previous_owner[:, None, :] == cols_axis[None, :, None])
            & (winner[:, None, :] != cols_axis[None, :, None])
        ).any(-1)
        col_row = jnp.where(won_any, won_row, jnp.where(displaced, -1, col_row))
        return prices, col_row, row_col, count + 1

    def keep_going(carry):
        """Stop once every valid column holds a row, or the round cap is reached."""
        _, col_row, _, count = carry
        return (count < max_rounds) & jnp.any(valid_col & (col_row < 0))

    unassigned_cols = jnp.full((num_problems, num_cols), -1, jnp.int32)
    unassigned_rows = jnp.full((num_problems, num_rows), -1, jnp.int32)
    prices = jnp.zeros((num_problems, num_rows), value.dtype)
    col_row = unassigned_cols
    for phase in range(phases):
        step = epsilon * float(4 ** (phases - 1 - phase))
        prices, col_row, _, _ = jax.lax.while_loop(
            keep_going,
            lambda carry, step=step: round_step(carry, step),
            (prices, unassigned_cols, unassigned_rows, jnp.int32(0)),
        )

    if not complete:
        return jnp.where(valid_col, col_row, -1)
    return jax.lax.cond(
        jnp.any(valid_col & (col_row < 0)),
        lambda: _greedy_complete(cost, valid_col, col_row),
        lambda: jnp.where(valid_col, col_row, -1),
    )


def _greedy_complete(cost: Array, valid_col: Array, col_row: Array) -> Array:
    """Fill any column the auction left unassigned with the cheapest row nobody else holds."""
    num_problems, num_rows, num_cols = cost.shape
    rows_axis = jnp.arange(num_rows)
    taken = (col_row[:, :, None] == rows_axis[None, None, :]).any(1)

    def assign(column, carry):
        """Give column ``column`` its cheapest free row when the auction skipped it."""
        col_row, taken = carry
        needed = valid_col[:, column] & (col_row[:, column] < 0)
        candidate = jnp.where(taken, jnp.inf, cost[:, :, column])
        chosen = jnp.argmin(candidate, axis=1).astype(jnp.int32)
        col_row = col_row.at[:, column].set(jnp.where(needed, chosen, col_row[:, column]))
        taken = taken | (needed[:, None] & (rows_axis[None, :] == chosen[:, None]))
        return col_row, taken

    col_row, _ = jax.lax.fori_loop(0, num_cols, assign, (col_row, taken))
    return jnp.where(valid_col, col_row, -1)


def auction_match(
    cost: Array,
    sizes: Array,
    group_detr: int,
    epsilon_scale: float = 1e-4,
    max_rounds: int = 32,
    complete: bool = True,
) -> tuple[Array, Array, Array]:
    """A device-only stand-in for :func:`hungarian_match` with the same signature and shapes.

    The exact SciPy solve uses ``jax.pure_callback``, requiring a device-to-host round trip
    inside the training step and host callback support. The auction algorithm solves the same
    problem on device. When bidding converges, the assignment is within
    ``T * epsilon`` of the optimum, where ``epsilon`` is set from each problem's own cost spread.
    If the round cap is reached, greedy completion preserves a valid assignment but does not
    preserve that optimality bound.

    Args:
        cost: ``[..., B, Q, T]`` padded cost; ``Q`` spans all query groups.
        sizes: Real target count per image ``[B]``.
        group_detr: Number of query groups.
        epsilon_scale: Bid increment as a fraction of the per-problem cost range.
        max_rounds: Cap on bidding rounds, bounding work even when near-tied columns keep
            competing for the same rows. Reaching the cap can leave an incomplete assignment;
            greedy completion does not preserve the converged auction's optimality bound.
        complete: Whether to greedily fill any column the round cap left unassigned.

    Returns:
        ``(rows, cols, valid)`` shaped ``[..., B, group_detr * T]``, with ``rows`` holding global
        query indices, exactly as :func:`hungarian_match` returns them.
    """
    leading = cost.shape[:-3]
    batch, num_queries, max_targets = cost.shape[-3:]
    if group_detr < 1 or num_queries % group_detr:
        raise ValueError("group_detr must be positive and divide the query count")
    per_group = num_queries // group_detr
    if max_targets < 1 or max_targets > per_group:
        raise ValueError(
            "auction_match requires 1 <= padded target count <= queries per group; "
            "reduce max_targets, increase num_queries, or use the Hungarian matcher"
        )
    if epsilon_scale <= 0 or max_rounds < 1:
        raise ValueError("epsilon_scale and max_rounds must be positive")

    blocks = jax.lax.stop_gradient(cost).astype(jnp.float32)
    blocks = blocks.reshape(-1, group_detr, per_group, max_targets).reshape(-1, per_group, max_targets)

    column_valid = jnp.arange(max_targets)[None, :] < sizes.astype(jnp.int32)[:, None]
    column_valid = jnp.broadcast_to(column_valid, (*leading, batch, max_targets))
    column_valid = jnp.repeat(column_valid.reshape(-1, max_targets), group_detr, axis=0)

    spread = jnp.where(column_valid[:, None, :], blocks, 0.0)
    scale = spread.max(axis=(1, 2)) - spread.min(axis=(1, 2))
    epsilon = (epsilon_scale * jnp.maximum(scale, 1e-6))[:, None]

    col_row = _auction_solve(blocks, column_valid, epsilon, max_rounds, complete=complete)
    valid = col_row >= 0
    offsets = (jnp.arange(group_detr) * per_group).repeat(max_targets)
    rows = jnp.where(valid, col_row, 0).reshape(-1, group_detr * max_targets) + offsets[None]
    cols = jnp.broadcast_to(jnp.arange(max_targets)[None], col_row.shape)

    shape = (*leading, batch, group_detr * max_targets)
    return (
        rows.astype(jnp.int32).reshape(shape),
        cols.reshape(-1, group_detr * max_targets).astype(jnp.int32).reshape(shape),
        valid.reshape(-1, group_detr * max_targets).reshape(shape),
    )
