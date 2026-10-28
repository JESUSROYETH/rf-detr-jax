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
# JAX/Flax port of rfdetr.utilities.box_ops (Apache-2.0, Roboflow / LW-DETR / DETR).
# ------------------------------------------------------------------------
"""Bounding-box conversions and IoU variants, ported one-to-one from the PyTorch original.

Every function keeps the upstream operation order and the ``eps=1e-7`` denominator clamps so a
degenerate box produces the same finite value it does in ``rfdetr.utilities.box_ops``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def box_cxcywh_to_xyxy(x: Array) -> Array:
    """Convert ``(cx, cy, w, h)`` boxes to ``(x0, y0, x1, y1)``, clamping negative sizes to zero."""
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    w = jnp.clip(w, 0.0, None)
    h = jnp.clip(h, 0.0, None)
    return jnp.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], axis=-1)


def box_xyxy_to_cxcywh(x: Array) -> Array:
    """Convert ``(x0, y0, x1, y1)`` boxes to ``(cx, cy, w, h)``."""
    x0, y0, x1, y1 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    return jnp.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], axis=-1)


def box_area(boxes: Array) -> Array:
    """Return ``(x1 - x0) * (y1 - y0)`` for xyxy boxes, matching ``torchvision.ops.box_area``."""
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def box_iou(boxes1: Array, boxes2: Array) -> tuple[Array, Array]:
    """Pairwise IoU and union for ``[N, 4]`` and ``[M, 4]`` xyxy boxes.

    Returns:
        A ``(iou, union)`` pair of ``[N, M]`` arrays.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = jnp.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = jnp.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = jnp.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / jnp.clip(union, 1e-7, None)
    return iou, union


def generalized_box_iou(boxes1: Array, boxes2: Array) -> Array:
    """Pairwise generalized IoU (https://giou.stanford.edu/) for xyxy boxes."""
    iou, union = box_iou(boxes1, boxes2)

    lt = jnp.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = jnp.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = jnp.clip(rb - lt, 0.0, None)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / jnp.clip(area, 1e-7, None)


def elementwise_box_iou(boxes1: Array, boxes2: Array) -> tuple[Array, Array]:
    """IoU and union for pre-matched box pairs; broadcasts over all leading axes."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = jnp.maximum(boxes1[..., :2], boxes2[..., :2])
    rb = jnp.minimum(boxes1[..., 2:], boxes2[..., 2:])

    wh = jnp.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]

    union = area1 + area2 - inter
    iou = inter / jnp.clip(union, 1e-7, None)
    return iou, union


def elementwise_generalized_box_iou(boxes1: Array, boxes2: Array) -> Array:
    """Generalized IoU for pre-matched box pairs; broadcasts over all leading axes."""
    iou, union = elementwise_box_iou(boxes1, boxes2)

    lt = jnp.minimum(boxes1[..., :2], boxes2[..., :2])
    rb = jnp.maximum(boxes1[..., 2:], boxes2[..., 2:])

    wh = jnp.clip(rb - lt, 0.0, None)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / jnp.clip(area, 1e-7, None)


def masks_to_boxes(masks: Array) -> Array:
    """Return the ``[N, 4]`` xyxy bounding boxes enclosing each ``[N, H, W]`` binary mask."""
    if masks.size == 0:
        return jnp.zeros((0, 4), dtype=jnp.float32)

    h, w = masks.shape[-2:]
    y = jnp.arange(h, dtype=jnp.float32)
    x = jnp.arange(w, dtype=jnp.float32)
    y_grid, x_grid = jnp.meshgrid(y, x, indexing="ij")

    x_mask = masks * x_grid[None]
    x_max = x_mask.reshape(masks.shape[0], -1).max(-1)
    x_min = jnp.where(masks.astype(bool), x_mask, 1e8).reshape(masks.shape[0], -1).min(-1)

    y_mask = masks * y_grid[None]
    y_max = y_mask.reshape(masks.shape[0], -1).max(-1)
    y_min = jnp.where(masks.astype(bool), y_mask, 1e8).reshape(masks.shape[0], -1).min(-1)

    boxes = jnp.stack([x_min, y_min, x_max, y_max], axis=1)
    keep = masks.reshape(masks.shape[0], -1).any(-1)
    return jnp.where(keep[:, None], boxes, 0.0)


def batch_dice_loss(inputs: Array, targets: Array) -> Array:
    """Pairwise Dice cost between ``[N, P]`` mask logits and ``[M, P]`` binary targets."""
    inputs = jax.nn.sigmoid(inputs)
    numerator = 2 * jnp.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


def batch_sigmoid_ce_loss(inputs: Array, targets: Array) -> Array:
    """Pairwise sigmoid cross-entropy cost between ``[N, P]`` logits and ``[M, P]`` targets."""
    hw = inputs.shape[1]
    pos = _bce_with_logits(inputs, jnp.ones_like(inputs))
    neg = _bce_with_logits(inputs, jnp.zeros_like(inputs))
    loss = jnp.einsum("nc,mc->nm", pos, targets) + jnp.einsum("nc,mc->nm", neg, 1 - targets)
    return loss / hw


def _bce_with_logits(logits: Array, targets: Array) -> Array:
    """Numerically stable elementwise ``binary_cross_entropy_with_logits``."""
    max_val = jnp.clip(-logits, 0.0, None)
    return (1 - targets) * logits + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp(-logits - max_val))
