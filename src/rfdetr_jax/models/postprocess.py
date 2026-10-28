# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""Detection post-processing, ported from ``rfdetr.models.postprocess``."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rfdetr_jax.utilities.box_ops import box_cxcywh_to_xyxy

Array = jax.Array


def post_process(
    pred_logits: Array,
    pred_boxes: Array,
    target_sizes: Array,
    num_select: int = 300,
) -> tuple[Array, Array, Array]:
    """Select the top scoring query/class pairs and scale their boxes to pixels.

    Args:
        pred_logits: Classification logits ``[B, Q, C]``.
        pred_boxes: Normalized ``cxcywh`` boxes ``[B, Q, 4]``.
        target_sizes: Per-image ``(height, width)`` ``[B, 2]``.
        num_select: Maximum detections per image.

    Returns:
        A ``(scores, labels, boxes)`` triple with shapes ``[B, K]``, ``[B, K]`` and ``[B, K, 4]``,
        where boxes are absolute ``xyxy`` clamped to the image bounds.
    """
    batch, _, num_classes = pred_logits.shape
    probabilities = jax.nn.sigmoid(pred_logits).reshape(batch, -1)
    k = min(num_select, probabilities.shape[1])
    scores, indexes = jax.lax.top_k(probabilities, k)
    query_index = indexes // num_classes
    labels = indexes % num_classes

    boxes = box_cxcywh_to_xyxy(pred_boxes)
    boxes = jnp.take_along_axis(boxes, query_index[..., None], axis=1)

    height = target_sizes[:, 0].astype(boxes.dtype)
    width = target_sizes[:, 1].astype(boxes.dtype)
    scale = jnp.stack([width, height, width, height], axis=1)[:, None, :]
    boxes = jnp.clip(boxes * scale, 0.0, scale)
    return scores, labels, boxes


def post_process_outputs(
    outputs: dict,
    target_sizes,
    num_select: int = 300,
    num_keypoints_per_class=(),
    trace_alpha: float = 0.2,
    score_threshold: float = 0.0,
    category_ids=None,
) -> list[dict]:
    """Decode all three tasks to per-image NumPy arrays in original-image coordinates.

    Select query/class pairs by descending sigmoid score (stable flattened-index
    tie breaking), as upstream does. Mask logits are resized with bilinear
    ``align_corners=False`` before thresholding at zero. Keypoint slots are indexed
    by raw category ID; column 2 is findability, 4:7 precision-Cholesky parameters.
    Upstream's optional uncertainty fusion is applied before score filtering.
    This host-side boundary supports variable original image sizes, not JIT.
    """
    import numpy as np
    from scipy.special import expit, logsumexp

    from rfdetr_jax.utilities.resample import resize_2d_numpy

    logits = np.asarray(outputs["pred_logits"], dtype=np.float32)
    boxes = np.asarray(outputs["pred_boxes"], dtype=np.float32)
    sizes = np.asarray(target_sizes)
    if logits.ndim != 3 or boxes.shape != (*logits.shape[:2], 4):
        raise ValueError("expected logits [B,Q,C] and boxes [B,Q,4]")
    if sizes.shape != (logits.shape[0], 2) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError("target_sizes must contain positive [height,width] per image")
    if not np.equal(sizes, sizes.astype(np.int64)).all():
        raise ValueError("target_sizes must be integer image dimensions")
    if num_select < 1 or not np.isfinite(trace_alpha) or trace_alpha < 0:
        raise ValueError("num_select must be positive and trace_alpha finite and nonnegative")
    if not np.isfinite(score_threshold):
        raise ValueError("score_threshold must be finite")
    if not np.isfinite(logits).all() or not np.isfinite(boxes).all():
        raise ValueError("nonfinite detection outputs")
    masks = outputs.get("pred_masks")
    keypoints = outputs.get("pred_keypoints")
    if masks is not None and keypoints is not None:
        raise ValueError("masks and keypoints cannot be used together in postprocessing")
    if masks is not None:
        masks = np.asarray(masks, dtype=np.float32)
        if masks.ndim != 4 or masks.shape[:2] != logits.shape[:2] or min(masks.shape[-2:]) < 1:
            raise ValueError("pred_masks must have shape [B,Q,Hm,Wm]")
        if not np.isfinite(masks).all():
            raise ValueError("nonfinite mask logits")
    schema = tuple(num_keypoints_per_class)
    max_points = max(schema, default=0)
    if keypoints is not None:
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if not schema or any(type(count) is not int or count < 0 for count in schema) or max_points < 1:
            raise ValueError("keypoint decoding requires nonnegative integer class counts and at least one keypoint")
        if (keypoints.ndim != 4 or keypoints.shape[:2] != logits.shape[:2]
                or keypoints.shape[2] != len(schema) * max_points or keypoints.shape[3] < 3):
            raise ValueError("pred_keypoints must have shape [B,Q,len(schema)*max(schema),D>=3]")
        if not np.isfinite(keypoints).all():
            raise ValueError("nonfinite keypoint predictions")

    probabilities = expit(logits).reshape(logits.shape[0], -1)
    results = []
    for image, (height, width) in enumerate(sizes.astype(int)):
        flat_indexes = np.argsort(-probabilities[image], kind="stable")[:num_select]
        query_indexes, labels = np.divmod(flat_indexes, logits.shape[-1])
        scores = probabilities[image, flat_indexes].copy()
        selected_boxes = boxes[image, query_indexes]
        corners = np.concatenate((selected_boxes[:, :2] - selected_boxes[:, 2:] / 2,
                                  selected_boxes[:, :2] + selected_boxes[:, 2:] / 2), axis=-1)
        scale = np.array([width, height, width, height], np.float32)
        result = {"scores": scores, "labels": labels, "boxes": np.clip(corners * scale, 0, scale)}
        if keypoints is not None:
            points = np.zeros((len(labels), max_points, 3), np.float32)
            precision = np.full_like(points, np.nan)
            selected = keypoints[image, query_indexes].reshape(len(labels), len(schema), max_points, -1)
            for category, count in enumerate(schema):
                slots = np.flatnonzero(labels == category)
                active = selected[slots, category, :count]
                if count:
                    points[slots, :count, :2] = active[..., :2] * np.array([width, height])
                    points[slots, :count, 2] = expit(active[..., 2])
                    if active.shape[-1] >= 7:
                        precision[slots, :count] = active[..., 4:7]
                if trace_alpha > 0 and selected.shape[-1] >= 7:
                    log_mean_trace = np.zeros(len(slots), np.float32)
                    if count:
                        t1, t2 = -2 * active[..., 4], -2 * active[..., 6]
                        t3 = 2 * np.log(np.maximum(np.abs(active[..., 5]), 1e-12)) + t1 + t2
                        log_trace = logsumexp(np.stack((t1, t2, t3), axis=-1), axis=-1)
                        log_weights = np.log(np.maximum(expit(active[..., 2]), 1e-12))
                        log_mean_trace = logsumexp(log_trace + log_weights, axis=-1) - logsumexp(log_weights, axis=-1)
                    with np.errstate(divide="ignore"):
                        scores[slots] = np.minimum(expit(np.log(scores[slots]) - trace_alpha * log_mean_trace),
                                                   np.nextafter(np.float32(1), np.float32(0)))
            result["keypoints"] = points
            result["keypoint_precision_cholesky"] = precision
        keep = scores > score_threshold
        if category_ids is not None:
            keep &= np.isin(labels, category_ids)
        result = {name: value[keep] for name, value in result.items()}
        if masks is not None:
            selected_masks = masks[image, query_indexes[keep]]
            resized = np.empty((len(selected_masks), height, width), dtype=bool)
            for start in range(0, len(selected_masks), 32):
                resized[start:start + 32] = resize_2d_numpy(selected_masks[start:start + 32], (height, width)) > 0
            result["masks"] = resized
        results.append(result)
    return results
