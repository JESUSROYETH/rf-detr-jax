# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""OKS keypoint metrics, ported from ``rfdetr.evaluation.keypoint_oks`` and the keypoint half of
``rfdetr.evaluation.coco_eval``.

Like the detection evaluator this is plain numpy over ``pycocotools``, so the same code scores the
JAX and the PyTorch model and their numbers are directly comparable. Categories are grouped by
keypoint count for scoring. The raw evaluator averages categories with labeled ground truth;
the public API can instead use the published group-size-weighted aggregation.
Unlike the historical upstream-derived implementation, categories without predictions contribute
zero rather than disappearing from the aggregate.
"""

from __future__ import annotations

import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

#: COCO's published per-keypoint OKS sigmas for the 17-point person skeleton.
COCO_PERSON_KEYPOINT_SIGMAS = (
    np.asarray(
        [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89],
        dtype=np.float64,
    )
    / 10.0
)

#: Sigma used for a keypoint layout COCO publishes no sigmas for.
DEFAULT_CUSTOM_KEYPOINT_OKS_SIGMA = 0.05

#: COCO fixes keypoint evaluation at 20 detections per image regardless of the detection budget.
KEYPOINT_MAX_DETS = 20


def resolve_keypoint_oks_sigmas(keypoint_count: int, sigmas: list[float] | None) -> list[float]:
    """Return the OKS sigma vector to score a group of ``keypoint_count``-point categories.

    Args:
        keypoint_count: Number of keypoints the categories in this group carry.
        sigmas: Explicit sigmas, or ``None`` to fall back to COCO's person sigmas for a 17-point
            layout and a uniform :data:`DEFAULT_CUSTOM_KEYPOINT_OKS_SIGMA` otherwise.

    Returns:
        A list of ``keypoint_count`` positive sigmas.

    Raises:
        ValueError: If explicit sigmas are empty, non-finite, non-positive, or shorter than
            ``keypoint_count``.
    """
    if keypoint_count < 1:
        raise ValueError("keypoint_count must be positive")
    if sigmas is None:
        if keypoint_count == len(COCO_PERSON_KEYPOINT_SIGMAS):
            return [float(value) for value in COCO_PERSON_KEYPOINT_SIGMAS]
        return [DEFAULT_CUSTOM_KEYPOINT_OKS_SIGMA] * keypoint_count

    values = np.asarray(sigmas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("keypoint_oks_sigmas must be a non-empty one-dimensional sequence.")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("keypoint_oks_sigmas values must be positive finite numbers.")
    if values.size < keypoint_count:
        raise ValueError(
            f"keypoint_oks_sigmas length {values.size} does not match dataset keypoint count {keypoint_count}."
        )
    return [float(value) for value in values[:keypoint_count]]


def accumulate_keypoint_detections(
    image_ids: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    keypoints: np.ndarray,
    category_ids: list[int],
    score_threshold: float = 0.0,
) -> list[dict]:
    """Convert one batch of keypoint predictions into COCO result records.

    Args:
        image_ids: COCO image ids ``[B]``.
        scores: Detection scores ``[B, K]``.
        labels: Predicted class indices ``[B, K]``; only real category ids are emitted.
        keypoints: Predicted keypoints ``[B, K, P, 3]`` as ``(x, y, confidence)`` in pixels.
        category_ids: The dataset's category ids.
        score_threshold: Drop detections at or below this score.

    Returns:
        A list of COCO keypoint result dicts with flattened ``keypoints`` triplets.
    """
    valid = set(category_ids)
    records: list[dict] = []
    for index, image_id in enumerate(image_ids):
        for score, label, points in zip(scores[index], labels[index], keypoints[index], strict=True):
            if score <= score_threshold or int(label) not in valid:
                continue
            records.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "keypoints": [float(value) for value in np.asarray(points, dtype=np.float64).reshape(-1)],
                    "score": float(score),
                }
            )
    return records


def _keypoint_counts_by_category(ground_truth) -> dict[int, int]:
    """Return each category's keypoint count, from its metadata or its annotations."""
    counts: dict[int, int] = {}
    for category_id, category in ground_truth.cats.items():
        names = category.get("keypoints")
        if isinstance(names, list) and names:
            counts[category_id] = len(names)
    observed: dict[int, set[int]] = defaultdict(set)
    for annotation in ground_truth.anns.values():
        points = annotation.get("keypoints")
        if isinstance(points, list) and points:
            if len(points) % 3:
                raise ValueError("keypoint annotations must contain x/y/visibility triplets")
            observed[annotation["category_id"]].add(len(points) // 3)
    for category_id, sizes in observed.items():
        if len(sizes) != 1 or (category_id in counts and counts[category_id] not in sizes):
            raise ValueError(f"inconsistent keypoint count for category {category_id}")
        counts.setdefault(category_id, next(iter(sizes)))
    return counts


def _backfill_num_keypoints(ground_truth) -> None:
    """Populate the ``num_keypoints`` field COCO's keypoint evaluator requires but datasets omit."""
    for annotation in ground_truth.anns.values():
        points = annotation.get("keypoints")
        if "num_keypoints" not in annotation and isinstance(points, list):
            annotation["num_keypoints"] = sum(1 for visibility in points[2::3] if visibility > 0)


def evaluate_keypoints(
    annotation_file: str | Path,
    detections: list[dict],
    keypoint_oks_sigmas: list[float] | None = None,
    *,
    aggregation: str = "eligible_categories",
) -> dict[str, float]:
    """Score keypoint detections with OKS-based COCO metrics.

    Categories are grouped by keypoint count and evaluated separately — COCO's evaluator assumes a
    single sigma vector — then reduced using the selected aggregation, including groups without predictions.

    Args:
        annotation_file: Ground-truth COCO keypoint JSON for the split.
        detections: Records from :func:`accumulate_keypoint_detections`.
        keypoint_oks_sigmas: Explicit sigmas, or ``None`` to resolve per group.
        aggregation: ``eligible_categories`` preserves raw CLI averaging;
            ``upstream_groups`` weights each group summary by all its category IDs.

    Returns:
        A dict with ``map``, ``map_50``, ``map_75`` and ``mar``. Annotated categories
        with no predictions contribute zero; ``-1.0`` denotes unavailable statistics
        (for example, a split without any labeled keypoints).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if aggregation not in {"eligible_categories", "upstream_groups"}:
        raise ValueError("unknown keypoint aggregation")

    keys = ["map", "map_50", "map_75", "mar"]
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotation_file))
    _backfill_num_keypoints(ground_truth)
    counts = _keypoint_counts_by_category(ground_truth)
    for annotation in ground_truth.anns.values():
        if annotation["category_id"] in counts and not annotation.get("keypoints"):
            annotation["keypoints"] = [0.0] * (counts[annotation["category_id"]] * 3)
            annotation["num_keypoints"] = 0
    groups: dict[int, list[int]] = defaultdict(list)
    for category_id, count in counts.items():
        if count > 0:
            groups[count].append(category_id)
    if not groups:
        return dict.fromkeys(keys, 0.0)
    for record in detections:
        count = counts.get(record["category_id"])
        points = np.asarray(record.get("keypoints", []))
        if count is None or points.ndim != 1 or len(points) != count * 3 or not np.isfinite(points).all():
            raise ValueError("predicted keypoints must match their category schema and contain finite triplets")

    per_category: list[list[float]] = []
    per_group = []
    for keypoint_count, category_group in sorted(groups.items()):
        subset = [record for record in detections if record["category_id"] in set(category_group)]
        with contextlib.redirect_stdout(io.StringIO()):
            if subset:
                predictions = ground_truth.loadRes(json.loads(json.dumps(subset)))
            else:
                # COCO.loadRes([]) crashes. An empty indexed result still lets COCOeval
                # score missed annotated categories as zero, rather than omit the group.
                predictions = COCO()
                predictions.dataset = {"images": ground_truth.dataset["images"],
                                       "categories": ground_truth.dataset["categories"], "annotations": []}
                predictions.createIndex()
            evaluator = COCOeval(ground_truth, predictions, iouType="keypoints")
            evaluator.params.catIds = sorted(category_group)
            evaluator.params.kpt_oks_sigmas = np.asarray(
                resolve_keypoint_oks_sigmas(keypoint_count, keypoint_oks_sigmas), dtype=np.float64
            )
            evaluator.evaluate()
            evaluator.accumulate()
        precision = np.asarray(evaluator.eval["precision"])  # IoU, recall, category, area, maxDets
        recall = np.asarray(evaluator.eval["recall"])
        thresholds = evaluator.params.iouThrs
        group_start = len(per_category)
        for category in range(len(category_group)):
            def mean_valid(values):
                valid = values[values > -1]
                return float(valid.mean()) if valid.size else -1.0

            values = precision[:, :, category, 0, -1]
            per_category.append([
                mean_valid(values),
                mean_valid(values[np.isclose(thresholds, 0.5)]),
                mean_valid(values[np.isclose(thresholds, 0.75)]),
                mean_valid(recall[:, category, 0, -1]),
            ])
        group_values = np.asarray(per_category[group_start:])
        per_group.append((len(category_group), np.asarray([
            float(column[column > -1].mean()) if np.any(column > -1) else -1.
            for column in group_values.T], dtype=np.float32)))
    if not per_category:
        return dict.fromkeys(keys, 0.0)
    if aggregation == "upstream_groups":
        # Published RF-DETR weights each schema group's available summary by all
        # category IDs in that group, including category metadata with no GT.
        result = {}
        for index, key in enumerate(keys):
            available = [(count, values[index]) for count, values in per_group if values[index] >= 0]
            result[key] = float(np.float32(sum(count * float(value) for count, value in available) /
                                          sum(count for count, _ in available))) if available else -1.
        return result
    stats = np.asarray(per_category)
    # Macro-average eligible categories, not schema groups. A missing group must
    # neither disappear nor outweigh a group containing several real categories.
    return {key: float(values[values > -1].mean()) if np.any(values > -1) else -1.0
            for key, values in zip(keys, stats.T, strict=True)}
