# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""COCO detection metrics via hotcoco when installed, else ``pycocotools``; framework-agnostic.

Predictions arrive as plain numpy so the same evaluator scores the JAX and the PyTorch model,
which is what makes their mAP directly comparable.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path

import numpy as np


def accumulate_detections(
    image_ids: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
    category_ids: list[int],
    score_threshold: float = 0.0,
    *,
    preserve_xyxy: bool = False,
) -> list[dict]:
    """Convert one batch of predictions into COCO result records.

    Args:
        image_ids: COCO image ids ``[B]``.
        scores: Detection scores ``[B, K]``.
        labels: Predicted class indices ``[B, K]``; these index the detector's head, which uses raw
            COCO category ids, so a label is emitted only when it is a real category id.
        boxes: Absolute ``xyxy`` boxes ``[B, K, 4]``.
        category_ids: The dataset's category ids, used to drop head slots with no category.
        score_threshold: Drop detections at or below this score.
        preserve_xyxy: Retain original endpoints for the public callback metric adapter.

    Returns:
        A list of COCO-format detection dicts.
    """
    # Filter candidates with vectorized operations before constructing individual COCO records.
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels)
    boxes = np.asarray(boxes, dtype=np.float32)
    keep = (scores > score_threshold) & np.isin(labels, np.asarray(category_ids))
    rows, _ = np.nonzero(keep)
    kept = boxes[keep].astype(np.float64)
    xywh = np.concatenate([kept[:, :2], kept[:, 2:] - kept[:, :2]], axis=1).tolist()
    records = [
        {"image_id": image_id, "category_id": label, "bbox": box, "score": score}
        for image_id, label, box, score in zip(
            np.asarray(image_ids)[rows].astype(np.int64).tolist(),
            labels[keep].astype(np.int64).tolist(),
            xywh,
            scores[keep].astype(np.float64).tolist(),
            strict=True,
        )
    ]
    if preserve_xyxy:
        for row, box in zip(records, kept.tolist(), strict=True):
            row["_boxes_xyxy"] = box
    return records


@functools.lru_cache(maxsize=8)
def _cached_ground_truth(annotation_file: str, backend: str, size: int, mtime_ns: int, inode: int, ctime_ns: int):
    coco = importlib.import_module("hotcoco" if backend == "hotcoco" else "pycocotools.coco").COCO
    with contextlib.redirect_stdout(io.StringIO()):
        return coco(annotation_file)


def _ground_truth(annotation_file: str, backend: str):
    """Load a ground-truth file once per process and reload it when the file changes."""
    status = os.stat(annotation_file)
    return _cached_ground_truth(annotation_file, backend, status.st_size, status.st_mtime_ns,
                                status.st_ino, status.st_ctime_ns)


def resolve_backend(backend: str = "auto") -> str:
    """Pick the COCO evaluator: ``hotcoco`` when installed (``auto``), otherwise ``pycocotools``.

    hotcoco provides an optional Rust implementation of COCO evaluation. Select a backend
    explicitly when comparing metrics across environments.
    """
    if backend == "auto":
        return "hotcoco" if importlib.util.find_spec("hotcoco") is not None else "pycocotools"
    if backend not in ("hotcoco", "pycocotools"):
        raise ValueError(f"Unknown COCO evaluation backend: {backend!r}")
    return backend


def evaluate_detections(
    annotation_file: str | Path,
    detections: list[dict],
    max_dets: int = 300,
    backend: str = "auto",
    *,
    iou_type: str = "bbox",
) -> dict[str, float]:
    """Score detections with ``COCOeval`` and return the standard summary metrics.

    Args:
        annotation_file: Ground-truth COCO JSON for the split.
        detections: Records from :func:`accumulate_detections`.
        max_dets: Maximum detections per image for the third ``maxDets`` entry.
        backend: ``"auto"``, ``"hotcoco"`` or ``"pycocotools"``; see :func:`resolve_backend`.
        iou_type: ``"bbox"`` for boxes or ``"segm"`` for original-size RLE masks.

    Returns:
        A dict with ``mAP``, ``mAP_50``, ``mAP_75``, ``mAP_small``, ``mAP_medium``, ``mAP_large``
        and ``mAR_max``. All zero when there are no detections.
    """
    if iou_type not in {"bbox", "segm"}:
        raise ValueError("iou_type must be bbox or segm")
    if iou_type == "segm":
        # loadRes otherwise derives detection area from bbox, corrupting mask area bins.
        detections = [{key: value for key, value in record.items() if key != "bbox"} for record in detections]
        if any("segmentation" not in record for record in detections):
            raise ValueError("segmentation evaluation requires an RLE for every prediction")
    if max_dets < 1:
        raise ValueError("max_dets must be positive")
    backend = resolve_backend(backend)
    COCOeval = importlib.import_module("hotcoco" if backend == "hotcoco" else "pycocotools.cocoeval").COCOeval  # noqa: N806

    keys = ["mAP", "mAP_50", "mAP_75", "mAP_small", "mAP_medium", "mAP_large", "mAR_max"]
    if not detections:
        return dict.fromkeys(keys, 0.0)

    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = _ground_truth(str(annotation_file), backend)
        predictions = ground_truth.loadRes(json.loads(json.dumps(detections)))
        evaluator = COCOeval(ground_truth, predictions, iouType=iou_type)
        # COCOeval sorts maxDets before evaluation; keep the requested budget last even below 10.
        evaluator.params.maxDets = [1, min(10, max_dets), max_dets]
        evaluator.evaluate()
        evaluator.accumulate()

    # `COCOeval.summarize()` is not used: its first entry hard-codes ``maxDets=100`` and returns
    # -1 whenever 100 is absent from ``params.maxDets``, which it is here. Reading the accumulated
    # arrays directly gives the same numbers at the configured detection budget.
    parameters = evaluator.params
    precision = np.asarray(evaluator.eval["precision"])  # [iou, recall, category, area, maxDets]
    recall = np.asarray(evaluator.eval["recall"])  # [iou, category, area, maxDets]
    area_index = {label: index for index, label in enumerate(parameters.areaRngLbl)}
    last = len(parameters.maxDets) - 1

    def mean_precision(area: str, iou: float | None = None) -> float:
        """Average precision over the requested IoU thresholds and area range."""
        values = precision[:, :, :, area_index[area], last]
        if iou is not None:
            match = np.where(np.isclose(parameters.iouThrs, iou))[0]
            values = values[match]
        values = values[values > -1]
        return float(values.mean()) if values.size else -1.0

    def mean_recall(area: str) -> float:
        """Average recall over IoU thresholds for the requested area range."""
        values = recall[:, :, area_index[area], last]
        values = values[values > -1]
        return float(values.mean()) if values.size else -1.0

    return {
        "mAP": mean_precision("all"),
        "mAP_50": mean_precision("all", 0.5),
        "mAP_75": mean_precision("all", 0.75),
        "mAP_small": mean_precision("small"),
        "mAP_medium": mean_precision("medium"),
        "mAP_large": mean_precision("large"),
        "mAR_max": mean_recall("all"),
    }
