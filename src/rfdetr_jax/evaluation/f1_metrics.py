# Modified NumPy adaptation of RF-DETR matching and confidence-sweep contracts.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Matching ancestry: LW-DETR, Copyright (c) 2024 Baidu. All Rights Reserved;
# Conditional DETR, Copyright (c) 2021 Microsoft. All Rights Reserved;
# DETR, Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
"""Upstream public macro-F1: IoU .5 matching and a 101-point confidence sweep."""
from __future__ import annotations

import contextlib
import io

import numpy as np


def _resize_masks(masks, size):
    """Torch nearest interpolation's floor-index rule (not PIL's center rule)."""
    h, w = size
    rows = np.minimum(np.arange(h) * masks.shape[-2] // h, masks.shape[-2] - 1)
    cols = np.minimum(np.arange(w) * masks.shape[-1] // w, masks.shape[-1] - 1)
    return masks[:, rows[:, None], cols[None, :]]


def _pairwise_iou(pred, target, iou_type):
    if iou_type == "segm":
        masks, targets = np.asarray(pred, bool), np.asarray(target, bool)
        if masks.shape[-2:] != targets.shape[-2:]:
            targets = _resize_masks(targets, masks.shape[-2:])
        p = masks.reshape(len(masks), -1).astype(np.float32)
        t = targets.reshape(len(targets), -1).astype(np.float32)
        intersection = p @ t.T
        union = p.sum(1)[:, None] + t.sum(1)[None, :] - intersection
    else:
        p, t = np.asarray(pred, np.float32), np.asarray(target, np.float32)
        wh = np.maximum(np.minimum(p[:, None, 2:], t[None, :, 2:]) -
                        np.maximum(p[:, None, :2], t[None, :, :2]), 0)
        intersection = wh.prod(-1)
        union = (p[:, 2:] - p[:, :2]).prod(-1)[:, None] + (t[:, 2:] - t[:, :2]).prod(-1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def build_matching_data(preds_list, targets_list, iou_threshold=.5, iou_type="bbox"):
    """Match each image/class by stable descending score, preferring unused noncrowd GT.

    Inputs are NumPy equivalents of upstream's boxes/scores/labels/masks/iscrowd
    dictionaries. Crowd overlap is ignored only if no noncrowd match is available.
    Unlike AP, this matching is not recomputed at each confidence threshold.
    """
    if iou_type not in {"bbox", "segm"}:
        raise ValueError("iou_type must be bbox or segm")
    accumulator = {}
    for pred, target in zip(preds_list, targets_list, strict=True):
        pl, tl = np.asarray(pred["labels"]), np.asarray(target["labels"])
        crowd = np.asarray(target.get("iscrowd", np.zeros(len(tl))), bool)
        if crowd.shape != (len(tl),):
            raise ValueError("iscrowd must be one-dimensional with one value per target")
        for label in np.union1d(pl, tl):
            pi, ti = np.flatnonzero(pl == label), np.flatnonzero(tl == label)
            scores = np.asarray(pred["scores"])[pi]
            order = np.argsort(-scores, kind="stable")
            pi, scores = pi[order], scores[order].astype(np.float32)
            matches, ignored = np.zeros(len(pi), np.int64), np.zeros(len(pi), bool)
            if len(pi) and len(ti):
                field = "masks" if iou_type == "segm" else "boxes"
                if field not in pred or field not in target:
                    raise ValueError(f"{iou_type} matching requires {field} on predictions and targets")
                overlap = _pairwise_iou(np.asarray(pred[field])[pi], np.asarray(target[field])[ti], iou_type)
                taken, is_crowd = np.zeros(len(ti), bool), crowd[ti]
                for i, row in enumerate(overlap):
                    available = np.where(is_crowd | taken, -1., row)
                    best = int(available.argmax())
                    if available[best] >= iou_threshold:
                        matches[i], taken[best] = 1, True
                    elif np.any(row[is_crowd] >= iou_threshold):
                        ignored[i] = True
            item = accumulator.setdefault(int(label), dict(scores=[], matches=[], ignore=[], total_gt=0))
            item["scores"].extend(scores)
            item["matches"].extend(matches)
            item["ignore"].extend(ignored)
            item["total_gt"] += int((~crowd[ti]).sum())
    return {label: dict(scores=np.asarray(item["scores"], np.float32),
                        matches=np.asarray(item["matches"], np.int64),
                        ignore=np.asarray(item["ignore"], bool), total_gt=item["total_gt"])
            for label, item in accumulator.items()}


def best_f1_metrics(matching):
    """Return macro metrics at the first best threshold in linspace(0, 1, 101).

    Macro averages include only categories with noncrowd GT. Threshold comparisons
    are float64, matching the reference sweep even at float32 score boundaries.
    """
    thresholds = np.linspace(0, 1, 101)
    precisions, recalls, f1s = [], [], []
    for label in sorted(matching):
        item = matching[label]
        if not item["total_gt"]:
            continue
        scores = np.asarray(item["scores"], np.float64)
        order = np.argsort(scores, kind="stable")
        scores = scores[order]
        valid = ~np.asarray(item["ignore"], bool)[order] & ~np.isnan(scores)
        positive = valid & (np.asarray(item["matches"])[order] != 0)
        first = np.searchsorted(scores, thresholds, side="left")
        tp = np.r_[np.cumsum(positive[::-1])[::-1], 0][first]
        total = np.r_[np.cumsum(valid[::-1])[::-1], 0][first]
        precision = np.divide(tp, total, out=np.zeros_like(tp, dtype=float), where=total > 0)
        recall = tp / item["total_gt"]
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(precision), where=precision + recall > 0)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    if not f1s:
        return dict(F1=0., precision=0., recall=0.)
    f1 = np.mean(f1s, axis=0)
    best = int(f1.argmax())
    return dict(F1=float(f1[best]), precision=float(np.mean(precisions, axis=0)[best]),
                recall=float(np.mean(recalls, axis=0)[best]))


def evaluate_coco_f1(annotation_file, records, *, task="detection", resolution=None):
    """Evaluate the older source-pin convention without restored crowd regions.

    This helper preserves the initial port's behavior for existing callers. The
    public facade uses ``public_evaluation.evaluate_public_predictions`` to match
    published RF-DETR 1.11.2, whose callback restores crowd regions. The older
    source-pin callback drops crowd and clipped-degenerate annotations. Its
    segmentation GT passes through the square
    evaluation resize then nearest-resizes to the prediction grid. ``resolution``
    reproduces that round trip; omitting it evaluates the original mask grid.
    Keypoint F1 uses boxes; OKS remains a separate metric.
    """
    from pycocotools import mask as mask_util
    from pycocotools.coco import COCO

    if task not in {"detection", "segmentation", "keypoints"}:
        raise ValueError(f"unknown task {task!r}")
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(annotation_file))
    by_image = {}
    for row in records:
        by_image.setdefault(row["image_id"], []).append(row)
    if set(by_image) - set(coco.imgs):
        raise ValueError("prediction image IDs are absent from annotations")
    aggregate = {}
    for image_id, image in coco.imgs.items():
        h, w = image["height"], image["width"]
        pred_rows = by_image.get(image_id, [])
        target_rows = [row for row in coco.imgToAnns[image_id] if not row.get("iscrowd", 0)]
        def boxes(rows):
            result = np.asarray([row["bbox"] for row in rows], np.float32).reshape(-1, 4)
            result[:, 2:] += result[:, :2]
            return result
        pb, tb = boxes(pred_rows), boxes(target_rows)
        tb[:, ::2] = np.clip(tb[:, ::2], 0, w)
        tb[:, 1::2] = np.clip(tb[:, 1::2], 0, h)
        keep = (tb[:, 2:] > tb[:, :2]).all(-1)
        target_rows = [row for row, valid in zip(target_rows, keep, strict=True) if valid]
        pred = dict(boxes=pb, labels=np.asarray([row["category_id"] for row in pred_rows]),
                    scores=np.asarray([row["score"] for row in pred_rows], np.float32))
        target = dict(boxes=tb[keep], labels=np.asarray([row["category_id"] for row in target_rows]))
        if task == "segmentation" and pred_rows and target_rows:
            pred["masks"] = np.stack([mask_util.decode(row["segmentation"]).astype(bool) for row in pred_rows])
            target["masks"] = np.stack([coco.annToMask(row).astype(bool) for row in target_rows])
            if resolution is not None:
                target["masks"] = _resize_masks(target["masks"], (resolution, resolution))
        matched = build_matching_data([pred], [target], iou_type="segm" if task == "segmentation" else "bbox")
        for label, item in matched.items():
            old = aggregate.setdefault(label, dict(scores=[], matches=[], ignore=[], total_gt=0))
            for key in ("scores", "matches", "ignore"):
                old[key].extend(item[key])
            old["total_gt"] += item["total_gt"]
    return best_f1_metrics(aggregate)
