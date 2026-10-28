# Modified NumPy adaptation of RF-DETR 1.11.2's public evaluation callback.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
"""Published public AP/F1 conventions, separate from the raw COCO CLI evaluator."""
from __future__ import annotations

import contextlib
import functools
import importlib
import io

import numpy as np

from rfdetr_jax.evaluation.coco_eval import resolve_backend
from rfdetr_jax.evaluation.f1_metrics import _resize_masks, best_f1_metrics, build_matching_data
from rfdetr_jax.evaluation.keypoint_oks import evaluate_keypoints


@functools.lru_cache(maxsize=1)
def public_ap_grids():
    """TorchMetrics constructs these grids in float32, before COCO widens them.

    NumPy linspace, even with dtype=float32, differs in the recall grid. Torch is
    a declared checkpoint dependency; using it here avoids inventing a different
    threshold contract. No model, training or prediction operation runs in Torch.
    """
    import torch

    grids = (torch.linspace(.5, .95, 10, dtype=torch.float32, device="cpu").numpy().astype(np.float64),
             torch.linspace(0, 1, 101, dtype=torch.float32, device="cpu").numpy().astype(np.float64))
    for grid in grids:
        grid.setflags(write=False)
    return grids


def _decode_mask(row, height, width):
    from pycocotools import mask as mask_util

    segmentation = row.get("segmentation")
    if segmentation is None or (not isinstance(segmentation, dict) and not segmentation):
        return np.zeros((height, width), dtype=bool)
    if isinstance(segmentation, dict):
        rle = (mask_util.frPyObjects(segmentation, height, width)
               if isinstance(segmentation.get("counts"), list) else segmentation)
    else:
        rle = mask_util.frPyObjects(segmentation, height, width)
    decoded = mask_util.decode(rle).astype(bool)
    return decoded.any(-1) if decoded.ndim == 3 else decoded


def convert_public_target(coco, image_id, *, resolution, image_size, segmentation=False):
    """Reproduce ConvertCoco -> square resize/normalize -> callback conversion.

    Crowd rows follow the ordinary rows. They retain raw, unclipped boxes and
    their masks bypass the ordinary target's model-resolution resize round trip.
    Labels remain raw category IDs, equivalent under a one-to-one model mapping.
    """
    h, w = image_size
    ordinary, crowds = [], []
    for row in coco.imgToAnns[image_id]:
        (crowds if row.get("iscrowd", 0) else ordinary).append(row)
    boxes = np.asarray([row["bbox"] for row in ordinary], np.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, ::2] = np.clip(boxes[:, ::2], 0, w)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
    keep = (boxes[:, 2:] > boxes[:, :2]).all(-1)
    ordinary = [row for row, valid in zip(ordinary, keep, strict=True) if valid]
    boxes = boxes[keep]
    boxes *= np.array([resolution / w, resolution / h] * 2, np.float32)
    center, size = (boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]
    normalized = np.concatenate([center, size], axis=-1) / np.float32(resolution)
    boxes = np.concatenate([normalized[:, :2] - .5 * normalized[:, 2:],
                            normalized[:, :2] + .5 * normalized[:, 2:]], axis=-1)
    boxes *= np.array([w, h, w, h], np.float32)
    crowd_boxes = np.asarray([row["bbox"] for row in crowds], np.float32).reshape(-1, 4)
    crowd_boxes[:, 2:] += crowd_boxes[:, :2]
    result = dict(boxes=np.concatenate([boxes, crowd_boxes]),
                  labels=np.asarray([row["category_id"] for row in ordinary + crowds], np.int64),
                  iscrowd=np.asarray([0] * len(ordinary) + [1] * len(crowds), np.int64))
    if segmentation:
        masks = []
        for row in ordinary:
            mask = _decode_mask(row, h, w)[None]
            masks.append(_resize_masks(_resize_masks(mask, (resolution, resolution)), (h, w))[0])
        # Upstream precompresses crowd polygons using annotation metadata, then
        # resizes their decoded masks to the callback's actual-image grid.
        image = coco.imgs[image_id]
        for row in crowds:
            metadata_h, metadata_w = image.get("height", h), image.get("width", w)
            mask = _decode_mask(row, metadata_h, metadata_w)[None]
            masks.append(_resize_masks(mask, (h, w))[0])
        result["masks"] = np.stack(masks) if masks else np.zeros((0, h, w), bool)
    return result


def _encode_mask(mask):
    from pycocotools import mask as mask_util

    rle = mask_util.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _score_public_coco(ground_truth, predictions, *, iou_type, max_dets, backend):
    """Score an already-converted callback dataset, retaining empty-result sentinels."""
    if not ground_truth["images"]:
        return dict.fromkeys(("mAP", "mAP_50", "mAP_75", "mAP_small", "mAP_medium", "mAP_large", "mAR_max"), -1.)
    backend = resolve_backend(backend)
    module = importlib.import_module("hotcoco" if backend == "hotcoco" else "pycocotools.coco")
    evaluator_module = importlib.import_module("hotcoco" if backend == "hotcoco" else "pycocotools.cocoeval")
    with contextlib.redirect_stdout(io.StringIO()):
        gt, dt = module.COCO(), module.COCO()
        gt.dataset = ground_truth
        gt.createIndex()
        # Building directly preserves both empty images and the explicitly chosen
        # bbox/mask areas; loadRes infers them from whichever fields it encounters.
        dt.dataset = dict(images=ground_truth["images"], categories=ground_truth["categories"],
                          annotations=predictions)
        dt.createIndex()
        evaluator = evaluator_module.COCOeval(gt, dt, iouType=iou_type)
        params = evaluator.params
        iou, recall = public_ap_grids()
        params.iouThrs, params.recThrs = iou.copy(), recall.copy()
        # The published callback passes [1, 10, configured] and TorchMetrics
        # sorts it. Below ten, AP therefore uses ten while public AR selects the
        # configured budget. Preserve that observable (if surprising) contract.
        params.maxDets = sorted([1, 10, max_dets])
        params.catIds = [row["id"] for row in ground_truth["categories"]]
        evaluator.params = params
        evaluator.evaluate()
        evaluator.accumulate()
    precision, recall = np.asarray(evaluator.eval["precision"]), np.asarray(evaluator.eval["recall"])

    def mean(values):
        values = values[values > -1]
        return float(np.float32(values.mean())) if values.size else -1.

    recall_budget = list(evaluator.params.maxDets).index(max_dets)
    return dict(mAP=mean(precision[:, :, :, 0, -1]),
                mAP_50=mean(precision[0, :, :, 0, -1]), mAP_75=mean(precision[5, :, :, 0, -1]),
                mAP_small=mean(precision[:, :, :, 1, -1]), mAP_medium=mean(precision[:, :, :, 2, -1]),
                mAP_large=mean(precision[:, :, :, 3, -1]), mAR_max=mean(recall[:, :, 0, recall_budget]))


def evaluate_public_predictions(annotation_file, records, *, task="detection", resolution,
                                image_sizes=None, max_dets=500, keypoint_oks_sigmas=None,
                                backend="pycocotools"):
    """Return backend-shaped task/bbox AP and F1 using RF-DETR 1.11.2 conventions.

    ``image_sizes`` is the image_id -> (decoded height, width) mapping collected
    from valid evaluation batches. Original annotations are retained for OKS.
    Dense masks are decoded per image and released after encoding/matching.
    """
    from pycocotools import mask as mask_util
    from pycocotools.coco import COCO

    if task not in {"detection", "segmentation", "keypoints"}:
        raise ValueError(f"unknown task {task!r}")
    if type(resolution) is not int or resolution < 1 or type(max_dets) is not int or max_dets < 1:
        raise ValueError("resolution and max_dets must be positive integers")
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(annotation_file))
    image_sizes = image_sizes if image_sizes is not None else {
        key: (row["height"], row["width"]) for key, row in coco.imgs.items()}
    if set(image_sizes) != set(coco.imgs):
        raise ValueError("evaluation image sizes must cover exactly the annotation image IDs")
    by_image = {}
    for row in records:
        if row["image_id"] not in image_sizes:
            raise ValueError("prediction image ID is absent from evaluation images")
        by_image.setdefault(row["image_id"], []).append(row)
    ground_rows, box_rows, mask_rows, matching, classes = [], [], [], {}, set()
    for image_id in coco.imgs:
        target = convert_public_target(coco, image_id, resolution=resolution,
                                        image_size=image_sizes[image_id], segmentation=task == "segmentation")
        current = by_image.get(image_id, [])
        boxes = np.asarray([row["bbox"] for row in current], np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        for index, row in enumerate(current):
            if "_boxes_xyxy" in row:
                boxes[index] = np.asarray(row["_boxes_xyxy"], np.float32)
        pred = dict(boxes=boxes, scores=np.asarray([row["score"] for row in current], np.float32),
                    labels=np.asarray([row["category_id"] for row in current], np.int64))
        if task == "segmentation":
            h, w = image_sizes[image_id]
            pred["masks"] = (np.stack([mask_util.decode(row["segmentation"]).astype(bool) for row in current])
                             if current else np.zeros((0, h, w), bool))
            if pred["masks"].shape[-2:] != (h, w):
                raise ValueError("public mask evaluation requires original-image-size prediction masks")
        data = build_matching_data([pred], [target], iou_type="segm" if task == "segmentation" else "bbox")
        for label, item in data.items():
            entry = matching.setdefault(label, dict(scores=[], matches=[], ignore=[], total_gt=0))
            for key in ("scores", "matches", "ignore"):
                entry[key].extend(item[key])
            entry["total_gt"] += item["total_gt"]
        classes.update(target["labels"].tolist())
        classes.update(pred["labels"].tolist())
        for i, (box, label, crowd) in enumerate(zip(target["boxes"], target["labels"], target["iscrowd"], strict=True)):
            xywh = np.r_[box[:2], box[2:] - box[:2]].astype(float).tolist()
            row = dict(id=len(ground_rows) + 1, image_id=image_id, category_id=int(label),
                       bbox=xywh, iscrowd=int(crowd), area=xywh[2] * xywh[3])
            if task == "segmentation":
                row.update(segmentation=_encode_mask(target["masks"][i]), area=int(target["masks"][i].sum()))
            ground_rows.append(row)
        for i, source in enumerate(current):
            xyxy = pred["boxes"][i]
            box = np.r_[xyxy[:2], xyxy[2:] - xyxy[:2]].astype(float).tolist()
            row = dict(id=len(box_rows) + 1, image_id=image_id, category_id=int(source["category_id"]),
                       bbox=list(box), score=float(pred["scores"][i]), area=float(box[2] * box[3]), iscrowd=0)
            box_rows.append(row)
            if task == "segmentation":
                mask_rows.append({**row, "segmentation": source["segmentation"], "area": int(pred["masks"][i].sum())})
    ground_truth = dict(images=[dict(id=key, height=size[0], width=size[1]) for key, size in image_sizes.items()],
                        categories=[dict(id=key, name=str(key)) for key in sorted(classes)], annotations=ground_rows)
    bbox = _score_public_coco(ground_truth, box_rows, iou_type="bbox", max_dets=max_dets, backend=backend)
    if task == "segmentation":
        metrics = _score_public_coco(ground_truth, mask_rows, iou_type="segm", max_dets=max_dets, backend=backend)
    elif task == "keypoints":
        oks = evaluate_keypoints(annotation_file, [{key: value for key, value in row.items() if key != "bbox"}
                                 for row in records if "keypoints" in row], keypoint_oks_sigmas,
                                 aggregation="upstream_groups")
        metrics = dict(mAP=oks["map"], mAP_50=oks["map_50"], mAP_75=oks["map_75"], mAR_max=oks["mar"])
    else:
        metrics = bbox
    return {**metrics, **({f"bbox_{key}": value for key, value in bbox.items()} if task != "detection" else {}),
            **best_f1_metrics(matching)}
