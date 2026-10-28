"""Shared detection, segmentation and keypoint evaluation boundaries for the CLI/trainer."""

from __future__ import annotations

import numpy as np

from rfdetr_jax.evaluation.coco_eval import evaluate_detections
from rfdetr_jax.evaluation.keypoint_oks import evaluate_keypoints
from rfdetr_jax.models.postprocess import post_process_outputs

_TASKS = {"detection", "segmentation", "keypoints"}


def accumulate_predictions(
    image_ids,
    outputs: dict,
    target_sizes,
    category_ids,
    *,
    task: str = "detection",
    num_select: int = 300,
    num_keypoints_per_class=(),
    trace_alpha: float = 0.2,
    score_threshold: float = 0.0,
    label_to_category=None,
    preserve_xyxy: bool = False,
) -> list[dict]:
    """Decode a valid (unpadded) batch into COCO records with original-size masks/keypoints."""
    if task not in _TASKS:
        raise ValueError(f"unknown task {task!r}")
    required = {"segmentation": "pred_masks", "keypoints": "pred_keypoints"}.get(task)
    if required is not None and required not in outputs:
        raise ValueError(f"{task} evaluation requires {required}")
    mapping = None if label_to_category is None else {int(key): int(value) for key, value in label_to_category.items()}
    if mapping is not None and (
        len(set(mapping.values())) != len(mapping) or not set(mapping.values()) <= set(category_ids)
    ):
        raise ValueError("label_to_category must map distinct model labels to dataset category IDs")
    allowed_labels = category_ids if mapping is None else list(mapping)
    results = post_process_outputs(outputs, target_sizes, num_select, num_keypoints_per_class,
                                   trace_alpha, score_threshold, allowed_labels)
    if len(image_ids) != len(results):
        raise ValueError("image_ids must match the prediction batch")
    records = []
    for image_id, result in zip(image_ids, results, strict=True):
        entries = zip(result["scores"], result["labels"], result["boxes"], strict=True)
        for index, (score, label, box) in enumerate(entries):
            record = {"image_id": int(image_id), "category_id": int(label) if mapping is None else mapping[int(label)],
                      "score": float(score),
                      "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]}
            if preserve_xyxy:
                record["_boxes_xyxy"] = np.asarray(box).astype(float).tolist()
            if task == "segmentation":
                from pycocotools import mask as mask_util

                encoded = mask_util.encode(np.asfortranarray(result["masks"][index], dtype=np.uint8))
                encoded["counts"] = encoded["counts"].decode("ascii")
                record["segmentation"] = encoded
            elif task == "keypoints":
                count = num_keypoints_per_class[int(label)] if int(label) < len(num_keypoints_per_class) else 0
                if count:
                    record["keypoints"] = result["keypoints"][index, :count].reshape(-1).astype(float).tolist()
            records.append(record)
    return records


def evaluate_predictions(
    annotation_file,
    records: list[dict],
    *,
    task: str = "detection",
    max_dets: int = 300,
    backend: str = "auto",
    keypoint_oks_sigmas=None,
) -> dict[str, float]:
    """Return task mAP and, for masks/keypoints, companion ``bbox_*`` metrics.

    Keypoint AP uses COCO's fixed 20-instance budget and macro-averages categories
    with labeled ground truth, including categories without predictions.
    """
    if task not in _TASKS:
        raise ValueError(f"unknown task {task!r}")
    if task == "detection":
        return evaluate_detections(annotation_file, records, max_dets, backend)
    boxes = [{key: value for key, value in row.items() if key in {"image_id", "category_id", "score", "bbox"}}
             for row in records]
    box_metrics = evaluate_detections(annotation_file, boxes, max_dets, backend)
    if task == "segmentation":
        metrics = evaluate_detections(annotation_file, records, max_dets, backend, iou_type="segm")
    else:
        points = [{key: value for key, value in row.items() if key != "bbox"}
                  for row in records if "keypoints" in row]
        oks = evaluate_keypoints(annotation_file, points, keypoint_oks_sigmas)
        metrics = {"mAP": oks["map"], "mAP_50": oks["map_50"], "mAP_75": oks["map_75"], "mAR_max": oks["mar"]}
    return {**metrics, **{f"bbox_{key}": value for key, value in box_metrics.items()}}
