"""Translate backend evaluation measurements into the public RF-DETR namespace."""
from __future__ import annotations


def format_public_metrics(metrics: dict[str, float], split: str, task: str) -> dict[str, float]:
    """Use box AP as the headline and separate mask/OKS measurements, as upstream does.

    The backend deliberately reports its requested task as ``mAP``. The public
    facade has a different contract: its headline is always box AP, including
    segmentation and keypoints. Missing companion measurements must fail instead
    of accidentally relabeling mask AP or OKS AP as box AP. F1 is not derived from
    AP; only a separately computed threshold sweep may supply it.
    """
    if task not in {"detection", "segmentation", "keypoints"}:
        raise ValueError(f"unknown task {task!r}")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown evaluation split {split!r}")
    prefix = "" if task == "detection" else "bbox_"
    names = {"mAP": "mAP_50_95", "mAP_50": "mAP_50", "mAP_75": "mAP_75", "mAR_max": "mAR"}
    result = {f"{split}/{target}": float(metrics[prefix + source]) for source, target in names.items()}
    if task == "segmentation":
        result.update({f"{split}/segm_mAP_50_95": float(metrics["mAP"]),
                       f"{split}/segm_mAP_50": float(metrics["mAP_50"])})
    elif task == "keypoints":
        for source, target in {"mAP": "keypoint_map_50_95", "mAP_50": "keypoint_map_50",
                               "mAP_75": "keypoint_map_75", "mAR_max": "keypoint_mAR"}.items():
            # Upstream does not log unavailable keypoint metrics (-1).
            if float(metrics[source]) >= 0:
                result[f"{split}/{target}"] = float(metrics[source])
    for name in ("F1", "precision", "recall"):
        if name in metrics:
            result[f"{split}/{name}"] = float(metrics[name])
    return result
