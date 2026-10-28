"""COCO metric computation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rfdetr_jax.evaluation.coco_eval import accumulate_detections, evaluate_detections


@pytest.fixture
def annotations(tmp_path: Path) -> Path:
    """Write a two-image ground truth with one box each."""
    payload = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 5, "bbox": [20, 20, 30, 30], "area": 900, "iscrowd": 0},
        ],
        "categories": [{"id": 3, "name": "x"}, {"id": 5, "name": "y"}],
    }
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(payload))
    return path


def test_accumulate_filters_unknown_categories() -> None:
    """Head slots with no matching dataset category never reach the evaluator."""
    records = accumulate_detections(
        np.array([1]),
        np.array([[0.9, 0.8]]),
        np.array([[3, 4]]),
        np.array([[[0, 0, 10, 10], [0, 0, 10, 10]]], dtype="float32"),
        category_ids=[3, 5],
    )
    assert [record["category_id"] for record in records] == [3]
    assert records[0]["bbox"] == [0.0, 0.0, 10.0, 10.0]


def test_accumulate_applies_score_threshold() -> None:
    """Detections at or below the threshold are dropped."""
    records = accumulate_detections(
        np.array([1]), np.array([[0.9, 0.1]]), np.array([[3, 3]]),
        np.zeros((1, 2, 4), dtype="float32"), category_ids=[3], score_threshold=0.5,
    )
    assert len(records) == 1


def test_perfect_predictions_score_one(annotations: Path) -> None:
    """Exact ground-truth boxes score mAP 1.0 at every IoU threshold."""
    detections = [
        {"image_id": 1, "category_id": 3, "bbox": [10, 10, 40, 40], "score": 0.99},
        {"image_id": 2, "category_id": 5, "bbox": [20, 20, 30, 30], "score": 0.99},
    ]
    metrics = evaluate_detections(annotations, detections, max_dets=300)
    assert metrics["mAP"] == pytest.approx(1.0)
    assert metrics["mAP_50"] == pytest.approx(1.0)
    assert metrics["mAR_max"] == pytest.approx(1.0)


def test_mAP_is_defined_at_a_non_default_detection_budget(annotations: Path) -> None:
    """`mAP` is a real number even when 100 is not among the configured maxDets."""
    detections = [{"image_id": 1, "category_id": 3, "bbox": [10, 10, 40, 40], "score": 0.9}]
    metrics = evaluate_detections(annotations, detections, max_dets=300)
    assert metrics["mAP"] >= 0.0


def test_no_detections_scores_zero(annotations: Path) -> None:
    """An empty prediction set is reported as zero, not as an error."""
    assert evaluate_detections(annotations, [])["mAP"] == 0.0


def test_small_detection_budget_does_not_include_lower_ranked_predictions(annotations: Path) -> None:
    """A budget of one must exclude a correct box ranked behind a false positive."""
    detections = [
        {"image_id": 1, "category_id": 3, "bbox": [70, 70, 10, 10], "score": 0.99},
        {"image_id": 1, "category_id": 3, "bbox": [10, 10, 40, 40], "score": 0.9},
    ]
    metrics = evaluate_detections(annotations, detections, max_dets=1, backend="pycocotools")
    assert metrics["mAP"] == 0.0
    assert metrics["mAR_max"] == 0.0


def test_ground_truth_reload_tracks_file_changes(annotations: Path) -> None:
    """A rewritten annotation file must not be scored against a cached ground truth."""
    detections = [{"image_id": 1, "category_id": 3, "bbox": [10.0, 10.0, 40.0, 40.0], "score": 0.9},
                  {"image_id": 2, "category_id": 5, "bbox": [20.0, 20.0, 30.0, 30.0], "score": 0.9}]
    first = evaluate_detections(annotations, detections, backend="pycocotools")["mAP"]
    content = json.loads(annotations.read_text())
    for row in content["annotations"]:
        row["bbox"] = [100.0, 100.0, 5.0, 5.0]
        row["area"] = 25.0
    annotations.write_text(json.dumps(content))
    second = evaluate_detections(annotations, detections, backend="pycocotools")["mAP"]
    assert first > 0.99 and second == 0.0
