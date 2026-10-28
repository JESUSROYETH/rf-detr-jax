"""OKS keypoint metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rfdetr_jax.evaluation.keypoint_oks import (
    COCO_PERSON_KEYPOINT_SIGMAS,
    accumulate_keypoint_detections,
    evaluate_keypoints,
    resolve_keypoint_oks_sigmas,
)


def _person_keypoints(rng: np.random.RandomState) -> list[float]:
    """Return 17 visible keypoints scattered inside a 100x100 image."""
    points = []
    for index in range(17):
        points += [10.0 + index * 4.0, 20.0 + (index % 5) * 6.0, 2.0]
    del rng
    return points


@pytest.fixture
def annotations(tmp_path: Path) -> Path:
    """Write a one-image, one-person COCO keypoint ground truth."""
    keypoints = _person_keypoints(np.random.RandomState(0))
    payload = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [5, 15, 75, 45],
                "area": 3375,
                "iscrowd": 0,
                "keypoints": keypoints,
                "num_keypoints": 17,
            }
        ],
        "categories": [{"id": 1, "name": "person", "keypoints": [f"k{i}" for i in range(17)], "skeleton": []}],
    }
    path = tmp_path / "kp.json"
    path.write_text(json.dumps(payload))
    return path


def test_person_layout_uses_coco_sigmas() -> None:
    """A 17-point layout falls back to COCO's published person sigmas."""
    assert resolve_keypoint_oks_sigmas(17, None) == pytest.approx(list(COCO_PERSON_KEYPOINT_SIGMAS))


def test_other_layouts_use_a_uniform_sigma() -> None:
    """A layout COCO publishes no sigmas for gets a uniform fallback."""
    assert resolve_keypoint_oks_sigmas(4, None) == [0.05, 0.05, 0.05, 0.05]


@pytest.mark.parametrize("bad", [[], [0.1, -0.2], [float("nan"), 0.1], [0.1]])
def test_invalid_sigmas_are_rejected(bad: list[float]) -> None:
    """Empty, non-positive, non-finite or too-short sigma vectors are errors."""
    with pytest.raises(ValueError):
        resolve_keypoint_oks_sigmas(2, bad)


def test_accumulate_filters_and_flattens() -> None:
    """Unknown categories are dropped and keypoints are flattened to COCO triplets."""
    records = accumulate_keypoint_detections(
        np.array([7]),
        np.array([[0.9, 0.8]]),
        np.array([[1, 99]]),
        np.zeros((1, 2, 3, 3), dtype="float32"),
        category_ids=[1],
    )
    assert len(records) == 1
    assert records[0]["image_id"] == 7 and records[0]["category_id"] == 1
    assert len(records[0]["keypoints"]) == 9


def test_exact_keypoints_score_one(annotations: Path) -> None:
    """Predicting the ground-truth keypoints exactly gives OKS mAP 1.0."""
    keypoints = json.loads(annotations.read_text())["annotations"][0]["keypoints"]
    detections = [{"image_id": 1, "category_id": 1, "keypoints": keypoints, "score": 0.99}]
    metrics = evaluate_keypoints(annotations, detections)
    assert metrics["map"] == pytest.approx(1.0)
    assert metrics["mar"] == pytest.approx(1.0)


def test_far_off_keypoints_score_zero(annotations: Path) -> None:
    """Keypoints displaced far outside the person score zero."""
    keypoints = json.loads(annotations.read_text())["annotations"][0]["keypoints"]
    shifted = list(keypoints)
    shifted[0::3] = [value + 500.0 for value in keypoints[0::3]]
    detections = [{"image_id": 1, "category_id": 1, "keypoints": shifted, "score": 0.99}]
    assert evaluate_keypoints(annotations, detections)["map"] == pytest.approx(0.0)


def test_tighter_sigmas_are_stricter(annotations: Path) -> None:
    """Shrinking the sigmas lowers the score for the same slightly-off prediction."""
    keypoints = json.loads(annotations.read_text())["annotations"][0]["keypoints"]
    nudged = list(keypoints)
    nudged[0::3] = [value + 6.0 for value in keypoints[0::3]]
    detections = [{"image_id": 1, "category_id": 1, "keypoints": nudged, "score": 0.99}]
    loose = evaluate_keypoints(annotations, detections, [0.5] * 17)["map"]
    tight = evaluate_keypoints(annotations, detections, [0.01] * 17)["map"]
    assert loose > tight


def test_no_detections_scores_zero(annotations: Path) -> None:
    """An empty prediction set is reported as zero, not as an error."""
    assert evaluate_keypoints(annotations, [])["map"] == 0.0
