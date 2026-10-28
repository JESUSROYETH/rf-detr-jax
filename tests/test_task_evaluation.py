"""Task decoding and COCO metrics, including original-size masks and missing keypoint groups."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pycocotools import mask as mask_util
from scipy.special import expit

from rfdetr_jax.evaluation.keypoint_oks import evaluate_keypoints
from rfdetr_jax.evaluation.tasks import accumulate_predictions, evaluate_predictions
from rfdetr_jax.models.postprocess import post_process_outputs


def _outputs():
    return {"pred_logits": np.array([[[-10, 10], [10, -10]]], np.float32),
            "pred_boxes": np.array([[[.5, .5, .5, .5], [.5, .5, 1, 1]]], np.float32)}


def _rle(mask):
    encoded = mask_util.encode(np.asfortranarray(mask, dtype=np.uint8))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def test_mask_decode_keeps_query_pairing_and_original_non_square_size():
    outputs = _outputs()
    outputs["pred_masks"] = np.array([[[[-5, 5], [-5, 5]], [[5, -5], [5, -5]]]], np.float32)
    result = post_process_outputs(outputs, [[4, 8]], num_select=2, category_ids=[1])[0]
    assert result["labels"].tolist() == [1]
    assert result["masks"].shape == (1, 4, 8)
    np.testing.assert_array_equal(result["masks"][0], np.tile(np.arange(8) >= 4, (4, 1)))
    np.testing.assert_allclose(result["boxes"][0], [2, 1, 6, 3])


def test_mask_decode_thresholds_after_interpolating_logits():
    outputs = _outputs()
    outputs["pred_masks"] = np.array([[[[-1, 3]], [[-1, -1]]]], np.float32)
    result = post_process_outputs(outputs, [[1, 4]], num_select=2, category_ids=[1])[0]
    # Half-pixel interpolation gives [-1,0,2,3], then >0; resizing a binary mask differs.
    np.testing.assert_array_equal(result["masks"][0], [[False, False, True, True]])


def test_keypoint_decode_routes_class_slots_and_upstream_uncertainty_fusion():
    outputs = _outputs()
    points = np.zeros((1, 2, 4, 8), np.float32)  # schema [0,2], maxK=2
    points[0, 0, 2:, :2] = [[.25, .5], [.75, .25]]
    points[0, 0, 2:, 2] = [0, 2]
    outputs["pred_keypoints"] = points
    result = post_process_outputs(outputs, [[100, 200]], num_select=2,
                                  num_keypoints_per_class=(0, 2), category_ids=[1])[0]
    np.testing.assert_allclose(result["keypoints"][0, :, :2], [[50, 50], [150, 25]])
    np.testing.assert_allclose(result["keypoints"][0, :, 2], expit([0, 2]), rtol=1e-6)
    np.testing.assert_allclose(result["keypoint_precision_cholesky"], 0)
    assert result["scores"][0] == pytest.approx(expit(np.log(expit(10)) - .2 * np.log(2)), rel=1e-6)


def test_official_keypoint_label_zero_maps_to_person_one():
    outputs = _outputs()
    outputs["pred_logits"] = outputs["pred_logits"][..., ::-1]
    outputs["pred_keypoints"] = np.zeros((1, 2, 2, 8), np.float32)
    outputs["pred_keypoints"][0, 0, :, :2] = [[.25, .5], [.75, .25]]
    records = accumulate_predictions([9], outputs, [[100, 200]], [1], task="keypoints", num_select=2,
                                     num_keypoints_per_class=(2,), label_to_category={"0": 1}, trace_alpha=0)
    assert len(records) == 1 and records[0]["category_id"] == 1
    assert records[0]["keypoints"][:3] == [50.0, 50.0, .5]


def test_keypoint_records_trim_class_padding():
    outputs = _outputs()
    outputs["pred_keypoints"] = np.zeros((1, 2, 6, 8), np.float32)
    records = accumulate_predictions([1], outputs, [[20, 30]], [1], task="keypoints", num_select=2,
                                     num_keypoints_per_class=(3, 1), trace_alpha=0)
    assert len(records) == 1 and len(records[0]["keypoints"]) == 3


def test_segmentation_records_roundtrip_and_perfect_mask_scores_one(tmp_path):
    outputs = _outputs()
    outputs["pred_masks"] = np.array([[[[-5, 5], [-5, 5]], [[5, -5], [5, -5]]]], np.float32)
    records = accumulate_predictions([1], outputs, [[4, 8]], [1], task="segmentation", num_select=2)
    mask = np.tile(np.arange(8) >= 4, (4, 1)).astype(np.uint8)
    np.testing.assert_array_equal(mask_util.decode(records[0]["segmentation"]), mask)
    assert isinstance(records[0]["segmentation"]["counts"], str)
    path = tmp_path / "seg.json"
    path.write_text(json.dumps({"images": [{"id": 1, "height": 4, "width": 8}],
                               "categories": [{"id": 1, "name": "object"}],
                               "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                                                "segmentation": _rle(mask), "area": 16,
                                                "bbox": [4, 0, 4, 4], "iscrowd": 0}]}))
    metrics = evaluate_predictions(path, records, task="segmentation", backend="pycocotools")
    assert metrics["mAP"] == pytest.approx(1)
    assert metrics["bbox_mAP"] < metrics["mAP"]
    records[0]["segmentation"] = _rle(1 - mask)
    assert evaluate_predictions(path, records, task="segmentation", backend="pycocotools")["mAP"] == 0
    assert evaluate_predictions(path, [], task="segmentation", backend="pycocotools")["mAP"] == 0


def _mixed_keypoint_ground_truth(tmp_path):
    categories = [{"id": 1, "name": "one", "keypoints": ["a"]},
                  {"id": 2, "name": "two", "keypoints": ["b"]},
                  {"id": 3, "name": "three", "keypoints": ["c", "d"]}]
    annotations = [{"id": c["id"], "image_id": 1, "category_id": c["id"], "bbox": [0, 0, 100, 100],
                    "area": 10000, "iscrowd": 0, "keypoints": [25, 25, 2] * len(c["keypoints"])}
                   for c in categories]
    path = tmp_path / "keypoints.json"
    path.write_text(json.dumps({"images": [{"id": 1, "width": 100, "height": 100}],
                               "categories": categories, "annotations": annotations}))
    return path


def test_missing_keypoint_group_counts_as_zero_and_categories_have_equal_weight(tmp_path):
    path = _mixed_keypoint_ground_truth(tmp_path)
    records = [{"image_id": 1, "category_id": 1, "score": .9, "keypoints": [25, 25, .9]}]
    metrics = evaluate_keypoints(path, records)
    assert metrics["map"] == pytest.approx(1 / 3)
    assert metrics["mar"] == pytest.approx(1 / 3)
    assert evaluate_keypoints(path, [])["map"] == 0
    records.extend([{"image_id": 1, "category_id": cat, "score": .9, "keypoints": [25, 25, .9] * count}
                    for cat, count in [(2, 1), (3, 2)]])
    assert evaluate_keypoints(path, records)["map"] == pytest.approx(1)


def test_keypoint_metadata_fallback_is_per_category(tmp_path):
    path = _mixed_keypoint_ground_truth(tmp_path)
    raw = json.loads(path.read_text())
    del raw["categories"][2]["keypoints"]
    path.write_text(json.dumps(raw))
    records = [{"image_id": 1, "category_id": 1, "score": .9, "keypoints": [25, 25, .9]}]
    assert evaluate_keypoints(path, records)["map"] == pytest.approx(1 / 3)


def test_keypoint_predictions_must_match_category_width(tmp_path):
    path = _mixed_keypoint_ground_truth(tmp_path)
    with pytest.raises(ValueError, match="category schema"):
        evaluate_keypoints(path, [{"image_id": 1, "category_id": 1, "score": .9, "keypoints": [1, 2, .9] * 2}])


def test_task_evaluation_refuses_a_missing_task_head():
    with pytest.raises(ValueError, match="pred_masks"):
        accumulate_predictions([1], _outputs(), [[10, 10]], [1], task="segmentation")


def test_empty_selection_preserves_mask_shape():
    outputs = _outputs()
    outputs["pred_masks"] = np.zeros((1, 2, 2, 2), np.float32)
    result = post_process_outputs(outputs, [[10, 20]], score_threshold=1)[0]
    assert result["masks"].shape == (0, 10, 20)


@pytest.mark.parametrize("task", ["masks", "keypoints"])
def test_postprocess_matches_pinned_upstream_fixture(task):
    """Frozen upstream outputs cover sparse class schemas and different original image sizes."""
    from pathlib import Path

    with np.load(Path(__file__).parent / "data" / "task_postprocess_reference.npz", allow_pickle=False) as data:
        head = {"pred_masks": data["masks"]} if task == "masks" else {"pred_keypoints": data["points"]}
        outputs = {"pred_logits": data["logits"], "pred_boxes": data["boxes"], **head}
        results = post_process_outputs(outputs, data["sizes"], num_select=7, num_keypoints_per_class=(1, 3, 2))
        for image, result in enumerate(results):
            for name, values in result.items():
                reference = data[f"{task}_{image}_{name}"]
                if name == "masks":
                    np.testing.assert_array_equal(values, reference[:, 0])
                else:
                    np.testing.assert_allclose(values, reference, rtol=2e-6, atol=2e-6, equal_nan=True)
