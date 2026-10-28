"""Real upstream matcher/sweep comparisons plus standalone failure regressions."""
import json

import numpy as np
import pytest

from rfdetr_jax.evaluation.f1_metrics import best_f1_metrics, build_matching_data, evaluate_coco_f1


def test_duplicate_and_missed_class_macro_average():
    pred = dict(boxes=np.array([[0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2]]),
                scores=np.array([.9, .8, .7]), labels=np.array([0, 0, 2]))
    target = dict(boxes=np.array([[0, 0, 2, 2], [0, 0, 2, 2]]), labels=np.array([0, 1]))
    matched = build_matching_data([pred], [target])
    assert matched[0]["matches"].tolist() == [1, 0]
    assert matched[1]["total_gt"] == 1
    # Class2 has predictions only and is excluded from upstream's macro average.
    assert best_f1_metrics(matched) == dict(F1=.5, precision=.5, recall=.5)


def test_crowd_is_fallback_after_noncrowd_and_can_ignore_multiple_predictions():
    pred = dict(boxes=np.tile([0, 0, 2, 2], (3, 1)), scores=np.array([.9, .8, .7]), labels=np.zeros(3))
    target = dict(boxes=np.tile([0, 0, 2, 2], (2, 1)), labels=np.zeros(2), iscrowd=np.array([1, 0]))
    result = build_matching_data([pred], [target])[0]
    assert result["matches"].tolist() == [1, 0, 0]
    assert result["ignore"].tolist() == [False, True, True]
    assert best_f1_metrics({0: result}) == dict(F1=1., precision=1., recall=1.)


def test_empty_metrics_and_shape_errors():
    assert best_f1_metrics({}) == dict(F1=0., precision=0., recall=0.)
    empty = dict(boxes=np.empty((0, 4)), labels=np.array([]), scores=np.array([]))
    with pytest.raises(ValueError, match="iscrowd"):
        build_matching_data([empty], [{**empty, "iscrowd": np.zeros((0, 1))}])
    with pytest.raises(ValueError):
        build_matching_data([empty], [])


@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
def test_actual_reference_matching_and_sweep(iou_type):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    pytest.importorskip("faster_coco_eval", minversion="1.7.2")
    from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds
    from rfdetr.evaluation.matching import build_matching_data as reference_match

    rng = np.random.default_rng(840)
    preds, targets = [], []
    for n, m in [(0, 3), (5, 0), (40, 12), (12, 12), (0, 0)]:
        pb, tb = rng.uniform(0, 4, (n, 4)).astype(np.float32), rng.uniform(0, 4, (m, 4)).astype(np.float32)
        pb[:, 2:] += pb[:, :2] + 1
        tb[:, 2:] += tb[:, :2] + 1
        p = dict(boxes=pb, labels=rng.integers(0, 3, n), scores=rng.choice([.2, .3, .7, .9], n).astype(np.float32))
        t = dict(boxes=tb, labels=rng.integers(0, 4, m), iscrowd=rng.integers(0, 2, m))
        p["masks"] = rng.uniform(size=(n, 8, 9)) > .2
        t["masks"] = rng.uniform(size=(m, 5, 7)) > .2
        preds.append(p)
        targets.append(t)
    expected = reference_match([{k: torch.from_numpy(v) for k, v in row.items()} for row in preds],
                               [{k: torch.from_numpy(v) for k, v in row.items()} for row in targets],
                               iou_type=iou_type)
    actual = build_matching_data(preds, targets, iou_type=iou_type)
    assert set(actual) == set(expected)
    for label in expected:
        for key in ("scores", "matches", "ignore", "total_gt"):
            np.testing.assert_array_equal(actual[label][key], expected[label][key])
    keys = sorted(expected)
    sweep = sweep_confidence_thresholds([expected[k] for k in keys], np.linspace(0, 1, 101),
                                       [i for i, k in enumerate(keys) if expected[k]["total_gt"]])
    best = max(sweep, key=lambda item: item["macro_f1"])
    assert best_f1_metrics(actual) == dict(F1=best["macro_f1"], precision=best["macro_precision"],
                                         recall=best["macro_recall"])


@pytest.mark.parametrize("task", ["detection", "segmentation", "keypoints"])
def test_coco_wrapper_drops_crowd_and_degenerate_gt_and_uses_masks(tmp_path, task):
    from pycocotools import mask as mask_util

    mask = np.zeros((6, 8), np.uint8)
    mask[1:5, 2:6] = 1
    rle = mask_util.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode()
    gt = dict(image_id=1, category_id=1, bbox=[2, 1, 4, 4], segmentation=rle, area=16)
    raw = dict(images=[dict(id=1, height=6, width=8)], categories=[dict(id=1, name="object")],
               annotations=[dict(gt, id=1, iscrowd=0), dict(gt, id=2, iscrowd=1),
                            dict(gt, id=3, bbox=[20, 20, 1, 1])])
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(raw))
    records = [dict(gt, score=.8)]
    assert evaluate_coco_f1(path, records, task=task) == dict(F1=1., precision=1., recall=1.)
    if task == "segmentation":
        # A matching box cannot hide a completely wrong segmentation.
        wrong = mask_util.encode(np.asfortranarray(1 - mask))
        wrong["counts"] = wrong["counts"].decode()
        records[0]["segmentation"] = wrong
        assert evaluate_coco_f1(path, records, task=task)["F1"] == 0.
    else:
        records.append(dict(gt, score=.8))
        assert evaluate_coco_f1(path, records, task=task)["F1"] == pytest.approx(2 / 3)


def test_float_boundary_and_nan_score_match_reference_sweep():
    pytest.importorskip("rfdetr")
    pytest.importorskip("faster_coco_eval", minversion="1.7.2")
    from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds
    item = dict(scores=np.array([np.float32(.7), np.nan, .9]), matches=np.array([0, 1, 1]),
                ignore=np.array([False, False, False]), total_gt=1)
    expected = max(sweep_confidence_thresholds([item], np.linspace(0, 1, 101), [0]), key=lambda x: x["macro_f1"])
    assert best_f1_metrics({3: item}) == dict(F1=expected["macro_f1"], precision=expected["macro_precision"],
                                            recall=expected["macro_recall"])


def test_mask_resize_roundtrip_matches_reference_public_targets(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    pytest.importorskip("faster_coco_eval", minversion="1.7.2")
    from pycocotools import mask as mask_util
    from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds
    from rfdetr.evaluation.matching import build_matching_data as reference_match

    mask = (np.random.default_rng(8).uniform(size=(5, 7)) > .5).astype(np.uint8)
    rle = mask_util.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode()
    target_mask = torch.nn.functional.interpolate(torch.tensor(mask)[None, None].float(),
                                                   size=(3, 3), mode="nearest")
    target_mask = torch.nn.functional.interpolate(target_mask, size=(5, 7), mode="nearest")[0].bool()
    predicted = target_mask[0].numpy().astype(np.uint8)
    predicted_rle = mask_util.encode(np.asfortranarray(predicted))
    predicted_rle["counts"] = predicted_rle["counts"].decode()
    row = dict(id=1, image_id=1, category_id=1, bbox=[0, 0, 7, 5], area=int(mask.sum()),
               segmentation=rle, iscrowd=0)
    path = tmp_path / "roundtrip.json"
    path.write_text(json.dumps(dict(images=[dict(id=1, height=5, width=7)],
                                    categories=[dict(id=1, name="object")], annotations=[row])))
    record = dict(row, segmentation=predicted_rle, score=.8)
    prediction = dict(boxes=torch.tensor([[0., 0., 7., 5.]]), scores=torch.tensor([.8]),
                      labels=torch.tensor([1]), masks=target_mask)
    target = dict(boxes=prediction["boxes"], labels=prediction["labels"], masks=target_mask)
    matching = reference_match([prediction], [target], iou_type="segm")
    expected = max(sweep_confidence_thresholds(list(matching.values()), np.linspace(0, 1, 101), [0]),
                   key=lambda x: x["macro_f1"])
    actual = evaluate_coco_f1(path, [record], task="segmentation", resolution=3)
    assert actual == dict(F1=expected["macro_f1"], precision=expected["macro_precision"],
                          recall=expected["macro_recall"])
    assert actual["F1"] == 1.
