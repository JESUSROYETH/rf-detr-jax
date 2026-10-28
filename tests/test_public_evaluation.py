"""Bounded comparisons with the published RF-DETR 1.11.2 callback and metrics."""
import contextlib
import io
import json

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_util
from pycocotools.coco import COCO

from rfdetr_jax.evaluation.public_evaluation import (
    convert_public_target,
    evaluate_public_predictions,
    public_ap_grids,
)


def _rle(mask):
    result = mask_util.encode(np.asfortranarray(mask, dtype=np.uint8))
    result["counts"] = result["counts"].decode()
    return result


@pytest.fixture
def reference():
    torch = pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    pytest.importorskip("faster_coco_eval", minversion="1.7.2")
    pytest.importorskip("torchmetrics", minversion="1.8.2")
    pytest.importorskip("pytorch_lightning")
    import importlib.metadata
    assert importlib.metadata.version("rfdetr") == "1.11.2", "public evaluation reference must be pinned to 1.11.2"
    from rfdetr.datasets._torchvision import Resize
    from rfdetr.datasets.coco import ConvertCoco, crowd_regions_from_coco
    from rfdetr.datasets.transforms import Normalize
    from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
    from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision

    return dict(torch=torch, Resize=Resize, ConvertCoco=ConvertCoco, Normalize=Normalize,
                crowd=crowd_regions_from_coco, callback=COCOEvalCallback, metric=OnePassCocoMeanAveragePrecision)


def _load(tmp_path, raw):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(raw))
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(path))
    return path, coco


def _reference_targets(reference, coco, resolution, segmentation, image_sizes=None):
    torch = reference["torch"]
    transformed = []
    for image_id, image in coco.imgs.items():
        h, w = image_sizes[image_id] if image_sizes else (image["height"], image["width"])
        pil = Image.new("RGB", (w, h))
        pil, target = reference["ConvertCoco"](include_masks=segmentation)(
            pil, dict(image_id=image_id, annotations=coco.imgToAnns[image_id]))
        _, target = reference["Resize"]((resolution, resolution))(pil, target)
        _, target = reference["Normalize"]()(torch.zeros((3, resolution, resolution)), target)
        transformed.append(target)
    callback = reference["callback"](segmentation=segmentation, eval_backend="faster_coco_eval")
    return callback._convert_targets(transformed, crowd_regions=reference["crowd"](coco))


def _annotation(image_id, bbox, mask, *, crowd=0, label=3):
    return dict(image_id=image_id, category_id=label, bbox=bbox, area=9999.,
                segmentation=_rle(mask), iscrowd=crowd)


def _fixture():
    rng = np.random.default_rng(61)
    annotations = [
        _annotation(1, [-1.25, .2, 6.7, 8.1], rng.random((9, 11)) > .3),
        _annotation(1, [9.2, 7.1, 4., 4.], rng.random((9, 11)) > .5),
        _annotation(1, [20., 20., 1., 1.], rng.random((9, 11)) > .5),
        _annotation(1, [-2., -1., 5., 5.], rng.random((9, 11)) > .2, crowd=1),
        _annotation(2, [1., 1., 3., 2.], rng.random((5, 7)) > .2, label=7),
    ]
    return dict(images=[dict(id=1, height=9, width=11), dict(id=2, height=5, width=7)],
                categories=[dict(id=3, name="a"), dict(id=7, name="b")],
                annotations=[dict(row, id=index + 1) for index, row in enumerate(annotations)])


@pytest.mark.parametrize("segmentation", [False, True])
def test_targets_match_actual_published_callback(reference, tmp_path, segmentation):
    _, coco = _load(tmp_path, _fixture())
    expected = _reference_targets(reference, coco, 4, segmentation)
    for (image_id, image), target in zip(coco.imgs.items(), expected, strict=True):
        actual = convert_public_target(coco, image_id, resolution=4,
                                        image_size=(image["height"], image["width"]), segmentation=segmentation)
        for name, value in target.items():
            np.testing.assert_allclose(actual[name], value.numpy(), atol=1e-6, rtol=1e-6, err_msg=name)
    assert expected[0]["boxes"][-1].tolist() == [-2., -1., 3., 4.]
    assert len(expected[0]["boxes"]) == 3  # degenerate ordinary box removed, crowd restored


def _compare_metrics(reference, path, targets, records, task, *, max_dets=500, backend="faster_coco_eval"):
    torch = reference["torch"]
    from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds
    from rfdetr.evaluation.matching import build_matching_data

    predictions = []
    for image_id, target in enumerate(targets, 1):
        rows = [row for row in records if row["image_id"] == image_id]
        boxes = torch.tensor([row["bbox"] for row in rows], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        for index, row in enumerate(rows):
            if "_boxes_xyxy" in row:
                boxes[index] = torch.tensor(row["_boxes_xyxy"], dtype=torch.float32)
        pred = dict(boxes=boxes, scores=torch.tensor([row["score"] for row in rows]),
                    labels=torch.tensor([row["category_id"] for row in rows], dtype=torch.int64))
        if task == "segmentation":
            pred["masks"] = (torch.from_numpy(np.stack([
                mask_util.decode(row["segmentation"]) for row in rows])).bool().contiguous()
                if rows else torch.zeros((0, *target["masks"].shape[-2:]), dtype=torch.bool))
        predictions.append(pred)
    metric = reference["metric"](iou_type=("bbox", "segm") if task == "segmentation" else "bbox",
                                  max_detection_thresholds=[1, 10, max_dets],
                                  backend=backend, sync_on_compute=False)
    metric.update(predictions, targets)
    expected = metric.compute()
    actual = evaluate_public_predictions(path, records, task=task, resolution=4, max_dets=max_dets)
    for ours, theirs in [("mAP", "map"), ("mAP_50", "map_50"), ("mAP_75", "map_75"),
                         ("mAP_small", "map_small"), ("mAP_medium", "map_medium"),
                         ("mAP_large", "map_large"), ("mAR_max", f"mar_{max_dets}")]:
        if task == "segmentation":
            assert actual[ours] == pytest.approx(float(expected["segm_" + theirs]), abs=1e-7)
            assert actual["bbox_" + ours] == pytest.approx(float(expected["bbox_" + theirs]), abs=1e-7)
        else:
            assert actual[ours] == pytest.approx(float(expected[theirs]), abs=1e-7)
    matched = build_matching_data(predictions, targets, iou_type="segm" if task == "segmentation" else "bbox")
    keys = sorted(matched)
    best = max(sweep_confidence_thresholds([matched[k] for k in keys], np.linspace(0, 1, 101),
                                          [i for i, k in enumerate(keys) if matched[k]["total_gt"]]),
               key=lambda item: item["macro_f1"])
    for ours, theirs in [("F1", "macro_f1"), ("precision", "macro_precision"), ("recall", "macro_recall")]:
        assert actual[ours] == pytest.approx(best[theirs], abs=1e-12)
    return actual


@pytest.mark.parametrize("task", ["detection", "segmentation"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("backend", ["faster_coco_eval", "vernier"])
def test_ap_f1_actual_callback_with_crowds_areas_and_masks(reference, tmp_path, task, empty, backend):
    if backend == "vernier":
        pytest.importorskip("vernier", minversion="0.5.3")
    path, coco = _load(tmp_path, _fixture())
    targets = _reference_targets(reference, coco, 4, task == "segmentation")
    records = []
    if not empty:
        for image_id, target in enumerate(targets, 1):
            # Duplicate crowd-overlapping rows exercise public F1's restored crowd handling.
            for i in [0, len(target["boxes"]) - 1, len(target["boxes"]) - 1]:
                box = target["boxes"][i].numpy()
                row = dict(image_id=image_id, category_id=int(target["labels"][i]), score=.8,
                           bbox=np.r_[box[:2], box[2:] - box[:2]].astype(float).tolist())
                if task == "segmentation":
                    row["segmentation"] = _rle(target["masks"][i].numpy())
                records.append(row)
    _compare_metrics(reference, path, targets, records, task, backend=backend)


def test_ap_recall_grid_boundary_uses_torch_not_numpy(reference, tmp_path):
    raw = dict(images=[dict(id=1, height=20, width=100)], categories=[dict(id=1, name="a")],
               annotations=[dict(id=i + 1, image_id=1, category_id=1, bbox=[i * 8, 0, 4, 4],
                                 area=16, iscrowd=0) for i in range(10)])
    path, coco = _load(tmp_path, raw)
    targets = _reference_targets(reference, coco, 4, False)
    records = [dict(image_id=1, category_id=1, bbox=row["bbox"], score=.9) for row in raw["annotations"][:3]]
    actual = _compare_metrics(reference, path, targets, records, "detection", max_dets=20)
    assert actual["mAP"] == pytest.approx(31 / 101, abs=1e-7)
    _, grid = public_ap_grids()
    np.testing.assert_array_equal(grid, reference["torch"].linspace(0, 1, 101).numpy().astype(float))
    assert not np.array_equal(grid, np.linspace(0, 1, 101, dtype=np.float32).astype(float))


def test_empty_gt_is_unavailable_not_zero(reference, tmp_path):
    path, coco = _load(tmp_path, dict(images=[dict(id=1, height=5, width=7)],
                                     categories=[dict(id=1, name="a")], annotations=[]))
    targets = _reference_targets(reference, coco, 4, False)
    actual = _compare_metrics(reference, path, targets, [], "detection")
    assert actual["mAP"] == -1 and actual["F1"] == 0


def test_keypoint_groups_match_published_original_gt_metric(reference, tmp_path):
    from faster_coco_eval import COCO as ReferenceCOCO
    from rfdetr.evaluation.keypoint_oks import MetricKeypointOKS

    from rfdetr_jax.evaluation.keypoint_oks import evaluate_keypoints

    torch = reference["torch"]
    raw = dict(images=[dict(id=1, height=32, width=32)],
               categories=[dict(id=1, name="a", keypoints=["a", "b"]),
                           dict(id=2, name="empty", keypoints=["a", "b"]),
                           dict(id=3, name="c", keypoints=["a", "b", "c"])],
               annotations=[dict(id=1, image_id=1, category_id=1, bbox=[1, 1, 10, 10], area=100,
                                 keypoints=[2, 2, 2, 4, 4, 2], num_keypoints=2, iscrowd=0),
                            dict(id=2, image_id=1, category_id=3, bbox=[1, 1, 10, 10], area=100,
                                 keypoints=[2, 2, 2, 4, 4, 2, 6, 6, 2], num_keypoints=3, iscrowd=0)])
    path, _ = _load(tmp_path, raw)
    with contextlib.redirect_stdout(io.StringIO()):
        metric = MetricKeypointOKS(ReferenceCOCO(str(path)))
    metric.update({1: dict(boxes=torch.tensor([[1., 1., 11., 11.]]), scores=torch.tensor([.9]),
                           labels=torch.tensor([1]), keypoints=torch.tensor([
                               [[2., 2., 1.], [4., 4., 1.], [0., 0., 0.]]]))})
    expected = metric.compute()
    records = [dict(image_id=1, category_id=1, bbox=[1., 1., 10., 10.], score=.9,
                    keypoints=[2., 2., 1., 4., 4., 1.])]
    actual = evaluate_public_predictions(path, records, task="keypoints", resolution=4)
    for ours, theirs in [("mAP", "map"), ("mAP_50", "map_50"), ("mAP_75", "map_75"), ("mAR_max", "mar")]:
        assert actual[ours] == pytest.approx(expected[theirs], abs=1e-7)
    assert actual["mAP"] == pytest.approx(2 / 3, abs=1e-7)
    assert evaluate_keypoints(path, records)["map"] == pytest.approx(.5)  # raw CLI convention preserved


def test_sizes_require_every_valid_image(tmp_path):
    path, _ = _load(tmp_path, _fixture())
    with pytest.raises(ValueError, match="cover exactly"):
        evaluate_public_predictions(path, [], resolution=4, image_sizes={1: (9, 11)})


@pytest.mark.parametrize("budget", [1, 5, 20, 500])
def test_configured_budget_is_distinct_from_selected_output_count(reference, tmp_path, budget):
    raw = dict(images=[dict(id=1, height=16, width=300)], categories=[dict(id=1, name="a")],
               annotations=[dict(id=i + 1, image_id=1, category_id=1, bbox=[i * 8, 0, 4, 4],
                                 area=16, iscrowd=0) for i in range(30)])
    path, coco = _load(tmp_path, raw)
    targets = _reference_targets(reference, coco, 4, False)
    records = [dict(image_id=1, category_id=1, bbox=row["bbox"], score=.9 - index / 100)
               for index, row in enumerate(raw["annotations"])]
    actual = _compare_metrics(reference, path, targets, records, "detection", max_dets=budget)
    assert actual["mAR_max"] == pytest.approx(min(budget, 30) / 30, abs=1e-7)


def test_empty_split_is_unavailable(reference, tmp_path):
    path, _ = _load(tmp_path, dict(images=[], annotations=[], categories=[dict(id=1, name="a")]))
    actual = evaluate_public_predictions(path, [], resolution=4)
    assert actual["mAP"] == -1 and actual["mAR_max"] == -1 and actual["F1"] == 0


@pytest.mark.parametrize("crowd", [False, True])
def test_polygon_uses_actual_image_grid_but_crowd_uses_metadata(reference, tmp_path, crowd):
    raw = dict(images=[dict(id=1, height=9, width=11)], categories=[dict(id=1, name="a")],
               annotations=[dict(id=1, image_id=1, category_id=1, bbox=[1, 1, 3, 3],
                                 area=9, iscrowd=int(crowd), segmentation=[[1, 1, 4, 1, 4, 4, 1, 4]])])
    path, coco = _load(tmp_path, raw)
    sizes = {1: (5, 7)}
    expected = _reference_targets(reference, coco, 4, True, sizes)[0]
    actual = convert_public_target(coco, 1, resolution=4, image_size=sizes[1], segmentation=True)
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name], value.numpy(), atol=1e-6, rtol=1e-6, err_msg=name)
    result = evaluate_public_predictions(path, [], task="segmentation", resolution=4, image_sizes=sizes)
    assert result["mAP"] == (-1 if crowd else 0)


@pytest.mark.parametrize("count,visible,crowd,sigmas,predicted", [
    (2, False, False, None, True),
    (2, True, True, None, True),
    (2, True, False, [.1, .2], True),
    (17, True, False, None, True),
    (2, True, False, None, False),
])
def test_keypoint_original_area_visibility_crowd_and_sigmas(
        reference, tmp_path, count, visible, crowd, sigmas, predicted):
    from faster_coco_eval import COCO as ReferenceCOCO
    from rfdetr.evaluation.keypoint_oks import MetricKeypointOKS

    torch = reference["torch"]
    points = [[2. + i, 3. + i, 2. if visible else 0.] for i in range(count)]
    raw = dict(images=[dict(id=1, height=40, width=40)],
               categories=[dict(id=3, name="joints", keypoints=[str(i) for i in range(count)])],
               annotations=[dict(id=1, image_id=1, category_id=3, bbox=[1, 1, 22, 22],
                                 area=19, iscrowd=int(crowd), keypoints=np.ravel(points).tolist(),
                                 num_keypoints=count if visible else 0)])
    path, _ = _load(tmp_path, raw)
    with contextlib.redirect_stdout(io.StringIO()):
        metric = MetricKeypointOKS(ReferenceCOCO(str(path)), keypoint_oks_sigmas=sigmas)
    shifted = [[x + .4, y + .4, 1.] for x, y, _ in points]
    n = int(predicted)
    metric.update({1: dict(boxes=torch.tensor([[1., 1., 23., 23.]] * n).reshape(n, 4),
                           labels=torch.tensor([3] * n, dtype=torch.int64), scores=torch.tensor([.9] * n),
                           keypoints=torch.tensor([shifted] * n).reshape(n, count, 3))})
    expected = metric.compute()
    records = [dict(image_id=1, category_id=3, bbox=[1., 1., 22., 22.], score=.9,
                    keypoints=np.ravel(shifted).tolist())] if predicted else []
    actual = evaluate_public_predictions(path, records, task="keypoints", resolution=4,
                                         keypoint_oks_sigmas=sigmas)
    for ours, theirs in [("mAP", "map"), ("mAP_50", "map_50"), ("mAP_75", "map_75"), ("mAR_max", "mar")]:
        assert actual[ours] == pytest.approx(expected[theirs], abs=1e-7)



def test_public_ap_grids_ignore_torch_default_device(reference):
    public_ap_grids.cache_clear()
    try:
        with reference["torch"].device("meta"):
            iou, recall = public_ap_grids()
        assert iou.shape == (10,) and recall.shape == (101,)
        assert not iou.flags.writeable and not recall.flags.writeable
        assert recall[-1] == 1
    finally:
        public_ap_grids.cache_clear()



def test_public_records_preserve_original_endpoints_at_f1_boundary(reference, tmp_path):
    from rfdetr_jax.evaluation.coco_eval import accumulate_detections

    raw = dict(images=[dict(id=1, height=256, width=256)], categories=[dict(id=1, name="a")],
               annotations=[dict(id=1, image_id=1, category_id=1,
                                 bbox=[40.63359832763672, 0., 73.51924896240234, 1.], area=73.5, iscrowd=0)])
    path, coco = _load(tmp_path, raw)
    box = np.array([[[16.127178192138672, 0., 89.64643096923828, 1.]]], np.float32)
    args = (np.array([1]), np.array([[.9]]), np.array([[1]]), box, [1])
    records = accumulate_detections(*args, preserve_xyxy=True)
    np.testing.assert_array_equal(records[0]["_boxes_xyxy"], box[0, 0])
    assert "_boxes_xyxy" not in accumulate_detections(*args)[0]
    targets = _reference_targets(reference, coco, 4, False)
    actual = _compare_metrics(reference, path, targets, records, "detection")
    assert actual["F1"] == 0
    legacy = [{key: value for key, value in row.items() if key != "_boxes_xyxy"} for row in records]
    assert evaluate_public_predictions(path, legacy, resolution=4)["F1"] == 1
