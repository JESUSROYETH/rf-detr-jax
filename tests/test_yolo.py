"""YOLO label edge cases and independently exercised upstream parsing parity."""
import json

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_util

from rfdetr_jax.datasets.yolo import convert_yolo_split, read_yolo_config, yolo_pose_schema


def dataset(tmp_path, label, config="names: [object]\n", split="train"):
    (tmp_path / "data.yaml").write_text(config)
    images, labels = tmp_path / split / "images", tmp_path / split / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (20, 10)).save(images / "a.png")
    Image.new("RGB", (20, 10)).save(images / "background.png")
    (labels / "a.txt").write_text(label)
    return tmp_path


@pytest.mark.parametrize("segmentation", [False, True])
def test_boxes_polygons_background_and_untouched_labels(tmp_path, segmentation):
    label = "0 .5 .5 .4 .6\n0 .1 .2 .8 .2 .5 .9\n"
    root = dataset(tmp_path, label)
    images, index = convert_yolo_split(root, "train", tmp_path / "index.json", segmentation=segmentation)
    raw = json.loads(index.read_text())
    assert images == root / "train" / "images"
    assert len(raw["images"]) == 2 and len(raw["annotations"]) == 2
    np.testing.assert_allclose(raw["annotations"][0]["bbox"], [6, 2, 8, 6], atol=1e-6)
    if segmentation:
        mask = mask_util.decode(raw["annotations"][0]["segmentation"])
        assert mask.shape == (10, 20) and mask[2:9, 6:15].all()
    else:
        assert "segmentation" not in raw["annotations"][0]
    assert (root / "train" / "labels" / "a.txt").read_text() == label


@pytest.mark.parametrize("dim,row,expected", [
    (2, "-.1 .5 .4 .8", [[0, 0, 0], [8, 8, 2]]),
    (3, "-.1 1.2 1 .4 .8 0", [[0, 10, 1], [0, 0, 0]]),
])
def test_pose_absence_clamping_schema_and_flips(tmp_path, dim, row, expected):
    root = dataset(tmp_path, "0 .5 .5 .4 .6 " + row,
                   f"names: {{0: person}}\nkpt_shape: [2, {dim}]\nflip_idx: [1, 0]\n")
    _, index = convert_yolo_split(root, "train", tmp_path / "index.json", keypoints=True)
    raw = json.loads(index.read_text())
    np.testing.assert_allclose(np.array(raw["annotations"][0]["keypoints"]).reshape(2, 3), expected)
    assert raw["annotations"][0]["num_keypoints"] == 1
    assert yolo_pose_schema(root)["flip_pairs"] == [0, 1]


@pytest.mark.parametrize("row", ["0 .1", "2 .5 .5 .1 .1", "0 nan .5 .1 .1", "0 .1 .2 .3 .4 .5"])
def test_invalid_rows_fail_with_source_location(tmp_path, row):
    root = dataset(tmp_path, row)
    with pytest.raises(ValueError, match=r"a.txt:1"):
        convert_yolo_split(root, "train", tmp_path / "index.json")


def test_yaml_split_priority_and_missing_test(tmp_path):
    root = dataset(tmp_path, "0 .5 .5 .4 .6", "names: [object]\nval: val/images\n", split="val")
    images, _ = convert_yolo_split(root, "val", tmp_path / "index.json")
    assert images == root / "val" / "images"
    with pytest.raises(FileNotFoundError):
        convert_yolo_split(root, "test", tmp_path / "index.json")
    (root / "data.yaml").write_text("names: [object]\ntest: ../escape/images\n")
    with pytest.raises(ValueError, match="test split"):
        convert_yolo_split(root, "test", tmp_path / "index.json")


def test_names_mapping_rejects_sparse_and_accepts_string_keys(tmp_path):
    path = tmp_path / "data.yml"
    path.write_text("names: {'0': first, '1': second}\n")
    assert read_yolo_config(tmp_path)["names"] == ["first", "second"]
    path.write_text("names: {0: first, 2: second}\n")
    with pytest.raises(ValueError, match="contiguous"):
        read_yolo_config(tmp_path)


@pytest.mark.parametrize("task", ["detection", "segmentation", "keypoints"])
def test_reference_parser_parity(tmp_path, task):
    pytest.importorskip("rfdetr")
    from rfdetr.datasets.yolo import _build_yolo_samples, _polygons_to_masks

    pose = task == "keypoints"
    mask = task == "segmentation"
    label = "0 .5 .5 .4 .6 -.1 .4 1 .8 .9 2" if pose else "0 .1 .2 .8 .2 .5 .9"
    root = dataset(tmp_path, label, "names: [object]\nkpt_shape: [2, 3]\n")
    _, index = convert_yolo_split(root, "train", root / "index.json", segmentation=mask, keypoints=pose)
    _, samples = _build_yolo_samples(str(root / "train/images"), str(root / "train/labels"),
                                    str(root / "data.yaml"), include_polygons=mask, include_keypoints=pose)
    annotation = json.loads(index.read_text())["annotations"][0]
    box = samples[0].xyxy[0]
    np.testing.assert_allclose(annotation["bbox"], [box[0], box[1], box[2] - box[0], box[3] - box[1]])
    if pose:
        np.testing.assert_allclose(annotation["keypoints"], samples[0].keypoints.ravel())
    if mask:
        np.testing.assert_array_equal(mask_util.decode(annotation["segmentation"]),
                                      _polygons_to_masks(samples[0].polygons, (20, 10))[0])
