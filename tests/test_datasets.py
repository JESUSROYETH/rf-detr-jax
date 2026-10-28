"""COCO-format reading, augmentation geometry and batching."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rfdetr_jax.datasets.coco import (
    CocoDetection,
    Sample,
    _crop_sample,
    _hflip_sample,
    _resize_sample,
    collate,
    iterate_batches,
    normalize,
)


@pytest.fixture
def dataset(tmp_path: Path) -> CocoDetection:
    """Write a two-image COCO split to disk and open it."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index, name in enumerate(("a.jpg", "b.jpg")):
        Image.fromarray(np.full((40, 60, 3), index * 40, dtype=np.uint8)).save(image_dir / name)
    annotations = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 60, "height": 40},
            {"id": 2, "file_name": "b.jpg", "width": 60, "height": 40},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 5, 20, 10], "area": 200, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 5, "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 3, "bbox": [30, 20, 10, 10], "area": 100, "iscrowd": 1},
        ],
        "categories": [{"id": 3, "name": "x"}, {"id": 5, "name": "y"}],
    }
    path = tmp_path / "_annotations.coco.json"
    path.write_text(json.dumps(annotations))
    return CocoDetection(image_dir, path)


def test_loads_boxes_as_xyxy(dataset: CocoDetection) -> None:
    """Annotations are decoded as absolute xyxy with their category ids preserved."""
    sample = dataset.load(0)
    assert sample.image.shape == (3, 40, 60)
    assert sample.orig_size == (40, 60)
    assert np.allclose(sample.boxes[0], [10, 5, 30, 15])
    assert sample.labels.tolist() == [3, 5]


def test_crowd_annotations_are_dropped(dataset: CocoDetection) -> None:
    """``iscrowd`` annotations are excluded, as the upstream reader does."""
    assert len(dataset.load(1).labels) == 0


def test_num_classes_uses_max_category_id(dataset: CocoDetection) -> None:
    """The head is sized by ``max(category_id) + 1``, not by the category count."""
    assert dataset.num_classes == 6


def test_resize_scales_boxes_with_image() -> None:
    """Resizing rescales boxes by the same factor as the image."""
    sample = Sample(np.zeros((3, 40, 60), "float32"), np.array([[10, 5, 30, 15]], "float32"),
                    np.array([1]), 1, (40, 60))
    resized = _resize_sample(sample, (80, 120))
    assert resized.image.shape == (3, 80, 120)
    assert np.allclose(resized.boxes[0], [20, 10, 60, 30])


def test_crop_drops_boxes_outside_the_window() -> None:
    """A box entirely outside the crop is removed along with its label."""
    sample = Sample(np.zeros((3, 40, 60), "float32"),
                    np.array([[10, 5, 30, 15], [50, 30, 58, 38]], "float32"),
                    np.array([1, 2]), 1, (40, 60))
    cropped = _crop_sample(sample, top=0, left=0, height=20, width=40)
    assert len(cropped.labels) == 1 and cropped.labels[0] == 1


def test_hflip_mirrors_boxes() -> None:
    """Horizontal flip mirrors box x-coordinates about the image centre."""
    sample = Sample(np.zeros((3, 40, 60), "float32"), np.array([[10, 5, 30, 15]], "float32"),
                    np.array([1]), 1, (40, 60))
    flipped = _hflip_sample(sample)
    assert np.allclose(flipped.boxes[0], [30, 5, 50, 15])


def test_normalize_produces_cxcywh_in_unit_range() -> None:
    """Normalisation converts to normalized cxcywh and drops zero-area boxes."""
    sample = Sample(np.ones((3, 40, 60), "float32"),
                    np.array([[0, 0, 60, 40], [10, 10, 10, 20]], "float32"),
                    np.array([1, 2]), 1, (40, 60))
    _, boxes, labels, keypoints = normalize(sample)
    assert len(labels) == 1
    assert np.allclose(boxes[0], [0.5, 0.5, 1.0, 1.0])
    assert keypoints is None


def test_collate_pads_and_masks() -> None:
    """Collation pads to a fixed target count and marks the real entries."""
    batch = collate(
        [
            (np.zeros((3, 8, 8), "float32"), np.zeros((2, 4), "float32"), np.array([1, 2]), None),
            (np.zeros((3, 8, 8), "float32"), np.zeros((0, 4), "float32"), np.array([], dtype=int), None),
        ],
        max_targets=4,
    )
    assert batch["labels"].shape == (2, 4)
    assert batch["target_mask"].sum(1).tolist() == [2, 0]
    assert "keypoints" not in batch


def test_collate_pads_keypoints_when_present() -> None:
    """A keypoint dataset yields a padded ``[B, T, K, 3]`` keypoint tensor."""
    keypoints = np.ones((2, 5, 3), dtype="float32")
    batch = collate(
        [
            (np.zeros((3, 8, 8), "float32"), np.zeros((2, 4), "float32"), np.array([1, 2]), keypoints),
            (np.zeros((3, 8, 8), "float32"), np.zeros((1, 4), "float32"), np.array([3]), keypoints[:1]),
        ],
        max_targets=4,
    )
    assert batch["keypoints"].shape == (2, 4, 5, 3)
    assert batch["keypoints"][0, :2].all() and not batch["keypoints"][0, 2:].any()
    assert batch["keypoints"][1, :1].all() and not batch["keypoints"][1, 1:].any()


def test_iterate_batches_is_deterministic(dataset: CocoDetection) -> None:
    """The same seed reproduces the same augmented batch, which the comparison relies on."""
    first = next(iterate_batches(dataset, 32, 2, 4, train=True, seed=3))
    second = next(iterate_batches(dataset, 32, 2, 4, train=True, seed=3))
    assert np.array_equal(first["pixel_values"], second["pixel_values"])
    assert np.array_equal(first["boxes"], second["boxes"])


@pytest.fixture
def keypoint_dataset(tmp_path: Path) -> CocoDetection:
    """Write a one-image COCO keypoint split with a 3-point skeleton."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(image_dir / "a.jpg")
    annotations = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 60, "height": 40}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 5, 20, 10], "area": 200,
                "iscrowd": 0,
                # Third point is flagged invisible and must be zeroed on load.
                "keypoints": [12, 6, 2, 20, 10, 1, 25, 12, 0],
            }
        ],
        "categories": [{"id": 1, "name": "thing", "keypoints": ["l", "r", "c"]}],
    }
    path = tmp_path / "kp.json"
    path.write_text(json.dumps(annotations))
    return CocoDetection(image_dir, path)


def test_keypoint_count_is_inferred(keypoint_dataset: CocoDetection) -> None:
    """The keypoint count comes from the category metadata."""
    assert keypoint_dataset.num_keypoints == 3


def test_invisible_keypoints_load_as_zero(keypoint_dataset: CocoDetection) -> None:
    """A visibility-0 keypoint is zeroed on load, matching the upstream convention."""
    sample = keypoint_dataset.load(0)
    assert sample.keypoints.shape == (1, 3, 3)
    assert np.allclose(sample.keypoints[0, 0], [12, 6, 2])
    assert np.allclose(sample.keypoints[0, 2], [0, 0, 0])


def test_resize_scales_keypoints(keypoint_dataset: CocoDetection) -> None:
    """Keypoints scale with the image and keep their visibility channel."""
    resized = _resize_sample(keypoint_dataset.load(0), (80, 120))
    assert np.allclose(resized.keypoints[0, 0], [24, 12, 2])


def test_hflip_mirrors_and_swaps_keypoints(keypoint_dataset: CocoDetection) -> None:
    """A horizontal flip mirrors x and swaps the left/right pair."""
    sample = keypoint_dataset.load(0)
    flipped = _hflip_sample(sample, keypoint_flip_pairs=[0, 1])
    # Point 0 mirrors to 60 - 1 - 12 = 47 and lands in slot 1 after the swap.
    assert np.allclose(flipped.keypoints[0, 1], [47, 6, 2])
    assert np.allclose(flipped.keypoints[0, 0], [60 - 1 - 20, 10, 1])


def test_hflip_without_pairs_keeps_joint_order(keypoint_dataset: CocoDetection) -> None:
    """Without flip pairs the joints mirror but stay in their original slots."""
    flipped = _hflip_sample(keypoint_dataset.load(0))
    assert np.allclose(flipped.keypoints[0, 0], [47, 6, 2])


def test_crop_offsets_and_drops_keypoints(keypoint_dataset: CocoDetection) -> None:
    """Cropping shifts keypoints and zeroes any that leave the window."""
    cropped = _crop_sample(keypoint_dataset.load(0), top=4, left=10, height=20, width=20)
    assert np.allclose(cropped.keypoints[0, 0], [2, 2, 2])
    # The second point at x=20 lands at x=10 inside the 20-wide crop and survives.
    assert np.allclose(cropped.keypoints[0, 1], [10, 6, 1])


def test_normalized_keypoints_are_unit_scaled(keypoint_dataset: CocoDetection) -> None:
    """Normalisation divides keypoint x/y by the image size and leaves visibility alone."""
    _, _, _, keypoints = normalize(keypoint_dataset.load(0))
    assert np.allclose(keypoints[0, 0], [12 / 60, 6 / 40, 2])


def test_keypoint_batches_carry_keypoints(keypoint_dataset: CocoDetection) -> None:
    """A keypoint dataset produces batches with a padded keypoint tensor."""
    batch = next(
        iterate_batches(
            keypoint_dataset, 32, 1, 4, train=True, seed=0, drop_last=False, keypoint_flip_pairs=[0, 1]
        )
    )
    assert batch["keypoints"].shape == (1, 4, 3, 3)


def test_parallel_loading_yields_identical_batches(dataset: CocoDetection) -> None:
    """Decoding and augmenting on a thread pool must not change a value in the batch stream."""
    sequential = list(iterate_batches(dataset, 32, 1, 4, train=True, seed=5, epochs=3))
    parallel = list(iterate_batches(dataset, 32, 1, 4, train=True, seed=5, epochs=3, workers=4))

    assert len(sequential) == len(parallel)
    for left, right in zip(sequential, parallel, strict=True):
        assert left.keys() == right.keys()
        for name in left:
            np.testing.assert_array_equal(left[name], right[name])


@pytest.mark.parametrize("train", [True, False])
def test_process_workers_yield_identical_batches(dataset: CocoDetection, train: bool) -> None:
    """Worker processes writing into shared memory must reproduce the sequential stream exactly."""
    options = {"train": train, "seed": 5, "epochs": 2, "drop_last": not train, "pad_last": not train}
    sequential = list(iterate_batches(dataset, 32, 2, 4, **options))
    parallel = list(iterate_batches(dataset, 32, 2, 4, workers=2, worker_mode="process", depth=2, **options))

    assert len(sequential) == len(parallel)
    for left, right in zip(sequential, parallel, strict=True):
        assert left.keys() == right.keys()
        for name in left:
            np.testing.assert_array_equal(left[name], right[name])


def test_pixel_dtype_is_applied_per_sample(dataset: CocoDetection) -> None:
    """A narrower pixel dtype equals casting the float32 batch afterwards."""
    wide = next(iterate_batches(dataset, 32, 2, 4, train=True, seed=1))
    narrow = next(iterate_batches(dataset, 32, 2, 4, train=True, seed=1, workers=2, pixel_dtype=np.float16))
    assert narrow["pixel_values"].dtype == np.float16
    np.testing.assert_array_equal(narrow["pixel_values"], wide["pixel_values"].astype(np.float16))


def test_target_buckets_pad_to_the_smallest_fitting_size(dataset: CocoDetection) -> None:
    """Bucketed batches carry the same targets, padded only as far as the smallest fitting bucket."""
    full = list(iterate_batches(dataset, 32, 2, 8, train=True, seed=4, epochs=2))
    bucketed = list(iterate_batches(dataset, 32, 2, 8, train=True, seed=4, epochs=2, target_buckets=[1, 2, 4]))
    for left, right in zip(full, bucketed, strict=True):
        needed = int(left["target_mask"].sum(1).max())
        expected = min([size for size in (1, 2, 4) if size >= needed] + [8])
        assert right["labels"].shape[1] == expected
        np.testing.assert_array_equal(left["pixel_values"], right["pixel_values"])
        for name in ("labels", "boxes", "target_mask"):
            np.testing.assert_array_equal(left[name][:, :expected], right[name])
            assert not left["target_mask"][:, expected:].any()


def test_collate_rejects_target_overflow() -> None:
    """Crowded images must not silently lose annotations at the padding boundary."""
    sample = (np.zeros((3, 8, 8)), np.zeros((3, 4)), np.array([1, 2, 3]), None)
    with pytest.raises(ValueError, match="3 targets.*max_targets=2"):
        collate([sample], max_targets=2)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("category", "unknown category"),
        ("image", "unknown image"),
        ("nan_box", "Invalid COCO bbox"),
        ("duplicate_categories", "category ids must be unique"),
        ("duplicate_images", "image ids must be unique"),
        ("empty_categories", "cannot be empty"),
    ],
)
def test_invalid_annotation_index_is_rejected(tmp_path: Path, change: str, message: str) -> None:
    """Malformed references and geometry fail before they can corrupt matching and metrics."""
    raw = {
        "images": [{"id": 1, "file_name": "image.jpg"}],
        "categories": [{"id": 1, "name": "thing"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100}],
    }
    if change == "category":
        raw["annotations"][0]["category_id"] = 2
    elif change == "image":
        raw["annotations"][0]["image_id"] = 2
    elif change == "nan_box":
        raw["annotations"][0]["bbox"][0] = float("nan")
    elif change == "duplicate_categories":
        raw["categories"] *= 2
    elif change == "duplicate_images":
        raw["images"] *= 2
    elif change == "empty_categories":
        raw["categories"] = []
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=message):
        CocoDetection(tmp_path, path)


def test_evaluation_preserves_targets_above_training_capacity(dataset: CocoDetection) -> None:
    """Inference can score crowded images without dropping targets or changing image shapes."""
    batch = next(iterate_batches(dataset, 32, 2, 1, train=False))
    assert batch["pixel_values"].shape == (2, 3, 32, 32)
    assert batch["target_mask"].sum(axis=1).tolist() == [2, 0]


@pytest.mark.parametrize("encoding", ["polygon", "rle", "uncompressed"])
def test_segmentation_decoding_and_geometry(tmp_path: Path, encoding: str) -> None:
    """Polygon and both COCO RLE forms survive the same resize/crop/flip as boxes."""
    from pycocotools import mask as mask_api

    Image.new("RGB", (32, 32)).save(tmp_path / "image.png")
    mask = np.zeros((32, 32), np.uint8)
    mask[8:20, 8:20] = 1
    if encoding == "polygon":
        segmentation = [[8, 8, 20, 8, 20, 20, 8, 20]]
    elif encoding == "rle":
        segmentation = mask_api.encode(np.asfortranarray(mask))
        segmentation["counts"] = segmentation["counts"].decode("ascii")
    else:
        counts, previous, length = [], 0, 0
        for value in mask.ravel(order="F"):
            if value == previous:
                length += 1
            else:
                counts.append(length)
                previous, length = value, 1
        counts.append(length)
        segmentation = {"size": [32, 32], "counts": counts}
    raw = {
        "images": [{"id": 1, "file_name": "image.png", "height": 32, "width": 32},
                   {"id": 2, "file_name": "image.png", "height": 32, "width": 32}],
        "categories": [{"id": 1, "name": "object"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [8, 8, 12, 12], "area": 144,
                         "segmentation": segmentation}],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(raw))
    dataset = CocoDetection(tmp_path, path, segmentation=True)
    sample = dataset.load(0)
    np.testing.assert_array_equal(sample.masks[0], mask)
    resized = _resize_sample(sample, (64, 64))
    np.testing.assert_array_equal(resized.masks[0], np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1))
    cropped = _crop_sample(sample, 6, 6, 20, 20)
    flipped = _hflip_sample(cropped)
    np.testing.assert_array_equal(flipped.masks[0], mask[6:26, 6:26][:, ::-1])
    batch = next(iterate_batches(dataset, 32, 3, 2, train=False, drop_last=False, pad_last=True))
    assert batch["masks"].shape == (3, 2, 32, 32)
    assert batch["num_valid"] == 2
    assert not batch["masks"][1:].any()
    assert not batch["masks"][:, 1].any()
    removed = _crop_sample(sample, 0, 0, 4, 4)
    assert removed.masks.shape == (0, 4, 4)
    assert normalize(removed, include_masks=True)[4].shape == (0, 4, 4)


def test_segmentation_requires_annotations(dataset: CocoDetection, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing or empty segmentation"):
        CocoDetection(dataset.image_dir, tmp_path / "_annotations.coco.json", segmentation=True)


def test_explicit_keypoint_label_mapping_preserves_coco_metadata(keypoint_dataset: CocoDetection) -> None:
    """Official person schema [17] can train label zero against raw COCO category one."""
    keypoint_dataset.set_category_mapping({1: 0})
    sample = keypoint_dataset.load(0)
    assert sample.labels.tolist() == [0]
    assert keypoint_dataset.category_ids == [1]
    assert keypoint_dataset.num_classes == 1
    assert sample.keypoints.shape == (1, 3, 3)


@pytest.mark.parametrize("mapping", [{3: 0}, {3: 0, 5: 0}, {3: -1, 5: 1}])
def test_category_mapping_rejects_missing_or_colliding_labels(dataset: CocoDetection, mapping: dict) -> None:
    with pytest.raises(ValueError, match="category_to_label"):
        dataset.set_category_mapping(mapping)


def test_disabling_keypoint_flip_preserves_joint_semantics(keypoint_dataset: CocoDetection, monkeypatch) -> None:
    from rfdetr_jax.datasets import coco

    monkeypatch.setattr(coco, "draw_train_augmentation",
                        lambda shape, rng, scale_jitter=True: coco.TrainAugmentation(None, None, True))
    no_flip = next(iterate_batches(keypoint_dataset, 32, 1, 4, train=True, allow_horizontal_flip=False))
    reference = next(iterate_batches(keypoint_dataset, 32, 1, 4, train=False))
    np.testing.assert_array_equal(no_flip["keypoints"], reference["keypoints"])
    np.testing.assert_array_equal(no_flip["boxes"], reference["boxes"])
    automatic = next(iterate_batches(keypoint_dataset, 32, 1, 4, train=True))
    np.testing.assert_array_equal(automatic["keypoints"], no_flip["keypoints"])
    flipped = next(iterate_batches(keypoint_dataset, 32, 1, 4, train=True, keypoint_flip_pairs=(0, 1)))
    assert not np.array_equal(flipped["keypoints"], no_flip["keypoints"])


def test_keypoint_capacity_includes_categories_without_metadata(tmp_path: Path) -> None:
    """One category's metadata cannot hide a larger annotation-only skeleton."""
    raw = {
        "images": [{"id": 1, "file_name": "unused.png"}],
        "categories": [{"id": 1, "name": "small", "keypoints": ["a"]}, {"id": 2, "name": "large"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "keypoints": [1, 1, 2]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "area": 100,
             "keypoints": [1, 1, 2, 2, 2, 2, 3, 3, 2]},
        ],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(raw))
    dataset = CocoDetection(tmp_path, path)
    assert dataset.num_keypoints == 3
    record = dataset.record(0)
    assert record.keypoints.shape == (2, 3, 3)
    np.testing.assert_array_equal(record.keypoints[1], [[1, 1, 2], [2, 2, 2], [3, 3, 2]])
    assert not record.keypoints[0, 1:].any()


def test_automatic_detection_augmentation_still_flips(dataset: CocoDetection) -> None:
    from rfdetr_jax.datasets.coco import TrainAugmentation, apply_train_augmentation

    sample = dataset.load(0)
    transformed = apply_train_augmentation(sample, 32, TrainAugmentation(None, None, True))
    expected = _hflip_sample(_resize_sample(sample, (32, 32)))
    np.testing.assert_array_equal(transformed.boxes, expected.boxes)


def test_crop_branch_draws_one_square_side_like_upstream() -> None:
    """Upstream samples a single crop size and applies it to both axes."""
    from rfdetr_jax.datasets.coco import draw_train_augmentation

    rng = np.random.default_rng(11)
    crops = [draw_train_augmentation((400, 600), rng).crop for _ in range(300)]
    crops = [crop for crop in crops if crop is not None]
    assert crops
    for top, left, crop_h, crop_w in crops:
        assert crop_h == crop_w
        assert 384 <= crop_h <= 600 and top >= 0 and left >= 0


def test_short_side_resize_rounds_like_upstream() -> None:
    from rfdetr_jax.datasets.coco import short_side_resize_shape

    assert short_side_resize_shape(120, 243, 500) == (500, 1012)
    assert short_side_resize_shape(243, 120, 500) == (1012, 500)
    assert short_side_resize_shape(40, 60, 400) == (400, 600)


def test_image_ids_above_int32_are_preserved(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_dir / "a.jpg")
    big = 2**40 + 3
    path = tmp_path / "ann.json"
    path.write_text(json.dumps({
        "images": [{"id": big, "file_name": "a.jpg", "width": 16, "height": 16}],
        "annotations": [{"id": 1, "image_id": big, "category_id": 1, "bbox": [1, 1, 4, 4], "area": 16, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "x"}],
    }))
    batch = next(iterate_batches(CocoDetection(image_dir, path), batch_size=1, resolution=16, max_targets=2,
                                 train=False, seed=0))
    assert batch["image_ids"].dtype == np.int64
    assert int(batch["image_ids"][0]) == big


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda row: row.pop("area"), "area"),
    (lambda row: row.update(area=-1), "area"),
    (lambda row: row.update(segmentation=[]), "segmentation"),
    (lambda row: row.update(segmentation=[[]]), "segmentation"),
    (lambda row: row.update(segmentation={"counts": [], "size": [16, 16]}), "segmentation"),
    (lambda row: row.update(segmentation={"counts": "", "size": [16, 16]}), "segmentation"),
])
def test_incomplete_annotations_are_rejected_at_load(tmp_path: Path, mutate, message: str) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_dir / "a.jpg")
    row = {"id": 7, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4], "area": 16, "iscrowd": 0,
           "segmentation": [[1, 1, 5, 1, 5, 5]]}
    mutate(row)
    path = tmp_path / "ann.json"
    path.write_text(json.dumps({"images": [{"id": 1, "file_name": "a.jpg", "width": 16, "height": 16}],
                                "annotations": [row], "categories": [{"id": 1, "name": "x"}]}))
    with pytest.raises(ValueError, match=message) as info:
        CocoDetection(image_dir, path, segmentation=True)
    assert "7" in str(info.value)


def test_detection_task_ignores_keypoint_metadata(tmp_path: Path) -> None:
    """Keypoint schema consistency is only enforced when keypoints are requested."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_dir / "a.jpg")
    path = tmp_path / "ann.json"
    path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.jpg", "width": 16, "height": 16}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4], "area": 16, "iscrowd": 0,
                         "keypoints": [2, 2, 2, 3, 3, 2, 4, 4, 2], "num_keypoints": 3}],
        "categories": [{"id": 1, "name": "x", "keypoints": ["a", "b"]}],
    }))
    assert CocoDetection(image_dir, path, keypoints=False).num_keypoints == 0
    with pytest.raises(ValueError, match="inconsistent keypoint count"):
        CocoDetection(image_dir, path, keypoints=True)


def test_raw_target_capacity_reports_the_largest_image(dataset: CocoDetection) -> None:
    assert dataset.max_targets_per_image() == 2
