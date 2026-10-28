# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# Source ancestry: LW-DETR / Conditional DETR / DETR / Deformable DETR.
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
"""COCO-format dataset and transforms, ported from ``rfdetr.datasets.coco``/``transforms``.

The default RF-DETR training pipeline (``square_resize_div_64=True``, ``aug_config=None``) is:

1. one of, with equal probability — a direct square resize, or a short-side resize to 400/500/600
   followed by a random crop and a square resize;
2. a horizontal flip with probability 0.5;
3. scale to ``[0, 1]`` and normalise with ImageNet statistics.

Resizing uses PyTorch's antialiased bilinear filter, reproduced by
:func:`rfdetr_jax.utilities.interpolate.resize_2d`. The torch pipeline resizes uint8 PIL images
and converts to float afterwards, so intermediate images here can differ from it by up to one
uint8 quantization step; the geometry of boxes, masks and keypoints matches exactly.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rfdetr_jax.utilities.resample import resize_2d_numpy as resize_2d

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: Short-side sizes the crop branch resizes to before cropping, per ``_build_train_resize_config``.
_CROP_SOURCE_SIZES = (400, 500, 600)
#: Crop height range for the crop branch.
_CROP_MIN_MAX_HEIGHT = (384, 600)


@dataclass
class Sample:
    """One decoded image with boxes, labels and optional keypoints and instance masks.

    Attributes:
        image: Image as ``[3, H, W]`` float32 in ``[0, 1]`` before normalisation.
        boxes: Boxes as ``[N, 4]`` in absolute ``xyxy`` until :func:`normalize` runs.
        labels: Class indices ``[N]``.
        image_id: The COCO image id, kept for evaluation.
        orig_size: The source ``(height, width)``, kept for evaluation.
        keypoints: Optional ``[N, K, 3]`` keypoints as absolute ``(x, y, visibility)``.
        masks: Optional binary ``[N, H, W]`` instance masks in the current image geometry.
    """

    image: np.ndarray
    boxes: np.ndarray
    labels: np.ndarray
    image_id: int
    orig_size: tuple[int, int]
    keypoints: np.ndarray | None = None
    masks: np.ndarray | None = None


def mark_invisible_keypoints(keypoints: np.ndarray, height: int, width: int) -> np.ndarray:
    """Zero every keypoint flagged invisible or moved outside the image.

    RF-DETR treats "outside the frame" and "not annotated" identically: both become an all-zero
    triplet, which the losses then exclude through the visibility channel.
    """
    if keypoints.size == 0:
        return keypoints
    visible = keypoints[..., 2] > 0
    inside = (
        (keypoints[..., 0] >= 0)
        & (keypoints[..., 0] < width)
        & (keypoints[..., 1] >= 0)
        & (keypoints[..., 1] < height)
    )
    return np.where((visible & inside)[..., None], keypoints, 0.0).astype(np.float32)


def _segmentation_is_empty(segmentation) -> bool:
    """Return whether a COCO ``segmentation`` field carries no mask: absent, no polygon with three
    points, or an RLE without counts."""
    if not segmentation:
        return True
    if isinstance(segmentation, list):
        return all(not isinstance(polygon, list) or len(polygon) < 6 for polygon in segmentation)
    if isinstance(segmentation, dict):
        return not segmentation.get("counts")
    return True


class CocoDetection:
    """A COCO-format detection split backed by an ``_annotations.coco.json`` or ``instances_*.json``."""

    def __init__(
        self, image_dir: str | Path, annotation_file: str | Path, *, segmentation: bool = False,
        category_to_label: dict[int, int] | None = None, keypoints: bool | None = None,
    ) -> None:
        """Load the annotation JSON and index it by image.

        Args:
            image_dir: Directory holding the split's images.
            annotation_file: The COCO JSON for this split.
            segmentation: Decode polygon/RLE masks and carry them through all transforms.
            category_to_label: Optional explicit raw category id to model label mapping.
            keypoints: ``True`` requires a consistent keypoint schema, ``False`` ignores keypoint
                metadata (detection and segmentation tasks), ``None`` infers it when present.
        """
        self.image_dir = Path(image_dir)
        self.segmentation = segmentation
        raw = json.loads(Path(annotation_file).read_text())
        self.images = raw["images"]
        self.categories = sorted(raw["categories"], key=lambda category: category["id"])
        self.category_ids = [category["id"] for category in self.categories]
        self.category_names = [category["name"] for category in self.categories]

        if not self.category_ids or any(type(value) is not int or value < 0 for value in self.category_ids):
            raise ValueError("COCO categories must have nonnegative integer ids and cannot be empty")
        if len(set(self.category_ids)) != len(self.category_ids):
            raise ValueError("COCO category ids must be unique")
        image_ids = [image["id"] for image in self.images]
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("COCO image ids must be unique")

        by_image: dict[int, list[dict[str, Any]]] = {image["id"]: [] for image in self.images}
        for annotation in raw["annotations"]:
            if annotation["image_id"] not in by_image:
                raise ValueError(f"Annotation references unknown image id {annotation['image_id']}")
            if annotation["category_id"] not in self.category_ids:
                raise ValueError(f"Annotation references unknown category id {annotation['category_id']}")
            identifier = annotation.get("id", "<unknown>")
            box = np.asarray(annotation["bbox"], dtype=np.float64)
            if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] < 0).any():
                raise ValueError(f"Invalid COCO bbox in annotation {identifier}")
            # COCO evaluation reads ``area`` for every annotation; fail now rather than after an epoch.
            area = annotation.get("area")
            if isinstance(area, bool) or not isinstance(area, int | float) or not np.isfinite(area) or area < 0:
                raise ValueError(f"Missing or invalid area in annotation {identifier}")
            if annotation.get("iscrowd", 0):
                continue
            if segmentation and _segmentation_is_empty(annotation.get("segmentation")):
                raise ValueError(f"Missing or empty segmentation in annotation {identifier}")
            by_image.setdefault(annotation["image_id"], []).append(annotation)
        self.annotations = by_image
        self.num_keypoints = 0 if keypoints is False else self._infer_keypoint_count(raw)
        self.set_category_mapping(category_to_label)

    def set_category_mapping(self, category_to_label: dict[int, int] | None) -> None:
        """Set an explicit one-to-one raw COCO category → model label mapping.

        Category metadata keeps its original ids for evaluation. Without an explicit mapping,
        model labels are raw category ids, preserving the detection reader's default behavior.
        """
        mapping = dict(category_to_label) if category_to_label is not None else {
            category: category for category in self.category_ids
        }
        if set(mapping) != set(self.category_ids):
            raise ValueError("category_to_label must cover exactly the dataset's category ids")
        if any(type(label) is not int or label < 0 for label in mapping.values()):
            raise ValueError("category_to_label values must be nonnegative integers")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("category_to_label must map categories to distinct labels")
        self.category_to_label = mapping

    @staticmethod
    def _infer_keypoint_count(raw: dict[str, Any]) -> int:
        """Return the dataset's keypoint count from category metadata or annotations; 0 if none."""
        counts: dict[int, set[int]] = {category["id"]: set() for category in raw["categories"]}
        for category in raw["categories"]:
            if category.get("keypoints"):
                counts[category["id"]].add(len(category["keypoints"]))
        for annotation in raw["annotations"]:
            points = annotation.get("keypoints")
            if points:
                if not isinstance(points, list) or len(points) % 3:
                    raise ValueError("keypoint annotations must contain x/y/visibility triplets")
                counts[annotation["category_id"]].add(len(points) // 3)
        for category, observed in counts.items():
            if len(observed) > 1:
                raise ValueError(f"inconsistent keypoint count for category {category}")
        return max((count for observed in counts.values() for count in observed), default=0)

    def max_targets_per_image(self) -> int:
        """Return the largest non-crowd annotation count of any image: the capacity training needs."""
        return max((len(rows) for rows in self.annotations.values()), default=0)

    @property
    def num_classes(self) -> int:
        """Return the required model label capacity (raw category ids unless explicitly mapped)."""
        return max(self.category_to_label.values()) + 1

    def __len__(self) -> int:
        """Number of images in the split."""
        return len(self.images)

    def record(self, index: int) -> SampleRecord:
        """Return image ``index``'s path and raw targets without decoding the image.

        The record is small and picklable, so a worker process can turn it into a :class:`Sample`
        without holding the annotation index.
        """
        record = self.images[index]
        annotations = self.annotations.get(record["id"], [])
        boxes = np.zeros((len(annotations), 4), dtype=np.float32)
        labels = np.zeros((len(annotations),), dtype=np.int32)
        for slot, annotation in enumerate(annotations):
            x, y, w, h = annotation["bbox"]
            boxes[slot] = [x, y, x + w, y + h]
            labels[slot] = self.category_to_label[annotation["category_id"]]

        keypoints = None
        if self.num_keypoints:
            keypoints = np.zeros((len(annotations), self.num_keypoints, 3), dtype=np.float32)
            for slot, annotation in enumerate(annotations):
                raw = annotation.get("keypoints")
                if raw:
                    points = np.asarray(raw, dtype=np.float32).reshape(-1, 3)
                    if points.shape[0] > self.num_keypoints:
                        raise ValueError("keypoint padding capacity is smaller than an annotation's joint count")
                    keypoints[slot, : points.shape[0]] = points
        segmentations = [annotation["segmentation"] for annotation in annotations] if self.segmentation else None
        return SampleRecord(
            str(self.image_dir / record["file_name"]), boxes, labels, keypoints, record["id"], segmentations
        )

    def load(self, index: int) -> Sample:
        """Decode image ``index`` and return it with absolute ``xyxy`` boxes."""
        return decode_record(self.record(index))


@dataclass
class SampleRecord:
    """What decoding one image needs: its path and its raw, unclipped targets.

    Attributes:
        path: The image file.
        boxes: Absolute ``xyxy`` boxes ``[N, 4]``, not yet clipped to the image.
        labels: Class indices ``[N]``.
        keypoints: Optional raw ``[N, K, 3]`` keypoints.
        image_id: The COCO image id.
        segmentations: Optional raw COCO polygon/RLE annotations, decoded alongside the image.
    """

    path: str
    boxes: np.ndarray
    labels: np.ndarray
    keypoints: np.ndarray | None
    image_id: int
    segmentations: list | None = None


def decode_record(record: SampleRecord) -> Sample:
    """Decode a record's image and clip its targets to the decoded size."""
    with Image.open(record.path) as handle:
        image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    height, width = image.shape[:2]
    boxes = record.boxes.copy()
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, height)
    keypoints = record.keypoints
    if keypoints is not None:
        keypoints = mark_invisible_keypoints(keypoints, height, width)
    masks = None
    if record.segmentations is not None:
        from pycocotools import mask as mask_api

        masks = np.zeros((len(record.segmentations), height, width), dtype=np.float32)
        for index, segmentation in enumerate(record.segmentations):
            if not segmentation:
                continue
            if isinstance(segmentation, list) or (
                isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list)
            ):
                encoded = mask_api.frPyObjects(segmentation, height, width)
            else:
                encoded = segmentation
            decoded = mask_api.decode(encoded)
            if decoded.ndim == 3:
                decoded = np.any(decoded, axis=2)
            if decoded.shape != (height, width):
                raise ValueError(f"Segmentation size {decoded.shape} differs from image size {(height, width)}")
            masks[index] = decoded
    return Sample(
        image=image.transpose(2, 0, 1).astype(np.float32) / 255.0,
        boxes=boxes,
        labels=record.labels,
        image_id=record.image_id,
        orig_size=(height, width),
        keypoints=keypoints,
        masks=masks,
    )


def _resize_sample(sample: Sample, size: tuple[int, int]) -> Sample:
    """Resize an image and rescale its absolute xyxy boxes and keypoints."""
    height, width = sample.image.shape[1:]
    target_h, target_w = size
    resized = resize_2d(
        sample.image[None], (target_h, target_w), mode="bilinear", align_corners=False, antialias=True
    )
    ratio_w, ratio_h = target_w / width, target_h / height
    scale = np.array([ratio_w, ratio_h, ratio_w, ratio_h], dtype=np.float32)

    keypoints = sample.keypoints
    if keypoints is not None:
        keypoints = keypoints * np.array([ratio_w, ratio_h, 1.0], dtype=np.float32)
        keypoints = mark_invisible_keypoints(keypoints, target_h, target_w)

    masks = sample.masks
    if masks is not None:
        masks = (
            resize_2d(masks[None], (target_h, target_w), mode="nearest")[0]
            if len(masks) else np.zeros((0, target_h, target_w), dtype=np.float32)
        )

    return Sample(
        image=np.asarray(resized[0]),
        boxes=sample.boxes * scale,
        labels=sample.labels,
        image_id=sample.image_id,
        orig_size=sample.orig_size,
        keypoints=keypoints,
        masks=masks,
    )


def _crop_sample(sample: Sample, top: int, left: int, height: int, width: int) -> Sample:
    """Crop an image and drop boxes that fall entirely outside the crop."""
    cropped = sample.image[:, top : top + height, left : left + width]
    boxes = sample.boxes.copy()
    boxes[:, 0::2] -= left
    boxes[:, 1::2] -= top
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, height)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

    keypoints = sample.keypoints
    if keypoints is not None:
        keypoints = keypoints - np.array([left, top, 0.0], dtype=np.float32)
        keypoints = mark_invisible_keypoints(keypoints, height, width)[keep]

    return Sample(
        image=cropped,
        boxes=boxes[keep],
        labels=sample.labels[keep],
        image_id=sample.image_id,
        orig_size=sample.orig_size,
        keypoints=keypoints,
        masks=sample.masks[keep, top : top + height, left : left + width] if sample.masks is not None else None,
    )


def _hflip_sample(sample: Sample, keypoint_flip_pairs: Sequence[int] | None = None) -> Sample:
    """Mirror an image horizontally, along with its boxes and keypoints.

    Args:
        sample: The sample to mirror.
        keypoint_flip_pairs: Flat list of index pairs ``[a0, b0, a1, b1, ...]`` naming keypoints
            that are left/right mirrors of each other. Without them a flipped pose keeps its
            original joint labels, which is wrong for any left/right-aware skeleton.

    Returns:
        The mirrored sample.
    """
    height, width = sample.image.shape[1:]
    boxes = sample.boxes.copy()
    flipped = np.stack([width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], axis=-1)

    keypoints = sample.keypoints
    if keypoints is not None:
        keypoints = keypoints.copy()
        visible = keypoints[..., 2] > 0
        # Mirror about the last pixel index, as `_torchvision.RandomHorizontalFlip` does.
        keypoints[..., 0] = (width - 1) - keypoints[..., 0]
        keypoints[..., :2] = np.where(visible[..., None], keypoints[..., :2], 0.0)
        if keypoint_flip_pairs:
            permutation = np.arange(keypoints.shape[1])
            pairs = list(keypoint_flip_pairs)
            for index in range(0, len(pairs) - 1, 2):
                left, right = pairs[index], pairs[index + 1]
                if left < keypoints.shape[1] and right < keypoints.shape[1]:
                    permutation[left], permutation[right] = right, left
            keypoints = keypoints[:, permutation, :]
        keypoints = mark_invisible_keypoints(keypoints, height, width)

    return Sample(
        image=sample.image[:, :, ::-1].copy(),
        boxes=flipped,
        labels=sample.labels,
        image_id=sample.image_id,
        orig_size=sample.orig_size,
        keypoints=keypoints,
        masks=sample.masks[:, :, ::-1].copy() if sample.masks is not None else None,
    )


@dataclass(frozen=True)
class TrainAugmentation:
    """One sample's random augmentation choices, drawn before any pixels are touched.

    Splitting the draw from the application is what lets the expensive resizes run on several
    threads: the draws still come off the shared generator strictly in sample order, so the batch
    stream is bit-for-bit the sequential one, while the pixel work — which touches no RNG — can be
    reordered freely.

    Attributes:
        resize_to: Short-side resize target for the crop branch, or ``None`` for a direct resize.
        crop: ``(top, left, height, width)`` crop window, set exactly when ``resize_to`` is.
        flip: Whether to mirror the sample horizontally.
    """

    resize_to: tuple[int, int] | None
    crop: tuple[int, int, int, int] | None
    flip: bool


def draw_train_augmentation(
    shape: tuple[int, int], rng: np.random.Generator, scale_jitter: bool = True
) -> TrainAugmentation:
    """Draw one sample's augmentation, consuming ``rng`` exactly as :func:`train_transform` does.

    Args:
        shape: The sample's ``(height, width)`` before augmentation.
        rng: The shared generator.
        scale_jitter: Whether the resize-and-crop branch is available.

    Returns:
        The choices to hand to :func:`apply_train_augmentation`.
    """
    height, width = shape
    resize_to = None
    crop = None
    if scale_jitter and rng.random() < 0.5:
        short_side = int(rng.choice(_CROP_SOURCE_SIZES))
        resize_to = short_side_resize_shape(height, width, short_side)

        # Upstream draws one square crop size and clamps it to each axis.
        height, width = resize_to
        low, high = _CROP_MIN_MAX_HEIGHT
        max_crop = min(high, height, width)
        min_crop = min(low, max_crop)
        crop_size = int(rng.integers(min_crop, max_crop + 1)) if max_crop > min_crop else max_crop
        crop_h, crop_w = min(crop_size, height), min(crop_size, width)
        top = int(rng.integers(0, height - crop_h + 1)) if height > crop_h else 0
        left = int(rng.integers(0, width - crop_w + 1)) if width > crop_w else 0
        crop = (top, left, crop_h, crop_w)
    return TrainAugmentation(resize_to, crop, bool(rng.random() < 0.5))


def short_side_resize_shape(height: int, width: int, size: int) -> tuple[int, int]:
    """Return upstream's ``(height, width)`` after resizing the short side to ``size``."""
    if (width <= height and width == size) or (height <= width and height == size):
        return height, width
    if width < height:
        return max(1, int(round(size * height / width))), size
    return size, max(1, int(round(size * width / height)))


def apply_train_augmentation(
    sample: Sample,
    resolution: int,
    augmentation: TrainAugmentation,
    keypoint_flip_pairs: Sequence[int] | None = None,
) -> Sample:
    """Apply drawn augmentation choices to one sample, touching no RNG."""
    if augmentation.resize_to is not None and augmentation.crop is not None:
        sample = _resize_sample(sample, augmentation.resize_to)
        sample = _crop_sample(sample, *augmentation.crop)
    sample = _resize_sample(sample, (resolution, resolution))
    if augmentation.flip and (sample.keypoints is None or keypoint_flip_pairs):
        sample = _hflip_sample(sample, keypoint_flip_pairs)
    return sample


def train_transform(
    sample: Sample,
    resolution: int,
    rng: np.random.Generator,
    scale_jitter: bool = True,
    keypoint_flip_pairs: Sequence[int] | None = None,
) -> Sample:
    """Apply the default RF-DETR training augmentation to one sample."""
    augmentation = draw_train_augmentation(sample.image.shape[1:], rng, scale_jitter)
    return apply_train_augmentation(sample, resolution, augmentation, keypoint_flip_pairs)


def val_transform(sample: Sample, resolution: int) -> Sample:
    """Resize a sample to the square evaluation resolution."""
    return _resize_sample(sample, (resolution, resolution))


def normalize(sample: Sample, *, include_masks: bool = False) -> tuple:
    """Normalise pixels, convert boxes to normalized ``cxcywh`` and scale keypoints to ``[0, 1]``.

    Returns:
        An ``(image, boxes, labels, keypoints)`` tuple. ``image`` is ``[3, H, W]`` normalised
        float32, ``boxes`` is ``[N, 4]`` normalized ``cxcywh``, and ``keypoints`` is ``[N, K, 3]``
        with x/y in ``[0, 1]`` and the visibility channel untouched, or ``None``.
        With ``include_masks=True``, append the filtered instance masks (or ``None``).
    """
    image = (sample.image - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    height, width = sample.image.shape[1:]
    boxes = sample.boxes
    centers = np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2 / width, (boxes[:, 1] + boxes[:, 3]) / 2 / height], axis=-1
    )
    sizes = np.stack([(boxes[:, 2] - boxes[:, 0]) / width, (boxes[:, 3] - boxes[:, 1]) / height], axis=-1)
    valid = (sizes > 0).all(-1)

    keypoints = sample.keypoints
    if keypoints is not None:
        keypoints = keypoints / np.array([width, height, 1.0], dtype=np.float32)
        keypoints = keypoints[valid]
    normalized = (image, np.concatenate([centers, sizes], axis=-1)[valid], sample.labels[valid], keypoints)
    if include_masks:
        return (*normalized, sample.masks[valid] if sample.masks is not None else None)
    return normalized


def collate(
    samples: list[tuple], max_targets: int
) -> dict[str, np.ndarray]:
    """Stack samples into a padded batch.

    Args:
        samples: ``(image, boxes, labels, keypoints[, masks])`` tuples from :func:`normalize`.
        max_targets: Fixed padded target count, so shapes stay static under ``jit``.

    Returns:
        Dict with ``pixel_values`` ``[B, 3, H, W]``, ``labels`` ``[B, T]``, ``boxes`` ``[B, T, 4]``,
        ``target_mask`` ``[B, T]``, ``keypoints`` ``[B, T, K, 3]`` when present, and
        ``masks`` ``[B, T, H, W]`` when mask decoding is enabled.
    """
    if not samples or max_targets < 1:
        raise ValueError("collate requires nonempty samples and max_targets >= 1")
    batch = len(samples)
    # `iterate_batches` writes pixels straight into the batch buffer and passes ``None`` here.
    images = np.stack([sample[0] for sample in samples]) if samples[0][0] is not None else None
    labels = np.zeros((batch, max_targets), dtype=np.int32)
    boxes = np.zeros((batch, max_targets, 4), dtype=np.float32)
    mask = np.zeros((batch, max_targets), dtype=bool)

    keypoint_counts = {sample[3].shape[1] for sample in samples if sample[3] is not None}
    keypoints = (
        np.zeros((batch, max_targets, next(iter(keypoint_counts)), 3), dtype=np.float32)
        if len(keypoint_counts) == 1
        else None
    )
    if len(keypoint_counts) > 1:
        raise ValueError("All samples must use the same padded keypoint count")
    sample_masks = [sample[4] if len(sample) > 4 else None for sample in samples]
    mask_shapes = {value.shape[1:] for value in sample_masks if value is not None}
    if len(mask_shapes) > 1:
        raise ValueError("All instance masks must share the transformed image size")
    masks = np.zeros((batch, max_targets, *next(iter(mask_shapes))), dtype=np.float32) if mask_shapes else None

    for index, sample in enumerate(samples):
        _, sample_boxes, sample_labels, sample_keypoints = sample[:4]
        count = len(sample_labels)
        if count > max_targets:
            raise ValueError(
                f"Sample {index} has {count} targets, exceeding max_targets={max_targets}; "
                "increase max_targets to avoid discarding annotations"
            )
        labels[index, :count] = sample_labels[:count]
        boxes[index, :count] = sample_boxes[:count]
        mask[index, :count] = True
        if keypoints is not None and sample_keypoints is not None:
            keypoints[index, :count] = sample_keypoints[:count]
        if masks is not None and sample_masks[index] is not None:
            masks[index, :count] = sample_masks[index]

    batch_dict = {"pixel_values": images, "labels": labels, "boxes": boxes, "target_mask": mask}
    if keypoints is not None:
        batch_dict["keypoints"] = keypoints
    if masks is not None:
        batch_dict["masks"] = masks
    return batch_dict


def _declared_shape(dataset: CocoDetection, index: int) -> tuple[int, int] | None:
    """Return the ``(height, width)`` the annotation file declares for an image, if it does."""
    record = dataset.images[index]
    height, width = record.get("height"), record.get("width")
    return (int(height), int(width)) if height and width else None


def _die_with_parent() -> None:
    """Have the kernel stop this worker when the process that spawned it dies.

    A normal exit or an exception shuts the pool down, but a parent killed by a signal runs no
    cleanup, and its workers would otherwise live on as orphans holding CPU and shared memory.
    Linux only; elsewhere this does nothing.
    """
    import sys

    if not sys.platform.startswith("linux"):
        return
    import contextlib
    import ctypes
    import signal

    with contextlib.suppress(OSError):
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


#: Shared-memory batch buffers this worker process has attached to, by name.
_ATTACHED: dict[str, Any] = {}


def _pixel_buffer(destination: tuple) -> np.ndarray:
    """Resolve where a sample's pixels go: an in-process array, or a shared-memory batch buffer."""
    if destination[0] == "array":
        return destination[1]
    from multiprocessing import shared_memory

    _, name, shape, dtype = destination
    if name not in _ATTACHED:
        # Buffers are reused round-robin, so each worker attaches to each one once. The parent owns
        # and unlinks them; spawned workers share its resource tracker, so attaching only re-adds a
        # name the tracker already holds and must not be undone here.
        _ATTACHED[name] = shared_memory.SharedMemory(name=name)
    return np.ndarray(shape, dtype=dtype, buffer=_ATTACHED[name].buf)


def _prepare_sample(
    source: SampleRecord | Sample,
    declared: tuple[int, int] | None,
    resolution: int,
    augmentation: TrainAugmentation | None,
    keypoint_flip_pairs: Sequence[int] | None,
    destination: tuple,
    slot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, tuple[int, int], np.ndarray | None]:
    """Decode, augment and normalise one sample, writing its pixels into its batch row.

    All the per-pixel work of a sample happens here, so it runs wherever the caller schedules it —
    the consuming thread, a worker thread or a worker process — and nothing heavier than the
    targets has to travel back.

    Args:
        source: The sample's record, or the sample itself when it had to be decoded early.
        declared: The size the annotation file states, which the augmentation was drawn against.
        resolution: Square output size.
        augmentation: Pre-drawn training choices, or ``None`` for the evaluation resize.
        keypoint_flip_pairs: Left/right keypoint index pairs to swap on a horizontal flip.
        destination: ``("array", images)`` or ``("shm", name, shape, dtype)`` for the batch pixels.
        slot: This sample's row in the batch.

    Returns:
        ``(boxes, labels, keypoints, image_id, orig_size, masks)`` with boxes in normalized ``cxcywh``.

    Raises:
        ValueError: If the decoded image does not have the size the annotation file declares.
    """
    sample = source if isinstance(source, Sample) else decode_record(source)
    if augmentation is not None and declared is not None and declared != tuple(sample.image.shape[1:]):
        raise ValueError(
            f"{getattr(source, 'path', sample.image_id)}: decoded size {tuple(sample.image.shape[1:])} "
            f"differs from the annotation's {declared}"
        )
    if augmentation is not None:
        sample = apply_train_augmentation(sample, resolution, augmentation, keypoint_flip_pairs)
    else:
        sample = val_transform(sample, resolution)
    image, boxes, labels, keypoints, masks = normalize(sample, include_masks=True)
    _pixel_buffer(destination)[slot] = image
    return boxes, labels, keypoints, sample.image_id, sample.orig_size, masks


def _collate_targets(
    images: np.ndarray, prepared: list, max_targets: int, valid: int
) -> dict[str, np.ndarray]:
    """Assemble a batch dict from a filled pixel buffer and the per-sample targets."""
    batch = collate(
        [(None, boxes, labels, keypoints, masks) for boxes, labels, keypoints, _, _, masks in prepared], max_targets
    )
    batch["pixel_values"] = images
    batch["image_ids"] = np.asarray([item[3] for item in prepared], dtype=np.int64)
    batch["orig_sizes"] = np.asarray([item[4] for item in prepared], dtype=np.int32)
    batch["num_valid"] = np.int32(valid)
    return batch


def iterate_batches(
    dataset: CocoDetection,
    resolution: int,
    batch_size: int,
    max_targets: int,
    train: bool,
    seed: int = 0,
    epochs: int = 1,
    drop_last: bool = True,
    keypoint_flip_pairs: Sequence[int] | None = None,
    workers: int = 0,
    pad_last: bool = False,
    depth: int = 4,
    pixel_dtype: Any = np.float32,
    worker_mode: str = "thread",
    target_buckets: Sequence[int] | None = None,
    allow_horizontal_flip: bool = True,
    scale_jitter: bool = True,
):
    """Yield padded batches from a dataset split.

    Args:
        dataset: The split to read.
        resolution: Square resize target.
        batch_size: Images per batch.
        max_targets: Padded target count.
        train: Whether to shuffle and augment.
        seed: Base RNG seed; each epoch derives its own stream from it.
        epochs: Number of passes over the split.
        drop_last: Whether to drop a trailing partial batch.
        keypoint_flip_pairs: Left/right keypoint index pairs to swap on a horizontal flip.
        workers: Threads to decode, augment and normalise on. ``0`` does it all on the consuming
            thread. The batches produced are identical either way: augmentation choices are always
            drawn from the shared generator in sample order, and only the pixel work is spread out.
        pad_last: Whether to repeat the final sample so a short trailing batch still has
            ``batch_size`` rows. Every batch then has one shape, which matters on an accelerator,
            where a second shape means a second compilation of the whole model.
        depth: Batches kept in flight on the workers ahead of the one being yielded.
        target_buckets: Padded target counts to choose from per batch. A batch is padded to the
            smallest bucket holding its largest image's targets (never above ``max_targets``)
            instead of always to ``max_targets``. Padding is masked out of the matcher and the loss,
            so padded entries do not contribute to either. Each bucket used compiles its own
            executable once.
        worker_mode: ``"thread"`` or ``"process"``. Threads share one interpreter, so the Python
            between NumPy calls is subject to the GIL. Processes use separate interpreters and
            write each sample straight into a shared-memory batch buffer.
        pixel_dtype: Dtype of ``pixel_values``. Each worker casts its own sample while writing it
            into the batch, so a narrower dtype (a model that computes in bfloat16 casts its input
            first anyway) halves the bytes sent to the device at no cost to the consuming thread.
        allow_horizontal_flip: Disable mirroring for keypoint schemas without known left/right
            correspondences. Random draws still occur, preserving all other augmentation choices.
        scale_jitter: Enable the upstream short-side resize/crop branch during training.

    Yields:
        Batch dicts from :func:`collate`, each carrying its ``image_ids``, ``orig_sizes`` and a
        ``num_valid`` count. ``num_valid`` is below ``batch_size`` only on a padded trailing batch,
        and the rows past it are duplicates the caller must drop.
    """
    import collections
    import concurrent.futures
    import contextlib

    if batch_size < 1 or resolution < 1 or max_targets < 1:
        raise ValueError("batch_size, resolution and max_targets must be positive")
    if epochs < 0 or workers < 0 or depth < 0:
        raise ValueError("epochs, workers and depth must be nonnegative")

    # The draws need each image's size. The annotation file normally states it, which lets a
    # sample be decoded, augmented and normalised in one task on a worker; only when a size is
    # missing does the image have to be decoded before its choices can be drawn.
    declared = [_declared_shape(dataset, index) for index in range(len(dataset))]
    sizes_known = all(shape is not None for shape in declared)

    def batches_of_work():
        """Yield ``(tasks, valid)`` per batch: each task is the arguments to one sample's work."""
        for epoch in range(epochs):
            rng = np.random.default_rng(seed + epoch)
            order = rng.permutation(len(dataset)) if train else np.arange(len(dataset))
            if drop_last:
                order = order[: len(order) - len(order) % batch_size]
            for start in range(0, len(order), batch_size):
                chosen = [int(index) for index in order[start : start + batch_size]]
                valid = len(chosen)
                preloaded: list[Sample | None] = [None] * valid
                augmentations: list[TrainAugmentation | None] = [None] * valid
                if train:
                    if not sizes_known:
                        preloaded = [dataset.load(index) for index in chosen]
                    shapes = [
                        tuple(sample.image.shape[1:]) if sample is not None else declared[index]
                        for index, sample in zip(chosen, preloaded, strict=True)
                    ]
                    # Drawn here, in order, on the consuming thread; applied wherever is convenient.
                    augmentations = [draw_train_augmentation(shape, rng, scale_jitter) for shape in shapes]
                    if not allow_horizontal_flip:
                        augmentations = [TrainAugmentation(item.resize_to, item.crop, False) for item in augmentations]
                if pad_last and valid < batch_size:
                    pad = batch_size - valid
                    chosen += [chosen[-1]] * pad
                    augmentations += [augmentations[-1]] * pad
                    preloaded += [preloaded[-1]] * pad
                yield list(zip(chosen, augmentations, preloaded, strict=True)), valid

    if worker_mode not in ("thread", "process"):
        raise ValueError(f"Unknown worker_mode: {worker_mode!r}")
    use_processes = workers > 0 and worker_mode == "process"
    shape = (batch_size, 3, resolution, resolution)

    with contextlib.ExitStack() as stack:
        pool = None
        free_buffers: collections.deque = collections.deque()
        if workers > 0 and not use_processes:
            pool = stack.enter_context(concurrent.futures.ThreadPoolExecutor(workers))
        elif use_processes:
            import multiprocessing
            from multiprocessing import shared_memory

            # Spawned, not forked: the parent may already run an accelerator runtime's threads.
            pool = stack.enter_context(
                concurrent.futures.ProcessPoolExecutor(
                    workers, mp_context=multiprocessing.get_context("spawn"), initializer=_die_with_parent
                )
            )
            nbytes = int(np.prod(shape)) * np.dtype(pixel_dtype).itemsize
            for _ in range(depth + 1):
                buffer = shared_memory.SharedMemory(create=True, size=nbytes)
                stack.callback(buffer.unlink)
                stack.callback(buffer.close)
                free_buffers.append(buffer)

        def launch(tasks):
            """Claim a batch buffer and start (or run) every sample's work into it."""
            if use_processes:
                buffer = free_buffers.popleft()
                destination = ("shm", buffer.name, shape, np.dtype(pixel_dtype))
                images = np.ndarray(shape, dtype=pixel_dtype, buffer=buffer.buf)
            else:
                buffer = None
                images = np.empty((len(tasks), 3, resolution, resolution), dtype=pixel_dtype)
                destination = ("array", images)
            jobs = [
                (
                    sample if sample is not None else dataset.record(index),
                    declared[index],
                    resolution,
                    augmentation,
                    keypoint_flip_pairs,
                    destination,
                    slot,
                )
                for slot, (index, augmentation, sample) in enumerate(tasks)
            ]
            if pool is None:
                return buffer, images, [_prepare_sample(*job) for job in jobs]
            return buffer, images, [pool.submit(_prepare_sample, *job) for job in jobs]

        def finish(launched, valid):
            """Wait for a batch's samples and assemble it."""
            buffer, images, results = launched
            prepared = [result.result() if pool is not None else result for result in results]
            if buffer is not None:
                # The buffer goes back to the ring for a later batch, so the batch keeps a copy.
                images = images[: len(prepared)].copy()
                free_buffers.append(buffer)
            padded = max_targets
            if target_buckets:
                needed = min(max((len(item[1]) for item in prepared), default=0), max_targets)
                padded = min([size for size in target_buckets if needed <= size <= max_targets] + [max_targets])
            if not train:
                # Evaluation only compiles images, not targets. Preserve crowded-image annotations
                # without making an inference-only pass depend on the training padding capacity.
                padded = max(padded, max((len(item[1]) for item in prepared), default=0))
            return _collate_targets(images, prepared, padded, valid)

        in_flight: collections.deque = collections.deque()
        for tasks, valid in batches_of_work():
            in_flight.append((launch(tasks), valid))
            if len(in_flight) <= (depth if pool is not None else 0):
                continue
            yield finish(*in_flight.popleft())
        while in_flight:
            yield finish(*in_flight.popleft())


def prefetch(iterator, depth: int = 4):
    """Run ``iterator`` on a background thread so data preparation overlaps with compute.

    Both training scripts wrap their batch stream in this, so the two backends see the same
    batches and the same host-side cost.

    Args:
        iterator: The source iterable.
        depth: How many batches to keep queued ahead.

    Yields:
        Items from ``iterator``, in order.
    """
    import queue
    import threading

    if depth < 1:
        raise ValueError("prefetch depth must be positive")
    channel: queue.Queue = queue.Queue(maxsize=depth)
    sentinel = object()
    stopped = threading.Event()

    def enqueue(item) -> bool:
        while not stopped.is_set():
            try:
                channel.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        """Drain the source; close it on this thread when the consumer stops early."""
        try:
            for item in iterator:
                if not enqueue(item):
                    break
        except Exception as error:  # noqa: BLE001 - re-raised on the consumer thread
            enqueue(error)
        finally:
            try:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            finally:
                enqueue(sentinel)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            item = channel.get()
            if item is sentinel:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stopped.set()
        thread.join(timeout=1.0)
