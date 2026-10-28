# Modified JAX dataset adapter from RF-DETR's YOLO dataset contracts.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
"""Read YOLO detection, polygon and pose datasets through the COCO pipeline.

Only a temporary annotation index is written. Images and original labels stay in
place. Masks use the same PIL rasterization as the upstream YOLO reader, encoded
as COCO RLE so subsequent COCO parsing does not change polygon edge semantics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

_EXTENSIONS = {".bmp", ".dng", ".jpg", ".jpeg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def read_yolo_config(root):
    root = Path(root)
    path = next((root / name for name in ("data.yaml", "data.yml") if (root / name).is_file()), None)
    if path is None:
        raise FileNotFoundError(f"YOLO dataset {root} needs data.yaml or data.yml")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YOLO config {path} must contain a mapping")
    names = data.get("names")
    if isinstance(names, dict):
        if any(not str(key).isdigit() for key in names):
            raise ValueError("YOLO names keys must be contiguous integers starting at zero")
        normalized = {int(key): value for key, value in names.items()}
        if len(normalized) != len(names) or sorted(normalized) != list(range(len(names))):
            raise ValueError("YOLO names keys must be contiguous integers starting at zero")
        names = [normalized[index] for index in range(len(names))]
    if not isinstance(names, list) or not names:
        raise ValueError("YOLO names must be a nonempty list or contiguous mapping")
    return {**data, "names": [str(name) for name in names]}


def yolo_pose_schema(root):
    """Return pose names, dimensions and flat mirror pairs declared in YAML."""
    data = read_yolo_config(root)
    shape = data.get("kpt_shape")
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2
            or any(type(value) is not int for value in shape) or shape[0] < 1 or shape[1] not in (2, 3)):
        raise ValueError("YOLO pose requires kpt_shape: [positive_count, 2_or_3]")
    count, dimension = shape
    names = data.get("kpt_names")
    if isinstance(names, dict):
        names = names.get(0, names.get("0"))
    if names is None:
        names = [f"keypoint_{index}" for index in range(count)]
    if not isinstance(names, list) or len(names) != count:
        raise ValueError("YOLO kpt_names length must equal kpt_shape count")
    flip = data.get("flip_idx", [])
    if flip and (not isinstance(flip, list) or any(type(index) is not int for index in flip)
                 or sorted(flip) != list(range(count))):
        raise ValueError("YOLO flip_idx must be a permutation of keypoint indexes")
    pairs = [index for left, right in enumerate(flip) if left < right and flip[right] == left
             for index in (left, right)]
    return dict(names=[str(name) for name in names], count=count, dimension=dimension,
                flip_pairs=pairs, oks_sigmas=[0.1] * count)


def _split_dirs(root, split, data):
    root = Path(root).resolve()
    if split not in {"train", "val", "test"}:
        raise ValueError("YOLO split must be train, val or test")
    declared = data.get(split, data.get("valid") if split == "val" else None)
    if declared is not None:
        base = root / data.get("path", "")
        if not isinstance(declared, str):
            raise ValueError("YOLO split paths must be directory strings")
        images = (base / declared).resolve()
        if images.is_relative_to(root):
            parts = images.relative_to(root).parts
            if "images" in parts:
                offset = parts.index("images")
                labels = root.joinpath(*parts[:offset], "labels", *parts[offset + 1:])
            else:
                labels, images = images / "labels", images / "images"
            if images.is_dir() and labels.is_dir():
                return images, labels
        if split == "test":
            raise ValueError("Declared YOLO test split could not be resolved inside the dataset")
    for name in (("valid", "val") if split == "val" else (split,)):
        images, labels = root / name / "images", root / name / "labels"
        if images.is_dir() and labels.is_dir():
            return images, labels
    raise FileNotFoundError(f"YOLO {split} split needs image and label directories in {root}")


def _parse_row(tokens, classes, width, height, pose):
    if len(tokens) < 5:
        raise ValueError("expected class and four box coordinates, or class and polygon points")
    category = int(tokens[0])
    if not 0 <= category < classes:
        raise ValueError("class ID is outside the dataset names")
    values = np.asarray(tokens[1:], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("label coordinates must be finite")
    keypoints = polygon = None
    if pose is not None or len(tokens) == 5:
        if pose is not None and len(tokens) != 5 + pose["count"] * pose["dimension"]:
            raise ValueError("pose row length disagrees with kpt_shape")
        cx, cy, bw, bh = values[:4]
        box = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dtype=np.float32)
        polygon = box[[0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 2)
        if pose is not None:
            raw = values[4:].reshape(pose["count"], pose["dimension"])
            xy = raw[:, :2].copy()
            if pose["dimension"] == 3:
                visibility = raw[:, 2]
                if np.any((visibility < 0) | (visibility > 2)):
                    raise ValueError("keypoint visibility must be in [0, 2]")
            else:
                xy[np.any(xy < 0, axis=1)] = 0
                visibility = np.any(xy != 0, axis=1).astype(np.float32) * 2
            xy = np.clip(xy, 0, 1) * [width, height]
            xy[visibility <= 0] = 0
            keypoints = np.column_stack([xy, visibility]).astype(np.float32)
    else:
        if len(values) < 6 or len(values) % 2:
            raise ValueError("polygon requires at least three paired coordinates")
        polygon = values.reshape(-1, 2)
        box = np.concatenate([polygon.min(axis=0), polygon.max(axis=0)])
    box *= np.array([width, height, width, height], dtype=np.float32)
    polygon = np.clip(polygon * np.array([width, height], dtype=np.float32), 0, [width - 1, height - 1])
    return category, box, polygon, keypoints


def convert_yolo_split(root, split, destination, *, segmentation=False, keypoints=False):
    """Write a COCO annotation index and return its image root and file path."""
    if segmentation and keypoints:
        raise ValueError("YOLO segmentation and pose cannot be enabled together")
    data = read_yolo_config(root)
    pose = yolo_pose_schema(root) if keypoints else None
    images_dir, labels_dir = _split_dirs(root, split, data)
    categories = [{"id": index, "name": name, "supercategory": "none"}
                  for index, name in enumerate(data["names"])]
    if pose is not None:
        for category in categories:
            category.update(keypoints=pose["names"], skeleton=[])
    images, annotations = [], []
    paths = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in _EXTENSIONS)
    for image_id, path in enumerate(paths):
        with Image.open(path) as image:
            width, height = image.size
        images.append(dict(id=image_id, file_name=path.name, width=width, height=height))
        label = labels_dir / (path.stem + ".txt")
        for number, line in enumerate(label.read_text().splitlines() if label.exists() else (), 1):
            if not line.strip():
                continue
            try:
                category, box, polygon, points = _parse_row(line.split(), len(categories), width, height, pose)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid YOLO label {label}:{number}: {exc}") from exc
            x1, y1, x2, y2 = box.tolist()
            annotation = dict(id=len(annotations) + 1, image_id=image_id, category_id=category,
                              bbox=[x1, y1, x2 - x1, y2 - y1], area=(x2 - x1) * (y2 - y1), iscrowd=0)
            if segmentation:
                from pycocotools import mask as mask_util

                mask = Image.new("L", (width, height), 0)
                ImageDraw.Draw(mask).polygon([tuple(point) for point in polygon.tolist()], fill=1)
                rle = mask_util.encode(np.asfortranarray(mask, dtype=np.uint8))
                rle["counts"] = rle["counts"].decode("ascii")
                annotation["segmentation"] = rle
            if points is not None:
                annotation.update(keypoints=points.ravel().tolist(), num_keypoints=int((points[:, 2] > 0).sum()))
            annotations.append(annotation)
    destination = Path(destination)
    destination.write_text(json.dumps(dict(info={}, images=images, annotations=annotations, categories=categories)))
    return images_dir, destination
