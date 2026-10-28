# Modified JAX adaptation of RF-DETR prediction and covariance contracts.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
"""Supervision result adapter for the public RF-DETR facade.

Image lists run as one JAX batch. Torch inputs use a host copy; this adapter
makes no zero-copy guarantee.
"""
from __future__ import annotations

import io
import operator
import sys
import warnings
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

COCO_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

COCO_CLASS_NAMES: list[str] = [name for _, name in sorted(COCO_CLASSES.items())]


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_shape(shape, patch_size, num_windows):
    """Validate public inference shape using integer-index and window divisibility rules."""
    patch = _positive_integer(patch_size, "patch_size")
    windows = _positive_integer(num_windows, "num_windows")
    try:
        height, width = shape
    except (TypeError, ValueError) as exc:
        raise ValueError("shape must contain exactly two dimensions") from exc
    shape = (_positive_integer(height, "shape height"), _positive_integer(width, "shape width"))
    if any(dim % (patch * windows) for dim in shape):
        raise ValueError(f"shape dimensions must be divisible by patch_size * num_windows ({patch * windows})")
    return shape


def _image_array(image):
    """Decode RGB HWC bytes or normalized floats without quantizing model input."""
    if isinstance(image, str | Path):
        if urlparse(str(image)).scheme in ("http", "https"):
            import requests

            response = requests.get(str(image), timeout=30)
            response.raise_for_status()
            image = io.BytesIO(response.content)
        with Image.open(image) as opened:
            return np.array(opened.convert("RGB"))
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(image, torch.Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Tensor image must have shape (3, H, W)")
        image = image.detach().to(device="cpu", dtype=torch.float32).numpy().transpose(1, 2, 0)
    if not isinstance(image, np.ndarray):
        raise TypeError("images must be paths, URLs, PIL images, RGB NumPy arrays or Torch tensors")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("NumPy image must have nonempty shape (H, W, 3)")
    if image.dtype != np.uint8 and (
        not np.issubdtype(image.dtype, np.floating) or not np.isfinite(image).all()
        or (image < 0).any() or (image > 1).any()
    ):
        raise ValueError("Image must be uint8 or finite normalized floating values in [0, 1]")
    return image


def precision_cholesky_to_pixel_covariance(precision_cholesky, source_shape):
    """Invert normalized lower-triangular precision factors and scale to pixels."""
    precision = np.asarray(precision_cholesky, dtype=np.float64)
    shape = np.asarray(source_shape, dtype=np.float64)
    if precision.ndim != 3 or precision.shape[-1] != 3:
        raise ValueError("precision_cholesky must have shape (N, K, 3)")
    if shape.shape != (precision.shape[0], 2):
        raise ValueError("source_shape must have shape (N, 2)")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        l11, l21, l22 = np.exp(precision[..., 0]), precision[..., 1], np.exp(precision[..., 2])
        inv_det = 1 / (l11 * l11 * l22 * l22)
        h, w = shape[:, 0:1], shape[:, 1:2]
        xx = w * w * inv_det * (l21 * l21 + l22 * l22)
        xy = w * h * (inv_det * (-l11 * l21) + 0.0)
        yy = h * h * inv_det * l11 * l11
        valid = np.isfinite(precision).all(-1) & np.isfinite(xx) & np.isfinite(xy) & np.isfinite(yy)
        result = np.full((*precision.shape[:2], 2, 2), np.nan, dtype=np.float32)
        result[..., 0, 0] = np.where(valid, xx, np.nan)
        result[..., 0, 1] = result[..., 1, 0] = np.where(valid, xy, np.nan)
        result[..., 1, 1] = np.where(valid, yy, np.nan)
    return result


def _class_names(model, labels):
    names = list(model.class_names)
    config = model.model_config
    count = config.num_classes
    schema = getattr(config, "num_keypoints_per_class", ()) or ()
    coco = count > len(names) and names == COCO_CLASS_NAMES
    legacy = bool(schema and schema[0] == 0)
    explicit_mapping = getattr(model._backend, "metadata", {}).get("category_to_label", {})
    if explicit_mapping and len(explicit_mapping) == len(names):
        # Training stores names in model-label order. Native COCO labels can be
        # sparse, so enumerating that list would attach names to different slots.
        mapping = dict(zip(sorted(int(value) for value in explicit_mapping.values()), names, strict=True))
    elif coco:
        mapping = COCO_CLASSES
    elif legacy:
        mapping = dict(zip((index for index, value in enumerate(schema) if value > 0), names, strict=False))
    else:
        mapping = dict(enumerate(names))
    sentinel = 0 if legacy else count
    unexpected = [int(value) for value in labels if value not in mapping and (coco or value != sentinel)]
    if unexpected:
        warnings.warn(f"predict encountered unmapped class_id(s): {unexpected[:5]}", UserWarning, stacklevel=3)
    return np.asarray([mapping.get(value, "__background__" if not coco and value == sentinel else "")
                       for value in labels], dtype=object)


def predict(model, images, threshold=.5, shape=None, patch_size=None, include_source_image=True, **kwargs):
    """Return actual Supervision Detections/KeyPoints with upstream field semantics.

    Lists and tuples return lists, including one-image lists. NumPy input uses RGB
    HWC uint8 or normalized floats; Torch input uses CHW values in [0, 1]. Shapes
    are (height, width). Unsupported keyword arguments fail explicitly.
    """
    from supervision import Detections, KeyPoints

    from rfdetr_jax.configuration import UnsupportedOptionError

    if kwargs:
        raise UnsupportedOptionError(f"Unsupported predict argument(s): {', '.join(sorted(kwargs))}")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")
    config = model.model_config
    model_spec = model._backend.spec
    if patch_size is not None and patch_size != model_spec.patch_size:
        raise ValueError(f"predict(patch_size={patch_size}) does not match the instantiated model's "
                         f"patch_size={model_spec.patch_size}")
    # The inference geometry is the loaded model's, as upstream reads ``model.resolution``.
    resolution = getattr(model_spec, "resolution", None) or config.resolution
    shape = validate_shape((resolution, resolution) if shape is None else shape,
                           model_spec.patch_size, getattr(config, "num_windows", 1))
    # A caller-provided validation patch size never changes the trained backbone.
    actual_block = model._backend.spec.patch_size * model._backend.spec.num_windows
    if any(dim % actual_block for dim in shape):
        raise ValueError(f"shape must also be divisible by the actual model block size ({actual_block})")
    single = not isinstance(images, list | tuple)
    arrays = [_image_array(value) for value in ([images] if single else images)]
    results = []
    options = dict(threshold=threshold, shape=shape, antialias=False,
                   trace_alpha=getattr(config, "postprocess_trace_alpha", .2), model_labels=True)
    if hasattr(model._backend, "predict_batch"):
        raw_results = model._backend.predict_batch(arrays, **options)
    else:
        raw_results = [model._backend.predict(array, **options) for array in arrays]
    for array, raw in zip(arrays, raw_results, strict=True):
        keep = raw["scores"] > threshold
        scores = np.asarray(raw["scores"][keep], dtype=np.float32)
        labels = np.asarray(raw["labels"][keep], dtype=int)
        boxes = np.asarray(raw["boxes"][keep], dtype=np.float32)
        source_shape = np.tile(np.array(array.shape[:2], dtype=np.int64), (len(scores), 1))
        data = {"source_shape": source_shape, "class_name": _class_names(model, labels)}
        source = None
        if include_source_image:
            source = array.copy() if array.dtype == np.uint8 else (array * 255).clip(0, 255).astype(np.uint8)
        if "keypoints" in raw:
            points = np.asarray(raw["keypoints"][keep], dtype=np.float32)
            data["xyxy"] = boxes
            if "keypoint_precision_cholesky" in raw:
                precision = np.asarray(raw["keypoint_precision_cholesky"][keep], dtype=np.float32)
                data["keypoint_precision_cholesky"] = precision
                if len(scores):
                    data["covariance"] = precision_cholesky_to_pixel_covariance(precision, source_shape)
            if include_source_image:
                data["source_image"] = [source for _ in scores]
            result = KeyPoints(xy=points[..., :2], keypoint_confidence=points[..., 2],
                               detection_confidence=scores, class_id=labels, visible=points[..., 2] > 0, data=data)
        else:
            result = Detections(xyxy=boxes, confidence=scores, class_id=labels,
                                mask=np.asarray(raw["masks"][keep], dtype=bool) if "masks" in raw else None, data=data)
            if include_source_image:
                result.metadata["source_image"] = source
        results.append(result)
    return results[0] if single else results
