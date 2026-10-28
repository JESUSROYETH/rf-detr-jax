"""Model metadata and category-label conventions shared by training and inference."""

from __future__ import annotations

import json
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rfdetr_jax.models.lwdetr import LWDETRSpec


class UnsupportedOptionError(TypeError, ValueError):
    """An option this port does not implement. Upstream raises ``pydantic.ValidationError`` (a
    ``ValueError``) for unknown fields. This error supports handlers for ``TypeError`` or
    ``ValueError``, but is not a ``ValidationError`` and does not provide Pydantic error details."""


def boolean_option(name, value):
    """Parse the boolean inputs accepted by upstream's Pydantic configuration."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Real) and value in (0, 1):
        return bool(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name} must be a boolean or an unambiguous boolean value") from exc
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in {"true", "on", "yes", "1", "t", "y"}:
            return True
        if normalized in {"false", "off", "no", "0", "f", "n"}:
            return False
    raise ValueError(f"{name} must be a boolean or an unambiguous boolean value")


def task_for_spec(spec: LWDETRSpec) -> str:
    """Return the model's task, rejecting ambiguous simultaneous task heads."""
    if spec.segmentation_head and spec.use_grouppose_keypoints:
        raise ValueError("choose one task: segmentation and keypoint heads cannot both be enabled")
    return "segmentation" if spec.segmentation_head else "keypoints" if spec.use_grouppose_keypoints else "detection"


def validate_model_spec(spec: LWDETRSpec) -> None:
    """Validate static geometry and task schema before initializing a model."""
    task = task_for_spec(spec)
    for name in ("resolution", "positional_encoding_size", "patch_size", "num_windows", "dec_layers",
                 "num_queries", "num_select", "group_detr", "hidden_dim", "sa_nheads", "ca_nheads",
                 "mask_downsample_ratio", "grouppose_keypoint_dim_downscale"):
        value = getattr(spec, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"model {name} must be a positive integer")
    if spec.num_classes < 0:
        raise ValueError("model num_classes must be nonnegative")
    if spec.resolution % (spec.patch_size * spec.num_windows):
        raise ValueError("resolution must be divisible by patch_size * num_windows")
    if spec.hidden_dim % spec.sa_nheads or spec.hidden_dim % spec.ca_nheads:
        raise ValueError("hidden_dim must be divisible by both attention head counts")
    if task == "keypoints":
        schema = spec.num_keypoints_per_class
        if not schema or any(type(count) is not int or count < 0 for count in schema) or not any(schema):
            raise ValueError("keypoint model needs nonnegative num_keypoints_per_class with an active slot")
        if len(schema) > spec.detector_classes:
            raise ValueError("keypoint schema cannot exceed detector class slots")
        if spec.hidden_dim % spec.grouppose_keypoint_dim_downscale:
            raise ValueError("keypoint downscale must divide hidden_dim")
    elif spec.num_keypoints_per_class:
        raise ValueError("num_keypoints_per_class requires the keypoints task")


def load_model_spec(path: str | Path) -> tuple[LWDETRSpec, dict]:
    """Read either a plain LWDETRSpec JSON or a checkpoint's metadata sidecar."""
    from rfdetr_jax.models.lwdetr import LWDETRSpec

    metadata = json.loads(Path(path).read_text())
    if not isinstance(metadata, dict):
        raise ValueError("model configuration must be a JSON object")
    config = dict(metadata.get("model_spec", metadata))
    for key in ("projector_scale", "out_feature_indexes", "num_keypoints_per_class"):
        if key in config:
            config[key] = tuple(config[key])
    spec = LWDETRSpec(**config)
    validate_model_spec(spec)
    return spec, metadata


def keypoint_category_counts(dataset) -> dict[int, int]:
    """Resolve each raw category's keypoint count, rejecting contradictory annotations."""
    counts = {}
    for category in dataset.categories:
        observed = {len(category["keypoints"])} if category.get("keypoints") else set()
        for records in dataset.annotations.values():
            for annotation in records:
                if annotation["category_id"] == category["id"] and annotation.get("keypoints"):
                    if len(annotation["keypoints"]) % 3:
                        raise ValueError("keypoint annotations must contain x/y/visibility triplets")
                    observed.add(len(annotation["keypoints"]) // 3)
        if len(observed) > 1:
            raise ValueError(f"inconsistent keypoint count for category {category['id']}")
        counts[category["id"]] = next(iter(observed), 0)
    return counts


def keypoint_category_mapping(dataset, schema: tuple[int, ...]) -> dict[int, int]:
    """Mirror upstream: sorted keypoint categories map to active model-schema slots.

    Thus COCO person category 1 maps to label 0 for preview schema [17], and to label
    1 for legacy schema [0, 17]. Remaining categories occupy unused model slots.
    """
    counts = keypoint_category_counts(dataset)
    categories = sorted(category for category, count in counts.items() if count)
    slots = [slot for slot, count in enumerate(schema) if count]
    if not categories or len(categories) > len(slots):
        raise ValueError("keypoint-bearing categories must fit the active keypoint schema slots")
    mapping = dict(zip(categories, slots, strict=False))
    for category, slot in mapping.items():
        if counts[category] != schema[slot]:
            raise ValueError(
                f"category {category} has {counts[category]} keypoints but schema slot {slot} has {schema[slot]}"
            )
    required = max(len(dataset.category_ids), max(slots) + 1)
    free = iter(slot for slot in range(required) if slot not in mapping.values())
    for category in sorted(dataset.category_ids):
        if category not in mapping:
            mapping[category] = next(free)
    return mapping


def default_model_spec(task: str = "detection", *, num_classes: int = 90,
                       keypoint_schema: tuple[int, ...] = ()) -> LWDETRSpec:
    """Pinned upstream Nano, SegNano and KeypointPreview geometries (no downloaded weights)."""
    from rfdetr_jax.models.lwdetr import LWDETRSpec

    if task == "detection":
        return LWDETRSpec(resolution=384, positional_encoding_size=24, dec_layers=2, num_classes=num_classes)
    if task == "segmentation":
        return LWDETRSpec(segmentation_head=True, patch_size=12, resolution=312, positional_encoding_size=26,
                          num_windows=1, dec_layers=4, num_queries=100, num_select=100, num_classes=num_classes)
    if task == "keypoints":
        return LWDETRSpec(use_grouppose_keypoints=True, patch_size=12, resolution=576, positional_encoding_size=48,
                          num_windows=2, dec_layers=4, num_queries=100, num_select=100, num_classes=num_classes,
                          dual_projector=True, dual_projector_kp_only=True,
                          num_keypoints_per_class=keypoint_schema or (17,))
    raise ValueError(f"unknown task {task!r}")


def validate_parameter_tree(params: dict, spec: LWDETRSpec) -> None:
    """Check every checkpoint path and shape against an abstract model initialization."""
    import jax
    import jax.numpy as jnp

    from rfdetr_jax.convert import flatten_params
    from rfdetr_jax.models.lwdetr import LWDETR

    model = LWDETR(spec=spec)
    resolution = spec.patch_size * spec.num_windows * 2
    template = jax.eval_shape(lambda: model.init(
        jax.random.key(0), jnp.zeros((1, 3, resolution, resolution)), None, train=True
    )["params"])
    import numpy as np

    actual, expected = flatten_params(params), flatten_params(template)
    if any(not np.isfinite(value).all() for value in actual.values()):
        raise ValueError("checkpoint contains nonfinite parameters")
    missing = expected.keys() - actual.keys()
    unexpected = actual.keys() - expected.keys()
    wrong_shape = [key for key in expected.keys() & actual.keys() if actual[key].shape != expected[key].shape]
    if missing or unexpected or wrong_shape:
        raise ValueError(f"checkpoint/model mismatch: {len(missing)} missing, {len(unexpected)} unexpected, "
                         f"{len(wrong_shape)} wrong shapes")
