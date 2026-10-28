"""Checkpoint-backed prediction for detection, instance segmentation and keypoints.

The interface follows RF-DETR's image-to-predictions workflow, using JAX arrays
internally and NumPy results. It does not download weights or import PyTorch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from rfdetr_jax.configuration import load_model_spec, task_for_spec, validate_model_spec
from rfdetr_jax.convert import flatten_params, load_params_npz
from rfdetr_jax.datasets.coco import IMAGENET_MEAN, IMAGENET_STD
from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec
from rfdetr_jax.models.postprocess import post_process_outputs
from rfdetr_jax.utilities.resample import resize_2d_numpy


class RFDETR:
    """A trained JAX model with explicit geometry and category metadata.

    Construct with :meth:`from_checkpoint`; :meth:`predict` accepts an RGB HWC
    NumPy image (uint8 or normalized float), a PIL image, or a local image path. Results are NumPy arrays
    in original-image pixel coordinates, with raw dataset category IDs when the
    checkpoint records a category mapping.
    """

    def __init__(self, spec: LWDETRSpec, params: dict, metadata: dict | None = None):
        validate_model_spec(spec)
        self.spec = spec
        self.params = params
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("checkpoint metadata must be an object")
        self.metadata = metadata if metadata is not None else {}
        self.task = task_for_spec(spec)
        self.model = LWDETR(spec=spec)
        if "task" in self.metadata and self.metadata["task"] != self.task:
            raise ValueError("checkpoint task metadata does not match the model specification")

        def integer(value, name, *, string_key=False):
            if string_key and isinstance(value, str):
                try:
                    value = int(value)
                except ValueError as exc:
                    raise ValueError(f"{name} must contain nonnegative integers") from exc
            if not isinstance(value, int | np.integer) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must contain nonnegative integers")
            return int(value)

        category_ids = self.metadata.get("category_ids")
        if category_ids is not None:
            if not isinstance(category_ids, list | tuple) or not category_ids:
                raise ValueError("category_ids must be a nonempty sequence")
            category_ids = [integer(value, "category_ids") for value in category_ids]
            if len(set(category_ids)) != len(category_ids):
                raise ValueError("category_ids must be unique")
        category_names = self.metadata.get("category_names")
        if category_names is not None and (
            not isinstance(category_names, list | tuple)
            or category_ids is None or len(category_names) != len(category_ids)
            or not all(isinstance(name, str) for name in category_names)
        ):
            raise ValueError("category_names must match category_ids")
        mapping = self.metadata.get("category_to_label")
        self.label_to_category = {}
        if mapping is not None:
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("category_to_label must be a nonempty object")
            normalized = {integer(category, "category_to_label", string_key=True):
                          integer(label, "category_to_label") for category, label in mapping.items()}
            if len(normalized) != len(mapping):
                raise ValueError("category_to_label has duplicate category ids")
            self.label_to_category = {label: category for category, label in normalized.items()}
            if len(self.label_to_category) != len(mapping):
                raise ValueError("category_to_label must map categories to distinct model labels")
            if category_ids is not None and set(normalized) != set(category_ids):
                raise ValueError("category_to_label must cover exactly category_ids")
        self.valid_labels = sorted(self.label_to_category) if mapping else category_ids
        if self.valid_labels is not None and any(label >= spec.detector_classes for label in self.valid_labels):
            raise ValueError("category metadata contains labels outside the model class slots")

        def forward(pixels):
            with jax.default_matmul_precision("highest"):
                outputs = self.model.apply({"params": self.params}, pixels, None, train=False)
            return {key: value for key, value in outputs.items() if key.startswith("pred_")}

        self._forward = jax.jit(forward)

    @classmethod
    def from_checkpoint(
        cls, params_path: str | Path, *, model_config: str | Path | None = None,
        spec: LWDETRSpec | None = None,
    ) -> RFDETR:
        """Load NPZ parameters and verify names/shapes against the declared model.

        Uses ``<params_path>.json`` automatically when present. A converted file
        without metadata needs an explicit ``model_config`` or ``spec``. No
        geometry or keypoint schema is guessed from a filename.
        """
        if model_config is not None and spec is not None:
            raise ValueError("provide model_config or spec, not both")
        metadata: dict[str, Any] = {}
        if spec is None:
            config_path = Path(model_config) if model_config is not None else Path(str(params_path) + ".json")
            if not config_path.is_file():
                raise ValueError("checkpoint requires model metadata; provide model_config or an explicit LWDETRSpec")
            spec, metadata = load_model_spec(config_path)
        validate_model_spec(spec)
        params = load_params_npz(params_path)
        model = LWDETR(spec=spec)
        # Shape tracing creates no random initialized model weights on the device.
        template = jax.eval_shape(lambda: model.init(
            jax.random.key(0), jnp.zeros((1, 3, spec.resolution, spec.resolution)), None, train=True,
        )["params"])
        actual, expected = flatten_params(params), flatten_params(template)
        if actual.keys() != expected.keys():
            raise ValueError("checkpoint parameter names do not match the model specification")
        for key in expected:
            if actual[key].shape != expected[key].shape:
                raise ValueError(f"checkpoint shape mismatch for {'.'.join(key)}")
            if not np.isfinite(actual[key]).all():
                raise ValueError(f"checkpoint contains nonfinite parameters at {'.'.join(key)}")
        return cls(spec, params, metadata)

    def predict(
        self, image: str | Path | Image.Image | np.ndarray, threshold: float = 0.5,
        *, shape: tuple[int, int] | None = None, trace_alpha: float = 0.2, antialias: bool = True,
        model_labels: bool = False,
    ) -> dict:
        """Predict one image; select/filter before resizing masks to bound memory.

        Returns ``scores``, ``labels`` and ``boxes`` (absolute xyxy), plus boolean
        ``masks`` or absolute ``keypoints`` and precision matrices for those tasks.
        Keypoint arrays are padded to the largest class schema; ``keypoint_counts``
        gives the number of real points in each detection. ``model_labels=True``
        retains all selected model slots, including slots absent from dataset metadata.
        The default filters dataset categories and restores their raw IDs.
        """
        return self.predict_batch([image], threshold=threshold, shape=shape, trace_alpha=trace_alpha,
                                  antialias=antialias, model_labels=model_labels)[0]

    def _prepare_image(self, image, shape, antialias):
        if isinstance(image, str | Path):
            # Stored orientation, as the training decoder and upstream predict() use it.
            with Image.open(image) as opened:
                pixels = np.asarray(opened.convert("RGB"))
        elif isinstance(image, Image.Image):
            pixels = np.asarray(image.convert("RGB"))
        else:
            pixels = np.asarray(image)
        if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 1:
            raise ValueError("NumPy input must be a nonempty HWC RGB uint8 or normalized floating image")
        if pixels.dtype == np.uint8:
            pixels = pixels.astype(np.float32) / 255
        elif (not np.issubdtype(pixels.dtype, np.floating) or not np.isfinite(pixels).all()
              or (pixels < 0).any() or (pixels > 1).any()):
            raise ValueError("NumPy input must be HWC RGB uint8 or finite floating values in [0, 1]")
        else:
            pixels = pixels.astype(np.float32)
        height, width = pixels.shape[:2]
        shape = shape if shape is not None else (self.spec.resolution, self.spec.resolution)
        block = self.spec.patch_size * self.spec.num_windows
        if len(shape) != 2 or any(isinstance(dim, bool) or not isinstance(dim, int | np.integer)
                                  or dim <= 0 or dim % block for dim in shape):
            raise ValueError(f"shape must contain two positive integer dimensions divisible by {block}")
        resized = resize_2d_numpy(pixels.transpose(2, 0, 1)[None], shape, mode="bilinear",
                                  align_corners=False, antialias=antialias)
        normalized = (resized - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
        return normalized, (height, width)

    def predict_batch(self, images, threshold=0.5, *, shape=None, trace_alpha=0.2, antialias=True,
                      model_labels=False):
        """Run one actual model batch, retaining each image's original output coordinates."""
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and between zero and one")
        if not isinstance(images, list | tuple):
            raise TypeError("predict_batch requires a list or tuple of images")
        if not images:
            return []
        prepared = [self._prepare_image(image, shape, antialias) for image in images]
        normalized = np.concatenate([item[0] for item in prepared], axis=0)
        sizes = np.asarray([item[1] for item in prepared])
        outputs = self._forward(jnp.asarray(normalized))
        results = post_process_outputs(
            outputs, sizes, self.spec.num_select,
            num_keypoints_per_class=self.spec.num_keypoints_per_class,
            score_threshold=threshold, category_ids=None if model_labels else self.valid_labels,
            trace_alpha=trace_alpha,
        )
        for result in results:
            labels = np.asarray(result["labels"], dtype=np.int32)
            if self.task == "keypoints":
                result["keypoint_counts"] = np.asarray(
                    [self.spec.num_keypoints_per_class[label] if label < len(self.spec.num_keypoints_per_class) else 0
                     for label in labels], dtype=np.int32,
                )
            if self.label_to_category and not model_labels:
                result["labels"] = np.asarray([self.label_to_category[int(label)] for label in labels], dtype=np.int32)
        return results


def prediction_records(result: dict) -> list[dict]:
    """Serialize a result without expanding boolean masks into huge JSON matrices."""
    records = []
    for i, (score, label, box) in enumerate(zip(result["scores"], result["labels"], result["boxes"], strict=True)):
        record = {"label": int(label), "score": float(score), "box_xyxy": np.asarray(box).tolist()}
        if "masks" in result:
            from pycocotools import mask as mask_utils

            encoded = mask_utils.encode(np.asfortranarray(result["masks"][i], dtype=np.uint8))
            record["segmentation"] = {"size": encoded["size"], "counts": encoded["counts"].decode("ascii")}
        if "keypoints" in result:
            count = int(result["keypoint_counts"][i])
            record["keypoints"] = np.asarray(result["keypoints"][i, :count]).tolist()
        records.append(record)
    return records


def main() -> None:
    """Predict a local image from installed checkpoint files and write JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True, help="Converted or trained NPZ parameter file.")
    parser.add_argument("--model-config", help="Model JSON; defaults to <params>.json.")
    parser.add_argument("--image", required=True, help="Local image path.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", help="JSON destination; print to stdout when omitted.")
    args = parser.parse_args()
    if args.out and Path(args.out).resolve() in {
        Path(args.params).resolve(), Path(str(args.params) + ".json").resolve(), Path(args.image).resolve(),
        *([Path(args.model_config).resolve()] if args.model_config else []),
    }:
        parser.error("--out must not overwrite an input file")
    try:
        detector = RFDETR.from_checkpoint(args.params, model_config=args.model_config)
        result = detector.predict(args.image, args.threshold)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    payload = json.dumps({"task": detector.task, "predictions": prediction_records(result)}, indent=2, allow_nan=False)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload + "\n")
    else:
        print(payload)


if __name__ == "__main__":
    main()
