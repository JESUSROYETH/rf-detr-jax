# ------------------------------------------------------------------------
# RF-DETR JAX — modified public interface of RF-DETR, Copyright (c) 2025 Roboflow.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""RF-DETR's user-facing model interface backed by JAX/Flax."""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import asdict, replace
from types import SimpleNamespace

import jax


class RFDETR:
    """Construct a named variant, train it and return Supervision predictions.

    Constructor arguments follow the pinned RF-DETR model configuration. Named
    variants load their official pretrained weights by default. Set
    ``pretrain_weights=None`` explicitly to initialize a model randomly.
    """

    _variant_name = "RFDETRBase"
    size = None
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    def __init__(self, *, trust_checkpoint=False, **kwargs):
        from rfdetr_jax.checkpoints import load_variant

        resolve_device(kwargs.get("device", "auto"))
        if self._variant_name in self._DEPRECATED_VARIANTS:
            warnings.warn(f"{type(self).__name__} is deprecated upstream; prefer a current named variant",
                          FutureWarning, stacklevel=2)
        if "pretrain_weights" in kwargs and kwargs["pretrain_weights"] is None:
            from rfdetr_jax.variants import VARIANT_DEFAULTS, PretrainWeightsCompatibilityWarning

            default_weights = VARIANT_DEFAULTS[self._variant_name].get("pretrain_weights")
            if default_weights:
                warnings.warn(
                    f"{type(self).__name__} was instantiated with pretrain_weights=None. The model will be "
                    "initialised from scratch, which typically produces lower accuracy than fine-tuning from "
                    f"the published checkpoint ({default_weights!r}).", PretrainWeightsCompatibilityWarning,
                    stacklevel=2)
        self._install(*load_variant(self._variant_name, trust_checkpoint=trust_checkpoint, **kwargs))
        self._has_been_trained = False

    _DEPRECATED_VARIANTS = frozenset({"RFDETRBase", "RFDETRLargeDeprecated", "RFDETRSegPreview"})

    def _install(self, spec, params, metadata, model_config):
        from rfdetr_jax.inference import RFDETR as Backend

        existing = getattr(self, "model_config", None)
        if existing is not None and existing is not model_config:
            # Upstream mutates model_config in place; callers holding a reference must see updates.
            vars(existing).clear()
            vars(existing).update(vars(model_config))
            model_config = existing
        self.model_config = model_config
        self._device = resolve_device(getattr(model_config, "device", "auto"))
        self._backend = Backend(spec, jax.device_put(params, self._device), metadata)
        if metadata.get("training_precision"):
            from rfdetr_jax.models.lwdetr import LWDETR

            # Lightning mixed precision leaves master parameters in FP32. Its
            # ordinary predict() does not enter autocast; retain the training
            # spec for resume, but execute this public inference path in FP32.
            self._backend.model = LWDETR(spec=replace(spec, compute_dtype="float32", head_dtype="float32"))
        self.callbacks = getattr(self, "callbacks", defaultdict(list))
        self._has_been_trained = getattr(self, "_has_been_trained", False)
        self._is_optimized_for_inference = False
        self._optimized_resolution = None
        self._optimized_batch_size = None
        self._optimized_dtype = None
        self._optimized_inplace = False
        self._optimized_has_been_compiled = False
        self._unoptimized_forward = None
        self.model = SimpleNamespace(model=self._backend.model, args=model_config,
                                     resolution=spec.resolution, device=self._device,
                                     inference_model=None, class_names=self.class_names)

    @property
    def class_names(self):
        """Names in model label order, matching upstream prediction consumers."""
        metadata = self._backend.metadata
        return list(metadata.get("class_names", metadata.get("category_names", ())))

    @property
    def params(self):
        """JAX parameter pytree with upstream parameter names and layouts."""
        return self._backend.params

    @property
    def spec(self):
        return self._backend.spec

    def get_model_config(self, **kwargs):
        from rfdetr_jax.variants import variant_config

        return variant_config(self._variant_name, **kwargs)

    def get_train_config(self, **kwargs):
        from rfdetr_jax.api_training import training_config
        from rfdetr_jax.configuration import task_for_spec

        return training_config(task=task_for_spec(self.spec), **kwargs)

    @classmethod
    def from_checkpoint(cls, path, *, trust_checkpoint=False, **kwargs):
        """Resolve an upstream or native checkpoint and return its concrete variant."""
        from rfdetr_jax import variants
        from rfdetr_jax.checkpoints import inspect_checkpoint, load_variant

        try:
            inferred, options = inspect_checkpoint(path, trust_checkpoint=trust_checkpoint)
        except ValueError as exc:
            if cls is RFDETR or "cannot identify checkpoint variant" not in str(exc):
                raise
            inferred, options = cls._variant_name, {"pretrain_weights": str(path)}
        # The checkpoint's recorded variant wins, as upstream resolves the class from ``model_name``.
        name = inferred
        options.update(kwargs)
        options["pretrain_weights"] = str(path)
        model_class = getattr(variants, name) if (cls is RFDETR or name != cls._variant_name) else cls
        model = object.__new__(model_class)
        model._install(*load_variant(name, trust_checkpoint=trust_checkpoint, **options))
        # An inferred class count must remain eligible for training-dataset alignment.
        model.model_config.model_fields_set = set(kwargs)
        model._has_been_trained = True
        return model

    def train(self, **kwargs):
        """Train this model and update the in-memory weights used by predict()."""
        from rfdetr_jax.api_training import train

        self._require_base_model("train")
        self.remove_optimized_model()
        return train(self, **kwargs)

    def evaluate(self, *, split="test", **kwargs):
        """Evaluate the in-memory parameters without changing inference geometry."""
        from rfdetr_jax.api_training import evaluate

        self._require_base_model("evaluate")
        return evaluate(self, split=split, **kwargs)

    def predict(self, images, threshold=0.5, shape=None, patch_size=None, include_source_image=True, **kwargs):
        """Return Supervision Detections/KeyPoints for one image or a list."""
        from rfdetr_jax.prediction import predict

        if self._is_optimized_for_inference:
            requested = tuple(shape) if shape is not None else (self.spec.resolution, self.spec.resolution)
            if requested != self._optimized_resolution:
                raise ValueError(f"model was optimized for shape {self._optimized_resolution}, got {requested}")
            count = len(images) if isinstance(images, list | tuple) else 1
            if self._optimized_has_been_compiled and count != self._optimized_batch_size:
                raise ValueError(f"model was compiled for batch size {self._optimized_batch_size}, got {count}")
        with jax.default_device(self._device):
            return predict(self, images, threshold, shape, patch_size, include_source_image, **kwargs)

    def save(self, path):
        """Save native inference parameters and the exact model/category metadata."""
        from rfdetr_jax.checkpoints import save_checkpoint

        self._require_base_model("save")
        save_checkpoint(path, self.params, self.spec, self._backend.metadata, self.model_config)

    def inference(self, compile=True, batch_size=1, dtype="float32", *, inplace=False,
                  shape=None, patch_size=None):
        """Build a dtype-specific inference forward; optionally compile and warm it.

        Normal optimization is reversible and retains original training weights.
        Inplace mode casts the stored weights and clears the base-model facade,
        matching upstream's inference-only lifetime. Compiled models enforce the
        selected batch size and image shape on subsequent prediction calls.
        """
        import jax.numpy as jnp

        from rfdetr_jax.models.lwdetr import LWDETR
        from rfdetr_jax.prediction import validate_shape

        self._require_base_model("optimize")
        if inplace and compile:
            raise ValueError("inference(inplace=True) requires compile=False")
        dtype_name = str(dtype).removeprefix("torch.")
        try:
            dtype_value = jnp.dtype(dtype_name if isinstance(dtype, str) or dtype_name.startswith(
                ("float", "bfloat", "int", "bool")) else dtype)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"dtype must be a floating-point dtype or dtype name, got {dtype!r}") from exc
        if not jnp.issubdtype(dtype_value, jnp.floating):
            raise ValueError("dtype must be floating point")
        if dtype_value.name not in {"float32", "float16", "bfloat16"}:
            raise NotImplementedError("JAX inference optimization supports float32, float16 and bfloat16")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        shape = validate_shape((self.spec.resolution, self.spec.resolution) if shape is None else shape,
                               self.spec.patch_size if patch_size is None else patch_size, self.spec.num_windows)
        block = self.spec.patch_size * self.spec.num_windows
        if any(dimension % block for dimension in shape):
            raise ValueError(f"shape must be divisible by the actual model block size ({block})")
        self.remove_optimized_model()
        original_forward = self._backend._forward
        optimized_spec = replace(self.spec, compute_dtype=dtype_value.name, head_dtype="activation")
        optimized_model = LWDETR(optimized_spec)
        with jax.default_device(self._device):
            optimized_params = jax.tree.map(lambda value: jnp.asarray(value, dtype=dtype_value), self.params)

            def forward(pixels):
                with jax.default_matmul_precision("highest"):
                    outputs = optimized_model.apply({"params": optimized_params},
                                                    pixels.astype(dtype_value), None, train=False)
                return {key: value for key, value in outputs.items() if key.startswith("pred_")}

            optimized_forward = jax.jit(forward) if compile else forward
            if compile:
                output = optimized_forward(jnp.zeros((batch_size, 3, *shape), dtype=dtype_value))
                jax.block_until_ready(output)
        # Commit the change only after tracing/execution succeeds.
        self._backend._forward = optimized_forward
        self.model.inference_model = optimized_forward
        self._unoptimized_forward = None if inplace else original_forward
        self._is_optimized_for_inference = True
        self._optimized_has_been_compiled = bool(compile)
        self._optimized_resolution = shape
        self._optimized_batch_size = batch_size if compile else None
        self._optimized_dtype = dtype_value
        self._optimized_inplace = bool(inplace)
        if inplace:
            self._backend.params = optimized_params
            self.model.model = None

    def optimize_for_inference(self, compile=True, batch_size=1, dtype="float32", *, inplace=False,
                               shape=None, patch_size=None):
        """Compatibility alias for upstream's renamed inference() method."""
        return self.inference(compile, batch_size, dtype, inplace=inplace, shape=shape, patch_size=patch_size)

    @property
    def is_optimized_inplace(self):
        return self._optimized_inplace

    def _require_base_model(self, operation):
        if self._optimized_inplace:
            raise RuntimeError(f"Cannot {operation} after inplace optimization; create or reload a model")

    def remove_optimized_model(self):
        """Restore the original inference forward unless the base model was cleared."""
        if self._optimized_inplace:
            warnings.warn("remove_optimized_model() has no effect after inplace optimization; reload the model",
                          UserWarning, stacklevel=2)
            return
        if self._unoptimized_forward is not None:
            self._backend._forward = self._unoptimized_forward
        self._unoptimized_forward = None
        self._is_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self.model.inference_model = None
        self._optimized_resolution = self._optimized_batch_size = self._optimized_dtype = None

    def export(self, **kwargs):
        """Export through the upstream interchange model built from JAX parameters."""
        from rfdetr_jax.export import export

        self._require_base_model("export")
        return export(self, **kwargs)

    def export_for_roboflow(self, output_dir):
        """Write an upstream-compatible deployment bundle without uploading it."""
        from rfdetr_jax.export import export_for_roboflow

        self._require_base_model("export")
        return export_for_roboflow(self, output_dir)

    def _checkpoint_metadata(self):
        return {**self._backend.metadata, "model_spec": asdict(self.spec),
                "model_name": type(self).__name__, "class_names": self.class_names}


def resolve_device(device):
    """Translate upstream CPU/CUDA/TPU device spelling to an available JAX device."""
    if device is None or str(device) == "auto":
        return jax.devices()[0]
    value = str(device)
    platform, _, index = value.partition(":")
    platform = {"cuda": "gpu", "xla": "tpu"}.get(platform, platform)
    if platform not in {"cpu", "gpu", "tpu"}:
        raise ValueError(f"unsupported JAX device {device!r}")
    try:
        devices = jax.devices(platform)
        ordinal = int(index or 0)
        if ordinal < 0:
            raise ValueError("negative device ordinal")
        return devices[ordinal]
    except (RuntimeError, IndexError, ValueError) as exc:
        raise ValueError(f"requested device {device!r} is unavailable") from exc
