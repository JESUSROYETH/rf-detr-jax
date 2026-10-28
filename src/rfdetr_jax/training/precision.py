# ------------------------------------------------------------------------
# RF-DETR JAX — modified precision policy of RF-DETR, Copyright (c) 2025 Roboflow.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""Resolve public RF-DETR precision requests without initializing a Torch device."""
from __future__ import annotations

import warnings


def normalize_amp_dtype(value):
    if value not in (None, "auto", "bf16", "fp16", "fp8"):
        warnings.warn(f"Unknown amp_dtype={value!r}; falling back to 'auto'", UserWarning, stacklevel=2)
        return "auto"
    return value


def _cuda_capability(device):
    value = getattr(device, "compute_capability", None)
    if value is None:
        from jaxlib import gpu_triton
        value = gpu_triton.get_compute_capability(device.local_hardware_id)
    if isinstance(value, str) and "." in value:
        major, minor = value.split(".", 1)
        return int(major), int(minor)
    if isinstance(value, tuple | list):
        return int(value[0]), int(value[1])
    numeric = int(value)
    return numeric // 10, numeric % 10


def resolve_training_precision(requested, legacy_amp, device):
    """Mirror 1.11.2's request precedence and selected-device precision.

    CPU always uses float32. CUDA auto uses native BF16 on Ampere or later;
    explicit BF16 can use emulation. FP16 requires dynamic loss scaling.
    """
    requested = normalize_amp_dtype(requested)
    effective = requested
    if requested == "auto" and not legacy_amp:
        warnings.warn("ModelConfig.amp is deprecated; pass amp_dtype=None instead", FutureWarning, stacklevel=2)
        effective = None
    if effective == "fp8":
        raise NotImplementedError("FP8 requires a Transformer Engine backend; JAX FP8 training is not implemented")
    if effective is None or device.platform == "cpu":
        dtype = "float32"
    elif device.platform == "tpu":
        if effective not in ("auto", "bf16"):
            raise NotImplementedError("TPU training supports amp_dtype=None, auto or bf16")
        dtype = "bfloat16"
    elif device.platform == "gpu":
        if effective == "fp16":
            dtype = "float16"
        else:
            native = _cuda_capability(device) >= (8, 0)
            if effective == "bf16" and not native:
                warnings.warn("BF16 on a pre-Ampere GPU uses emulation; fp16 is faster", UserWarning, stacklevel=2)
            dtype = "bfloat16" if native or effective == "bf16" else "float16"
    else:
        raise ValueError(f"unsupported precision device {device.platform!r}")
    return dict(requested=requested, effective=effective, compute_dtype=dtype,
                head_dtype="float32" if dtype == "float32" else "activation",
                pixel_dtype="float32", loss_scaling=dtype == "float16")
