# Modified JAX adaptation of RF-DETR export contracts.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
"""Optional export-only bridge to the pinned upstream Torch exporters.

Training and prediction remain JAX. This module reconstructs a separate Torch
model from the current JAX tensors; it never downloads weights or uploads files.
Format availability and conversion limits belong to the installed upstream
exporter. A successful conversion alone does not establish deployment parity.
"""
from __future__ import annotations

import inspect
from dataclasses import asdict, fields, replace
from types import SimpleNamespace


def _upstream_api():
    try:
        from rfdetr.detr import RFDETR
    except ImportError as exc:
        raise ImportError(
            "Export requires the optional upstream RF-DETR package and the requested format's extras. "
            "Install rfdetr-jax[export] for the tested rfdetr[onnx]==1.11.2 bridge "
            "and the corresponding upstream format extra for other exporters. "
            "Only static ONNX and the local Roboflow bundle have cross-runtime tests here."
        ) from exc
    return RFDETR


def build_export_model(model):
    """Reconstruct an independent upstream model and strictly load every parameter/buffer."""
    import torch
    from rfdetr._namespace import _namespace_from_configs
    from rfdetr.config import ModelConfig, TrainConfig
    from rfdetr.models._defaults import MODEL_DEFAULTS
    from rfdetr.models.lwdetr import build_model

    from rfdetr_jax.checkpoints import torch_state_dict

    upstream_api = _upstream_api()
    values = asdict(model.spec)
    config_values = {key: value for key, value in values.items() if key in ModelConfig.model_fields}
    config_values.update(device="cpu", pretrain_weights="jax-export-local", backbone_lora=False, freeze_encoder=False)
    config_values["postprocess_trace_alpha"] = getattr(model.model_config, "postprocess_trace_alpha", .2)
    config = ModelConfig(**config_values)
    default_names = {field.name for field in fields(MODEL_DEFAULTS)}
    defaults = replace(MODEL_DEFAULTS, **{key: value for key, value in values.items() if key in default_names})
    defaults = replace(defaults, force_no_pretrain=True, pretrained_encoder=None)
    args = _namespace_from_configs(config, TrainConfig(dataset_dir=".", output_dir="output"), defaults)
    args.class_names = list(model.class_names)
    # Construction is CPU-only and cannot request pretrained DINOv2 weights.
    with torch.device("cpu"):
        torch_model = build_model(args)
    # Upstream registers all DINO blocks even though its forward stops after the
    # last requested stage. Flax never instantiates those unreachable parameters.
    # Remove exactly the unreachable tail before the otherwise strict state load.
    encoder = torch_model.backbone[0].encoder.encoder.encoder
    encoder.layer = encoder.layer[: min(len(encoder.layer), model.spec.out_feature_indexes[-1] + 1)]
    state = torch_state_dict(model.params, model.spec)
    torch_model.load_state_dict(state, strict=True)
    torch_model.eval()
    facade = object.__new__(upstream_api)
    facade.model_config = config
    facade.model = SimpleNamespace(model=torch_model, args=args, resolution=model.spec.resolution,
                                   device=torch.device("cpu"), class_names=list(model.class_names))
    facade.size = getattr(model, "size", None)
    facade._optimized_inplace = False
    return facade


def export(model, **kwargs):
    """Delegate all supported upstream export arguments without discarding options.

    The format implementation returns its actual artifact path and owns shape,
    backend, quantization and dependency validation. Current JAX parameters are
    copied into a separate CPU Torch model, leaving the live model untouched.
    """
    api = _upstream_api()
    # Reject unknown arguments before constructing another model or creating files.
    inspect.signature(api.export).bind(None, **kwargs)
    reconstructed = build_export_model(model)
    return reconstructed.export(**kwargs)


def export_for_roboflow(model, output_dir):
    """Write the upstream weights.pt + class_names.txt bundle locally; never upload."""
    _upstream_api()
    reconstructed = build_export_model(model)
    return reconstructed.export_for_roboflow(output_dir)
