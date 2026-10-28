# ------------------------------------------------------------------------
# RF-DETR JAX — modified from RF-DETR, Copyright (c) 2025 Roboflow.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""Validated upstream checkpoint loading, with explicit random initialization only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import tempfile
import types
import urllib.request
import warnings
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from rfdetr_jax.configuration import load_model_spec, task_for_spec, validate_model_spec, validate_parameter_tree
from rfdetr_jax.convert import convert_state_dict, flatten_params, load_params_npz, unflatten_params
from rfdetr_jax.models.lwdetr import LWDETRSpec
from rfdetr_jax.variants import VARIANT_DEFAULTS, spec_from_config, variant_config

OFFICIAL_WEIGHTS = {'rf-detr-base.pth': ('https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth',
                      'b4d3ce46099eaed50626ede388caf979'),
 'rf-detr-base-o365.pth': ('https://storage.googleapis.com/rfdetr/top-secret-1234/lwdetr_dinov2_small_o365_checkpoint.pth',
                           'd93f4921ccbb0f0a2e4364bed290892b'),
 'rf-detr-base-2.pth': ('https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth',
                        '462f4d9df407ddc1812f42614040e913'),
 'rf-detr-large.pth': ('https://storage.googleapis.com/rfdetr/rf-detr-large.pth',
                       '992c8e862aa733a7bb2777e45d49f1a0'),
 'rf-detr-large-2026.pth': ('https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth',
                            '5cb72153541cbcb9aa6efa26222acc75'),
 'rf-detr-nano.pth': ('https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth',
                      'fb6504cce7fbdc783f7a46991f07639f'),
 'rf-detr-small.pth': ('https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth',
                       'fb37061c1af7bace359c91b723a8d5c1'),
 'rf-detr-medium.pth': ('https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth',
                        '7223f764a87b863f02eb8d52bf0ce2ee'),
 'rf-detr-keypoint-preview-xlarge.pth': ('https://storage.googleapis.com/rfdetr/rf-detr-keypoint-preview-xlarge.pth',
                                         '6de511943ee85a547d4c5cb527daf0eb'),
 'rf-detr-seg-preview.pt': ('https://storage.googleapis.com/rfdetr/rf-detr-seg-preview.pt',
                            'e35820c28fb86080558123e47a5e49ca'),
 'rf-detr-seg-nano.pt': ('https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth',
                         '9995497791d0ff1664a1d9ddee9cfd20'),
 'rf-detr-seg-small.pt': ('https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth',
                          '0a2a3006381d0c42853907e700eadd08'),
 'rf-detr-seg-medium.pt': ('https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth',
                           'a49af1562c3719227ad43d0ca53b4c7a'),
 'rf-detr-seg-large.pt': ('https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth',
                          '275f7b094909544ed2841c94a677d07e'),
 'rf-detr-seg-xlarge.pt': ('https://storage.googleapis.com/rfdetr/rf-detr-seg-xl-ft.pth',
                           '3693b35d0eea86ebb3e0444f4a611fba'),
 'rf-detr-seg-xxlarge.pt': ('https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth',
                            '040bc3412af840fa8a47e0ff69b552ba')}

COCO_CLASSES = {1: 'person',
 2: 'bicycle',
 3: 'car',
 4: 'motorcycle',
 5: 'airplane',
 6: 'bus',
 7: 'train',
 8: 'truck',
 9: 'boat',
 10: 'traffic light',
 11: 'fire hydrant',
 13: 'stop sign',
 14: 'parking meter',
 15: 'bench',
 16: 'bird',
 17: 'cat',
 18: 'dog',
 19: 'horse',
 20: 'sheep',
 21: 'cow',
 22: 'elephant',
 23: 'bear',
 24: 'zebra',
 25: 'giraffe',
 27: 'backpack',
 28: 'umbrella',
 31: 'handbag',
 32: 'tie',
 33: 'suitcase',
 34: 'frisbee',
 35: 'skis',
 36: 'snowboard',
 37: 'sports ball',
 38: 'kite',
 39: 'baseball bat',
 40: 'baseball glove',
 41: 'skateboard',
 42: 'surfboard',
 43: 'tennis racket',
 44: 'bottle',
 46: 'wine glass',
 47: 'cup',
 48: 'fork',
 49: 'knife',
 50: 'spoon',
 51: 'bowl',
 52: 'banana',
 53: 'apple',
 54: 'sandwich',
 55: 'orange',
 56: 'broccoli',
 57: 'carrot',
 58: 'hot dog',
 59: 'pizza',
 60: 'donut',
 61: 'cake',
 62: 'chair',
 63: 'couch',
 64: 'potted plant',
 65: 'bed',
 67: 'dining table',
 70: 'toilet',
 72: 'tv',
 73: 'laptop',
 74: 'mouse',
 75: 'remote',
 76: 'keyboard',
 77: 'cell phone',
 78: 'microwave',
 79: 'oven',
 80: 'toaster',
 81: 'sink',
 82: 'refrigerator',
 84: 'book',
 85: 'clock',
 86: 'vase',
 87: 'scissors',
 88: 'teddy bear',
 89: 'hair drier',
 90: 'toothbrush'}


def get_model_cache_dir() -> Path:
    return Path(os.environ.get("RF_HOME") or os.environ.get("ROBOFLOW_HOME") or "~/.roboflow/models").expanduser()


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest()


def resolve_weights(value: str | Path) -> Path:
    """Resolve explicit paths or fetch a registered filename atomically with integrity checking."""
    path = Path(value).expanduser()
    if path.is_file():
        return path
    if str(path.parent) != "." or path.name not in OFFICIAL_WEIGHTS:
        raise FileNotFoundError(f"checkpoint not found: {path}")
    url, expected_md5 = OFFICIAL_WEIGHTS[path.name]
    destination = get_model_cache_dir() / path.name
    if destination.is_file():
        if _digest(destination) != expected_md5:
            raise ValueError(f"cached checkpoint checksum mismatch: {destination}; remove it to download again")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        if _digest(temporary) != expected_md5:
            raise ValueError(f"download checksum mismatch for {path.name}")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def safe_torch_load(path, *, trust_checkpoint=False):
    """Load tensors with scoped inert namespace support; never automatically enable pickle."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("loading .pth/.pt/.ckpt requires PyTorch >=2.10.0") from exc
    if tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 10):
        raise RuntimeError("checkpoint loading requires PyTorch >=2.10.0 for safe deserialization")
    try:
        with torch.serialization.safe_globals([argparse.Namespace, types.SimpleNamespace]):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        if not trust_checkpoint:
            raise ValueError("safe checkpoint loading failed; trust_checkpoint=True explicitly permits pickle") from exc
        warnings.warn("trust_checkpoint=True permits arbitrary Python code in this checkpoint", stacklevel=2)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a model state dictionary")
    if "model" not in checkpoint and "state_dict" in checkpoint:
        state = {key.removeprefix("model.").removeprefix("_orig_mod."): value
                 for key, value in checkpoint["state_dict"].items() if key.startswith("model.")}
        if not state:
            raise ValueError("Lightning checkpoint has no model.* parameter keys")
        checkpoint = {**checkpoint, "model": state}
    return checkpoint


def _mapping(value):
    return value if isinstance(value, dict) else vars(value) if hasattr(value, "__dict__") else {}


_BEHAVIOR_OPTIONS = ("amp", "freeze_encoder", "fused_optimizer", "ia_bce_loss", "postprocess_trace_alpha")


def _saved_behavior_options(config, metadata):
    """Restore portable supported behavior, excluding source paths and devices."""
    from rfdetr_jax.configuration import boolean_option

    values = {**metadata.get("model_options", {})}
    if "postprocess_trace_alpha" in metadata:
        values["postprocess_trace_alpha"] = metadata["postprocess_trace_alpha"]
    values.update(config)
    return {key: values[key] if key == "postprocess_trace_alpha" else boolean_option(key, values[key])
            for key in _BEHAVIOR_OPTIONS if key in values}


def inspect_checkpoint(path, *, trust_checkpoint=False) -> tuple[str, dict]:
    """Return (public variant name, constructor options) for upstream or JAX checkpoints."""
    path = resolve_weights(path)
    if path.suffix == ".npz":
        spec, metadata = load_model_spec(str(path) + ".json")
        options = asdict(spec)
        options.update(_saved_behavior_options({}, metadata))
        saved_name = metadata.get("model_name")
        reference = metadata.get("pretrain_weights", path.name)
    else:
        checkpoint = safe_torch_load(path, trust_checkpoint=trust_checkpoint)
        args = _mapping(checkpoint.get("args", checkpoint.get("hyper_parameters", {})))
        saved = _mapping(checkpoint.get("model_config", args.get("model_config", {})))
        values = saved or args
        options = {key: value for key, value in values.items() if key in {f.name for f in fields(LWDETRSpec)}}
        options.update(_saved_behavior_options(values, checkpoint.get("metadata", {})))
        saved_name = checkpoint.get("model_name") or values.get("model_name", args.get("model_name"))
        reference = values.get("pretrain_weights") or path.name
        state = checkpoint.get("model", checkpoint)
        if "class_embed.weight" in state:
            options["num_classes"] = int(state["class_embed.weight"].shape[0]) - 1
        mask = state.get("_kp_active_mask")
        if mask is not None and mask.ndim == 2 and mask.numel() and mask.any():
            options["num_keypoints_per_class"] = mask.sum(dim=1).tolist()
    if saved_name in VARIANT_DEFAULTS:
        name = saved_name
    else:
        filename = Path(str(reference)).name
        if filename.lower() in {"none", "null", ""}:
            filename = path.name
        matches = [name for name, defaults in VARIANT_DEFAULTS.items() if defaults["pretrain_weights"] == filename]
        if not matches:
            # Sidecars from the lower-level API may not carry a named variant. Match geometry exactly.
            keys = ("encoder", "patch_size", "num_windows", "dec_layers", "resolution", "segmentation_head",
                    "use_grouppose_keypoints", "hidden_dim")
            matches = [name for name in VARIANT_DEFAULTS
                       if all(getattr(spec_from_config(variant_config(name)), key) == options.get(
                           key, getattr(LWDETRSpec(), key)) for key in keys)]
        if len(matches) != 1:
            if path.suffix == ".npz":
                # A complete native sidecar is authoritative even for a custom low-level architecture.
                matches = [{"detection": "RFDETRNano", "segmentation": "RFDETRSegNano",
                            "keypoints": "RFDETRKeypointPreview"}[task_for_spec(spec)]]
            else:
                raise ValueError("cannot identify checkpoint variant; use a named class with pretrain_weights=path")
        name = matches[0]
    options["pretrain_weights"] = str(path)
    return name, options


def _resize_rows(value, count):
    if value.shape[0] == 0:
        raise ValueError("cannot resize an empty checkpoint parameter")
    return np.tile(value, (int(np.ceil(count / value.shape[0])),) + (1,) * (value.ndim - 1))[:count]


def _resize_position_embedding(value, pe_size):
    """Use the upstream checkpoint interpolation kernel on a copied PE tensor."""
    if value.shape[1] == pe_size**2 + 1:
        return value
    import torch
    import torch.nn.functional as functional

    value = np.asarray(value)
    side = int(np.sqrt(value.shape[1] - 1))
    if side * side != value.shape[1] - 1:
        raise ValueError("checkpoint position embedding grid is not square")
    patches = torch.from_numpy(value[:, 1:].copy()).reshape(1, side, side, -1).permute(0, 3, 1, 2)
    resized = functional.interpolate(patches.float(), size=(pe_size,) * 2,
                                     mode="bicubic", align_corners=False, antialias=True)
    return np.concatenate([value[:, :1], resized.permute(0, 2, 3, 1).reshape(
        1, pe_size**2, -1).numpy().astype(value.dtype)], axis=1)


def resize_position_embeddings(params, pe_size):
    """Copy a parameter tree and adapt only PE grids for temporary evaluation."""
    flat = flatten_params(params)
    for key, value in flat.items():
        if ".".join(key).endswith("embeddings.position_embeddings"):
            flat[key] = _resize_position_embedding(value, pe_size)
    return unflatten_params(flat)


def _adapt_parameters(params, spec, *, source_queries, source_groups):
    flat = flatten_params(params)
    for key, value in list(flat.items()):
        name = ".".join(key)
        if key[0] == "class_embed" or re.search(r"enc_out_class_embed_\d+\.(weight|bias)$", name):
            flat[key] = _resize_rows(value, spec.detector_classes)
        if key in {("query_feat",), ("refpoint_embed",)}:
            if value.shape[0] != source_queries * source_groups:
                raise ValueError("checkpoint query packing does not match its query/group metadata")
            if spec.num_queries > source_queries or spec.group_detr > source_groups:
                raise ValueError("pretrained query/group expansion is unsupported; use pretrain_weights=None")
            flat[key] = value.reshape(source_groups, source_queries, -1)[
                :spec.group_detr, :spec.num_queries].reshape(spec.group_detr * spec.num_queries, -1)
        group = re.search(r"(?:enc_out_(?:class_embed|bbox_embed|keypoint_embed)|enc_output(?:_norm)?)_(\d+)", name)
        if group and int(group[1]) >= spec.group_detr:
            del flat[key]
        if name.endswith(("keypoint_query_initializer.queries", "keypoint_query_initializer_enc.queries",
                          "decoder.keypoint_pos_embed")):
            flat[key] = _resize_rows(value, sum(spec.num_keypoints_per_class))
        if name.endswith("embeddings.position_embeddings"):
            flat[key] = _resize_position_embedding(value, spec.positional_encoding_size)
    if spec.dual_projector and not any(key[:2] == ("backbone_0", "cross_attn_projector") for key in flat):
        flat.update({("backbone_0", "cross_attn_projector", *key[2:]): value.copy()
                     for key, value in list(flat.items()) if key[:2] == ("backbone_0", "projector")})
    return unflatten_params(flat)


def load_variant(variant_name: str, *, trust_checkpoint=False, seed=0, **kwargs):
    """Return (spec, parameters, metadata, config), with no silent random fallback."""
    config = variant_config(variant_name, **kwargs)
    metadata = {}
    source_options = {}
    raw_state = None
    if config.pretrain_weights is None:
        import jax
        import jax.numpy as jnp

        from rfdetr_jax.models.lwdetr import LWDETR

        spec = spec_from_config(config)
        validate_model_spec(spec)
        side = spec.patch_size * spec.num_windows * 2
        params = LWDETR(spec).init(jax.random.key(seed), jnp.zeros((1, 3, side, side)), None, train=True)["params"]
    else:
        path = resolve_weights(config.pretrain_weights)
        config.pretrain_weights = str(path)
        if path.suffix == ".npz":
            source_spec, metadata = load_model_spec(str(path) + ".json")
            source_options = asdict(source_spec)
            options = {**source_options, **kwargs, "pretrain_weights": str(path)}
            config = variant_config(variant_name, **options)
            config.model_fields_set = set(kwargs)
            params = load_params_npz(path)
        else:
            checkpoint = safe_torch_load(path, trust_checkpoint=trust_checkpoint)
            raw_state = checkpoint.get("model", checkpoint)
            if not isinstance(raw_state, dict) or not raw_state:
                raise ValueError("checkpoint model state must be a nonempty tensor mapping")
            if not all(isinstance(key, str) and hasattr(value, "detach") for key, value in raw_state.items()):
                raise ValueError("checkpoint state must map parameter names to tensors")
            state = {key: value.detach().cpu().numpy() for key, value in raw_state.items()
                     if not key.startswith("keypoint_head.keypoint_proj.")}
            if any(not np.isfinite(value).all() for value in state.values()):
                raise ValueError("checkpoint contains nonfinite tensors")
            source_options = _mapping(checkpoint.get("args", checkpoint.get("hyper_parameters", {})))
            saved_model_config = _mapping(checkpoint.get("model_config", source_options.get("model_config", {})))
            source_options = {**source_options, **saved_model_config}
            metadata.update(checkpoint.get("metadata", {}))
            names = source_options.get("class_names") or checkpoint.get("class_names")
            if names:
                metadata["class_names"] = list(names)
            if config.use_grouppose_keypoints and "num_keypoints_per_class" not in kwargs:
                mask = state.get("_kp_active_mask")
                if mask is not None and mask.ndim == 2 and mask.any():
                    config.num_keypoints_per_class = mask.sum(axis=1).astype(int).tolist()
            params, _, _ = convert_state_dict(state)
        if "num_classes" not in kwargs:
            config.num_classes = int(params["class_embed"]["weight"].shape[0]) - 1
        for key, value in _saved_behavior_options(source_options, metadata).items():
            if key not in kwargs:
                setattr(config, key, value)
        spec = spec_from_config(config)
        validate_model_spec(spec)
        defaults = VARIANT_DEFAULTS[variant_name]
        source_queries = int(source_options.get("num_queries", defaults["num_queries"]))
        source_groups = int(source_options.get("group_detr", 13))
        params = _adapt_parameters(params, spec,
                                   source_queries=source_queries, source_groups=source_groups)
        validate_parameter_tree(params, spec)
        metadata["pretrain_weights"] = str(path)
    metadata.update(model_spec=asdict(spec), model_name=variant_name, task=task_for_spec(spec),
                    postprocess_trace_alpha=config.postprocess_trace_alpha,
                    model_options={key: getattr(config, key) for key in _BEHAVIOR_OPTIONS})
    if "num_classes" in kwargs and any(int(label) >= spec.detector_classes
                                       for label in metadata.get("category_to_label", {}).values()):
        for key in ("category_ids", "category_names", "category_to_label", "class_names"):
            metadata.pop(key, None)
    if "category_to_label" not in metadata:
        # No dataset schema travelled with these weights: the mapping below is a convention,
        # and evaluate() re-derives it from the evaluation dataset like train() does.
        metadata["category_source"] = "inferred"
        if spec.use_grouppose_keypoints:
            active = [slot for slot, count in enumerate(spec.num_keypoints_per_class) if count]
            person_schema = spec.num_keypoints_per_class in {(17,), (0, 17)}
            metadata["category_to_label"] = {1: active[0]} if person_schema else {slot: slot for slot in active}
            metadata.setdefault("class_names", ["person"] if person_schema
                                else [f"class_{index}" for index in range(len(active))])
        elif spec.num_classes == 90:
            metadata["category_to_label"] = {key: key for key in COCO_CLASSES}
            metadata.setdefault("class_names", list(COCO_CLASSES.values()))
        else:
            metadata["category_to_label"] = {key: key for key in range(spec.detector_classes)}
            metadata.setdefault("class_names", [f"class_{key}" for key in range(spec.num_classes)])
    mapping = {int(key): int(value) for key, value in metadata["category_to_label"].items()}
    metadata.setdefault("category_ids", list(mapping))
    if "category_names" not in metadata:
        names = metadata.get("class_names", [])
        metadata["category_names"] = [names[index] if index < len(names) else f"class_{key}"
                                      for index, key in enumerate(mapping)]
    return spec, params, metadata, config


def reset_keypoint_gaussian_parameters(params):
    """Copy and zero only the pretrained keypoint precision-Cholesky rows.

    The native decoder and encoder heads each use a three-layer MLP. Rows
    4:7 of its final linear encode the Gaussian precision; the other outputs
    and all earlier layers must retain their pretrained values.
    """
    flat = flatten_params(params)
    for key, value in flat.items():
        decoder = key[:-1] == ("keypoint_embed", "layers_2")
        encoder = (len(key) == 4 and key[0] == "transformer"
                   and re.fullmatch(r"enc_out_keypoint_embed_\d+", key[1])
                   and key[2] == "layers_2")
        if (decoder or encoder) and key[-1] in {"weight", "bias"} and value.shape[0] > 6:
            reset = np.array(value, copy=True)
            reset[4:7] = 0
            flat[key] = reset
    return unflatten_params(flat)


def load_pretrained_training_parameters(variant_name, model_config, aligned_spec: LWDETRSpec):
    """Reload the configured source once into the effective training geometry.

    Dataset/schema and precision decisions are already made by the caller.
    Pass them explicitly to the safe loader, but do not persist that temporary
    config's explicit fields or replace the current dataset metadata.
    """
    source = model_config.pretrain_weights
    if source is None:
        raise ValueError("fresh pretrained training requires a configured checkpoint")
    options = {**model_config.model_dump(), **asdict(aligned_spec), "pretrain_weights": source}
    loaded_spec, params, _, _ = load_variant(variant_name, trust_checkpoint=False, **options)
    if loaded_spec != aligned_spec:
        changed = [field.name for field in fields(LWDETRSpec)
                   if getattr(loaded_spec, field.name) != getattr(aligned_spec, field.name)]
        raise ValueError(f"training checkpoint load changed the aligned model specification: {changed}")
    if aligned_spec.use_grouppose_keypoints:
        params = reset_keypoint_gaussian_parameters(params)
    return params


def adapt_parameters(params, old_spec: LWDETRSpec, new_spec: LWDETRSpec, seed=42):
    """Resize only upstream-supported heads/query rows and verify every other parameter strictly.

    Head growth tiles pretrained rows, matching upstream; it does not initialize random weights.
    The seed argument is accepted for callers that share initialization APIs.
    """
    del seed
    allowed = {"num_classes", "num_keypoints_per_class", "num_queries", "group_detr", "num_select",
               "resolution", "positional_encoding_size", "gradient_checkpointing", "compute_dtype",
               "head_dtype", "deform_sampler"}
    changed = {field.name for field in fields(LWDETRSpec)
               if getattr(old_spec, field.name) != getattr(new_spec, field.name)}
    if changed - allowed:
        raise ValueError(f"unsupported pretrained architecture changes: {sorted(changed - allowed)}")
    validate_model_spec(new_spec)
    result = _adapt_parameters(params, new_spec, source_queries=old_spec.num_queries,
                               source_groups=old_spec.group_detr)
    validate_parameter_tree(result, new_spec)
    return result


def torch_state_dict(params, spec: LWDETRSpec):
    """Reverse the parameter rename and restore deterministic upstream state buffers."""
    import torch

    state = {}
    for path, value in flatten_params(params).items():
        name = re.sub(r"_(\d+)(?=[_.]|$)", r".\1", ".".join(path))
        name = name.replace("backbone.0.encoder.", "backbone.0.encoder.encoder.", 1)
        if name in {"query_feat", "refpoint_embed"}:
            name += ".weight"
        state[name] = torch.from_numpy(np.array(value, copy=True))
        if name.endswith("running_mean"):
            state[name.removesuffix("running_mean") + "num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
    schema = spec.num_keypoints_per_class
    state["_kp_active_mask"] = torch.zeros((len(schema), max(schema, default=0)), dtype=torch.bool)
    for slot, count in enumerate(schema):
        state["_kp_active_mask"][slot, :count] = True
    if spec.use_grouppose_keypoints:
        labels = np.repeat(np.arange(len(schema)), schema)
        mask = torch.zeros((1 + len(labels), 1 + len(labels)), dtype=torch.bool)
        mask[1:, 1:] = torch.from_numpy(labels[:, None] != labels[None, :])
        state["transformer.decoder.keypoint_class_mask"] = mask
    return state


def save_checkpoint(path, params, spec: LWDETRSpec, metadata: dict, model_config=None):
    """Save native NPZ+JSON or upstream Torch state without model source paths.

    Training metadata and embedded resume state may retain configured dataset/output
    locations. Use export_for_roboflow for a sanitized distribution bundle.
    """
    path = Path(path)
    if path.suffix not in {".npz", ".pt", ".pth", ".ckpt"}:
        raise ValueError("checkpoint destination must end in .npz, .pt, .pth, or .ckpt")
    validate_parameter_tree(params, spec)
    payload = {**metadata, "model_spec": asdict(spec), "task": task_for_spec(spec)}
    config = model_config.model_dump() if model_config is not None else asdict(spec)
    config.update(asdict(spec))
    payload["postprocess_trace_alpha"] = config.get("postprocess_trace_alpha",
                                                   payload.get("postprocess_trace_alpha", .2))
    payload["model_options"] = _saved_behavior_options(config, payload)
    name = config.get("model_name", payload.get("model_name"))
    payload["model_name"] = name
    payload.pop("pretrain_weights", None)
    config.update(pretrain_weights=None, device="cpu")
    serialized_metadata = json.dumps(payload, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            if path.suffix == ".npz":
                np.savez(stream, **{".".join(key): np.asarray(value) for key, value in flatten_params(params).items()})
            else:
                import torch

                args = {**config, "class_names": payload.get("class_names", payload.get("category_names", [])),
                        "dataset_dir": None, "output_dir": "output", "resume": ""}
                torch.save({"model": torch_state_dict(params, spec), "model_config": config, "model_name": name,
                            "args": args, "class_names": args["class_names"], "metadata": payload}, stream)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if path.suffix == ".npz":
        sidecar = Path(str(path) + ".json")
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=sidecar.name + ".",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(serialized_metadata)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(sidecar)


def embed_native_resume(path, native_checkpoint):
    """Attach native optimizer/RNG state as inert bytes to a safely loaded Torch checkpoint."""
    import torch

    path, native_checkpoint = Path(path), Path(native_checkpoint)
    if path.resolve() == native_checkpoint.resolve():
        raise ValueError("native resume source must differ from the Torch checkpoint")
    checkpoint = safe_torch_load(path)
    checkpoint["rfdetr_jax_training_state"] = torch.from_numpy(
        np.frombuffer(native_checkpoint.read_bytes(), dtype=np.uint8).copy())
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(checkpoint, stream)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
