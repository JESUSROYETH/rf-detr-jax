# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""Command-line entry points for the JAX RF-DETR port."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def convert() -> None:
    """Convert a PyTorch RF-DETR checkpoint into a Flax parameter ``.npz``.

    Reads the ``.pth`` with PyTorch (required only for this step), maps every key onto the port's
    parameter tree, and writes the result as a dotted-path ``.npz`` that
    :func:`rfdetr_jax.convert.load_params_npz` can read back without PyTorch.
    """
    parser = argparse.ArgumentParser(description="Convert an RF-DETR .pth checkpoint to Flax parameters.")
    parser.add_argument("checkpoint", help="Path to the PyTorch .pth checkpoint.")
    parser.add_argument("--out", required=True, help="Destination .npz path.")
    parser.add_argument(
        "--model-config", help="LWDETRSpec/metadata JSON; validates every parameter and writes a sidecar."
    )
    parser.add_argument("--annotations", help="Optional COCO JSON for keypoint label mapping and category metadata.")
    parser.add_argument(
        "--trust-checkpoint", action="store_true",
        help="Allow Python pickle execution for a trusted legacy checkpoint. Never use for untrusted files.",
    )
    args = parser.parse_args()
    if Path(args.out).suffix != ".npz":
        parser.error("--out must end in .npz")
    outputs = {Path(args.out).resolve()}
    if args.model_config:
        outputs.add(Path(str(args.out) + ".json").resolve())
    inputs = {Path(value).resolve() for value in (args.checkpoint, args.model_config, args.annotations) if value}
    if outputs & inputs:
        parser.error("conversion output and metadata must not overwrite an input file")

    try:
        import torch
    except ImportError:
        parser.error('conversion requires PyTorch; install rfdetr-jax[reference]')

    # Older releases have known weights-only unpickler vulnerabilities (CVE-2026-24747).
    version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if version < (2, 10):
        parser.error("checkpoint conversion requires PyTorch >=2.10.0; upgrade rfdetr-jax[reference]")

    from rfdetr_jax.checkpoints import safe_torch_load
    from rfdetr_jax.convert import convert_state_dict, save_params_npz

    try:
        raw = safe_torch_load(args.checkpoint, trust_checkpoint=args.trust_checkpoint)
    except ValueError as exc:
        parser.error(
            f"safe checkpoint loading failed: {exc}\n"
            "Only for a checkpoint you trust, retry with --trust-checkpoint (allows Python code execution)."
        )
    if not isinstance(raw, dict):
        parser.error("checkpoint must contain a state dict or a 'model' state dict")
    state_dict = raw.get("model", raw)
    if not isinstance(state_dict, dict) or not state_dict:
        parser.error("checkpoint state dict must be a nonempty tensor mapping")
    if not all(isinstance(key, str) and hasattr(value, "detach") for key, value in state_dict.items()):
        parser.error("checkpoint state dict must map parameter names to tensors")
    numpy_state = {key: value.detach().cpu().numpy() for key, value in state_dict.items()}
    if any(not np.isfinite(value).all() for value in numpy_state.values()):
        parser.error("checkpoint contains nonfinite tensors")
    params, missing, unexpected = convert_state_dict(numpy_state, template=None)
    metadata = None
    if args.model_config:
        from rfdetr_jax.configuration import (
            keypoint_category_mapping,
            load_model_spec,
            task_for_spec,
            validate_parameter_tree,
        )
        from rfdetr_jax.datasets.coco import CocoDetection

        spec, original_metadata = load_model_spec(args.model_config)
        validate_parameter_tree(params, spec)
        metadata = dict(original_metadata) if "model_spec" in original_metadata else {}
        metadata.update(model_spec=spec.__dict__, task=task_for_spec(spec))
        if args.annotations:
            dataset = CocoDetection(Path(args.annotations).parent, args.annotations,
                                    keypoints=spec.use_grouppose_keypoints)
            mapping = (keypoint_category_mapping(dataset, spec.num_keypoints_per_class)
                       if spec.use_grouppose_keypoints else {
                           int(key): int(value) for key, value in metadata.get(
                               "category_to_label", {key: key for key in dataset.category_ids}
                           ).items()
                       })
            dataset.set_category_mapping(mapping)
            metadata.update(category_ids=dataset.category_ids, category_names=dataset.category_names,
                            category_to_label=mapping)
    elif args.annotations:
        parser.error("--annotations requires --model-config")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_params_npz(args.out, params)
    if metadata is not None:
        Path(str(args.out) + ".json").write_text(json.dumps(metadata, indent=2))
    print(f"wrote {args.out} ({len(numpy_state)} source tensors, {len(unexpected)} unmapped)")
    del missing


def evaluate() -> None:
    """Score detection, segmentation or keypoint parameters on a COCO split."""
    parser = argparse.ArgumentParser(description="Evaluate JAX RF-DETR parameters with task-specific COCO metrics.")
    parser.add_argument("--params", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument(
        "--model-config", help="Model JSON; defaults to the parameter file's .json sidecar when present."
    )
    parser.add_argument("--task", choices=["detection", "segmentation", "keypoints"])
    parser.add_argument("--keypoint-schema", help="Comma-separated keypoint counts indexed by model label slot.")
    parser.add_argument("--keypoint-oks-sigmas", help="Comma-separated positive OKS sigmas.")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--dec-layers", type=int)
    parser.add_argument("--num-select", type=int)
    parser.add_argument("--eval-backend", choices=["auto", "pycocotools", "hotcoco"], default="auto")
    args = parser.parse_args()

    from dataclasses import replace

    import jax
    import jax.numpy as jnp

    from rfdetr_jax.configuration import (
        default_model_spec,
        keypoint_category_mapping,
        load_model_spec,
        task_for_spec,
        validate_model_spec,
        validate_parameter_tree,
    )
    from rfdetr_jax.convert import load_params_npz
    from rfdetr_jax.datasets.coco import CocoDetection, iterate_batches
    from rfdetr_jax.evaluation.tasks import accumulate_predictions, evaluate_predictions
    from rfdetr_jax.models.lwdetr import LWDETR

    jax.config.update("jax_default_matmul_precision", "highest")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    metadata = {}
    sidecar = Path(str(args.params) + ".json")
    config_path = args.model_config or (str(sidecar) if sidecar.is_file() else None)
    try:
        if config_path:
            spec, metadata = load_model_spec(config_path)
            if args.task and args.task != task_for_spec(spec):
                raise ValueError("--task disagrees with the checkpoint model configuration")
        else:
            schema = tuple(int(value) for value in args.keypoint_schema.split(",")) if args.keypoint_schema else ()
            spec = default_model_spec(args.task or "detection", keypoint_schema=schema)
        changes = {key: getattr(args, key) for key in ("resolution", "num_classes", "dec_layers", "num_select")
                   if getattr(args, key) is not None}
        if args.keypoint_schema:
            changes["num_keypoints_per_class"] = tuple(int(value) for value in args.keypoint_schema.split(","))
        if args.resolution is not None and not config_path:
            changes["positional_encoding_size"] = args.resolution // spec.patch_size
        spec = replace(spec, **changes)
        validate_model_spec(spec)
        params = load_params_npz(args.params)
        validate_parameter_tree(params, spec)
        task = task_for_spec(spec)
        dataset = CocoDetection(args.images, args.annotations, keypoints=task == "keypoints")
        # Metadata inferred from a bare checkpoint is a convention, not the dataset's schema.
        if metadata.get("category_source") != "inferred":
            if "category_ids" in metadata and metadata["category_ids"] != dataset.category_ids:
                raise ValueError("evaluation category ids must match the model metadata")
            if "category_names" in metadata and metadata["category_names"] != dataset.category_names:
                raise ValueError("evaluation category names must match the model metadata")
        if task == "keypoints":
            mapping = keypoint_category_mapping(dataset, spec.num_keypoints_per_class)
            if "category_to_label" in metadata:
                saved = {int(key): int(value) for key, value in metadata["category_to_label"].items()}
                if saved != mapping:
                    raise ValueError("evaluation category mapping must match the model metadata")
            dataset.set_category_mapping(mapping)
        elif "category_to_label" in metadata and metadata.get("category_source") != "inferred":
            dataset.set_category_mapping({int(key): int(value) for key, value in metadata["category_to_label"].items()})
        # An inferred sidecar keeps the CLI contract: raw category ids are the model labels.
        if max(dataset.category_to_label.values()) >= spec.detector_classes:
            raise ValueError("--num-classes must cover the largest model label")
        sigmas = ([float(value) for value in args.keypoint_oks_sigmas.split(",")]
                  if args.keypoint_oks_sigmas else metadata.get("keypoint_oks_sigmas") or None)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    model = LWDETR(spec=spec)

    @jax.jit
    def forward(parameters, pixels):
        outputs = model.apply({"params": parameters}, pixels, None, train=False)
        return {key: value for key, value in outputs.items() if key.startswith("pred_")}

    records = []
    for batch in iterate_batches(dataset, spec.resolution, args.batch_size, 1,
                                 train=False, drop_last=False, pad_last=True):
        outputs = forward(params, jnp.asarray(batch["pixel_values"]))
        valid = int(batch["num_valid"])
        records.extend(accumulate_predictions(
            batch["image_ids"][:valid], {key: np.asarray(value)[:valid] for key, value in outputs.items()},
            batch["orig_sizes"][:valid], dataset.category_ids, task=task, num_select=spec.num_select,
            num_keypoints_per_class=spec.num_keypoints_per_class,
            label_to_category={label: category for category, label in dataset.category_to_label.items()},
        ))
    metrics = evaluate_predictions(args.annotations, records, task=task, max_dets=spec.num_select,
                                   backend=args.eval_backend, keypoint_oks_sigmas=sigmas)
    print(json.dumps(metrics, indent=2))


def train() -> None:
    """Train a detector using the packaged training loop."""
    from rfdetr_jax.training.run import main

    main()
