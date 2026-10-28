# ------------------------------------------------------------------------
# RF-DETR JAX — adapted from RF-DETR, Copyright (c) 2025 Roboflow.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""Public training/evaluation adapters; the optimizer and model remain JAX."""

from __future__ import annotations

import json
import shutil
import tempfile
import warnings
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rfdetr_jax.configuration import (
    UnsupportedOptionError,
    keypoint_category_counts,
    keypoint_category_mapping,
    task_for_spec,
)
from rfdetr_jax.configuration import boolean_option as _boolean_option
from rfdetr_jax.datasets.coco import CocoDetection
from rfdetr_jax.variants import ModelConfig

_DEFAULTS = dict(
    dataset_dir=None, dataset_file="roboflow", output_dir="output", epochs=100, batch_size=4,
    grad_accum_steps=1, multi_scale=True, expanded_scales=True, scale_jitter=True,
    eval_batch_size=None, eval_max_dets=500, eval_masks_head_resolution=False,
    num_workers=2, lr=1e-4, lr_encoder=1.5e-4,
    weight_decay=1e-4, lr_vit_layer_decay=.8, lr_component_decay=.7, clip_max_norm=.1,
    lr_scheduler="step", lr_drop=100, lr_min_factor=0., warmup_epochs=0., seed=None,
    use_ema=True, ema_decay=.993, ema_tau=100, resume=None, resolution=None, device=None, amp_dtype="auto",
    checkpoint_interval=10, eval_interval=1, class_names=None, notes=None,
    keypoint_flip_pairs=(), keypoint_oks_sigmas=None, tensorboard=True,
    early_stopping=False, early_stopping_patience=10, early_stopping_min_delta=.001,
    early_stopping_use_ema=False, eval_base_model=False, eval_ema_only=False,
    best_model_metric="map", skip_best_epochs=0, smooth_alpha=0.,
    cls_loss_coef=None, bbox_loss_coef=5., giou_loss_coef=2., mask_point_sample_ratio=16,
    mask_ce_loss_coef=5., mask_dice_loss_coef=5., keypoint_l1_loss_coef=1.,
    keypoint_findable_loss_coef=1., keypoint_visible_loss_coef=1., keypoint_nll_loss_coef=1.,
    # Backend extensions are explicit; they are not advertised as upstream options.
    max_targets=None, matcher="hungarian", no_precompile=False, no_compile_cache=False,
)


def _normalize_boolean_options(values):
    return {name: (_boolean_option(name, value) if type(_DEFAULTS.get(name)) is bool and name != "multi_scale"
                   else value) for name, value in values.items()}


def _multi_scale_enabled(value):
    """Normalize the published enum without treating the string 'off' as true."""
    value = getattr(value, "value", value)
    if type(value) is bool:
        return value
    if value == "off":
        return False
    if value == "per-batch":
        return True
    if value == "per-sample":
        raise NotImplementedError("per-sample multi_scale is not implemented; use per-batch or off")
    raise ValueError("multi_scale must be a boolean, 'off', or 'per-batch'")


def training_config(*, task="detection", **kwargs):
    """Validate supported options without accepting a misspelled or ignored setting."""
    unknown = kwargs.keys() - _DEFAULTS.keys()
    if unknown:
        raise UnsupportedOptionError(f"training options not implemented yet: {', '.join(sorted(unknown))}")
    defaults = {**_DEFAULTS}
    if task == "keypoints":
        defaults.update(smooth_alpha=.5, skip_best_epochs=10, cls_loss_coef=2.)
    values = _normalize_boolean_options({**defaults, **kwargs})
    from rfdetr_jax.training.precision import normalize_amp_dtype
    values["amp_dtype"] = normalize_amp_dtype(values["amp_dtype"])
    values["multi_scale"] = _multi_scale_enabled(values["multi_scale"])
    if "eval_ema_only" in kwargs:
        if "eval_base_model" not in kwargs:
            values["eval_base_model"] = not values["eval_ema_only"]
        warnings.warn("eval_ema_only is deprecated; use eval_base_model instead", FutureWarning, stacklevel=2)
    if values["eval_ema_only"] and (not values["use_ema"] or values["eval_base_model"]):
        raise ValueError("eval_ema_only requires use_ema=True and eval_base_model=False")
    if not np.isfinite(values["smooth_alpha"]) or not 0 <= values["smooth_alpha"] < 1:
        raise ValueError("smooth_alpha must be finite and in [0, 1)")
    for name in ("skip_best_epochs", "ema_tau"):
        if type(values[name]) is not int or values[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if values["eval_masks_head_resolution"]:
        raise NotImplementedError("eval_masks_head_resolution=True is not implemented; evaluate full-resolution masks")
    if type(values["eval_max_dets"]) is not int or values["eval_max_dets"] <= 0:
        raise ValueError("eval_max_dets must be a positive integer")
    for name in ("eval_batch_size", "max_targets", "resolution", "mask_point_sample_ratio", "lr_drop"):
        if values[name] is not None and (type(values[name]) is not int or values[name] <= 0):
            raise ValueError(f"{name} must be a positive integer")
    if values["seed"] is not None and (type(values["seed"]) is not int or values["seed"] < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    for name in ("lr", "lr_encoder", "weight_decay", "lr_vit_layer_decay", "lr_component_decay",
                 "clip_max_norm", "lr_min_factor", "warmup_epochs", "ema_decay", "early_stopping_min_delta",
                 "cls_loss_coef", "bbox_loss_coef", "giou_loss_coef", "mask_ce_loss_coef", "mask_dice_loss_coef",
                 "keypoint_l1_loss_coef", "keypoint_findable_loss_coef", "keypoint_visible_loss_coef",
                 "keypoint_nll_loss_coef"):
        value = values[name]
        if value is None and name == "cls_loss_coef":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if values["ema_decay"] > 1 or values["lr_min_factor"] > 1:
        raise ValueError("ema_decay and lr_min_factor must be at most 1")
    for name in ("epochs", "batch_size", "grad_accum_steps", "checkpoint_interval", "eval_interval",
                 "early_stopping_patience"):
        if type(values[name]) is not int or values[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(values["num_workers"]) is not int or values["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if values["dataset_file"] not in {"roboflow", "coco", "yolo"}:
        raise NotImplementedError("Objects365 dataset adapter is not implemented yet")
    if values["best_model_metric"] not in {"map", "mar"}:
        raise ValueError("best_model_metric must be 'map' or 'mar'")
    if values["lr_scheduler"] not in {"step", "cosine"}:
        raise NotImplementedError("only step and cosine schedulers are implemented")
    result = ModelConfig(**values)
    result.model_fields_set = set(kwargs)
    return result


def _dataset_format(root, configured):
    """Match Roboflow's COCO-first automatic format detection."""
    if configured != "roboflow":
        return configured
    root = Path(root)
    if (root / "train" / "_annotations.coco.json").exists():
        return "roboflow"
    if any((root / name).exists() for name in ("data.yaml", "data.yml")) and (root / "train" / "images").exists():
        return "yolo"
    raise ValueError(f"Could not detect Roboflow dataset format in {root}: expected train/_annotations.coco.json "
                     "or data.yaml/data.yml with train/images")


def _validate_category_schema(actual, expected, context):
    """Reject a changed label or ordered joint schema before rewriting a split index."""
    by_id = {row["id"]: row for row in actual}
    for row in expected:
        observed = by_id.get(row["id"])
        if observed is None:
            continue
        if observed["name"] != row["name"]:
            raise ValueError(f"{context} category names disagree with the saved/training schema")
        for field in ("keypoints", "skeleton"):
            if observed.get(field, []) != row.get(field, []):
                raise ValueError(f"{context} ordered {field} schema disagrees with the saved/training schema")


def _paths(root, split, dataset_file, *, keypoints=False):
    root = Path(root)
    if dataset_file == "coco":
        name = "train2017" if split == "train" else "val2017"
        image_dir = root / name
        if not image_dir.exists() and (root / "images" / name).exists():
            image_dir = root / "images" / name
        prefix = "person_keypoints" if keypoints else "instances"
        return image_dir, root / "annotations" / f"{prefix}_{name}.json"
    directory = root / ("valid" if split == "val" else split)
    return directory, directory / "_annotations.coco.json"


def _categories(raw, *, keypoints=False):
    ordered = sorted(raw["categories"], key=lambda row: row["id"])
    if keypoints:
        return ordered
    parents = {row.get("supercategory") for row in ordered
               if row.get("supercategory") not in {None, "", "none", "None"}
               and row.get("supercategory") != row["name"]}
    used = {row["category_id"] for row in raw["annotations"]}
    return [row for row in ordered if row["name"] not in parents or row["id"] in used] or ordered


def _prepare_split(root, split, config, scratch, *, categories=None, keypoints=False, segmentation=False):
    dataset_file = _dataset_format(root, config.dataset_file)
    if dataset_file == "yolo":
        from rfdetr_jax.datasets.yolo import convert_yolo_split, read_yolo_config
        effective_split = split
        if split == "test" and read_yolo_config(root).get("test") is None:
            test_images = Path(root) / "test" / "images"
            if not test_images.exists() and not test_images.is_symlink():
                warnings.warn("YOLO test split is unavailable; evaluating the validation split", stacklevel=2)
                effective_split = "val"
        images, annotations = convert_yolo_split(root, effective_split, scratch / f"{split}-coco.json",
                                                 segmentation=segmentation,
                                                 keypoints=keypoints)
    else:
        images, annotations = _paths(root, split, dataset_file, keypoints=keypoints)
    raw = json.loads(annotations.read_text())
    if dataset_file == "yolo" and split == "test" and effective_split == "test" and not raw["images"]:
        raise ValueError("YOLO test images directory contains no supported images")
    categories = categories if categories is not None else _categories(raw, keypoints=keypoints)
    allowed = {row["id"] for row in categories}
    if any(row["category_id"] not in allowed for row in raw["annotations"]):
        raise ValueError(f"{split} contains annotated categories missing from the training schema")
    _validate_category_schema(raw["categories"], categories, split)
    raw["categories"] = categories
    destination = scratch / f"{split}.json"
    destination.write_text(json.dumps(raw))
    dataset = CocoDetection(images, destination, segmentation=False, keypoints=keypoints)
    capacity = max(Counter(row["image_id"] for row in raw["annotations"] if not row.get("iscrowd", 0)).values(),
                   default=0)
    return images, destination, dataset, max(1, capacity)


def _schema(model, dataset, config):
    spec = model.spec
    raw_coco_labels = config.dataset_file == "coco" and not spec.use_grouppose_keypoints
    mapping = ({category: category for category in dataset.category_ids} if raw_coco_labels else
               {category: index for index, category in enumerate(dataset.category_ids)})
    if spec.use_grouppose_keypoints:
        counts = keypoint_category_counts(dataset)
        if "num_keypoints_per_class" not in model.model_config.model_fields_set:
            schema = tuple(counts[category] for category in sorted(counts) if counts[category])
            if not schema:
                raise ValueError("training keypoints requires a dataset keypoint schema")
        else:
            schema = spec.num_keypoints_per_class
        mapping = keypoint_category_mapping(dataset, schema)
        spec = replace(spec, num_keypoints_per_class=schema)
    expected = max(len(dataset.category_ids), len(spec.num_keypoints_per_class))
    if raw_coco_labels:
        # The standard COCO builder uses sparse source IDs and keeps its configured head.
        # Its 80 categories require slots through ID 90, not a contiguous 80-way replacement.
        pass
    elif "num_classes" not in model.model_config.model_fields_set:
        spec = replace(spec, num_classes=expected)
    elif spec.num_classes != expected:
        warnings.warn("Explicit num_classes differs from the dataset; preserving the configured head", stacklevel=3)
    if max(mapping.values()) >= spec.detector_classes:
        raise ValueError("configured class head cannot represent every dataset category")
    if config.resolution is not None:
        changes = {"resolution": config.resolution}
        if spec.positional_encoding_size == spec.resolution // spec.patch_size:
            changes["positional_encoding_size"] = config.resolution // spec.patch_size
        spec = replace(spec, **changes)
    return spec, mapping


def _resume_configuration(kwargs, task):
    """Inherit omitted public options while keeping explicit overrides visible to validation."""
    kwargs = _normalize_boolean_options(kwargs)
    if "multi_scale" in kwargs:
        kwargs["multi_scale"] = _multi_scale_enabled(kwargs["multi_scale"])
    if not kwargs.get("resume"):
        return training_config(task=task, **kwargs), None
    from rfdetr_jax.training.checkpoint import checkpoint_metadata
    saved = checkpoint_metadata(kwargs["resume"])
    fixed = {name: value for name, value in saved["run_config"].items() if name in (
        "epochs", "batch_size", "grad_accum_steps", "lr", "lr_encoder", "weight_decay", "lr_vit_layer_decay",
        "lr_component_decay", "clip_max_norm", "lr_scheduler", "lr_drop", "lr_min_factor", "warmup_epochs",
        "use_ema", "ema_decay", "ema_tau", "seed", "multi_scale", "expanded_scales", "scale_jitter")}
    fixed.update({name: value for name, value in saved.get("criterion_config", {}).items() if name in _DEFAULTS})
    for name, value in kwargs.items():
        if name in fixed and value is not None and value != fixed[name]:
            raise ValueError(f"resume must preserve {name}; start a new fine-tuning run to change it")
    inherited = {key: value for key, value in saved["run_config"].items()
                 if key in _DEFAULTS and key not in ("seed", "dataset_dir")}
    inherited.update({key: value for key, value in saved.get("criterion_config", {}).items() if key in _DEFAULTS})
    public = saved["model_metadata"].get("public_training_config", {})
    inherited.update({key: value for key, value in public.items() if key in _DEFAULTS and key != "resume"})
    options = {**inherited, **kwargs}
    # The deprecated flag's presence itself changes upstream defaults. Stored defaults are not new user input.
    if "eval_ema_only" not in kwargs:
        options.pop("eval_ema_only", None)
    return training_config(task=task, **options), saved


class _TrainingMonitor:
    """Keep strict best-checkpoint comparison separate from early-stopping min_delta."""

    def __init__(self, config, task):
        self.config, self.task = config, task
        self.best = {"val": -float("inf"), "val_ema": -float("inf")}
        self.best_raw = dict(self.best)
        self.smoothed_regular = 0.
        self.selected_track = None
        self.stop_best, self.bad_epochs = -float("inf"), 0

    def update(self, record):
        key = "mAP" if self.config.best_model_metric == "map" else (
            "bbox_mAR_max" if self.task == "segmentation" else "mAR_max")
        scores = {track: record[track][key] for track in self.best
                  if key in record.get(track, {})}
        if "val" in scores and self.config.smooth_alpha:
            alpha = self.config.smooth_alpha
            self.smoothed_regular = alpha * self.smoothed_regular + (1 - alpha) * scores["val"]
        if record["epoch"] < self.config.skip_best_epochs:
            return [], None, False
        improved = []
        for track, raw_score in scores.items():
            score = self.smoothed_regular if track == "val" and self.config.smooth_alpha else raw_score
            if np.isfinite(score) and score > self.best[track]:
                self.best[track], self.best_raw[track] = score, raw_score
                improved.append(track)
        winner = max(self.best_raw, key=self.best_raw.get)
        if not np.isfinite(self.best_raw[winner]):
            winner = None
        changed = winner != self.selected_track or winner in improved
        self.selected_track = winner
        winner = winner if changed else None
        if not scores:
            return improved, winner, False
        score = scores.get("val_ema") if self.config.early_stopping_use_ema else None
        if score is None:
            score = max(scores.values())
        if not np.isfinite(score):
            stop = True
        elif score > self.stop_best + self.config.early_stopping_min_delta:
            self.stop_best, self.bad_epochs, stop = score, 0, False
        else:
            self.bad_epochs += 1
            stop = self.bad_epochs >= self.config.early_stopping_patience
        return improved, winner, self.config.early_stopping and stop


def train(model, **kwargs):
    """Run the public train method and retain trained weights for later prediction."""
    from rfdetr_jax.checkpoints import (
        embed_native_resume,
        load_pretrained_training_parameters,
        save_checkpoint,
    )
    from rfdetr_jax.detr import resolve_device
    from rfdetr_jax.evaluation.public_evaluation import evaluate_public_predictions
    from rfdetr_jax.models.criterion import CriterionConfig
    from rfdetr_jax.training.run import run_training

    if getattr(model, "_has_been_trained", False):
        warnings.warn(
            "Calling train() on an already trained or checkpoint-loaded model. Without resume, "
            "the configured checkpoint is reloaded, or native parameters are initialized afresh "
            "when pretrain_weights=None. Pass resume=<checkpoint_path> to restore native training state.",
            UserWarning, stacklevel=3,
        )
    config, resumed = _resume_configuration(kwargs, task_for_spec(model.spec))
    if config.dataset_dir is None:
        raise ValueError("dataset_dir is required")
    config.dataset_file = _dataset_format(config.dataset_dir, config.dataset_file)
    if config.dataset_file == "yolo" and model.spec.use_grouppose_keypoints:
        from rfdetr_jax.datasets.yolo import yolo_pose_schema
        pose = yolo_pose_schema(config.dataset_dir)
        if "keypoint_flip_pairs" not in kwargs and not config.keypoint_flip_pairs:
            config.keypoint_flip_pairs = pose["flip_pairs"]
        if "keypoint_oks_sigmas" not in kwargs and not config.keypoint_oks_sigmas:
            config.keypoint_oks_sigmas = pose["oks_sigmas"]
    device = resolve_device(config.device or getattr(model.model_config, "device", "auto"))
    from rfdetr_jax.training.precision import resolve_training_precision
    legacy_amp = getattr(model.model_config, "amp", True)
    if resumed and "amp" not in getattr(model.model_config, "model_fields_set", set()):
        # A fresh default constructor must inherit the resumed run's legacy AMP
        # policy, just as it inherits freeze_encoder and fused_optimizer below.
        saved_options = resumed["model_metadata"].get("model_options", {})
        legacy_amp = _boolean_option("amp", saved_options.get("amp", legacy_amp))
    precision = resolve_training_precision(config.amp_dtype, legacy_amp, device)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rfdetr-dataset-") as temp:
        scratch = Path(temp)
        images, annotations, dataset, capacity = _prepare_split(
            config.dataset_dir, "train", config, scratch, keypoints=model.spec.use_grouppose_keypoints,
            segmentation=model.spec.segmentation_head)
        if config.max_targets is not None and config.max_targets < capacity:
            raise ValueError(f"max_targets={config.max_targets} is below the {capacity} annotations of the "
                             "largest training image; raise it to avoid discarding annotations")
        val_images, val_annotations, _, _ = _prepare_split(
            config.dataset_dir, "val", config, scratch, categories=dataset.categories,
            keypoints=model.spec.use_grouppose_keypoints, segmentation=model.spec.segmentation_head)
        if resumed:
            from rfdetr_jax.models.lwdetr import LWDETRSpec
            saved_spec = dict(resumed["model_metadata"]["model_spec"])
            for name in ("projector_scale", "out_feature_indexes", "num_keypoints_per_class"):
                saved_spec[name] = tuple(saved_spec[name])
            resumed_spec = LWDETRSpec(**saved_spec)
            if task_for_spec(resumed_spec) != task_for_spec(model.spec):
                raise ValueError("resume task differs from the constructed model")
            if config.resolution is not None and config.resolution != resumed_spec.resolution:
                raise ValueError("resume must preserve resolution")
            spec = resumed_spec
            saved_mapping = {int(key): int(value) for key, value in
                             resumed["model_metadata"]["category_to_label"].items()}
            if set(saved_mapping) != set(dataset.category_ids):
                raise ValueError("resume must preserve the dataset category mapping")
            if spec.use_grouppose_keypoints and keypoint_category_mapping(
                    dataset, spec.num_keypoints_per_class) != saved_mapping:
                raise ValueError("resume keypoint schema disagrees with the dataset")
            mapping = saved_mapping
            params = None
        else:
            spec, mapping = _schema(model, dataset, config)
            # Let the native trainer initialize a new aligned model for None.
            # Reusing live constructor/trained weights would silently continue
            # a previous model despite this being a fresh training call.
            params = None
        if resumed and "amp_dtype" not in kwargs and resumed["model_metadata"].get("training_precision"):
            # An omitted request inherits the saved run's resolved precision: a run started on
            # one device must not change dtype because the resuming device resolves "auto" differently.
            precision = dict(resumed["model_metadata"]["training_precision"])
        if resumed and (spec.compute_dtype != precision["compute_dtype"] or spec.head_dtype != precision["head_dtype"]):
            raise ValueError("resume must preserve resolved training precision")
        spec = replace(spec, compute_dtype=precision["compute_dtype"], head_dtype=precision["head_dtype"])
        if not resumed and getattr(model.model_config, "pretrain_weights", None) is not None:
            params = load_pretrained_training_parameters(model._variant_name, model.model_config, spec)
        names_by_category = dict(zip(dataset.category_ids, dataset.category_names, strict=True))
        names = list(config.class_names) if config.class_names is not None else [
            names_by_category[category] for category in sorted(mapping, key=mapping.get)]
        if len(names) != len(dataset.category_ids):
            raise ValueError("class_names must contain one name per dataset category")
        model_config = model.model_config.model_copy(update=asdict(spec))
        model_config.device = config.device or getattr(model.model_config, "device", "auto")
        model_config.amp = legacy_amp
        trace_alpha = (resumed["run_config"].get("postprocess_trace_alpha", .2) if resumed else
                       getattr(model.model_config, "postprocess_trace_alpha", .2))
        model_config.postprocess_trace_alpha = trace_alpha
        # Dataset-inferred fields must remain distinguishable from explicit user overrides.
        model_config.model_fields_set = set(model.model_config.model_fields_set)
        metadata = {"model_spec": asdict(spec), "model_name": type(model).__name__,
                    "class_names": names, "category_ids": dataset.category_ids,
                    "category_names": dataset.category_names, "category_to_label": mapping,
                    "category_schema": dataset.categories,
                    "postprocess_trace_alpha": trace_alpha,
                    "task": task_for_spec(spec), "notes": config.notes, "training_precision": precision,
                    "model_options": {name: getattr(model_config, name) for name in (
                        "amp", "freeze_encoder", "fused_optimizer", "ia_bce_loss", "postprocess_trace_alpha")
                        if hasattr(model_config, name)},
                    "keypoint_flip_pairs": tuple(config.keypoint_flip_pairs or ()),
                    "keypoint_oks_sigmas": tuple(config.keypoint_oks_sigmas or ()),
                    "public_training_config": json.loads(json.dumps(
                        {key: getattr(config, key) for key in _DEFAULTS if key != "resume"}, default=str))}
        argv = ["--train-images", str(images), "--train-annotations", str(annotations),
                "--val-images", str(val_images), "--val-annotations", str(val_annotations),
                "--output-dir", str(output), "--max-targets", str(config.max_targets or capacity),
                "--workers", str(config.num_workers), "--worker-mode", "thread", "--target-buckets", "",
                "--eval-every", str(config.eval_interval), "--matcher", config.matcher,
                "--postprocess-trace-alpha", str(trace_alpha)]
        argv += ["--compute-dtype", precision["compute_dtype"], "--head-dtype", precision["head_dtype"],
                 "--pixel-dtype", "float32"]
        for name in ("epochs", "batch_size", "grad_accum_steps", "lr", "lr_encoder", "weight_decay",
                     "lr_vit_layer_decay", "lr_component_decay", "clip_max_norm", "lr_scheduler", "lr_drop",
                     "lr_min_factor", "warmup_epochs", "ema_decay", "ema_tau"):
            argv += ["--" + name.replace("_", "-"), str(getattr(config, name))]
        argv += ["--use-ema" if config.use_ema else "--no-use-ema"]
        ema_first_update_copy = resumed["run_config"].get("ema_first_update_copy", False) if resumed else True
        argv += ["--ema-first-update-copy" if ema_first_update_copy else "--no-ema-first-update-copy"]
        metadata["ema_first_update_copy"] = ema_first_update_copy
        keypoint_accumulation = (resumed["run_config"].get("keypoint_box_normalized_accumulation", False)
                                 if resumed else spec.use_grouppose_keypoints)
        argv += [("--" if keypoint_accumulation else "--no-") + "keypoint-box-normalized-accumulation"]
        metadata["keypoint_box_normalized_accumulation"] = keypoint_accumulation
        automatic_accumulation = (resumed["run_config"].get("automatic_accumulation_normalization", False)
                                  if resumed else not spec.use_grouppose_keypoints)
        argv += [("--" if automatic_accumulation else "--no-") + "automatic-accumulation-normalization"]
        metadata["automatic_accumulation_normalization"] = automatic_accumulation
        if resumed:
            freeze_encoder = resumed["run_config"].get("freeze_encoder", False)
        else:
            freeze_encoder = getattr(model.model_config, "freeze_encoder", False)
        argv += ["--freeze-encoder" if freeze_encoder else "--no-freeze-encoder"]
        fused_optimizer = (resumed["run_config"].get("fused_optimizer", True) if resumed else
                           getattr(model.model_config, "fused_optimizer", True))
        argv += ["--fused-optimizer" if fused_optimizer else "--no-fused-optimizer"]
        for name in ("multi_scale", "expanded_scales", "scale_jitter"):
            argv += [("--" if getattr(config, name) else "--no-") + name.replace("_", "-")]
        if resumed:
            # Requested public seed=None does not erase the saved native
            # streams, and a matching explicit seed must not collapse them.
            argv += ["--seed", str(resumed["run_config"]["seed"]),
                     "--data-seed", str(resumed["run_config"]["data_seed"])]
        elif config.seed is not None:
            argv += ["--seed", str(config.seed), "--data-seed", str(config.seed)]
        if config.eval_batch_size is not None:
            argv += ["--eval-batch-size", str(config.eval_batch_size)]
        for name in ("keypoint_flip_pairs", "keypoint_oks_sigmas"):
            values = getattr(config, name)
            if values:
                argv += ["--" + name.replace("_", "-"), ",".join(map(str, values))]
        for name in ("no_precompile", "no_compile_cache"):
            if getattr(config, name):
                argv += ["--" + name.replace("_", "-")]
        if config.resume:
            argv += ["--resume", str(config.resume)]
        if config.eval_base_model:
            argv += ["--eval-raw"]
        criterion = CriterionConfig(
            num_classes=spec.detector_classes, group_detr=spec.group_detr, matcher=config.matcher,
            ia_bce_loss=(resumed["criterion_config"]["ia_bce_loss"] if resumed else
                         getattr(model.model_config, "ia_bce_loss", True)),
            segmentation_head=spec.segmentation_head, use_grouppose_keypoints=spec.use_grouppose_keypoints,
            num_keypoints_per_class=spec.num_keypoints_per_class,
            cls_loss_coef=config.cls_loss_coef if config.cls_loss_coef is not None else (
                2. if spec.use_grouppose_keypoints else 1.),
            **{name: getattr(config, name) for name in (
                "bbox_loss_coef", "giou_loss_coef", "mask_point_sample_ratio", "mask_ce_loss_coef",
                "mask_dice_loss_coef", "keypoint_l1_loss_coef", "keypoint_findable_loss_coef",
                "keypoint_visible_loss_coef", "keypoint_nll_loss_coef")},
        )
        writer = None
        if config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(log_dir=str(output / "tensorboard"))
            except ImportError:
                warnings.warn("TensorBoard is unavailable; install tensorboard to enable logging", stacklevel=2)
        monitor = _TrainingMonitor(config, task_for_spec(spec))
        for record in resumed.get("epoch_log", []) if resumed else ():
            monitor.update(record)

        def epoch_end(record, state):
            epoch = record["epoch"]
            epoch_metadata = {**metadata, "step": int(state.step)}
            native = output / "checkpoint_last.npz"

            def save(name, parameters, source="regular"):
                destination = output / name
                save_checkpoint(destination, parameters, spec,
                                {**epoch_metadata, "weights": "ema" if source == "ema" else "raw",
                                 "best_total_source": source}, model_config)
                embed_native_resume(destination, native)

            save("checkpoint.pth", state.params)
            if (epoch + 1) % config.checkpoint_interval == 0:
                save(f"checkpoint{epoch:04}.pth", state.params)
            improved, winner, stop = monitor.update(record)
            for track in improved:
                ema = track == "val_ema"
                save("checkpoint_best_ema.pth" if ema else "checkpoint_best_regular.pth",
                     state.ema_params if ema else state.params, "ema" if ema else "regular")
            if winner is not None:
                source = output / ("checkpoint_best_ema.pth" if winner == "val_ema"
                                   else "checkpoint_best_regular.pth")
                if source.exists():
                    shutil.copyfile(source, output / "checkpoint_best_total.pth")
                else:
                    warnings.warn("Historical best weights are absent from this resume directory; "
                                  "best_total will be written when a new winning checkpoint is available", stacklevel=2)
            if writer:
                for group in ("losses", "val", "val_ema"):
                    for name, value in record.get(group, {}).items():
                        writer.add_scalar(f"{group}/{name}", value, epoch)
            for callback in model.callbacks.get("on_fit_epoch_end", ()):
                callback(record)
            return stop

        def validation_records(records, image_sizes):
            return evaluate_public_predictions(
                val_annotations, records, task=task_for_spec(spec), resolution=spec.resolution,
                image_sizes=image_sizes, max_dets=config.eval_max_dets,
                keypoint_oks_sigmas=list(config.keypoint_oks_sigmas or ()) or None,
            )

        def write_training_config(current_config, current_spec):
            payload = {"train_config": metadata["public_training_config"],
                       "model_config": current_config.model_dump(),
                       "model_config_type": f"{getattr(model, '_variant_name', type(model).__name__)}Config",
                       "class_names": names,
                       "num_classes": len(names), "model_name": type(model).__name__}
            (output / "training_config.json").write_text(json.dumps(payload, indent=2, default=str))

        write_training_config(model_config, spec)
        try:
            with jax.default_device(device):
                result = run_training(argv, initial_params=params, model_spec=spec,
                                      category_to_label=mapping, metadata=metadata,
                                      on_epoch_end=epoch_end, criterion_config=criterion,
                                      evaluation_record_evaluator=validation_records)
        finally:
            if writer:
                writer.close()
        actual_spec = replace(spec, **result.metadata.get("model_spec", {}))
        explicit_model_fields = set(model_config.model_fields_set)
        model_config = model_config.model_copy(update=asdict(actual_spec))
        model_config.model_fields_set = explicit_model_fields
        # Published RF-DETR keeps module.model after fit (run_test=False), so
        # public predict/save/evaluate use the final raw optimizer parameters.
        # The lower-level trainer's inference export deliberately keeps EMA.
        model._install(actual_spec, result.state.params, {**result.metadata, "weights": "raw"}, model_config)
        ema_params = getattr(getattr(result, "state", None), "ema_params", None)
        if ema_params is not None:
            # Upstream always writes the final EMA weights beside the epoch checkpoints.
            save_checkpoint(output / "last_ema.pth", ema_params, actual_spec,
                            {**result.metadata, "weights": "ema", "best_total_source": "ema"}, model_config)
            if (output / "checkpoint_last.npz").exists():
                embed_native_resume(output / "last_ema.pth", output / "checkpoint_last.npz")
        write_training_config(model.model_config, actual_spec)
        model._has_been_trained = True
        model._last_train_kwargs = metadata["public_training_config"]
        model._training_result = result
        for callback in model.callbacks.get("on_train_end", ()):
            callback(result.log)
    return None


def _inferred_evaluation_mapping(model, dataset, metadata, split, *, native_coco=False):
    """Map the evaluation dataset onto a checkpoint that carries no dataset schema of its own."""
    spec = model.spec
    if spec.use_grouppose_keypoints:
        mapping = keypoint_category_mapping(dataset, spec.num_keypoints_per_class)
    elif native_coco:
        # The standard COCO builder keeps sparse source ids as labels, as train() does.
        mapping = {category: category for category in dataset.category_ids}
    else:
        # Roboflow and YOLO label spaces are the sorted kept categories, as upstream builds them.
        mapping = {category: index for index, category in enumerate(dataset.category_ids)}
    if not native_coco and len(dataset.category_ids) != spec.num_classes:
        # Native COCO keeps sparse raw ids inside the configured head, so a category count below
        # the class count is the normal case there; the capacity check below still applies.
        warnings.warn(f"the checkpoint has {spec.num_classes} classes but the {split} split has "
                      f"{len(dataset.category_ids)} categories; labels follow the dataset and "
                      "train() would adapt the class head", UserWarning, stacklevel=3)
    if max(mapping.values(), default=-1) >= spec.detector_classes:
        raise ValueError("the evaluation dataset needs more class slots than the checkpoint provides")
    if native_coco and "category_ids" in metadata:
        names = dict(zip(metadata["category_ids"], metadata.get("category_names", []), strict=False))
        differing = [f"{name!r} is category {category} ({names[category]!r} in the checkpoint)"
                     for category, name in zip(dataset.category_ids, dataset.category_names, strict=True)
                     if category in names and names[category] != name]
    else:
        names = list(metadata.get("class_names", []))
        differing = [f"{name!r} is label {mapping[category]} ({names[mapping[category]]!r} in the checkpoint)"
                     for category, name in zip(dataset.category_ids, dataset.category_names, strict=True)
                     if mapping[category] < len(names) and names[mapping[category]] != name]
    if differing:
        warnings.warn("evaluation dataset class names differ from the checkpoint: " + "; ".join(differing),
                      UserWarning, stacklevel=3)
    return mapping


def evaluate(model, *, split="test", **kwargs):
    """Evaluate in-memory weights; validation image shape does not mutate the model."""
    from rfdetr_jax.detr import resolve_device
    from rfdetr_jax.evaluation.public_evaluation import evaluate_public_predictions
    from rfdetr_jax.models.lwdetr import LWDETR
    from rfdetr_jax.training.run import evaluate as evaluate_backend
    from rfdetr_jax.training.run_config import RunConfig

    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    # Published evaluate() builds a fresh task config. Saved training options
    # describe the old run; they are not defaults for this independent call.
    config = training_config(task=task_for_spec(model.spec), **kwargs)
    if config.dataset_dir is None:
        raise ValueError("dataset_dir is required")
    config.dataset_file = _dataset_format(config.dataset_dir, config.dataset_file)
    with tempfile.TemporaryDirectory(prefix="rfdetr-evaluation-") as temp:
        scratch = Path(temp)
        training_path = _paths(config.dataset_dir, "train", config.dataset_file,
                               keypoints=model.spec.use_grouppose_keypoints)[1]
        categories = None
        if training_path.exists():
            categories = _categories(json.loads(training_path.read_text()),
                                     keypoints=model.spec.use_grouppose_keypoints)
        images, annotations, dataset, capacity = _prepare_split(
            config.dataset_dir, split, config, scratch, categories=categories,
            keypoints=model.spec.use_grouppose_keypoints, segmentation=model.spec.segmentation_head)
        saved_metadata = model._backend.metadata
        inferred = saved_metadata.get("category_source") == "inferred" or "category_to_label" not in saved_metadata
        if inferred:
            # Upstream checkpoints carry label-indexed class names only; the label space is the
            # dataset's, derived exactly as train() derives it.
            mapping = _inferred_evaluation_mapping(model, dataset, saved_metadata, split,
                                                   native_coco=config.dataset_file == "coco")
        else:
            saved_ids = saved_metadata.get("category_ids", [])
            saved_names_by_id = dict(zip(saved_ids, saved_metadata.get("category_names", []), strict=True))
            for category, name in zip(dataset.category_ids, dataset.category_names, strict=True):
                if category in saved_names_by_id and saved_names_by_id[category] != name:
                    raise ValueError("evaluation category names disagree with the checkpoint")
            if "category_schema" in saved_metadata:
                _validate_category_schema(dataset.categories, saved_metadata["category_schema"], "evaluation")
            mapping = {int(key): int(value) for key, value in saved_metadata.get("category_to_label", {}).items()}
            if mapping and not set(dataset.category_ids).issubset(mapping):
                raise ValueError("evaluation dataset categories are absent from the checkpoint label mapping")
            if mapping:
                mapping = {category: mapping[category] for category in dataset.category_ids}
            elif model.spec.use_grouppose_keypoints:
                mapping = keypoint_category_mapping(dataset, model.spec.num_keypoints_per_class)
            elif config.dataset_file == "coco":
                # Defensive fallback for metadata-free contexts. Normal constructors
                # already provide a mapping; native COCO retains sparse source IDs.
                mapping = {category: category for category in dataset.category_ids}
            else:
                mapping = {category: index for index, category in enumerate(dataset.category_ids)}
        dataset.set_category_mapping(mapping)
        from rfdetr_jax.training.precision import resolve_training_precision
        device = resolve_device(config.device or getattr(model.model_config, "device", "auto"))
        precision = resolve_training_precision(config.amp_dtype, getattr(model.model_config, "amp", True), device)
        spec = replace(model.spec, compute_dtype=precision["compute_dtype"], head_dtype=precision["head_dtype"])
        resolution = config.resolution or spec.resolution
        if resolution % (spec.patch_size * spec.num_windows):
            raise ValueError("resolution must be divisible by patch_size * num_windows")
        pe_size = spec.positional_encoding_size
        if pe_size == spec.resolution // spec.patch_size:
            pe_size = resolution // spec.patch_size
        spec = replace(spec, resolution=resolution, positional_encoding_size=pe_size)
        params = model.params
        if pe_size != model.spec.positional_encoding_size:
            from rfdetr_jax.checkpoints import resize_position_embeddings

            # Only positional tensors change; current heads and Gaussian rows
            # survive evaluation without reopening the initial checkpoint.
            params = resize_position_embeddings(params, pe_size)
        network = LWDETR(spec=spec)
        task = task_for_spec(spec)

        @jax.jit
        def forward(params, pixels, sizes):
            outputs = network.apply({"params": params}, pixels, None, train=False)
            if task == "detection":
                from rfdetr_jax.models.postprocess import post_process

                return post_process(outputs["pred_logits"], outputs["pred_boxes"], sizes, spec.num_select)
            return {key: value for key, value in outputs.items() if key.startswith("pred_")}

        run = RunConfig(resolution=resolution, task=task, max_targets=config.max_targets or capacity,
                        num_select=spec.num_select, num_keypoints_per_class=spec.num_keypoints_per_class,
                        postprocess_trace_alpha=getattr(model.model_config, "postprocess_trace_alpha", .2),
                        keypoint_oks_sigmas=tuple(config.keypoint_oks_sigmas or ()))
        with jax.default_device(device), jax.default_matmul_precision("highest"):
            metrics = evaluate_backend(forward, jax.device_put(params, device), run, dataset, str(annotations),
                                       config.eval_batch_size or config.batch_size, workers=config.num_workers,
                                       pixel_dtype=jnp.float32,
                                       record_evaluator=lambda records, image_sizes: evaluate_public_predictions(
                                           annotations, records, task=task, resolution=resolution,
                                           image_sizes=image_sizes, max_dets=config.eval_max_dets,
                                           keypoint_oks_sigmas=list(config.keypoint_oks_sigmas or ()) or None))
    from rfdetr_jax.evaluation.public_metrics import format_public_metrics

    return format_public_metrics(metrics, split, task)
