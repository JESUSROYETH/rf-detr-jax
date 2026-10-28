"""Train the JAX RF-DETR port on the shared batch stream and log per-epoch metrics.

Bound thread counts on shared machines (the ``rfdetr-jax-train`` command honours the usual
``OMP_NUM_THREADS``-style variables). Pass ``--init-weights`` to start from a converted PyTorch
initialisation so a JAX loss curve can be compared with a PyTorch run on the same batches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from rfdetr_jax.utilities.gpu_flags import apply_gpu_xla_defaults  # noqa: E402

apply_gpu_xla_defaults()  # before the backend starts; a no-op without the CUDA plugin

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rfdetr_jax.configuration import (  # noqa: E402
    keypoint_category_counts,
    keypoint_category_mapping,
    load_model_spec,
    task_for_spec,
    validate_model_spec,
)
from rfdetr_jax.convert import (  # noqa: E402
    convert_state_dict,
    flatten_params,
    load_params_npz,
    save_params_npz,
)
from rfdetr_jax.datasets.coco import CocoDetection, iterate_batches  # noqa: E402
from rfdetr_jax.evaluation.coco_eval import accumulate_detections, evaluate_detections  # noqa: E402
from rfdetr_jax.models.criterion import CriterionConfig  # noqa: E402
from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec  # noqa: E402
from rfdetr_jax.training.checkpoint import (  # noqa: E402
    checkpoint_metadata,
    load_training_checkpoint,
    save_training_checkpoint,
)
from rfdetr_jax.training.optimizer import OptimizerSpec, build_optimizer, managed_schedule  # noqa: E402
from rfdetr_jax.training.run_config import DEFAULT_RUN, RunConfig  # noqa: E402
from rfdetr_jax.training.trainer import (  # noqa: E402
    TrainSettings,
    create_train_state,
    data_parallel,
    device_batches,
    init_params,
    make_predict_step,
    make_train_step,
    stack_microbatches,
)


def spec_for(run: RunConfig) -> LWDETRSpec:
    """Build the port's model spec from the shared run configuration."""
    return LWDETRSpec(
        encoder=run.encoder,
        hidden_dim=run.hidden_dim,
        patch_size=run.patch_size,
        num_windows=run.num_windows,
        dec_layers=run.dec_layers,
        sa_nheads=run.sa_nheads,
        ca_nheads=run.ca_nheads,
        dec_n_points=run.dec_n_points,
        num_queries=run.num_queries,
        num_select=run.num_select,
        projector_scale=run.projector_scale,
        out_feature_indexes=run.out_feature_indexes,
        resolution=run.resolution,
        positional_encoding_size=run.positional_encoding_size,
        num_classes=run.num_classes,
        group_detr=run.group_detr,
        gradient_checkpointing=run.gradient_checkpointing,
        segmentation_head=run.task == "segmentation",
        mask_downsample_ratio=run.mask_downsample_ratio,
        use_grouppose_keypoints=run.task == "keypoints",
        num_keypoints_per_class=run.num_keypoints_per_class,
    )


def evaluate(
    predict,
    params: dict,
    run: RunConfig,
    dataset: CocoDetection,
    annotation_file: str,
    batch_size: int,
    workers: int = 0,
    worker_mode: str = "thread",
    batch_sharding=None,
    pixel_dtype=np.float32,
    backend: str = "auto",
    prediction_metrics=None,
    record_evaluator=None,
) -> dict[str, float]:
    """Score validation batches, transferring and encoding each result before the next prediction.

    Dense mask logits must not accumulate on device across the whole validation split.
    ``record_evaluator(records, image_sizes)`` may replace raw COCO scoring entirely.
    Sizes are decoded original ``(height, width)`` pairs for valid images, never padding.
    """
    if record_evaluator is not None and prediction_metrics is not None:
        raise ValueError("record_evaluator and additive prediction_metrics are mutually exclusive")
    detections: list[dict] = []
    # Published metric callbacks retain all top-k predictions, including float32 underflow to zero.
    score_threshold = -1. if record_evaluator is not None else 0.
    image_sizes: dict[int, tuple[int, int]] = {}
    for batch in iterate_batches(
        dataset,
        run.resolution,
        batch_size,
        run.max_targets,
        train=False,
        drop_last=False,
        pad_last=True,
        workers=workers,
        pixel_dtype=pixel_dtype,
        worker_mode=worker_mode,
    ):
        pixels = (
            jnp.asarray(batch["pixel_values"])
            if batch_sharding is None
            else jax.device_put(batch["pixel_values"], batch_sharding)
        )
        outputs = predict(params, pixels, jnp.asarray(batch["orig_sizes"]))
        # A short trailing batch is padded with repeats so every call has one shape; drop them here.
        image_ids, valid, original_sizes = batch["image_ids"], int(batch["num_valid"]), batch["orig_sizes"]
        image_sizes.update({int(image_id): tuple(int(value) for value in size)
                            for image_id, size in zip(image_ids[:valid], original_sizes[:valid], strict=True)})
        if run.task != "detection":
            from rfdetr_jax.evaluation.tasks import accumulate_predictions

            detections.extend(accumulate_predictions(
                image_ids[:valid], {key: np.asarray(value)[:valid] for key, value in outputs.items()},
                original_sizes[:valid], dataset.category_ids, task=run.task, num_select=run.num_select,
                num_keypoints_per_class=run.num_keypoints_per_class,
                trace_alpha=run.postprocess_trace_alpha,
                score_threshold=score_threshold,
                preserve_xyxy=record_evaluator is not None,
                label_to_category={label: category for category, label in dataset.category_to_label.items()},
            ))
            continue
        scores, labels, boxes = (np.asarray(value)[:valid] for value in outputs)
        inverse = {label: category for category, label in dataset.category_to_label.items()}
        labels = np.asarray([[inverse.get(int(label), -1) for label in row] for row in labels], dtype=np.int32)
        detections.extend(
            accumulate_detections(image_ids[:valid], scores, labels, boxes, dataset.category_ids,
                                  score_threshold=score_threshold, preserve_xyxy=record_evaluator is not None)
        )
    if record_evaluator is not None:
        return record_evaluator(detections, image_sizes)
    extra_metrics = {} if prediction_metrics is None else prediction_metrics(detections)
    if run.task != "detection":
        from rfdetr_jax.evaluation.tasks import evaluate_predictions

        metrics = evaluate_predictions(annotation_file, detections, task=run.task, max_dets=run.num_select,
                                       backend=backend, keypoint_oks_sigmas=list(run.keypoint_oks_sigmas) or None)
    else:
        metrics = evaluate_detections(annotation_file, detections, max_dets=run.num_select, backend=backend)
    overlap = set(metrics) & set(extra_metrics)
    if overlap:
        raise ValueError(f"prediction_metrics cannot overwrite backend measurements: {sorted(overlap)}")
    return {**metrics, **extra_metrics}


def precompile_steps(step_fn, state, key, widths, batch_size, resolution, pixel_dtype, sharding,
                     task="detection", keypoint_count=0) -> dict:
    """Compile the train step for every padded target width at once, in parallel threads.

    Each width is its own XLA program, and compiled lazily they run one after another inside the
    first epoch. XLA compiles outside the GIL, so threads overlap them; the result maps a batch's
    target width to its executable.
    """
    import concurrent.futures

    # lower/compile executes in a fresh worker thread, which does not inherit
    # the public train(device=...) context from the dispatching thread.
    default_device = jax.config.jax_default_device

    def spec(shape, dtype):
        if sharding is None:
            return jax.ShapeDtypeStruct(shape, dtype)
        return jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)

    def one(width: int):
        batch = {
            "pixel_values": spec((batch_size, 3, resolution, resolution), pixel_dtype),
            "labels": spec((batch_size, width), jnp.int32),
            "boxes": spec((batch_size, width, 4), jnp.float32),
            "target_mask": spec((batch_size, width), jnp.bool_),
        }
        if task == "segmentation":
            batch["masks"] = spec((batch_size, width, resolution, resolution), jnp.float32)
        elif task == "keypoints":
            batch["keypoints"] = spec((batch_size, width, keypoint_count, 3), jnp.float32)
        with jax.default_device(default_device):
            return width, step_fn.lower(state, batch, key).compile()

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(len(widths)) as pool:
        compiled = dict(pool.map(one, widths))
    print(f"compiled {len(widths)} step variants {widths} in {time.time() - started:.1f}s", flush=True)
    return compiled


def load_initial_params(path: str | Path, template: dict) -> dict:
    """Load an already converted checkpoint without renaming its paths a second time."""
    params = load_params_npz(path)
    actual = flatten_params(params)
    expected = flatten_params(template)
    missing = expected.keys() - actual.keys()
    unexpected = actual.keys() - expected.keys()
    mismatched = [key for key in expected.keys() & actual.keys() if expected[key].shape != actual[key].shape]
    if any(not np.isfinite(value).all() for value in actual.values()):
        raise ValueError("initial parameters contain nonfinite values")
    if missing or unexpected or mismatched:
        raise ValueError(
            f"initial parameter mismatch: {len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(mismatched)} wrong shapes; use the same model geometry and --num-classes"
        )
    return params


@dataclass
class TrainingResult:
    """In-memory and persisted results of a complete or deliberately interrupted run."""

    params: dict
    state: object
    metadata: dict
    log: dict
    paths: dict


def run_training(
    argv: list[str] | None = None, *, initial_params: dict | None = None,
    model_spec: LWDETRSpec | None = None, category_to_label: dict[int, int] | None = None,
    metadata: dict | None = None, on_epoch_end=None, stop_after_epoch: int | None = None,
    criterion_config: CriterionConfig | None = None, evaluation_record_evaluator=None,
) -> TrainingResult:
    """Train without mutating process arguments; resume checkpoints preserve the original schedule.

    ``epochs`` is the planned total horizon, including completed epochs. ``stop_after_epoch``
    stops at a saved epoch boundary without changing that horizon (useful for interruption tests).
    The callback receives ``(epoch_record, state)`` after the full-state checkpoint is saved;
    returning True requests a clean stop. ``evaluation_record_evaluator(records, image_sizes)``
    replaces raw validation scoring for both the regular and EMA parameters.
    """
    provided_metadata = dict(metadata or {})
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["detection", "segmentation", "keypoints"])
    parser.add_argument("--model-config", help="LWDETRSpec JSON or saved metadata; preserves checkpoint geometry.")
    parser.add_argument("--keypoint-schema", help="Comma-separated keypoint counts indexed by model label slot.")
    parser.add_argument(
        "--keypoint-flip-pairs", default="", help="Comma-separated left/right index pairs, e.g. 0,1,2,3."
    )
    parser.add_argument("--keypoint-oks-sigmas", default="", help="Comma-separated positive OKS sigmas.")
    parser.add_argument("--mask-downsample-ratio", type=int)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--dec-layers", type=int)
    parser.add_argument("--num-queries", type=int)
    parser.add_argument("--num-select", type=int)
    parser.add_argument("--postprocess-trace-alpha", type=float, default=DEFAULT_RUN.postprocess_trace_alpha)
    parser.add_argument("--epochs", type=int, default=DEFAULT_RUN.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_RUN.batch_size)
    parser.add_argument("--out", default="artifacts/train_jax.json")
    parser.add_argument("--output-dir", help="Directory for metrics, inference weights and resumable checkpoints.")
    parser.add_argument("--resume", help="Full-state checkpoint_last.npz from an interrupted run.")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    for name in ("multi_scale", "expanded_scales", "scale_jitter", "freeze_encoder", "fused_optimizer",
                 "ema_first_update_copy", "keypoint_box_normalized_accumulation",
                 "automatic_accumulation_normalization"):
        parser.add_argument("--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction,
                            default=getattr(DEFAULT_RUN, name))
    parser.add_argument("--seed", type=int, default=DEFAULT_RUN.seed)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_RUN.data_seed)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=DEFAULT_RUN.use_ema)
    parser.add_argument("--lr-scheduler", choices=["step", "cosine"], default=DEFAULT_RUN.lr_scheduler)
    parser.add_argument("--lr-drop", type=int, default=DEFAULT_RUN.lr_drop)
    parser.add_argument("--ema-tau", type=int, default=DEFAULT_RUN.ema_tau)
    for option in ("lr", "lr_encoder", "weight_decay", "lr_vit_layer_decay", "lr_component_decay",
                   "clip_max_norm", "warmup_epochs", "lr_min_factor", "ema_decay"):
        parser.add_argument("--" + option.replace("_", "-"), type=float, default=getattr(DEFAULT_RUN, option))
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--init-weights", help="Initial PyTorch-keyed state_dict .npz.")
    initial.add_argument(
        "--init-params", help="Initial converted/saved Flax .npz; omit both options for random weights."
    )
    parser.add_argument(
        "--save-params", help="Final inference weights .npz (EMA when enabled); defaults to OUT.params.npz."
    )
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-steps", type=int, default=10, help="Record this many per-step losses.")
    parser.add_argument(
        "--precision",
        default="highest",
        choices=["highest", "float32", "default", "bfloat16"],
        help="Matmul precision; 'highest' disables TF32 so numbers match float32 PyTorch.",
    )
    parser.add_argument(
        "--matcher",
        default="hungarian",
        choices=["hungarian", "auction"],
        help="'auction' solves the assignment on device instead of through a host callback.",
    )
    parser.add_argument(
        "--sampler",
        default="gather",
        choices=["gather", "gather-f32", "matmul", "auto", "pallas"],
        help="Deformable-attention sampling kernel; 'matmul' trades arithmetic for indexed reads.",
    )
    parser.add_argument(
        "--no-compile-cache",
        action="store_true",
        help="Do not keep compiled steps in the persistent XLA cache (~/.cache/rfdetr_jax/xla or "
        "$JAX_COMPILATION_CACHE_DIR), so every run recompiles from scratch.",
    )
    parser.add_argument(
        "--no-precompile",
        action="store_true",
        help="Compile each target-bucket step lazily on first use instead of all at once, in parallel, "
        "before the first epoch.",
    )
    parser.add_argument(
        "--head-dtype",
        default=None,
        choices=["float32", "activation"],
        help="Dtype of the box/class head matmuls; 'activation' matches torch autocast. Defaults to "
        "'activation' on a GPU and 'float32' elsewhere.",
    )
    parser.add_argument(
        "--attention",
        default=None,
        choices=["xla", "cudnn"],
        help="Softmax-attention kernel; 'cudnn' uses cuDNN's fused attention on a GPU with bf16 "
        "activations (defaults to $RFDETR_JAX_ATTENTION, else 'xla').",
    )
    parser.add_argument(
        "--grad-checkpoint",
        default="config",
        choices=["config", "on", "off"],
        help="Override the run config's gradient checkpointing.",
    )
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Replicate the state and split each batch across every local device.",
    )
    parser.add_argument("--host-init", action="store_true", help="Initialise parameters on the host.")
    parser.add_argument(
        "--compute-dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Activation dtype; float16 enables dynamic loss scaling. Master parameters remain float32.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Threads decoding and augmenting images; 0 keeps the single-threaded reader.",
    )
    parser.add_argument(
        "--target-buckets",
        default="32,64",
        help="Comma-separated padded target counts: each training batch is padded to the smallest one "
        "holding its targets instead of always to --max-targets. Padding is masked out of matching "
        "and losses; one compilation per bucket used. '' pads everything to --max-targets.",
    )
    parser.add_argument(
        "--dp-mode",
        default="shard_map",
        choices=["shard_map", "gspmd"],
        help="With --data-parallel: run the step as an explicit per-device program (shard_map), or "
        "leave the partitioning to XLA's sharding propagation (gspmd).",
    )
    parser.add_argument(
        "--eval-raw",
        action="store_true",
        help="Also score the raw weights each epoch; by default only the EMA weights are scored.",
    )
    parser.add_argument(
        "--worker-mode",
        default="process",
        choices=["process", "thread"],
        help="Run --workers as processes (scale with the host's cores) or threads (share one interpreter).",
    )
    parser.add_argument("--dataset-dir", default=None, help="Roboflow-layout root (train/, valid/ splits).")
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--train-images", default=None, help="Override the training image directory.")
    parser.add_argument("--train-annotations", default=None, help="Override the training COCO JSON.")
    parser.add_argument("--val-images", default=None, help="Override the validation image directory.")
    parser.add_argument("--val-annotations", default=None, help="Override the validation COCO JSON.")
    parser.add_argument("--num-classes", type=int, default=None, help="Head classes (COCO ids: 90).")
    parser.add_argument("--max-targets", type=int, default=None, help="Padded targets per image.")
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument(
        "--pixel-dtype",
        default="auto",
        choices=["auto", "float32", "bfloat16"],
        help="Dtype images travel to the device in; 'auto' matches --compute-dtype, which the model "
        "casts its input to first, so the step sees identical pixels either way.",
    )
    parser.add_argument(
        "--eval-backend",
        default="auto",
        choices=["auto", "hotcoco", "pycocotools"],
        help="COCO evaluator; 'auto' uses hotcoco when it is installed.",
    )
    args = parser.parse_args(argv)
    # Restore unspecified training settings before constructing the optimizer or input stream.
    import sys
    tokens = sys.argv[1:] if argv is None else argv
    explicit = {token.split("=", 1)[0] for token in tokens if token.startswith("--")}
    resumed = checkpoint_metadata(args.resume) if args.resume else None
    if resumed:
        for name, value in resumed["run_config"].items():
            flag = "--" + name.replace("_", "-")
            if hasattr(args, name) and flag not in explicit and "--no-" + name.replace("_", "-") not in explicit:
                if name in ("keypoint_flip_pairs", "keypoint_oks_sigmas"):
                    value = ",".join(map(str, value))
                setattr(args, name, value)
        for name, value in resumed.get("runtime_config", {}).items():
            if "--" + name.replace("_", "-") not in explicit:
                setattr(args, name, value)
        if model_spec is None and not args.model_config:
            configuration = dict(resumed["model_metadata"]["model_spec"])
            for name in ("projector_scale", "out_feature_indexes", "num_keypoints_per_class"):
                configuration[name] = tuple(configuration[name])
            model_spec = LWDETRSpec(**configuration)
        provided_metadata = {**resumed["model_metadata"], **provided_metadata}
    if args.output_dir and "--out" not in explicit:
        args.out = str(Path(args.output_dir) / "metrics.json")
    if args.resume and (args.init_weights or args.init_params or initial_params is not None):
        parser.error("--resume cannot be combined with initial weights or parameters")
    if initial_params is not None and (args.init_weights or args.init_params):
        parser.error("provide initial_params or a checkpoint flag, not both")
    if model_spec is not None and args.model_config:
        parser.error("provide model_spec or --model-config, not both")
    for name in ("lr", "lr_encoder", "weight_decay", "lr_vit_layer_decay", "lr_component_decay",
                 "clip_max_norm", "warmup_epochs", "lr_min_factor", "ema_decay", "postprocess_trace_alpha"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.ema_decay > 1 or args.lr_min_factor > 1 or args.ema_tau < 0:
        parser.error("EMA decay and minimum LR factor must be at most one; EMA tau must be nonnegative")
    for name in ("epochs", "batch_size", "eval_every", "grad_accum_steps", "lr_drop"):

        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be positive")
    if args.max_targets is not None and args.max_targets <= 0:
        parser.error("--max-targets must be positive")
    if stop_after_epoch is not None and stop_after_epoch <= 0:
        parser.error("stop_after_epoch must be positive")
    if args.seed < 0 or args.data_seed < 0:
        parser.error("seeds must be nonnegative")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    for name in ("init_weights", "init_params"):
        value = getattr(args, name)
        if value and not Path(value).is_file():
            parser.error(f"--{name.replace('_', '-')} file does not exist: {value}")
    try:
        buckets = [int(size) for size in args.target_buckets.split(",") if size]
    except ValueError:
        parser.error("--target-buckets must contain comma-separated positive integers")
    if any(size <= 0 for size in buckets):
        parser.error("--target-buckets must contain positive integers")
    if args.data_parallel and args.matcher != "auction":
        parser.error("--data-parallel requires --matcher auction")
    save_path = Path(args.save_params) if args.save_params else Path(args.out).with_suffix(".params.npz")
    if save_path.suffix != ".npz":
        parser.error("--save-params must end in .npz")
    metadata_path = Path(str(save_path) + ".json")
    checkpoint_path = Path(args.output_dir or Path(args.out).parent) / "checkpoint_last.npz"
    outputs = [save_path.resolve(), metadata_path.resolve(), Path(args.out).resolve(), checkpoint_path.resolve()]
    if len(set(outputs)) != len(outputs):
        parser.error("--save-params, its JSON metadata, --out and the training checkpoint must be different files")

    jax.config.update("jax_default_matmul_precision", args.precision)
    if not args.no_compile_cache and not jax.config.jax_compilation_cache_dir:
        # Each step variant costs one to three minutes to compile; a later run with the same shapes
        # then loads it from disk instead.
        cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR") or str(Path.home() / ".cache/rfdetr_jax/xla")
        jax.config.update("jax_compilation_cache_dir", cache_dir)
    if args.attention is not None:
        from rfdetr_jax.models import layers

        layers.set_attention_implementation(args.attention)
    overrides = {
        name: value
        for name, value in (
            ("dataset_dir", args.dataset_dir),
            ("train_split", args.train_split),
            ("val_split", args.val_split),
            ("num_classes", args.num_classes),
            ("max_targets", args.max_targets),
        )
        if value is not None
    }
    overrides.update({name: getattr(args, name) for name in (
        "lr", "lr_encoder", "weight_decay", "lr_vit_layer_decay", "lr_component_decay", "clip_max_norm",
        "warmup_epochs", "lr_min_factor", "lr_scheduler", "lr_drop", "seed", "data_seed", "grad_accum_steps",
        "use_ema", "ema_decay", "ema_tau", "multi_scale", "expanded_scales", "scale_jitter",
        "freeze_encoder", "fused_optimizer", "postprocess_trace_alpha", "ema_first_update_copy",
        "keypoint_box_normalized_accumulation",
        "automatic_accumulation_normalization",
    )})
    metadata = provided_metadata
    configured_spec = model_spec
    if args.model_config:
        configured_spec, file_metadata = load_model_spec(args.model_config)
        metadata = {**file_metadata, **provided_metadata}
    task = args.task or (task_for_spec(configured_spec) if configured_spec is not None else "detection")
    if configured_spec is not None and task_for_spec(configured_spec) != task:
        parser.error("--task disagrees with --model-config")
    run = RunConfig(epochs=args.epochs, batch_size=args.batch_size, task=task, **overrides)
    if configured_spec is None and task != "detection":
        run = replace(run, patch_size=12, resolution=312 if task == "segmentation" else 576,
                      num_windows=1 if task == "segmentation" else 2, dec_layers=4,
                      num_queries=100, num_select=100,
                      num_classes=args.num_classes if args.num_classes is not None else 90)
    if configured_spec is not None:
        run = replace(run, **{name: getattr(configured_spec, name) for name in (
            "resolution", "num_classes", "dec_layers", "num_queries", "num_select", "patch_size",
            "num_windows", "hidden_dim", "out_feature_indexes", "mask_downsample_ratio", "encoder",
            "group_detr", "sa_nheads", "ca_nheads", "dec_n_points", "projector_scale", "gradient_checkpointing",
        )})
        if args.num_classes is not None:
            run = replace(run, num_classes=args.num_classes)
            configured_spec = replace(configured_spec, num_classes=args.num_classes)
    geometry_changes = {name: getattr(args, name) for name in ("resolution", "dec_layers", "num_queries", "num_select")
                        if getattr(args, name) is not None}
    run = replace(run, **geometry_changes)
    if configured_spec is not None:
        configured_spec = replace(configured_spec, **geometry_changes)
    if args.mask_downsample_ratio is not None and args.mask_downsample_ratio <= 0:
        parser.error("--mask-downsample-ratio must be positive")
    if args.grad_checkpoint != "config":
        run = replace(run, gradient_checkpointing=args.grad_checkpoint == "on")
    train_paths = (args.train_images or run.train_paths()[0], args.train_annotations or run.train_paths()[1])
    val_paths = (args.val_images or run.val_paths()[0], args.val_annotations or run.val_paths()[1])
    train_dataset = CocoDetection(*train_paths, segmentation=task == "segmentation", keypoints=task == "keypoints")
    val_dataset = CocoDetection(*val_paths, segmentation=task == "segmentation", keypoints=task == "keypoints")
    if list(zip(train_dataset.category_ids, train_dataset.category_names, strict=True)) != list(
        zip(val_dataset.category_ids, val_dataset.category_names, strict=True)
    ):
        parser.error("training and validation category ids and names must match")
    try:
        schema = tuple(int(item) for item in args.keypoint_schema.split(",")) if args.keypoint_schema else (
            configured_spec.num_keypoints_per_class if configured_spec is not None else ()
        )
        flip_pairs = tuple(int(item) for item in args.keypoint_flip_pairs.split(",") if item) or tuple(
            metadata.get("keypoint_flip_pairs", ())
        )
        sigmas = tuple(float(item) for item in args.keypoint_oks_sigmas.split(",") if item) or tuple(
            metadata.get("keypoint_oks_sigmas", ())
        )
        if task == "keypoints":
            if not schema:
                counts = keypoint_category_counts(train_dataset)
                schema = tuple(counts[category] for category in sorted(counts) if counts[category])
            if any(count < 0 for count in schema) or not any(schema):
                raise ValueError("keypoint schema needs nonnegative counts and an active slot")
            train_mapping = keypoint_category_mapping(train_dataset, schema)
            val_mapping = keypoint_category_mapping(val_dataset, schema)
            if train_mapping != val_mapping:
                raise ValueError("training and validation keypoint schemas must match")
            if "category_to_label" in metadata:
                saved_mapping = {int(key): int(value) for key, value in metadata["category_to_label"].items()}
                if saved_mapping != train_mapping:
                    raise ValueError("checkpoint category_to_label does not match the dataset schema")
            train_dataset.set_category_mapping(train_mapping)
            val_dataset.set_category_mapping(val_mapping)
            train_dataset.num_keypoints = val_dataset.num_keypoints = max(schema)
            if len(flip_pairs) % 2 or any(index < 0 or index >= max(schema) for index in flip_pairs):
                raise ValueError("keypoint flip pairs must be in-range left/right index pairs")
            if len(set(flip_pairs)) != len(flip_pairs):
                raise ValueError("keypoint flip pairs cannot repeat an index")
            if sigmas and (len(sigmas) < max(schema) or not np.isfinite(sigmas).all() or min(sigmas) <= 0):
                raise ValueError("OKS sigmas must be positive finite values covering every keypoint")
        elif schema or flip_pairs or sigmas:
            raise ValueError("keypoint schema, flip pairs and sigmas require --task keypoints")
        elif "category_to_label" in metadata:
            saved_mapping = {int(key): int(value) for key, value in metadata["category_to_label"].items()}
            train_dataset.set_category_mapping(saved_mapping)
            val_dataset.set_category_mapping(saved_mapping)
    except ValueError as exc:
        parser.error(str(exc))
    run = replace(run, num_keypoints_per_class=schema, keypoint_flip_pairs=flip_pairs,
                  keypoint_oks_sigmas=sigmas,
                  mask_downsample_ratio=args.mask_downsample_ratio or run.mask_downsample_ratio)
    spec = configured_spec or spec_for(run)
    spec = replace(spec, num_keypoints_per_class=schema, mask_downsample_ratio=run.mask_downsample_ratio)
    if configured_spec is None and task == "keypoints":
        spec = replace(spec, dual_projector=True, dual_projector_kp_only=True)
    if args.grad_checkpoint != "config":
        spec = replace(spec, gradient_checkpointing=run.gradient_checkpointing)
    head_dtype = args.head_dtype or ("activation" if jax.default_backend() == "gpu" else "float32")
    spec = replace(spec, deform_sampler=args.sampler, compute_dtype=args.compute_dtype, head_dtype=head_dtype)
    requested_precision = metadata.get("training_precision", {})
    run = replace(run, requested_amp_dtype=requested_precision.get("requested"),
                  effective_amp_dtype=requested_precision.get("effective"),
                  resolved_compute_dtype=spec.compute_dtype, loss_scaling=spec.compute_dtype == "float16")
    try:
        validate_model_spec(spec)
    except ValueError as exc:
        parser.error(str(exc))
    if category_to_label is not None:
        if task == "keypoints" and category_to_label != train_dataset.category_to_label:
            parser.error("explicit category mapping disagrees with the keypoint schema")
        train_dataset.set_category_mapping(category_to_label)
        val_dataset.set_category_mapping(category_to_label)
    if max(train_dataset.category_to_label.values()) >= spec.detector_classes:
        parser.error("--num-classes must be at least the largest raw COCO category id or mapped model label")
    # Augmentation can only remove targets, so the raw count bounds every batch; check it now rather
    # than at whichever step a crop happens to keep every annotation.
    if train_dataset.max_targets_per_image() > run.max_targets:
        parser.error(f"--max-targets {run.max_targets} is below the {train_dataset.max_targets_per_image()} "
                     "annotations of the largest training image; raise it to avoid discarding annotations")
    model = LWDETR(spec=spec)
    pixel_dtype = jnp.dtype(("float32" if run.loss_scaling else args.compute_dtype)
                            if args.pixel_dtype == "auto" else args.pixel_dtype)
    eval_batch_size = args.eval_batch_size or run.batch_size
    batches_per_epoch = len(train_dataset) // run.batch_size
    steps_per_epoch = (batches_per_epoch + run.grad_accum_steps - 1) // run.grad_accum_steps
    if steps_per_epoch == 0:
        parser.error("training split must contain at least --batch-size images")
    if len(val_dataset) == 0:
        parser.error("validation split must contain at least one image")
    total_steps = steps_per_epoch * run.epochs
    dataset_fingerprints = [hashlib.sha256(Path(path).read_bytes()).hexdigest()
                            for path in (train_paths[1], val_paths[1])]

    started = time.time()
    if args.host_init:
        # Tracing the initialiser on the accelerator compiles a full-model graph that is never
        # reused; the host produces the same tree without paying for that.
        with jax.default_device(jax.devices("cpu")[0]):
            params = init_params(model, spec, jax.random.key(run.seed))
        params = jax.device_put(jax.tree.map(np.asarray, params))
    else:
        params = init_params(model, spec, jax.random.key(run.seed))
    if initial_params is not None:
        actual, expected = flatten_params(initial_params), flatten_params(params)
        if actual.keys() != expected.keys() or any(actual[name].shape != expected[name].shape for name in expected):
            parser.error("initial_params must exactly match the model parameter names and shapes")
        if any(not np.isfinite(np.asarray(value)).all() for value in actual.values()):
            parser.error("initial_params contains nonfinite values")
        params = jax.tree.map(lambda value: jax.device_put(np.asarray(value).copy()), initial_params)
    if args.init_weights:
        with np.load(args.init_weights) as data:
            state_dict = {key: data[key] for key in data.files}
        params, missing, unexpected = convert_state_dict(state_dict, params, strict=True)
        if missing or unexpected:
            raise SystemExit(f"weight conversion incomplete: missing={missing[:3]} unexpected={unexpected[:3]}")
        print(f"loaded initial weights from {args.init_weights}", flush=True)
    if args.init_params:
        try:
            params = load_initial_params(args.init_params, params)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"loaded initial parameters from {args.init_params}", flush=True)
    print(f"init: {time.time() - started:.1f}s", flush=True)

    optimizer_spec = OptimizerSpec(
        lr=run.lr,
        lr_encoder=0. if run.freeze_encoder else run.lr_encoder,
        weight_decay=run.weight_decay,
        lr_vit_layer_decay=run.lr_vit_layer_decay,
        lr_component_decay=run.lr_component_decay,
        clip_max_norm=run.clip_max_norm,
    )
    schedule = managed_schedule(
        run.lr_scheduler,
        total_steps,
        steps_per_epoch,
        int(run.warmup_epochs * steps_per_epoch),
        run.lr_drop,
        run.lr_min_factor,
    )
    optimizer = build_optimizer(params, optimizer_spec, spec.out_feature_indexes,
                                None if run.loss_scaling else schedule, fused=run.fused_optimizer,
                                lite_refpoint_refine=spec.lite_refpoint_refine)
    state = create_train_state(params, optimizer, use_ema=run.use_ema, loss_scaling=run.loss_scaling)
    start_epoch = 0
    key = jax.random.key(run.seed)
    if resumed:
        for name in ("lr", "lr_encoder", "weight_decay", "lr_vit_layer_decay", "lr_component_decay",
                     "clip_max_norm", "warmup_epochs", "lr_min_factor", "lr_scheduler", "lr_drop", "seed",
                     "data_seed", "grad_accum_steps", "use_ema", "ema_decay", "ema_tau", "epochs", "batch_size",
                     "multi_scale", "expanded_scales", "scale_jitter", "freeze_encoder", "fused_optimizer"):
            if getattr(run, name) != resumed["run_config"][name]:
                parser.error(f"resume must preserve {name}; start a new fine-tuning run to change its schedule")
        if run.postprocess_trace_alpha != resumed["run_config"].get("postprocess_trace_alpha", .2):
            parser.error("resume must preserve postprocess_trace_alpha")
        if run.ema_first_update_copy != resumed["run_config"].get("ema_first_update_copy", False):
            parser.error("resume must preserve ema_first_update_copy")
        if run.keypoint_box_normalized_accumulation != resumed["run_config"].get(
                "keypoint_box_normalized_accumulation", False):
            parser.error("resume must preserve keypoint_box_normalized_accumulation")
        if run.automatic_accumulation_normalization != resumed["run_config"].get(
                "automatic_accumulation_normalization", False):
            parser.error("resume must preserve automatic_accumulation_normalization")
        if json.loads(json.dumps(spec.__dict__)) != resumed["model_metadata"]["model_spec"]:
            parser.error("resume must preserve the saved model specification")
        saved_categories = {int(k): int(v) for k, v in resumed["model_metadata"]["category_to_label"].items()}
        if saved_categories != train_dataset.category_to_label or batches_per_epoch != resumed["batches_per_epoch"]:
            parser.error("resume must preserve the dataset category mapping and number of batches")
        if dataset_fingerprints != resumed["dataset_fingerprints"]:
            parser.error("resume requires the original dataset annotations")
        for name, value in resumed.get("runtime_config", {}).items():
            if name in ("matcher", "precision", "pixel_dtype") and getattr(args, name) != value:
                parser.error(f"resume must preserve {name}")
        state, key, _ = load_training_checkpoint(args.resume, state)
        start_epoch = resumed["completed_epochs"]
    batch_sharding = None
    if args.data_parallel:
        state, batch_sharding = data_parallel(state)
        if run.batch_size % jax.device_count():
            raise SystemExit(
                f"--data-parallel needs a batch size divisible by {jax.device_count()} devices, "
                f"got {run.batch_size}"
            )

    if criterion_config is None and resumed and "criterion_config" in resumed:
        restored_criterion = dict(resumed["criterion_config"])
        restored_criterion["num_keypoints_per_class"] = tuple(restored_criterion["num_keypoints_per_class"])
        criterion_config = CriterionConfig(**restored_criterion)
    criterion_config = criterion_config or CriterionConfig(
        num_classes=spec.detector_classes, group_detr=spec.group_detr, matcher=args.matcher,
        segmentation_head=spec.segmentation_head, use_grouppose_keypoints=spec.use_grouppose_keypoints,
        num_keypoints_per_class=spec.num_keypoints_per_class,
        cls_loss_coef=2.0 if task == "keypoints" else 1.0,
        keypoint_l1_loss_coef=1.0, keypoint_findable_loss_coef=1.0,
        keypoint_visible_loss_coef=1.0, keypoint_nll_loss_coef=1.0,
    )
    criterion_contract = json.loads(json.dumps(criterion_config.__dict__))
    if resumed and criterion_contract != resumed.get("criterion_config", criterion_contract):
        parser.error("resume must preserve the loss and matching configuration")
    for name, expected in (("num_classes", spec.detector_classes), ("group_detr", spec.group_detr),
                           ("segmentation_head", spec.segmentation_head),
                           ("use_grouppose_keypoints", spec.use_grouppose_keypoints),
                           ("num_keypoints_per_class", spec.num_keypoints_per_class)):
        if getattr(criterion_config, name) != expected:
            parser.error(f"criterion {name} disagrees with the model specification")
    if args.data_parallel and criterion_config.matcher != "auction":
        parser.error("data parallel training requires the auction criterion matcher")
    settings = TrainSettings(
        epochs=run.epochs,
        batch_size=run.batch_size,
        max_targets=run.max_targets,
        use_ema=run.use_ema,
        ema_decay=run.ema_decay,
        ema_tau=run.ema_tau,
        ema_first_update_copy=run.ema_first_update_copy,
        keypoint_box_normalized_accumulation=run.keypoint_box_normalized_accumulation,
        automatic_accumulation_normalization=run.automatic_accumulation_normalization,
        lr_scheduler=run.lr_scheduler,
        lr_drop=run.lr_drop,
        seed=run.seed,
        grad_accum_steps=run.grad_accum_steps,
        freeze_encoder=run.freeze_encoder,
        loss_scaling=run.loss_scaling,
    )
    weights = criterion_config.weight_dict(spec.dec_layers)
    explicit = batch_sharding if args.dp_mode == "shard_map" else None
    step_fn = make_train_step(model, optimizer, criterion_config, weights, settings, batch_sharding=explicit,
                              optimizer_schedule=schedule if run.loss_scaling else None)
    if task == "detection":
        predict = make_predict_step(model, run.num_select, batch_sharding)
    else:
        @jax.jit
        def predict(parameters, pixels, original_sizes):
            del original_sizes
            outputs = model.apply({"params": parameters}, pixels, None, train=False)
            return {key: value for key, value in outputs.items() if key.startswith("pred_")}
    compiled_steps = {}
    if not args.no_precompile and run.grad_accum_steps == 1 and not run.multi_scale:
        widths = sorted({size for size in buckets if size <= run.max_targets} | {run.max_targets})
        compiled_steps = precompile_steps(
            step_fn, state, jax.random.key(run.seed), widths, run.batch_size, run.resolution, pixel_dtype,
            batch_sharding, task=task, keypoint_count=train_dataset.num_keypoints,
        )
    if batch_sharding is not None and eval_batch_size % jax.device_count():
        raise SystemExit(f"--eval-batch-size must be divisible by {jax.device_count()} devices")

    step_inputs = ("pixel_values", "labels", "boxes", "target_mask")
    if task == "segmentation":
        step_inputs += ("masks",)
    elif task == "keypoints":
        step_inputs += ("keypoints",)
    from rfdetr_jax.training.augmentation import compute_multi_scale_scales, resize_training_batch, scale_for_step
    scales = compute_multi_scale_scales(run.resolution, run.expanded_scales, spec.patch_size, spec.num_windows) \
        if run.multi_scale else [run.resolution]
    batches = device_batches(
        iterate_batches(
            train_dataset,
            max(scales),
            run.batch_size,
            run.max_targets,
            train=True,
            seed=run.data_seed + start_epoch,
            epochs=max(0, run.epochs - start_epoch),
            workers=args.workers,
            pixel_dtype=pixel_dtype,
            worker_mode=args.worker_mode,
            target_buckets=buckets or None,
            keypoint_flip_pairs=run.keypoint_flip_pairs,
            allow_horizontal_flip=task != "keypoints" or bool(run.keypoint_flip_pairs),
            scale_jitter=run.scale_jitter,
        ),
        step_inputs,
        batch_sharding,
    )

    log: dict = {"framework": "jax", "config": run.__dict__,
                 "epochs": list(resumed.get("epoch_log", [])) if resumed else [], "first_steps": []}
    model_metadata = {**metadata, "model_spec": spec.__dict__, "task": task,
                      "postprocess_trace_alpha": run.postprocess_trace_alpha,
                      "category_ids": train_dataset.category_ids, "category_names": train_dataset.category_names,
                      "category_to_label": train_dataset.category_to_label,
                      "keypoint_flip_pairs": run.keypoint_flip_pairs, "keypoint_oks_sigmas": run.keypoint_oks_sigmas}
    try:
        for epoch in range(start_epoch, run.epochs):
            epoch_started = time.time()
            totals: dict = {}
            count = 0
            for step_index in range(steps_per_epoch):
                group_size = min(run.grad_accum_steps, batches_per_epoch - step_index * run.grad_accum_steps)
                microbatches = [{name: batch[name] for name in step_inputs}
                                for batch in (next(batches) for _ in range(group_size))]
                if run.multi_scale:
                    scale = scale_for_step(scales, epoch * steps_per_epoch + step_index)
                    microbatches = [resize_training_batch(batch, scale) for batch in microbatches]
                device_batch = stack_microbatches(microbatches) if run.grad_accum_steps > 1 else microbatches[0]
                key, dropout_key = jax.random.split(key)
                step = (compiled_steps.get(device_batch["labels"].shape[1], step_fn)
                        if run.grad_accum_steps == 1 else step_fn)
                state, metrics = step(state, device_batch, dropout_key)
                if epoch == 0 and len(log["first_steps"]) < args.log_steps:
                    log["first_steps"].append({name: float(value) for name, value in metrics.items()})
                # Accumulate on device: reading a metric per step would synchronize every iteration.
                weighted = {name: value * group_size for name, value in metrics.items()}
                totals = weighted if not totals else {name: totals[name] + value for name, value in weighted.items()}
                count += group_size
                if step_index % 20 == 0:
                    print(f"  epoch {epoch} step {step_index}/{steps_per_epoch}", flush=True)

            # Reading the losses waits for the last queued step; only then has the epoch finished on
            # the device, so the clock stops after it rather than when the host ran out of dispatches.
            losses = {name: float(value) / max(1, count) for name, value in totals.items()}
            record = {
                "epoch": epoch,
                "seconds": time.time() - epoch_started,
                "steps": steps_per_epoch,
                "micro_batches": count,
                "images_per_second": count * run.batch_size / (time.time() - epoch_started),
                "losses": losses,
            }
            if (epoch + 1) % args.eval_every == 0 or epoch == run.epochs - 1:
                eval_started = time.time()
                scoring = {
                    "run": run,
                    "dataset": val_dataset,
                    "annotation_file": val_paths[1],
                    "batch_size": eval_batch_size,
                    "workers": args.workers,
                    "worker_mode": args.worker_mode,
                    "batch_sharding": batch_sharding,
                    "pixel_dtype": pixel_dtype,
                    "backend": args.eval_backend,
                    "record_evaluator": evaluation_record_evaluator,
                }
                # With EMA on, the EMA weights are the ones a run keeps, so only they are scored unless
                # --eval-raw asks for both; official RF-DETR dropped the second evaluation the same way.
                if state.ema_params is None or args.eval_raw:
                    record["val"] = evaluate(predict, state.params, **scoring)
                if state.ema_params is not None:
                    record["val_ema"] = evaluate(predict, state.ema_params, **scoring)
                record["eval_seconds"] = time.time() - eval_started
            log["epochs"].append(record)
            save_training_checkpoint(checkpoint_path, state, key, {
                "model_metadata": model_metadata, "run_config": run.__dict__, "completed_epochs": epoch + 1,
                "batches_per_epoch": batches_per_epoch, "epoch_log": log["epochs"],
                "dataset_fingerprints": dataset_fingerprints, "criterion_config": criterion_contract,
                "runtime_config": {name: getattr(args, name) for name in (
                    "matcher", "precision", "pixel_dtype", "compute_dtype", "head_dtype", "sampler",
                    "train_images", "train_annotations", "val_images", "val_annotations",
                )},
            })
            callback_stop = on_epoch_end(record, state) if on_epoch_end is not None else False
            print(
                f"epoch {epoch}: loss={record['losses']['loss']:.4f} "
                f"mAP={record.get('val_ema', record.get('val', {})).get('mAP', float('nan')):.4f} "
                f"({record['seconds']:.1f}s, {record['images_per_second']:.1f} img/s)",
                flush=True,
            )
            if callback_stop is True or (stop_after_epoch is not None and epoch + 1 >= stop_after_epoch):
                break
    finally:
        batches.close()

    if resumed and start_epoch >= run.epochs:
        save_training_checkpoint(checkpoint_path, state, key, resumed)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    inference_params = state.ema_params if state.ema_params is not None else state.params
    save_params_npz(save_path, inference_params)
    model_metadata.update({
        "model_spec": spec.__dict__,
        "weights": "ema" if state.ema_params is not None else "raw",
        "step": int(state.step),
        "category_ids": train_dataset.category_ids,
        "category_names": train_dataset.category_names,
        "category_to_label": train_dataset.category_to_label,
        "task": task,
        "keypoint_flip_pairs": run.keypoint_flip_pairs,
        "keypoint_oks_sigmas": run.keypoint_oks_sigmas,
    })
    metadata_path.write_text(json.dumps(model_metadata, indent=2))
    log["saved_params"] = str(save_path)
    log["model_metadata"] = str(metadata_path)
    log["training_checkpoint"] = str(checkpoint_path)
    print(f"wrote inference parameters {save_path} and resumable checkpoint {checkpoint_path}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(log, indent=2, default=str))
    print(f"wrote {args.out}")
    return TrainingResult(inference_params, state, model_metadata, log,
                          {"params": str(save_path), "metadata": str(metadata_path),
                           "checkpoint": str(checkpoint_path), "log": str(args.out)})


def main(argv: list[str] | None = None) -> TrainingResult:
    """CLI-compatible entry point; Python callers should use run_training."""
    return run_training(argv)


if __name__ == "__main__":
    main()
