"""Benchmark one RF-DETR training step and one inference forward on the current backend.

Separates runtime init, host->device transfer, lowering/compile and warm steady execution, as the
TPU measurement protocol requires: dispatch is asynchronous, so every timed region ends with
``block_until_ready`` and inputs are placed before the clock starts.

The knobs mirror the candidate TPU optimizations one at a time so an A/B changes a single variable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    """Return the benchmark configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline", help="Name recorded in the result JSON.")
    parser.add_argument("--out", default=None, help="Where to write the raw result JSON.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--group-detr", type=int, default=13)
    parser.add_argument("--dec-layers", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--max-targets", type=int, default=20)
    parser.add_argument("--precision", default="highest", choices=["highest", "float32", "default", "bfloat16"])
    parser.add_argument("--grad-checkpoint", default="on", choices=["on", "off"])
    parser.add_argument("--matcher", default="hungarian", choices=["hungarian", "auction"])
    parser.add_argument("--sampler", default="gather", choices=["gather", "matmul", "auto"])
    parser.add_argument("--compute-dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--data-parallel", action="store_true", help="Shard the batch over every device.")
    parser.add_argument("--host-init", action="store_true", help="Initialise parameters on the host.")
    parser.add_argument("--steps", type=int, default=12, help="Timed steady-state steps.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def synthetic_batch(rng: np.random.Generator, args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Build one fixed-shape batch with a deterministic target count per image."""
    batch = args.batch_size
    pixel_values = rng.standard_normal((batch, 3, args.resolution, args.resolution), dtype=np.float32)
    labels = rng.integers(0, args.num_classes, size=(batch, args.max_targets)).astype(np.int32)
    centres = rng.uniform(0.25, 0.75, size=(batch, args.max_targets, 2)).astype(np.float32)
    sizes = rng.uniform(0.05, 0.3, size=(batch, args.max_targets, 2)).astype(np.float32)
    boxes = np.concatenate([centres, sizes], axis=-1)
    mask = np.zeros((batch, args.max_targets), dtype=bool)
    mask[:, :4] = True  # four real targets per image, the rest padding
    return {"pixel_values": pixel_values, "labels": labels, "boxes": boxes, "target_mask": mask}


def timed(fn, repeats: int) -> list[float]:
    """Run ``fn`` ``repeats`` times, returning per-call seconds with device work forced to complete."""
    import jax

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - started)
    return samples


def main() -> None:
    """Measure compile and steady-state cost for the train step and the eval forward."""
    args = parse_args()

    runtime_started = time.perf_counter()
    import jax
    import jax.numpy as jnp

    devices = jax.devices()
    runtime_seconds = time.perf_counter() - runtime_started

    jax.config.update("jax_default_matmul_precision", args.precision)

    from shared_run_config import RunConfig
    from train_jax import spec_for

    from rfdetr_jax.models.criterion import CriterionConfig
    from rfdetr_jax.models.lwdetr import LWDETR
    from rfdetr_jax.training.optimizer import OptimizerSpec, build_optimizer
    from rfdetr_jax.training.trainer import (
        TrainSettings,
        create_train_state,
        data_parallel,
        init_params,
        make_eval_step,
        make_train_step,
    )

    run = RunConfig(
        resolution=args.resolution,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        max_targets=args.max_targets,
        dec_layers=args.dec_layers,
        group_detr=args.group_detr,
        gradient_checkpointing=args.grad_checkpoint == "on",
    )
    spec = dataclasses.replace(
        spec_for(run), deform_sampler=args.sampler, compute_dtype=args.compute_dtype
    )
    model = LWDETR(spec=spec)

    init_started = time.perf_counter()
    if args.host_init:
        with jax.default_device(jax.devices("cpu")[0]):
            params = init_params(model, spec, jax.random.key(0))
        params = jax.device_put(jax.tree.map(np.asarray, params))
    else:
        params = init_params(model, spec, jax.random.key(0))
    jax.block_until_ready(params)
    init_seconds = time.perf_counter() - init_started
    param_count = int(sum(np.prod(leaf.shape) for leaf in jax.tree.leaves(params)))

    optimizer = build_optimizer(
        params,
        OptimizerSpec(
            lr=run.lr,
            lr_encoder=run.lr_encoder,
            weight_decay=run.weight_decay,
            lr_vit_layer_decay=run.lr_vit_layer_decay,
            lr_component_decay=run.lr_component_decay,
            clip_max_norm=run.clip_max_norm,
        ),
        spec.out_feature_indexes,
    )
    state = create_train_state(params, optimizer, run.use_ema)
    criterion = CriterionConfig(
        num_classes=spec.detector_classes, group_detr=spec.group_detr, matcher=args.matcher
    )
    settings = TrainSettings(
        batch_size=run.batch_size,
        max_targets=run.max_targets,
        use_ema=run.use_ema,
        ema_decay=run.ema_decay,
        ema_tau=run.ema_tau,
    )
    weights = criterion.weight_dict(spec.dec_layers)

    rng = np.random.default_rng(0)
    host_batch = synthetic_batch(rng, args)

    result: dict = {
        "label": args.label,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "device_kind": devices[0].device_kind,
        "device_count": len(devices),
        "config": vars(args),
        "runtime_init_seconds": runtime_seconds,
        "param_init_seconds": init_seconds,
        "param_count": param_count,
    }

    batch_sharding = None
    if args.data_parallel:
        state, batch_sharding = data_parallel(state)

    def place(values: dict) -> dict:
        """Move one host batch onto the device (or across every device when sharding)."""
        if batch_sharding is None:
            return {name: jnp.asarray(value) for name, value in values.items()}
        return {name: jax.device_put(value, batch_sharding) for name, value in values.items()}

    transfer = timed(lambda: place(host_batch), 5)
    result["transfer_seconds"] = {"median": statistics.median(transfer), "samples": transfer}
    device_batch = place(host_batch)

    if not args.skip_train:
        step_fn = make_train_step(model, optimizer, criterion, weights, settings)
        key = jax.random.key(0)
        key, dropout_key = jax.random.split(key)

        compile_started = time.perf_counter()
        compiled = step_fn.lower(state, device_batch, dropout_key).compile()
        result["train_compile_seconds"] = time.perf_counter() - compile_started
        try:
            analysis = compiled.cost_analysis()
            entry = analysis[0] if isinstance(analysis, list) else analysis
            result["train_flops"] = float(entry.get("flops", 0.0))
            result["train_bytes_accessed"] = float(entry.get("bytes accessed", 0.0))
        except Exception as error:  # pragma: no cover - cost analysis is best effort
            result["train_flops"] = f"unavailable: {error}"

        for _ in range(args.warmup):
            state, metrics = step_fn(state, device_batch, dropout_key)
            jax.block_until_ready((state, metrics))

        samples = []
        for _ in range(args.steps):
            started = time.perf_counter()
            state, metrics = step_fn(state, device_batch, dropout_key)
            jax.block_until_ready((state, metrics))
            samples.append(time.perf_counter() - started)
        result["steady_state_note"] = "median of warm steps, dispatch forced to completion each step"
        result["train_step_seconds"] = {
            "median": statistics.median(samples),
            "min": min(samples),
            "max": max(samples),
            "samples": samples,
        }
        result["train_images_per_second"] = args.batch_size / statistics.median(samples)
        result["train_loss"] = float(metrics["loss"])

    if not args.skip_eval:
        forward = make_eval_step(model)
        eval_params = state.params
        compile_started = time.perf_counter()
        outputs = forward(eval_params, device_batch["pixel_values"])
        jax.block_until_ready(outputs)
        result["eval_compile_plus_first_seconds"] = time.perf_counter() - compile_started
        for _ in range(args.warmup):
            jax.block_until_ready(forward(eval_params, device_batch["pixel_values"]))
        samples = timed(lambda: forward(eval_params, device_batch["pixel_values"]), args.steps)
        result["eval_step_seconds"] = {
            "median": statistics.median(samples),
            "min": min(samples),
            "max": max(samples),
            "samples": samples,
        }
        result["eval_images_per_second"] = args.batch_size / statistics.median(samples)

    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
