"""Measure the RF-DETR training step across TPU configurations, one variable at a time.

Parameters are initialised once on the host and re-uploaded for every configuration, because the
step donates its state buffers. Each configuration is compiled, warmed and then timed over
``--steps`` executions with dispatch forced to completion, and the median is reported.
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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def _config(
    sampler="gather",
    precision="default",
    grad_checkpoint=False,
    matcher="auction",
    batch=2,
    parallel=False,
    compute_dtype="float32",
):
    """Build one sweep entry; defaults are the TPU-oriented settings."""
    return {
        "precision": precision,
        "grad_checkpoint": grad_checkpoint,
        "matcher": matcher,
        "sampler": sampler,
        "batch": batch,
        "parallel": parallel,
        "compute_dtype": compute_dtype,
    }


CONFIGS = {
    # The port's own settings, so the sweep carries its own reference point.
    "reference": _config(precision="highest", grad_checkpoint=True, matcher="hungarian"),
    "no_remat": _config(precision="highest", matcher="hungarian"),
    "auction": _config(precision="highest"),
    "matmul": _config(precision="highest", sampler="matmul"),
    "bf16": _config(sampler="matmul"),
    "batch8": _config(sampler="matmul", batch=8),
    "batch8_parallel": _config(sampler="matmul", batch=8, parallel=True),
    "batch32_parallel": _config(sampler="matmul", batch=32, parallel=True),
    "batch64_parallel": _config(sampler="matmul", batch=64, parallel=True),
    "bf16_activations": _config(sampler="matmul", compute_dtype="bfloat16"),
    "best_parallel": _config(sampler="matmul", batch=32, parallel=True, compute_dtype="bfloat16"),
    # Per-chip batch is the remaining lever once the step has no pathology left in it.
    "batch64_best": _config(sampler="matmul", batch=64, parallel=True, compute_dtype="bfloat16"),
    "batch64_auto": _config(sampler="auto", batch=64, parallel=True, compute_dtype="bfloat16"),
    "batch128_best": _config(sampler="matmul", batch=128, parallel=True, compute_dtype="bfloat16"),
}


def parse_args() -> argparse.Namespace:
    """Return the sweep configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/sweep.json")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--configs", default=",".join(CONFIGS), help="Comma-separated subset to run.")
    return parser.parse_args()


def make_batch(batch: int, resolution: int, num_classes: int, max_targets: int) -> dict[str, np.ndarray]:
    """Build one fixed-shape synthetic batch with four real targets per image."""
    rng = np.random.default_rng(0)
    mask = np.zeros((batch, max_targets), dtype=bool)
    mask[:, :4] = True
    return {
        "pixel_values": rng.standard_normal((batch, 3, resolution, resolution), dtype=np.float32),
        "labels": rng.integers(0, num_classes, size=(batch, max_targets)).astype(np.int32),
        "boxes": np.concatenate(
            [
                rng.uniform(0.25, 0.75, size=(batch, max_targets, 2)),
                rng.uniform(0.05, 0.30, size=(batch, max_targets, 2)),
            ],
            axis=-1,
        ).astype(np.float32),
        "target_mask": mask,
    }


def main() -> None:
    """Run every requested configuration and write the records as JSON."""
    args = parse_args()
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

    base_run = RunConfig(resolution=args.resolution)
    base_spec = spec_for(base_run)

    jax.config.update("jax_default_matmul_precision", "highest")
    started = time.perf_counter()
    with jax.default_device(jax.devices("cpu")[0]):
        host_params = jax.tree.map(np.asarray, init_params(LWDETR(spec=base_spec), base_spec, jax.random.key(0)))
    init_seconds = time.perf_counter() - started
    print(f"host init: {init_seconds:.1f}s", flush=True)

    devices = jax.devices()
    records = []
    for name in args.configs.split(","):
        if not name:
            continue
        cfg = CONFIGS[name]
        batch = cfg["batch"]
        parallel = cfg.get("parallel", False)
        if parallel and batch % len(devices):
            raise SystemExit(f"{name}: batch {batch} is not divisible by {len(devices)} devices")

        jax.config.update("jax_default_matmul_precision", cfg["precision"])
        run = RunConfig(
            resolution=args.resolution, batch_size=batch, gradient_checkpointing=cfg["grad_checkpoint"]
        )
        spec = dataclasses.replace(
            spec_for(run), deform_sampler=cfg["sampler"], compute_dtype=cfg["compute_dtype"]
        )
        model = LWDETR(spec=spec)

        params = jax.device_put(host_params)
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
        state = create_train_state(params, optimizer, use_ema=True)
        sharding = None
        if parallel:
            state, sharding = data_parallel(state, devices)

        host_batch = make_batch(batch, args.resolution, run.num_classes, run.max_targets)
        device_batch = {
            key: (jnp.asarray(value) if sharding is None else jax.device_put(value, sharding))
            for key, value in host_batch.items()
        }

        criterion = CriterionConfig(
            num_classes=spec.detector_classes, group_detr=spec.group_detr, matcher=cfg["matcher"]
        )
        settings = TrainSettings(batch_size=batch, max_targets=run.max_targets)
        step_fn = make_train_step(model, optimizer, criterion, criterion.weight_dict(spec.dec_layers), settings)
        dropout_key = jax.random.key(1)

        compile_started = time.perf_counter()
        state, metrics = step_fn(state, device_batch, dropout_key)
        jax.block_until_ready((state, metrics))
        compile_seconds = time.perf_counter() - compile_started

        for _ in range(args.warmup):
            state, metrics = step_fn(state, device_batch, dropout_key)
            jax.block_until_ready((state, metrics))
        samples = []
        for _ in range(args.steps):
            began = time.perf_counter()
            state, metrics = step_fn(state, device_batch, dropout_key)
            jax.block_until_ready((state, metrics))
            samples.append(time.perf_counter() - began)

        forward = make_eval_step(model)
        eval_started = time.perf_counter()
        jax.block_until_ready(forward(state.params, device_batch["pixel_values"]))
        eval_compile = time.perf_counter() - eval_started
        for _ in range(args.warmup):
            jax.block_until_ready(forward(state.params, device_batch["pixel_values"]))
        eval_samples = []
        for _ in range(args.steps):
            began = time.perf_counter()
            jax.block_until_ready(forward(state.params, device_batch["pixel_values"]))
            eval_samples.append(time.perf_counter() - began)

        record = {
            "config": name,
            **cfg,
            "devices": len(devices) if parallel else 1,
            "compile_seconds": compile_seconds,
            "eval_compile_seconds": eval_compile,
            "train_step_ms": statistics.median(samples) * 1e3,
            "train_step_min_ms": min(samples) * 1e3,
            "train_step_max_ms": max(samples) * 1e3,
            "train_images_per_second": batch / statistics.median(samples),
            "eval_step_ms": statistics.median(eval_samples) * 1e3,
            "eval_images_per_second": batch / statistics.median(eval_samples),
            "loss": float(metrics["loss"]),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        del state, step_fn, forward

    payload = {
        "device_kind": devices[0].device_kind,
        "device_count": len(devices),
        "jax_version": jax.__version__,
        "resolution": args.resolution,
        "host_init_seconds": init_seconds,
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
