"""Attribute the RF-DETR train step's TPU cost to one candidate cause at a time.

Builds the parameter tree once, then re-jits the step under each variant so an A/B changes exactly
one variable. Every timed region ends with ``block_until_ready``; the reported number is the median
of ``--steps`` warm executions after ``--warmup`` discarded ones.
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    """Return the ablation configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/ablation.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--variants",
        default="base,precision,gradckpt,matcher,forward,combined",
        help="Comma-separated subset of the variants to run.",
    )
    return parser.parse_args()


def dummy_match(cost, sizes, group_detr):
    """A device-only stand-in for the matcher, used to bound the host callback's cost.

    Assigns target ``t`` of group ``g`` to query ``g * per_group + t``. The result has the same
    shapes, dtypes and validity pattern as :func:`hungarian_match`, so the losses downstream do the
    same work; only the assignment is different.
    """
    batch, num_queries, max_targets = cost.shape
    per_group = num_queries // group_detr
    offsets = (jnp.arange(group_detr) * per_group)[:, None]
    rows = (offsets + jnp.arange(max_targets)[None, :]).reshape(-1)
    rows = jnp.broadcast_to(rows[None], (batch, group_detr * max_targets)).astype(jnp.int32)
    cols = jnp.broadcast_to(jnp.tile(jnp.arange(max_targets), group_detr)[None], rows.shape).astype(jnp.int32)
    valid = cols < sizes.astype(jnp.int32)[:, None]
    return rows, cols, valid


def measure(call, args):
    """Compile through one call, then return compile seconds and the warm per-call median.

    ``call`` takes no arguments and performs one step; the training step donates its state buffers,
    so the caller is responsible for threading the returned state into the next call.
    """
    started = time.perf_counter()
    jax.block_until_ready(call())
    compile_seconds = time.perf_counter() - started

    for _ in range(args.warmup):
        jax.block_until_ready(call())
    samples = []
    for _ in range(args.steps):
        began = time.perf_counter()
        result = call()
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - began)
    return compile_seconds, samples


def main() -> None:
    """Run the requested variants and write one JSON record per variant."""
    args = parse_args()
    from shared_run_config import RunConfig
    from train_jax import spec_for

    from rfdetr_jax.models import criterion as criterion_module
    from rfdetr_jax.models.criterion import CriterionConfig, Targets, compute_losses, weighted_total
    from rfdetr_jax.models.lwdetr import LWDETR
    from rfdetr_jax.models.matcher import hungarian_match
    from rfdetr_jax.training.optimizer import OptimizerSpec, build_optimizer
    from rfdetr_jax.training.trainer import (
        TrainSettings,
        create_train_state,
        init_params,
        make_train_step,
    )

    base_run = RunConfig(resolution=args.resolution, batch_size=args.batch_size)
    base_spec = spec_for(base_run)

    jax.config.update("jax_default_matmul_precision", "highest")
    started = time.perf_counter()
    with jax.default_device(jax.devices("cpu")[0]):
        params = init_params(LWDETR(spec=base_spec), base_spec, jax.random.key(0))
    params = jax.device_put(jax.tree.map(np.asarray, params))
    jax.block_until_ready(params)
    init_seconds = time.perf_counter() - started
    print(f"host init: {init_seconds:.1f}s", flush=True)

    rng = np.random.default_rng(0)
    batch = args.batch_size
    host_batch = {
        "pixel_values": rng.standard_normal((batch, 3, args.resolution, args.resolution), dtype=np.float32),
        "labels": rng.integers(0, base_run.num_classes, size=(batch, base_run.max_targets)).astype(np.int32),
        "boxes": np.concatenate(
            [
                rng.uniform(0.25, 0.75, size=(batch, base_run.max_targets, 2)),
                rng.uniform(0.05, 0.30, size=(batch, base_run.max_targets, 2)),
            ],
            axis=-1,
        ).astype(np.float32),
        "target_mask": np.zeros((batch, base_run.max_targets), dtype=bool),
    }
    host_batch["target_mask"][:, :4] = True
    device_batch = {name: jnp.asarray(value) for name, value in host_batch.items()}

    criterion = CriterionConfig(num_classes=base_spec.detector_classes, group_detr=base_spec.group_detr)
    weights = criterion.weight_dict(base_spec.dec_layers)
    settings = TrainSettings(batch_size=batch, max_targets=base_run.max_targets)
    optimizer = build_optimizer(
        params,
        OptimizerSpec(
            lr=base_run.lr,
            lr_encoder=base_run.lr_encoder,
            weight_decay=base_run.weight_decay,
            lr_vit_layer_decay=base_run.lr_vit_layer_decay,
            lr_component_decay=base_run.lr_component_decay,
            clip_max_norm=base_run.clip_max_norm,
        ),
        base_spec.out_feature_indexes,
    )
    state = create_train_state(params, optimizer, use_ema=True)
    dropout_key = jax.random.key(1)

    variants = {
        "base": dict(precision="highest", grad_checkpoint=True, matcher="scipy", mode="train"),
        "precision": dict(precision="default", grad_checkpoint=True, matcher="scipy", mode="train"),
        "gradckpt": dict(precision="highest", grad_checkpoint=False, matcher="scipy", mode="train"),
        "matcher": dict(precision="highest", grad_checkpoint=True, matcher="dummy", mode="train"),
        "forward": dict(precision="highest", grad_checkpoint=True, matcher="scipy", mode="forward"),
        "combined": dict(precision="default", grad_checkpoint=False, matcher="dummy", mode="train"),
    }
    wanted = [name for name in args.variants.split(",") if name]

    records = []
    for name in wanted:
        cfg = variants[name]
        jax.config.update("jax_default_matmul_precision", cfg["precision"])
        criterion_module.hungarian_match = dummy_match if cfg["matcher"] == "dummy" else hungarian_match

        spec = spec_for(
            RunConfig(
                resolution=args.resolution,
                batch_size=batch,
                gradient_checkpointing=cfg["grad_checkpoint"],
            )
        )
        model = LWDETR(spec=spec)

        if cfg["mode"] == "train":
            step_fn = make_train_step(model, optimizer, criterion, weights, settings)

            def call(step_fn=step_fn):
                """Advance the held state by one donated step."""
                nonlocal state
                state, metrics = step_fn(state, device_batch, dropout_key)
                return metrics
        else:

            def loss_only(p, b, k, model=model):
                """Forward and loss without the optimizer update, isolating the backward pass."""
                outputs = model.apply({"params": p}, b["pixel_values"], None, train=True, rngs={"dropout": k})
                targets = Targets(labels=b["labels"], boxes=b["boxes"], mask=b["target_mask"])
                return weighted_total(compute_losses(outputs, targets, criterion, train=True), weights)

            forward_fn = jax.jit(loss_only)

            def call(forward_fn=forward_fn):
                """Run the forward and loss only, leaving the state untouched."""
                return forward_fn(state.params, device_batch, dropout_key)

        compile_seconds, samples = measure(call, args)
        record = {
            "variant": name,
            **cfg,
            "compile_seconds": compile_seconds,
            "median_seconds": statistics.median(samples),
            "min_seconds": min(samples),
            "max_seconds": max(samples),
            "images_per_second": batch / statistics.median(samples),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    criterion_module.hungarian_match = hungarian_match
    payload = {
        "device_kind": jax.devices()[0].device_kind,
        "device_count": len(jax.devices()),
        "jax_version": jax.__version__,
        "batch_size": batch,
        "resolution": args.resolution,
        "host_init_seconds": init_seconds,
        "variants": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
