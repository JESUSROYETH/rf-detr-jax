"""Decompose the real training step by timing nested prefixes of it.

Component benchmarks measure each piece in its own ``jit``, where it fuses differently and nothing
is queued behind it. This instead times four nested stages of the same step — forward, forward plus
loss, plus backward, plus the optimizer — so each difference is that stage's cost *in context*.
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


def timed(call, repeats: int, warmup: int) -> dict:
    """Compile through one call, then return compile seconds and the warm per-call median."""
    started = time.perf_counter()
    jax.block_until_ready(call())
    compile_seconds = time.perf_counter() - started
    for _ in range(warmup):
        jax.block_until_ready(call())
    samples = []
    for _ in range(repeats):
        began = time.perf_counter()
        result = call()
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - began)
    return {"compile_seconds": compile_seconds, "median_ms": statistics.median(samples) * 1e3}


def main() -> None:
    """Time each nested stage and report what every step of the ladder adds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/stages.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--precision", default="default", choices=["highest", "float32", "default"])
    parser.add_argument("--compute-dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--sampler", default="auto", choices=["gather", "matmul", "auto"])
    parser.add_argument("--matcher", default="auction", choices=["hungarian", "auction"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    from shared_run_config import RunConfig
    from train_jax import spec_for

    from rfdetr_jax.models.criterion import CriterionConfig, Targets, compute_losses, weighted_total
    from rfdetr_jax.models.lwdetr import LWDETR
    from rfdetr_jax.training.optimizer import OptimizerSpec, build_optimizer
    from rfdetr_jax.training.trainer import (
        TrainSettings,
        create_train_state,
        init_params,
        make_train_step,
    )

    run = RunConfig(resolution=args.resolution, batch_size=args.batch_size, gradient_checkpointing=False)
    spec = dataclasses.replace(
        spec_for(run), deform_sampler=args.sampler, compute_dtype=args.compute_dtype
    )
    model = LWDETR(spec=spec)

    jax.config.update("jax_default_matmul_precision", "highest")
    with jax.default_device(jax.devices("cpu")[0]):
        host_params = jax.tree.map(np.asarray, init_params(model, spec, jax.random.key(0)))
    jax.config.update("jax_default_matmul_precision", args.precision)
    params = jax.device_put(host_params)

    rng = np.random.default_rng(0)
    batch, targets = args.batch_size, run.max_targets
    mask = np.zeros((batch, targets), dtype=bool)
    mask[:, :4] = True
    device_batch = {
        "pixel_values": jnp.asarray(
            rng.standard_normal((batch, 3, args.resolution, args.resolution), dtype=np.float32)
        ),
        "labels": jnp.asarray(rng.integers(0, run.num_classes, size=(batch, targets)).astype(np.int32)),
        "boxes": jnp.asarray(
            np.concatenate(
                [
                    rng.uniform(0.25, 0.75, size=(batch, targets, 2)),
                    rng.uniform(0.05, 0.30, size=(batch, targets, 2)),
                ],
                axis=-1,
            ).astype(np.float32)
        ),
        "target_mask": jnp.asarray(mask),
    }

    criterion = CriterionConfig(
        num_classes=spec.detector_classes, group_detr=spec.group_detr, matcher=args.matcher
    )
    weights = criterion.weight_dict(spec.dec_layers)
    dropout_key = jax.random.key(1)

    def forward(p):
        """Model forward in training mode; returns the main logits so nothing is dead code."""
        outputs = model.apply({"params": p}, device_batch["pixel_values"], None, train=True,
                              rngs={"dropout": dropout_key})
        return outputs["pred_logits"].sum()

    def forward_and_loss(p):
        """Model forward plus the full criterion, including the match."""
        outputs = model.apply({"params": p}, device_batch["pixel_values"], None, train=True,
                              rngs={"dropout": dropout_key})
        targets_struct = Targets(
            labels=device_batch["labels"], boxes=device_batch["boxes"], mask=device_batch["target_mask"]
        )
        return weighted_total(compute_losses(outputs, targets_struct, criterion, train=True), weights)

    optimizer = build_optimizer(params, OptimizerSpec(), spec.out_feature_indexes)
    state = create_train_state(params, optimizer, use_ema=True)
    step_fn = make_train_step(model, optimizer, criterion, weights, TrainSettings(
        batch_size=batch, max_targets=targets
    ))

    stages = {}
    jitted_forward = jax.jit(forward)
    stages["forward"] = timed(lambda: jitted_forward(params), args.steps, args.warmup)

    jitted_loss = jax.jit(forward_and_loss)
    stages["forward_and_loss"] = timed(lambda: jitted_loss(params), args.steps, args.warmup)

    jitted_grad = jax.jit(jax.grad(forward_and_loss))
    stages["with_backward"] = timed(lambda: jitted_grad(params), args.steps, args.warmup)

    def full_step():
        """Advance the held state by one donated step."""
        nonlocal state
        state, metrics = step_fn(state, device_batch, dropout_key)
        return metrics

    stages["full_step"] = timed(full_step, args.steps, args.warmup)

    order = ["forward", "forward_and_loss", "with_backward", "full_step"]
    previous = 0.0
    for name in order:
        stages[name]["adds_ms"] = stages[name]["median_ms"] - previous
        previous = stages[name]["median_ms"]

    payload = {
        "device": jax.devices()[0].device_kind,
        "batch_size": batch,
        "precision": args.precision,
        "compute_dtype": args.compute_dtype,
        "sampler": args.sampler,
        "matcher": args.matcher,
        "stages": stages,
    }
    print(json.dumps(payload, indent=2), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
