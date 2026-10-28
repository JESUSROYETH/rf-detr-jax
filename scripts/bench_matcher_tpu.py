"""Price the assignment solvers and the cost-matrix build separately, at production shapes.

A host callback looks cheap in isolation, because nothing else is queued behind it. This measures
each piece on its own; the end-to-end sweep is what shows what a callback costs inside a real step.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rfdetr_jax.models.matcher import auction_match, build_cost_matrix, hungarian_match  # noqa: E402


def timed(fn, args, repeats: int = 10, warmup: int = 3) -> dict:
    """Compile, warm up and time ``fn(*args)``, returning compile seconds and the warm median."""
    started = time.perf_counter()
    jax.block_until_ready(fn(*args))
    compile_seconds = time.perf_counter() - started
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeats):
        began = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - began)
    return {"compile_seconds": compile_seconds, "median_ms": statistics.median(samples) * 1e3}


def main() -> None:
    """Measure the cost build and both solvers over a range of real target counts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/matcher.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--layers", type=int, default=3, help="Output layers matched per step.")
    parser.add_argument("--group-detr", type=int, default=13)
    parser.add_argument("--per-group", type=int, default=300)
    parser.add_argument("--max-targets", type=int, default=20)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--real-targets", default="1,4,8,20")
    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", "highest")
    key = jax.random.key(0)
    batch, queries, targets = args.batch_size, args.group_detr * args.per_group, args.max_targets

    logits = jax.random.normal(key, (batch, queries, args.num_classes), jnp.float32)
    boxes = jnp.concatenate(
        [
            jax.random.uniform(key, (batch, queries, 2), jnp.float32) * 0.5 + 0.25,
            jax.random.uniform(key, (batch, queries, 2), jnp.float32) * 0.25 + 0.05,
        ],
        axis=-1,
    )
    # Two target sets. Distinct targets are what a real batch looks like. Identical ones are the
    # auction's worst case: every column prefers the same rows, so prices have to separate them.
    distinct_boxes = jnp.concatenate(
        [
            jax.random.uniform(key, (batch, targets, 2), jnp.float32) * 0.5 + 0.25,
            jax.random.uniform(key, (batch, targets, 2), jnp.float32) * 0.25 + 0.05,
        ],
        axis=-1,
    )
    distinct_labels = jax.random.randint(key, (batch, targets), 0, args.num_classes, jnp.int32)
    identical_boxes = jnp.tile(jnp.asarray([0.5, 0.5, 0.2, 0.2], jnp.float32), (batch, targets, 1))
    identical_labels = jnp.zeros((batch, targets), jnp.int32)

    records = []
    for kind, target_boxes, target_labels in (
        ("distinct", distinct_boxes, distinct_labels),
        ("identical", identical_boxes, identical_labels),
    ):
        for real in (int(v) for v in args.real_targets.split(",")):
            mask = jnp.arange(targets)[None, :] < real
            mask = jnp.broadcast_to(mask, (batch, targets))

            def cost_only(logits, boxes, mask=mask, labels=target_labels, tboxes=target_boxes):
                """Build one layer's cost matrix."""
                return build_cost_matrix(logits, boxes, labels, tboxes, mask, 2.0, 5.0, 2.0, 0.25)

            record = {
                "targets": kind,
                "real_targets": real,
                "cost_build": timed(jax.jit(cost_only), (logits, boxes)),
            }
            stacked = jnp.stack([cost_only(logits, boxes)] * args.layers)
            sizes = jnp.full((batch,), real, jnp.int32)
            solvers = {
                "hungarian": hungarian_match,
                "auction": auction_match,
                # The completion only fires when the bidding hit its round cap, so this prices what its
                # presence costs even when it does nothing.
                "auction_without_completion": lambda c, s, g: auction_match(c, s, g, complete=False),
            }
            for name, solver in solvers.items():
                record[name] = timed(
                    jax.jit(lambda c, s, solver=solver: solver(c, s, args.group_detr)), (stacked, sizes)
                )
            records.append(record)
            print(json.dumps(record), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"device": jax.devices()[0].device_kind, "records": records}, indent=2))


if __name__ == "__main__":
    main()
