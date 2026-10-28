"""Find which part of two-stage query selection costs what, at the exact shapes it runs on.

This rebuilds the query-selection block standalone and takes one piece out at a time. The shapes
are RF-DETR Nano's in training mode: thirteen query groups selecting 300 of 576 encoder positions each.
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

from rfdetr_jax.models.layers import grouped_layer_norm, grouped_linear  # noqa: E402


def timed(fn, arguments, repeats: int, warmup: int) -> dict:
    """Compile ``fn``, warm it and return the median warm call in milliseconds."""
    call = jax.jit(fn)
    started = time.perf_counter()
    jax.block_until_ready(call(*arguments))
    compile_seconds = time.perf_counter() - started
    for _ in range(warmup):
        jax.block_until_ready(call(*arguments))
    samples = []
    for _ in range(repeats):
        began = time.perf_counter()
        jax.block_until_ready(call(*arguments))
        samples.append(time.perf_counter() - began)
    return {"compile_seconds": compile_seconds, "median_ms": statistics.median(samples) * 1e3}


def main() -> None:
    """Time the block and five reductions of it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/two_stage.json")
    parser.add_argument("--groups", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--positions", type=int, default=576)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--classes", type=int, default=7)
    parser.add_argument("--precision", default="default", choices=["highest", "float32", "default"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", args.precision)
    groups, batch = args.groups, args.batch_size
    positions, queries, width, classes = args.positions, args.queries, args.width, args.classes
    key = jax.random.key(0)

    memory = jax.random.normal(key, (batch, positions, width), jnp.float32)
    proposals = jax.random.uniform(key, (batch, positions, 4), jnp.float32)
    project_w = jax.random.normal(key, (groups, width, width), jnp.float32) * 0.05
    project_b = jnp.zeros((groups, width), jnp.float32)
    norm_w = jnp.ones((groups, width), jnp.float32)
    norm_b = jnp.zeros((groups, width), jnp.float32)
    class_w = jax.random.normal(key, (groups, classes, width), jnp.float32) * 0.05
    class_b = jnp.zeros((groups, classes), jnp.float32)
    bbox = [
        (jax.random.normal(key, (groups, width, width), jnp.float32) * 0.05, jnp.zeros((groups, width))),
        (jax.random.normal(key, (groups, width, width), jnp.float32) * 0.05, jnp.zeros((groups, width))),
        (jax.random.normal(key, (groups, 4, width), jnp.float32) * 0.05, jnp.zeros((groups, 4))),
    ]
    arguments = (memory, proposals, project_w, project_b, norm_w, norm_b, class_w, class_b, bbox)
    fixed_indices = jnp.broadcast_to(jnp.arange(queries)[None, None, :], (groups, batch, queries))

    def block(memory, proposals, pw, pb, nw, nb, cw, cb, bbox, *, stage="full"):
        """The two-stage head, with one piece skipped when ``stage`` names it."""
        projected = grouped_linear(memory, pw, pb, grouped_input=False)
        if stage != "no_norm":
            projected = grouped_layer_norm(projected, nw, nb)
        logits = grouped_linear(projected, cw, cb)

        if stage == "no_topk":
            indices = fixed_indices
        else:
            _, indices = jax.lax.top_k(logits.max(-1), queries)

        if stage == "no_select":
            selected = projected[:, :, :queries]
            chosen = jnp.broadcast_to(proposals[None, :, :queries], (groups, batch, queries, 4))
        elif stage == "onehot_select":
            # The same selection as a contraction: a matmul's speed does not depend on the layout
            # its operand happened to be produced in, and an indexed read's very much does.
            picker = (indices[..., None] == jnp.arange(positions)[None, None, None, :])
            picker = picker.astype(projected.dtype)
            selected = jnp.einsum("gbqk,gbkc->gbqc", picker, projected)
            chosen = jnp.einsum("gbqk,bkc->gbqc", picker, proposals)
        elif stage == "flat_gather":
            flat = projected.reshape(groups * batch, positions, -1)
            flat_index = indices.reshape(groups * batch, queries)
            selected = jnp.take_along_axis(flat, flat_index[..., None], axis=1).reshape(
                groups, batch, queries, -1
            )
            chosen = jnp.take_along_axis(
                jnp.broadcast_to(proposals[None], (groups, *proposals.shape)), indices[..., None], axis=2
            )
        else:
            selected = jnp.take_along_axis(projected, indices[..., None], axis=2)
            chosen = jnp.take_along_axis(
                jnp.broadcast_to(proposals[None], (groups, *proposals.shape)), indices[..., None], axis=2
            )

        if stage == "no_heads":
            return selected.sum() + chosen.sum() + logits.sum()

        selected_logits = grouped_linear(selected, cw, cb)
        delta = selected
        for index, (weight, bias) in enumerate(bbox):
            delta = grouped_linear(delta, weight, bias)
            if index < 2:
                delta = jax.nn.relu(delta)
        centre = delta[..., :2] * chosen[..., 2:] + chosen[..., :2]
        size = jnp.exp(delta[..., 2:]) * chosen[..., 2:]
        return selected.sum() + selected_logits.sum() + centre.sum() + size.sum() + logits.sum()

    stages = [
        "full", "onehot_select", "flat_gather", "no_norm", "no_topk", "no_select", "no_heads",
        "projection_only",
    ]
    records = []
    for stage in stages:
        if stage == "projection_only":

            def fn(memory, proposals, pw, pb, nw, nb, cw, cb, bbox):
                """Only the per-group encoder projection."""
                del proposals, nw, nb, cw, cb, bbox
                return grouped_linear(memory, pw, pb, grouped_input=False).sum()
        else:

            def fn(*a, stage=stage):
                """The block with one piece removed."""
                return block(*a, stage=stage)

        record = {"stage": stage, **timed(fn, arguments, args.steps, args.warmup)}
        records.append(record)
        print(json.dumps(record), flush=True)

    baseline = records[0]["median_ms"]
    for record in records[1:]:
        record["removed_ms"] = baseline - record["median_ms"]
    payload = {"device": jax.devices()[0].device_kind, "baseline_ms": baseline, "records": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
