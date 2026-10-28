"""Find where the gather and matmul deformable-attention samplers cross over.

The matmul sampler's cost grows with the feature map it contracts against, while the gather's does
not, so which one wins is a property of the level's size rather than of the model. This sweeps the
feature-map side length at otherwise fixed decoder shapes and reports both, forward and backward.
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

from rfdetr_jax.models.ops import ms_deform_attn_core_rows  # noqa: E402


def timed(fn, args, repeats: int = 8, warmup: int = 2) -> float:
    """Return the warm median seconds per call of ``fn(*args)``."""
    for _ in range(warmup + 1):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main() -> None:
    """Sweep the feature-map side length and report both samplers at each size."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/sampler_crossover.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--queries", type=int, default=300 * 13)
    parser.add_argument("--points", type=int, default=2)
    parser.add_argument("--sides", default="12,24,32,48,64,96")
    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", "highest")
    key = jax.random.key(0)
    rows = args.batch_size * args.heads
    records = []

    for side in (int(value) for value in args.sides.split(",")):
        length = side * side
        value = jax.random.normal(key, (rows, length, args.head_dim), jnp.float32)
        locations = jax.random.uniform(
            key, (args.batch_size, args.queries, args.heads, 1, args.points, 2), jnp.float32
        )
        weights = jax.random.uniform(key, (args.batch_size, args.queries, args.heads, args.points), jnp.float32)

        kernel_bytes = rows * args.queries * args.points * length * 4
        record = {"side": side, "length": length, "kernel_megabytes": kernel_bytes / (1024 * 1024)}
        for sampler in ("gather", "matmul"):

            def core(v, loc, w, sampler=sampler, side=side):
                """Run one deformable-attention level with the chosen sampler."""
                return ms_deform_attn_core_rows(v, ((side, side),), loc, w, sampler)

            forward = jax.jit(core)
            backward = jax.jit(jax.grad(lambda *a, core=core: jnp.sum(core(*a))))
            record[f"{sampler}_forward_ms"] = timed(forward, (value, locations, weights)) * 1e3
            record[f"{sampler}_grad_ms"] = timed(backward, (value, locations, weights)) * 1e3
        record["matmul_speedup_forward"] = record["gather_forward_ms"] / record["matmul_forward_ms"]
        record["matmul_speedup_grad"] = record["gather_grad_ms"] / record["matmul_grad_ms"]
        records.append(record)
        print(json.dumps(record), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"device": jax.devices()[0].device_kind, "records": records}, indent=2))


if __name__ == "__main__":
    main()
