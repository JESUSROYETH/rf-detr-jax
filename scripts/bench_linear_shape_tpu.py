"""Ask whether a linear layer's weight gradient cares how many leading axes its input has.

Indexed reads that carry more leading axes than XLA expects can be far slower than their flat
equivalent. A matmul's transpose is a contraction over every leading axis at once, so the same question applies:
does ``[8, 149, 384] @ [384, 384]`` cost more than the same arithmetic reshaped to ``[1192, 384]``?
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

#: Shapes the port actually runs: the backbone's windowed blocks and the decoder's query stream.
CASES = {
    "backbone_block": ((8, 149, 384), 384, 384),
    "backbone_mlp": ((8, 149, 384), 384, 1536),
    "decoder_ffn": ((2, 3900, 256), 256, 2048),
    "decoder_proj": ((26, 300, 256), 256, 256),
}


def timed(fn, arguments, repeats: int, warmup: int) -> float:
    """Return the warm median milliseconds per call."""
    call = jax.jit(fn)
    for _ in range(warmup + 1):
        jax.block_until_ready(call(*arguments))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(call(*arguments))
        samples.append(time.perf_counter() - started)
    return statistics.median(samples) * 1e3


def main() -> None:
    """Time each shape written both ways, forward and with gradients."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/linear_shape.json")
    parser.add_argument("--precision", default="default", choices=["highest", "float32", "default"])
    parser.add_argument("--compute-dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--stack", type=int, default=6, help="Layers per timed call, to lift it off the floor.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", args.precision)
    dtype = jnp.bfloat16 if args.compute_dtype == "bfloat16" else jnp.float32
    key = jax.random.key(0)
    records = []

    for name, (shape, inputs, outputs) in CASES.items():
        x = jax.random.normal(key, shape, jnp.float32).astype(dtype)
        # A stack of square-ish layers so the timing is not dominated by call overhead.
        widths = [(inputs, outputs)] + [(outputs, outputs)] * (args.stack - 1)
        weights = [
            jax.random.normal(key, (out_dim, in_dim), jnp.float32).astype(dtype) * 0.03
            for in_dim, out_dim in widths
        ]

        def as_written(x, ws):
            """The port's `Linear`: an einsum over however many leading axes arrive."""
            for weight in ws:
                x = jnp.einsum("...i,oi->...o", x, weight)
            return x.astype(jnp.float32).sum()

        def flattened(x, ws):
            """The same layers with the leading axes folded into one before each matmul."""
            leading = x.shape[:-1]
            value = x.reshape(-1, x.shape[-1])
            for weight in ws:
                value = value @ weight.T
            return value.reshape(*leading, value.shape[-1]).astype(jnp.float32).sum()

        record = {
            "case": name,
            "shape": list(shape),
            "as_written_forward_ms": timed(as_written, (x, weights), args.steps, args.warmup),
            "flattened_forward_ms": timed(flattened, (x, weights), args.steps, args.warmup),
            "as_written_grad_ms": timed(
                jax.grad(as_written, argnums=(0, 1)), (x, weights), args.steps, args.warmup
            ),
            "flattened_grad_ms": timed(
                jax.grad(flattened, argnums=(0, 1)), (x, weights), args.steps, args.warmup
            ),
        }
        record["grad_speedup"] = record["as_written_grad_ms"] / record["flattened_grad_ms"]
        records.append(record)
        print(json.dumps(record), flush=True)

    payload = {"device": jax.devices()[0].device_kind, "compute_dtype": args.compute_dtype, "records": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
