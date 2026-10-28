"""Price the projector's convolution stack in NCHW against NHWC.

The port keeps RF-DETR's NCHW activations so it can follow the original line by line. Parameters
must stay in torch's ``[O, I, kH, kW]`` layout for checkpoints to load, but the activations flowing
through the convolutions need not, and XLA has a preferred layout. This runs the shapes RF-DETR
Nano's multi-scale projector actually sees, both ways, forward and with gradients.
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

#: Channel widths of `C2f(1536 -> 256, 3 blocks)`, the projector's fusion block for RF-DETR Nano.
FUSED_IN = 1536
HIDDEN = 128
BLOCKS = 3
OUT = 256


def timed(fn, arguments, repeats: int, warmup: int) -> dict:
    """Compile, warm and time ``fn``."""
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
    """Run the stack in both layouts and report the difference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/conv_layout.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--side", type=int, default=24)
    parser.add_argument("--precision", default="default", choices=["highest", "float32", "default"])
    parser.add_argument("--compute-dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", args.precision)
    dtype = jnp.bfloat16 if args.compute_dtype == "bfloat16" else jnp.float32
    batch, side = args.batch_size, args.side
    key = jax.random.key(0)

    # Parameters stay in torch's `[O, I, kH, kW]` layout in both arms; only the activations differ.
    def weight(out_dim, in_dim, size):
        """One convolution kernel in torch's layout."""
        return jax.random.normal(key, (out_dim, in_dim, size, size), jnp.float32).astype(dtype) * 0.02

    weights = {
        "cv1": weight(2 * HIDDEN, FUSED_IN, 1),
        "cv2": weight(OUT, (2 + BLOCKS) * HIDDEN, 1),
        "blocks": [(weight(HIDDEN, HIDDEN, 3), weight(HIDDEN, HIDDEN, 3)) for _ in range(BLOCKS)],
    }
    nchw = jax.random.normal(key, (batch, FUSED_IN, side, side), jnp.float32).astype(dtype)
    nhwc = jnp.transpose(nchw, (0, 2, 3, 1))

    def run(x, ws, layout):
        """The projector's `C2f` fusion block, in whichever layout is active."""
        channel = 1 if layout == "NCHW" else 3
        numbers = (layout, "OIHW", layout)

        def convolve(value, kernel):
            """One convolution followed by the block's normalisation and activation."""
            pad = kernel.shape[-1] // 2
            value = jax.lax.conv_general_dilated(
                value, kernel, (1, 1), ((pad, pad), (pad, pad)), dimension_numbers=numbers
            )
            promoted = value.astype(jnp.float32)
            mean = promoted.mean(channel, keepdims=True)
            variance = promoted.var(channel, keepdims=True)
            normed = ((promoted - mean) * jax.lax.rsqrt(variance + 1e-6)).astype(value.dtype)
            return jax.nn.silu(normed)

        fused = convolve(x, ws["cv1"])
        first = jax.lax.slice_in_dim(fused, 0, HIDDEN, axis=channel)
        parts = [first, jax.lax.slice_in_dim(fused, HIDDEN, 2 * HIDDEN, axis=channel)]
        for inner, outer in ws["blocks"]:
            parts.append(convolve(convolve(parts[-1], inner), outer))
        return convolve(jnp.concatenate(parts, axis=channel), ws["cv2"]).astype(jnp.float32).sum()

    records = []
    for layout, activations in (("NCHW", nchw), ("NHWC", nhwc)):
        record = {
            "layout": layout,
            "forward": timed(lambda x, w, layout=layout: run(x, w, layout), (activations, weights),
                             args.steps, args.warmup),
            "with_grad": timed(
                jax.grad(lambda x, w, layout=layout: run(x, w, layout), argnums=(0, 1)),
                (activations, weights), args.steps, args.warmup,
            ),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    speedup = records[0]["with_grad"]["median_ms"] / records[1]["with_grad"]["median_ms"]
    payload = {
        "device": jax.devices()[0].device_kind,
        "batch_size": batch,
        "side": side,
        "compute_dtype": args.compute_dtype,
        "nhwc_speedup_with_grad": speedup,
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
