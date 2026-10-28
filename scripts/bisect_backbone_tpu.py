"""Find what the DINOv2 backbone's forward and backward spend their time on.

The backbone is a small ViT — twelve blocks over 149-token windows. This runs the block stack
standalone at those shapes and removes one piece at a time, forward and with gradients.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rfdetr_jax.models.backbone import dinov2_windowed  # noqa: E402
from rfdetr_jax.models.backbone.dinov2_windowed import (  # noqa: E402
    WindowedDinov2Config,
    load_dinov2_json,
)


class Stack(nn.Module):
    """A run of DINOv2 blocks, with one piece optionally removed."""

    config: WindowedDinov2Config
    layers: int
    stage: str = "full"

    @nn.compact
    def __call__(self, hidden_states):
        """Run ``layers`` blocks, skipping whatever ``stage`` names."""
        cfg = self.config
        for index in range(self.layers):
            shortcut = hidden_states
            normed = dinov2_windowed.LayerNorm(
                cfg.hidden_size, eps=cfg.layer_norm_eps, name=f"norm1_{index}"
            )(hidden_states)
            attended = (
                normed
                if self.stage == "no_attention"
                else dinov2_windowed.Attention(cfg, name=f"attention_{index}")(normed)
            )
            if self.stage != "no_layerscale":
                attended = dinov2_windowed.LayerScale(cfg, name=f"scale1_{index}")(attended)
            hidden_states = attended + shortcut

            normed = dinov2_windowed.LayerNorm(
                cfg.hidden_size, eps=cfg.layer_norm_eps, name=f"norm2_{index}"
            )(hidden_states)
            mlp = (
                normed
                if self.stage == "no_mlp"
                else dinov2_windowed.Mlp(cfg, name=f"mlp_{index}")(normed)
            )
            if self.stage != "no_layerscale":
                mlp = dinov2_windowed.LayerScale(cfg, name=f"scale2_{index}")(mlp)
            hidden_states = mlp + hidden_states
        return hidden_states


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
    """Time the stack and its reductions, forward and with gradients."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/backbone_bisect.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=149)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--precision", default="default", choices=["highest", "float32", "default"])
    parser.add_argument("--compute-dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", args.precision)
    config = WindowedDinov2Config(**{
        key: value
        for key, value in load_dinov2_json("small", use_registers=True).items()
        if key in WindowedDinov2Config.__dataclass_fields__
    })
    config = WindowedDinov2Config(**{**config.__dict__, "num_windows": args.windows})

    sequences = args.batch_size * args.windows**2
    dtype = jnp.bfloat16 if args.compute_dtype == "bfloat16" else jnp.float32
    hidden = jax.random.normal(
        jax.random.key(0), (sequences, args.tokens, config.hidden_size), jnp.float32
    ).astype(dtype)

    records = []
    for stage in ("full", "no_attention", "no_mlp", "no_layerscale"):
        module = Stack(config, args.layers, stage)
        params = module.init(jax.random.key(0), hidden)

        def forward(p, x, module=module):
            """Run the stack and reduce to a scalar."""
            return module.apply(p, x).astype(jnp.float32).sum()

        record = {
            "stage": stage,
            "forward": timed(forward, (params, hidden), args.steps, args.warmup),
            "with_grad": timed(jax.grad(forward), (params, hidden), args.steps, args.warmup),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    baseline = records[0]
    for record in records[1:]:
        record["removed_forward_ms"] = baseline["forward"]["median_ms"] - record["forward"]["median_ms"]
        record["removed_grad_ms"] = baseline["with_grad"]["median_ms"] - record["with_grad"]["median_ms"]

    payload = {
        "device": jax.devices()[0].device_kind,
        "sequences": sequences,
        "tokens": args.tokens,
        "width": config.hidden_size,
        "layers": args.layers,
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
