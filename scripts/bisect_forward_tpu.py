"""Localize the forward pass's cost by removing one component at a time, in context.

Timing a component in its own ``jit`` measures it fusing against nothing. This instead times the
whole forward with one piece replaced by a stub of the same shape, so each difference is what that
piece costs where it actually runs. Every stub still consumes its inputs through a cheap reduction,
so XLA cannot delete the computation feeding it and credit the stub with that too.
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

import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def _cheap(reference, shape):
    """A tensor of ``shape`` that still depends on ``reference``, so nothing upstream is dead."""
    return jnp.broadcast_to(reference.mean().astype(jnp.float32), shape)


class StubAttention(nn.Module):
    """Stands in for :class:`MultiheadAttention` without the attention math."""

    embed_dim: int
    num_heads: int
    use_bias: bool = True
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, query, key, value, attn_mask=None):
        """Return something query-shaped that depends on all three inputs."""
        del attn_mask
        return query + _cheap(key, ()) + _cheap(value, ())


def main() -> None:
    """Time the forward once per stub and report what each component costs in place."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/bisect_forward.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--precision", default="default", choices=["highest", "float32", "default"])
    parser.add_argument("--compute-dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--sampler", default="auto", choices=["gather", "matmul", "auto"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--backward", action="store_true", help="Time forward plus backward instead.")
    args = parser.parse_args()

    from shared_run_config import RunConfig
    from train_jax import spec_for

    from rfdetr_jax.models import ops, transformer
    from rfdetr_jax.models.lwdetr import LWDETR
    from rfdetr_jax.training.trainer import init_params

    run = RunConfig(resolution=args.resolution, batch_size=args.batch_size, gradient_checkpointing=False)
    spec = dataclasses.replace(
        spec_for(run), deform_sampler=args.sampler, compute_dtype=args.compute_dtype
    )
    pixel_values = jnp.asarray(
        np.random.default_rng(0)
        .standard_normal((args.batch_size, 3, args.resolution, args.resolution))
        .astype(np.float32)
    )

    original = {
        "deform_core": ops.ms_deform_attn_core_rows,
        "attention": transformer.MultiheadAttention,
        "sine_embed": transformer.gen_sineembed_for_position,
        "grouped_linear": transformer.grouped_linear,
    }

    def stub_deform(value, spatial_shapes, sampling_locations, attention_weights, sampler="gather"):
        """Skip the sampling, keep the output shape and a dependency on every input."""
        del spatial_shapes, sampler
        batch, len_query, n_heads = sampling_locations.shape[:3]
        head_dim = value.shape[-1]
        shape = (batch, len_query, n_heads * head_dim)
        return _cheap(value, shape) + _cheap(sampling_locations, ()) + _cheap(attention_weights, ())

    def stub_sine(pos_tensor, dim=128):
        """Skip the sine embedding, keep its shape."""
        width = dim * pos_tensor.shape[-1]
        return _cheap(pos_tensor, (*pos_tensor.shape[:2], width))

    def stub_grouped_linear(x, weight, bias, grouped_input=True):
        """Skip the batched per-group matmul, keep its shape."""
        leading = x.shape if grouped_input else (weight.shape[0], *x.shape)
        shape = (*leading[:-1], weight.shape[1])
        return _cheap(x, shape) + _cheap(weight, ()) + _cheap(bias, ())

    variants = {
        "full": {},
        "no_deform_attn": {"deform_core": stub_deform},
        "no_self_attention": {"attention": StubAttention},
        "no_sine_embed": {"sine_embed": stub_sine},
        "no_two_stage_heads": {"grouped_linear": stub_grouped_linear},
    }

    records = []
    for name, patches in variants.items():
        ops.ms_deform_attn_core_rows = patches.get("deform_core", original["deform_core"])
        transformer.MultiheadAttention = patches.get("attention", original["attention"])
        transformer.gen_sineembed_for_position = patches.get("sine_embed", original["sine_embed"])
        transformer.grouped_linear = patches.get("grouped_linear", original["grouped_linear"])

        jax.config.update("jax_default_matmul_precision", "highest")
        model = LWDETR(spec=spec)
        with jax.default_device(jax.devices("cpu")[0]):
            params = jax.tree.map(np.asarray, init_params(model, spec, jax.random.key(0)))
        params = jax.device_put(params)
        jax.config.update("jax_default_matmul_precision", args.precision)

        def forward(p, model=model):
            """The detector's training-mode forward, reduced to a scalar."""
            outputs = model.apply(
                {"params": p}, pixel_values, None, train=True, rngs={"dropout": jax.random.key(1)}
            )
            return outputs["pred_logits"].sum() + outputs["pred_boxes"].sum()

        function = jax.jit(jax.grad(forward)) if args.backward else jax.jit(forward)
        started = time.perf_counter()
        jax.block_until_ready(function(params))
        compile_seconds = time.perf_counter() - started
        for _ in range(args.warmup):
            jax.block_until_ready(function(params))
        samples = []
        for _ in range(args.steps):
            began = time.perf_counter()
            jax.block_until_ready(function(params))
            samples.append(time.perf_counter() - began)

        record = {
            "variant": name,
            "compile_seconds": compile_seconds,
            "median_ms": statistics.median(samples) * 1e3,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        del params, function

    for key, value in original.items():
        setattr(ops if key == "deform_core" else transformer,
                "ms_deform_attn_core_rows" if key == "deform_core"
                else {"attention": "MultiheadAttention", "sine_embed": "gen_sineembed_for_position",
                      "grouped_linear": "grouped_linear"}[key], value)

    baseline = records[0]["median_ms"]
    for record in records[1:]:
        record["component_ms"] = baseline - record["median_ms"]
        record["share"] = record["component_ms"] / baseline

    payload = {
        "device": jax.devices()[0].device_kind,
        "measuring": "forward and backward" if args.backward else "forward",
        "baseline_ms": baseline,
        "records": records,
    }
    print(json.dumps(payload, indent=2), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
