"""Compare each stage's measured time against what its arithmetic and its memory traffic imply.

XLA's cost analysis reports the flops and the bytes a compiled module touches. Dividing those by the
chip's peak gives two lower bounds; whichever is larger is what the stage is bound by, and the ratio
to the measured time says how much is left on the table. It is a roofline, not a profile — it says
which wall a stage is against, not which op is against it.
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

#: v6e peak bf16 matmul throughput and HBM bandwidth, from Google's published figures.
PEAK_FLOPS = 918e12
PEAK_BANDWIDTH = 1.6e12


def analyse(name: str, fn, arguments, repeats: int, warmup: int) -> dict:
    """Compile ``fn``, read its cost analysis, and time it warm."""
    compiled = jax.jit(fn).lower(*arguments).compile()
    analysis = compiled.cost_analysis()
    entry = analysis[0] if isinstance(analysis, list) else analysis
    flops = float(entry.get("flops", 0.0))
    written = float(entry.get("bytes accessed", 0.0))

    call = jax.jit(fn)
    for _ in range(warmup + 1):
        jax.block_until_ready(call(*arguments))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(call(*arguments))
        samples.append(time.perf_counter() - started)
    measured = statistics.median(samples)

    record = {
        "stage": name,
        "measured_ms": measured * 1e3,
        "gflops": flops / 1e9,
        "gigabytes_accessed": written / 1e9,
        "compute_bound_ms": flops / PEAK_FLOPS * 1e3,
        "bandwidth_bound_ms": written / PEAK_BANDWIDTH * 1e3,
    }
    record["bound_by"] = (
        "bandwidth" if record["bandwidth_bound_ms"] >= record["compute_bound_ms"] else "compute"
    )
    record["headroom"] = record["measured_ms"] / max(
        record["bandwidth_bound_ms"], record["compute_bound_ms"], 1e-9
    )
    print(json.dumps(record), flush=True)
    return record


def main() -> None:
    """Roofline the forward, the backward and the whole step."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/roofline.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--precision", default="default", choices=["highest", "float32", "default"])
    parser.add_argument("--compute-dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--sampler", default="auto", choices=["gather", "matmul", "auto"])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    from shared_run_config import RunConfig
    from train_jax import spec_for

    from rfdetr_jax.models.criterion import CriterionConfig, Targets, compute_losses, weighted_total
    from rfdetr_jax.models.lwdetr import LWDETR
    from rfdetr_jax.training.trainer import init_params

    run = RunConfig(resolution=args.resolution, batch_size=args.batch_size, gradient_checkpointing=False)
    spec = dataclasses.replace(
        spec_for(run), deform_sampler=args.sampler, compute_dtype=args.compute_dtype
    )
    model = LWDETR(spec=spec)

    jax.config.update("jax_default_matmul_precision", "highest")
    with jax.default_device(jax.devices("cpu")[0]):
        params = jax.tree.map(np.asarray, init_params(model, spec, jax.random.key(0)))
    params = jax.device_put(params)
    jax.config.update("jax_default_matmul_precision", args.precision)

    rng = np.random.default_rng(0)
    batch, targets = args.batch_size, run.max_targets
    mask = np.zeros((batch, targets), dtype=bool)
    mask[:, :4] = True
    pixel_values = jnp.asarray(
        rng.standard_normal((batch, 3, args.resolution, args.resolution), dtype=np.float32)
    )
    target_struct = Targets(
        labels=jnp.asarray(rng.integers(0, run.num_classes, size=(batch, targets)).astype(np.int32)),
        boxes=jnp.asarray(
            np.concatenate(
                [
                    rng.uniform(0.25, 0.75, size=(batch, targets, 2)),
                    rng.uniform(0.05, 0.30, size=(batch, targets, 2)),
                ],
                axis=-1,
            ).astype(np.float32)
        ),
        mask=jnp.asarray(mask),
    )
    criterion = CriterionConfig(
        num_classes=spec.detector_classes, group_detr=spec.group_detr, matcher="auction"
    )
    weights = criterion.weight_dict(spec.dec_layers)

    def backbone_only(p):
        """Just the DINOv2 backbone and projector."""
        return model.apply({"params": p}, pixel_values, True, method=_backbone).sum()

    def _backbone(module, x, train):
        """Module method selecting the backbone branch."""
        return module.backbone_0(x, train=train)[0][0]

    def forward(p):
        """Detector forward in training mode."""
        outputs = model.apply(
            {"params": p}, pixel_values, None, train=True, rngs={"dropout": jax.random.key(1)}
        )
        return weighted_total(compute_losses(outputs, target_struct, criterion, train=True), weights)

    records = [
        analyse("backbone_forward", backbone_only, (params,), args.steps, args.warmup),
        analyse("forward_and_loss", forward, (params,), args.steps, args.warmup),
        analyse("with_backward", jax.grad(forward), (params,), args.steps, args.warmup),
    ]

    payload = {
        "device": jax.devices()[0].device_kind,
        "peak_flops": PEAK_FLOPS,
        "peak_bandwidth": PEAK_BANDWIDTH,
        "batch_size": batch,
        "compute_dtype": args.compute_dtype,
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
