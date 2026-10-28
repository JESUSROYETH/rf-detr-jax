"""Check that the TPU changes hold across RF-DETR's variants, not just Nano.

The samplers, matcher and fused query-selection head scale with quantities that
change between variants — the level's
size, the decoder's depth — so each variant is measured at its own native resolution, once at the
port's settings and once at the TPU settings.
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

#: Geometry of each released variant, from ``rfdetr.config``.
VARIANTS = {
    "nano": dict(resolution=384, patch_size=16, num_windows=2, dec_layers=2, encoder="dinov2_windowed_small"),
    "small": dict(resolution=512, patch_size=16, num_windows=2, dec_layers=3, encoder="dinov2_windowed_small"),
    "medium": dict(resolution=576, patch_size=16, num_windows=2, dec_layers=4, encoder="dinov2_windowed_small"),
    "large": dict(resolution=704, patch_size=16, num_windows=2, dec_layers=4, encoder="dinov2_windowed_small"),
}

SETTINGS = {
    "port": dict(precision="highest", grad_checkpoint=True, matcher="hungarian", sampler="gather"),
    "tpu": dict(precision="default", grad_checkpoint=False, matcher="auction", sampler="auto"),
}


def main() -> None:
    """Time one training step per variant under both settings and report the ratio."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/variants.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--max-targets", type=int, default=20)
    args = parser.parse_args()

    from rfdetr_jax.models.criterion import CriterionConfig
    from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec
    from rfdetr_jax.training.optimizer import OptimizerSpec, build_optimizer
    from rfdetr_jax.training.trainer import TrainSettings, create_train_state, init_params, make_train_step

    batch, targets = args.batch_size, args.max_targets
    rng = np.random.default_rng(0)
    records = []

    for name in args.variants.split(","):
        geometry = VARIANTS[name]
        resolution = geometry["resolution"]
        base = LWDETRSpec(
            encoder=geometry["encoder"],
            patch_size=geometry["patch_size"],
            num_windows=geometry["num_windows"],
            dec_layers=geometry["dec_layers"],
            resolution=resolution,
            positional_encoding_size=resolution // geometry["patch_size"],
            num_classes=args.num_classes,
            out_feature_indexes=(3, 6, 9, 12),
        )

        jax.config.update("jax_default_matmul_precision", "highest")
        with jax.default_device(jax.devices("cpu")[0]):
            host_params = jax.tree.map(np.asarray, init_params(LWDETR(spec=base), base, jax.random.key(0)))

        host_batch = {
            "pixel_values": rng.standard_normal((batch, 3, resolution, resolution), dtype=np.float32),
            "labels": rng.integers(0, args.num_classes, size=(batch, targets)).astype(np.int32),
            "boxes": np.concatenate(
                [
                    rng.uniform(0.25, 0.75, size=(batch, targets, 2)),
                    rng.uniform(0.05, 0.30, size=(batch, targets, 2)),
                ],
                axis=-1,
            ).astype(np.float32),
            "target_mask": np.zeros((batch, targets), dtype=bool),
        }
        host_batch["target_mask"][:, :4] = True
        device_batch = {key: jnp.asarray(value) for key, value in host_batch.items()}

        record = {
            "variant": name,
            "resolution": resolution,
            "dec_layers": geometry["dec_layers"],
            "level": resolution // geometry["patch_size"],
        }
        for label, settings in SETTINGS.items():
            jax.config.update("jax_default_matmul_precision", settings["precision"])
            spec = dataclasses.replace(
                base,
                gradient_checkpointing=settings["grad_checkpoint"],
                deform_sampler=settings["sampler"],
            )
            model = LWDETR(spec=spec)
            params = jax.device_put(host_params)
            optimizer = build_optimizer(params, OptimizerSpec(), spec.out_feature_indexes)
            state = create_train_state(params, optimizer, use_ema=True)
            criterion = CriterionConfig(
                num_classes=spec.detector_classes, group_detr=spec.group_detr, matcher=settings["matcher"]
            )
            step_fn = make_train_step(
                model,
                optimizer,
                criterion,
                criterion.weight_dict(spec.dec_layers),
                TrainSettings(batch_size=batch, max_targets=targets),
            )
            dropout_key = jax.random.key(1)

            started = time.perf_counter()
            state, metrics = step_fn(state, device_batch, dropout_key)
            jax.block_until_ready((state, metrics))
            compile_seconds = time.perf_counter() - started

            for _ in range(args.warmup):
                state, metrics = step_fn(state, device_batch, dropout_key)
                jax.block_until_ready((state, metrics))
            samples = []
            for _ in range(args.steps):
                began = time.perf_counter()
                state, metrics = step_fn(state, device_batch, dropout_key)
                jax.block_until_ready((state, metrics))
                samples.append(time.perf_counter() - began)

            record[label] = {
                "compile_seconds": compile_seconds,
                "step_ms": statistics.median(samples) * 1e3,
                "images_per_second": batch / statistics.median(samples),
                "loss": float(metrics["loss"]),
            }
            del state, step_fn

        record["speedup"] = record["port"]["step_ms"] / record["tpu"]["step_ms"]
        records.append(record)
        print(json.dumps(record), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {"device_kind": jax.devices()[0].device_kind, "batch_size": batch, "records": records}, indent=2
        )
    )


if __name__ == "__main__":
    main()
