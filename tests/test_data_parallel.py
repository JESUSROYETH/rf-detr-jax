"""The explicit per-device (shard_map) step must reproduce the step XLA partitions on its own."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

_SCRIPT = textwrap.dedent(
    """
    import json
    import jax, jax.numpy as jnp, numpy as np
    from rfdetr_jax.models.criterion import CriterionConfig
    from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec
    from rfdetr_jax.training.optimizer import OptimizerSpec, build_optimizer
    from rfdetr_jax.training.trainer import (
        TrainSettings, create_train_state, data_parallel, init_params, make_predict_step, make_train_step,
    )

    spec = LWDETRSpec(dec_layers=2, num_windows=1, patch_size=16, resolution=32, positional_encoding_size=2,
                      num_classes=3, group_detr=2, num_queries=6, num_select=6, hidden_dim=32, sa_nheads=2,
                      ca_nheads=2, dim_feedforward=32, out_feature_indexes=(1, 2))
    model = LWDETR(spec=spec)
    params = jax.tree.map(np.asarray, init_params(model, spec, jax.random.key(0)))
    optimizer = build_optimizer(params, OptimizerSpec(), spec.out_feature_indexes)
    config = CriterionConfig(num_classes=spec.detector_classes, group_detr=spec.group_detr, matcher="auction")
    settings = TrainSettings(ema_decay=0.9, ema_tau=0)
    weights = config.weight_dict(spec.dec_layers)

    rng = np.random.RandomState(0)
    batch = {
        "pixel_values": rng.randn(4, 3, 32, 32).astype("float32"),
        "labels": np.array([[1, 2, 0], [2, 0, 0], [0, 0, 0], [1, 1, 2]], dtype="int32"),
        "boxes": rng.uniform(0.2, 0.4, (4, 3, 4)).astype("float32"),
        # Uneven counts, one image with none: the loss denominator must be the global count.
        "target_mask": np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=bool),
    }

    results = {}
    for mode in ("gspmd", "shard_map"):
        state, sharding = data_parallel(create_train_state(jax.device_put(params), optimizer, use_ema=True))
        step = make_train_step(model, optimizer, config, weights, settings,
                               batch_sharding=sharding if mode == "shard_map" else None)
        placed = {k: jax.device_put(v, sharding) for k, v in batch.items()}
        for _ in range(2):
            state, metrics = step(state, placed, jax.random.key(3))
        predict = make_predict_step(model, spec.num_select, sharding if mode == "shard_map" else None)
        scores, labels, boxes = predict(state.ema_params, placed["pixel_values"], jnp.full((4, 2), 32.0))
        results[mode] = {
            "metrics": {k: float(v) for k, v in metrics.items()},
            "params": [np.asarray(x).ravel().tolist() for x in jax.tree.leaves(state.params)],
            "ema": [np.asarray(x).ravel().tolist() for x in jax.tree.leaves(state.ema_params)],
            "scores": np.asarray(scores).tolist(),
            "labels": np.asarray(labels).tolist(),
            "devices": jax.device_count(),
        }
    print("RESULT" + json.dumps(results))
    """
)


def test_shard_map_step_matches_partitioned_step() -> None:
    """Two steps, then predictions, agree between the two data-parallel implementations."""
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    output = subprocess.run(
        [sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, timeout=600, check=True
    ).stdout
    results = json.loads(output.split("RESULT", 1)[1])
    gspmd, explicit = results["gspmd"], results["shard_map"]
    assert gspmd["devices"] == 4

    for name, value in gspmd["metrics"].items():
        assert abs(explicit["metrics"][name] - value) <= 1e-5 * max(1.0, abs(value)), name
    for key in ("params", "ema"):
        for left, right in zip(gspmd[key], explicit[key], strict=True):
            worst = max((abs(a - b) for a, b in zip(left, right, strict=True)), default=0.0)
            assert worst <= 1e-5, (key, worst)
    assert gspmd["labels"] == explicit["labels"]
    worst_score = max(abs(a - b) for row_a, row_b in zip(gspmd["scores"], explicit["scores"], strict=True)
                      for a, b in zip(row_a, row_b, strict=True))
    assert worst_score <= 1e-5
