"""Compare the JAX keypoint head, losses and matching costs against the PyTorch reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")

from parity_utils import compare_named  # noqa: E402

from rfdetr_jax.convert import convert_state_dict  # noqa: E402
from rfdetr_jax.models.heads.keypoints import (  # noqa: E402
    ConditionalQueryInitializer,
    compute_keypoint_matching_cost,
    compute_l1_keypoint_loss,
)


def main() -> int:
    """Run each keypoint primitive on the recorded inputs and report the error."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="artifacts/torch_keypoint_ref.npz")
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()
    if not np.isfinite(args.tol) or args.tol < 0:
        parser.error("--tol must be finite and nonnegative")

    data = np.load(args.bundle)
    schema = [int(value) for value in data["schema"]]
    actual = {}

    initializer = ConditionalQueryInitializer(
        dim=int(data["dim"]), num_queries=int(data["num_queries"]), out_dim=int(data["out_dim"])
    )
    conditioning = jnp.asarray(data["conditioning"])
    template = initializer.init(jax.random.key(0), conditioning)["params"]
    state_dict = {key[4:]: data[key] for key in data.files if key.startswith("sd::")}
    params, missing, unexpected = convert_state_dict(state_dict, template, strict=False)
    print(f"initializer: converted {len(state_dict)} tensors; missing={missing} unexpected={unexpected}")
    if missing or unexpected:
        return 1

    actual["queries_out"] = initializer.apply({"params": params}, conditioning)

    losses = compute_l1_keypoint_loss(
        jnp.asarray(data["predictions"]),
        jnp.asarray(data["target_keypoints"]),
        jnp.asarray(data["target_classes"]),
        jnp.asarray(data["target_areas"]),
        schema,
    )
    for name, value in zip(("location", "findable", "visible", "nll"), losses, strict=True):
        actual[f"loss_{name}"] = value

    costs = compute_keypoint_matching_cost(
        jnp.asarray(data["cost_predictions"]),
        jnp.asarray(data["target_keypoints"]),
        jnp.asarray(data["target_classes"]),
        jnp.asarray(data["target_areas"]),
        schema,
    )
    for name, value in zip(("l1", "findable", "visible", "nll"), costs, strict=True):
        actual[f"cost_{name}"] = value

    reference = {
        key: data[key] for key in data.files
        if key == "queries_out" or key.startswith(("loss_", "cost_")) and key != "cost_predictions"
    }
    return 1 if compare_named(actual, reference, args.tol) else 0


if __name__ == "__main__":
    raise SystemExit(main())
