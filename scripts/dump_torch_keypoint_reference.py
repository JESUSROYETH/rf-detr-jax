"""Dump PyTorch keypoint-head references (initializer, losses, matching costs) for parity checks."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from rfdetr.models.heads.keypoints import (  # noqa: E402
    ConditionalQueryInitializer,
    compute_keypoint_matching_cost,
    compute_l1_keypoint_loss,
)


def main() -> None:
    """Run each keypoint primitive once and save its inputs, weights and outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/torch_keypoint_ref.npz")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dim, num_queries, out_dim = 16, 5, 16
    initializer = ConditionalQueryInitializer(dim=dim, num_queries=num_queries, out_dim=out_dim)
    with torch.no_grad():
        for parameter in initializer.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)
    initializer.eval()

    generator = torch.Generator().manual_seed(args.seed + 1)
    conditioning = torch.randn(2, 4, dim, generator=generator)
    with torch.no_grad():
        queries = initializer(conditioning)

    schema = [3, 2]
    per_class = max(schema)
    total = len(schema) * per_class
    targets = 4
    predictions = torch.randn(targets, total, 7, generator=generator)
    target_keypoints = torch.rand(targets, per_class, 3, generator=generator)
    target_keypoints[:, :, 2] = torch.tensor([[2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
    target_classes = torch.tensor([0, 1, 0, 1])
    target_areas = torch.tensor([0.05, 0.2, 0.01, 0.5])

    losses = compute_l1_keypoint_loss(predictions, target_keypoints, target_classes, target_areas, schema)

    batch, queries_count = 2, 3
    cost_predictions = torch.randn(batch, queries_count, total, 7, generator=generator)
    costs = compute_keypoint_matching_cost(
        cost_predictions, target_keypoints, target_classes, target_areas, schema
    )

    bundle = {
        "conditioning": conditioning.numpy(),
        "queries_out": queries.numpy(),
        "dim": np.asarray(dim),
        "num_queries": np.asarray(num_queries),
        "out_dim": np.asarray(out_dim),
        "schema": np.asarray(schema),
        "predictions": predictions.numpy(),
        "target_keypoints": target_keypoints.numpy(),
        "target_classes": target_classes.numpy(),
        "target_areas": target_areas.numpy(),
        "cost_predictions": cost_predictions.numpy(),
    }
    for name, value in zip(("location", "findable", "visible", "nll"), losses, strict=True):
        bundle[f"loss_{name}"] = value.detach().numpy()
    for name, value in zip(("l1", "findable", "visible", "nll"), costs, strict=True):
        bundle[f"cost_{name}"] = value.detach().numpy()
    for key, value in initializer.state_dict().items():
        bundle[f"sd::{key}"] = value.numpy()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **bundle)
    print(f"wrote {args.out} ({len(bundle)} arrays)")


if __name__ == "__main__":
    main()
