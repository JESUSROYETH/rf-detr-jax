"""Compare a single training-mode forward + loss between PyTorch and JAX on identical inputs.

Two modes matter here. With ``group_detr=1`` the two-stage encoder makes one selection per image,
and the whole step should agree to float32 noise. With the configured ``group_detr`` there are that
many independent 300-of-N selections per image, and near-tied ranks re-pair proposals with
different learned query slots — the residual this script quantifies.

Pass ``--framework torch`` under the torch environment first, then ``--framework jax``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parity_utils import compare_named  # noqa: E402
from shared_run_config import RunConfig  # noqa: E402

from rfdetr_jax.datasets.coco import CocoDetection, iterate_batches  # noqa: E402


def make_batch(run: RunConfig) -> dict:
    """Return the first deterministic training batch of the shared stream."""
    dataset = CocoDetection(*run.train_paths())
    return next(
        iterate_batches(
            dataset, run.resolution, run.batch_size, run.max_targets, train=True, seed=run.data_seed
        )
    )


def run_torch(run: RunConfig, group_detr: int, weights: str, device: str = "cpu") -> dict[str, float]:
    """Compute the loss dict for one batch with the upstream PyTorch model."""
    import torch
    from rfdetr.config import RFDETRNanoConfig, TrainConfig
    from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config
    from rfdetr.utilities.tensors import NestedTensor

    torch.set_num_threads(4)
    model_config = RFDETRNanoConfig(
        pretrain_weights=None,
        device=device,
        resolution=run.resolution,
        positional_encoding_size=run.positional_encoding_size,
        num_classes=run.num_classes,
        group_detr=group_detr,
        num_queries=run.num_queries,
        num_select=run.num_select,
        dec_layers=run.dec_layers,
    )
    train_config = TrainConfig(dataset_dir=run.dataset_dir, output_dir="output")
    model = build_model_from_config(model_config, train_config)
    with np.load(weights) as data:
        state = {key: torch.from_numpy(data[key]) for key in data.files}
    # A `group_detr` override changes only the group-replicated query rows, so trim them.
    own = model.state_dict()
    trimmed = {
        key: value[: own[key].shape[0]] if key in own and value.shape != own[key].shape else value
        for key, value in state.items()
        if key in own
    }
    model.load_state_dict(trimmed, strict=False)
    model.to(device)
    model.train()

    criterion, _ = build_criterion_from_config(model_config, train_config)
    criterion.to(device)
    criterion.train()

    batch = make_batch(run)
    pixel_values = torch.from_numpy(batch["pixel_values"]).to(device)
    mask = torch.zeros((pixel_values.shape[0], *pixel_values.shape[2:]), dtype=torch.bool, device=device)
    targets = [
        {
            "labels": torch.from_numpy(batch["labels"][i][batch["target_mask"][i]].astype("int64")).to(device),
            "boxes": torch.from_numpy(batch["boxes"][i][batch["target_mask"][i]]).to(device),
        }
        for i in range(pixel_values.shape[0])
    ]
    with torch.no_grad():
        outputs = model(NestedTensor(pixel_values, mask), targets)
        losses = criterion(outputs, targets)
    weight_dict = criterion.weight_dict
    result = {name: float(value) for name, value in losses.items() if name in weight_dict}
    result["total"] = float(sum(losses[name] * weight_dict[name] for name in result if name in weight_dict))
    return result


def run_jax(run: RunConfig, group_detr: int, weights: str) -> dict[str, float]:
    """Compute the loss dict for one batch with the JAX port."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_default_matmul_precision", "highest")

    from train_jax import spec_for

    from rfdetr_jax.convert import convert_state_dict
    from rfdetr_jax.models.criterion import CriterionConfig, Targets, compute_losses
    from rfdetr_jax.models.lwdetr import LWDETR
    from rfdetr_jax.training.trainer import init_params

    spec = spec_for(run).__class__(**{**spec_for(run).__dict__, "group_detr": group_detr})
    model = LWDETR(spec=spec)
    params = init_params(model, spec, jax.random.key(run.seed))
    with np.load(weights) as data:
        state_dict = {key: data[key] for key in data.files}
    from rfdetr_jax.convert import flatten_params

    shapes = {".".join(path): value.shape for path, value in flatten_params(params).items()}
    from rfdetr_jax.convert import torch_key_to_jax_path

    trimmed = {}
    for key, value in state_dict.items():
        path = torch_key_to_jax_path(key)
        if path is None:
            continue
        target = shapes.get(".".join(path))
        if target is None:
            # A `group_detr` override drops the extra per-group encoder heads entirely.
            continue
        trimmed[key] = value[: target[0]] if value.shape != target else value
    params, missing, unexpected = convert_state_dict(trimmed, params, strict=False)
    if missing or unexpected:
        raise SystemExit(f"conversion incomplete: missing={missing[:3]} unexpected={unexpected[:3]}")

    batch = make_batch(run)
    outputs = model.apply({"params": params}, jnp.asarray(batch["pixel_values"]), None, train=True)
    targets = Targets(
        labels=jnp.asarray(batch["labels"]),
        boxes=jnp.asarray(batch["boxes"]),
        mask=jnp.asarray(batch["target_mask"]),
    )
    config = CriterionConfig(num_classes=spec.detector_classes, group_detr=group_detr)
    losses = compute_losses(outputs, targets, config, train=True)
    weights_map = config.weight_dict(spec.dec_layers)
    result = {name: float(value) for name, value in losses.items()}
    result["total"] = float(sum(losses[name] * weights_map[name] for name in losses if name in weights_map))
    return result


def main() -> int:
    """Run one framework and either save or compare its loss dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", required=True, choices=["torch", "jax"])
    parser.add_argument("--group-detr", type=int, default=1)
    parser.add_argument("--weights", default="artifacts/init_state_dict.npz")
    parser.add_argument("--out", default=None)
    parser.add_argument("--compare", default=None)
    parser.add_argument("--device", default="cpu", help="torch only: 'cpu' or 'cuda'.")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="Comparison tolerance scaled by max(1, abs(reference)).")
    args = parser.parse_args()
    if not np.isfinite(args.tol) or args.tol < 0:
        parser.error("--tol must be finite and nonnegative")

    run = RunConfig()
    losses = (
        run_torch(run, args.group_detr, args.weights, args.device)
        if args.framework == "torch"
        else run_jax(run, args.group_detr, args.weights)
    )

    if not losses or any(not np.isfinite(value) for value in losses.values()):
        parser.error("framework returned empty or nonfinite losses")
    for name in sorted(losses):
        print(f"  {name:16s} {losses[name]:.6f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(losses, indent=2))
        print(f"wrote {args.out}")

    if args.compare:
        reference = json.loads(Path(args.compare).read_text())
        return 1 if compare_named(losses, reference, args.tol, relative=True) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
