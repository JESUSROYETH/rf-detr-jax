"""Compare the JAX criterion against the PyTorch SetCriterion reference bundle."""

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

from rfdetr_jax.models.criterion import CriterionConfig, Targets, compute_losses  # noqa: E402


def replay_mask_draws(data, aux_layers: int) -> dict[str, dict]:
    """Assign the recorded ``torch.rand`` rows to layers in upstream's call order.

    Upstream matches every layer first (one ``(1, P, 2)`` draw each, main, aux..., encoder) and
    then computes each layer's mask loss (a ``(N, S, 2)`` candidate draw followed by a
    ``(N, R, 2)`` random draw).
    """
    suffixes = ["", *(f"_{index}" for index in range(aux_layers)), "_enc"]
    draws = []
    index = 0
    while f"rand::{index}" in data.files:
        role = str(data[f"rand_role::{index}"]) if f"rand_role::{index}" in data.files else None
        shape = tuple(int(v) for v in data[f"rand_shape::{index}"])
        draws.append((role or ("matcher" if shape[0] == 1 else "loss"), jnp.asarray(data[f"rand::{index}"])))
        index += 1
    cost_draws = [row for role, row in draws if role == "matcher"]
    loss_draws = [row for role, row in draws if role == "loss"]
    if len(cost_draws) != len(suffixes) or len(loss_draws) != 2 * len(suffixes):
        raise ValueError(f"expected {len(suffixes)} matcher draws and {2 * len(suffixes)} loss draws, "
                         f"found {len(cost_draws)} and {len(loss_draws)}")
    return {
        suffix: {"cost_points": cost_draws[position],
                 "loss_draws": (loss_draws[2 * position], loss_draws[2 * position + 1])}
        for position, suffix in enumerate(suffixes)
    }


def main() -> int:
    """Recompute every loss in JAX on the recorded batch and compare with torch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="artifacts/torch_loss_ref.npz")
    parser.add_argument("--group-detr", type=int, default=1)
    parser.add_argument("--ia-bce", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()
    if not np.isfinite(args.tol) or args.tol < 0:
        parser.error("--tol must be finite and nonnegative")

    data = np.load(args.bundle)
    outputs = {
        "pred_logits": jnp.asarray(data["pred_logits"]),
        "pred_boxes": jnp.asarray(data["pred_boxes"]),
        "aux_outputs": [],
        "enc_outputs": {
            "pred_logits": jnp.asarray(data["enc_pred_logits"]),
            "pred_boxes": jnp.asarray(data["enc_pred_boxes"]),
        },
    }
    index = 0
    while f"aux{index}_pred_logits" in data.files:
        outputs["aux_outputs"].append(
            {
                "pred_logits": jnp.asarray(data[f"aux{index}_pred_logits"]),
                "pred_boxes": jnp.asarray(data[f"aux{index}_pred_boxes"]),
            }
        )
        index += 1
    segmentation = "target_masks" in data.files
    sampling_overrides = None
    if segmentation:
        outputs["pred_masks"] = jnp.asarray(data["pred_masks"])
        outputs["enc_outputs"]["pred_masks"] = jnp.asarray(data["enc_pred_masks"])
        for position in range(len(outputs["aux_outputs"])):
            outputs["aux_outputs"][position]["pred_masks"] = jnp.asarray(data[f"aux{position}_pred_masks"])
        sampling_overrides = replay_mask_draws(data, len(outputs["aux_outputs"]))

    schema = tuple(int(value) for value in data["schema"]) if "schema" in data.files else ()
    if schema:
        outputs["pred_keypoints"] = jnp.asarray(data["pred_keypoints"])
        outputs["enc_outputs"]["pred_keypoints"] = jnp.asarray(data["enc_pred_keypoints"])
        for position in range(len(outputs["aux_outputs"])):
            outputs["aux_outputs"][position]["pred_keypoints"] = jnp.asarray(
                data[f"aux{position}_pred_keypoints"]
            )

    targets = Targets(
        labels=jnp.asarray(data["target_labels"].astype("int32")),
        boxes=jnp.asarray(data["target_boxes"]),
        mask=jnp.asarray(data["target_mask"]),
        masks=jnp.asarray(data["target_masks"]) if segmentation else None,
        keypoints=jnp.asarray(data["target_keypoints"]) if schema else None,
    )
    config = CriterionConfig(
        num_classes=int(data["pred_logits"].shape[-1]),
        group_detr=args.group_detr,
        ia_bce_loss=bool(args.ia_bce),
        segmentation_head=segmentation,
        mask_ce_loss_coef=5.0,
        mask_dice_loss_coef=5.0,
        mask_point_sample_ratio=16,
        use_grouppose_keypoints=bool(schema),
        num_keypoints_per_class=schema,
        keypoint_l1_loss_coef=5.0 if schema else 0.0,
        keypoint_findable_loss_coef=1.0 if schema else 0.0,
        keypoint_visible_loss_coef=1.0 if schema else 0.0,
        keypoint_nll_loss_coef=0.5 if schema else 0.0,
    )
    actual = compute_losses(outputs, targets, config, train=args.group_detr > 1,
                            sampling_overrides=sampling_overrides)

    # Upstream logs classification accuracy beside its loss terms; this port's criterion
    # returns optimization losses only. Exempt this one named diagnostic, not unknown losses.
    if "loss::class_error" in data.files:
        print("  INFO upstream class_error accuracy diagnostic is outside loss parity")
    reference = {
        key[6:]: data[key] for key in data.files
        if key.startswith("loss::") and key != "loss::class_error"
    }
    failures = compare_named(actual, reference, args.tol, relative=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
