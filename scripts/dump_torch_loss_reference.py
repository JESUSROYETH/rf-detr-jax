"""Dump PyTorch SetCriterion losses for a fixed random batch, for the JAX criterion parity test."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from rfdetr.models.criterion import SetCriterion  # noqa: E402
from rfdetr.models.matcher import HungarianMatcher  # noqa: E402


def main() -> None:
    """Build random outputs/targets, evaluate every loss, and save both sides to an npz."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/torch_loss_ref.npz")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--classes", type=int, default=91)
    parser.add_argument("--max-targets", type=int, default=6)
    parser.add_argument("--group-detr", type=int, default=1)
    parser.add_argument("--ia-bce", type=int, default=1)
    parser.add_argument(
        "--keypoints",
        default="",
        help="Comma-separated keypoints-per-class schema, e.g. '3,2'. Empty disables keypoint losses.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--segmentation", action="store_true",
                        help="Add mask logits/targets, upstream's mask matching costs and mask losses; "
                             "records every torch.rand draw so the JAX side can replay the sampled points.")
    parser.add_argument("--mask-size", type=int, default=16, help="Side of the predicted mask logit maps.")
    parser.add_argument("--target-mask-size", type=int, default=32, help="Side of the ground-truth masks.")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    batch, queries, classes = args.batch, args.queries, args.classes

    schema = [int(value) for value in args.keypoints.split(",") if value]
    padded_keypoints = len(schema) * max(schema) if schema else 0

    def make_layer() -> dict[str, torch.Tensor]:
        """Draw one layer of random logits, boxes and (optionally) keypoints."""
        logits = rng.randn(batch, queries, classes).astype("float32")
        centers = rng.uniform(0.2, 0.8, size=(batch, queries, 2)).astype("float32")
        sizes = rng.uniform(0.05, 0.3, size=(batch, queries, 2)).astype("float32")
        boxes = np.concatenate([centers, sizes], axis=-1)
        layer = {"pred_logits": torch.from_numpy(logits), "pred_boxes": torch.from_numpy(boxes)}
        if args.segmentation:
            layer["pred_masks"] = torch.from_numpy(
                (rng.randn(batch, queries, args.mask_size, args.mask_size) * 3).astype("float32")
            )
        if schema:
            layer["pred_keypoints"] = torch.from_numpy(
                rng.randn(batch, queries, padded_keypoints, 8).astype("float32")
            )
        return layer

    outputs = make_layer()
    outputs["aux_outputs"] = [make_layer() for _ in range(2)]
    outputs["enc_outputs"] = make_layer()

    counts = [3, args.max_targets]
    targets = []
    padded_labels = np.zeros((batch, args.max_targets), dtype="int64")
    padded_boxes = np.zeros((batch, args.max_targets, 4), dtype="float32")
    padded_mask = np.zeros((batch, args.max_targets), dtype=bool)
    keypoints_per_target = max(schema) if schema else 0
    padded_keypoint_targets = np.zeros((batch, args.max_targets, keypoints_per_target, 3), dtype="float32")
    padded_masks = np.zeros((batch, args.max_targets, args.target_mask_size, args.target_mask_size), dtype="float32")
    for image in range(batch):
        count = counts[image % len(counts)]
        # Keypoint classes must index the schema, so keep labels inside its range.
        label_ceiling = len(schema) if schema else classes
        labels = rng.randint(0, label_ceiling, size=(count,)).astype("int64")
        centers = rng.uniform(0.2, 0.8, size=(count, 2)).astype("float32")
        sizes = rng.uniform(0.05, 0.3, size=(count, 2)).astype("float32")
        boxes = np.concatenate([centers, sizes], axis=-1)
        target = {"labels": torch.from_numpy(labels), "boxes": torch.from_numpy(boxes)}
        if args.segmentation:
            masks = (rng.rand(count, args.target_mask_size, args.target_mask_size) > 0.5).astype("float32")
            target["masks"] = torch.from_numpy(masks)
            padded_masks[image, :count] = masks
        if schema:
            keypoints = rng.uniform(0.1, 0.9, size=(count, keypoints_per_target, 3)).astype("float32")
            keypoints[:, :, 2] = rng.randint(0, 3, size=(count, keypoints_per_target))
            target["keypoints"] = torch.from_numpy(keypoints)
            padded_keypoint_targets[image, :count] = keypoints
        targets.append(target)
        padded_labels[image, :count] = labels
        padded_boxes[image, :count] = boxes
        padded_mask[image, :count] = True

    mask_kwargs = ({"cost_mask_ce": 5.0, "cost_mask_dice": 5.0, "mask_point_sample_ratio": 16}
                   if args.segmentation else {})
    matcher = HungarianMatcher(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        focal_alpha=0.25,
        **mask_kwargs,
        num_keypoints_per_class=schema,
        keypoint_l1_loss_coef=5.0 if schema else 0.0,
        keypoint_findable_loss_coef=1.0 if schema else 0.0,
        keypoint_visible_loss_coef=1.0 if schema else 0.0,
        keypoint_nll_loss_coef=0.5 if schema else 0.0,
    )
    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    losses_requested = ["labels", "boxes"]
    if args.segmentation:
        losses_requested.append("masks")
        weight_dict.update({"loss_mask_ce": 5.0, "loss_mask_dice": 5.0})
    if schema:
        losses_requested.append("keypoints")
        weight_dict.update(
            {
                "loss_keypoints_l1": 5.0,
                "loss_keypoints_findable": 1.0,
                "loss_keypoints_visible": 1.0,
                "loss_keypoints_nll": 0.5,
            }
        )
    criterion = SetCriterion(
        classes,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=0.25,
        losses=losses_requested,
        group_detr=args.group_detr,
        sum_group_losses=False,
        ia_bce_loss=bool(args.ia_bce),
        mask_point_sample_ratio=16,
        num_keypoints_per_class=schema,
    )
    criterion.train(args.group_detr > 1)

    # Upstream samples mask points with ``torch.rand``: once per layer in the matcher (one row shared
    # by the batch) and twice per layer in the mask loss (one row per matched mask). Recording a
    # single row per call and broadcasting it over the leading axis keeps every mask on the same
    # points, which lets the JAX side replay the draw regardless of its padded match layout.
    recorded_draws: list[tuple[tuple[int, ...], np.ndarray]] = []
    original_rand = torch.rand
    draw_generator = torch.Generator().manual_seed(args.seed + 7)

    def recording_rand(*size, **kwargs):
        import sys

        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else tuple(size)
        row = original_rand(*shape[1:], generator=draw_generator)
        # The caller's module tells matcher draws from loss draws without guessing from shapes.
        caller = Path(sys._getframe(1).f_code.co_filename).as_posix()
        role = "matcher" if caller.endswith("rfdetr/models/matcher.py") else "loss"
        recorded_draws.append((shape, role, row.numpy().copy()))
        return row.unsqueeze(0).expand(shape).clone().to(kwargs.get("device", "cpu"))

    if args.segmentation:
        torch.rand = recording_rand
    try:
        losses = criterion(outputs, targets)
    finally:
        torch.rand = original_rand

    bundle: dict[str, np.ndarray] = {
        "pred_logits": outputs["pred_logits"].numpy(),
        "pred_boxes": outputs["pred_boxes"].numpy(),
        "enc_pred_logits": outputs["enc_outputs"]["pred_logits"].numpy(),
        "enc_pred_boxes": outputs["enc_outputs"]["pred_boxes"].numpy(),
        "target_labels": padded_labels,
        "target_boxes": padded_boxes,
        "target_mask": padded_mask,
        "schema": np.asarray(schema, dtype="int64"),
    }
    if args.segmentation:
        bundle["target_masks"] = padded_masks
        bundle["pred_masks"] = outputs["pred_masks"].numpy()
        bundle["enc_pred_masks"] = outputs["enc_outputs"]["pred_masks"].numpy()
        for index, aux in enumerate(outputs["aux_outputs"]):
            bundle[f"aux{index}_pred_masks"] = aux["pred_masks"].numpy()
        for index, (shape, role, row) in enumerate(recorded_draws):
            bundle[f"rand::{index}"] = row
            bundle[f"rand_shape::{index}"] = np.asarray(shape, dtype="int64")
            bundle[f"rand_role::{index}"] = np.asarray(role)
    if schema:
        bundle["target_keypoints"] = padded_keypoint_targets
        bundle["pred_keypoints"] = outputs["pred_keypoints"].numpy()
        bundle["enc_pred_keypoints"] = outputs["enc_outputs"]["pred_keypoints"].numpy()
        for index, aux in enumerate(outputs["aux_outputs"]):
            bundle[f"aux{index}_pred_keypoints"] = aux["pred_keypoints"].numpy()
    for index, aux in enumerate(outputs["aux_outputs"]):
        bundle[f"aux{index}_pred_logits"] = aux["pred_logits"].numpy()
        bundle[f"aux{index}_pred_boxes"] = aux["pred_boxes"].numpy()
    for key, value in losses.items():
        bundle[f"loss::{key}"] = value.detach().numpy()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **bundle)
    print(f"wrote {args.out}")
    for key in sorted(losses):
        print(f"  {key:20s} {float(losses[key]):.6f}")


if __name__ == "__main__":
    main()
