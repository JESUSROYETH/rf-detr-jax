"""Build an RF-DETR model in PyTorch and dump weights, inputs and outputs for the JAX parity test.

Run with the torch reference environment: ``./.venv-torch/bin/python scripts/dump_torch_reference.py``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

#: Variants whose training config carries keypoint-specific fields.
_KEYPOINT_VARIANTS = {"keypoint"}


def _dense(value) -> np.ndarray:
    """Materialise a mask output, expanding the training-time sparse factorisation.

    In training the segmentation head returns the factors rather than the mask logits, to let the
    matcher sample points without building the full map. The dense form the JAX port returns is
    ``einsum("bchw,bnc->bnhw", spatial_features, query_features) + bias``.
    """
    if isinstance(value, dict):
        import torch

        value = (
            torch.einsum("bchw,bnc->bnhw", value["spatial_features"], value["query_features"])
            + value["bias"]
        )
    return value.detach().numpy()


def record_outputs(bundle: dict, out: dict, mode: str) -> None:
    """Record every detection/mask/keypoint output, including decoder auxiliaries."""
    groups = [("", out)]
    groups.extend((f"aux{index}_", aux) for index, aux in enumerate(out.get("aux_outputs", [])))
    if "enc_outputs" in out:
        groups.append(("enc_", out["enc_outputs"]))
    for prefix, values in groups:
        for name in ("pred_logits", "pred_boxes", "pred_masks", "pred_keypoints"):
            if name in values:
                bundle[f"{mode}_{prefix}{name}"] = _dense(values[name])


def main() -> None:
    """Dump a torch reference bundle for one RF-DETR variant."""
    import torch
    from rfdetr.config import (
        KeypointTrainConfig,
        RFDETRKeypointPreviewConfig,
        RFDETRMediumConfig,
        RFDETRNanoConfig,
        RFDETRSegNanoConfig,
        RFDETRSmallConfig,
        TrainConfig,
    )
    from rfdetr.models.lwdetr import build_model_from_config

    warnings.filterwarnings("ignore")
    configs = {
        "nano": RFDETRNanoConfig,
        "small": RFDETRSmallConfig,
        "medium": RFDETRMediumConfig,
        "seg-nano": RFDETRSegNanoConfig,
        "keypoint": RFDETRKeypointPreviewConfig,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="nano", choices=sorted(configs))
    parser.add_argument("--out", default="artifacts/torch_ref_nano.npz")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--group-detr", type=int, default=None)
    parser.add_argument(
        "--keypoints-per-class",
        default=None,
        help="Comma-separated keypoint slots per class label, e.g. '0,17' for the published keypoint checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-padding", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32",
                        help="Run the reference in float64 to separate rounding noise from formula differences.")
    parser.add_argument("--checkpoint", default=None, help="Load this .pth into the model before dumping.")
    parser.add_argument(
        "--perturb-zeros",
        action="store_true",
        help=(
            "Fill zero-initialised parameters with small random values. Several RF-DETR heads start "
            "at exactly zero (the box head's last layer, the keypoint head's last layer, the "
            "instance/keypoint layer scale), which makes their contribution vanish and lets a parity "
            "test pass without ever exercising them."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    overrides = {"pretrain_weights": None, "device": "cpu"}
    if args.resolution is not None:
        overrides["resolution"] = args.resolution
        overrides["positional_encoding_size"] = args.resolution // configs[args.variant]().patch_size
    if args.num_classes is not None:
        overrides["num_classes"] = args.num_classes
    if args.group_detr is not None:
        overrides["group_detr"] = args.group_detr
    if args.keypoints_per_class is not None:
        overrides["num_keypoints_per_class"] = [int(value) for value in args.keypoints_per_class.split(",")]
    model_config = configs[args.variant](**overrides)
    train_config = (
        KeypointTrainConfig(dataset_dir=".", output_dir=".")
        if args.variant in _KEYPOINT_VARIANTS
        else TrainConfig(dataset_dir=".", output_dir=".")
    )
    model = build_model_from_config(model_config, train_config)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded {args.checkpoint}: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  first missing:", missing[:5])
        if unexpected:
            print("  first unexpected:", unexpected[:5])
    if args.perturb_zeros:
        perturbed = 0
        with torch.no_grad():
            for parameter in model.parameters():
                if bool((parameter == 0).all()):
                    parameter.copy_(torch.randn_like(parameter) * 0.05)
                    perturbed += 1
        print(f"perturbed {perturbed} zero-initialised parameter tensors")
    model.eval()
    if args.dtype == "float64":
        model = model.double()

    resolution = model_config.resolution
    generator = torch.Generator().manual_seed(args.seed + 1)
    pixel_values = torch.randn(args.batch, 3, resolution, resolution, generator=generator)
    if args.dtype == "float64":
        pixel_values = pixel_values.double()
    pixel_mask = torch.zeros(args.batch, resolution, resolution, dtype=torch.bool)
    if args.batch > 1 and not args.no_padding:
        # Pad the second image on the right and bottom so the valid-ratio path is exercised.
        pixel_mask[1, resolution - 32 :, :] = True
        pixel_mask[1, :, resolution - 48 :] = True

    from rfdetr.utilities.tensors import NestedTensor

    bundle: dict[str, np.ndarray] = {}
    original_topk = torch.topk
    # Record the two-stage proposal selection (every ``torch.topk`` call in the forward is one
    # group's selection, or the batched ``[groups, batch, queries]`` selection) so the JAX check
    # can inject it and judge the rest of the network under identical proposals.
    selections: list[torch.Tensor] = []

    def recording_topk(*args, **kwargs):
        result = original_topk(*args, **kwargs)
        selections.append(result.indices.detach().cpu())
        return result

    for mode in ("eval", "train"):
        model.train(mode == "train")
        selections.clear()
        torch.topk = recording_topk
        try:
            with torch.no_grad():
                out = model(NestedTensor(pixel_values, pixel_mask))
        finally:
            torch.topk = original_topk
        record_outputs(bundle, out, mode)
        if selections:
            stacked = selections[0] if len(selections) == 1 and selections[0].dim() == 3 else torch.stack(selections)
            bundle[f"topk::{mode}"] = stacked.numpy().astype("int64")
    model.eval()

    bundle["input_pixel_values"] = pixel_values.numpy()
    bundle["input_pixel_mask"] = pixel_mask.numpy()
    for key, value in model.state_dict().items():
        bundle[f"sd::{key}"] = value.detach().cpu().numpy()

    meta = {
        "variant": args.variant,
        "dtype": args.dtype,
        "resolution": resolution,
        "num_classes": model_config.num_classes,
        "group_detr": model_config.group_detr,
        "patch_size": model_config.patch_size,
        "num_windows": model_config.num_windows,
        "dec_layers": model_config.dec_layers,
        "positional_encoding_size": model_config.positional_encoding_size,
        "sa_nheads": model_config.sa_nheads,
        "ca_nheads": model_config.ca_nheads,
        "dec_n_points": model_config.dec_n_points,
        "num_queries": model_config.num_queries,
        "hidden_dim": model_config.hidden_dim,
        "encoder": model_config.encoder,
        "projector_scale": ",".join(model_config.projector_scale),
        "out_feature_indexes": ",".join(str(v) for v in model_config.out_feature_indexes),
        "segmentation_head": model_config.segmentation_head,
        "use_grouppose_keypoints": model_config.use_grouppose_keypoints,
        "num_keypoints_per_class": ",".join(str(v) for v in model_config.num_keypoints_per_class),
        "grouppose_keypoint_dim_downscale": model_config.grouppose_keypoint_dim_downscale,
        "keypoint_cross_attn": model_config.keypoint_cross_attn,
        "inter_instance_kp_attn": model_config.inter_instance_kp_attn,
        "dual_projector": model_config.dual_projector,
        "dual_projector_kp_only": model_config.dual_projector_kp_only,
        "mask_downsample_ratio": model_config.mask_downsample_ratio,
    }
    for key, value in meta.items():
        bundle[f"meta::{key}"] = np.asarray(str(value))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **bundle)
    elements = sum(value.size for value in bundle.values() if value.dtype != object)
    print(f"wrote {args.out}: {len(bundle)} arrays, {elements:,} elems")


if __name__ == "__main__":
    main()
