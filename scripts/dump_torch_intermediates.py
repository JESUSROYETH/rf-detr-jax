"""Dump intermediate activations from the PyTorch RF-DETR so the JAX port can be bisected.

Run with the torch reference environment.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from rfdetr.config import RFDETRNanoConfig, TrainConfig  # noqa: E402
from rfdetr.models.lwdetr import build_model_from_config  # noqa: E402
from rfdetr.utilities.tensors import NestedTensor  # noqa: E402


def main() -> None:
    """Register forward hooks on the interesting submodules and dump their outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="artifacts/torch_ref_nano.npz")
    parser.add_argument("--out", default="artifacts/torch_intermediates.npz")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = np.load(args.bundle, allow_pickle=True)
    state_dict = {key[4:]: torch.from_numpy(source[key]) for key in source.files if key.startswith("sd::")}
    meta = {key[6:]: str(source[key]) for key in source.files if key.startswith("meta::")}

    torch.manual_seed(args.seed)
    model_config = RFDETRNanoConfig(
        pretrain_weights=None,
        device="cpu",
        resolution=int(meta["resolution"]),
        positional_encoding_size=int(meta["positional_encoding_size"]),
        num_classes=int(meta["num_classes"]),
        group_detr=int(meta["group_detr"]),
    )
    model = build_model_from_config(model_config, TrainConfig(dataset_dir=".", output_dir="."))
    model.load_state_dict(state_dict)
    model.eval()

    captured: dict[str, np.ndarray] = {}

    def record(name: str, value) -> None:
        """Store one tensor (or the flattened elements of a sequence) under ``name``."""
        if isinstance(value, torch.Tensor):
            captured[name] = value.detach().cpu().numpy()
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                record(f"{name}.{index}", item)

    targets = {
        "patch_embeddings": model.backbone[0].encoder.encoder.embeddings.patch_embeddings,
        "embeddings": model.backbone[0].encoder.encoder.embeddings,
        "encoder_layer0": model.backbone[0].encoder.encoder.encoder.layer[0],
        "encoder_layer3": model.backbone[0].encoder.encoder.encoder.layer[3],
        "encoder_layer11": model.backbone[0].encoder.encoder.encoder.layer[11],
        "dino_backbone": model.backbone[0].encoder,
        "projector": model.backbone[0].projector,
        "transformer": model.transformer,
        "decoder": model.transformer.decoder,
        "decoder_layer0": model.transformer.decoder.layers[0],
        "class_embed": model.class_embed,
        "bbox_embed": model.bbox_embed,
    }
    handles = [
        module.register_forward_hook(lambda _m, _i, output, name=name: record(name, output))
        for name, module in targets.items()
    ]

    pixel_values = torch.from_numpy(source["input_pixel_values"])
    pixel_mask = torch.from_numpy(source["input_pixel_mask"])
    with torch.no_grad():
        model(NestedTensor(pixel_values, pixel_mask))
    for handle in handles:
        handle.remove()

    # The backbone's position embeddings are produced by the Joiner, not a hooked submodule.
    with torch.no_grad():
        features, poss, _ = model.backbone(NestedTensor(pixel_values, pixel_mask))
    for index, (feat, pos) in enumerate(zip(features, poss, strict=True)):
        captured[f"joiner_feat.{index}"] = feat.tensors.numpy()
        captured[f"joiner_mask.{index}"] = feat.mask.numpy()
        captured[f"joiner_pos.{index}"] = pos.numpy()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **captured)
    print(f"wrote {args.out}: {len(captured)} arrays")
    for key in sorted(captured):
        print(f"  {key:34s} {captured[key].shape}")


if __name__ == "__main__":
    main()
