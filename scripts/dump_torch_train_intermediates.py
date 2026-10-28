"""Dump train-mode transformer intermediates so the group-DETR path can be bisected."""

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
    """Hook the decoder and its first layer in train mode and dump their outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="artifacts/torch_ref_nano.npz")
    parser.add_argument("--out", default="artifacts/torch_train_intermediates.npz")
    parser.add_argument("--group-detr", type=int, default=None)
    args = parser.parse_args()

    source = np.load(args.bundle, allow_pickle=True)
    state_dict = {key[4:]: torch.from_numpy(source[key]) for key in source.files if key.startswith("sd::")}
    meta = {key[6:]: str(source[key]) for key in source.files if key.startswith("meta::")}

    model_config = RFDETRNanoConfig(
        pretrain_weights=None,
        device="cpu",
        resolution=int(meta["resolution"]),
        positional_encoding_size=int(meta["positional_encoding_size"]),
        num_classes=int(meta["num_classes"]),
        group_detr=args.group_detr if args.group_detr is not None else int(meta["group_detr"]),
    )
    model = build_model_from_config(model_config, TrainConfig(dataset_dir=".", output_dir="."))
    model.load_state_dict(state_dict)
    model.train(True)

    captured: dict[str, np.ndarray] = {}

    def record(name: str, value) -> None:
        """Store a tensor or the elements of a sequence under ``name``."""
        if isinstance(value, torch.Tensor):
            captured[name] = value.detach().cpu().numpy()
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                record(f"{name}.{index}", item)

    targets = {
        "transformer": model.transformer,
        "decoder": model.transformer.decoder,
        "decoder_layer0": model.transformer.decoder.layers[0],
        "decoder_layer0_selfattn": model.transformer.decoder.layers[0].self_attn,
        "decoder_layer0_crossattn": model.transformer.decoder.layers[0].cross_attn,
        "ref_point_head": model.transformer.decoder.ref_point_head,
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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **captured)
    print(f"wrote {args.out}")
    for key in sorted(captured):
        print(f"  {key:34s} {captured[key].shape}")


if __name__ == "__main__":
    main()
