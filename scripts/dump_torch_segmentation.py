"""Dump a PyTorch ``SegmentationHead`` with random weights, its inputs and its outputs."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from rfdetr.models.heads.segmentation import SegmentationHead  # noqa: E402


def main() -> None:
    """Run the head once and save everything the JAX parity check needs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/torch_seg_ref.npz")
    parser.add_argument("--in-dim", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--downsample-ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    head = SegmentationHead(args.in_dim, args.num_blocks, downsample_ratio=args.downsample_ratio)
    # Random rather than default init, so a zero-initialised parameter cannot mask a mapping error.
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)
    head.eval()

    generator = torch.Generator().manual_seed(args.seed + 1)
    spatial = torch.randn(2, args.in_dim, 12, 12, generator=generator)
    queries = [torch.randn(2, 5, args.in_dim, generator=generator) for _ in range(args.num_blocks)]
    image_size = (96, 96)

    with torch.no_grad():
        dense = head(spatial, queries, image_size)
        skipped = head(spatial, queries[:1], image_size, skip_blocks=True)

    bundle: dict[str, np.ndarray] = {
        "spatial": spatial.numpy(),
        "image_size": np.asarray(image_size),
        "in_dim": np.asarray(args.in_dim),
        "num_blocks": np.asarray(args.num_blocks),
        "downsample_ratio": np.asarray(args.downsample_ratio),
    }
    for index, query in enumerate(queries):
        bundle[f"query_{index}"] = query.numpy()
    for index, output in enumerate(dense):
        bundle[f"out_{index}"] = output.numpy()
    bundle["out_skip"] = skipped[0].numpy()
    for key, value in head.state_dict().items():
        bundle[f"sd::{key}"] = value.numpy()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **bundle)
    print(f"wrote {args.out} ({len(bundle)} arrays)")


if __name__ == "__main__":
    main()
