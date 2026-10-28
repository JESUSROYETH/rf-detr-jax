"""Compare the JAX segmentation head against the PyTorch reference bundle."""

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
from rfdetr_jax.models.heads.segmentation import SegmentationHead  # noqa: E402


def main() -> int:
    """Load the bundle, run the JAX head, and report per-output errors."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="artifacts/torch_seg_ref.npz")
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()
    if not np.isfinite(args.tol) or args.tol < 0:
        parser.error("--tol must be finite and nonnegative")

    data = np.load(args.bundle)
    in_dim = int(data["in_dim"])
    num_blocks = int(data["num_blocks"])
    image_size = tuple(int(value) for value in data["image_size"])
    head = SegmentationHead(in_dim, num_blocks, downsample_ratio=int(data["downsample_ratio"]))

    spatial = jnp.asarray(data["spatial"])
    queries = [jnp.asarray(data[f"query_{index}"]) for index in range(num_blocks)]
    template = head.init(jax.random.key(0), spatial, queries, image_size)["params"]

    state_dict = {key[4:]: data[key] for key in data.files if key.startswith("sd::")}
    params, missing, unexpected = convert_state_dict(state_dict, template, strict=False)
    print(f"converted {len(state_dict)} tensors; missing={missing} unexpected={unexpected}")
    if missing or unexpected:
        return 1

    dense = head.apply({"params": params}, spatial, queries, image_size)
    actual = {f"out_{index}": output for index, output in enumerate(dense)}
    skipped = head.apply({"params": params}, spatial, queries[:1], image_size, skip_blocks=True)
    if len(skipped) != 1:
        print("  FAIL skip_blocks must produce exactly one output")
        return 1
    actual["out_skip"] = skipped[0]
    reference = {key: data[key] for key in data.files if key.startswith("out_")}
    return 1 if compare_named(actual, reference, args.tol) else 0


if __name__ == "__main__":
    raise SystemExit(main())
