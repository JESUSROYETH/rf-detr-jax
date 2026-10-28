"""Regenerate ``tests/data/interpolate_reference.npz`` from ``torch.nn.functional.interpolate``.

Run with a PyTorch environment. The fixture keeps its stored input ``x``; every case is recomputed
and compared with the existing entry so an accidental change is reported rather than silently
rewritten. ``--write`` replaces the file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

CASES = {
    "bilinear|False|False|16x16": ("bilinear", False, False, (16, 16)),
    "bilinear|False|False|5x5": ("bilinear", False, False, (5, 5)),
    "bilinear|True|False|5x5": ("bilinear", True, False, (5, 5)),
    "bicubic|False|True|4x4": ("bicubic", False, True, (4, 4)),
    "bicubic|False|False|16x16": ("bicubic", False, False, (16, 16)),
    "bicubic|False|True|3x5": ("bicubic", False, True, (3, 5)),
    "nearest|False|False|4x4": ("nearest", False, False, (4, 4)),
    "nearest|False|False|5x13": ("nearest", False, False, (5, 13)),
    "bicubic|False|True|16x16": ("bicubic", False, True, (16, 16)),
    "bicubic|False|True|13x5": ("bicubic", False, True, (13, 5)),
    "bilinear|False|True|16x16": ("bilinear", False, True, (16, 16)),
    "bilinear|False|True|12x6": ("bilinear", False, True, (12, 6)),
    "bicubic|False|True|6x4": ("bicubic", False, True, (6, 4)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_fixture = Path(__file__).resolve().parents[1] / "tests/data/interpolate_reference.npz"
    parser.add_argument("--fixture", type=Path, default=default_fixture)
    parser.add_argument("--write", action="store_true", help="Replace the fixture instead of only comparing.")
    args = parser.parse_args()
    existing = dict(np.load(args.fixture)) if args.fixture.exists() else {}
    x = existing.get("x")
    if x is None:
        x = np.random.RandomState(0).normal(size=(1, 2, 8, 8)).astype(np.float32)
    output = {"x": x}
    mismatches = 0
    for name, (mode, align_corners, antialias, size) in CASES.items():
        kwargs = {} if mode == "nearest" else {"align_corners": align_corners}
        if antialias:
            kwargs["antialias"] = True
        value = torch.nn.functional.interpolate(torch.from_numpy(x), size=size, mode=mode, **kwargs).numpy()
        output[name] = value.astype(np.float32)
        if name in existing and not np.array_equal(existing[name], output[name]):
            mismatches += 1
            print(f"  CHANGED {name}: maxabs {np.abs(existing[name] - output[name]).max():.3e}")
    print(f"{len(CASES)} cases from torch {torch.__version__}; {mismatches} differ from the stored fixture")
    if args.write:
        np.savez(args.fixture, **output)
        print(f"wrote {args.fixture}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
