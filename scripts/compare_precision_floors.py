"""Diagnose median float32 error relative to upstream's float32 error.

Use two dump_torch_reference.py bundles with identical weights, inputs, masks and model
configuration, differing only in float32/float64 precision. The port runs with upstream's
float64 proposal selection injected. Only whole query groups with identical upstream
selections in both precisions are compared, and their coverage is reported.
The ratio denominator is floored at float32 epsilon times max(1, reference magnitude)
over the compared elements, so a zero upstream median does not cause division by zero.

Exit zero means every recorded output has comparable, finite evidence and meets the median
error margin. It does not certify maximum error, excluded groups, native proposal selection,
end-to-end numerical parity or accuracy. Missing evidence is a failure. Keep strict parity
checks separate; this diagnostic cannot turn a failed strict check into a pass.

Run: python scripts/compare_precision_floors.py --float32 a.npz --float64 b.npz
"""

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
from check_parity import named_outputs, spec_from_meta  # noqa: E402

from rfdetr_jax.convert import convert_state_dict  # noqa: E402
from rfdetr_jax.models.lwdetr import LWDETR  # noqa: E402


def _rows_with_same_selection(selection_a: np.ndarray, selection_b: np.ndarray, queries: int) -> np.ndarray:
    """Return a ``[B, G * Q]`` boolean mask of query slots whose whole group kept the same proposals.

    Self-attention mixes every query of a group, so one reordered proposal changes all of that
    group's outputs; only groups whose selection agrees slot for slot are comparable.
    """
    selection_a, selection_b = np.asarray(selection_a), np.asarray(selection_b)
    if (selection_a.shape != selection_b.shape or selection_a.ndim != 3
            or not selection_a.size or selection_a.shape[-1] != queries):
        raise ValueError("proposal selections must have the same nonempty [G, B, Q] shape")
    for selection in (selection_a, selection_b):
        if not np.issubdtype(selection.dtype, np.integer) or np.any(selection < 0):
            raise ValueError("proposal selections must contain nonnegative integer indices")
        if np.any(np.diff(np.sort(selection, axis=-1), axis=-1) == 0):
            raise ValueError("proposal selections must contain distinct indices per group")
    groups, batch, _ = selection_a.shape
    same_group = (selection_a == selection_b).all(-1)  # [G, B]
    same = np.broadcast_to(same_group[..., None], (groups, batch, queries))
    return same.transpose(1, 0, 2).reshape(batch, groups * queries)


def validate_bundles(data32, data64, modes: list[str]) -> dict[str, np.ndarray]:
    """Reject mismatched evidence before initializing a model; return comparable query masks."""
    for data, dtype in ((data32, "float32"), (data64, "float64")):
        if str(data.get("meta::dtype", "")) != dtype:
            raise ValueError(f"expected a {dtype} reference bundle")
    meta32 = {key: str(data32[key]) for key in data32 if key.startswith("meta::") and key != "meta::dtype"}
    meta64 = {key: str(data64[key]) for key in data64 if key.startswith("meta::") and key != "meta::dtype"}
    if meta32 != meta64:
        raise ValueError("reference model metadata differs between bundles")
    for key in ("input_pixel_values", "input_pixel_mask"):
        if key not in data32 or key not in data64:
            raise ValueError(f"missing {key}")
        left, right = np.asarray(data32[key]), np.asarray(data64[key])
        if not left.size or not right.size or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"empty or nonfinite {key}")
        expected = right.astype(np.float32) if key == "input_pixel_values" else right
        if not np.array_equal(left, expected):
            raise ValueError(f"reference inputs differ: {key}")
        if key == "input_pixel_values" and (left.dtype != np.float32 or right.dtype != np.float64):
            raise ValueError("reference pixels must use their declared float32/float64 dtype")
    weights32 = {key for key in data32 if key.startswith("sd::")}
    weights64 = {key for key in data64 if key.startswith("sd::")}
    if not weights32 or weights32 != weights64:
        raise ValueError("missing or mismatched checkpoint keys")
    for key in sorted(weights32):
        left, right = np.asarray(data32[key]), np.asarray(data64[key])
        if (not np.isfinite(left).all() or not np.isfinite(right).all()
                or not np.array_equal(left, right.astype(left.dtype))):
            raise ValueError(f"different or nonfinite weights: {key}")
    common = {}
    for mode in modes:
        key = f"topk::{mode}"
        if key not in data32 or key not in data64:
            raise ValueError(f"missing {key}")
        selection = np.asarray(data64[key])
        if selection.ndim != 3:
            raise ValueError(f"invalid selection shape: {key}")
        common[mode] = _rows_with_same_selection(data32[key], selection, selection.shape[-1])
        if not common[mode].any():
            raise ValueError(f"{mode}: no comparable query groups")
        prefix = f"{mode}_"
        names32 = {key for key in data32 if key.startswith(prefix)}
        names64 = {key for key in data64 if key.startswith(prefix)}
        if names32 != names64 or not {prefix + "pred_logits", prefix + "pred_boxes"} <= names32:
            raise ValueError(f"{mode}: missing or mismatched reference outputs")
        for name in sorted(names32):
            left, right = np.asarray(data32[name]), np.asarray(data64[name])
            if (left.shape != right.shape or left.shape[:2] != common[mode].shape
                    or not left.size or not np.isfinite(left).all() or not np.isfinite(right).all()):
                raise ValueError(f"invalid shape, empty or nonfinite reference output: {name}")
    return common


def compare_outputs(named: dict, data32, data64, mode: str, common: np.ndarray, margin: float) -> int:
    """Check every output without dropping missing tensors or broadcasting incompatible shapes."""
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be finite and positive")
    if common.ndim != 2 or common.dtype != np.bool_ or not common.any():
        print(f"FAIL {mode}: no comparable query groups")
        return 1
    prefix = f"{mode}_"
    names = set(named) | {"pred_logits", "pred_boxes"}
    names |= {key[len(prefix):] for data in (data32, data64) for key in data if key.startswith(prefix)}
    failures = 0
    for name in sorted(names):
        key = prefix + name
        if name not in named or key not in data32 or key not in data64:
            print(f"  FAIL {key}: missing model or reference output")
            failures += 1
            continue
        actual, reference, truth = (np.asarray(value, dtype=np.float64)
                                    for value in (named[name], data32[key], data64[key]))
        if (actual.shape != truth.shape or reference.shape != truth.shape
                or truth.shape[:2] != common.shape or not truth.size):
            print(f"  FAIL {key}: empty or incompatible output shapes")
            failures += 1
            continue
        # Validate the full tensors, including excluded groups, before computing diagnostics.
        if not all(np.isfinite(value).all() for value in (actual, reference, truth)):
            print(f"  FAIL {key}: nonfinite output")
            failures += 1
            continue
        jax_compared = np.abs(actual - truth)[common].ravel()
        torch_compared = np.abs(reference - truth)[common].ravel()
        scale = float(np.max(np.abs(truth[common])))
        resolution = np.finfo(np.float32).eps * max(1.0, scale)
        jax_median, torch_median = float(np.median(jax_compared)), float(np.median(torch_compared))
        ratio = jax_median / max(torch_median, resolution)
        ok = ratio <= margin
        failures += not ok
        verdict = "WITHIN MEDIAN MARGIN" if ok else "FAIL MEDIAN MARGIN"
        print(f"  {verdict} {key}: median port {jax_median:.2e} / upstream {torch_median:.2e} "
              f"(ratio {ratio:.2f}); max port {jax_compared.max():.2e} / upstream {torch_compared.max():.2e}; "
              f"compared {jax_compared.size}/{truth.size} elements")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--float32", required=True)
    parser.add_argument("--float64", required=True)
    parser.add_argument("--modes", default="eval,train")
    parser.add_argument("--margin", type=float, default=2.0,
                        help="Diagnostic median-error ratio limit; does not bound maximum error or certify parity.")
    args = parser.parse_args()
    modes = args.modes.split(",")
    if any(mode not in {"eval", "train"} for mode in modes):
        parser.error("--modes must be eval, train, or eval,train")
    if not np.isfinite(args.margin) or args.margin <= 0:
        parser.error("--margin must be finite and positive")
    data32 = np.load(args.float32, allow_pickle=False)
    data64 = np.load(args.float64, allow_pickle=False)
    try:
        common_by_mode = validate_bundles(data32, data64, modes)
    except ValueError as exc:
        print(f"FAIL evidence: {exc}")
        return 1
    meta = {key[6:]: str(data64[key]) for key in data64.files if key.startswith("meta::")}

    spec = spec_from_meta(meta)
    model = LWDETR(spec=spec)
    pixel_values = jnp.asarray(data32["input_pixel_values"])
    pixel_mask = jnp.asarray(data32["input_pixel_mask"])
    template = model.init(jax.random.key(0), pixel_values[:1], pixel_mask[:1], train=True)["params"]
    state_dict = {key[4:]: data32[key] for key in data32.files if key.startswith("sd::")}
    params, missing, unexpected = convert_state_dict(state_dict, template, strict=False)
    if missing or unexpected:
        print(f"conversion problem: missing={len(missing)} unexpected={len(unexpected)}")
        return 1

    failures = 0
    for mode in modes:
        selection64 = data64[f"topk::{mode}"]
        queries = selection64.shape[-1]
        common = common_by_mode[mode]
        out = model.apply({"params": params}, pixel_values, pixel_mask, train=(mode == "train"),
                          proposal_indices=jnp.asarray(selection64))
        named = named_outputs(out)
        print(f"[{mode}] compared {int(common.sum()) // queries}/{common.size // queries} "
              "(group, image) pairs with unchanged upstream selection")
        failures += compare_outputs(named, data32, data64, mode, common, args.margin)
    print("\nMedian-error diagnostic only. Maximum errors and excluded groups are not qualified; "
          "native-forward strict parity must be checked separately.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
