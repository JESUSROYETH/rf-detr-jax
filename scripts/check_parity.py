"""Compare this JAX port against a PyTorch reference bundle produced by ``dump_torch_reference.py``.

Run with the JAX environment: ``./.venv-jax/bin/python scripts/check_parity.py``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# GPU matmuls default to TF32; parity against float32 PyTorch needs the full mantissa.
jax.config.update("jax_default_matmul_precision", "highest")

from rfdetr_jax.convert import convert_state_dict  # noqa: E402
from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec  # noqa: E402


def spec_from_meta(meta: dict[str, str]) -> LWDETRSpec:
    """Rebuild the port's model spec from the metadata the torch dumper recorded."""
    return LWDETRSpec(
        encoder=meta["encoder"],
        hidden_dim=int(meta["hidden_dim"]),
        patch_size=int(meta["patch_size"]),
        num_windows=int(meta["num_windows"]),
        dec_layers=int(meta["dec_layers"]),
        sa_nheads=int(meta["sa_nheads"]),
        ca_nheads=int(meta["ca_nheads"]),
        dec_n_points=int(meta["dec_n_points"]),
        num_queries=int(meta["num_queries"]),
        projector_scale=tuple(meta["projector_scale"].split(",")),
        out_feature_indexes=tuple(int(v) for v in meta["out_feature_indexes"].split(",")),
        resolution=int(meta["resolution"]),
        positional_encoding_size=int(meta["positional_encoding_size"]),
        num_classes=int(meta["num_classes"]),
        group_detr=int(meta["group_detr"]),
        segmentation_head=meta.get("segmentation_head", "False") == "True",
        mask_downsample_ratio=int(meta.get("mask_downsample_ratio", 4)),
        use_grouppose_keypoints=meta.get("use_grouppose_keypoints", "False") == "True",
        num_keypoints_per_class=tuple(
            int(value) for value in meta.get("num_keypoints_per_class", "").split(",") if value
        ),
        grouppose_keypoint_dim_downscale=int(meta.get("grouppose_keypoint_dim_downscale", 1)),
        keypoint_cross_attn=meta.get("keypoint_cross_attn", "True") == "True",
        inter_instance_kp_attn=meta.get("inter_instance_kp_attn", "False") == "True",
        dual_projector=meta.get("dual_projector", "False") == "True",
        dual_projector_kp_only=meta.get("dual_projector_kp_only", "False") == "True",
    )


def compare_outputs(named: dict, reference: dict, mode: str, tol: float) -> tuple[int, float]:
    """Check ordered outputs, treating missing, malformed or nonfinite tensors as failures.

    ``reference`` uses the dumper's mode-prefixed names. No query permutation or
    inferred near-tie exemption can change this strict verdict.
    """
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    prefix = f"{mode}_"
    expected = {key[len(prefix):]: reference[key] for key in reference if key.startswith(prefix)}
    names = set(named) | set(expected) | {"pred_logits", "pred_boxes"}
    failures = 0
    worst = 0.0
    for name in sorted(names):
        key = f"{prefix}{name}"
        if name not in named or name not in expected:
            absent = "model output" if name not in named else "reference bundle"
            print(f"  FAIL {key}: absent from {absent}")
            failures += 1
            continue
        actual = np.asarray(named[name])
        target = np.asarray(expected[name])
        if actual.shape != target.shape:
            print(f"  FAIL {key}: shape {actual.shape} vs {target.shape}")
            failures += 1
            continue
        if actual.size == 0 or target.size == 0:
            print(f"  FAIL {key}: empty tensor")
            failures += 1
            continue
        if not np.isfinite(actual).all() or not np.isfinite(target).all():
            print(f"  FAIL {key}: nonfinite values")
            failures += 1
            worst = float("inf")
            continue
        # Float64 subtraction avoids overflow in float32 diagnostics.
        error = float(np.max(np.abs(actual.astype(np.float64) - target.astype(np.float64)), initial=0.0))
        scale = float(np.max(np.abs(target.astype(np.float64)), initial=0.0))
        worst = max(worst, error)
        tolerated = error <= tol * max(1.0, scale)
        failures += not tolerated
        status = "OK  " if tolerated else "FAIL"
        print(f"  {status} {key:28s} maxabs={error:.3e} (ref scale {scale:.2f})")
    return failures, worst


def compare_selection(reference_indices: np.ndarray, actual_indices: np.ndarray, scores: np.ndarray, tol: float):
    """Judge the two-stage selection without depending on the order of tied proposals.

    Both selections are scored with the JAX per-position scores: a selection is equivalent when
    the sorted selected scores agree within ``tol`` (relative to the score scale), which holds
    when the frameworks ranked near-tied proposals differently but chose equally good ones.

    Returns:
        ``(failures, worst_score_gap, differing_indices)``.
    """
    reference_indices = np.asarray(reference_indices)
    actual_indices = np.asarray(actual_indices)
    if reference_indices.shape != actual_indices.shape:
        print(f"  FAIL selection shape {actual_indices.shape} vs {reference_indices.shape}")
        return 1, float("inf"), -1
    groups, batch, queries = reference_indices.shape
    positions = scores.shape[-1]
    worst, differing = 0.0, 0
    failures = 0
    for group in range(groups):
        for image in range(batch):
            row = np.asarray(scores[group, image], dtype=np.float64)
            candidates = (("reference", reference_indices[group, image]), ("actual", actual_indices[group, image]))
            for label, indices in candidates:
                if indices.min() < 0 or indices.max() >= positions or len(set(indices.tolist())) != queries:
                    print(f"  FAIL {label} selection for group {group}, image {image}: indices must be distinct "
                          f"positions in [0, {positions})")
                    failures += 1
            if failures:
                continue
            chosen = np.sort(row[actual_indices[group, image]])
            expected = np.sort(row[reference_indices[group, image]])
            worst = max(worst, float(np.max(np.abs(chosen - expected), initial=0.0)))
            differing += len(set(reference_indices[group, image].tolist()) ^ set(actual_indices[group, image].tolist()))
    scale = float(np.max(np.abs(scores), initial=0.0))
    failures += int(worst > tol * max(1.0, scale))
    status = "OK  " if not failures else "FAIL"
    print(f"  {status} proposal selection: sorted selected-score gap {worst:.3e} (ref scale {scale:.2f}); "
          f"{differing} index differences across {groups * batch} (group, image) pairs")
    return failures, worst, differing


def named_outputs(out: dict) -> dict:
    """Flatten a model output dict into the dumper's tensor names (without the mode prefix)."""
    named = {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
    for extra in ("pred_masks", "pred_keypoints"):
        if extra in out:
            named[extra] = out[extra]
    for index, aux in enumerate(out.get("aux_outputs", [])):
        named[f"aux{index}_pred_logits"] = aux["pred_logits"]
        named[f"aux{index}_pred_boxes"] = aux["pred_boxes"]
        for extra in ("pred_masks", "pred_keypoints"):
            if extra in aux:
                named[f"aux{index}_{extra}"] = aux[extra]
    if "enc_outputs" in out:
        named["enc_pred_logits"] = out["enc_outputs"]["pred_logits"]
        named["enc_pred_boxes"] = out["enc_outputs"]["pred_boxes"]
        for extra in ("pred_masks", "pred_keypoints"):
            if extra in out["enc_outputs"]:
                named[f"enc_{extra}"] = out["enc_outputs"][extra]
    return named


def main() -> int:
    """Load the bundle, run the JAX model on the same inputs, and report per-tensor errors.

    The unmodified forward must match the reference. When upstream's two-stage selection is
    recorded, also check selection scores and the forward with that selection injected. These
    additional checks help localize a difference; they cannot override a native-forward failure.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="artifacts/torch_ref_nano.npz")
    parser.add_argument("--tol", type=float, default=2e-4)
    parser.add_argument("--modes", default="eval,train")
    args = parser.parse_args()
    if not np.isfinite(args.tol) or args.tol < 0:
        parser.error("--tol must be finite and nonnegative")
    modes = args.modes.split(",")
    if not modes or any(mode not in {"eval", "train"} for mode in modes):
        parser.error("--modes must be eval, train, or eval,train")

    data = np.load(args.bundle, allow_pickle=False)
    meta = {key[6:]: str(data[key]) for key in data.files if key.startswith("meta::")}
    state_dict = {key[4:]: data[key] for key in data.files if key.startswith("sd::")}

    spec = spec_from_meta(meta)
    if meta.get("dtype", "float32") == "float64":
        # A float64 reference isolates formula differences from float32 rounding.
        jax.config.update("jax_enable_x64", True)
        spec = replace(spec, compute_dtype="float64")
        print("float64 reference: JAX x64 enabled; parameters and activations run in float64")
    model = LWDETR(spec=spec)
    pixel_values = jnp.asarray(data["input_pixel_values"])
    pixel_mask = jnp.asarray(data["input_pixel_mask"])
    # Initialise in train mode so all `group_detr` encoder heads exist; the eval path uses a subset.
    template = model.init(jax.random.key(0), pixel_values[:1], pixel_mask[:1], train=True)["params"]

    params, missing, unexpected = convert_state_dict(state_dict, template, strict=False)
    print(f"converted {len(state_dict)} torch tensors; missing={len(missing)} unexpected={len(unexpected)}")
    for name in missing[:12]:
        print("  MISSING  ", name)
    for name in unexpected[:12]:
        print("  UNEXPECTED", name)
    if missing or unexpected:
        return 1

    worst = 0.0
    failures = 0
    for mode in modes:
        out = model.apply({"params": params}, pixel_values, pixel_mask, train=(mode == "train"))
        recorded_selection = data[f"topk::{mode}"] if f"topk::{mode}" in data.files else None
        if recorded_selection is not None and "proposal_indices" in out:
            print(f"[{mode}] native-forward parity (required even when selection scores agree)")
            selection_failures, _, _ = compare_selection(
                recorded_selection, np.asarray(out["proposal_indices"]), np.asarray(out["proposal_scores"]), args.tol)
            own_failures, own_worst = compare_outputs(named_outputs(out), data, mode, args.tol)
            failures += own_failures
            worst = max(worst, own_worst)
            print(f"[{mode}] native forward: {own_failures} mismatching tensors (max {own_worst:.3e})")
            print(f"[{mode}] injected-selection comparison (does not replace native-forward parity)")
            out = model.apply({"params": params}, pixel_values, pixel_mask, train=(mode == "train"),
                              proposal_indices=jnp.asarray(recorded_selection))
            failures += selection_failures
        named = named_outputs(out)
        mode_failures, mode_worst = compare_outputs(named, data, mode, args.tol)
        failures += mode_failures
        worst = max(worst, mode_worst)

    print(f"\nworst absolute error: {worst:.3e}")
    print("Strict ordered comparison: near-tie hypotheses do not exempt mismatches."
          " Native outputs must pass; an injected-selection comparison cannot erase their failures.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
