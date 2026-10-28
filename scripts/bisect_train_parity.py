"""Bisect the train-mode (group-DETR) path against the PyTorch reference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")

from check_parity import spec_from_meta  # noqa: E402

from rfdetr_jax.convert import convert_state_dict  # noqa: E402
from rfdetr_jax.models.lwdetr import LWDETR  # noqa: E402


def report(name: str, actual, reference) -> None:
    """Print the max absolute difference for one tensor pair."""
    actual = np.asarray(actual).astype(np.float64)
    reference = np.asarray(reference).astype(np.float64)
    if actual.shape != reference.shape:
        print(f"  SHAPE {name:26s} {actual.shape} vs {reference.shape}")
        return
    error = float(np.abs(actual - reference).max())
    scale = float(np.abs(reference).max())
    status = "OK  " if error < 1e-3 else "DIFF"
    print(f"  {status} {name:26s} maxabs={error:.3e}  ref|max|={scale:.3f}")


def main() -> None:
    """Run the JAX transformer in train mode and compare its stages against torch."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="artifacts/torch_ref_nano.npz")
    parser.add_argument("--mid", default="artifacts/torch_train_intermediates.npz")
    options = parser.parse_args()
    bundle = np.load(options.bundle, allow_pickle=True)
    mid = np.load(options.mid)
    meta = {key[6:]: str(bundle[key]) for key in bundle.files if key.startswith("meta::")}
    state_dict = {key[4:]: bundle[key] for key in bundle.files if key.startswith("sd::")}

    spec = spec_from_meta(meta)
    model = LWDETR(spec=spec)
    pixel_values = jnp.asarray(bundle["input_pixel_values"])
    pixel_mask = jnp.asarray(bundle["input_pixel_mask"])
    template = model.init(jax.random.key(0), pixel_values[:1], pixel_mask[:1], train=True)["params"]
    params, _, _ = convert_state_dict(state_dict, template, strict=True)

    captured: dict[str, jnp.ndarray] = {}
    module = model.bind({"params": params}, mutable=False)

    features, _ = module.backbone_0(pixel_values, train=True)
    from rfdetr_jax.models.backbone.backbone import downsample_mask

    masks = [downsample_mask(pixel_mask, (f.shape[2], f.shape[3])) for f in features]
    outputs = module.transformer(
        features,
        masks,
        [],
        module.refpoint_embed,
        module.query_feat,
        bbox_embed=None,
        train=True,
    )
    captured["hs"] = outputs["hs"]
    captured["references"] = outputs["references"]
    captured["hs_enc"] = outputs["hs_enc"]
    captured["ref_enc"] = outputs["ref_enc"]

    report("hs", captured["hs"], mid["transformer.0"])
    report("references", captured["references"], mid["transformer.1"])
    report("hs_enc", captured["hs_enc"], mid["transformer.2"])
    report("ref_enc", captured["ref_enc"], mid["transformer.3"])

    print("\n== decoder internals ==")
    from rfdetr_jax.models.backbone.backbone import downsample_mask as _dm

    del _dm
    memory = jnp.concatenate([f.reshape(*f.shape[:2], -1).transpose(0, 2, 1) for f in features], axis=1)
    del memory

    print("\n== per-group hs_enc ==")
    groups = spec.group_detr
    per = captured["hs_enc"].shape[1] // groups
    actual = np.asarray(captured["hs_enc"])
    reference = mid["transformer.2"]
    for group in range(groups):
        window = slice(group * per, (group + 1) * per)
        error = float(np.abs(actual[:, window] - reference[:, window]).max())
        best = min(
            (float(np.abs(actual[:, window] - reference[:, slice(other * per, (other + 1) * per)]).max()), other)
            for other in range(groups)
        )
        rows = int((np.abs(actual[:, window] - reference[:, window]).max(-1) > 1e-3).sum())
        print(
            f"  group {group:2d}: maxabs={error:.3e}   best match = group {best[1]} ({best[0]:.3e})"
            f"   mismatched queries={rows}"
        )


if __name__ == "__main__":
    main()
