"""Bisect the JAX/PyTorch mismatch by comparing intermediate activations stage by stage."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# GPU matmuls default to TF32; parity against float32 PyTorch needs the full mantissa.
jax.config.update("jax_default_matmul_precision", "highest")

from rfdetr_jax.convert import convert_state_dict  # noqa: E402
from rfdetr_jax.models.backbone.backbone import downsample_mask  # noqa: E402
from rfdetr_jax.models.backbone.dinov2_windowed import Embeddings, PatchEmbeddings  # noqa: E402
from rfdetr_jax.models.lwdetr import LWDETR  # noqa: E402
from rfdetr_jax.models.position_encoding import position_embedding_sine  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_parity import spec_from_meta  # noqa: E402


def report(name: str, actual: np.ndarray, reference: np.ndarray) -> None:
    """Print the max absolute difference between one JAX tensor and its torch counterpart."""
    actual = np.asarray(actual).astype(np.float64)
    reference = np.asarray(reference).astype(np.float64)
    if actual.shape != reference.shape:
        print(f"  SHAPE {name:26s} {actual.shape} vs {reference.shape}")
        return
    error = float(np.abs(actual - reference).max())
    scale = float(np.abs(reference).max())
    print(f"  {'OK  ' if error < 1e-3 else 'DIFF'} {name:26s} maxabs={error:.3e}  ref|max|={scale:.3f}")


def main() -> None:
    """Run each stage of the JAX model on the recorded inputs and compare against torch."""
    bundle = np.load("artifacts/torch_ref_nano.npz", allow_pickle=True)
    mid = np.load("artifacts/torch_intermediates.npz")
    meta = {key[6:]: str(bundle[key]) for key in bundle.files if key.startswith("meta::")}
    state_dict = {key[4:]: bundle[key] for key in bundle.files if key.startswith("sd::")}

    spec = spec_from_meta(meta)
    model = LWDETR(spec=spec)
    pixel_values = jnp.asarray(bundle["input_pixel_values"])
    pixel_mask = jnp.asarray(bundle["input_pixel_mask"])
    template = model.init(jax.random.key(0), pixel_values[:1], pixel_mask[:1], train=True)["params"]
    params, _, _ = convert_state_dict(state_dict, template, strict=True)

    backbone = model.backbone_0 if hasattr(model, "backbone_0") else None
    del backbone
    dino_config = LWDETR(spec=spec).clone().spec  # keep spec handy for readability
    del dino_config

    from rfdetr_jax.models.backbone.backbone import build_backbone

    backbone_module = build_backbone(
        encoder=spec.encoder,
        out_feature_indexes=list(spec.out_feature_indexes),
        projector_scale=list(spec.projector_scale),
        hidden_dim=spec.hidden_dim,
        patch_size=spec.patch_size,
        num_windows=spec.num_windows,
        positional_encoding_size=spec.positional_encoding_size,
    )
    backbone_params = params["backbone_0"]
    config = backbone_module.dino_config

    print("== embeddings ==")
    patch_out = PatchEmbeddings(config).apply(
        {"params": backbone_params["encoder"]["embeddings"]["patch_embeddings"]}, pixel_values
    )
    report("patch_embeddings", patch_out, mid["patch_embeddings"])

    emb_out = Embeddings(config).apply({"params": backbone_params["encoder"]["embeddings"]}, pixel_values)
    report("embeddings", emb_out, mid["embeddings"])

    print("== encoder layers ==")
    from rfdetr_jax.models.backbone.dinov2_windowed import Layer, WindowedDinov2Backbone

    hidden = emb_out
    for index in range(config.num_hidden_layers):
        hidden = Layer(config).apply(
            {"params": backbone_params["encoder"]["encoder"][f"layer_{index}"]},
            hidden,
            index not in config.window_block_indexes,
            True,
        )
        key = f"encoder_layer{index}"
        if key in mid.files:
            report(key, hidden, mid[key + ".0"] if key + ".0" in mid.files else mid[key])

    print("== dino feature maps ==")
    raw = WindowedDinov2Backbone(config).apply(
        {"params": backbone_params["encoder"]}, pixel_values, deterministic=True
    )
    for index, feature in enumerate(raw):
        report(f"dino_backbone.{index}", feature, mid[f"dino_backbone.{index}"])

    print("== projector ==")
    features, _ = backbone_module.apply({"params": backbone_params}, pixel_values, train=False)
    report("projector_out", features[0], mid["projector.0"])

    print("== position embedding ==")
    mask_level = downsample_mask(pixel_mask, (features[0].shape[2], features[0].shape[3]))
    report("joiner_mask", mask_level, mid["joiner_mask.0"])
    pos = position_embedding_sine(mask_level, spec.hidden_dim // 2, normalize=True)
    report("joiner_pos", pos, mid["joiner_pos.0"])

    print("== full model ==")
    out = model.apply({"params": params}, pixel_values, pixel_mask, train=False)
    report("pred_logits", out["pred_logits"], bundle["eval_pred_logits"])


if __name__ == "__main__":
    main()
