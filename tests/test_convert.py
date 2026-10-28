"""Checkpoint key translation between the PyTorch tree and this port's Flax tree."""

from __future__ import annotations

import numpy as np
import pytest

from rfdetr_jax.convert import (
    convert_state_dict,
    flatten_params,
    torch_key_to_jax_path,
    unflatten_params,
)


@pytest.mark.parametrize(
    ("torch_key", "expected"),
    [
        ("class_embed.weight", ("class_embed", "weight")),
        ("bbox_embed.layers.2.bias", ("bbox_embed", "layers_2", "bias")),
        ("refpoint_embed.weight", ("refpoint_embed",)),
        ("query_feat.weight", ("query_feat",)),
        (
            "backbone.0.encoder.encoder.encoder.layer.3.norm1.weight",
            ("backbone_0", "encoder", "encoder", "layer_3", "norm1", "weight"),
        ),
        (
            "backbone.0.projector.stages_sampling.0.1.0.conv.weight",
            ("backbone_0", "projector", "stages_sampling_0_1_0", "conv", "weight"),
        ),
        (
            "transformer.decoder.layers.1.self_attn.in_proj_weight",
            ("transformer", "decoder", "layers_1", "self_attn", "in_proj_weight"),
        ),
        (
            "transformer.enc_out_bbox_embed.7.layers.0.weight",
            ("transformer", "enc_out_bbox_embed_7", "layers_0", "weight"),
        ),
    ],
)
def test_key_translation(torch_key: str, expected: tuple[str, ...]) -> None:
    """Every structural rename the port needs is a pure key transformation."""
    assert torch_key_to_jax_path(torch_key) == expected


@pytest.mark.parametrize("key", ["encoder.layer.0.num_batches_tracked", "_kp_active_mask"])
def test_skipped_keys(key: str) -> None:
    """Buffers with no Flax counterpart are dropped rather than mapped."""
    assert torch_key_to_jax_path(key) is None


def test_flatten_roundtrip() -> None:
    """Flattening and unflattening a parameter tree is lossless."""
    tree = {"a": {"b": np.zeros(3), "c": {"d": np.ones(2)}}, "e": np.arange(4)}
    assert set(flatten_params(unflatten_params(flatten_params(tree)))) == set(flatten_params(tree))


def test_convert_reports_shape_mismatch() -> None:
    """A shape mismatch against the template is raised, not silently loaded."""
    template = {"class_embed": {"weight": np.zeros((4, 8))}}
    with pytest.raises(ValueError, match="shape mismatch"):
        convert_state_dict({"class_embed.weight": np.zeros((5, 8))}, template, strict=True)


def test_convert_reports_missing_and_unexpected() -> None:
    """Non-strict conversion reports both directions instead of failing."""
    template = {"class_embed": {"weight": np.zeros((4, 8)), "bias": np.zeros(4)}}
    params, missing, unexpected = convert_state_dict(
        {"class_embed.weight": np.zeros((4, 8)), "mystery.weight": np.zeros(3)}, template, strict=False
    )
    assert "class_embed" in params
    assert missing == ["class_embed.bias"]
    assert unexpected == ["mystery.weight"]


def test_decoder_box_head_alias_is_not_a_separate_tensor() -> None:
    """``lite_refpoint_refine=False`` aliases the shared box head under the decoder in upstream state dicts."""
    assert torch_key_to_jax_path("transformer.decoder.bbox_embed.layers.0.weight") is None
    assert torch_key_to_jax_path("bbox_embed.layers.0.weight") == ("bbox_embed", "layers_0", "weight")
