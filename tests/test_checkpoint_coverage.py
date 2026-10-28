"""The converter must cover an entire real RF-DETR checkpoint, with matching shapes.

``tests/data/nano_state_dict_shapes.json`` records every key and shape of an upstream RF-DETR Nano
``state_dict``, so this runs without PyTorch installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from rfdetr_jax.convert import convert_state_dict, flatten_params
from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec
from rfdetr_jax.training.trainer import init_params

_FIXTURE = json.loads((Path(__file__).parent / "data" / "nano_state_dict_shapes.json").read_text())


@pytest.fixture(scope="module")
def nano_params() -> dict:
    """Initialise the JAX model with the recorded upstream Nano configuration."""
    config = _FIXTURE["config"]
    spec = LWDETRSpec(
        encoder=config["encoder"],
        hidden_dim=config["hidden_dim"],
        patch_size=config["patch_size"],
        num_windows=config["num_windows"],
        dec_layers=config["dec_layers"],
        sa_nheads=config["sa_nheads"],
        ca_nheads=config["ca_nheads"],
        dec_n_points=config["dec_n_points"],
        num_queries=config["num_queries"],
        projector_scale=tuple(config["projector_scale"]),
        out_feature_indexes=tuple(config["out_feature_indexes"]),
        resolution=config["resolution"],
        positional_encoding_size=config["positional_encoding_size"],
        num_classes=config["num_classes"],
        group_detr=config["group_detr"],
    )
    return init_params(LWDETR(spec=spec), spec, jax.random.key(0))


def test_every_checkpoint_key_maps_onto_the_parameter_tree(nano_params: dict) -> None:
    """A real Nano checkpoint converts with nothing missing and nothing left over."""
    state_dict = {
        key: np.zeros(shape, dtype="float32") for key, shape in _FIXTURE["state_dict_shapes"].items()
    }
    params, missing, unexpected = convert_state_dict(state_dict, nano_params, strict=True)
    assert missing == []
    assert unexpected == []
    assert len(flatten_params(params)) == len(flatten_params(nano_params))


def test_parameter_counts_agree(nano_params: dict) -> None:
    """The port has exactly as many learnable elements as the upstream checkpoint."""
    shapes = _FIXTURE["state_dict_shapes"]
    torch_elements = sum(
        int(np.prod(shape)) for key, shape in shapes.items() if not key.endswith("num_batches_tracked")
        and key != "_kp_active_mask"
    )
    jax_elements = sum(int(value.size) for value in flatten_params(nano_params).values())
    assert jax_elements == torch_elements


def test_a_wrong_shape_is_rejected(nano_params: dict) -> None:
    """Strict conversion refuses a checkpoint whose head width does not match."""
    shapes = dict(_FIXTURE["state_dict_shapes"])
    shapes["class_embed.weight"] = [shapes["class_embed.weight"][0] + 1, shapes["class_embed.weight"][1]]
    state_dict = {key: np.zeros(shape, dtype="float32") for key, shape in shapes.items()}
    with pytest.raises(ValueError, match="shape mismatch for class_embed.weight"):
        convert_state_dict(state_dict, nano_params, strict=True)
