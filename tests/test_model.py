"""Model construction, output shapes and the two-stage/aux wiring."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec, initial_class_bias

_SPEC = LWDETRSpec(
    dec_layers=2,
    num_windows=2,
    patch_size=16,
    resolution=64,
    positional_encoding_size=4,
    num_classes=5,
    group_detr=2,
    num_queries=8,
    hidden_dim=32,
    sa_nheads=2,
    ca_nheads=2,
    dim_feedforward=64,
    out_feature_indexes=(1, 2),
)


@pytest.fixture(scope="module")
def model_and_params() -> tuple[LWDETR, dict]:
    """Build a tiny detector and initialise it once for the whole module."""
    model = LWDETR(spec=_SPEC)
    inputs = jnp.zeros((1, 3, _SPEC.resolution, _SPEC.resolution), jnp.float32)
    mask = jnp.zeros((1, _SPEC.resolution, _SPEC.resolution), bool)
    return model, model.init(jax.random.key(0), inputs, mask, train=True)["params"]


def test_output_shapes_in_eval_mode(model_and_params) -> None:
    """Inference uses a single query group and emits aux and encoder outputs."""
    model, params = model_and_params
    inputs = jnp.zeros((2, 3, _SPEC.resolution, _SPEC.resolution), jnp.float32)
    outputs = model.apply({"params": params}, inputs, None, train=False)
    assert outputs["pred_logits"].shape == (2, _SPEC.num_queries, _SPEC.detector_classes)
    assert outputs["pred_boxes"].shape == (2, _SPEC.num_queries, 4)
    assert len(outputs["aux_outputs"]) == _SPEC.dec_layers - 1
    assert outputs["enc_outputs"]["pred_logits"].shape == outputs["pred_logits"].shape


def test_training_mode_expands_query_groups(model_and_params) -> None:
    """Training runs ``group_detr`` independent query groups."""
    model, params = model_and_params
    inputs = jnp.zeros((1, 3, _SPEC.resolution, _SPEC.resolution), jnp.float32)
    outputs = model.apply({"params": params}, inputs, None, train=True)
    assert outputs["pred_logits"].shape[1] == _SPEC.num_queries * _SPEC.group_detr


def test_padding_mask_changes_predictions(model_and_params) -> None:
    """A padding mask feeds the valid-ratio path, so it must change the output."""
    model, params = model_and_params
    inputs = jnp.asarray(
        np.random.RandomState(0).randn(1, 3, _SPEC.resolution, _SPEC.resolution).astype("float32")
    )
    mask = np.zeros((1, _SPEC.resolution, _SPEC.resolution), dtype=bool)
    mask[0, -16:, :] = True
    unmasked = model.apply({"params": params}, inputs, None, train=False)["pred_boxes"]
    masked = model.apply({"params": params}, inputs, jnp.asarray(mask), train=False)["pred_boxes"]
    assert not np.allclose(np.asarray(unmasked), np.asarray(masked))


def test_resolution_must_divide_window_grid(model_and_params) -> None:
    """A resolution that is not a multiple of ``patch_size * num_windows`` is rejected."""
    model, params = model_and_params
    bad = jnp.zeros((1, 3, 48, 48), jnp.float32)
    with pytest.raises(ValueError, match="divisible by patch_size"):
        model.apply({"params": params}, bad, None, train=False)


def test_parameter_tree_names_match_torch_layout(model_and_params) -> None:
    """The parameter tree carries the names the checkpoint converter targets."""
    _, params = model_and_params
    assert set(params) >= {"backbone_0", "transformer", "class_embed", "bbox_embed"}
    assert "refpoint_embed" in params and "query_feat" in params
    assert "encoder" in params["backbone_0"] and "projector" in params["backbone_0"]
    assert "layer_0" in params["backbone_0"]["encoder"]["encoder"]
    assert "layers_0" in params["transformer"]["decoder"]
    assert "enc_out_class_embed_0" in params["transformer"]


def test_class_prior_bias() -> None:
    """The focal-loss prior bias matches the upstream constant."""
    assert initial_class_bias(91) == pytest.approx(-np.log((1 - 0.01) / 0.01))


def test_head_dtype_activation_runs_the_head_matmuls_in_bf16(model_and_params) -> None:
    """``head_dtype="activation"`` changes only the heads' matmul dtype; outputs stay float32."""
    import dataclasses
    import re

    _, params = model_and_params
    size = (1, 3, _SPEC.resolution, _SPEC.resolution)
    inputs = jnp.asarray(np.random.default_rng(0).standard_normal(size), jnp.float32)
    outputs, head_dots = {}, {}
    for head_dtype in ("float32", "activation"):
        spec = dataclasses.replace(_SPEC, compute_dtype="bfloat16", head_dtype=head_dtype)
        model = LWDETR(spec=spec)
        run = lambda p, x, model=model: model.apply({"params": p}, x, None, train=False)  # noqa: E731
        outputs[head_dtype] = run(params, inputs)
        text = str(jax.make_jaxpr(run)(params, inputs))
        # The decoder's class head: [dec_layers, batch, queries, classes], one row per decoder layer.
        shape = rf"\[{_SPEC.dec_layers},1,\d+,{_SPEC.detector_classes}\]"
        head_dots[head_dtype] = set(re.findall(rf":(\w+){shape} = dot_general", text))
    assert outputs["activation"]["pred_logits"].dtype == jnp.float32
    assert outputs["activation"]["pred_boxes"].dtype == jnp.float32
    np.testing.assert_allclose(
        outputs["activation"]["pred_boxes"], outputs["float32"]["pred_boxes"], atol=2e-2
    )
    assert "f32" in head_dots["float32"] and "bf16" not in head_dots["float32"]
    assert "bf16" in head_dots["activation"]


def test_unknown_head_dtype_is_rejected(model_and_params) -> None:
    """A typo must fail loudly rather than silently pick one."""
    import dataclasses

    _, params = model_and_params
    model = LWDETR(spec=dataclasses.replace(_SPEC, head_dtype="half"))
    inputs = jnp.zeros((1, 3, _SPEC.resolution, _SPEC.resolution), jnp.float32)
    with pytest.raises(ValueError, match="unknown head_dtype"):
        model.apply({"params": params}, inputs, None, train=False)


def test_proposal_selection_can_be_injected_for_parity_checks() -> None:
    """Injecting the model's own two-stage selection reproduces its outputs; a different valid
    selection changes the encoder proposals accordingly."""
    import numpy as np

    from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec

    spec = LWDETRSpec(resolution=32, positional_encoding_size=2, patch_size=16, num_windows=1, dec_layers=1,
                      group_detr=2, num_queries=3, num_select=3, hidden_dim=32, sa_nheads=2, ca_nheads=2,
                      dim_feedforward=32, out_feature_indexes=(1, 2), num_classes=3)
    model = LWDETR(spec=spec)
    pixels = jax.random.normal(jax.random.key(0), (2, 3, 32, 32))
    params = model.init(jax.random.key(1), pixels, None, train=True)["params"]
    outputs = model.apply({"params": params}, pixels, None, train=True)
    assert outputs["proposal_indices"].shape == (2, 2, 3) and outputs["proposal_scores"].shape[:2] == (2, 2)
    replay = model.apply({"params": params}, pixels, None, train=True, proposal_indices=outputs["proposal_indices"])
    for key in ("pred_logits", "pred_boxes"):
        np.testing.assert_array_equal(np.asarray(replay[key]), np.asarray(outputs[key]))
        np.testing.assert_array_equal(np.asarray(replay["enc_outputs"][key]), np.asarray(outputs["enc_outputs"][key]))
    positions = outputs["proposal_scores"].shape[-1]
    other = (np.asarray(outputs["proposal_indices"]) + 1) % positions
    changed = model.apply({"params": params}, pixels, None, train=True, proposal_indices=jnp.asarray(other))
    assert not np.allclose(np.asarray(changed["enc_outputs"]["pred_boxes"]),
                           np.asarray(outputs["enc_outputs"]["pred_boxes"]))
    with pytest.raises(ValueError, match="proposal_indices must have shape"):
        model.apply({"params": params}, pixels, None, train=True, proposal_indices=jnp.zeros((1, 2, 3), jnp.int32))
    evaluation = model.apply({"params": params}, pixels, None, train=False)
    assert evaluation["proposal_indices"].shape == (1, 2, 3)
