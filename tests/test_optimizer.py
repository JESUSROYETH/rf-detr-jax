"""Layer-wise learning rates and the managed LR schedule."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rfdetr_jax.convert import flatten_params
from rfdetr_jax.training.optimizer import (
    OptimizerSpec,
    build_optimizer,
    build_param_group_trees,
    dinov2_lr_decay_rate,
    dinov2_weight_decay_rate,
    ema_decay_at,
    ema_update,
    managed_schedule,
)

_PARAMS = {
    "backbone_0": {
        "encoder": {
            "embeddings": {"cls_token": jnp.zeros((1, 1, 4))},
            "encoder": {
                "layer_0": {"norm1": {"weight": jnp.zeros(4)}, "mlp": {"fc1": {"weight": jnp.zeros((4, 4))}}},
                "layer_11": {"mlp": {"fc1": {"weight": jnp.zeros((4, 4))}}},
            },
        },
        "projector": {"stages_0_0": {"cv1": {"conv": {"weight": jnp.zeros((4, 4, 1, 1))}}}},
    },
    "transformer": {
        "decoder": {"layers_0": {"norm1": {"weight": jnp.zeros(4)}}},
        "enc_output_0": {"weight": jnp.zeros((4, 4))},
    },
    "class_embed": {"weight": jnp.zeros((3, 4))},
}


def test_layer_decay_follows_block_index() -> None:
    """Deeper encoder blocks get a larger multiplier; embeddings get the smallest."""
    embeddings = dinov2_lr_decay_rate("backbone_0.encoder.embeddings.cls_token", 0.8, 13)
    first = dinov2_lr_decay_rate("backbone_0.encoder.encoder.layer_0.norm1.weight", 0.8, 13)
    last = dinov2_lr_decay_rate("backbone_0.encoder.encoder.layer_11.mlp.fc1.weight", 0.8, 13)
    assert embeddings == pytest.approx(0.8**14)
    assert first == pytest.approx(0.8**13)
    assert last == pytest.approx(0.8**2)
    assert embeddings < first < last


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("backbone_0.encoder.encoder.layer_0.mlp.fc1.weight", 1.0),
        ("backbone_0.encoder.encoder.layer_0.mlp.fc1.bias", 0.0),
        ("backbone_0.encoder.encoder.layer_0.norm1.weight", 0.0),
        ("backbone_0.encoder.embeddings.cls_token", 0.0),
    ],
)
def test_weight_decay_exemptions(path: str, expected: float) -> None:
    """Biases, norms and embeddings are exempt from weight decay."""
    assert dinov2_weight_decay_rate(path) == expected


def test_param_groups_split_backbone_decoder_and_rest() -> None:
    """The three upstream groups land on the right parameters with the right rates."""
    spec = OptimizerSpec(lr=1e-4, lr_encoder=1.5e-4, lr_component_decay=0.7, lr_vit_layer_decay=0.8)
    rates, decays = build_param_group_trees(_PARAMS, spec, (3, 6, 9, 12))
    flat_rates = {".".join(path): float(value) for path, value in flatten_params(rates).items()}
    flat_decays = {".".join(path): float(value) for path, value in flatten_params(decays).items()}

    assert flat_rates["class_embed.weight"] == pytest.approx(spec.lr)
    assert flat_rates["backbone_0.projector.stages_0_0.cv1.conv.weight"] == pytest.approx(spec.lr)
    assert flat_rates["transformer.enc_output_0.weight"] == pytest.approx(spec.lr)
    assert flat_rates["transformer.decoder.layers_0.norm1.weight"] == pytest.approx(spec.lr * 0.7)
    assert flat_rates["backbone_0.encoder.encoder.layer_11.mlp.fc1.weight"] == pytest.approx(
        spec.lr_encoder * 0.8**2 * 0.7**2
    )
    assert flat_decays["backbone_0.encoder.encoder.layer_0.norm1.weight"] == 0.0
    assert flat_decays["class_embed.weight"] == pytest.approx(spec.weight_decay)


def test_step_schedule_drops_by_ten() -> None:
    """The step preset holds at 1.0 then drops 10x after ``lr_drop`` epochs."""
    schedule = managed_schedule("step", total_steps=100, steps_per_epoch=10, warmup_steps=0, lr_drop=5)
    assert float(schedule(jnp.int32(0))) == pytest.approx(1.0)
    assert float(schedule(jnp.int32(49))) == pytest.approx(1.0)
    assert float(schedule(jnp.int32(50))) == pytest.approx(0.1)


def test_warmup_ramps_linearly() -> None:
    """Warmup ramps from 0 to 1 over the configured step count."""
    schedule = managed_schedule("step", total_steps=100, steps_per_epoch=10, warmup_steps=10, lr_drop=100)
    assert float(schedule(jnp.int32(0))) == pytest.approx(0.0)
    assert float(schedule(jnp.int32(5))) == pytest.approx(0.5)
    assert float(schedule(jnp.int32(10))) == pytest.approx(1.0)


def test_cosine_schedule_decays_to_min_factor() -> None:
    """The cosine preset starts at 1.0 and ends at ``min_factor``."""
    schedule = managed_schedule("cosine", total_steps=100, steps_per_epoch=10, warmup_steps=0, lr_drop=100)
    assert float(schedule(jnp.int32(0))) == pytest.approx(1.0)
    assert float(schedule(jnp.int32(100))) == pytest.approx(0.0, abs=1e-6)


def test_ema_matches_torch_formula() -> None:
    """The EMA decay and update reproduce ``ModelEma``'s tau-warmed formula."""
    import math

    assert float(ema_decay_at(jnp.int32(1), 0.993, 0)) == pytest.approx(0.993)
    assert float(ema_decay_at(jnp.int32(100), 0.993, 100)) == pytest.approx(0.993 * (1 - math.exp(-1.0)), rel=1e-6)
    updated = ema_update({"w": jnp.ones(3)}, {"w": jnp.zeros(3)}, jnp.float32(0.9))
    assert np.allclose(np.asarray(updated["w"]), 0.9)


def test_optimizer_applies_per_parameter_rates() -> None:
    """A unit gradient moves each parameter by exactly its own learning rate."""
    spec = OptimizerSpec(clip_max_norm=0.0, weight_decay=0.0)
    optimizer = build_optimizer(_PARAMS, spec, (3, 6, 9, 12))
    state = optimizer.init(_PARAMS)
    grads = jax.tree.map(jnp.ones_like, _PARAMS)
    updates, _ = optimizer.update(grads, state, _PARAMS)
    flat = {".".join(path): np.asarray(value) for path, value in flatten_params(updates).items()}
    # After one Adam step the normalised update magnitude is ~1, so the step size is the rate.
    assert float(flat["class_embed.weight"].mean()) == pytest.approx(-spec.lr, rel=1e-3)
    assert float(flat["transformer.decoder.layers_0.norm1.weight"].mean()) == pytest.approx(
        -spec.lr * spec.lr_component_decay, rel=1e-3
    )


def test_fused_adamw_matches_the_optax_chain() -> None:
    """The single-pass optimizer must reproduce the staged chain term for term, over many steps."""
    from rfdetr_jax.training.optimizer import build_optimizer, managed_schedule

    key = jax.random.key(0)
    params = {
        "backbone_0": {"encoder": {"layer_1": {"weight": jax.random.normal(key, (8, 4))}}},
        "class_embed": {"weight": jax.random.normal(key, (3, 8)), "bias": jnp.zeros((3,))},
        "transformer": {"decoder": {"layers_0": {"weight": jax.random.normal(key, (4, 4))}}},
    }
    spec = OptimizerSpec(clip_max_norm=0.1)
    schedule = managed_schedule("step", total_steps=20, steps_per_epoch=4, warmup_steps=3, lr_drop=2)

    staged = build_optimizer(params, spec, (3, 6, 9, 12), schedule, fused=False)
    fused = build_optimizer(params, spec, (3, 6, 9, 12), schedule, fused=True)
    staged_state, fused_state = staged.init(params), fused.init(params)
    staged_params = fused_params = params

    for step in range(6):
        grads = jax.tree.map(
            lambda leaf, step=step: jax.random.normal(jax.random.key(step), leaf.shape) * (step + 1),
            params,
        )
        staged_updates, staged_state = staged.update(grads, staged_state, staged_params)
        fused_updates, fused_state = fused.update(grads, fused_state, fused_params)
        staged_params = optax.apply_updates(staged_params, staged_updates)
        fused_params = optax.apply_updates(fused_params, fused_updates)

        for left, right in zip(
            jax.tree.leaves(staged_params), jax.tree.leaves(fused_params), strict=True
        ):
            np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=1e-6, atol=1e-7)


def test_fused_adamw_needs_params_for_the_decay_term() -> None:
    """Decoupled decay reads the parameters, so omitting them must fail loudly."""
    from rfdetr_jax.training.optimizer import build_optimizer

    params = {"class_embed": {"weight": jnp.ones((2, 2))}}
    optimizer = build_optimizer(params, OptimizerSpec(), (3, 6, 9, 12))
    state = optimizer.init(params)
    with pytest.raises(ValueError, match="needs params"):
        optimizer.update(jax.tree.map(jnp.ones_like, params), state, None)


def test_shared_box_head_joins_the_decoder_group_without_lite_refinement() -> None:
    """Upstream registers ``transformer.decoder.bbox_embed`` first, so the shared head takes the decoder rate."""
    spec = OptimizerSpec(lr=1e-4, lr_encoder=1.5e-4, lr_component_decay=0.7, weight_decay=1e-4)
    params = {**_PARAMS, "bbox_embed": {"layers_0": {"weight": jnp.zeros((2, 2))}}}
    lite, _ = build_param_group_trees(params, spec, (3, 6, 9, 12))
    shared, _ = build_param_group_trees(params, spec, (3, 6, 9, 12), lite_refpoint_refine=False)
    assert float(lite["bbox_embed"]["layers_0"]["weight"]) == pytest.approx(1e-4)
    assert float(shared["bbox_embed"]["layers_0"]["weight"]) == pytest.approx(0.7e-4)
