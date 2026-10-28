"""GroupPose keypoint head, losses and matching costs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfdetr_jax.models.heads.keypoints import (
    ConditionalQueryInitializer,
    active_keypoint_mask,
    compute_keypoint_matching_cost,
    compute_l1_keypoint_loss,
    modulate,
)
from rfdetr_jax.models.transformer import keypoint_class_mask

_SCHEMA = [3, 2]


def _predictions(targets: int = 4, seed: int = 0) -> jnp.ndarray:
    """Random keypoint predictions in the class-padded layout."""
    total = len(_SCHEMA) * max(_SCHEMA)
    return jnp.asarray(np.random.RandomState(seed).randn(targets, total, 7).astype("float32"))


def _targets(count: int = 4) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Ground-truth keypoints, classes and areas with a mix of visibility flags."""
    rng = np.random.RandomState(1)
    keypoints = rng.rand(count, max(_SCHEMA), 3).astype("float32")
    keypoints[:, :, 2] = np.array([[2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])[:count]
    return (
        jnp.asarray(keypoints),
        jnp.asarray(np.array([0, 1, 0, 1])[:count]),
        jnp.asarray(np.array([0.05, 0.2, 0.01, 0.5], dtype="float32")[:count]),
    )


def test_active_mask_follows_the_schema() -> None:
    """Each class exposes exactly as many keypoint slots as it owns."""
    mask = np.asarray(active_keypoint_mask(_SCHEMA))
    assert mask.shape == (2, 3)
    assert mask[0].tolist() == [True, True, True]
    assert mask[1].tolist() == [True, True, False]


def test_class_mask_blocks_cross_class_keypoints() -> None:
    """Keypoints of one class cannot attend to another's; the instance token attends to all."""
    mask = np.asarray(keypoint_class_mask(_SCHEMA))
    assert mask.shape == (6, 6)
    assert not mask[0].any() and not mask[:, 0].any()
    assert mask[1, 4] and mask[4, 1]
    assert not mask[1, 2] and not mask[4, 5]


def test_modulate_matches_the_adaln_formula() -> None:
    """``modulate`` is ``(scale + 1) * features + shift``."""
    features = jnp.ones((2, 4))
    result = np.asarray(modulate(features, jnp.full((2, 4), 0.1), jnp.full((2, 4), -0.5)))
    assert np.allclose(result, 1.1 - 0.5)


def test_query_initializer_shape_and_residual() -> None:
    """The initializer adds a query axis and stays close to its bank when modulation is zero."""
    initializer = ConditionalQueryInitializer(dim=8, num_queries=5, out_dim=8)
    conditioning = jnp.zeros((2, 3, 8))
    params = initializer.init(jax.random.key(0), conditioning)["params"]
    output = initializer.apply({"params": params}, conditioning)
    assert output.shape == (2, 3, 5, 8)


def test_losses_ignore_unlabelled_keypoints() -> None:
    """A keypoint with visibility 0 contributes nothing to the location loss."""
    keypoints, classes, areas = _targets()
    predictions = _predictions()
    location, _, _, _ = compute_l1_keypoint_loss(predictions, keypoints, classes, areas, _SCHEMA)
    shifted = keypoints.at[:, :, :2].set(keypoints[:, :, :2])
    # Move only the unlabelled slots; the loss must not react.
    unlabelled = keypoints[:, :, 2] == 0
    shifted = jnp.where(unlabelled[..., None], shifted[:, :, :2] + 10.0, shifted[:, :, :2])
    moved = jnp.concatenate([shifted, keypoints[:, :, 2:]], axis=-1)
    location_after, _, _, _ = compute_l1_keypoint_loss(predictions, moved, classes, areas, _SCHEMA)
    assert np.allclose(np.asarray(location), np.asarray(location_after), atol=1e-6)


def test_losses_survive_a_non_finite_prediction() -> None:
    """One NaN channel must not poison the other keypoints of the same target."""
    keypoints, classes, areas = _targets()
    predictions = _predictions().at[0, 0, 0].set(jnp.nan)
    losses = compute_l1_keypoint_loss(predictions, keypoints, classes, areas, _SCHEMA)
    for value in losses:
        assert np.isfinite(np.asarray(value)).all()


def test_losses_reject_an_empty_schema() -> None:
    """An empty keypoint schema is a configuration error, not a silent no-op."""
    keypoints, classes, areas = _targets()
    with pytest.raises(ValueError, match="must be non-empty"):
        compute_l1_keypoint_loss(_predictions(), keypoints, classes, areas, [])


def test_matching_cost_shapes_and_class_routing() -> None:
    """Costs are ``[B, Q, N]`` and each target is scored against its own class's slots."""
    keypoints, classes, areas = _targets()
    total = len(_SCHEMA) * max(_SCHEMA)
    predictions = jnp.asarray(np.random.RandomState(2).randn(2, 3, total, 7).astype("float32"))
    costs = compute_keypoint_matching_cost(predictions, keypoints, classes, areas, _SCHEMA)
    for cost in costs:
        assert cost.shape == (2, 3, 4)
        assert np.isfinite(np.asarray(cost)).all()


def test_matching_cost_handles_no_targets() -> None:
    """With no targets every cost is an empty ``[B, Q, 0]`` array."""
    total = len(_SCHEMA) * max(_SCHEMA)
    costs = compute_keypoint_matching_cost(
        jnp.zeros((1, 2, total, 7)),
        jnp.zeros((0, max(_SCHEMA), 3)),
        jnp.zeros((0,), jnp.int32),
        jnp.zeros((0,)),
        _SCHEMA,
    )
    assert all(cost.shape == (1, 2, 0) for cost in costs)


def test_box_only_class_outside_compact_schema_has_zero_keypoint_loss_and_gradient() -> None:
    """Out-of-schema classes must not clamp onto another class's keypoint supervision."""
    predictions = jnp.ones((1, 2, 8))
    targets = jnp.array([[[.2, .4, 2.], [.4, .6, 2.]]])

    def loss(predictions):
        terms = compute_l1_keypoint_loss(predictions, targets, jnp.array([1]), jnp.array([.2]), (2,))
        return sum(value.sum() for value in terms), terms

    (total, terms), gradients = jax.jit(jax.value_and_grad(loss, has_aux=True))(predictions)
    assert float(total) == 0.0
    assert all(float(term[0]) == 0.0 for term in terms)
    np.testing.assert_array_equal(gradients, np.zeros_like(predictions))
