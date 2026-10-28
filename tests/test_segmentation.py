"""Segmentation head, point sampling and mask losses."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfdetr_jax.models.criterion import Targets, loss_masks
from rfdetr_jax.models.heads.segmentation import (
    SegmentationHead,
    calculate_uncertainty,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)


def test_head_output_shape_follows_downsample_ratio() -> None:
    """Masks come out at ``image_size // downsample_ratio``, one map per decoder layer."""
    head = SegmentationHead(in_dim=8, num_blocks=2, downsample_ratio=4)
    spatial = jnp.zeros((2, 8, 6, 6))
    queries = [jnp.zeros((2, 5, 8)), jnp.zeros((2, 5, 8))]
    params = head.init(jax.random.key(0), spatial, queries, (64, 64))["params"]
    outputs = head.apply({"params": params}, spatial, queries, (64, 64))
    assert len(outputs) == 2
    assert outputs[0].shape == (2, 5, 16, 16)


def test_skip_blocks_requires_one_query_tensor() -> None:
    """The encoder path takes exactly one query tensor."""
    head = SegmentationHead(in_dim=8, num_blocks=2)
    spatial = jnp.zeros((1, 8, 6, 6))
    queries = [jnp.zeros((1, 5, 8))]
    params = head.init(jax.random.key(0), spatial, queries, (64, 64))["params"]
    with pytest.raises(ValueError, match="single query-feature tensor"):
        head.apply({"params": params}, spatial, [queries[0], queries[0]], (64, 64), skip_blocks=True)


def test_point_sample_reads_the_expected_pixel() -> None:
    """Nearest sampling at a pixel centre returns that pixel."""
    values = jnp.asarray([[[[0.0, 1.0], [2.0, 3.0]]]])
    coords = jnp.asarray([[[0.75, 0.75]]])
    assert float(point_sample(values, coords, mode="nearest")[0, 0, 0]) == 3.0


def test_point_sample_rejects_other_modes() -> None:
    """An unsupported sampling mode is rejected."""
    with pytest.raises(ValueError, match="bilinear and nearest"):
        point_sample(jnp.zeros((1, 1, 2, 2)), jnp.zeros((1, 1, 2)), mode="bicubic")


def test_uncertainty_prefers_the_decision_boundary() -> None:
    """A logit near zero is more uncertain than a confident one."""
    values = calculate_uncertainty(jnp.asarray([0.0, 5.0, -5.0]))
    assert float(values[0]) > float(values[1]) and float(values[0]) > float(values[2])


def test_point_sampling_returns_the_requested_count() -> None:
    """Importance plus random sampling returns exactly ``num_points`` coordinates in range."""
    logits = jnp.asarray(np.random.RandomState(0).randn(3, 1, 8, 8).astype("float32"))
    coords = get_uncertain_point_coords_with_randomness(logits, 20, jax.random.key(0))
    assert coords.shape == (3, 20, 2)
    assert float(coords.min()) >= 0.0 and float(coords.max()) <= 1.0


def test_mask_losses_reward_a_correct_prediction() -> None:
    """Predicting the ground-truth mask scores lower than predicting its inverse."""
    rng = np.random.RandomState(0)
    target = (rng.rand(1, 2, 16, 16) > 0.5).astype("float32")
    match = (
        jnp.asarray([[0, 1]]),
        jnp.asarray([[0, 1]]),
        jnp.asarray([[True, True]]),
    )
    targets = jnp.asarray(target)
    good = jnp.asarray(target * 20 - 10)
    bad = -good
    num_boxes = jnp.float32(2.0)
    good_ce, good_dice = loss_masks(good, targets, match, num_boxes, jax.random.key(0))
    bad_ce, bad_dice = loss_masks(bad, targets, match, num_boxes, jax.random.key(0))
    assert float(good_ce) < float(bad_ce)
    assert float(good_dice) < float(bad_dice)


def test_mask_losses_ignore_padded_matches() -> None:
    """Invalid match slots contribute nothing to the mask losses."""
    target = jnp.asarray(np.zeros((1, 2, 8, 8), dtype="float32"))
    predictions = jnp.zeros((1, 2, 8, 8))
    all_valid = (jnp.asarray([[0, 1]]), jnp.asarray([[0, 1]]), jnp.asarray([[True, True]]))
    one_valid = (jnp.asarray([[0, 1]]), jnp.asarray([[0, 1]]), jnp.asarray([[True, False]]))
    both = loss_masks(predictions, target, all_valid, jnp.float32(2.0), jax.random.key(0))
    single = loss_masks(predictions, target, one_valid, jnp.float32(2.0), jax.random.key(0))
    assert float(single[0]) == pytest.approx(float(both[0]) / 2, rel=1e-5)


def test_targets_carry_optional_masks() -> None:
    """``Targets`` stays a valid pytree with and without masks."""
    without = Targets(jnp.zeros((1, 2), jnp.int32), jnp.zeros((1, 2, 4)), jnp.zeros((1, 2), bool))
    with_masks = Targets(
        jnp.zeros((1, 2), jnp.int32), jnp.zeros((1, 2, 4)), jnp.zeros((1, 2), bool), jnp.zeros((1, 2, 4, 4))
    )
    assert len(jax.tree.leaves(without)) == 3
    assert len(jax.tree.leaves(with_masks)) == 4


def test_point_sample_uses_border_padding_like_upstream() -> None:
    """``grid_sample(padding_mode="border")``: a point outside the pixel centres clamps to the edge."""
    values = jnp.asarray([[[[2.0, 4.0], [6.0, 8.0]]]])
    corner = point_sample(values, jnp.asarray([[[0.0, 0.0]]]))  # pixel coordinate -0.5: outside the centres
    assert float(corner[0, 0, 0]) == 2.0  # zeros padding would return 0.5
    far = point_sample(values, jnp.asarray([[[1.0, 1.0]]]))
    assert float(far[0, 0, 0]) == 8.0
    centre = point_sample(values, jnp.asarray([[[0.5, 0.5]]]))
    assert float(centre[0, 0, 0]) == 5.0
