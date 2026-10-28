"""Transformer helpers: proposals, valid ratios, query-group folding and reference refinement."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfdetr_jax.models.position_encoding import gen_sineembed_for_position, position_embedding_sine
from rfdetr_jax.models.transformer import (
    _merge_groups,
    _split_groups,
    gen_encoder_output_proposals,
    get_valid_ratio,
    refpoints_refine,
)


def test_group_folding_roundtrips() -> None:
    """Folding query groups onto the batch axis and back is the identity."""
    x = jnp.asarray(np.arange(2 * 6 * 3, dtype="float32").reshape(2, 6, 3))
    folded = _split_groups(x, per_group=2)
    assert folded.shape == (6, 2, 3)
    assert np.allclose(np.asarray(_merge_groups(folded, batch=2)), np.asarray(x))


def test_group_folding_matches_torch_ordering() -> None:
    """Group ``g`` of image ``b`` lands at row ``g * batch + b``, as ``split``/``cat`` produce."""
    batch, groups, per_group = 2, 3, 4
    x = jnp.asarray(
        np.arange(batch * groups * per_group, dtype="float32").reshape(batch, groups * per_group, 1)
    )
    folded = np.asarray(_split_groups(x, per_group))
    for group in range(groups):
        for image in range(batch):
            expected = np.asarray(x)[image, group * per_group : (group + 1) * per_group]
            assert np.allclose(folded[group * batch + image], expected)


def test_valid_ratio_counts_unpadded_rows_and_columns() -> None:
    """The valid ratio is the unpadded fraction of each axis."""
    mask = np.zeros((1, 10, 20), dtype=bool)
    mask[0, 8:, :] = True
    mask[0, :, 15:] = True
    ratios = np.asarray(get_valid_ratio(jnp.asarray(mask)))
    assert ratios[0, 0] == pytest.approx(15 / 20)
    assert ratios[0, 1] == pytest.approx(8 / 10)


def test_proposals_cover_the_grid() -> None:
    """Anchor proposals are pixel centres with a level-dependent size."""
    memory = jnp.zeros((1, 16, 8))
    _, proposals = gen_encoder_output_proposals(memory, None, ((4, 4),), unsigmoid=False)
    assert proposals.shape == (1, 16, 4)
    centers = np.asarray(proposals[0, :, :2])
    assert centers.min() == pytest.approx(0.125)
    assert centers.max() == pytest.approx(0.875)
    assert np.allclose(np.asarray(proposals[0, :, 2:]), 0.05)


def test_padded_positions_are_zeroed() -> None:
    """Masked encoder positions contribute neither memory nor a proposal."""
    memory = jnp.ones((1, 16, 8))
    mask = np.zeros((1, 16), dtype=bool)
    mask[0, 8:] = True
    output_memory, proposals = gen_encoder_output_proposals(
        memory, jnp.asarray(mask), ((4, 4),), unsigmoid=False
    )
    assert np.allclose(np.asarray(output_memory)[0, 8:], 0.0)
    assert np.allclose(np.asarray(proposals)[0, 8:], 0.0)


def test_reparametrised_refinement_scales_by_the_reference() -> None:
    """With ``bbox_reparam`` the delta is applied relative to the reference box."""
    reference = jnp.asarray([[[0.5, 0.5, 0.2, 0.4]]])
    delta = jnp.asarray([[[1.0, -1.0, 0.0, 0.0]]])
    refined = np.asarray(refpoints_refine(reference, delta, bbox_reparam=True))[0, 0]
    assert refined[0] == pytest.approx(0.5 + 1.0 * 0.2)
    assert refined[1] == pytest.approx(0.5 - 1.0 * 0.4)
    assert refined[2] == pytest.approx(0.2)


def test_additive_refinement_is_a_plain_sum() -> None:
    """Without ``bbox_reparam`` refinement is addition in logit space."""
    reference = jnp.asarray([[[0.5, 0.5, 0.2, 0.4]]])
    delta = jnp.asarray([[[0.1, 0.1, 0.1, 0.1]]])
    assert np.allclose(
        np.asarray(refpoints_refine(reference, delta, bbox_reparam=False)),
        np.asarray(reference) + np.asarray(delta),
    )


def test_position_embedding_channels_and_range() -> None:
    """The sine embedding has ``2 * num_pos_feats`` channels bounded by [-1, 1]."""
    mask = jnp.zeros((2, 6, 8), bool)
    embedding = position_embedding_sine(mask, num_pos_feats=16)
    assert embedding.shape == (2, 32, 6, 8)
    assert float(jnp.abs(embedding).max()) <= 1.0 + 1e-6


def test_reference_sine_embedding_widths() -> None:
    """A 2-coordinate reference gives ``2 * dim`` features and a 4-coordinate one ``4 * dim``."""
    two = gen_sineembed_for_position(jnp.zeros((1, 5, 2)), dim=8)
    four = gen_sineembed_for_position(jnp.zeros((1, 5, 4)), dim=8)
    assert two.shape == (1, 5, 16)
    assert four.shape == (1, 5, 32)
    with pytest.raises(ValueError, match="Unknown pos_tensor shape"):
        gen_sineembed_for_position(jnp.zeros((1, 5, 3)), dim=8)


@pytest.mark.parametrize("height", [24, 48])
def test_proposals_carry_torch_float32_rounding(height: int) -> None:
    """Proposal centres are the IEEE float32 quotients ``(i + 0.5) / valid_size`` upstream computes.

    Both frameworks build this grid in float32 whatever the activation dtype, so a one-ulp
    difference here is a formula difference that the decoder amplifies, not rounding noise. The
    48-point grid also covers ``jnp.linspace``, whose float32 step is not exactly one there.
    """
    batch, width = 2, height
    mask = np.zeros((batch, height, width), dtype=bool)
    mask[1, height - 2 :, :] = True
    mask[1, :, width - 3 :] = True
    valid_h = (~mask[:, :, 0]).sum(1).astype(np.float32)
    valid_w = (~mask[:, 0, :]).sum(1).astype(np.float32)
    grid_y, grid_x = np.meshgrid(
        np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij"
    )
    grid = np.stack([grid_x, grid_y], -1)[None]
    scale = np.stack([valid_w, valid_h], -1).reshape(batch, 1, 1, 2)
    centres = (grid + np.float32(0.5)) / scale
    expected = np.concatenate([centres, np.full_like(centres, np.float32(0.05))], -1).reshape(batch, -1, 4)
    keep = ((expected > 0.01) & (expected < 0.99)).all(-1, keepdims=True) & ~mask.reshape(batch, -1, 1)
    expected = np.where(keep, expected, np.float32(0))

    def proposals(memory, padding):
        """Proposal half of the two-stage anchor generator."""
        return gen_encoder_output_proposals(memory, padding, ((height, width),), unsigmoid=False)[1]

    memory = jnp.zeros((batch, height * width, 8), jnp.float32)
    actual = np.asarray(jax.jit(proposals)(memory, jnp.asarray(mask.reshape(batch, -1))))
    assert np.array_equal(actual, expected)


def test_valid_ratio_is_the_rounded_quotient() -> None:
    """Every valid fraction equals the float32 quotient torch computes, for every valid size."""
    size = 24
    for valid in range(1, size + 1):
        mask = np.ones((1, size, size), dtype=bool)
        mask[:, :valid, :valid] = False
        ratios = np.asarray(jax.jit(get_valid_ratio)(jnp.asarray(mask)))
        assert ratios.dtype == np.float32
        assert ratios[0, 0] == np.float32(valid) / np.float32(size)
        assert ratios[0, 1] == np.float32(valid) / np.float32(size)
