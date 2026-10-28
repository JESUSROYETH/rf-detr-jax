"""The on-device auction solver, against the SciPy solve it stands in for."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from rfdetr_jax.models.matcher import auction_match, hungarian_match


def _padded_cost(rng: np.random.Generator, shape: tuple[int, ...], real: int) -> np.ndarray:
    """Build a ``[..., B, Q, T]`` cost whose padded columns carry the sentinel `build_cost_matrix` uses."""
    cost = rng.standard_normal(shape).astype(np.float32) * 3.0
    cost[..., real:] = 1e4
    return cost


def _objective(cost: np.ndarray, rows: np.ndarray, cols: np.ndarray, valid: np.ndarray) -> float:
    """Total cost of an assignment, summed over every leading axis."""
    leading = tuple(axis[..., None] for axis in np.indices(rows.shape[:-1]))
    return float((cost[(*leading, rows, cols)] * valid).sum())


@pytest.mark.parametrize(
    ("groups", "per_group", "targets", "real"),
    [(1, 40, 6, 3), (4, 25, 6, 6), (3, 30, 8, 1), (2, 20, 5, 0)],
)
def test_auction_matches_the_scipy_objective(groups, per_group, targets, real) -> None:
    """The auction must reach the same total cost as the exact solve it replaces."""
    rng = np.random.default_rng(0)
    batch = 2
    cost = _padded_cost(rng, (2, batch, groups * per_group, targets), real)
    device_cost = jnp.asarray(cost)
    sizes = jnp.full((batch,), real, jnp.int32)

    exact = [np.asarray(x) for x in jax.jit(lambda c, s: hungarian_match(c, s, groups))(device_cost, sizes)]
    solved = [np.asarray(x) for x in jax.jit(lambda c, s: auction_match(c, s, groups))(device_cost, sizes)]

    np.testing.assert_array_equal(exact[2], solved[2])
    best = _objective(cost, *exact)
    found = _objective(cost, *solved)
    assert found == pytest.approx(best, rel=1e-5, abs=1e-4)


def test_auction_never_gives_two_targets_the_same_query() -> None:
    """Within a group the assigned queries must be distinct, however the bidding went."""
    rng = np.random.default_rng(1)
    groups, per_group, targets, real, batch = 3, 25, 8, 8, 2
    # Identical columns are the worst case: every target wants the query every other target wants.
    column = rng.standard_normal((1, batch, groups * per_group, 1)).astype(np.float32)
    cost = np.repeat(column, targets, axis=-1)
    cost[..., real:] = 1e4

    rows, _, valid = jax.jit(lambda c, s: auction_match(c, s, groups))(
        jnp.asarray(cost), jnp.full((batch,), real, jnp.int32)
    )
    rows, valid = np.asarray(rows), np.asarray(valid)
    for image in range(batch):
        assert valid[0, image].sum() == groups * real
        for group in range(groups):
            window = slice(group * targets, group * targets + targets)
            assigned = rows[0, image, window][valid[0, image, window]]
            assert len(set(assigned.tolist())) == len(assigned)
            assert ((assigned >= group * per_group) & (assigned < (group + 1) * per_group)).all()


def test_auction_assigns_every_valid_target_even_when_capped() -> None:
    """The greedy completion must leave no valid target unmatched at a punishing round cap."""
    rng = np.random.default_rng(2)
    groups, per_group, targets, real, batch = 2, 30, 10, 10, 2
    base = rng.standard_normal((1, batch, groups * per_group, 1)).astype(np.float32)
    cost = (base + 1e-4 * rng.standard_normal((1, batch, groups * per_group, targets))).astype(np.float32)
    cost[..., real:] = 1e4

    rows, cols, valid = auction_match(
        jnp.asarray(cost), jnp.full((batch,), real, jnp.int32), groups, max_rounds=1
    )
    assert int(np.asarray(valid).sum()) == batch * groups * real
    for image in range(batch):
        for group in range(groups):
            window = slice(group * targets, group * targets + targets)
            assigned = np.asarray(rows)[0, image, window][np.asarray(valid)[0, image, window]]
            assert len(set(assigned.tolist())) == len(assigned)


def test_hungarian_folds_leading_axes_into_one_call() -> None:
    """Stacked layers must give exactly what solving each layer on its own gives."""
    rng = np.random.default_rng(3)
    groups, per_group, targets, real, batch = 2, 15, 4, 3, 2
    cost = _padded_cost(rng, (3, batch, groups * per_group, targets), real)
    sizes = jnp.full((batch,), real, jnp.int32)

    stacked = [np.asarray(x) for x in hungarian_match(jnp.asarray(cost), sizes, groups)]
    for layer in range(cost.shape[0]):
        alone = [np.asarray(x) for x in hungarian_match(jnp.asarray(cost[layer]), sizes, groups)]
        for folded, single in zip(stacked, alone, strict=True):
            np.testing.assert_array_equal(folded[layer], single)


def test_auction_agrees_with_a_hand_solved_problem() -> None:
    """A tiny instance the reference solver settles unambiguously."""
    cost = np.array([[[[4.0, 1.0], [2.0, 3.0], [9.0, 9.0]]]], np.float32)
    rows, cols, valid = auction_match(jnp.asarray(cost), jnp.asarray([2], jnp.int32), 1)
    rows, cols, valid = np.asarray(rows)[0, 0], np.asarray(cols)[0, 0], np.asarray(valid)[0, 0]
    expected_rows, expected_cols = linear_sum_assignment(cost[0, 0])
    assert sorted(zip(rows[valid].tolist(), cols[valid].tolist(), strict=True)) == sorted(
        zip(expected_rows.tolist(), expected_cols.tolist(), strict=True)
    )


def test_auction_rejects_more_targets_than_queries() -> None:
    """Greedy completion cannot assign three targets to two queries without duplicates."""
    cost = jnp.array([[[1., 2., 3.], [4., 5., 6.]]])
    with pytest.raises(ValueError, match="padded target count <= queries per group"):
        jax.jit(lambda c: auction_match(c, jnp.array([3]), 1))(cost)
