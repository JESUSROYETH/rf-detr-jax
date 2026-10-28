"""Deformable-attention sampling primitives."""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfdetr_jax.models.ops import (
    MSDeformAttn,
    bilinear_grid_sample,
    ms_deform_attn_core,
    ms_deform_attn_core_rows,
    precise_divide,
    sampling_offsets_bias_init,
)


def test_grid_sample_reads_pixel_centres() -> None:
    """Sampling at a pixel centre returns that pixel exactly."""
    values = jnp.asarray([[[[1.0, 2.0], [3.0, 4.0]]]])
    # align_corners=False puts pixel centres at -0.5 and +0.5 for a 2-wide axis.
    grid = jnp.asarray([[[[-0.5, -0.5], [0.5, 0.5]]]])
    sampled = np.asarray(bilinear_grid_sample(values, grid))[0, 0, 0]
    assert sampled[0] == pytest.approx(1.0)
    assert sampled[1] == pytest.approx(4.0)


def test_grid_sample_zero_pads_outside() -> None:
    """Out-of-bounds samples contribute zero, matching ``padding_mode='zeros'``."""
    values = jnp.ones((1, 1, 2, 2))
    grid = jnp.asarray([[[[5.0, 5.0]]]])
    assert float(bilinear_grid_sample(values, grid)[0, 0, 0, 0]) == 0.0


def test_grid_sample_interpolates_linearly() -> None:
    """A point midway between two pixel centres averages them."""
    values = jnp.asarray([[[[0.0, 4.0]]]])
    grid = jnp.asarray([[[[0.0, 0.0]]]])
    assert float(bilinear_grid_sample(values, grid)[0, 0, 0, 0]) == pytest.approx(2.0)


def test_core_weights_sum_to_the_sampled_value() -> None:
    """With uniform attention weights the core returns the mean of the sampled points."""
    batch, heads, head_dim, height, width, queries, points = 1, 1, 1, 2, 2, 1, 2
    value = jnp.asarray(np.arange(4, dtype="float32")).reshape(batch, heads, head_dim, height * width)
    locations = jnp.asarray([[[[[[0.25, 0.25], [0.75, 0.75]]]]]])
    weights = jnp.full((batch, queries, heads, points), 0.5)
    output = np.asarray(ms_deform_attn_core(value, ((height, width),), locations, weights))
    assert output.shape == (batch, queries, heads * head_dim)
    assert output[0, 0, 0] == pytest.approx(1.5)


def test_offset_bias_spreads_points_radially() -> None:
    """The sampling-offset bias places point ``k`` at radius ``k + 1`` around each head's angle."""
    bias = np.asarray(sampling_offsets_bias_init(n_heads=4, n_levels=1, n_points=3))
    grid = bias.reshape(4, 1, 3, 2)
    radii = np.abs(grid).max(-1)
    assert np.allclose(radii[:, 0, 0], 1.0)
    assert np.allclose(radii[:, 0, 1], 2.0)
    assert np.allclose(radii[:, 0, 2], 3.0)


@pytest.mark.parametrize("reference_dim", [2, 4])
def test_module_runs_for_both_reference_forms(reference_dim: int) -> None:
    """Centre references and box references both produce the right output shape."""
    module = MSDeformAttn(d_model=16, n_levels=1, n_heads=2, n_points=2)
    query = jnp.zeros((2, 5, 16))
    reference = jnp.full((2, 5, 1, reference_dim), 0.5)
    flattened = jnp.zeros((2, 16, 16))
    params = module.init(jax.random.key(0), query, reference, flattened, ((4, 4),))
    output = module.apply(params, query, reference, flattened, ((4, 4),))
    assert output.shape == (2, 5, 16)


def test_module_rejects_bad_reference_dim() -> None:
    """A reference with 3 trailing values is rejected rather than silently mis-sampled."""
    module = MSDeformAttn(d_model=16, n_levels=1, n_heads=2, n_points=2)
    query = jnp.zeros((1, 3, 16))
    flattened = jnp.zeros((1, 16, 16))
    with pytest.raises(ValueError, match="must be 2 or 4"):
        module.init(jax.random.key(0), query, jnp.zeros((1, 3, 1, 3)), flattened, ((4, 4),))


def test_padding_mask_zeroes_value_positions() -> None:
    """Fully masked memory gives the same output regardless of what the memory held."""
    module = MSDeformAttn(d_model=8, n_levels=1, n_heads=2, n_points=2)
    query = jnp.zeros((1, 2, 8))
    reference = jnp.full((1, 2, 1, 2), 0.5)
    mask = jnp.ones((1, 4), bool)
    params = module.init(jax.random.key(0), query, reference, jnp.ones((1, 4, 8)), ((2, 2),))
    ones = module.apply(params, query, reference, jnp.ones((1, 4, 8)), ((2, 2),), mask)
    other = module.apply(params, query, reference, jnp.full((1, 4, 8), 7.0), ((2, 2),), mask)
    assert np.allclose(np.asarray(ones), np.asarray(other), atol=1e-6)


def test_padding_mask_changes_the_output() -> None:
    """Masking is not a no-op: an unmasked forward differs from a fully masked one."""
    module = MSDeformAttn(d_model=8, n_levels=1, n_heads=2, n_points=2)
    query = jnp.zeros((1, 2, 8))
    reference = jnp.full((1, 2, 1, 2), 0.5)
    memory = jnp.asarray(np.random.RandomState(0).randn(1, 4, 8).astype("float32"))
    params = module.init(jax.random.key(0), query, reference, memory, ((2, 2),))
    unmasked = module.apply(params, query, reference, memory, ((2, 2),))
    masked = module.apply(params, query, reference, memory, ((2, 2),), jnp.ones((1, 4), bool))
    assert not np.allclose(np.asarray(unmasked), np.asarray(masked))


def test_matmul_sampler_matches_the_gather_sampler() -> None:
    """The dense-kernel sampler must reproduce the gather sampler, edges and all."""
    batch, heads, head_dim, height, width, queries, points = 2, 4, 8, 7, 5, 11, 3
    key = jax.random.key(0)
    value = jax.random.normal(key, (batch * heads, height * width, head_dim), jnp.float32)
    # Deliberately sample outside [0, 1] so the zero-padding branch is exercised.
    locations = jax.random.uniform(key, (batch, queries, heads, 1, points, 2), jnp.float32) * 1.4 - 0.2
    weights = jax.random.uniform(key, (batch, queries, heads, points), jnp.float32)

    shapes = ((height, width),)
    gathered = ms_deform_attn_core_rows(value, shapes, locations, weights, "gather")
    contracted = ms_deform_attn_core_rows(value, shapes, locations, weights, "matmul")
    assert np.abs(np.asarray(gathered) - np.asarray(contracted)).max() < 1e-5


def test_gather_f32_sampler_matches_the_gather_sampler() -> None:
    """``gather-f32`` is the gather on float32 features: identical in float32, close in bf16."""
    batch, heads, head_dim, height, width, queries, points = 2, 4, 8, 7, 5, 11, 3
    key = jax.random.key(1)
    value = jax.random.normal(key, (batch * heads, height * width, head_dim), jnp.float32)
    locations = jax.random.uniform(key, (batch, queries, heads, 1, points, 2), jnp.float32) * 1.4 - 0.2
    weights = jax.random.uniform(key, (batch, queries, heads, points), jnp.float32)
    shapes = ((height, width),)

    gathered = ms_deform_attn_core_rows(value, shapes, locations, weights, "gather")
    widened = ms_deform_attn_core_rows(value, shapes, locations, weights, "gather-f32")
    np.testing.assert_array_equal(np.asarray(gathered), np.asarray(widened))

    # On bf16 features the output keeps the activation dtype, and the gradient into the feature map
    # (the scatter-add this sampler exists for) tracks the float32 one to bf16 precision.
    def value_grad(sampler: str, dtype) -> np.ndarray:
        def loss(v):
            out = ms_deform_attn_core_rows(v, shapes, locations, weights.astype(dtype), sampler)
            return (out.astype(jnp.float32) ** 2).sum()

        return np.asarray(jax.grad(loss)(value.astype(dtype)).astype(jnp.float32))

    out_bf16 = ms_deform_attn_core_rows(
        value.astype(jnp.bfloat16), shapes, locations, weights.astype(jnp.bfloat16), "gather-f32"
    )
    assert out_bf16.dtype == jnp.bfloat16
    reference = value_grad("gather", jnp.float32)
    candidate = value_grad("gather-f32", jnp.bfloat16)
    assert np.linalg.norm(candidate - reference) / np.linalg.norm(reference) < 2e-2

    # The point of the sampler: its backward scatter-add runs on float32, where CUDA has a native atomic.
    def scatter_dtypes(sampler: str) -> set[str]:
        def loss(v):
            out = ms_deform_attn_core_rows(v, shapes, locations, weights.astype(jnp.bfloat16), sampler)
            return out.astype(jnp.float32).sum()

        text = str(jax.make_jaxpr(jax.grad(loss))(value.astype(jnp.bfloat16)))
        return set(re.findall(r":(\w+)\[[\d,]*\] = scatter-add", text))

    assert scatter_dtypes("gather") == {"bf16"}
    assert scatter_dtypes("gather-f32") == {"f32"}


def test_auto_sampler_picks_the_matmul_kernel_when_it_is_small() -> None:
    """``auto`` must agree with whichever sampler it selects, on both sides of the budget."""
    from rfdetr_jax.models import ops

    batch, heads, head_dim, side, queries, points = 1, 2, 4, 6, 5, 2
    key = jax.random.key(3)
    value = jax.random.normal(key, (batch * heads, side * side, head_dim), jnp.float32)
    locations = jax.random.uniform(key, (batch, queries, heads, 1, points, 2), jnp.float32)
    weights = jax.random.uniform(key, (batch, queries, heads, points), jnp.float32)
    shapes = ((side, side),)

    automatic = ms_deform_attn_core_rows(value, shapes, locations, weights, "auto")
    contracted = ms_deform_attn_core_rows(value, shapes, locations, weights, "matmul")
    assert np.abs(np.asarray(automatic) - np.asarray(contracted)).max() < 1e-5

    # Above the budget `auto` must fall back to the gather, and still agree with it.
    original = ops._MATMUL_SAMPLER_MAX_LEVEL
    try:
        ops._MATMUL_SAMPLER_MAX_LEVEL = 1
        fallback = ops.ms_deform_attn_core_rows(value, shapes, locations, weights, "auto")
    finally:
        ops._MATMUL_SAMPLER_MAX_LEVEL = original
    gathered = ms_deform_attn_core_rows(value, shapes, locations, weights, "gather")
    assert np.abs(np.asarray(fallback) - np.asarray(gathered)).max() == 0.0


def test_unknown_sampler_is_rejected() -> None:
    """A typo in the sampler name must fail loudly rather than silently pick one."""
    value = jnp.zeros((2, 4, 3), jnp.float32)
    locations = jnp.zeros((1, 2, 2, 1, 1, 2), jnp.float32)
    weights = jnp.zeros((1, 2, 2, 1), jnp.float32)
    with pytest.raises(ValueError, match="unknown sampler"):
        ms_deform_attn_core_rows(value, ((2, 2),), locations, weights, "nearest")


def test_select_rows_matches_a_plain_gather() -> None:
    """Collapsing the batch axes must not change which rows come back."""
    from rfdetr_jax.models.layers import select_rows

    groups, batch, rows, width, picks = 3, 2, 11, 5, 4
    key = jax.random.key(7)
    values = jax.random.normal(key, (groups, batch, rows, width), jnp.float32)
    indices = jax.random.randint(key, (groups, batch, picks), 0, rows, jnp.int32)

    expected = jnp.take_along_axis(values, indices[..., None], axis=2)
    np.testing.assert_array_equal(np.asarray(select_rows(values, indices)), np.asarray(expected))

    # It must also work with a single leading axis, which is the shape it collapses to.
    flat_values, flat_indices = values[0], indices[0]
    np.testing.assert_array_equal(
        np.asarray(select_rows(flat_values, flat_indices)),
        np.asarray(jnp.take_along_axis(flat_values, flat_indices[..., None], axis=1)),
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_pallas_sampler_matches_the_gather_sampler(dtype) -> None:
    """The fused kernel reproduces the gather sampler's output and all three gradients."""
    rng = np.random.default_rng(3)
    batch, queries, heads, levels, points, head_dim = 2, 70, 2, 2, 3, 8
    shapes = ((8, 16), (4, 8))
    length = sum(h * w for h, w in shapes)
    value = jnp.asarray(rng.standard_normal((batch * heads, length, head_dim)), dtype)
    locations = jnp.asarray(rng.uniform(-0.1, 1.1, (batch, queries, heads, levels, points, 2)), jnp.float32)
    weights = jnp.asarray(rng.uniform(0, 1, (batch, queries, heads, levels * points)), jnp.float32)
    cotangent = jnp.asarray(rng.standard_normal((batch, queries, heads * head_dim)), jnp.float32)

    def loss(sampler, v, loc, w):
        out = ms_deform_attn_core_rows(v.astype(jnp.float32) if sampler == "gather" else v, shapes, loc, w, sampler)
        return jnp.sum(out.astype(jnp.float32) * cotangent)

    with jax.default_matmul_precision("highest"):
        expected = jax.value_and_grad(lambda *a: loss("gather", *a), argnums=(0, 1, 2))(value, locations, weights)
        actual = jax.value_and_grad(lambda *a: loss("pallas", *a), argnums=(0, 1, 2))(value, locations, weights)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-5)
    for got, want in zip(actual[1], expected[1], strict=True):
        got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
        tolerance = 1e-5 if dtype == jnp.float32 else 1e-2
        assert np.max(np.abs(got - want)) <= tolerance * max(1.0, np.max(np.abs(want)))


def test_pallas_tile_fits_the_vmem_budget_and_the_samples() -> None:
    """Tiles widen to 4096 on Nano's map, narrow on bigger maps, and never exceed what a call needs."""
    from rfdetr_jax.models.deform_pallas import _TILE_BUDGET_BYTES, _tile_for

    assert _tile_for(24 * 24, 3900 * 2) == 4096
    assert _tile_for(40 * 40, 3900 * 2) == 1024
    assert _tile_for(24 * 24, 100) == 128
    for positions in (144, 576, 1600, 6400, 25600):
        tile = _tile_for(positions, 10**6)
        assert tile >= 128 and tile & (tile - 1) == 0
        assert 3 * positions * tile * 4 <= _TILE_BUDGET_BYTES or tile == 128


def test_precise_divide_rounds_once_like_torch() -> None:
    """Dividing by a broadcast divisor gives the IEEE quotient, not a reciprocal product.

    XLA rewrites ``x / broadcast(y)`` into ``x * broadcast(1 / y)``; the extra rounding moves about a
    quarter of the quotients by one ulp, which the decoder amplifies through its sine embeddings.
    """
    rng = np.random.default_rng(0)
    numerator = rng.uniform(0.0, 6.3, (512, 1)).astype(np.float32)
    frequencies = (10000.0 ** (2 * (np.arange(128) // 2) / 128)).astype(np.float32)
    expected = numerator / frequencies[None]
    quotient = jax.jit(precise_divide)(jnp.asarray(numerator), jnp.asarray(frequencies)[None])
    assert np.array_equal(np.asarray(quotient), expected)
    scalar = jax.jit(lambda x: precise_divide(x, 24))(jnp.asarray(numerator))
    assert np.array_equal(np.asarray(scalar), numerator / np.float32(24))
    gradient = jax.grad(lambda x: precise_divide(x, jnp.asarray(frequencies)).sum())(jnp.ones((128,), jnp.float32))
    assert np.allclose(np.asarray(gradient), 1.0 / frequencies, rtol=1e-6)
