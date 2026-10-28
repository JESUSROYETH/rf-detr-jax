"""Scaled dot-product attention: the reference path, its mask, and the cuDNN route on a GPU."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfdetr_jax.models import layers
from rfdetr_jax.models.layers import scaled_dot_product_attention, set_attention_implementation


def _reference(q, k, v, mask=None):
    """Softmax attention written out in float64 NumPy."""
    q, k, v = (np.asarray(t, np.float64) for t in (q, k, v))
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(np.asarray(mask), -np.inf, scores)
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    return (scores / scores.sum(-1, keepdims=True)) @ v


def _qkv(length: int, dtype=jnp.float32, seed: int = 0):
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.standard_normal((2, 3, length, 16)), dtype) for _ in range(3))


def test_xla_path_matches_softmax_attention() -> None:
    """The default path is plain softmax attention, with ``True`` in the mask blocking a key."""
    q, k, v = _qkv(9)
    with jax.default_matmul_precision("highest"):  # a GPU defaults to TF32
        _check_xla_path(q, k, v)


def _check_xla_path(q, k, v) -> None:
    np.testing.assert_allclose(scaled_dot_product_attention(q, k, v), _reference(q, k, v), atol=1e-5)
    mask = np.zeros((9, 9), bool)
    mask[:, -3:] = True
    np.testing.assert_allclose(
        scaled_dot_product_attention(q, k, v, jnp.asarray(mask)), _reference(q, k, v, mask), atol=1e-5
    )


def test_unknown_implementation_is_rejected() -> None:
    """A typo must fail loudly rather than silently keep the previous kernel."""
    with pytest.raises(ValueError, match="unknown attention implementation"):
        set_attention_implementation("flash")


@pytest.mark.parametrize("platform", ["cpu", "gpu", "tpu"])
def test_platform_string_context_overrides_default_backend(monkeypatch, platform):
    # Selecting the policy needs no accelerator initialization or execution.
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu" if platform == "gpu" else "gpu")
    with jax.default_device(platform):
        assert layers._on_gpu() is (platform == "gpu")


@pytest.mark.parametrize("string_context", [False, True])
def test_cpu_trace_preserves_cpu_geometry_on_gpu_default_host(monkeypatch, string_context):
    from rfdetr_jax.models.transformer import refpoints_refine

    cpu = jax.devices("cpu")[0]
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    with jax.default_device("cpu" if string_context else cpu):
        reference = jnp.asarray([[[.5, .5, .2, .4]]], jnp.float16)
        delta = jnp.asarray([[[.1, -.1, .2, -.2]]], jnp.float16)
        result = jax.jit(lambda r, d: refpoints_refine(r, d, True))(reference, delta)
        expected = jnp.concatenate((delta[..., :2] * reference[..., 2:] + reference[..., :2],
                                    jnp.exp(delta[..., 2:]) * reference[..., 2:]), axis=-1)
    assert result.device.platform == "cpu"
    assert result.dtype == jnp.float16
    np.testing.assert_allclose(result, expected, rtol=1e-3, atol=0)


@pytest.mark.parametrize("platform", ["cpu", "gpu", "tpu"])
def test_platform_policy_without_context_uses_default_backend(monkeypatch, platform):
    monkeypatch.setattr(jax, "default_backend", lambda: platform)
    with jax.default_device(None):
        assert layers._on_gpu() is (platform == "gpu")


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="cuDNN attention needs a CUDA device")
@pytest.mark.parametrize("length", [145, 300])
def test_cudnn_path_matches_the_reference_on_odd_and_even_lengths(length: int) -> None:
    """cuDNN rejects odd lengths in training, so they are padded; the pad must not leak in."""
    q, k, v = _qkv(length, jnp.bfloat16, seed=length)
    previous = layers._ATTENTION
    try:
        set_attention_implementation("cudnn")
        out = scaled_dot_product_attention(q, k, v)
        grad = jax.grad(lambda q_: (scaled_dot_product_attention(q_, k, v).astype(jnp.float32) ** 2).sum())(q)
        set_attention_implementation("xla")
        grad_xla = jax.grad(lambda q_: (scaled_dot_product_attention(q_, k, v).astype(jnp.float32) ** 2).sum())(q)
    finally:
        layers._ATTENTION = previous
    assert out.shape == q.shape
    np.testing.assert_allclose(np.asarray(out, np.float32), _reference(q, k, v), atol=2e-2)
    grad, grad_xla = (np.asarray(g, np.float32) for g in (grad, grad_xla))
    assert np.linalg.norm(grad - grad_xla) / np.linalg.norm(grad_xla) < 3e-2


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="cuDNN attention needs a CUDA device")
def test_cudnn_padding_row_takes_no_attention_weight() -> None:
    """Every real key scores -4 here, so a leaked zero pad key (score 0) would take ~27% of the weight."""
    q = jnp.ones((2, 3, 145, 16), jnp.bfloat16)
    k = -jnp.ones((2, 3, 145, 16), jnp.bfloat16)
    v = jnp.asarray(np.random.default_rng(0).standard_normal((2, 3, 145, 16)), jnp.bfloat16)
    previous = layers._ATTENTION
    try:
        set_attention_implementation("cudnn")
        out = np.asarray(scaled_dot_product_attention(q, k, v), np.float32)
    finally:
        layers._ATTENTION = previous
    # Equal scores make the attention a plain mean over the 145 real values.
    expected = np.asarray(v, np.float32).mean(axis=2, keepdims=True)
    np.testing.assert_allclose(out, np.broadcast_to(expected, out.shape), atol=2e-2)


def test_multihead_attention_dropout_only_acts_in_training() -> None:
    """``nn.MultiheadAttention(dropout=p)`` drops attention probabilities while training only."""
    from rfdetr_jax.models.layers import MultiheadAttention

    inputs = jax.random.normal(jax.random.key(0), (2, 5, 8))
    plain = MultiheadAttention(8, 2)
    params = plain.init(jax.random.key(1), inputs, inputs, inputs)["params"]
    reference = plain.apply({"params": params}, inputs, inputs, inputs)
    dropped = MultiheadAttention(8, 2, dropout=0.5)
    evaluation = dropped.apply({"params": params}, inputs, inputs, inputs, train=False)
    np.testing.assert_allclose(np.asarray(evaluation), np.asarray(reference), rtol=1e-6)
    first = dropped.apply({"params": params}, inputs, inputs, inputs, train=True, rngs={"dropout": jax.random.key(2)})
    second = dropped.apply({"params": params}, inputs, inputs, inputs, train=True, rngs={"dropout": jax.random.key(3)})
    assert not np.allclose(np.asarray(first), np.asarray(reference))
    assert not np.allclose(np.asarray(first), np.asarray(second))
