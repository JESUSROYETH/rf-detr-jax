"""GPU XLA defaults: added once, never over a user's own choice."""

from __future__ import annotations

import importlib.util

import pytest

from rfdetr_jax.utilities import gpu_flags


@pytest.fixture
def cuda(monkeypatch):
    """Pretend the CUDA plugin is installed."""
    real = importlib.util.find_spec
    monkeypatch.setattr(gpu_flags.importlib.util, "find_spec", lambda name: object() if "cuda" in name else real(name))


def test_adds_the_defaults_after_existing_flags(cuda) -> None:
    env = {"XLA_FLAGS": "--xla_dump_to=/tmp/x"}
    assert gpu_flags.apply_gpu_xla_defaults(env) == ["--xla_gpu_enable_triton_gemm=false"]
    assert env["XLA_FLAGS"] == "--xla_dump_to=/tmp/x --xla_gpu_enable_triton_gemm=false"
    assert gpu_flags.apply_gpu_xla_defaults(env) == []  # idempotent


def test_keeps_a_user_choice(cuda) -> None:
    env = {"XLA_FLAGS": "--xla_gpu_enable_triton_gemm=true"}
    assert gpu_flags.apply_gpu_xla_defaults(env) == []
    assert env["XLA_FLAGS"] == "--xla_gpu_enable_triton_gemm=true"


def test_does_nothing_without_the_cuda_plugin(monkeypatch) -> None:
    monkeypatch.setattr(gpu_flags.importlib.util, "find_spec", lambda name: None)
    env: dict = {}
    assert gpu_flags.apply_gpu_xla_defaults(env) == []
    assert env == {}
