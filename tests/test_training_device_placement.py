"""Worker threads retain the public caller's selected JAX execution device."""
from contextlib import ExitStack
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfdetr_jax.training.run import precompile_steps
from rfdetr_jax.training.trainer import device_batches


def cpu_context(stack, spelling):
    selected = jax.devices("cpu")[0] if spelling == "device" else "cpu"
    try:
        stack.enter_context(jax.default_device(selected))
    except ValueError:
        if spelling != "platform":
            raise
        pytest.skip("this JAX version does not accept a platform-string default device")
    return selected


@pytest.mark.parametrize("spelling", ["device", "platform"])
@pytest.mark.parametrize("explicit_sharding", [False, True])
def test_prefetch_preserves_selected_cpu_device_and_dtype(spelling, explicit_sharding, monkeypatch):
    """Also runs in CUDA-default environments: selecting CPU must keep transfers there."""
    cpu = jax.devices("cpu")[0]
    sharding = jax.sharding.SingleDeviceSharding(cpu) if explicit_sharding else None
    original_context = jax.config.jax_default_device
    original_asarray, original_put = jnp.asarray, jax.device_put
    observed = []

    def asarray(*args, **kwargs):
        observed.append(jax.config.jax_default_device)
        return original_asarray(*args, **kwargs)

    def put(*args, **kwargs):
        observed.append(jax.config.jax_default_device)
        return original_put(*args, **kwargs)

    monkeypatch.setattr(jnp, "asarray", asarray)
    monkeypatch.setattr(jax, "device_put", put)
    marker = object()
    batches = [{"pixels": np.ones((1, 3, 2, 2), np.float16),
                "labels": np.array([[90]], np.int32), "marker": marker}]
    with ExitStack() as stack:
        selected = cpu_context(stack, spelling)
        iterator = device_batches(iter(batches), ("pixels", "labels"), sharding=sharding)
        try:
            result = next(iterator)
            with pytest.raises(StopIteration):
                next(iterator)
        finally:
            iterator.close()
        assert observed and all(context == selected for context in observed)
        for name in ("pixels", "labels"):
            assert result[name].devices() == {cpu}
            assert result[name].dtype == batches[0][name].dtype
            np.testing.assert_array_equal(result[name], batches[0][name])
            if explicit_sharding:
                assert result[name].sharding == sharding
        assert result["marker"] is marker
    assert jax.config.jax_default_device == original_context


@pytest.mark.parametrize("spelling", ["device", "platform"])
@pytest.mark.parametrize("explicit_sharding", [False, True])
def test_precompile_workers_preserve_device_context_and_sharding(spelling, explicit_sharding):
    cpu = jax.devices("cpu")[0]
    sharding = jax.sharding.SingleDeviceSharding(cpu) if explicit_sharding else None
    original_context = jax.config.jax_default_device
    observed = []

    def lower(state, batch, key):
        observed.append(jax.config.jax_default_device)
        assert batch["pixel_values"].dtype == jnp.float16
        assert batch["labels"].dtype == jnp.int32
        for value in batch.values():
            assert value.sharding == sharding
        return SimpleNamespace(compile=lambda: batch["labels"].shape[1])

    with ExitStack() as stack:
        selected = cpu_context(stack, spelling)
        result = precompile_steps(SimpleNamespace(lower=lower), None, None, [1, 3],
                                  1, 2, jnp.float16, sharding)
        assert result == {1: 1, 3: 3}
        assert observed == [selected, selected]
    assert jax.config.jax_default_device == original_context


def test_precompiled_tiny_function_executes_on_selected_cpu():
    """Exercise actual lowering; no RF-DETR graph or training workload is involved."""
    cpu = jax.devices("cpu")[0]

    @jax.jit
    def step(state, batch, key):
        del key
        return state + jnp.sum(batch["pixel_values"])

    batch = {"pixel_values": np.ones((1, 3, 2, 2), np.float32),
             "labels": np.zeros((1, 1), np.int32), "boxes": np.zeros((1, 1, 4), np.float32),
             "target_mask": np.ones((1, 1), bool)}
    with jax.default_device(cpu):
        state, key = jnp.asarray(1., jnp.float32), jax.random.key(0)
        compiled = precompile_steps(step, state, key, [1], 1, 2, jnp.float32, None)
        result = compiled[1](state, batch, key)
    assert result.devices() == {cpu}
    assert float(result) == 13.
