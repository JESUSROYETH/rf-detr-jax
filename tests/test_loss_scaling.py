"""FP16 overflow, accumulation and complete dynamic-scaler checkpoint semantics."""
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import serialization

from rfdetr_jax.models.criterion import CriterionConfig
from rfdetr_jax.training import trainer
from rfdetr_jax.training.checkpoint import load_training_checkpoint, save_training_checkpoint


class HalfPrecisionModel:
    spec = SimpleNamespace(segmentation_head=False, use_grouppose_keypoints=False)

    def apply(self, variables, pixels, mask, **kwargs):
        # Casting the actual differentiable operation forces real FP16 gradient overflow on CPU.
        return {"prediction": (variables["params"]["weight"].astype(jnp.float16) *
                               pixels.astype(jnp.float16)).astype(jnp.float32)}


def batch(value):
    return {"pixel_values": jnp.full((1, 1, 1, 1), value, jnp.float32),
            "labels": jnp.zeros((1, 1), jnp.int32), "boxes": jnp.zeros((1, 1, 4)),
            "target_mask": jnp.ones((1, 1), bool)}


def state(optimizer, *, scaled=True, scale=8.):
    return trainer.create_train_state({"weight": jnp.array(1., jnp.float32)}, optimizer, True,
                                     loss_scaling=scaled, initial_loss_scale=scale)


def step(monkeypatch, optimizer, *, scaled=True, accumulation=1, growth=2000, schedule=None, first_copy=False):
    monkeypatch.setattr(trainer, "compute_losses", lambda outputs, *args, **kwargs: {
        "loss_bbox": jnp.mean(jnp.square(outputs["prediction"]))})
    return trainer.make_train_step(HalfPrecisionModel(), optimizer, CriterionConfig(), {"loss_bbox": 1.},
                                  trainer.TrainSettings(loss_scaling=scaled, grad_accum_steps=accumulation,
                                                        ema_decay=.5, ema_tau=0,
                                                        ema_first_update_copy=first_copy,
                                                        loss_scale_growth_interval=growth),
                                  optimizer_schedule=schedule)


def assert_tree_equal(actual, expected):
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(left, right)


def test_real_fp16_overflow_preserves_adam_but_advances_attempt_and_ema(monkeypatch):
    optimizer = optax.adamw(.1)
    train_step = step(monkeypatch, optimizer, schedule=lambda count: .5 ** count)
    current, first = train_step(state(optimizer), batch(1.), jax.random.key(0))
    params_before = jax.tree.map(np.asarray, current.params)
    moments_before = jax.tree.map(np.asarray, current.opt_state)
    ema_before = np.asarray(current.ema_params["weight"])
    current, overflow = train_step(current, batch(100.), jax.random.key(1))
    assert np.isfinite(overflow["loss"]) and float(overflow["overflow"]) == 1.
    assert_tree_equal(current.params, params_before)
    assert_tree_equal(current.opt_state, moments_before)
    assert int(current.step) == 2 and int(current.loss_scale.skipped_steps) == 1
    assert float(current.loss_scale.scale) == 4.
    assert int(current.loss_scale.growth_tracker) == 0
    np.testing.assert_array_equal(current.ema_params["weight"], .5 * ema_before + .5 * params_before["weight"])
    # The third attempt uses schedule index 2, while Adam's bias-correction count is only 2.
    reference = jax.tree.map(lambda value: jax.device_put(np.asarray(value).copy()),
                             trainer.TrainState(current.params, current.opt_state, current.ema_params, current.step))
    ordinary = step(monkeypatch, optimizer, scaled=False, schedule=lambda count: .5 ** count)
    expected, _ = ordinary(reference, batch(1.), jax.random.key(2))
    current, _ = train_step(current, batch(1.), jax.random.key(2))
    assert_tree_equal(current.params, expected.params)
    for actual, wanted in zip(jax.tree.leaves(current.opt_state), jax.tree.leaves(expected.opt_state), strict=True):
        np.testing.assert_allclose(actual, wanted, rtol=1e-7, atol=1e-10)
    assert int(current.opt_state[0].count) == 2 and int(current.step) == 3
    assert float(first["loss"]) == 1.


@pytest.mark.parametrize("clip", [False, True])
def test_scaled_accumulation_unscales_before_clipping_and_flushes_partial(monkeypatch, clip):
    optimizer = optax.chain(optax.clip_by_global_norm(.1), optax.sgd(.1)) if clip else optax.sgd(.01)
    scaled_step = step(monkeypatch, optimizer, accumulation=3)
    plain_step = step(monkeypatch, optimizer, scaled=False, accumulation=3)
    current, expected = state(optimizer), state(optimizer, scaled=False)
    for values in ((1., 2., 3.), (2.,)):
        inputs = trainer.stack_microbatches([batch(value) for value in values])
        current, actual_loss = scaled_step(current, inputs, jax.random.key(0))
        expected, expected_loss = plain_step(expected, inputs, jax.random.key(0))
        # Dividing by three before an FP16 backward rounds differently from dividing FP32 gradients afterwards.
        np.testing.assert_allclose(current.params["weight"], expected.params["weight"], rtol=2e-5, atol=1e-6)
        assert_tree_equal(current.opt_state, expected.opt_state)
        np.testing.assert_array_equal(actual_loss["loss"], expected_loss["loss"])
    assert int(current.step) == 2


def test_one_overflow_microbatch_discards_entire_group_and_resets_growth(monkeypatch):
    optimizer = optax.sgd(.01)
    train_step = step(monkeypatch, optimizer, accumulation=2, growth=2)
    current = state(optimizer)
    for _ in range(2):
        current, _ = train_step(current, trainer.stack_microbatches([batch(1.)]), jax.random.key(0))
    assert float(current.loss_scale.scale) == 16.
    before = jax.tree.map(np.asarray, current.params)
    current, metrics = train_step(current, trainer.stack_microbatches([batch(1.), batch(100.)]), jax.random.key(1))
    assert_tree_equal(current.params, before)
    assert float(current.loss_scale.scale) == 8.
    assert int(current.loss_scale.growth_tracker) == 0
    assert float(metrics["overflow"]) == 1.


@pytest.mark.parametrize("standalone", [False, True])
def test_scaled_resume_restores_overflow_growth_rng_and_schedule(tmp_path, monkeypatch, standalone):
    optimizer = optax.adamw(.01)
    train_step = step(monkeypatch, optimizer, accumulation=2, growth=2, schedule=lambda count: .8 ** count)
    current, key = state(optimizer), jax.random.key(7)
    checkpoint = tmp_path / "resume.npz"
    inputs = [(1.,), (1., 100.), (1., 2.), (1.,)]
    for index, values in enumerate(inputs):
        key, subkey = jax.random.split(key)
        current, _ = train_step(current, trainer.stack_microbatches([batch(value) for value in values]), subkey)
        if index == 1:
            save_training_checkpoint(checkpoint, current, key, {"precision": "float16"})
    final_key = np.asarray(jax.random.key_data(key)).copy()
    if standalone:
        torch = pytest.importorskip("torch")
        from rfdetr_jax.checkpoints import embed_native_resume
        public = tmp_path / "public.pth"
        torch.save({"model": {"weight": torch.tensor([1.])}}, public)
        embed_native_resume(public, checkpoint)
        copied = tmp_path / "copied.pth"
        copied.write_bytes(public.read_bytes())
        checkpoint.unlink()
        public.unlink()
        checkpoint = copied
    resumed, key, metadata = load_training_checkpoint(checkpoint, state(optimizer))
    assert metadata["checkpoint_version"] == 2 and int(resumed.loss_scale.skipped_steps) == 1
    for values in inputs[2:]:
        key, subkey = jax.random.split(key)
        resumed, _ = train_step(resumed, trainer.stack_microbatches([batch(value) for value in values]), subkey)
    assert_tree_equal(resumed, current)
    np.testing.assert_array_equal(jax.random.key_data(key), final_key)
    assert float(resumed.loss_scale.scale) == 8.
    with pytest.raises(ValueError, match="structure"):
        load_training_checkpoint(checkpoint, state(optimizer, scaled=False))


def test_version_one_checkpoint_migrates_only_original_unscaled_state(tmp_path):
    optimizer = optax.adamw(.01)
    original = state(optimizer, scaled=False)
    payload = {name: getattr(original, name) for name in ("params", "opt_state", "ema_params", "step")}
    leaves, structure = jax.tree.flatten(payload)
    path = tmp_path / "legacy.npz"
    np.savez(path, state=np.frombuffer(serialization.to_bytes(jax.device_get(leaves)), np.uint8),
             rng=np.asarray(jax.random.key_data(jax.random.key(0))),
             metadata=json.dumps({"checkpoint_version": 1, "state_structure": str(structure)}))
    restored, _, _ = load_training_checkpoint(path, state(optimizer, scaled=False))
    assert_tree_equal(restored, original)
    assert restored.loss_scale is None
    with pytest.raises(ValueError, match="no FP16 loss scaler"):
        load_training_checkpoint(path, state(optimizer))


def test_loss_scaling_recovers_a_real_fp16_underflowed_gradient(monkeypatch):
    optimizer = optax.sgd(1e6)
    scaled, scaled_metrics = step(monkeypatch, optimizer)(
        state(optimizer, scale=65536.), batch(1e-4), jax.random.key(0))
    plain, plain_metrics = step(monkeypatch, optimizer, scaled=False)(
        state(optimizer, scaled=False), batch(1e-4), jax.random.key(0))
    assert float(plain_metrics["grad_norm"]) == 0. and float(plain.params["weight"]) == 1.
    np.testing.assert_allclose(scaled_metrics["grad_norm"], 2e-8, rtol=.01)
    np.testing.assert_allclose(scaled.params["weight"], .98, atol=1e-4)


def test_accumulation_divides_before_backward_to_avoid_premature_fp16_overflow(monkeypatch):
    optimizer = optax.sgd(1e-6)
    train_step = step(monkeypatch, optimizer, accumulation=4)
    full, metrics = train_step(state(optimizer), trainer.stack_microbatches([batch(100.)] * 4), jax.random.key(0))
    assert float(metrics["overflow"]) == 0.
    np.testing.assert_allclose(full.params["weight"], .98, atol=1e-6)
    # A trailing single microbatch has no denominator of four and really does overflow at this scale.
    partial, metrics = train_step(state(optimizer), trainer.stack_microbatches([batch(100.)]), jax.random.key(0))
    assert float(metrics["overflow"]) == 1. and float(partial.params["weight"]) == 1.


@pytest.mark.parametrize("first_value", [1., 100.])
def test_public_ema_first_attempt_copies_even_when_optimizer_overflows(monkeypatch, first_value):
    optimizer = optax.sgd(.01)
    train_step = step(monkeypatch, optimizer, first_copy=True)
    current, _ = train_step(state(optimizer), batch(first_value), jax.random.key(0))
    np.testing.assert_array_equal(current.ema_params["weight"], current.params["weight"])
    before = np.asarray(current.ema_params["weight"]).copy()
    current, _ = train_step(current, batch(1.), jax.random.key(1))
    np.testing.assert_allclose(current.ema_params["weight"], .5 * before + .5 * current.params["weight"], rtol=1e-7)
