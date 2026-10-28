"""Actual CUDA mixed-precision updates and full-state recovery for all three tasks."""

import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rfdetr_jax.models.criterion import CriterionConfig
from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec
from rfdetr_jax.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from rfdetr_jax.training.trainer import TrainSettings, create_train_state, init_params, make_train_step

pytestmark = [
    pytest.mark.skipif(
        not any(device.platform == "gpu" and "NVIDIA" in device.device_kind for device in jax.devices()),
        reason="requires an NVIDIA CUDA device; CPU checks do not qualify mixed-precision GPU training",
    ),
    pytest.mark.skipif(
        "--xla_gpu_deterministic_ops=true" not in os.environ.get("XLA_FLAGS", ""),
        reason="exact GPU resume comparison requires XLA_FLAGS=--xla_gpu_deterministic_ops=true",
    ),
]


def _snapshot(tree):
    return jax.tree.map(lambda value: np.asarray(value).copy(), tree)


def _assert_same(left, right):
    for expected, actual in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize("task", ["detection", "segmentation", "keypoints"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_cuda_mixed_training_keeps_master_state_and_recovers_from_overflow(tmp_path, task, dtype):
    spec = LWDETRSpec(
        resolution=32, positional_encoding_size=2, patch_size=16, num_windows=1,
        hidden_dim=32, sa_nheads=2, ca_nheads=2, dim_feedforward=32, dec_layers=1,
        group_detr=1, num_queries=2, num_select=2, num_classes=1, out_feature_indexes=(1,),
        gradient_checkpointing=False, compute_dtype=dtype, head_dtype="activation",
        segmentation_head=task == "segmentation", use_grouppose_keypoints=task == "keypoints",
        num_keypoints_per_class=(2,) if task == "keypoints" else (),
        dual_projector=task == "keypoints", dual_projector_kp_only=task == "keypoints",
    )
    model = LWDETR(spec)
    params = init_params(model, spec, jax.random.key(230))
    assert all(value.device.platform == "gpu" for value in jax.tree.leaves(params))
    optimizer = optax.chain(optax.clip_by_global_norm(.1), optax.adamw(1e-4))
    scaled = dtype == "float16"
    settings = TrainSettings(grad_accum_steps=2, loss_scaling=scaled, initial_loss_scale=128.,
                             ema_decay=.9, ema_tau=0)
    state = create_train_state(params, optimizer, True, loss_scaling=scaled, initial_loss_scale=128.)
    criterion = CriterionConfig(
        num_classes=spec.detector_classes, group_detr=1,
        segmentation_head=spec.segmentation_head, use_grouppose_keypoints=spec.use_grouppose_keypoints,
        num_keypoints_per_class=spec.num_keypoints_per_class, keypoint_nll_loss_coef=.1,
    )
    step = make_train_step(model, optimizer, criterion, criterion.weight_dict(1), settings)
    batch = {
        "pixel_values": jax.random.normal(jax.random.key(231), (2, 1, 3, 32, 32)),
        "labels": jnp.zeros((2, 1, 2), jnp.int32),
        "boxes": jnp.tile(jnp.array([[[[.5, .5, .4, .4], [0., 0., 0., 0.]]]]), (2, 1, 1, 1)),
        "target_mask": jnp.tile(jnp.array([[[True, False]]]), (2, 1, 1)),
    }
    if task == "segmentation":
        batch["masks"] = jnp.zeros((2, 1, 2, 32, 32)).at[:, 0, 0, 10:22, 10:22].set(1.)
    elif task == "keypoints":
        batch["keypoints"] = jnp.zeros((2, 1, 2, 2, 3)).at[:, 0, 0].set(
            jnp.array([[.4, .4, 2.], [.6, .6, 1.]]))
    before = _snapshot(state.params)
    state, metrics = step(state, batch, jax.random.key(232))
    assert all(np.isfinite(float(value)) for value in metrics.values())
    assert float(metrics.get("overflow", 0.)) == 0.
    assert any(not np.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(before), jax.tree.leaves(state.params), strict=True))
    for value in jax.tree.leaves((state.params, state.opt_state, state.ema_params)):
        if jnp.issubdtype(value.dtype, jnp.floating):
            assert value.dtype == jnp.float32

    if scaled:
        # Keep a finite forward/Hungarian match, but force backward FP16 overflow.
        state = replace(state, loss_scale=replace(state.loss_scale, scale=jnp.float32(1e30)))
        saved_params, saved_optimizer = _snapshot(state.params), _snapshot(state.opt_state)
        state, metrics = step(state, batch, jax.random.key(233))
        assert float(metrics["overflow"]) == 1. and int(state.loss_scale.skipped_steps) == 1
        assert int(state.step) == 2
        _assert_same(saved_params, state.params)
        _assert_same(saved_optimizer, state.opt_state)

    checkpoint = tmp_path / "training.npz"
    key = jax.random.key(234)
    save_training_checkpoint(checkpoint, state, key, {"precision": dtype, "task": task})
    restored, restored_key, metadata = load_training_checkpoint(checkpoint, state)
    _assert_same(state, restored)
    np.testing.assert_array_equal(jax.random.key_data(key), jax.random.key_data(restored_key))
    assert metadata["precision"] == dtype
    if scaled:
        # The saved overflowing scale was restored exactly above. Reset both
        # branches to a finite scale to exercise a successful recovery update.
        state = replace(state, loss_scale=replace(state.loss_scale, scale=jnp.float32(128.)))
        restored = replace(restored, loss_scale=replace(restored.loss_scale, scale=jnp.float32(128.)))
    partial = jax.tree.map(lambda value: value[:1], batch)
    state, metrics = step(state, partial, key)
    restored, restored_metrics = step(restored, partial, restored_key)
    assert float(metrics.get("overflow", 0.)) == 0.
    _assert_same(state, restored)
    _assert_same(metrics, restored_metrics)
