"""The jitted training step, EMA and state construction."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfdetr_jax.convert import flatten_params
from rfdetr_jax.models.criterion import CriterionConfig
from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec
from rfdetr_jax.training.optimizer import OptimizerSpec, build_optimizer
from rfdetr_jax.training.trainer import (
    TrainSettings,
    create_train_state,
    init_params,
    make_eval_step,
    make_train_step,
)

_SPEC = LWDETRSpec(
    dec_layers=2,
    num_windows=1,
    patch_size=16,
    resolution=32,
    positional_encoding_size=2,
    num_classes=3,
    group_detr=2,
    num_queries=6,
    hidden_dim=32,
    sa_nheads=2,
    ca_nheads=2,
    dim_feedforward=32,
    out_feature_indexes=(1, 2),
)


@pytest.fixture(scope="module")
def trained_once():
    """Build a tiny model, take one optimizer step, and return everything for inspection."""
    model = LWDETR(spec=_SPEC)
    params = init_params(model, _SPEC, jax.random.key(0))
    optimizer = build_optimizer(params, OptimizerSpec(), _SPEC.out_feature_indexes)
    state = create_train_state(params, optimizer, use_ema=True)
    config = CriterionConfig(num_classes=_SPEC.detector_classes, group_detr=_SPEC.group_detr)
    settings = TrainSettings(ema_decay=0.9, ema_tau=0)
    step = make_train_step(model, optimizer, config, config.weight_dict(_SPEC.dec_layers), settings)
    rng = np.random.RandomState(0)
    batch = {
        "pixel_values": jnp.asarray(rng.randn(1, 3, 32, 32).astype("float32")),
        "labels": jnp.asarray(np.array([[1, 2]], dtype="int32")),
        "boxes": jnp.asarray(np.array([[[0.4, 0.4, 0.2, 0.2], [0.7, 0.7, 0.1, 0.1]]], dtype="float32")),
        "target_mask": jnp.asarray(np.array([[True, True]])),
    }
    # The step donates its input buffers, so snapshot anything needed afterwards as numpy first.
    before = np.asarray(state.params["class_embed"]["weight"])
    new_state, metrics = step(state, batch, jax.random.key(1))
    return model, before, new_state, metrics, batch


def test_init_applies_the_upstream_head_initialisation() -> None:
    """The box head's last layer starts at zero and the class bias at the focal prior."""
    model = LWDETR(spec=_SPEC)
    params = init_params(model, _SPEC, jax.random.key(0))
    flat = {".".join(path): value for path, value in flatten_params(params).items()}
    assert np.allclose(np.asarray(flat["bbox_embed.layers_2.weight"]), 0.0)
    assert np.allclose(np.asarray(flat["bbox_embed.layers_2.bias"]), 0.0)
    assert np.allclose(np.asarray(flat["refpoint_embed"]), 0.0)
    assert np.allclose(np.asarray(flat["class_embed.bias"]), -np.log(99.0))
    assert np.allclose(np.asarray(flat["transformer.decoder.layers_0.cross_attn.attention_weights.weight"]), 0.0)


def test_encoder_heads_start_as_copies_not_aliases() -> None:
    """Each two-stage head starts equal to the decoder head but as its own buffer."""
    model = LWDETR(spec=_SPEC)
    params = init_params(model, _SPEC, jax.random.key(0))
    source = params["class_embed"]["weight"]
    for group in range(_SPEC.group_detr):
        copy = params["transformer"][f"enc_out_class_embed_{group}"]["weight"]
        assert np.allclose(np.asarray(copy), np.asarray(source))
        assert copy is not source


def test_step_produces_finite_losses(trained_once) -> None:
    """One step yields finite losses and a finite gradient norm."""
    _, _, _, metrics, _ = trained_once
    assert np.isfinite(float(metrics["loss"]))
    assert np.isfinite(float(metrics["grad_norm"]))
    assert {"loss_ce", "loss_bbox", "loss_giou", "loss_ce_enc"} <= set(metrics)


def test_step_advances_parameters_and_counter(trained_once) -> None:
    """Parameters move and the step counter increments."""
    _, before, new_state, _, _ = trained_once
    assert int(new_state.step) == 1
    assert not np.allclose(before, np.asarray(new_state.params["class_embed"]["weight"]))


def test_ema_lags_the_live_parameters(trained_once) -> None:
    """The EMA tree moves toward, but not all the way to, the updated parameters."""
    _, start, new_state, _, _ = trained_once
    live = np.asarray(new_state.params["class_embed"]["weight"])
    ema = np.asarray(new_state.ema_params["class_embed"]["weight"])
    assert np.allclose(ema, 0.9 * start + 0.1 * live, atol=1e-6)


def test_eval_step_uses_a_single_query_group(trained_once) -> None:
    """Inference collapses the query groups back to ``num_queries``."""
    model, _, new_state, _, batch = trained_once
    logits, boxes = make_eval_step(model)(new_state.params, batch["pixel_values"])
    assert logits.shape == (1, _SPEC.num_queries, _SPEC.detector_classes)
    assert boxes.shape == (1, _SPEC.num_queries, 4)


def test_state_without_ema_is_supported() -> None:
    """Disabling EMA leaves the slot empty rather than allocating a second tree."""
    model = LWDETR(spec=_SPEC)
    params = init_params(model, _SPEC, jax.random.key(0))
    optimizer = build_optimizer(params, OptimizerSpec(), _SPEC.out_feature_indexes)
    assert create_train_state(params, optimizer, use_ema=False).ema_params is None


def test_data_parallel_replicates_state_and_shards_batches() -> None:
    """The helper must hand back the same values plus a sharding a batch can be placed on."""
    from rfdetr_jax.training.trainer import data_parallel

    spec = LWDETRSpec(
        resolution=64,
        positional_encoding_size=4,
        num_classes=2,
        dec_layers=1,
        group_detr=1,
        num_queries=6,
        num_select=6,
    )
    params = init_params(LWDETR(spec=spec), spec, jax.random.key(0))
    optimizer = build_optimizer(params, OptimizerSpec(), spec.out_feature_indexes)
    state = create_train_state(params, optimizer, use_ema=False)

    devices = jax.devices()
    replicated, batch_sharding = data_parallel(state, devices)
    for before, after in zip(jax.tree.leaves(state.params), jax.tree.leaves(replicated.params), strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))

    images = np.zeros((len(devices), 3, 64, 64), np.float32)
    placed = jax.device_put(images, batch_sharding)
    assert placed.shape == images.shape


@pytest.mark.parametrize("task", ["segmentation_head", "use_grouppose_keypoints"])
def test_builtin_trainer_rejects_mismatched_task_configuration(task: str) -> None:
    """An enabled task must not appear to train while its supervision is discarded."""
    from dataclasses import replace

    model = LWDETR(spec=replace(_SPEC, **{task: True}))
    config = CriterionConfig(num_classes=_SPEC.detector_classes)
    with pytest.raises(ValueError, match="settings must agree"):
        make_train_step(model, None, config, config.weight_dict(_SPEC.dec_layers), TrainSettings())


@pytest.mark.parametrize("task", ["segmentation", "keypoints", "keypoints_compact"])
def test_task_training_updates_head_and_ignores_padding(task: str) -> None:
    """Real task heads receive supervision through the compiled public training step."""
    from dataclasses import replace

    import optax

    segmentation = task == "segmentation"
    spec = replace(
        _SPEC, dec_layers=1, group_detr=1, num_queries=2, num_select=2, num_classes=1,
        out_feature_indexes=(1,), segmentation_head=segmentation,
        use_grouppose_keypoints=not segmentation,
        num_keypoints_per_class=() if segmentation else ((2,) if task == "keypoints_compact" else (0, 2)),
    )
    model = LWDETR(spec=spec)
    params = init_params(model, spec, jax.random.key(90))
    optimizer = optax.sgd(0.001)
    config = CriterionConfig(
        num_classes=spec.detector_classes, group_detr=1, segmentation_head=segmentation,
        use_grouppose_keypoints=not segmentation, num_keypoints_per_class=spec.num_keypoints_per_class,
        keypoint_l1_loss_coef=1., keypoint_findable_loss_coef=1.,
        keypoint_visible_loss_coef=1., keypoint_nll_loss_coef=0.1,
    )
    step = make_train_step(model, optimizer, config, config.weight_dict(spec.dec_layers), TrainSettings())
    batch = {
        "pixel_values": jnp.ones((2, 3, 32, 32)),
        "labels": jnp.array([[0 if task == "keypoints_compact" else 1, 0], [0, 0]], jnp.int32),
        "boxes": jnp.array([[[.5, .5, .4, .4], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0]]]),
        "target_mask": jnp.array([[True, False], [False, False]]),
    }
    if segmentation:
        batch["masks"] = jnp.zeros((2, 2, 32, 32)).at[0, 0, 10:22, 10:22].set(1.)
        prefix, loss_name = "segmentation_head", "loss_mask_ce"
    else:
        batch["keypoints"] = jnp.zeros((2, 2, 2, 3)).at[0, 0].set(jnp.array([[.4, .4, 2.], [.6, .6, 1.]]))
        prefix, loss_name = "keypoint", "loss_keypoints_l1"
    before = {path: np.asarray(value).copy() for path, value in flatten_params(params).items()}
    state = create_train_state(params, optimizer, use_ema=False)
    state, metrics = step(state, batch, jax.random.key(91))
    assert all(np.isfinite(float(value)) for value in metrics.values())
    assert float(metrics[loss_name]) > 0
    changed = [path for path, value in flatten_params(state.params).items()
               if prefix in ".".join(path) and not np.array_equal(before[path], np.asarray(value))]
    assert changed, f"{task} head received no parameter update"
    # The second image and trailing target slot above are empty; now the whole batch has no objects.
    batch["target_mask"] = jnp.zeros_like(batch["target_mask"])
    state, empty_metrics = step(state, batch, jax.random.key(92))
    assert all(np.isfinite(float(value)) for value in empty_metrics.values())
    assert float(empty_metrics[loss_name]) == 0.0


@pytest.mark.parametrize("task", ["segmentation", "keypoints"])
def test_task_training_requires_supervision(task: str) -> None:
    """A task cannot silently turn into box-only training when its target tensor is absent."""
    from dataclasses import replace

    import optax

    segmentation = task == "segmentation"
    spec = replace(_SPEC, segmentation_head=segmentation, use_grouppose_keypoints=not segmentation,
                   num_keypoints_per_class=() if segmentation else (0, 2))
    config = CriterionConfig(segmentation_head=segmentation, use_grouppose_keypoints=not segmentation,
                             num_keypoints_per_class=spec.num_keypoints_per_class, keypoint_l1_loss_coef=1.)
    optimizer = optax.sgd(.01)
    step = make_train_step(LWDETR(spec=spec), optimizer, config, config.weight_dict(1), TrainSettings())
    state = create_train_state({}, optimizer, use_ema=False)
    with pytest.raises(ValueError, match="training requires batch"):
        step(state, {}, jax.random.key(0))


def test_fresh_keypoint_initialisation_matches_upstream_resets() -> None:
    """GroupPose zeroes the keypoint head's last layer, copies it into the encoder heads, and zeroes
    adaLN and attention output biases, as the upstream constructors do."""
    spec = replace(_SPEC, use_grouppose_keypoints=True, num_keypoints_per_class=(2,), dual_projector=True,
                   dual_projector_kp_only=True, group_detr=2, grouppose_keypoint_dim_downscale=2)
    model = LWDETR(spec=spec)
    params = init_params(model, spec, jax.random.key(1))
    # The keypoint cross-attention has ca_nheads // downscale heads: its offsets bias keeps that size.
    template = flatten_params(model.init(jax.random.key(0), jnp.zeros((1, 3, 32, 32)), None, train=True)["params"])
    for path, value in flatten_params(params).items():
        assert value.shape == template[path].shape, path
    flat = {".".join(path): np.asarray(value) for path, value in flatten_params(params).items()}
    assert np.allclose(flat["keypoint_embed.layers_2.weight"], 0.0)
    assert np.allclose(flat["keypoint_embed.layers_2.bias"], 0.0)
    for group in range(2):
        for layer in range(3):
            assert np.array_equal(flat[f"transformer.enc_out_keypoint_embed_{group}.layers_{layer}.weight"],
                                  flat[f"keypoint_embed.layers_{layer}.weight"])
    adaln = [name for name in flat if name.endswith("adaLN_modulation_2.bias")]
    out_proj = [name for name in flat if name.startswith("transformer") and name.endswith("out_proj.bias")
                and name[:-len("out_proj.bias")] + "in_proj_weight" in flat]
    plain_out_proj = [name for name in flat if name.endswith("out_proj.bias") and name not in out_proj]
    backbone_bias = [name for name in flat if name.startswith("backbone_0.encoder") and name.endswith(".bias")
                     and flat.get(name[:-5] + ".weight", np.zeros(())).ndim in (2, 4)]
    assert adaln and out_proj and plain_out_proj and backbone_bias
    assert all(np.allclose(flat[name], 0.0) for name in adaln + out_proj + backbone_bias)
    assert all(not np.allclose(flat[name], 0.0) for name in plain_out_proj)  # plain nn.Linear keeps its init
