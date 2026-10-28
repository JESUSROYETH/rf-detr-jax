"""Nonzero stochastic depth must work inside compiled model and optimizer calls."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rfdetr_jax.models.criterion import CriterionConfig
from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec
from rfdetr_jax.training.trainer import TrainSettings, create_train_state, init_params, make_train_step


def test_nonzero_drop_path_compiles_inference_and_stochastic_training():
    spec = LWDETRSpec(
        resolution=32, positional_encoding_size=2, patch_size=16, num_windows=1,
        hidden_dim=32, sa_nheads=2, ca_nheads=2, dim_feedforward=32, dec_layers=1,
        group_detr=1, num_queries=2, num_classes=1, out_feature_indexes=(1, 2), drop_path=.8,
    )
    model = LWDETR(spec)
    params = init_params(model, spec, jax.random.key(0))
    pixels = jax.random.normal(jax.random.key(1), (2, 3, 32, 32))

    @jax.jit
    def predict(params, pixels):
        return model.apply({"params": params}, pixels, None, train=False)["pred_logits"]

    expected = model.apply({"params": params}, pixels, None, train=False)["pred_logits"]
    np.testing.assert_allclose(predict(params, pixels), expected, atol=1e-5, rtol=1e-5)

    @jax.jit
    def training_logits(params, pixels, key):
        return model.apply({"params": params}, pixels, None, train=True,
                           rngs={"dropout": key})["pred_logits"]

    first = training_logits(params, pixels, jax.random.key(2))
    np.testing.assert_array_equal(first, training_logits(params, pixels, jax.random.key(2)))
    assert not np.array_equal(first, training_logits(params, pixels, jax.random.key(3)))
    optimizer = optax.sgd(.001)
    config = CriterionConfig(num_classes=spec.detector_classes, group_detr=1)
    step = make_train_step(model, optimizer, config, config.weight_dict(1), TrainSettings())
    state = create_train_state(params, optimizer, use_ema=False)
    before = np.asarray(state.params["class_embed"]["weight"]).copy()
    state, metrics = step(state, {
        "pixel_values": pixels, "labels": jnp.zeros((2, 1), jnp.int32),
        "boxes": jnp.tile(jnp.array([[[.5, .5, .3, .3]]]), (2, 1, 1)),
        "target_mask": jnp.ones((2, 1), bool),
    }, jax.random.key(4))
    assert int(state.step) == 1
    assert np.isfinite(float(metrics["loss"])) and np.isfinite(float(metrics["grad_norm"]))
    assert not np.array_equal(before, state.params["class_embed"]["weight"])
