"""Public keypoint accumulation follows the actual published manual-optimization helpers."""
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rfdetr_jax.models.criterion import CriterionConfig, Targets, compute_losses, num_boxes_for_targets
from rfdetr_jax.training import trainer


class ScalarKeypointModel:
    spec = SimpleNamespace(segmentation_head=False, use_grouppose_keypoints=True, num_keypoints_per_class=(1,))

    def apply(self, variables, pixels, mask, **kwargs):
        return {"prediction": variables["params"]["gain"] * pixels.mean()}


def microbatch(count, value):
    return {"pixel_values": jnp.full((1, 1, 1, 1), value), "labels": jnp.zeros((1, 3), jnp.int32),
            "boxes": jnp.full((1, 3, 4), .2), "target_mask": jnp.arange(3)[None] < count,
            "keypoints": jnp.ones((1, 3, 1, 3))}


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("group,sum_groups", [(1, False), (3, False), (3, True)])
def test_matches_published_manual_accumulation_with_unequal_counts_empty_and_partial(
        monkeypatch, scaled, group, sum_groups):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    import importlib.metadata
    assert importlib.metadata.version("rfdetr") == "1.11.2"
    from rfdetr.models.criterion import SetCriterion
    from rfdetr.training.module_model import RFDETRModelModule

    config = CriterionConfig(group_detr=group, sum_group_losses=sum_groups, use_grouppose_keypoints=True,
                             num_keypoints_per_class=(1,), keypoint_l1_loss_coef=1.)
    weights = {"loss_keypoints_l1": 1.}

    def losses(outputs, targets, config, *, num_boxes=None, **kwargs):
        assert num_boxes is not None, "manual accumulation must request raw criterion numerators"
        prediction = outputs["prediction"]
        raw = ((prediction - .2) ** 2 * targets.mask.sum() + .1 * prediction ** 2) * group
        return {"loss_keypoints_l1": raw / num_boxes, "cardinality_error": jnp.float32(7.)}

    monkeypatch.setattr(trainer, "compute_losses", losses)
    optimizer = optax.sgd(.01)
    current = trainer.create_train_state({"gain": jnp.float32(1.)}, optimizer, False,
                                         loss_scaling=scaled, initial_loss_scale=8.)
    train_step = trainer.make_train_step(ScalarKeypointModel(), optimizer, config, weights,
                                        trainer.TrainSettings(grad_accum_steps=2, loss_scaling=scaled,
                                            keypoint_box_normalized_accumulation=True))
    gain = torch.nn.Parameter(torch.tensor(1.))
    reference = SimpleNamespace(parameters=lambda: iter([gain]), _accumulated_box_normalizer=None)
    reference._rescale_accumulated_gradients = partial(RFDETRModelModule._rescale_accumulated_gradients, reference)
    criterion = SimpleNamespace(training=True, group_detr=group, sum_group_losses=sum_groups,
                                _output_device=lambda outputs: torch.device("cpu"))
    optimizer_torch = torch.optim.SGD([gain], lr=.01)
    scaler = torch.amp.GradScaler("cpu", init_scale=8., enabled=scaled)
    groups = [[(1, 1.), (3, 2.)], [(0, 1.5), (2, .5)], [(1, 2.5)]]
    for values in groups:
        optimizer_torch.zero_grad()
        reference._accumulated_box_normalizer = None
        logged = []
        for count, value in values:
            prediction = gain * value
            raw = ((prediction - .2) ** 2 * count + .1 * prediction ** 2) * group
            denominator = SetCriterion.num_boxes_for_targets(
                criterion, {"prediction": prediction}, [{"labels": torch.zeros(count, dtype=torch.int64)}])
            logged.append(float((raw / denominator).detach()))
            backward_loss = RFDETRModelModule._scale_loss_for_accumulation(reference, raw, denominator)
            scaler.scale(backward_loss).backward()
        scaler.step(optimizer_torch)
        scaler.update()
        current, metrics = train_step(current, trainer.stack_microbatches(
            [microbatch(count, value) for count, value in values]), jax.random.key(0))
        np.testing.assert_allclose(current.params["gain"], gain.detach(), rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(metrics["loss"], np.mean(logged), rtol=2e-6)
        assert float(metrics["cardinality_error"]) == 7.
    assert int(current.step) == 3


def test_raw_numerator_override_applies_to_actual_keypoint_criterion():
    config = CriterionConfig(num_classes=1, group_detr=1, use_grouppose_keypoints=True,
                             num_keypoints_per_class=(1,), keypoint_l1_loss_coef=1.)
    outputs = {"pred_logits": jnp.zeros((1, 3, 1)), "pred_boxes": jnp.full((1, 3, 4), .4),
               "pred_keypoints": jnp.zeros((1, 3, 1, 8))}
    targets = Targets(jnp.zeros((1, 3), jnp.int32), jnp.full((1, 3, 4), .3),
                      jnp.array([[True, True, False]]), keypoints=jnp.ones((1, 3, 1, 3)))
    ordinary = compute_losses(outputs, targets, config)
    raw = compute_losses(outputs, targets, config, num_boxes=jnp.float32(1.))
    denominator = num_boxes_for_targets(targets, 1, False)
    assert float(denominator) == 2.
    for name, value in ordinary.items():
        np.testing.assert_allclose(raw[name], value * denominator, rtol=1e-6)


@pytest.mark.parametrize("scaled", [False, True])
def test_public_automatic_partial_group_matches_actual_lightning_trainer(monkeypatch, tmp_path, scaled):
    torch = pytest.importorskip("torch")
    pl = pytest.importorskip("pytorch_lightning")
    from pytorch_lightning.plugins.precision import MixedPrecision

    class Reference(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(1.))

        def training_step(self, batch, batch_idx):
            return self.gain * batch.mean()

        def configure_optimizers(self):
            return torch.optim.SGD([self.gain], lr=.1)

    reference = Reference()
    # Lightning's CPU autocast backend supports this real GradScaler experiment without a GPU.
    plugins = [MixedPrecision("16-mixed", "cpu", scaler=torch.amp.GradScaler("cpu", init_scale=8.))] if scaled else None
    trainer_torch = pl.Trainer(accelerator="cpu", max_epochs=1, accumulate_grad_batches=2,
                               logger=False, enable_checkpointing=False, enable_progress_bar=False,
                               enable_model_summary=False, default_root_dir=tmp_path, plugins=plugins)
    trainer_torch.fit(reference, torch.utils.data.DataLoader(torch.tensor([2., 4., 6.]), batch_size=1))
    assert trainer_torch.global_step == 2
    np.testing.assert_allclose(reference.gain.detach(), .4, atol=1e-7)

    class Model(ScalarKeypointModel):
        spec = SimpleNamespace(segmentation_head=False, use_grouppose_keypoints=False)

    monkeypatch.setattr(trainer, "compute_losses", lambda outputs, *args, **kwargs: {
        "loss_bbox": outputs["prediction"]})
    optimizer = optax.sgd(.1)
    current = trainer.create_train_state({"gain": jnp.float32(1.)}, optimizer, False,
                                         loss_scaling=scaled, initial_loss_scale=8.)
    train_step = trainer.make_train_step(Model(), optimizer, CriterionConfig(), {"loss_bbox": 1.},
        trainer.TrainSettings(grad_accum_steps=2, loss_scaling=scaled, automatic_accumulation_normalization=True))
    for values in ((2., 4.), (6.,)):
        current, metrics = train_step(current, trainer.stack_microbatches([microbatch(1, value) for value in values]),
                                      jax.random.key(0))
    np.testing.assert_allclose(current.params["gain"], reference.gain.detach(), atol=1e-7)
    # Logging remains the actual partial batch loss; only backward uses the configured divisor.
    np.testing.assert_allclose(metrics["loss"], 4.2, atol=1e-6)
