"""Accumulation semantics and uninterrupted-versus-restored optimizer state."""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rfdetr_jax.models.criterion import CriterionConfig
from rfdetr_jax.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from rfdetr_jax.training.trainer import TrainSettings, create_train_state, make_train_step, stack_microbatches


class TinyDetector:
    spec = SimpleNamespace(segmentation_head=False, use_grouppose_keypoints=False)

    def apply(self, variables, pixels, mask, **kwargs):
        del mask, kwargs
        params = variables["params"]
        batch = pixels.shape[0]
        logits = params["logits"][None, None] * pixels[:, 0, 0, 0, None, None]
        return {"pred_logits": jnp.broadcast_to(logits, (batch, 2, 2)),
                "pred_boxes": jnp.broadcast_to(jax.nn.sigmoid(params["boxes"]), (batch, 2, 4))}


def parameters():
    return {"logits": jnp.array([.2, -.3]), "boxes": jnp.array([.1, -.2, -.4, -.5])}


def batch(value):
    return {"pixel_values": jnp.full((1, 3, 4, 4), value), "labels": jnp.array([[1]]),
            "boxes": jnp.array([[[.4, .6, .3, .2]]]), "target_mask": jnp.ones((1, 1), bool)}


def step_for(optimizer, accumulation=1):
    config = CriterionConfig(num_classes=2, group_detr=1, aux_loss=False, two_stage=False)
    return make_train_step(TinyDetector(), optimizer, config, config.weight_dict(1),
                           TrainSettings(grad_accum_steps=accumulation, ema_decay=.9, ema_tau=0))


def assert_trees_equal(left, right):
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-6)


def test_accumulation_matches_combined_batch_and_flushes_partial_group():
    optimizer = optax.sgd(.01)
    accumulated = create_train_state(parameters(), optimizer, True)
    combined = create_train_state(parameters(), optimizer, True)
    accumulate_step, combined_step = step_for(optimizer, 2), step_for(optimizer)
    for values in ((1., 2.), (3.,)):
        inputs = [batch(value) for value in values]
        stacked = stack_microbatches(inputs)
        merged = jax.tree.map(lambda value: value.reshape((-1, *value.shape[2:])), stacked)
        accumulated, metrics = accumulate_step(accumulated, stacked, jax.random.key(0))
        combined, reference = combined_step(combined, merged, jax.random.key(0))
        assert_trees_equal(accumulated, combined)
        np.testing.assert_allclose(metrics["loss"], reference["loss"], rtol=1e-6)
    assert int(accumulated.step) == 2


def test_checkpoint_restores_optimizer_ema_step_and_rng_exactly(tmp_path):
    optimizer = optax.adamw(optax.cosine_decay_schedule(.01, 4), weight_decay=.1)
    state = create_train_state(parameters(), optimizer, True)
    step = step_for(optimizer, 2)
    key = jax.random.key(42)
    checkpoint = tmp_path / "checkpoint_last.npz"
    for index in range(4):
        key, subkey = jax.random.split(key)
        state, _ = step(state, stack_microbatches([batch(float(index + 1))]), subkey)
        if index == 1:
            save_training_checkpoint(checkpoint, state, key, {"completed_epochs": 2})
    template = create_train_state(parameters(), optimizer, True)
    resumed, resumed_key, metadata = load_training_checkpoint(checkpoint, template)
    assert metadata["completed_epochs"] == 2
    assert int(resumed.step) == 2
    for index in range(2, 4):
        resumed_key, subkey = jax.random.split(resumed_key)
        resumed, _ = step(resumed, stack_microbatches([batch(float(index + 1))]), subkey)
    assert_trees_equal(state, resumed)
    np.testing.assert_array_equal(jax.random.key_data(key), jax.random.key_data(resumed_key))


def test_reusable_run_resume_preserves_data_order_schedule_and_partial_groups(tmp_path, monkeypatch):
    import json

    from PIL import Image

    from rfdetr_jax.models.lwdetr import LWDETRSpec
    from rfdetr_jax.training import run

    root = tmp_path / "dataset"
    for split in ("train", "val"):
        directory = root / split
        directory.mkdir(parents=True)
        images, annotations = [], []
        for index in range(3 if split == "train" else 1):
            name = f"{index}.png"
            Image.new("RGB", (32, 32), (30 + index * 50,) * 3).save(directory / name)
            images.append({"id": index + 1, "file_name": name, "height": 32, "width": 32})
            annotations.append({"id": index + 1, "image_id": index + 1, "category_id": 1,
                                "bbox": [8, 8, 12, 12], "area": 144, "iscrowd": 0})
        (directory / "_annotations.coco.json").write_text(json.dumps({
            "images": images, "annotations": annotations, "categories": [{"id": 1, "name": "object"}],
        }))
    spec = LWDETRSpec(resolution=32, positional_encoding_size=2, num_windows=1, num_classes=1,
                      num_queries=2, num_select=2, group_detr=1, dec_layers=1)

    def model_for(spec):
        model = TinyDetector()
        model.spec = spec
        return model

    monkeypatch.setattr(run, "LWDETR", model_for)
    monkeypatch.setattr(run, "init_params", lambda *args: parameters())
    common = ["--dataset-dir", str(root), "--epochs", "2", "--batch-size", "1", "--grad-accum-steps", "2",
              "--lr-scheduler", "cosine", "--seed", "5", "--data-seed", "11", "--no-precompile",
              "--no-compile-cache", "--eval-backend", "pycocotools"]
    common += ["--eval-raw"]
    evaluations = []

    def evaluate_records(records, image_sizes):
        assert isinstance(records, list)
        assert image_sizes == {1: (32, 32)}
        evaluations.append(records)
        return {"mAP": .314, "F1": .2}

    monkeypatch.setattr(run, "evaluate_detections", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("custom validation must bypass the raw AP evaluator")))
    criterion = CriterionConfig(num_classes=2, group_detr=1, bbox_loss_coef=7.5)
    full = run.run_training(common + ["--output-dir", str(tmp_path / "full")], model_spec=spec,
                            initial_params=parameters(), category_to_label={1: 0},
                            metadata={"custom": "retained"}, criterion_config=criterion,
                            evaluation_record_evaluator=evaluate_records)
    partial = run.run_training(common + ["--output-dir", str(tmp_path / "partial")], model_spec=spec,
                               initial_params=parameters(), category_to_label={1: 0},
                               metadata={"custom": "retained"}, on_epoch_end=lambda record, state: True,
                               criterion_config=criterion, evaluation_record_evaluator=evaluate_records)
    resumed = run.run_training(["--resume", partial.paths["checkpoint"], "--output-dir", str(tmp_path / "resumed"),
                                "--no-precompile", "--no-compile-cache", "--eval-backend", "pycocotools", "--eval-raw"],
                               evaluation_record_evaluator=evaluate_records)
    assert len(evaluations) == 8
    assert all(row["val"]["mAP"] == row["val_ema"]["mAP"] == .314 for row in resumed.log["epochs"])
    assert_trees_equal(full.state, resumed.state)
    assert int(full.state.step) == int(resumed.state.step) == 4
    assert [epoch["micro_batches"] for epoch in resumed.log["epochs"]] == [3, 3]
    from rfdetr_jax.training.checkpoint import checkpoint_metadata
    assert checkpoint_metadata(resumed.paths["checkpoint"])["criterion_config"]["bbox_loss_coef"] == 7.5
    assert resumed.metadata["custom"] == "retained"
    assert resumed.metadata["category_to_label"] == {1: 0}
    np.testing.assert_allclose([epoch["losses"]["loss"] for epoch in full.log["epochs"]],
                               [epoch["losses"]["loss"] for epoch in resumed.log["epochs"]], rtol=1e-6)


def test_closed_prefetch_closes_source():
    import threading

    from rfdetr_jax.datasets.coco import prefetch

    closed = threading.Event()

    def source():
        try:
            while True:
                yield 1
        finally:
            closed.set()

    batches = prefetch(source(), depth=1)
    assert next(batches) == 1
    batches.close()
    assert closed.wait(2), "early stop left the data producer alive"


def test_standalone_pth_restores_full_state_without_native_file(tmp_path):
    import pytest
    torch = pytest.importorskip("torch")
    from rfdetr_jax.checkpoints import embed_native_resume

    optimizer = optax.adamw(.01)
    state = create_train_state(parameters(), optimizer, True)
    state, _ = step_for(optimizer)(state, batch(2.), jax.random.key(1))
    key = jax.random.key(13)
    native = tmp_path / "checkpoint_last.npz"
    save_training_checkpoint(native, state, key, {"completed_epochs": 1})
    public = tmp_path / "checkpoint.pth"
    torch.save({"model": {"weight": torch.tensor([1.])}}, public)
    embed_native_resume(public, native)
    standalone = tmp_path / "copied.pth"
    standalone.write_bytes(public.read_bytes())
    native.unlink()
    public.unlink()
    restored, restored_key, metadata = load_training_checkpoint(
        standalone, create_train_state(parameters(), optimizer, True))
    assert_trees_equal(restored, state)
    np.testing.assert_array_equal(jax.random.key_data(restored_key), jax.random.key_data(key))
    assert metadata["completed_epochs"] == 1


def test_frozen_encoder_excludes_gradients_from_clipping_and_weight_decay():
    class Detector(TinyDetector):
        def __init__(self, reference=False):
            self.reference = reference

        def apply(self, variables, pixels, mask, **kwargs):
            params = variables["params"]
            encoder = params["backbone_0"]["encoder"]["weight"]
            if self.reference:
                encoder = jax.lax.stop_gradient(encoder)
            combined = {**params, "logits": params["logits"] + 1000 * encoder}
            return super().apply({"params": combined}, pixels, mask, **kwargs)

    params = {**parameters(), "backbone_0": {"encoder": {"weight": jnp.array(.001)}}}
    optimizer = optax.chain(optax.clip_by_global_norm(.1), optax.adamw(.01, weight_decay=.1))
    config = CriterionConfig(num_classes=2, group_detr=1)
    frozen = make_train_step(Detector(), optimizer, config, config.weight_dict(1),
                             TrainSettings(freeze_encoder=True))
    reference = make_train_step(Detector(reference=True), optimizer, config, config.weight_dict(1), TrainSettings())
    state, _ = frozen(create_train_state(params, optimizer, True), batch(1.), jax.random.key(0))
    expected, _ = reference(create_train_state(params, optimizer, True), batch(1.), jax.random.key(0))
    np.testing.assert_array_equal(state.params["backbone_0"]["encoder"]["weight"], np.float32(.001))
    np.testing.assert_allclose(state.params["logits"], expected.params["logits"], rtol=1e-6)
    np.testing.assert_allclose(state.params["boxes"], expected.params["boxes"], rtol=1e-6)
