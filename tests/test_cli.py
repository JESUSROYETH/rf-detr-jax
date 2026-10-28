"""User-facing command regressions: safe loading, validation, and reusable training output."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from contextlib import nullcontext
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from rfdetr_jax import cli
from rfdetr_jax.convert import flatten_params, load_params_npz, save_params_npz
from rfdetr_jax.training import run
from rfdetr_jax.training.run_config import RunConfig


@pytest.mark.parametrize("trusted", [False, True])
def test_converter_prefers_safe_loading_even_with_trust_opt_in(monkeypatch, tmp_path, trusted):
    calls = []
    tensor = SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.ones(2))))

    def load(path, **kwargs):
        calls.append(kwargs)
        return {"model": {"class_embed.bias": tensor}}

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=load, __version__="2.10.0",
        serialization=SimpleNamespace(safe_globals=lambda values: nullcontext())))
    output = tmp_path / "params.npz"
    monkeypatch.setattr(sys, "argv", ["convert", "input.pth", "--out", str(output)]
                        + (["--trust-checkpoint"] if trusted else []))
    cli.convert()
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    np.testing.assert_array_equal(load_params_npz(output)["class_embed"]["bias"], np.ones(2))


def test_rejected_checkpoint_is_not_retried_unsafely(monkeypatch, tmp_path, capsys):
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        raise pickle.UnpicklingError("unsupported global")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=load, __version__="2.10.0",
        serialization=SimpleNamespace(safe_globals=lambda values: nullcontext())))
    output = tmp_path / "params.npz"
    monkeypatch.setattr(sys, "argv", ["convert", "input.pth", "--out", str(output)])
    with pytest.raises(SystemExit) as exc:
        cli.convert()
    assert exc.value.code == 2
    assert len(calls) == 1 and calls[0]["weights_only"] is True
    assert not output.exists()
    assert "--trust-checkpoint" in capsys.readouterr().err


@pytest.mark.parametrize("namespace_type", [argparse.Namespace, SimpleNamespace])
def test_converter_accepts_inert_namespace_metadata_safely(monkeypatch, tmp_path, namespace_type):
    """Official Nano checkpoints include argparse.Namespace alongside their tensors."""
    import torch

    checkpoint = tmp_path / "nano.pth"
    output = tmp_path / "nano.npz"
    expected = torch.tensor([1., -2.])
    torch.save({"model": {"class_embed.bias": expected}, "args": namespace_type(num_classes=90)}, checkpoint)
    safe_globals_before = set(torch.serialization.get_safe_globals())
    monkeypatch.setattr(sys, "argv", ["convert", str(checkpoint), "--out", str(output)])
    cli.convert()
    np.testing.assert_array_equal(load_params_npz(output)["class_embed"]["bias"], expected.numpy())
    assert set(torch.serialization.get_safe_globals()) == safe_globals_before


class _UntrustedCheckpointMetadata:
    """Deliberately outside the inert namespace allowlist."""


@pytest.mark.parametrize("trusted", [False, True])
def test_converter_unknown_globals_require_explicit_trust(monkeypatch, tmp_path, capsys, trusted):
    import torch

    checkpoint = tmp_path / "custom.pth"
    output = tmp_path / "custom.npz"
    torch.save({"model": {"class_embed.bias": torch.tensor([3., 4.])},
                "metadata": _UntrustedCheckpointMetadata()}, checkpoint)
    original_load = torch.load
    calls = []

    def observe_load(*args, **kwargs):
        calls.append(kwargs["weights_only"])
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", observe_load)
    monkeypatch.setattr(sys, "argv", ["convert", str(checkpoint), "--out", str(output)]
                        + (["--trust-checkpoint"] if trusted else []))
    if trusted:
        with pytest.warns(UserWarning, match="permits arbitrary Python code"):
            cli.convert()
        assert calls == [True, False]
        np.testing.assert_array_equal(load_params_npz(output)["class_embed"]["bias"], [3., 4.])
    else:
        with pytest.raises(SystemExit) as exc:
            cli.convert()
        assert exc.value.code == 2 and calls == [True]
        assert not output.exists()
        assert "--trust-checkpoint" in capsys.readouterr().err


@pytest.mark.parametrize(("flags", "message"), [
    (["--epochs", "0"], "--epochs must be positive"),
    (["--batch-size", "0"], "--batch-size must be positive"),
    (["--eval-every", "0"], "--eval-every must be positive"),
    (["--init-weights", "/missing/checkpoint.npz"], "--init-weights file does not exist"),
    (["--target-buckets", "-1"], "--target-buckets must contain positive integers"),
    (["--data-parallel"], "--data-parallel requires --matcher auction"),
    (["--save-params", "x.npz", "--out", "x.npz.json"], "must be different files"),
])
def test_training_rejects_invalid_configuration_early(monkeypatch, capsys, flags, message):
    monkeypatch.setattr(sys, "argv", ["train", *flags])
    with pytest.raises(SystemExit) as exc:
        cli.train()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def write_dataset(tmp_path, *, val_name="object", category_id=1, task="detection"):
    for split in ("train", "val"):
        directory = tmp_path / split
        directory.mkdir()
        Image.new("RGB", (32, 32), "gray").save(directory / "image.png")
        annotations = {
            "images": [{"id": 1, "file_name": "image.png", "width": 32, "height": 32}],
            "categories": [{"id": category_id, "name": val_name if split == "val" else "object"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": category_id,
                             "bbox": [8, 8, 12, 12], "area": 144, "iscrowd": 0}],
        }
        if task == "segmentation":
            annotations["annotations"][0]["segmentation"] = [[8, 8, 20, 8, 20, 20, 8, 20]]
        elif task == "keypoints":
            annotations["categories"][0]["keypoints"] = ["left", "right"]
            annotations["categories"][0]["skeleton"] = [[1, 2]]
            annotations["annotations"][0].update(keypoints=[10, 12, 2, 18, 12, 1], num_keypoints=2)
        (directory / "_annotations.coco.json").write_text(json.dumps(annotations))


@pytest.mark.parametrize(("val_name", "category_id", "batch_size", "message"), [
    ("different", 1, 1, "category ids and names must match"),
    ("object", 9, 1, "largest raw COCO category id"),
    ("object", 1, 2, "at least --batch-size images"),
])
def test_training_rejects_invalid_dataset(monkeypatch, tmp_path, capsys, val_name, category_id, batch_size, message):
    write_dataset(tmp_path, val_name=val_name, category_id=category_id)
    monkeypatch.setattr(sys, "argv", ["train", "--dataset-dir", str(tmp_path), "--batch-size", str(batch_size)])
    with pytest.raises(SystemExit) as exc:
        cli.train()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_training_exports_reloadable_inference_weights(monkeypatch, tmp_path):
    """One actual CPU training step writes weights and enough metadata to evaluate them."""
    write_dataset(tmp_path)
    original_spec_for = run.spec_for

    def small_run(**kwargs):
        return replace(RunConfig(**kwargs), resolution=32, num_classes=1, max_targets=2,
                       num_windows=1, dec_layers=1, group_detr=1, num_queries=2, num_select=2,
                       hidden_dim=32, sa_nheads=2, ca_nheads=2, out_feature_indexes=(1, 2),
                       gradient_checkpointing=False)

    monkeypatch.setattr(run, "RunConfig", small_run)
    monkeypatch.setattr(run, "spec_for", lambda config: replace(original_spec_for(config), dim_feedforward=32))
    output = tmp_path / "run.json"
    monkeypatch.setattr(sys, "argv", ["train", "--dataset-dir", str(tmp_path), "--batch-size", "1",
                                      "--epochs", "1", "--no-precompile", "--no-compile-cache",
                                      "--out", str(output), "--eval-backend", "pycocotools"])
    cli.train()
    log = json.loads(output.read_text())
    assert log["epochs"][0]["steps"] == 1
    assert np.isfinite(log["epochs"][0]["losses"]["loss"])
    params = load_params_npz(log["saved_params"])
    assert params["class_embed"]["weight"].shape[0] == 2
    metadata = json.loads((tmp_path / "run.params.npz.json").read_text())
    assert metadata["weights"] == "ema" and metadata["step"] == 1
    assert metadata["model_spec"]["resolution"] == 32
    monkeypatch.setattr(sys, "argv", ["eval", "--params", log["saved_params"],
                                      "--model-config", log["model_metadata"], "--resolution", "64",
                                      "--images", str(tmp_path / "val"),
                                      "--annotations", str(tmp_path / "val" / "_annotations.coco.json"),
                                      "--batch-size", "1"])
    cli.evaluate()
    from rfdetr_jax.inference import RFDETR

    detector = RFDETR.from_checkpoint(log["saved_params"])
    result = detector.predict(tmp_path / "val" / "image.png", threshold=0)
    assert detector.task == "detection" and result["boxes"].shape[1] == 4


def test_converter_rejects_vulnerable_torch_before_loading(monkeypatch, tmp_path, capsys):
    def load(*args, **kwargs):
        pytest.fail("must reject unsupported PyTorch before reading a checkpoint")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=load, __version__="2.9.1+cpu"))
    monkeypatch.setattr(sys, "argv", ["convert", "input.pth", "--out", str(tmp_path / "params.npz")])
    with pytest.raises(SystemExit) as exc:
        cli.convert()
    assert exc.value.code == 2
    assert "PyTorch >=2.10.0" in capsys.readouterr().err


def test_converted_parameters_keep_their_paths_when_reloaded(tmp_path):
    params = {"backbone_0": {"encoder": {"encoder": {"layer_0": {"weight": np.ones((2, 3))}}}}}
    path = tmp_path / "params.npz"
    save_params_npz(path, params)
    loaded = run.load_initial_params(path, params)
    np.testing.assert_array_equal(loaded["backbone_0"]["encoder"]["encoder"]["layer_0"]["weight"], np.ones((2, 3)))
    with pytest.raises(ValueError, match="initial parameter mismatch"):
        run.load_initial_params(path, {"different": np.ones(2)})


@pytest.mark.parametrize(("task", "schema"), [
    ("segmentation", ()), ("keypoints", (2,)), ("keypoints", (2, 3)),
])
def test_task_training_export_reload_and_evaluation(monkeypatch, tmp_path, capsys, task, schema):
    from rfdetr_jax.configuration import load_model_spec, validate_parameter_tree
    from rfdetr_jax.models.lwdetr import LWDETRSpec

    write_dataset(tmp_path, category_id=1, task=task)
    model_spec = LWDETRSpec(
        resolution=32, positional_encoding_size=2, patch_size=16, num_windows=1,
        dec_layers=1, group_detr=1, num_queries=2, num_select=2, hidden_dim=32,
        sa_nheads=2, ca_nheads=2, dim_feedforward=32, out_feature_indexes=(1, 2), num_classes=1,
        segmentation_head=task == "segmentation", use_grouppose_keypoints=task == "keypoints",
        num_keypoints_per_class=schema,
        dual_projector=task == "keypoints", dual_projector_kp_only=task == "keypoints",
    )
    config = tmp_path / "model.json"
    config.write_text(json.dumps(asdict(model_spec)))
    output = tmp_path / "train.json"
    monkeypatch.setattr(sys, "argv", ["train", "--model-config", str(config), "--task", task,
                                      "--dataset-dir", str(tmp_path), "--batch-size", "1", "--epochs", "1",
                                      "--max-targets", "2", "--out", str(output), "--no-precompile",
                                      "--no-compile-cache", "--eval-backend", "pycocotools"])
    cli.train()
    log = json.loads(output.read_text())
    losses = log["epochs"][0]["losses"]
    task_key = "loss_mask_ce" if task == "segmentation" else "loss_keypoints_l1"
    assert task_key in losses and np.isfinite(losses[task_key])
    params = load_params_npz(log["saved_params"])
    saved_spec, metadata = load_model_spec(log["model_metadata"])
    validate_parameter_tree(params, saved_spec)
    assert metadata["task"] == task
    assert metadata["category_to_label"] == {"1": 0 if task == "keypoints" else 1}
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["eval", "--params", log["saved_params"],
                                      "--images", str(tmp_path / "val"),
                                      "--annotations", str(tmp_path / "val" / "_annotations.coco.json"),
                                      "--batch-size", "1", "--eval-backend", "pycocotools"])
    cli.evaluate()
    metrics = json.loads(capsys.readouterr().out)
    assert "mAP" in metrics and "bbox_mAP" in metrics
    assert np.isfinite(metrics["mAP"])
    # Saved model slots must reload directly, without torch-path conversion or task-head loss.
    reloaded = run.load_initial_params(log["saved_params"], params)
    assert set(reloaded) == set(params)
    from rfdetr_jax.inference import RFDETR

    class Tensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    torch_state = {}
    for path, value in flatten_params(params).items():
        key = re.sub(r"_(\d+)(?=_|\.|$)", r".\1", ".".join(path))
        key = key.replace("backbone.0.encoder.", "backbone.0.encoder.encoder.", 1)
        if key in ("refpoint_embed", "query_feat"):
            key += ".weight"
        torch_state[key] = Tensor(value)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        load=lambda *args, **kwargs: {"model": torch_state}, __version__="2.10.0",
        serialization=SimpleNamespace(safe_globals=lambda values: nullcontext())))
    converted_path = tmp_path / "converted.npz"
    monkeypatch.setattr(sys, "argv", ["convert", "model.pth", "--out", str(converted_path),
                                      "--model-config", log["model_metadata"],
                                      "--annotations", str(tmp_path / "val" / "_annotations.coco.json")])
    cli.convert()
    converted = flatten_params(load_params_npz(converted_path))
    for path, value in flatten_params(params).items():
        np.testing.assert_array_equal(converted[path], value)
    detector = RFDETR.from_checkpoint(converted_path)
    predicted = detector.predict(tmp_path / "val" / "image.png", threshold=0)
    assert detector.task == task
    if task == "segmentation":
        assert predicted["masks"].shape[1:] == (32, 32)
    else:
        assert predicted["keypoints"].shape[1:] == (max(schema), 3)
        assert set(predicted["labels"].tolist()) <= {1}


@pytest.mark.parametrize("task", ["segmentation", "keypoints"])
def test_training_geometry_overrides_are_applied_after_task_defaults(monkeypatch, tmp_path, task):
    write_dataset(tmp_path, task=task)
    seen = []

    def stop_before_initialization(model, spec, key):
        seen.append(spec)
        raise RuntimeError("geometry inspected")

    monkeypatch.setattr(run, "init_params", stop_before_initialization)
    monkeypatch.setattr(sys, "argv", ["train", "--task", task, "--dataset-dir", str(tmp_path),
                                      "--batch-size", "1", "--resolution", "48", "--dec-layers", "2",
                                      "--num-queries", "4", "--num-select", "3"])
    with pytest.raises(RuntimeError, match="geometry inspected"):
        cli.train()
    assert seen[0].resolution == 48 and seen[0].dec_layers == 2
    assert seen[0].num_queries == 4 and seen[0].num_select == 3
    assert seen[0].positional_encoding_size == 4


def test_converter_rejects_nonfinite_tensors(monkeypatch, tmp_path, capsys):
    tensor = SimpleNamespace(
        detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([np.nan])))
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *args, **kwargs: {"bad.weight": tensor},
                                                              __version__="2.10.0",
        serialization=SimpleNamespace(safe_globals=lambda values: nullcontext())))
    monkeypatch.setattr(sys, "argv", ["convert", "input.pth", "--out", str(tmp_path / "params.npz")])
    with pytest.raises(SystemExit) as exc:
        cli.convert()
    assert exc.value.code == 2 and "nonfinite" in capsys.readouterr().err


@pytest.mark.parametrize("input_kind", ["checkpoint", "model-config", "annotations"])
def test_conversion_rejects_input_output_collisions_before_loading(monkeypatch, tmp_path, capsys, input_kind):
    output = tmp_path / "weights.npz"
    args = ["convert", str(output) if input_kind == "checkpoint" else "weights.pth", "--out", str(output)]
    if input_kind != "checkpoint":
        args += ["--model-config", str(output) + ".json" if input_kind == "model-config" else "config.json"]
    if input_kind == "annotations":
        args += ["--annotations", str(output) + ".json"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as exc:
        cli.convert()
    assert exc.value.code == 2 and "must not overwrite an input" in capsys.readouterr().err


@pytest.mark.parametrize("task", ["detection", "segmentation", "keypoints"])
def test_training_evaluation_encodes_each_batch_before_predicting_next(monkeypatch, task):
    from rfdetr_jax.evaluation import tasks

    calls = []
    batch = {"pixel_values": np.zeros((1, 3, 4, 4)), "orig_sizes": np.array([[4, 4]]),
             "image_ids": np.array([1]), "num_valid": 1}
    monkeypatch.setattr(run, "iterate_batches", lambda *args, **kwargs: iter([batch, batch, batch]))

    def predict(*args):
        assert len(calls) % 2 == 0, "previous device output has not been encoded"
        calls.append("predict")
        if task == "detection":
            return np.ones((1, 1)), np.ones((1, 1)), np.zeros((1, 1, 4))
        return {"pred_logits": np.ones((1, 1, 2))}

    def accumulate(*args, **kwargs):
        calls.append("encode")
        return []

    monkeypatch.setattr(run, "accumulate_detections", accumulate)
    monkeypatch.setattr(tasks, "accumulate_predictions", accumulate)
    monkeypatch.setattr(run, "evaluate_detections", lambda *args, **kwargs: {"mAP": 0.0})
    monkeypatch.setattr(tasks, "evaluate_predictions", lambda *args, **kwargs: {"mAP": 0.0})
    dataset = SimpleNamespace(category_ids=[1], category_to_label={1: 1})
    result = run.evaluate(predict, {}, RunConfig(task=task), dataset, "annotations.json", 1)
    assert result == {"mAP": 0.0}
    assert calls == ["predict", "encode"] * 3


def test_evaluation_accepts_inferred_sidecar_metadata_with_raw_ids(monkeypatch, tmp_path, capsys):
    """A model saved before any training carries conventional metadata; the CLI evaluates the
    dataset's raw category ids against the head instead of demanding that convention."""
    pytest.importorskip("torch")
    from rfdetr_jax import RFDETRNano
    from rfdetr_jax.cli import evaluate

    write_dataset(tmp_path, category_id=3)
    model = RFDETRNano(pretrain_weights=None, device="cpu", resolution=32, positional_encoding_size=2, patch_size=16,
                       num_windows=1, dec_layers=1, group_detr=1, num_queries=2, num_select=2, hidden_dim=32,
                       sa_nheads=2, ca_nheads=2, dim_feedforward=32, out_feature_indexes=[1, 2], num_classes=5)
    model.save(tmp_path / "untrained.npz")
    monkeypatch.setattr("sys.argv", ["rfdetr-jax-eval", "--params", str(tmp_path / "untrained.npz"),
                                     "--images", str(tmp_path / "val"),
                                     "--annotations", str(tmp_path / "val" / "_annotations.coco.json"),
                                     "--batch-size", "1"])
    evaluate()
    assert "mAP" in capsys.readouterr().out
