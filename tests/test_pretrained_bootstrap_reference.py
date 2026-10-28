"""Tiny published bootstrap oracles using local synthetic checkpoints.

No model builder or official weights are used. The native loader comparison
deliberately bypasses whole-model tree validation for head-only fixture files;
it certifies row algebra/loading only, not acceptance of partial checkpoints.
"""
import importlib.metadata
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def oracle(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    assert importlib.metadata.version("rfdetr") == "1.11.2"
    from rfdetr.config import RFDETRNanoConfig
    from rfdetr.models import weights
    from rfdetr.models.lwdetr import LWDETR
    from rfdetr.utilities import io

    calls = []
    real_safe_load = io._safe_torch_load

    def safe_load(path, *, trust=False):
        calls.append((str(path), trust))
        return real_safe_load(path, trust=trust)

    def local_only(path, **kwargs):
        assert Path(path).is_file(), "oracle must never fetch a missing file"

    monkeypatch.setattr(io, "_safe_torch_load", safe_load)
    monkeypatch.setattr(weights, "download_pretrain_weights", local_only)
    monkeypatch.setattr(weights, "validate_pretrain_weights", local_only)

    class Heads(torch.nn.Module):
        reinitialize_detection_head = LWDETR.reinitialize_detection_head

        def __init__(self, rows, offset=0):
            super().__init__()
            self.two_stage = True
            self.class_embed = torch.nn.Linear(2, rows)
            self.transformer = torch.nn.Module()
            self.transformer.enc_out_class_embed = torch.nn.ModuleList([
                torch.nn.Linear(2, rows), torch.nn.Linear(2, rows)])
            with torch.no_grad():
                for index, head in enumerate([self.class_embed, *self.transformer.enc_out_class_embed]):
                    head.weight.copy_(torch.arange(rows * 2).reshape(rows, 2) + offset + 100 * index)
                    head.bias.copy_(torch.arange(rows) + offset + 1000 * index + 50)

    def write(rows, offset=0, path=None):
        path = path or tmp_path / "custom-head-fixture.pth"
        source = Heads(rows, offset)
        state = {name: value.clone() for name, value in source.state_dict().items()}
        torch.save({"model": state, "args": {"num_queries": 2, "group_detr": 2},
                    "model_name": "RFDETRNano"}, path)
        return path, state

    def config(path, rows, explicit):
        result = RFDETRNanoConfig(pretrain_weights=str(path), num_classes=rows - 1,
                                 num_queries=2, group_detr=2, device="cpu")
        if not explicit:
            result.model_fields_set.discard("num_classes")
        return result

    return SimpleNamespace(torch=torch, Heads=Heads, write=write, config=config,
                           load=weights.load_pretrain_weights, calls=calls)


@pytest.mark.parametrize("source_rows,target_rows,explicit,final_rows", [
    (6, 3, True, 3), (3, 6, True, 6), (3, 3, True, 3),
    (3, 6, False, 3), (6, 3, False, 3), (3, 3, False, 3),
])
def test_published_classifier_resize_includes_background_rows(
        oracle, source_rows, target_rows, explicit, final_rows):
    path, state = oracle.write(source_rows)
    model = oracle.Heads(target_rows, offset=-900)
    config = oracle.config(path, target_rows, explicit)
    oracle.load(model, config)
    assert config.num_classes == final_rows - 1
    assert oracle.calls == [(str(path), False)]
    indices = np.arange(final_rows) % source_rows
    for key, value in model.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), state[key].numpy()[indices])
    assert model.class_embed.out_features == final_rows
    assert all(head.out_features == final_rows for head in model.transformer.enc_out_class_embed)


@pytest.mark.parametrize("source_rows,target_rows", [(6, 3), (3, 6), (3, 3)])
def test_native_existing_loader_matches_explicit_classifier_row_oracle(
        oracle, monkeypatch, source_rows, target_rows):
    from rfdetr_jax import checkpoints
    from rfdetr_jax.convert import flatten_params, torch_key_to_jax_path

    path, _ = oracle.write(source_rows)
    published = oracle.Heads(target_rows)
    oracle.load(published, oracle.config(path, target_rows, True))
    # The fixture has no backbone. Preserve real safe load, conversion and
    # adaptation; bypass only validation that requires a complete transformer.
    monkeypatch.setattr(checkpoints, "validate_parameter_tree", lambda params, spec: None)
    spec, params, _, _ = checkpoints.load_variant(
        "RFDETRNano", pretrain_weights=str(path), num_classes=target_rows - 1,
        num_queries=2, group_detr=2, device="cpu")
    assert spec.detector_classes == target_rows
    native = flatten_params(params)
    for key, value in published.state_dict().items():
        np.testing.assert_array_equal(native[torch_key_to_jax_path(key)], value.numpy())


@pytest.mark.parametrize("source_rows,target_rows", [(6, 3), (3, 6), (3, 3)])
def test_native_training_bootstrap_matches_effective_aligned_classifier_target(
        oracle, monkeypatch, source_rows, target_rows):
    from rfdetr_jax import checkpoints
    from rfdetr_jax.convert import flatten_params, torch_key_to_jax_path
    from rfdetr_jax.variants import spec_from_config, variant_config

    path, _ = oracle.write(source_rows)
    published = oracle.Heads(target_rows)
    # Dataset alignment assigns the effective target on upstream's Pydantic
    # config, making it explicit to the subsequent checkpoint loader.
    oracle.load(published, oracle.config(path, target_rows, True))
    model_config = variant_config("RFDETRNano", pretrain_weights=str(path),
                                  num_queries=2, group_detr=2, device="cpu")
    original_options = model_config.model_dump()
    original_fields = model_config.model_fields_set.copy()
    aligned_spec = replace(spec_from_config(model_config), num_classes=target_rows - 1)
    monkeypatch.setattr(checkpoints, "validate_parameter_tree", lambda params, spec: None)
    params = checkpoints.load_pretrained_training_parameters("RFDETRNano", model_config, aligned_spec)
    native = flatten_params(params)
    for key, value in published.state_dict().items():
        np.testing.assert_array_equal(native[torch_key_to_jax_path(key)], value.numpy())
    assert model_config.model_dump() == original_options
    assert model_config.model_fields_set == original_fields


def test_native_training_bootstrap_rereads_original_before_expansion(oracle, monkeypatch):
    from rfdetr_jax import checkpoints
    from rfdetr_jax.variants import spec_from_config, variant_config

    path, source = oracle.write(6)
    config = variant_config("RFDETRNano", pretrain_weights=str(path), num_queries=2, group_detr=2,
                            num_classes=2, device="cpu")
    small_spec = spec_from_config(config)
    larger_spec = replace(small_spec, num_classes=4)
    monkeypatch.setattr(checkpoints, "validate_parameter_tree", lambda params, spec: None)
    first = checkpoints.load_pretrained_training_parameters("RFDETRNano", config, small_spec)
    second = checkpoints.load_pretrained_training_parameters("RFDETRNano", config, larger_spec)
    np.testing.assert_array_equal(second["class_embed"]["bias"], source["class_embed.bias"].numpy()[:5])
    assert second["class_embed"]["bias"][3] != first["class_embed"]["bias"][0]
    oracle.write(6, offset=700, path=path)
    third = checkpoints.load_pretrained_training_parameters("RFDETRNano", config, larger_spec)
    published = oracle.Heads(5)
    oracle.load(published, oracle.config(path, 5, True))
    np.testing.assert_array_equal(third["class_embed"]["bias"], published.class_embed.bias.detach().numpy())


def test_published_loader_rereads_file_and_reexpands_original_rows(oracle):
    path, state = oracle.write(6)
    first = oracle.Heads(3)
    oracle.load(first, oracle.config(path, 3, True))
    second = oracle.Heads(5)
    oracle.load(second, oracle.config(path, 5, True))
    np.testing.assert_array_equal(second.class_embed.bias.detach().numpy(),
                                  state["class_embed.bias"].numpy()[:5])
    # Loading the initial file again must not use the prior 3-row live state.
    assert second.class_embed.bias[3].item() != first.class_embed.bias[0].item()
    _, changed = oracle.write(6, offset=700, path=path)
    third = oracle.Heads(5)
    oracle.load(third, oracle.config(path, 5, True))
    np.testing.assert_array_equal(third.class_embed.bias.detach().numpy(),
                                  changed["class_embed.bias"].numpy()[:5])
    assert oracle.calls == [(str(path), False)] * 3


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_published_gaussian_reset_exact_rows_all_groups(oracle, bias, enabled):
    from rfdetr.models.lwdetr import LWDETR

    from rfdetr_jax.checkpoints import reset_keypoint_gaussian_parameters
    from rfdetr_jax.convert import convert_state_dict, flatten_params, torch_key_to_jax_path

    torch = oracle.torch

    class MLP(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(3, 3), torch.nn.Linear(3, 3), torch.nn.Linear(3, 8, bias=bias)])
            with torch.no_grad():
                for parameter in self.parameters():
                    parameter.copy_(torch.arange(parameter.numel()).reshape(parameter.shape) + offset)

    root = torch.nn.Module()
    root.use_grouppose_keypoints = enabled
    root.keypoint_embed = MLP(11)
    root.transformer = torch.nn.Module()
    root.transformer.enc_out_keypoint_embed = torch.nn.ModuleList([MLP(101), MLP(201)])
    before = {key: value.clone() for key, value in root.state_dict().items()}
    LWDETR.reset_keypoint_gaussian_parameters(root)
    for key, value in root.state_dict().items():
        expected = before[key].clone()
        if enabled and ".layers.2." in key:
            expected[4:7] = 0
        np.testing.assert_array_equal(value.numpy(), expected.numpy())
    native, _, _ = convert_state_dict({key: value.numpy().copy() for key, value in before.items()})
    # The tree-only helper has no task flag. Its caller must gate disabled
    # keypoint models; native public-bootstrap tests cover that call boundary.
    actual = reset_keypoint_gaussian_parameters(native) if enabled else native
    flat_actual, flat_original = flatten_params(actual), flatten_params(native)
    for key, expected in root.state_dict().items():
        path = torch_key_to_jax_path(key)
        np.testing.assert_array_equal(flat_actual[path], expected.numpy())
        assert flat_actual[path].dtype == expected.numpy().dtype
        np.testing.assert_array_equal(flat_original[path], before[key].numpy())


@pytest.mark.parametrize("rows", [6, 8])
def test_published_gaussian_leaf_guard_and_missing_head(oracle, rows):
    from rfdetr.models.lwdetr import _reset_keypoint_gaussian_output_rows

    from rfdetr_jax.checkpoints import reset_keypoint_gaussian_parameters

    torch = oracle.torch
    head = torch.nn.Linear(2, rows)
    with torch.no_grad():
        head.weight.fill_(7)
        head.bias.fill_(9)
    native = {"keypoint_embed": {"layers_2": {
        "weight": head.weight.detach().numpy().copy(), "bias": head.bias.detach().numpy().copy()}}}
    _reset_keypoint_gaussian_output_rows(SimpleNamespace(layers=[head]))
    assert torch.count_nonzero(head.weight[4:7]).item() == (0 if rows > 6 else 4)
    actual = reset_keypoint_gaussian_parameters(native)["keypoint_embed"]["layers_2"]
    np.testing.assert_array_equal(actual["weight"], head.weight.detach().numpy())
    np.testing.assert_array_equal(actual["bias"], head.bias.detach().numpy())
    np.testing.assert_array_equal(native["keypoint_embed"]["layers_2"]["weight"], np.full((rows, 2), 7))
    _reset_keypoint_gaussian_output_rows(SimpleNamespace())
    _reset_keypoint_gaussian_output_rows(SimpleNamespace(layers=[]))


def test_native_gaussian_reset_leaves_non_keypoint_tree_untouched():
    from rfdetr_jax.checkpoints import reset_keypoint_gaussian_parameters

    values = np.arange(24, dtype=np.float32).reshape(8, 3)
    native = {"class_embed": {"weight": values}, "unrelated": {"layers_2": {"weight": values.copy()}}}
    actual = reset_keypoint_gaussian_parameters(native)
    np.testing.assert_array_equal(actual["class_embed"]["weight"], values)
    np.testing.assert_array_equal(actual["unrelated"]["layers_2"]["weight"], values)
    np.testing.assert_array_equal(native["class_embed"]["weight"], values)
    assert reset_keypoint_gaussian_parameters({}) == {}


@pytest.mark.parametrize("explicit_classes", [False, True])
def test_published_from_checkpoint_path_and_identity_without_step(oracle, monkeypatch, explicit_classes):
    from rfdetr import RFDETR, RFDETRNano
    from rfdetr.config import RFDETRNanoConfig

    path, _ = oracle.write(3)
    captured = []

    def fake_init(self, *, trust_checkpoint=False, **kwargs):
        captured.append((dict(kwargs), trust_checkpoint))
        self.model_config = RFDETRNanoConfig(**kwargs)
        self._has_been_trained = False

    monkeypatch.setattr(RFDETRNano, "__init__", fake_init)
    options = {"pretrain_weights": str(path.parent / "must-not-win.pth")}
    if explicit_classes:
        options["num_classes"] = 4
    result = RFDETR.from_checkpoint(path, **options)
    assert result._has_been_trained is True
    assert captured[-1][0]["pretrain_weights"] == str(path)
    assert captured[-1][1] is False
    assert ("num_classes" in result.model_config.model_fields_set) is explicit_classes
    assert result.model_config.num_classes == (4 if explicit_classes else 2)


def test_published_ordinary_constructor_starts_untrained_with_checkpoint(oracle, monkeypatch):
    from rfdetr import RFDETRNano

    path, _ = oracle.write(3)
    monkeypatch.setattr(RFDETRNano, "maybe_download_pretrain_weights", lambda self: None)
    seen = []

    def tiny_model(self, config, *, trust_checkpoint=False):
        seen.append((str(config.pretrain_weights), trust_checkpoint))
        # Constructor owns the lifecycle identity, regardless of what the
        # supplied builder knows about the loaded checkpoint's training step.
        return SimpleNamespace(step=27, inference_model=None)

    monkeypatch.setattr(RFDETRNano, "get_model", tiny_model)
    result = RFDETRNano(pretrain_weights=str(path))
    assert seen == [(str(path), False)]
    assert result._has_been_trained is False


@pytest.mark.parametrize("trained,resume", [(False, None), (True, None), (True, "native-resume.pth")])
def test_published_retraining_warning_precedes_run_preparation(oracle, monkeypatch, trained, resume):
    import rfdetr.detr as detr
    from rfdetr import RFDETRNano

    class StopBeforeBuild(Exception):
        pass

    def stop(*args, **kwargs):
        assert kwargs.get("resume") == resume
        raise StopBeforeBuild

    monkeypatch.setattr(detr, "_prepare_run_config", stop)
    model = object.__new__(RFDETRNano)
    model._has_been_trained = trained
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(StopBeforeBuild):
            model.train(resume=resume)
    relevant = [item for item in caught if "original pretrained weights" in str(item.message)]
    assert bool(relevant) is trained
