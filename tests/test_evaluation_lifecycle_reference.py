"""Bounded oracle: execute published evaluate, stub only heavy construction/runners."""
import importlib.metadata
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(params=["detection", "segmentation", "keypoints"])
def published_eval_case(request, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    assert importlib.metadata.version("rfdetr") == "1.11.2"
    import rfdetr.training as training
    from rfdetr import RFDETRKeypointPreview, RFDETRNano, RFDETRSegNano
    from rfdetr.detr import _prepare_run_config

    cls = {"detection": RFDETRNano, "segmentation": RFDETRSegNano,
           "keypoints": RFDETRKeypointPreview}[request.param]
    detector = object.__new__(cls)
    options = dict(device="cpu", resolution=32, positional_encoding_size=2,
                   patch_size=16, num_windows=1, num_classes=1,
                   pretrain_weights=str(tmp_path / "absent-original.pth"))
    if request.param == "keypoints":
        options["num_keypoints_per_class"] = [2]
    detector.model_config = detector._model_config_class(**options)
    pe_key = "backbone.0.encoder.embeddings.position_embeddings"
    live_state = {"live.weight": torch.tensor([19., 23.]),
                  "keypoint_embed.layers.2.bias": torch.arange(8, dtype=torch.float32) + 10,
                  pe_key: torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)}
    detector.model = SimpleNamespace(model=SimpleNamespace(state_dict=lambda: dict(live_state)),
                                     device="cpu", resolution=32,
                                     args=SimpleNamespace(resolution=32, positional_encoding_size=2))
    seen = SimpleNamespace(modules=[], data=[], trainers=[], states=[], fail=False)

    class Module:
        def __init__(self, model_config, train_config):
            seen.modules.append((model_config, train_config))
            self.model = SimpleNamespace(load_state_dict=lambda state: seen.states.append(state))

    class DataModule:
        class_names = ["custom"]

        def __init__(self, model_config, train_config):
            seen.data.append((model_config, train_config))

        def setup(self, stage):
            assert stage in {"test", "validate"}

    def runner(module, datamodule):
        if seen.fail:
            raise RuntimeError("deliberate evaluation failure")
        return [{"test/F1": .3}]

    def trainer(train_config, model_config, **kwargs):
        seen.trainers.append((train_config, model_config, kwargs))
        return SimpleNamespace(test=runner, validate=runner)

    monkeypatch.setattr(training, "RFDETRModelModule", Module)
    monkeypatch.setattr(training, "RFDETRDataModule", DataModule)
    monkeypatch.setattr(training, "build_trainer", trainer)
    # Use actual upstream preparation to model prior training config; no fit or
    # model construction is necessary to prove a later evaluate's fresh defaults.
    prior, accelerator, devices = _prepare_run_config(
        detector, dataset_dir=str(tmp_path / "prior"), batch_size=2,
        eval_batch_size=1, eval_max_dets=17, keypoint_oks_sigmas=[.13, .17],
        amp_dtype="fp16", output_dir=str(tmp_path / "old-output"), device="cpu")
    assert accelerator == "cpu"
    detector.model.args = SimpleNamespace(**prior.model_dump(), resolution=32, positional_encoding_size=2)
    detector._last_train_kwargs = prior.model_dump()
    detector._has_been_trained = True
    return SimpleNamespace(detector=detector, seen=seen, task=request.param,
                            root=str(tmp_path / "fresh"), live=live_state, pe_key=pe_key)


def test_published_evaluate_builds_fresh_per_call_config(published_eval_case):
    case = published_eval_case
    assert case.detector.evaluate(dataset_dir=case.root) == {"test/F1": .3}
    model_config, config = case.seen.modules[-1]
    assert config.batch_size == 4 and config.eval_batch_size is None
    assert config.eval_max_dets == 500 and config.keypoint_oks_sigmas is None
    assert config.amp_dtype == "auto" and str(config.output_dir) == "output"
    assert config.smooth_alpha == (.5 if case.task == "keypoints" else 0.)
    assert config.skip_best_epochs == (10 if case.task == "keypoints" else 0)
    assert config.cls_loss_coef == (2. if case.task == "keypoints" else 1.)
    assert model_config.pretrain_weights is None
    assert case.seen.trainers[-1][2] == {"include_training_callbacks": False}
    # An old dataset on model.args does not satisfy a new call's required field.
    with pytest.raises(ValueError, match="dataset_dir"):
        case.detector.evaluate()


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("source_pe", [2, 3])
def test_published_evaluate_transplants_live_state_and_restores_geometry(published_eval_case, fail, source_pe):
    case = published_eval_case
    case.seen.fail = fail
    import torch

    case.detector.model_config.positional_encoding_size = source_pe
    case.detector.model.args.positional_encoding_size = source_pe
    case.live[case.pe_key] = torch.arange((source_pe**2 + 1) * 4, dtype=torch.float32).reshape(
        1, source_pe**2 + 1, 4)
    expected_pe = 4 if source_pe == 2 else source_pe
    original_config = case.detector.model_config.model_dump()
    args_resolution = case.detector.model.args.resolution
    args_pe = case.detector.model.args.positional_encoding_size
    original_tensors = {key: value.clone() for key, value in case.live.items()}
    if fail:
        with pytest.raises(RuntimeError, match="deliberate evaluation failure"):
            case.detector.evaluate(dataset_dir=case.root, resolution=64, device="cuda:3")
    else:
        case.detector.evaluate(dataset_dir=case.root, resolution=64, device="cuda:3")
    eval_model, config = case.seen.modules[-1]
    assert eval_model.resolution == 64 and eval_model.positional_encoding_size == expected_pe
    assert eval_model.pretrain_weights is None
    assert case.seen.data[-1][0] is eval_model
    assert case.seen.trainers[-1][2] == {
        "include_training_callbacks": False, "accelerator": "gpu", "devices": [3]}
    assert case.detector.model_config.resolution == original_config["resolution"]
    assert case.detector.model_config.positional_encoding_size == original_config["positional_encoding_size"]
    assert case.detector.model_config.device == original_config["device"]
    assert case.detector.model.resolution == args_resolution
    assert case.detector.model.args.resolution == args_resolution
    assert case.detector.model.args.positional_encoding_size == args_pe
    transplanted = case.seen.states[-1]
    assert tuple(transplanted[case.pe_key].shape) == (1, expected_pe**2 + 1, 4)
    for key, value in case.live.items():
        np.testing.assert_array_equal(value, original_tensors[key])
        if key != case.pe_key:
            np.testing.assert_array_equal(transplanted[key], value)
    # Independently compare native temporary PE adaptation to the actual
    # published evaluate transplant, including untouched Gaussian channels.
    from dataclasses import replace

    from rfdetr_jax.checkpoints import _adapt_parameters, resize_position_embeddings
    from rfdetr_jax.models.lwdetr import LWDETRSpec

    native = {"backbone_0": {"encoder": {"embeddings": {
        "position_embeddings": original_tensors[case.pe_key].numpy()}}}}
    spec = replace(LWDETRSpec(), resolution=64, positional_encoding_size=expected_pe)
    adapted = _adapt_parameters(native, spec, source_queries=spec.num_queries, source_groups=spec.group_detr)
    np.testing.assert_allclose(adapted["backbone_0"]["encoder"]["embeddings"]["position_embeddings"],
                               transplanted[case.pe_key].numpy(), rtol=0, atol=1e-6)
    # Exercise the actual evaluation wrapper separately from checkpoint
    # adaptation: changing PE must not resize heads or reset Gaussian rows.
    native["class_embed"] = {"bias": original_tensors["live.weight"].numpy()}
    native["keypoint_embed"] = {"layers_2": {
        "bias": original_tensors["keypoint_embed.layers.2.bias"].numpy()}}
    temporary = resize_position_embeddings(native, expected_pe)
    np.testing.assert_allclose(temporary["backbone_0"]["encoder"]["embeddings"]["position_embeddings"],
                               transplanted[case.pe_key].numpy(), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(temporary["class_embed"]["bias"], transplanted["live.weight"].numpy())
    np.testing.assert_array_equal(temporary["keypoint_embed"]["layers_2"]["bias"],
                                  transplanted["keypoint_embed.layers.2.bias"].numpy())
    np.testing.assert_array_equal(native["backbone_0"]["encoder"]["embeddings"]["position_embeddings"],
                                  original_tensors[case.pe_key].numpy())
