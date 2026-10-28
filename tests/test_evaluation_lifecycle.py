"""Standalone evaluation owns fresh options and temporary execution state."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(params=["detection", "segmentation", "keypoints"])
def eval_case(request, tmp_path, monkeypatch, write_dataset):
    import jax

    from rfdetr_jax import api_training, checkpoints, detr
    from rfdetr_jax.evaluation import public_evaluation
    from rfdetr_jax.models import lwdetr
    from rfdetr_jax.training import run
    from rfdetr_jax.variants import ModelConfig

    task = request.param
    root = tmp_path / "data"
    write_dataset(root, task)
    prior = dict(dataset_dir=str(root), output_dir=str(tmp_path / "old-output"),
                 batch_size=2, eval_batch_size=1, eval_max_dets=17, num_workers=0,
                 keypoint_oks_sigmas=[.13, .17], amp_dtype="fp16", eval_base_model=True,
                 smooth_alpha=.8, skip_best_epochs=3, cls_loss_coef=7.)
    metadata = dict(category_ids=[5], category_names=["custom"], category_to_label={5: 0},
                    public_training_config=prior.copy(), weights="raw")
    spec = lwdetr.LWDETRSpec(resolution=32, positional_encoding_size=2, patch_size=16,
                           num_windows=1, num_classes=1, num_select=2,
                           segmentation_head=task == "segmentation",
                           use_grouppose_keypoints=task == "keypoints",
                           num_keypoints_per_class=(2,) if task == "keypoints" else ())
    # Sentinel heads expose a reload/reset without constructing a transformer.
    params = {"class_embed": {"bias": np.array([3., 7.], np.float32)},
              "keypoint_embed": {"layers_2": {"bias": np.arange(8, dtype=np.float32) + 10}},
              "backbone_0": {"encoder": {"embeddings": {
                  "position_embeddings": np.arange(20, dtype=np.float32).reshape(1, 5, 4)}}}}
    config = ModelConfig(device="cpu", amp=True, resolution=32, positional_encoding_size=2,
                         pretrain_weights=str(tmp_path / "deleted-original.pth"))
    model = SimpleNamespace(spec=spec, params=params, model_config=config,
                            _backend=SimpleNamespace(metadata=metadata),
                            _last_train_kwargs=prior.copy(), _has_been_trained=True,
                            model=SimpleNamespace(resolution=32, args=config, device="cpu"))
    seen = SimpleNamespace(configs=[], networks=[], runs=[], metrics=[], devices=[], fail=False)
    original_config = api_training.training_config

    def capture_config(*args, **kwargs):
        result = original_config(*args, **kwargs)
        seen.configs.append(result)
        return result

    def resolve_device(value):
        # Test per-call device routing without requiring a second physical GPU.
        seen.devices.append(value)
        return jax.devices("cpu")[0]

    def reject_reload(*args, **kwargs):
        pytest.fail("standalone evaluate must not reopen the initial checkpoint")

    monkeypatch.setattr(api_training, "training_config", capture_config)
    monkeypatch.setattr(detr, "resolve_device", resolve_device)
    monkeypatch.setattr(checkpoints, "load_variant", reject_reload)
    monkeypatch.setattr(lwdetr, "LWDETR", lambda *, spec: seen.networks.append(spec) or SimpleNamespace())

    def metrics(annotations, records, **kwargs):
        seen.metrics.append(kwargs)
        return {"mAP": .2, "mAP_50": .3, "mAP_75": .1, "mAR_max": .4,
                "bbox_mAP": .2, "bbox_mAP_50": .3, "bbox_mAP_75": .1,
                "bbox_mAR_max": .4, "F1": .3}

    def backend(forward, actual_params, run_config, dataset, annotations, batch_size, **kwargs):
        seen.runs.append((jax.tree.map(np.asarray, actual_params), run_config, batch_size, kwargs))
        if seen.fail:
            raise RuntimeError("deliberate evaluation failure")
        return kwargs["record_evaluator"]([], {})

    monkeypatch.setattr(public_evaluation, "evaluate_public_predictions", metrics)
    monkeypatch.setattr(run, "evaluate", backend)
    return SimpleNamespace(model=model, seen=seen, root=root, task=task, evaluate=api_training.evaluate)


def test_evaluation_uses_fresh_task_defaults_after_training(eval_case):
    case = eval_case
    case.evaluate(case.model, dataset_dir=case.root)
    config = case.seen.configs[-1]
    assert config.batch_size == 4 and config.eval_batch_size is None
    assert config.eval_max_dets == 500 and config.num_workers == 2
    assert config.keypoint_oks_sigmas is None and config.amp_dtype == "auto"
    assert config.output_dir == "output" and config.device is None
    assert config.eval_base_model is False and config.eval_ema_only is False
    assert config.smooth_alpha == (.5 if case.task == "keypoints" else 0.)
    assert config.skip_best_epochs == (10 if case.task == "keypoints" else 0)
    # Detection/segmentation loss coefficient is resolved by the training adapter;
    # standalone evaluation does not use a criterion.
    assert config.cls_loss_coef == (2. if case.task == "keypoints" else None)
    assert case.seen.runs[-1][2] == 4
    assert case.seen.metrics[-1]["max_dets"] == 500
    assert case.seen.metrics[-1]["keypoint_oks_sigmas"] is None


def test_evaluation_requires_dataset_even_with_training_history(eval_case):
    with pytest.raises(ValueError, match="dataset_dir"):
        eval_case.evaluate(eval_case.model)
    assert not eval_case.seen.runs


def test_evaluation_explicit_options_apply_only_to_that_call(eval_case):
    case = eval_case
    case.evaluate(case.model, dataset_dir=case.root, eval_max_dets=23, batch_size=3,
                  eval_batch_size=2, keypoint_oks_sigmas=[.1, .2], amp_dtype=None,
                  device="cuda:3")
    assert case.seen.runs[-1][2] == 2
    assert case.seen.metrics[-1]["max_dets"] == 23
    assert case.seen.metrics[-1]["keypoint_oks_sigmas"] == [.1, .2]
    assert case.seen.configs[-1].amp_dtype is None
    assert "cuda:3" in case.seen.devices
    assert case.model.model_config.device == "cpu"
    case.evaluate(case.model, dataset_dir=case.root)
    assert case.seen.runs[-1][2] == 4
    assert case.seen.metrics[-1]["max_dets"] == 500
    assert case.seen.configs[-1].amp_dtype == "auto"
    assert case.seen.devices[-1] == "cpu"


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("source_pe", [2, 3])
def test_evaluation_resolution_and_device_are_transient_even_on_failure(eval_case, fail, source_pe):
    case = eval_case
    from dataclasses import replace

    case.model.spec = replace(case.model.spec, positional_encoding_size=source_pe)
    case.model.model_config.positional_encoding_size = source_pe
    case.model.params["backbone_0"]["encoder"]["embeddings"]["position_embeddings"] = np.arange(
        (source_pe**2 + 1) * 4, dtype=np.float32).reshape(1, source_pe**2 + 1, 4)
    expected_pe = 4 if source_pe == 2 else source_pe
    original_spec = case.model.spec
    original_metadata = deepcopy(case.model._backend.metadata)
    original_params = deepcopy(case.model.params)
    case.seen.fail = fail
    if fail:
        with pytest.raises(RuntimeError, match="deliberate evaluation failure"):
            case.evaluate(case.model, dataset_dir=case.root, resolution=64, device="cuda:3")
    else:
        case.evaluate(case.model, dataset_dir=case.root, resolution=64, device="cuda:3")
    assert case.seen.runs[-1][1].resolution == 64
    assert case.seen.networks[-1].resolution == 64
    assert case.seen.networks[-1].positional_encoding_size == expected_pe
    forwarded = case.seen.runs[-1][0]
    assert forwarded["backbone_0"]["encoder"]["embeddings"]["position_embeddings"].shape == (1, expected_pe**2 + 1, 4)
    np.testing.assert_array_equal(forwarded["class_embed"]["bias"], original_params["class_embed"]["bias"])
    np.testing.assert_array_equal(forwarded["keypoint_embed"]["layers_2"]["bias"],
                                  original_params["keypoint_embed"]["layers_2"]["bias"])
    assert case.model.spec is original_spec
    assert case.model.model_config.resolution == case.model.model.resolution == 32
    assert case.model.model_config.positional_encoding_size == source_pe
    assert case.model.model_config.device == case.model.model.device == "cpu"
    assert case.model._backend.metadata == original_metadata
    import jax
    for value, expected in zip(jax.tree.leaves(case.model.params), jax.tree.leaves(original_params), strict=True):
        np.testing.assert_array_equal(value, expected)


def test_evaluation_reads_current_parameters_without_reloading_or_resetting(eval_case):
    case = eval_case
    case.evaluate(case.model, dataset_dir=case.root)
    first = case.seen.runs[-1][0]
    np.testing.assert_array_equal(first["keypoint_embed"]["layers_2"]["bias"], np.arange(8) + 10)
    case.model.params["class_embed"]["bias"] = np.array([-8., 6.], np.float32)
    case.model.params["keypoint_embed"]["layers_2"]["bias"] = np.arange(8, dtype=np.float32) + 20
    case.evaluate(case.model, dataset_dir=case.root)
    current = case.seen.runs[-1][0]
    np.testing.assert_array_equal(current["class_embed"]["bias"], [-8., 6.])
    np.testing.assert_array_equal(current["keypoint_embed"]["layers_2"]["bias"], np.arange(8) + 20)
    assert current["class_embed"]["bias"].shape == first["class_embed"]["bias"].shape
