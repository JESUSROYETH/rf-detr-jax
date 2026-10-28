"""Public scoring replaces raw AP once and serves standalone and in-fit validation."""
import json
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from rfdetr_jax import api_training
from rfdetr_jax.models.lwdetr import LWDETRSpec
from rfdetr_jax.training import run
from rfdetr_jax.training.run_config import RunConfig
from rfdetr_jax.variants import ModelConfig


@pytest.mark.parametrize("task", ["detection", "segmentation", "keypoints"])
def test_record_evaluator_bypasses_raw_ap_and_receives_only_valid_decoded_sizes(monkeypatch, task):
    from rfdetr_jax.evaluation import tasks
    batch = {"pixel_values": np.zeros((2, 3, 4, 4)), "orig_sizes": np.array([[10, 20], [999, 999]]),
             "image_ids": np.array([1, 999]), "num_valid": 1}
    monkeypatch.setattr(run, "iterate_batches", lambda *args, **kwargs: iter([batch]))

    def fail_raw(*args, **kwargs):
        pytest.fail("raw AP must not run before the public evaluator")

    monkeypatch.setattr(run, "evaluate_detections", fail_raw)
    monkeypatch.setattr(tasks, "evaluate_predictions", fail_raw)

    def encode(image_ids, *args, **kwargs):
        assert list(image_ids) == [1]
        if task != "detection":
            assert kwargs["trace_alpha"] == .7
        return [{"image_id": 1, "category_id": 5}]

    monkeypatch.setattr(run, "accumulate_detections", encode)
    monkeypatch.setattr(tasks, "accumulate_predictions", encode)
    seen = []

    def evaluator(records, image_sizes):
        seen.append((records, image_sizes))
        return {"mAP": .37}

    def predict(*args):
        if task == "detection":
            return np.ones((2, 1)), np.zeros((2, 1)), np.zeros((2, 1, 4))
        return {"pred_logits": np.ones((2, 1, 2))}

    dataset = SimpleNamespace(category_ids=[5], category_to_label={5: 0})
    result = run.evaluate(predict, {}, RunConfig(task=task, postprocess_trace_alpha=.7), dataset, "unused.json", 2,
                          record_evaluator=evaluator)
    assert result == {"mAP": .37}
    assert seen == [([{"image_id": 1, "category_id": 5}], {1: (10, 20)})]


def test_record_and_additive_metric_callbacks_cannot_be_combined():
    with pytest.raises(ValueError, match="mutually exclusive"):
        run.evaluate(None, {}, None, None, "unused.json", 1,
                     record_evaluator=lambda *args: {}, prediction_metrics=lambda *args: {})


@pytest.mark.parametrize("value", [0, -1, None, 2.5, True])
def test_public_max_dets_requires_positive_integer(value):
    with pytest.raises(ValueError, match="eval_max_dets"):
        api_training.training_config(eval_max_dets=value)


def test_native_mask_grid_option_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match="eval_masks_head_resolution"):
        api_training.training_config(eval_masks_head_resolution=True)
    assert api_training.training_config().eval_max_dets == 500


@pytest.mark.parametrize("task", ["detection", "segmentation", "keypoints"])
def test_public_train_and_evaluate_share_record_evaluator_and_independent_budget(tmp_path, monkeypatch, task):
    from rfdetr_jax import checkpoints
    from rfdetr_jax.evaluation import public_evaluation
    spec = LWDETRSpec(resolution=32, patch_size=16, num_windows=1, num_classes=1, num_select=2,
                      segmentation_head=task == "segmentation", use_grouppose_keypoints=task == "keypoints",
                      num_keypoints_per_class=(2,) if task == "keypoints" else ())
    category = {"id": 5, "name": "object"}
    if task == "keypoints":
        category.update(keypoints=["a", "b"], skeleton=[[1, 2]])
    for split in ("train", "valid", "test"):
        directory = tmp_path / split
        directory.mkdir()
        Image.new("RGB", (20, 10)).save(directory / "image.png")
        (directory / "_annotations.coco.json").write_text(json.dumps({
            "images": [{"id": 1, "file_name": "image.png", "width": 20, "height": 10}],
            "annotations": [], "categories": [category],
        }))
    metadata = {"category_ids": [5], "category_names": ["object"], "category_to_label": {5: 0}}
    model = SimpleNamespace(spec=spec, params={},
                            model_config=ModelConfig(device="cpu", amp=False, model_fields_set=set(),
                                                     postprocess_trace_alpha=.7),
                            _backend=SimpleNamespace(metadata=metadata), callbacks=defaultdict(list),
                            _install=lambda *args: None)
    monkeypatch.setattr(checkpoints, "adapt_parameters", lambda params, *args, **kwargs: params)
    records, sizes = [{"image_id": 1}], {1: (10, 20)}
    calls = []
    raw = {"mAP": .7, "mAP_50": .8, "mAP_75": .6, "mAR_max": .9,
           "bbox_mAP": .1, "bbox_mAP_50": .2, "bbox_mAP_75": .05, "bbox_mAR_max": .3,
           "F1": .4, "precision": .5, "recall": .6}

    def evaluator(annotations, received, **kwargs):
        assert received == records
        assert kwargs["image_sizes"] == sizes
        assert kwargs["task"] == task and kwargs["resolution"] == 32
        assert kwargs["max_dets"] == 17
        calls.append(json.loads(annotations.read_text()))
        return raw

    monkeypatch.setattr(public_evaluation, "evaluate_public_predictions", evaluator)

    def train_backend(argv, **kwargs):
        assert argv[argv.index("--postprocess-trace-alpha") + 1] == "0.7"
        assert kwargs["metadata"]["postprocess_trace_alpha"] == .7
        metrics = kwargs["evaluation_record_evaluator"](records, sizes)
        assert metrics is raw
        return run.TrainingResult({}, SimpleNamespace(params={}), kwargs["metadata"],
                                  {"epochs": [{"val_ema": metrics}]}, {})

    monkeypatch.setattr(run, "run_training", train_backend)
    api_training.train(model, dataset_dir=tmp_path, output_dir=tmp_path / "output", tensorboard=False,
                       eval_max_dets=17)

    def standalone_backend(*args, **kwargs):
        assert args[2].num_select == 2
        assert args[2].postprocess_trace_alpha == .7
        assert "prediction_metrics" not in kwargs
        return kwargs["record_evaluator"](records, sizes)

    monkeypatch.setattr(run, "evaluate", standalone_backend)
    metrics = api_training.evaluate(model, dataset_dir=tmp_path, tensorboard=False, eval_max_dets=17)
    assert len(calls) == 2
    assert metrics["test/mAP_50_95"] == (.7 if task == "detection" else .1)
    assert metrics["test/F1"] == .4
    if task == "segmentation":
        assert metrics["test/segm_mAP_50_95"] == .7
    if task == "keypoints":
        assert metrics["test/keypoint_map_50_95"] == .7


@pytest.mark.parametrize("task", ["detection", "segmentation"])
def test_standard_coco_keeps_sparse_ids_and_pretrained_head(task):
    spec = LWDETRSpec(num_classes=90, segmentation_head=task == "segmentation")
    model = SimpleNamespace(spec=spec, model_config=ModelConfig(model_fields_set=set()))
    dataset = SimpleNamespace(category_ids=[5, 17, 90])
    actual, mapping = api_training._schema(model, dataset, api_training.training_config(dataset_file="coco"))
    assert actual.num_classes == 90
    assert mapping == {5: 5, 17: 17, 90: 90}
    for dataset_file in ("roboflow", "yolo"):
        actual, mapping = api_training._schema(model, dataset, api_training.training_config(dataset_file=dataset_file))
        assert actual.num_classes == 3 and mapping == {5: 0, 17: 1, 90: 2}


def test_standard_coco_rejects_a_head_too_small_for_raw_ids():
    model = SimpleNamespace(spec=LWDETRSpec(num_classes=2), model_config=ModelConfig(model_fields_set=set()))
    with pytest.raises(ValueError, match="cannot represent"):
        api_training._schema(model, SimpleNamespace(category_ids=[5]),
                             api_training.training_config(dataset_file="coco"))


@pytest.mark.parametrize("value,enabled", [(True, True), (False, False), ("off", False), ("per-batch", True)])
def test_public_multiscale_strings_preserve_their_meaning(value, enabled):
    assert api_training.training_config(multi_scale=value).multi_scale is enabled


def test_public_multiscale_per_sample_is_not_silently_per_batch():
    with pytest.raises(NotImplementedError, match="per-sample"):
        api_training.training_config(multi_scale="per-sample")


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False), (1, True), (0, False), (1., True), (0., False),
    ("TrUe", True), ("FALSE", False), ("on", True), ("off", False),
    ("YES", True), ("no", False), ("1", True), ("0", False),
])
def test_public_boolean_training_options(value, expected):
    options = {name: value for name, default in api_training._DEFAULTS.items()
               if type(default) is bool and name not in {"multi_scale", "eval_masks_head_resolution", "eval_ema_only"}}
    config = api_training.training_config(**options)
    assert all(getattr(config, name) is expected for name in options)


@pytest.mark.parametrize("value", [None, 2, -.1, float("nan"), "", "maybe", [], {}, object()])
def test_public_boolean_training_options_reject_ambiguous_values(value):
    with pytest.raises(ValueError, match="use_ema must be a boolean"):
        api_training.training_config(use_ema=value)


def test_deprecated_boolean_flag_uses_normalized_value():
    with pytest.warns(FutureWarning):
        config = api_training.training_config(eval_ema_only="false", use_ema="false")
    assert config.eval_base_model is True
    assert config.use_ema is False
    assert api_training.training_config(eval_masks_head_resolution="false").eval_masks_head_resolution is False
    with pytest.raises(NotImplementedError, match="eval_masks_head_resolution"):
        api_training.training_config(eval_masks_head_resolution="TRUE")


def test_resume_boolean_options_compare_normalized_values(monkeypatch):
    from rfdetr_jax.training import checkpoint
    monkeypatch.setattr(checkpoint, "checkpoint_metadata", lambda path: {
        "run_config": {"use_ema": False, "scale_jitter": False}, "model_metadata": {},
    })
    config, _ = api_training._resume_configuration(
        {"resume": "checkpoint.pth", "use_ema": "false", "scale_jitter": "OFF"}, "detection")
    assert config.use_ema is False and config.scale_jitter is False
    with pytest.raises(ValueError, match="resume must preserve use_ema"):
        api_training._resume_configuration({"resume": "checkpoint.pth", "use_ema": "true"}, "detection")


@pytest.mark.parametrize("task", ["detection", "segmentation", "keypoints"])
def test_public_record_path_keeps_zero_score_topk_predictions(monkeypatch, task):
    from rfdetr_jax.evaluation import tasks
    batch = {"pixel_values": np.zeros((1, 3, 4, 4)), "orig_sizes": np.array([[10, 20]]),
             "image_ids": np.array([1]), "num_valid": 1}
    monkeypatch.setattr(run, "iterate_batches", lambda *args, **kwargs: iter([batch]))
    outputs = {"pred_logits": np.array([[[-1000.]]]), "pred_boxes": np.array([[[.5, .5, .5, .5]]])}
    if task == "segmentation":
        outputs["pred_masks"] = np.ones((1, 1, 2, 2))
    if task == "keypoints":
        outputs["pred_keypoints"] = np.zeros((1, 1, 1, 8))

    def predict(*args):
        if task == "detection":
            return np.zeros((1, 1)), np.zeros((1, 1)), np.array([[[5., 2.5, 15., 7.5]]])
        return outputs

    dataset = SimpleNamespace(category_ids=[1], category_to_label={1: 0})
    config = RunConfig(task=task, num_select=1, num_keypoints_per_class=(1,))
    recorded = []
    run.evaluate(predict, {}, config, dataset, "unused.json", 1,
                 record_evaluator=lambda records, sizes: recorded.extend(records) or {})
    assert len(recorded) == 1 and recorded[0]["score"] == 0.
    raw = []
    monkeypatch.setattr(run, "evaluate_detections", lambda path, records, **kwargs: raw.extend(records) or {})
    monkeypatch.setattr(tasks, "evaluate_predictions", lambda path, records, **kwargs: raw.extend(records) or {})
    run.evaluate(predict, {}, config, dataset, "unused.json", 1)
    assert raw == []


def test_keypoint_evaluation_uses_nondefault_trace_alpha_numerically(monkeypatch):
    batch = {"pixel_values": np.zeros((1, 3, 4, 4)), "orig_sizes": np.array([[10, 20]]),
             "image_ids": np.array([1]), "num_valid": 1}
    monkeypatch.setattr(run, "iterate_batches", lambda *args, **kwargs: iter([batch]))
    outputs = {"pred_logits": np.array([[[2.]]]), "pred_boxes": np.array([[[.5, .5, .5, .5]]]),
               "pred_keypoints": np.zeros((1, 1, 1, 8))}
    dataset = SimpleNamespace(category_ids=[1], category_to_label={1: 0})
    scores = []
    for alpha in (0., .7):
        run.evaluate(lambda *args: outputs, {}, RunConfig(task="keypoints", num_select=1,
                     num_keypoints_per_class=(1,), postprocess_trace_alpha=alpha), dataset, "unused", 1,
                     record_evaluator=lambda records, sizes: scores.append(records[0]["score"]) or {})
    np.testing.assert_allclose(scores, [.880797, .351573], atol=1e-6)
