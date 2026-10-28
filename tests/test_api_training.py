"""Public workflows exercise real tiny models, real optimizers and standalone checkpoints."""
import json
import shutil
from dataclasses import asdict, replace

import jax
import numpy as np
import pytest
from PIL import Image

from rfdetr_jax.api_training import _TrainingMonitor, training_config


def test_best_checkpoint_and_early_stopping_have_independent_thresholds():
    config = training_config(early_stopping=True, early_stopping_min_delta=.1,
                             early_stopping_patience=2)
    monitor = _TrainingMonitor(config, "detection")
    assert monitor.update({"epoch": 0, "val": {"mAP": .5}, "val_ema": {"mAP": .4}}) == (
        ["val", "val_ema"], "val", False)
    assert monitor.update({"epoch": 1, "val": {"mAP": .51}, "val_ema": {"mAP": .45}}) == (
        ["val", "val_ema"], "val", False)
    assert monitor.update({"epoch": 2, "val": {"mAP": .52}, "val_ema": {"mAP": .49}}) == (
        ["val", "val_ema"], "val", True)


def test_early_stopping_ema_option_and_segmentation_recall_select_the_right_metric():
    config = training_config(early_stopping=True, early_stopping_use_ema=True,
                             early_stopping_patience=1, best_model_metric="mar")
    monitor = _TrainingMonitor(config, "segmentation")
    monitor.update({"epoch": 0, "val": {"mAR_max": .9, "bbox_mAR_max": .1},
                    "val_ema": {"mAR_max": .8, "bbox_mAR_max": .2}})
    improved, winner, stop = monitor.update({"epoch": 1, "val": {"bbox_mAR_max": .3},
                                            "val_ema": {"bbox_mAR_max": .2}})
    assert improved == ["val"] and winner == "val" and stop


@pytest.mark.parametrize("kwargs", [{"smooth_alpha": 1.}, {"weight_decay": float("nan")},
                                    {"seed": -1}, {"mask_point_sample_ratio": 0},
                                    {"skip_best_epochs": -1}, {"eval_batch_size": 0}])
def test_invalid_or_unimplemented_options_fail_explicitly(kwargs):
    with pytest.raises((ValueError, NotImplementedError)):
        training_config(**kwargs)


@pytest.mark.parametrize("task", ["detection", "segmentation", "keypoints"])
@pytest.mark.parametrize("fresh_none", [False, True], ids=["checkpoint-source", "fresh-native"])
def test_public_train_save_reload_standalone_resume_evaluate_predict(tmp_path, task, fresh_none, write_dataset):
    pytest.importorskip("torch")
    import supervision as sv

    from rfdetr_jax import RFDETR, RFDETRKeypointPreview, RFDETRNano, RFDETRSegNano
    from rfdetr_jax.checkpoints import save_checkpoint
    from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec
    from rfdetr_jax.training.checkpoint import checkpoint_metadata
    from rfdetr_jax.training.trainer import init_params

    dataset = tmp_path / "data"
    write_dataset(dataset, task)
    spec = LWDETRSpec(resolution=32, positional_encoding_size=2, patch_size=16, num_windows=1,
                      dec_layers=1, group_detr=1, num_queries=2, num_select=2, hidden_dim=32,
                      sa_nheads=2, ca_nheads=2, dim_feedforward=32, out_feature_indexes=(1, 2),
                      num_classes=2, gradient_checkpointing=False,
                      segmentation_head=task == "segmentation", use_grouppose_keypoints=task == "keypoints",
                      num_keypoints_per_class=(2,) if task == "keypoints" else (),
                      dual_projector=task == "keypoints", dual_projector_kp_only=task == "keypoints")
    params = init_params(LWDETR(spec=spec), spec, jax.random.key(3))
    initial = tmp_path / "initial.npz"
    save_checkpoint(initial, params, spec, {"task": task})
    options = dict(dataset_dir=str(dataset), epochs=2, batch_size=1, num_workers=0, tensorboard=False,
                   no_precompile=True, no_compile_cache=True, multi_scale=False, seed=None,
                   lr=2e-4, lr_scheduler="cosine",
                   bbox_loss_coef=7., eval_max_dets=17, eval_base_model=True,
                   keypoint_oks_sigmas=[.1, .1] if task == "keypoints" else None)
    def make_model(poison):
        if not fresh_none:
            return RFDETR.from_checkpoint(initial)
        cls = {"detection": RFDETRNano, "segmentation": RFDETRSegNano,
               "keypoints": RFDETRKeypointPreview}[task]
        fresh = cls(pretrain_weights=None, device="cpu", **asdict(spec))
        # Distinct deliberately corrupted live trees must never become a fresh
        # run's initialization. The real optimizer/backend executes below.
        fresh._backend.params = jax.tree.map(lambda value: np.full_like(value, poison), fresh.params)
        return fresh

    model = make_model(12345)
    assert model.train(output_dir=str(tmp_path / "full"), **options) is None
    expected = jax.tree.map(lambda value: np.asarray(value).copy(), model._training_result.state)

    interrupted = make_model(23456)

    class Interrupted(Exception):
        pass

    def interrupt(record):
        assert record["epoch"] == 0
        raise Interrupted

    interrupted.callbacks["on_fit_epoch_end"].append(interrupt)
    partial = tmp_path / "partial"
    with pytest.raises(Interrupted):
        interrupted.train(output_dir=str(partial), **options)
    standalone = tmp_path / "standalone.pth"
    shutil.copyfile(partial / "checkpoint.pth", standalone)
    shutil.rmtree(partial)
    saved = checkpoint_metadata(standalone)
    assert saved["run_config"]["seed"] == 42
    assert saved["run_config"]["data_seed"] == 7
    assert saved["model_metadata"]["public_training_config"]["seed"] is None
    assert saved["criterion_config"]["bbox_loss_coef"] == 7.
    resumed = RFDETR.from_checkpoint(standalone)
    resume_options = {} if fresh_none else {"seed": 42}
    assert resumed.train(resume=standalone, output_dir=str(tmp_path / "resumed"), **resume_options) is None
    pairs = zip(jax.tree.leaves(resumed._training_result.state), jax.tree.leaves(expected), strict=True)
    for actual, reference in pairs:
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=1e-7)
    assert resumed.class_names == ["custom"]
    assert int(resumed._training_result.state.step) == 2
    raw = resumed._training_result.state.params
    ema = resumed._training_result.state.ema_params
    # One update cannot catch accidental EMA installation: upstream copies
    # its first update. The second update must distinguish the two trees.
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(raw), jax.tree.leaves(ema), strict=True))
    for actual, reference in zip(jax.tree.leaves(resumed.params), jax.tree.leaves(raw), strict=True):
        np.testing.assert_array_equal(actual, reference)
    assert resumed._backend.metadata["weights"] == "raw"
    # Public installation must not mutate the lower-level trainer/export policy.
    assert resumed._training_result.metadata["weights"] == "ema"
    for actual, reference in zip(jax.tree.leaves(resumed._training_result.params), jax.tree.leaves(ema), strict=True):
        np.testing.assert_array_equal(actual, reference)
    exported = tmp_path / "trained.pth"
    resumed.save(exported)
    restored = RFDETR.from_checkpoint(exported)
    assert restored._backend.metadata["weights"] == "raw"
    for actual, reference in zip(jax.tree.leaves(restored.params), jax.tree.leaves(raw), strict=True):
        np.testing.assert_array_equal(actual, reference)
    image = dataset / "test" / "image.png"
    before_save = resumed.predict(image, threshold=0)
    result = restored.predict(image, threshold=0)
    assert isinstance(result, sv.KeyPoints if task == "keypoints" else sv.Detections)
    fields = ("xy", "keypoint_confidence", "detection_confidence", "class_id") if task == "keypoints" else (
        "xyxy", "confidence", "class_id")
    for field in fields:
        np.testing.assert_array_equal(getattr(result, field), getattr(before_save, field))
    if task == "segmentation":
        np.testing.assert_array_equal(result.mask, before_save.mask)
    metrics = restored.evaluate(dataset_dir=str(dataset), batch_size=1, num_workers=0, tensorboard=False,
                                eval_max_dets=17,
                                keypoint_oks_sigmas=[.1, .1] if task == "keypoints" else None)
    assert metrics and all(np.isfinite(value) for value in metrics.values())
    # Published train() retains raw module.model; standalone evaluation must
    # agree with the regular validation track, not implicitly select EMA.
    validation = resumed._training_result.log["epochs"][-1]["val"]
    assert "F1" in validation and "precision" in validation and "recall" in validation
    assert validation["F1"] == metrics["test/F1"]
    bbox_key = "mAP" if task == "detection" else "bbox_mAP"
    assert validation[bbox_key] == metrics["test/mAP_50_95"]
    assert resumed._training_result.metadata["public_training_config"]["eval_max_dets"] == 17
    assert {"test/F1", "test/precision", "test/recall", "test/mAP_50_95"}.issubset(metrics)
    if task == "segmentation":
        assert "test/segm_mAP_50_95" in metrics
    elif task == "keypoints":
        assert "test/keypoint_map_50_95" in metrics
    assert restored.spec == replace(restored.spec, resolution=32)
    # Run the real network at a larger evaluation geometry. The temporary PE
    # transplant must leave the model ready for its original inference shape.
    inference_spec = restored.spec
    inference_params = restored.params
    inference_leaves = jax.tree.leaves(inference_params)
    resized_metrics = restored.evaluate(
        dataset_dir=str(dataset), resolution=64, batch_size=1, num_workers=0,
        tensorboard=False, eval_max_dets=17,
        keypoint_oks_sigmas=[.1, .1] if task == "keypoints" else None)
    assert resized_metrics.keys() == metrics.keys()
    assert all(np.isfinite(value) for value in resized_metrics.values())
    assert restored.spec is inference_spec and restored.params is inference_params
    assert restored.model_config.resolution == 32
    assert restored.model_config.positional_encoding_size == 2
    for actual, original in zip(jax.tree.leaves(restored.params), inference_leaves, strict=True):
        assert actual is original


def test_regular_smoothing_warms_during_skipped_epochs_and_compares_raw_winner():
    config = training_config(task="keypoints", skip_best_epochs=1)
    assert config.smooth_alpha == .5 and config.cls_loss_coef == 2.
    monitor = _TrainingMonitor(config, "keypoints")
    assert monitor.update({"epoch": 0, "val": {"mAP": .8}, "val_ema": {"mAP": .5}}) == ([], None, False)
    assert monitor.smoothed_regular == .4
    improved, winner, _ = monitor.update({"epoch": 1, "val": {"mAP": .5}, "val_ema": {"mAP": .55}})
    assert improved == ["val", "val_ema"] and winner == "val_ema"
    assert monitor.best["val"] == .45 and monitor.best_raw["val"] == .5


def test_default_public_multiscale_runs_real_optimizer_update(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from rfdetr_jax import RFDETRNano
    from rfdetr_jax.training import augmentation, run
    from rfdetr_jax.training.checkpoint import checkpoint_metadata

    root = tmp_path / "data"
    write_yolo_dataset(root)
    model = RFDETRNano(pretrain_weights=None, resolution=32, positional_encoding_size=2,
                       patch_size=16, num_windows=1, dec_layers=1, group_detr=1, num_queries=2,
                       num_select=2, hidden_dim=32, sa_nheads=2, ca_nheads=2, dim_feedforward=32,
                       out_feature_indexes=[1, 2], num_classes=1, gradient_checkpointing=False,
                       freeze_encoder=True, fused_optimizer=False, ia_bce_loss=False)
    encoder_before = []
    initialize = run.init_params

    def record_initial_encoder(*args, **kwargs):
        params = initialize(*args, **kwargs)
        encoder_before.append(jax.tree.map(lambda value: np.asarray(value).copy(), params["backbone_0"]["encoder"]))
        return params

    monkeypatch.setattr(run, "init_params", record_initial_encoder)
    observed = []
    original_resize = augmentation.resize_training_batch

    def record_resize(batch, scale):
        observed.append((batch["pixel_values"].shape[-1], scale))
        return original_resize(batch, scale)

    monkeypatch.setattr(augmentation, "resize_training_batch", record_resize)
    model.train(dataset_dir=root, output_dir=tmp_path / "output", epochs=1, batch_size=1,
                grad_accum_steps=2, num_workers=0, tensorboard=False,
                no_precompile=True, no_compile_cache=True, scale_jitter=False)
    scales = augmentation.compute_multi_scale_scales(32, True, 16, 1)
    assert observed == [(max(scales), augmentation.scale_for_step(scales, 0))]
    assert int(model._training_result.state.step) == 1
    assert np.isfinite(model._training_result.log["epochs"][0]["losses"]["loss"])
    metadata = checkpoint_metadata(model._training_result.paths["checkpoint"])
    assert metadata["run_config"]["multi_scale"] is True
    assert metadata["run_config"]["fused_optimizer"] is False
    assert metadata["criterion_config"]["ia_bce_loss"] is False
    assert len(encoder_before) == 1
    for actual, before in zip(jax.tree.leaves(model.params["backbone_0"]["encoder"]),
                              jax.tree.leaves(encoder_before[0]), strict=True):
        np.testing.assert_array_equal(actual, before)
    with pytest.raises(ValueError, match="preserve lr"):
        model.train(resume=model._training_result.paths["checkpoint"], lr=.9)


def write_yolo_dataset(root, *, filename="data.yaml"):
    root.mkdir(parents=True, exist_ok=True)
    (root / filename).write_text("names: [object]\n")
    for split in ("train", "valid"):
        images, labels = root / split / "images", root / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        Image.new("RGB", (32, 32), (60, 120, 180)).save(images / "image.png")
        (labels / "image.txt").write_text("0 .5 .5 .5 .5\n")


@pytest.mark.parametrize("filename", ["data.yaml", "data.yml"])
def test_roboflow_format_detects_yolo_but_prioritizes_coco(tmp_path, filename):
    from rfdetr_jax.api_training import _dataset_format
    write_yolo_dataset(tmp_path, filename=filename)
    assert _dataset_format(tmp_path, "roboflow") == "yolo"
    (tmp_path / "train" / "_annotations.coco.json").write_text("{}")
    assert _dataset_format(tmp_path, "roboflow") == "roboflow"


@pytest.mark.parametrize("field,value", [("keypoints", ["right", "left"]), ("skeleton", [[2, 1]])])
def test_validation_rejects_reordered_keypoint_schema(tmp_path, field, value, write_dataset):
    from rfdetr_jax.api_training import _prepare_split
    root = tmp_path / "data"
    write_dataset(root, "keypoints")
    train = json.loads((root / "train" / "_annotations.coco.json").read_text())
    path = root / "valid" / "_annotations.coco.json"
    validation = json.loads(path.read_text())
    validation["categories"][0][field] = value
    path.write_text(json.dumps(validation))
    with pytest.raises(ValueError, match=f"ordered {field}"):
        _prepare_split(root, "val", training_config(dataset_dir=root), tmp_path,
                       categories=train["categories"], keypoints=True)


def evaluation_model(metadata, *, keypoints=False):
    from types import SimpleNamespace

    from rfdetr_jax.models.lwdetr import LWDETRSpec
    return SimpleNamespace(_backend=SimpleNamespace(metadata=metadata), params={},
                           spec=LWDETRSpec(resolution=32, patch_size=16, num_windows=1,
                                           use_grouppose_keypoints=keypoints,
                                           num_keypoints_per_class=(2,) if keypoints else ()),
                           model_config=SimpleNamespace(device="cpu"))


def test_evaluate_rejects_checkpoint_category_name_mismatch_before_inference(tmp_path, monkeypatch, write_dataset):
    from rfdetr_jax.api_training import evaluate
    from rfdetr_jax.training import run
    write_dataset(tmp_path, "detection")
    model = evaluation_model({"category_ids": [5], "category_names": ["unrelated"], "category_to_label": {5: 0}})
    monkeypatch.setattr(run, "evaluate", lambda *args, **kwargs: pytest.fail("inference must not run"))
    with pytest.raises(ValueError, match="category names disagree with the checkpoint"):
        evaluate(model, dataset_dir=tmp_path)


def test_evaluate_rejects_saved_joint_order_even_when_splits_agree(tmp_path, monkeypatch, write_dataset):
    from rfdetr_jax.api_training import evaluate
    from rfdetr_jax.training import run
    write_dataset(tmp_path, "keypoints")
    model = evaluation_model({"category_ids": [5], "category_names": ["custom"], "category_to_label": {5: 0},
                              "category_schema": [{"id": 5, "name": "custom", "keypoints": ["right", "left"],
                                                   "skeleton": [[1, 2]]}]}, keypoints=True)
    monkeypatch.setattr(run, "evaluate", lambda *args, **kwargs: pytest.fail("inference must not run"))
    with pytest.raises(ValueError, match="ordered keypoints"):
        evaluate(model, dataset_dir=tmp_path)


def test_evaluate_automatic_yolo_missing_test_falls_back_to_validation(tmp_path, monkeypatch):
    from rfdetr_jax.api_training import evaluate
    from rfdetr_jax.training import run
    write_yolo_dataset(tmp_path)
    model = evaluation_model({"category_ids": [0], "category_names": ["object"], "category_to_label": {0: 0}})
    seen = []

    def metrics(*args, **kwargs):
        seen.append(args[3].images[0]["file_name"])
        return {"mAP": .2, "mAP_50": .3, "mAP_75": .1, "mAR_max": .4}

    monkeypatch.setattr(run, "evaluate", metrics)
    with pytest.warns(UserWarning, match="test split is unavailable"):
        result = evaluate(model, dataset_dir=tmp_path)
    assert result["test/mAP_50_95"] == .2 and seen == ["image.png"]


@pytest.mark.parametrize("task,dataset_file,schema,saved_mapping,expected", [
    ("detection", "coco", (), None, {1: 1, 3: 3, 90: 90}),
    ("segmentation", "coco", (), None, {1: 1, 3: 3, 90: 90}),
    ("detection", "roboflow", (), None, {1: 0, 3: 1, 90: 2}),
    ("segmentation", "roboflow", (), None, {1: 0, 3: 1, 90: 2}),
    ("keypoints", "coco", (2,), None, {90: 0, 1: 1, 3: 2}),
    ("keypoints", "coco", (0, 2), None, {90: 1, 1: 0, 3: 2}),
    ("keypoints", "roboflow", (2,), None, {90: 0, 1: 1, 3: 2}),
    ("detection", "coco", (), {1: 2, 3: 1, 90: 0}, {1: 2, 3: 1, 90: 0}),
    ("segmentation", "coco", (), {1: 2, 3: 1, 90: 0}, {1: 2, 3: 1, 90: 0}),
], ids=["native-detection", "native-segmentation", "roboflow-detection", "roboflow-segmentation",
        "native-keypoints-compact", "native-keypoints-legacy", "roboflow-keypoints",
        "saved-detection", "saved-segmentation"])
def test_standalone_evaluate_preserves_dataset_label_contract(
        tmp_path, monkeypatch, task, dataset_file, schema, saved_mapping, expected):
    """Pristine constructor metadata must not make native COCO labels contiguous.

    Stub only model execution: the public evaluation path still reads the real
    split and derives its mapping. No weights or model initialization are needed.
    """
    from rfdetr_jax.api_training import evaluate
    from rfdetr_jax.training import run

    keypoints = task == "keypoints"
    if dataset_file == "coco":
        image_dir = tmp_path / "val2017"
        annotation_path = tmp_path / "annotations" / (
            "person_keypoints_val2017.json" if keypoints else "instances_val2017.json")
    else:
        image_dir = tmp_path / "valid"
        annotation_path = image_dir / "_annotations.coco.json"
    image_dir.mkdir(parents=True)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32)).save(image_dir / "image.png")
    categories = [{"id": category, "name": f"category-{category}"} for category in (1, 3, 90)]
    if keypoints:
        categories[-1]["keypoints"] = ["left", "right"]
    annotation_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "image.png", "height": 32, "width": 32}],
        "categories": categories,
        "annotations": [{"id": 1, "image_id": 1, "category_id": 90,
                         "bbox": [8, 8, 16, 16], "area": 256, "iscrowd": 0,
                         "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]],
                         **({"keypoints": [12, 12, 2, 20, 20, 2], "num_keypoints": 2}
                            if keypoints else {})}],
    }))
    if dataset_file == "roboflow":
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        shutil.copyfile(annotation_path, train_dir / "_annotations.coco.json")
    model = evaluation_model({} if saved_mapping is None else {"category_to_label": saved_mapping},
                             keypoints=keypoints)
    model.spec = replace(model.spec, segmentation_head=task == "segmentation", num_keypoints_per_class=schema)
    observed = []

    def metrics(*args, **kwargs):
        observed.append(args[3].category_to_label)
        values = {"mAP": 1., "mAP_50": 1., "mAP_75": 1., "mAR_max": 1.}
        return {**values, **{f"bbox_{name}": value for name, value in values.items()}}

    monkeypatch.setattr(run, "evaluate", metrics)
    result = evaluate(model, dataset_dir=tmp_path, dataset_file=dataset_file, split="val", num_workers=0)
    assert observed == [expected]
    assert result["val/mAP_50_95"] == 1.


@pytest.mark.parametrize("invalid", ["declared", "labels", "empty"])
def test_yolo_invalid_test_does_not_fall_back(tmp_path, invalid):
    from rfdetr_jax.api_training import _prepare_split
    write_yolo_dataset(tmp_path)
    if invalid == "declared":
        (tmp_path / "data.yaml").write_text("names: [object]\ntest: missing/images\n")
    else:
        images = tmp_path / "test" / "images"
        images.mkdir(parents=True)
        if invalid == "labels":
            Image.new("RGB", (32, 32)).save(images / "image.png")
        else:
            (tmp_path / "test" / "labels").mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        _prepare_split(tmp_path, "test", training_config(dataset_dir=tmp_path), tmp_path)


def test_native_coco_sparse_category_survives_training_reload_evaluation_prediction(
        tmp_path, monkeypatch, write_dataset):
    pytest.importorskip("torch")
    import jax.numpy as jnp

    from rfdetr_jax import RFDETR, RFDETRNano
    from rfdetr_jax.training import run

    root = tmp_path / "coco"
    write_dataset(root, "detection")
    annotations = root / "annotations"
    annotations.mkdir()
    for source, destination in (("train", "train2017"), ("valid", "val2017")):
        (root / source).rename(root / destination)
        (root / destination / "_annotations.coco.json").rename(annotations / f"instances_{destination}.json")
    model = RFDETRNano(pretrain_weights=None, resolution=32, positional_encoding_size=2,
                       patch_size=16, num_windows=1, dec_layers=1, group_detr=1, num_queries=2,
                       num_select=2, hidden_dim=32, sa_nheads=2, ca_nheads=2, dim_feedforward=32,
                       out_feature_indexes=[1, 2], num_classes=5, gradient_checkpointing=False)
    params = jax.tree.map(lambda value: value, model.params)
    params["class_embed"]["weight"] = jnp.zeros_like(params["class_embed"]["weight"])
    params["class_embed"]["bias"] = jnp.full_like(params["class_embed"]["bias"], -8.).at[5].set(8.)
    model._install(model.spec, params, model._backend.metadata, model.model_config)
    initial = tmp_path / "initial.npz"
    model.save(initial)
    # Reload makes the class count inferred, so the regression also catches accidental automatic shrinking.
    model = RFDETR.from_checkpoint(initial)
    observed_labels = []
    iterate = run.iterate_batches

    def record_training_labels(dataset, *args, **kwargs):
        for batch in iterate(dataset, *args, **kwargs):
            if kwargs.get("train"):
                assert dataset.category_to_label == {5: 5}
                observed_labels.extend(batch["labels"][batch["target_mask"]].tolist())
            yield batch

    monkeypatch.setattr(run, "iterate_batches", record_training_labels)
    model.train(dataset_dir=str(root), dataset_file="coco", output_dir=str(tmp_path / "output"),
                epochs=1, batch_size=1, num_workers=0, tensorboard=False, no_precompile=True,
                no_compile_cache=True, multi_scale=False, scale_jitter=False, seed=0)
    assert observed_labels == [5]
    assert model.spec.num_classes == 5 and model.params["class_embed"]["bias"].shape == (6,)
    assert int(model._training_result.state.step) == 1
    exported = tmp_path / "trained.pth"
    model.save(exported)
    restored = RFDETR.from_checkpoint(exported)
    assert restored.spec.num_classes == 5 and restored.class_names == ["custom"]
    assert {int(key): value for key, value in restored._backend.metadata["category_to_label"].items()} == {5: 5}
    detections = restored.predict(root / "val2017" / "image.png", threshold=.5)
    assert len(detections) == 2
    assert detections.class_id.tolist() == [5, 5]
    assert detections.data["class_name"].tolist() == ["custom", "custom"]
    metrics = restored.evaluate(dataset_dir=str(root), dataset_file="coco", split="val", batch_size=1,
                                num_workers=0, tensorboard=False)
    assert all(np.isfinite(value) for value in metrics.values())
    validation = model._training_result.log["epochs"][0]["val_ema"]
    assert metrics["val/mAP_50_95"] == validation["mAP"]
    assert metrics["val/F1"] == validation["F1"]


def _upstream_style_checkpoint(path, task, class_names, *, num_classes=None):
    """Write the payload upstream train() produces: weights plus args/model_config, no port metadata."""
    import torch

    from rfdetr_jax.checkpoints import torch_state_dict
    from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec
    from rfdetr_jax.training.trainer import init_params

    num_classes = len(class_names) if num_classes is None else num_classes
    spec = LWDETRSpec(resolution=32, positional_encoding_size=2, patch_size=16, num_windows=1, dec_layers=1,
                      group_detr=1, num_queries=2, num_select=2, hidden_dim=32, sa_nheads=2, ca_nheads=2,
                      dim_feedforward=32, out_feature_indexes=(1, 2), num_classes=num_classes,
                      segmentation_head=task == "segmentation", use_grouppose_keypoints=task == "keypoints",
                      num_keypoints_per_class=(2,) if task == "keypoints" else (),
                      dual_projector=task == "keypoints", dual_projector_kp_only=task == "keypoints")
    params = init_params(LWDETR(spec=spec), spec, jax.random.key(5))
    variant = {"detection": "RFDETRNano", "segmentation": "RFDETRSegNano", "keypoints": "RFDETRKeypointPreview"}[task]
    model_config = {key: (list(value) if isinstance(value, tuple) else value) for key, value in asdict(spec).items()}
    model_config.update(model_name=variant, pretrain_weights=None, device="cpu")
    payload = {"model": torch_state_dict(params, spec), "epoch": 0,
               "args": {"class_names": list(class_names), "dataset_dir": "/data/example", "output_dir": "output"},
               "model_config": model_config, "model_name": variant}
    torch.save(payload, path)
    return spec


@pytest.mark.parametrize("task", ["detection", "segmentation", "keypoints"])
def test_evaluate_accepts_upstream_finetuned_checkpoint_on_its_dataset(tmp_path, task, write_dataset):
    """Upstream checkpoints carry label-indexed class names; the dataset keeps raw category ids."""
    pytest.importorskip("torch")
    from rfdetr_jax import RFDETR

    dataset = tmp_path / "data"
    write_dataset(dataset, task)
    checkpoint = tmp_path / "checkpoint_best_total.pth"
    _upstream_style_checkpoint(checkpoint, task, ["custom"])
    model = RFDETR.from_checkpoint(checkpoint)
    assert model.class_names == ["custom"]
    options = dict(dataset_dir=str(dataset), batch_size=1, num_workers=0, tensorboard=False,
                   keypoint_oks_sigmas=[.1, .1] if task == "keypoints" else None)
    metrics = model.evaluate(split="val", **options)
    assert {"val/mAP_50_95", "val/F1"}.issubset(metrics)
    assert all(np.isfinite(value) for value in metrics.values())
    prediction = model.predict(dataset / "valid" / "image.png", threshold=0)
    assert len(prediction) == 2  # every query survives threshold 0; labels stay model-indexed


def test_evaluate_reports_class_schema_mismatches_against_upstream_checkpoints(tmp_path, write_dataset):
    pytest.importorskip("torch")
    from rfdetr_jax import RFDETR

    dataset = tmp_path / "data"
    write_dataset(dataset, "detection")
    renamed = tmp_path / "renamed.pth"
    _upstream_style_checkpoint(renamed, "detection", ["widget"])
    with pytest.warns(UserWarning, match="class names"):
        metrics = RFDETR.from_checkpoint(renamed).evaluate(
            split="val", dataset_dir=str(dataset), batch_size=1, num_workers=0, tensorboard=False)
    assert all(np.isfinite(value) for value in metrics.values())
    wider = tmp_path / "wider.pth"
    _upstream_style_checkpoint(wider, "detection", ["custom", "other"])
    with pytest.warns(UserWarning, match="2 classes"):
        metrics = RFDETR.from_checkpoint(wider).evaluate(split="val", dataset_dir=str(dataset), batch_size=1,
                                                         num_workers=0, tensorboard=False)
    assert all(np.isfinite(value) for value in metrics.values())


def test_resume_inherits_the_saved_resolved_precision_unless_requested(tmp_path, write_dataset, monkeypatch):
    """A CPU FP32 run resumed on a BF16-capable device keeps FP32 unless amp_dtype is passed."""
    pytest.importorskip("torch")
    from rfdetr_jax import RFDETR, RFDETRNano
    from rfdetr_jax.training import precision as precision_module

    dataset = tmp_path / "data"
    write_dataset(dataset, "detection")
    tiny = dict(pretrain_weights=None, device="cpu", resolution=32, positional_encoding_size=2, patch_size=16,
                num_windows=1, dec_layers=1, group_detr=1, num_queries=2, num_select=2, hidden_dim=32, sa_nheads=2,
                ca_nheads=2, dim_feedforward=32, out_feature_indexes=[1, 2], num_classes=1)
    options = dict(dataset_dir=str(dataset), epochs=2, batch_size=1, num_workers=0, tensorboard=False,
                   no_precompile=True, no_compile_cache=True, multi_scale=False)
    model = RFDETRNano(**tiny)

    class Stop(Exception):
        pass

    def stop(record):
        raise Stop

    model.callbacks["on_fit_epoch_end"].append(stop)
    with pytest.raises(Stop):
        model.train(output_dir=str(tmp_path / "run"), **options)
    checkpoint = tmp_path / "run" / "checkpoint.pth"
    original = precision_module.resolve_training_precision

    def bf16_device(requested, legacy_amp, device):
        resolved = original(requested, legacy_amp, device)
        if resolved["effective"] in ("auto", "bf16"):
            resolved.update(compute_dtype="bfloat16", head_dtype="activation")
        return resolved

    monkeypatch.setattr(precision_module, "resolve_training_precision", bf16_device)
    resumed = RFDETR.from_checkpoint(checkpoint)
    assert resumed.train(resume=str(checkpoint), output_dir=str(tmp_path / "resumed")) is None
    assert resumed._training_result.metadata["training_precision"]["compute_dtype"] == "float32"
    explicit = RFDETR.from_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="resolved training precision"):
        explicit.train(resume=str(checkpoint), output_dir=str(tmp_path / "explicit"), amp_dtype="bf16")


def test_subclass_from_checkpoint_returns_the_recorded_variant(tmp_path):
    pytest.importorskip("torch")
    from rfdetr_jax import RFDETRSegNano

    checkpoint = tmp_path / "nano.pth"
    _upstream_style_checkpoint(checkpoint, "detection", ["custom"])
    model = RFDETRSegNano.from_checkpoint(checkpoint)
    assert type(model).__name__ == "RFDETRNano" and model.size == "rfdetr-nano"


def test_train_updates_model_config_in_place_and_writes_upstream_files(tmp_path, write_dataset):
    pytest.importorskip("torch")
    from rfdetr_jax import RFDETRNano
    from rfdetr_jax.checkpoints import safe_torch_load

    dataset = tmp_path / "data"
    write_dataset(dataset, "detection")
    tiny = dict(pretrain_weights=None, device="cpu", resolution=32, positional_encoding_size=2, patch_size=16,
                num_windows=1, dec_layers=1, group_detr=1, num_queries=2, num_select=2, hidden_dim=32, sa_nheads=2,
                ca_nheads=2, dim_feedforward=32, out_feature_indexes=[1, 2])
    model = RFDETRNano(**tiny)
    config = model.model_config
    assert config.num_classes == 90
    model.train(dataset_dir=str(dataset), epochs=1, batch_size=1, num_workers=0, tensorboard=False,
                no_precompile=True, no_compile_cache=True, multi_scale=False, output_dir=str(tmp_path / "out"))
    assert model.model_config is config and config.num_classes == 1
    files = {path.name for path in (tmp_path / "out").iterdir()}
    assert {"last_ema.pth", "training_config.json", "checkpoint.pth"} <= files
    recorded = json.loads((tmp_path / "out" / "training_config.json").read_text())
    assert recorded["class_names"] == ["custom"] and recorded["num_classes"] == 1
    assert recorded["model_config"]["num_classes"] == 1 and recorded["train_config"]["epochs"] == 1
    ema = safe_torch_load(tmp_path / "out" / "last_ema.pth")
    assert ema["metadata"]["weights"] == "ema"


def test_inferred_checkpoint_evaluates_native_coco_layout_with_raw_ids(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image

    from rfdetr_jax import RFDETRNano

    root = tmp_path / "coco"
    for split in ("train2017", "val2017"):
        (root / split).mkdir(parents=True)
        Image.new("RGB", (32, 32), (90, 90, 90)).save(root / split / "image.png")
    (root / "annotations").mkdir()
    payload = {"images": [{"id": 1, "file_name": "image.png", "width": 32, "height": 32}],
               "annotations": [{"id": 1, "image_id": 1, "category_id": 5, "bbox": [8, 8, 16, 16], "area": 256,
                                "iscrowd": 0}],
               "categories": [{"id": 5, "name": "airplane"}]}
    for split in ("train2017", "val2017"):
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(payload))
    tiny = dict(pretrain_weights=None, device="cpu", resolution=32, positional_encoding_size=2, patch_size=16,
                num_windows=1, dec_layers=1, group_detr=1, num_queries=2, num_select=2, hidden_dim=32, sa_nheads=2,
                ca_nheads=2, dim_feedforward=32, out_feature_indexes=[1, 2])
    import warnings as warnings_module

    model = RFDETRNano(**tiny)  # 90-class head with inferred COCO metadata
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        metrics = model.evaluate(split="val", dataset_dir=str(root), dataset_file="coco", batch_size=1,
                                 num_workers=0, tensorboard=False)
    assert not [item for item in caught if "classes" in str(item.message)]  # sparse raw ids are the normal case
    assert all(np.isfinite(value) for value in metrics.values())


def test_train_rejects_max_targets_below_the_dataset_capacity(tmp_path, write_dataset):
    pytest.importorskip("torch")
    from rfdetr_jax import RFDETRNano

    dataset = tmp_path / "data"
    write_dataset(dataset, "detection")
    annotations = dataset / "train" / "_annotations.coco.json"
    raw = json.loads(annotations.read_text())
    raw["annotations"].append({"id": 2, "image_id": 1, "category_id": 5, "bbox": [2, 2, 8, 8], "area": 64,
                               "iscrowd": 0})
    annotations.write_text(json.dumps(raw))
    tiny = dict(pretrain_weights=None, device="cpu", resolution=32, positional_encoding_size=2, patch_size=16,
                num_windows=1, dec_layers=1, group_detr=1, num_queries=2, num_select=2, hidden_dim=32, sa_nheads=2,
                ca_nheads=2, dim_feedforward=32, out_feature_indexes=[1, 2])
    with pytest.raises(ValueError, match="max_targets=1 is below the 2"):
        RFDETRNano(**tiny).train(dataset_dir=str(dataset), epochs=1, batch_size=1, num_workers=0, tensorboard=False,
                                 max_targets=1, output_dir=str(tmp_path / "out"))
