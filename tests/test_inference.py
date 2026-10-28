"""Installed prediction API: checkpoint contracts and original-image task outputs."""

from __future__ import annotations

import json
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfdetr_jax.convert import load_params_npz, save_params_npz
from rfdetr_jax.inference import RFDETR, prediction_records
from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec


@pytest.fixture(scope="module", params=["detection", "segmentation", "keypoints"])
def checkpoint(request, tmp_path_factory):
    task = request.param
    spec = LWDETRSpec(
        resolution=32, positional_encoding_size=2, patch_size=16, num_windows=1,
        dec_layers=1, num_queries=2, num_select=2, group_detr=1, num_classes=1,
        hidden_dim=32, sa_nheads=2, ca_nheads=2, dim_feedforward=32,
        out_feature_indexes=(1, 2), segmentation_head=task == "segmentation",
        use_grouppose_keypoints=task == "keypoints",
        num_keypoints_per_class=(0, 2) if task == "keypoints" else (),
    )
    model = LWDETR(spec)
    params = model.init(jax.random.key(0), jnp.zeros((1, 3, 32, 32)), None, train=True)["params"]
    path = tmp_path_factory.mktemp(task) / "model.npz"
    save_params_npz(path, params)
    metadata = {"model_spec": asdict(spec), "category_ids": [17], "category_names": ["object"],
                "category_to_label": {"17": 1}}
    path.with_suffix(".npz.json").write_text(json.dumps(metadata))
    return path, spec, task


def test_predict_restores_schema_and_original_image_coordinates(checkpoint):
    path, _, task = checkpoint
    detector = RFDETR.from_checkpoint(path)
    result = detector.predict(np.zeros((19, 7, 3), dtype=np.uint8), threshold=0)
    assert detector.task == task
    assert len(result["scores"]) > 0
    assert set(result["labels"].tolist()) == {17}
    assert np.isfinite(result["boxes"]).all()
    assert (result["boxes"] >= 0).all()
    assert (result["boxes"] <= [7, 19, 7, 19]).all()
    records = prediction_records(result)
    if task == "segmentation":
        from pycocotools import mask as mask_utils

        assert result["masks"].shape[1:] == (19, 7)
        decoded = mask_utils.decode(records[0]["segmentation"])
        np.testing.assert_array_equal(decoded, result["masks"][0])
    elif task == "keypoints":
        assert result["keypoints"].shape[1:] == (2, 3)
        assert result["keypoint_counts"].tolist() == [2] * len(records)
        assert len(records[0]["keypoints"]) == 2
    else:
        assert "segmentation" not in records[0] and "keypoints" not in records[0]
    json.dumps(records, allow_nan=False)
    assert prediction_records(detector.predict(np.zeros((19, 7, 3), dtype=np.uint8), threshold=1)) == []


def test_checkpoint_shape_mismatch_fails_before_prediction(checkpoint, tmp_path):
    original, spec, _ = checkpoint
    params = load_params_npz(original)
    params["class_embed"]["bias"] = np.zeros(9)
    path = tmp_path / "bad.npz"
    save_params_npz(path, params)
    with pytest.raises(ValueError, match="shape mismatch"):
        RFDETR.from_checkpoint(path, spec=spec)


def test_checkpoint_requires_geometry_without_sidecar(tmp_path):
    path = tmp_path / "model.npz"
    save_params_npz(path, {"weight": np.ones(1)})
    with pytest.raises(ValueError, match="requires model metadata"):
        RFDETR.from_checkpoint(path)


@pytest.mark.parametrize("threshold", [-1, 2, np.nan, np.inf])
def test_predict_rejects_invalid_threshold_before_loading_image(threshold):
    detector = object.__new__(RFDETR)
    with pytest.raises(ValueError, match="threshold"):
        detector.predict("missing.png", threshold=threshold)


@pytest.mark.parametrize("pixels", [np.zeros((2, 2)), np.zeros((2, 2, 3), np.int32), np.zeros((0, 2, 3), np.uint8)])
def test_predict_rejects_ambiguous_numpy_inputs(pixels):
    detector = object.__new__(RFDETR)
    with pytest.raises(ValueError, match="HWC RGB uint8"):
        detector.predict(pixels)


def test_compact_schema_preserves_box_only_class_predictions():
    """A compact keypoint schema can leave other detector classes without keypoints."""
    spec = LWDETRSpec(resolution=32, positional_encoding_size=2, num_windows=1, num_classes=1,
                      num_queries=1, num_select=1, use_grouppose_keypoints=True, num_keypoints_per_class=(2,))
    detector = RFDETR(spec, {})
    detector._forward = lambda pixels: {
        "pred_logits": np.array([[[-10., 10.]]]),
        "pred_boxes": np.array([[[.5, .5, .4, .4]]]),
        "pred_keypoints": np.zeros((1, 1, 2, 8)),
    }
    result = detector.predict(np.zeros((16, 16, 3), np.uint8), threshold=0)
    assert result["labels"].tolist() == [1]
    assert result["keypoint_counts"].tolist() == [0]
    assert prediction_records(result)[0]["keypoints"] == []


@pytest.mark.parametrize("metadata", [
    {"category_to_label": {"1": -1}},
    {"category_to_label": {"1": 99}},
    {"category_to_label": {"1": 0.5}},
    {"category_to_label": {"1": True}},
    {"category_to_label": {"one": 0}},
    {"category_to_label": {"1": 0, "2": 0}},
    {"category_to_label": {"1": 0}, "category_ids": [2]},
    {"category_to_label": []},
    {"category_ids": []},
    {"category_ids": [1, 1]},
    {"category_ids": [99]},
    {"category_ids": "1"},
    {"category_ids": [1], "category_names": []},
    {"task": "segmentation"},
])
def test_corrupt_category_metadata_fails_before_prediction(metadata):
    with pytest.raises(ValueError, match="category|task"):
        RFDETR(LWDETRSpec(num_classes=1), {}, metadata)


def test_prediction_supports_rectangular_shape_and_normalized_float_input(checkpoint):
    path, _, task = checkpoint
    detector = RFDETR.from_checkpoint(path)
    image = np.full((11, 19, 3), .12345, np.float32)
    result = detector.predict(image, threshold=0, shape=(32, 48), antialias=False, trace_alpha=0)
    assert result['boxes'].shape[-1] == 4
    assert np.isfinite(result['scores']).all()
    if task == 'segmentation':
        assert result['masks'].shape[1:] == (11, 19)


def test_float_preprocessing_does_not_quantize_and_matches_resize_contract():
    from rfdetr_jax.datasets.coco import IMAGENET_MEAN, IMAGENET_STD
    from rfdetr_jax.utilities.resample import resize_2d_numpy

    spec = LWDETRSpec(resolution=32, positional_encoding_size=2, num_windows=1, num_classes=1,
                      num_queries=1, num_select=1)
    detector = RFDETR(spec, {})
    captured = []

    def forward(pixels):
        captured.append(np.asarray(pixels))
        return {'pred_logits': np.array([[[10., -10.]]]), 'pred_boxes': np.array([[[.5, .5, .4, .4]]])}

    detector._forward = forward
    image = np.random.default_rng(7).random((37, 51, 3), dtype=np.float32)
    detector.predict(image, shape=(32, 48), antialias=False)
    expected = resize_2d_numpy(image.transpose(2, 0, 1)[None], (32, 48), mode='bilinear', antialias=False)
    expected = (expected - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
    np.testing.assert_array_equal(captured[0], expected)


def test_model_label_mode_preserves_slots_missing_from_category_metadata():
    spec = LWDETRSpec(resolution=32, positional_encoding_size=2, num_windows=1, num_classes=1,
                      num_queries=1, num_select=2)
    detector = RFDETR(spec, {}, {'category_ids': [17], 'category_to_label': {'17': 0}})
    detector._forward = lambda pixels: {'pred_logits': np.array([[[5., 10.]]]),
                                        'pred_boxes': np.array([[[.5, .5, .4, .4]]])}
    image = np.zeros((16, 16, 3), np.uint8)
    assert detector.predict(image, threshold=0)['labels'].tolist() == [17]
    assert detector.predict(image, threshold=0, model_labels=True)['labels'].tolist() == [1, 0]


def test_prepare_image_keeps_stored_orientation_like_training_and_upstream(tmp_path):
    """EXIF orientation is not applied: the dataset decoder and upstream predict() ignore it too."""
    from PIL import Image

    pixels = np.zeros((20, 40, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    exif = image.getexif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise when honoured
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif.tobytes())
    model = object.__new__(RFDETR)
    model.spec = LWDETRSpec()
    _, size = model._prepare_image(str(path), None, True)
    assert tuple(size) == (20, 40)
