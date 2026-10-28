"""Adversarial checks of the public facade against the pinned upstream contract."""
from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

import rfdetr_jax
from rfdetr_jax import RFDETR
from rfdetr_jax.detr import resolve_device


@pytest.fixture(params=["RFDETRNano", "RFDETRSegNano", "RFDETRKeypointPreview"])
def model(request):
    kwargs = dict(pretrain_weights=None, resolution=32, patch_size=16, positional_encoding_size=2,
                  num_windows=1, dec_layers=1, group_detr=1, num_queries=2, num_select=2, hidden_dim=32,
                  sa_nheads=2, ca_nheads=2, dim_feedforward=32, out_feature_indexes=[1, 2], num_classes=1,
                  device="cpu")
    if request.param == "RFDETRKeypointPreview":
        kwargs["num_keypoints_per_class"] = [0, 2]
    return getattr(rfdetr_jax, request.param)(**kwargs)


def test_named_variants_are_stable_public_classes():
    from rfdetr_jax import variants

    for name in rfdetr_jax._VARIANTS:
        cls = getattr(rfdetr_jax, name)
        assert cls is getattr(rfdetr_jax, name) is getattr(variants, name)
        assert issubclass(cls, RFDETR) and cls.__name__ == name
    assert issubclass(rfdetr_jax.RFDETRSegNano, rfdetr_jax.RFDETRSeg)
    assert rfdetr_jax.RFDETRSeg2XLarge.size == "rfdetr-seg-2xlarge"


def test_public_save_load_prediction_and_callbacks(model, tmp_path):
    def callback(*args):
        return None
    model.callbacks["on_fit_epoch_end"].append(callback)
    model._install(model.spec, model.params, model._backend.metadata, model.model_config)
    assert model.callbacks["on_fit_epoch_end"] == [callback]
    path = tmp_path / "trained.npz"
    model.save(path)
    restored = RFDETR.from_checkpoint(path)
    assert type(restored) is type(model)
    assert restored.spec == model.spec
    image = np.zeros((19, 7, 3), np.uint8)
    result = restored.predict(image, threshold=0, shape=(32, 48))
    assert isinstance(result, sv.KeyPoints if model.spec.use_grouppose_keypoints else sv.Detections)
    result_list = restored.predict([image], threshold=0)
    assert isinstance(result_list, list) and len(result_list) == 1
    if model.spec.segmentation_head:
        assert result.mask.shape[1:] == (19, 7)


def test_optimization_compiles_and_can_restore_original_forward(model):
    image = np.zeros((32, 32, 3), np.uint8)
    original = model._backend._forward
    expected = model.predict(image, threshold=0)
    model.optimize_for_inference()
    assert model._is_optimized_for_inference and model._optimized_has_been_compiled
    assert model.model.inference_model is model._backend._forward
    actual = model.predict(image, threshold=0)
    np.testing.assert_allclose(actual.confidence, expected.confidence, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="batch size"):
        model.predict([image, image])
    with pytest.raises(ValueError, match="optimized for shape"):
        model.predict(image, shape=(32, 48))
    model.remove_optimized_model()
    assert model._backend._forward is original and not model._is_optimized_for_inference
    assert len(model.predict([image, image])) == 2


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_noncompiled_optimization_honors_dtype_without_mutating_weights(model, dtype):
    original = model.params["query_feat"]
    model.inference(compile=False, batch_size=7, dtype=dtype)
    assert model._optimized_dtype.name == dtype
    assert not model._optimized_has_been_compiled and model._optimized_batch_size is None
    assert model.params["query_feat"] is original and original.dtype == np.float32
    result = model.predict(np.zeros((32, 32, 3), np.uint8), threshold=0)
    assert np.isfinite(result.confidence).all()
    model.remove_optimized_model()
    assert model.params["query_feat"] is original


def test_inplace_optimization_is_explicitly_destructive_and_guarded(model):
    with pytest.raises(ValueError, match="requires compile=False"):
        model.inference(inplace=True)
    model.inference(compile=False, inplace=True, dtype="bfloat16")
    assert model.is_optimized_inplace and model.model.model is None
    assert str(model.params["query_feat"].dtype) == "bfloat16"
    assert model.predict(np.zeros((32, 32, 3), np.uint8), threshold=0) is not None
    with pytest.warns(UserWarning, match="no effect"):
        model.remove_optimized_model()
    assert model._is_optimized_for_inference
    for operation in (lambda: model.inference(), lambda: model.export(),
                      lambda: model.train(dataset_dir="missing"), lambda: model.evaluate(dataset_dir="missing")):
        with pytest.raises(RuntimeError, match="inplace"):
            operation()


@pytest.mark.parametrize(("options", "error"), [
    ({"dtype": "int32"}, ValueError), ({"dtype": "bad-dtype"}, TypeError),
    ({"dtype": "float64"}, NotImplementedError), ({"batch_size": 0}, ValueError),
    ({"batch_size": True}, ValueError), ({"shape": (31, 32)}, ValueError),
])
def test_invalid_optimization_does_not_set_success_flags(model, options, error):
    with pytest.raises(error):
        model.inference(**options)
    assert not model._is_optimized_for_inference and model.model.inference_model is None


def test_device_resolution_rejects_negative_or_unavailable_ordinals():
    assert resolve_device("cpu:0").platform == "cpu"
    for value in ("cpu:-1", "cpu:999999", "cpu:abc", "bad:0"):
        with pytest.raises(ValueError):
            resolve_device(value)


def test_compiled_multiimage_prediction_runs_one_real_batch(model):
    images = [np.zeros((19, 7, 3), np.uint8), np.full((11, 23, 3), 127, np.uint8)]
    expected = model.predict(images, threshold=0)
    model.inference(batch_size=2)
    original_forward = model._backend._forward
    seen = []

    def record_forward(pixels):
        seen.append(pixels.shape)
        return original_forward(pixels)

    model._backend._forward = record_forward
    actual = model.predict(images, threshold=0)
    assert seen == [(2, 3, 32, 32)]
    assert len(actual) == 2
    for reference, prediction in zip(expected, actual, strict=True):
        np.testing.assert_allclose(prediction.confidence, reference.confidence, atol=1e-5, rtol=1e-5)
        assert prediction.data["source_shape"].shape[1] == 2
    if model.spec.segmentation_head:
        assert actual[0].mask.shape[1:] == (19, 7)
        assert actual[1].mask.shape[1:] == (11, 23)
    with pytest.raises(ValueError, match="batch size"):
        model.predict(images[0])


@pytest.mark.parametrize("variant", ["RFDETRNano", "RFDETRSegNano", "RFDETRKeypointPreview"])
def test_saved_checkpoint_loads_through_real_upstream_public_api(tmp_path, variant):
    torch = pytest.importorskip("torch")
    reference = pytest.importorskip("rfdetr")
    options = dict(pretrain_weights=None, resolution=32, patch_size=16, positional_encoding_size=2,
                   num_windows=1, dec_layers=1, group_detr=1, num_queries=2, num_select=2, hidden_dim=32,
                   sa_nheads=2, ca_nheads=2, out_feature_indexes=[3, 6, 9, 12], num_classes=1, device="cpu")
    if variant == "RFDETRKeypointPreview":
        options["num_keypoints_per_class"] = [0, 2]
    original = getattr(rfdetr_jax, variant)(**options)
    destination = tmp_path / "checkpoint_best_total.pth"
    original.save(destination)
    from rfdetr_jax.checkpoints import safe_torch_load, torch_state_dict

    assert safe_torch_load(destination)["model_name"] == variant
    restored = reference.RFDETR.from_checkpoint(destination)
    assert type(restored).__name__ == variant
    assert restored.model_config.num_classes == original.spec.num_classes
    expected = torch_state_dict(original.params, original.spec)
    actual = restored.model.model.state_dict()
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    image = np.zeros((32, 32, 3), np.uint8)
    result = restored.predict(image, threshold=0, include_source_image=False)
    assert isinstance(result, sv.KeyPoints if original.spec.use_grouppose_keypoints else sv.Detections)



def test_constructor_rejects_unsupported_torch_compile_before_loading_weights():
    with pytest.raises(NotImplementedError, match="JAX steps already use jit"):
        rfdetr_jax.RFDETRNano(compile=True)


def test_supervision_readme_annotator_consumers(model):
    image = np.zeros((19, 27, 3), np.uint8)
    detections = model.predict(image, threshold=0)
    if model.spec.use_grouppose_keypoints:
        annotated = sv.VertexAnnotator().annotate(image.copy(), detections)
        assert annotated.shape == image.shape and annotated.dtype == np.uint8
        return
    annotated = sv.BoxAnnotator().annotate(image.copy(), detections)
    annotated = sv.LabelAnnotator().annotate(annotated, detections,
                                            labels=detections.data["class_name"].tolist())
    if model.spec.segmentation_head:
        annotated = sv.MaskAnnotator().annotate(annotated, detections)
    assert annotated.shape == image.shape and annotated.dtype == np.uint8


def test_public_coco_asset_names_match_prediction_mapping():
    from rfdetr_jax.assets.coco_classes import COCO_CLASS_NAMES, COCO_CLASSES
    from rfdetr_jax.prediction import COCO_CLASSES as expected

    assert expected == COCO_CLASSES
    assert [expected[key] for key in sorted(expected)] == COCO_CLASS_NAMES


def test_constructor_warnings_and_option_errors_follow_upstream():
    import warnings as warnings_module

    from rfdetr_jax import RFDETRBase, RFDETRNano
    from rfdetr_jax.configuration import UnsupportedOptionError
    from rfdetr_jax.variants import PretrainWeightsCompatibilityWarning

    tiny = dict(resolution=32, patch_size=16, positional_encoding_size=2, num_windows=1, dec_layers=1, group_detr=1,
                num_queries=2, num_select=2, hidden_dim=32, sa_nheads=2, ca_nheads=2, dim_feedforward=32,
                out_feature_indexes=[1, 2], num_classes=1, device="cpu")
    with pytest.warns(PretrainWeightsCompatibilityWarning, match="pretrain_weights=None"):
        model = RFDETRNano(pretrain_weights=None, **tiny)
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        RFDETRBase(pretrain_weights=None, **{**tiny, "patch_size": 14, "resolution": 28})
    assert any(issubclass(item.category, FutureWarning) and "deprecated" in str(item.message) for item in caught)
    with pytest.raises(UnsupportedOptionError) as info:
        RFDETRNano(pretrain_weights=None, bogus_option=1, **tiny)
    assert isinstance(info.value, TypeError) and isinstance(info.value, ValueError)
    with pytest.raises(ValueError, match="Unknown attribute"):
        model.model_config.not_a_field = 1
    with pytest.raises(ValueError, match="patch_size"):
        model.predict(np.zeros((32, 32, 3), np.uint8), patch_size=8)
