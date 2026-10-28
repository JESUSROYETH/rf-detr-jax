"""Public prediction contract against actual Supervision containers."""
from types import SimpleNamespace

import numpy as np
import pytest
import supervision as sv
from PIL import Image

from rfdetr_jax.prediction import COCO_CLASS_NAMES, precision_cholesky_to_pixel_covariance, predict


class Backend:
    def __init__(self, task='detection', labels=(0, 1), metadata=None):
        self.spec = SimpleNamespace(patch_size=16, num_windows=1)
        self.metadata = metadata or {}
        self.task = task
        self.labels = labels
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append((image, kwargs))
        count = len(self.labels)
        result = {'scores': np.linspace(.9, .5, count, dtype=np.float32),
                  'labels': np.array(self.labels), 'boxes': np.tile([1, 2, 3, 4], (count, 1))}
        if self.task == 'segmentation':
            result['masks'] = np.ones((count, *image.shape[:2]), bool)
        if self.task == 'keypoints':
            result['keypoints'] = np.tile([[[3, 4, .7], [5, 6, .2]]], (count, 1, 1))
            result['keypoint_precision_cholesky'] = np.zeros((count, 2, 3))
        return result


def model(task='detection', **kwargs):
    return SimpleNamespace(_backend=Backend(task, **kwargs), class_names=['cat', 'dog'],
                           model_config=SimpleNamespace(patch_size=16, num_windows=1, resolution=32,
                                                        num_classes=2, num_keypoints_per_class=(),
                                                        postprocess_trace_alpha=.4))


@pytest.mark.parametrize('task', ['detection', 'segmentation', 'keypoints'])
def test_actual_supervision_types_and_metadata(task):
    detector = model(task)
    image = np.full((10, 20, 3), 111, np.uint8)
    result = predict(detector, image)
    assert isinstance(result, sv.KeyPoints if task == 'keypoints' else sv.Detections)
    assert len(result) == 1  # strict greater-than threshold, .5 itself excluded
    assert result.class_id.tolist() == [0]
    assert result.data['class_name'].tolist() == ['cat']
    np.testing.assert_array_equal(result.data['source_shape'], [[10, 20]])
    if task == 'keypoints':
        np.testing.assert_allclose(result.detection_confidence, [.9])
        np.testing.assert_allclose(result.keypoint_confidence, [[.7, .2]])
        np.testing.assert_array_equal(result.data['xyxy'], [[1, 2, 3, 4]])
        np.testing.assert_array_equal(result.data['covariance'][0, 0], [[400, 0], [0, 100]])
        np.testing.assert_array_equal(result.data['source_image'][0], image)
    else:
        np.testing.assert_array_equal(result.metadata['source_image'], image)
        if task == 'segmentation':
            assert result.mask.shape == (1, 10, 20)
    call = detector._backend.calls[0]
    assert call[1]['antialias'] is False
    assert call[1]['trace_alpha'] == .4


@pytest.mark.parametrize('task', ['detection', 'segmentation'])
def test_native_coco_sparse_labels_use_saved_class_name_mapping(task):
    # JSON keys are strings after checkpoint reload; model-label order, rather
    # than dictionary insertion order, determines class_names alignment.
    detector = model(task, labels=(5, 90), metadata={'category_to_label': {'90': 90, '5': 5}})
    detector.class_names = ['custom airplane', 'custom toothbrush']
    detector.model_config.num_classes = 90
    result = predict(detector, Image.new('RGB', (8, 8)), threshold=0)
    assert result.class_id.tolist() == [5, 90]
    assert result.data['class_name'].tolist() == ['custom airplane', 'custom toothbrush']


@pytest.mark.parametrize('task', ['detection', 'segmentation', 'keypoints'])
def test_empty_predictions_and_no_source(task):
    detector = model(task)
    result = predict(detector, np.zeros((8, 9, 3), np.uint8), threshold=1, include_source_image=False)
    assert len(result) == 0
    assert result.data['source_shape'].shape == (0, 2)
    assert 'source_image' not in result.data
    if task == 'keypoints':
        assert result.xy.shape == (0, 2, 2)
    else:
        assert 'source_image' not in result.metadata
        if task == 'segmentation':
            assert result.mask.shape == (0, 8, 9)


def test_list_and_tuple_stay_lists_and_float_inputs_are_not_quantized():
    detector = model()
    image = np.full((11, 7, 3), .12345, np.float32)
    for batch in ([image], (image,), [image, image]):
        results = predict(detector, batch, shape=(32, 48))
        assert isinstance(results, list) and len(results) == len(batch)
    assert predict(detector, []) == []
    np.testing.assert_array_equal(detector._backend.calls[0][0], image)
    assert detector._backend.calls[0][1]['shape'] == (32, 48)


def test_local_and_url_pil_inputs_convert_rgb(tmp_path, monkeypatch):
    import io

    import requests

    gray = Image.new('L', (8, 12), 50)
    path = tmp_path / 'image.png'
    gray.save(path)
    stream = io.BytesIO()
    gray.save(stream, format='PNG')
    response = SimpleNamespace(content=stream.getvalue(), raise_for_status=lambda: None)
    monkeypatch.setattr(requests, 'get', lambda url, timeout: response)
    detector = model()
    for image in (path, str(path), gray, 'https://example.test/image.png'):
        result = predict(detector, image)
        assert result.metadata['source_image'].shape == (12, 8, 3)
        assert detector._backend.calls[-1][0].shape == (12, 8, 3)


@pytest.mark.parametrize('shape', [(0, 32), (-1, 32), (32.0, 32), (True, 32), (31, 32), (32,), 32])
def test_invalid_shape_fails_before_forward(shape):
    detector = model()
    with pytest.raises(ValueError):
        predict(detector, 'missing.png', shape=shape)
    assert not detector._backend.calls


@pytest.mark.parametrize('image', [np.zeros((3, 8, 8)), np.zeros((0, 8, 3)),
                                    np.full((8, 8, 3), np.nan), np.full((8, 8, 3), 1.1),
                                    np.full((8, 8, 3), -.1), np.zeros((8, 8, 3), np.int32)])
def test_invalid_arrays_fail_before_forward(image):
    detector = model()
    with pytest.raises(ValueError):
        predict(detector, image)
    assert not detector._backend.calls


def test_unknown_kwargs_rejected():
    with pytest.raises(TypeError, match='unsupported_option'):
        predict(model(), None, unsupported_option=True)


def test_public_adapter_requests_model_labels():
    detector = model(labels=(0,), metadata={'category_to_label': {'17': 0}})
    result = predict(detector, Image.new('RGB', (8, 8)))
    assert result.class_id.tolist() == [0]
    assert result.data['class_name'].tolist() == ['cat']
    assert detector._backend.calls[0][1]['model_labels'] is True


@pytest.mark.parametrize('schema,label,name', [((0, 17), 0, '__background__'), ((0, 17), 1, 'person'),
                                               ((17,), 0, 'person'), ((17,), 1, '__background__')])
def test_keypoint_class_name_layouts(schema, label, name):
    detector = model('keypoints', labels=(label,))
    detector.class_names = ['person']
    detector.model_config.num_classes = 1
    detector.model_config.num_keypoints_per_class = schema
    assert predict(detector, Image.new('RGB', (8, 8))).data['class_name'].tolist() == [name]


def test_sparse_coco_names():
    detector = model(labels=(18, 90))
    detector.class_names = COCO_CLASS_NAMES
    detector.model_config.num_classes = 90
    assert predict(detector, Image.new('RGB', (8, 8)), threshold=0).data['class_name'].tolist() == ['dog', 'toothbrush']


def test_covariance_matches_matrix_inverse_and_invalid_slots():
    precision = np.array([[[.2, .3, -.4], [np.nan, 0, 0]]])
    covariance = precision_cholesky_to_pixel_covariance(precision, [[13, 29]])
    lower = np.array([[np.exp(.2), 0], [.3, np.exp(-.4)]])
    scale = np.diag([29, 13])
    np.testing.assert_allclose(covariance[0, 0], scale @ np.linalg.inv(lower @ lower.T) @ scale, rtol=1e-6)
    assert np.isnan(covariance[0, 1]).all()


def test_torch_input_preserves_float_values():
    torch = pytest.importorskip('torch')
    detector = model()
    tensor = torch.full((3, 10, 20), .12345)
    predict(detector, tensor)
    np.testing.assert_array_equal(detector._backend.calls[0][0], tensor.numpy().transpose(1, 2, 0))
    with pytest.raises(ValueError, match='shape'):
        predict(detector, torch.zeros((10, 20, 3)))


@pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan")])
def test_out_of_range_thresholds_are_rejected_before_inference(threshold):
    """Stricter than upstream, which accepts any threshold: see docs/COMPATIBILITY.md."""
    with pytest.raises(ValueError, match="zero and one"):
        predict(model(), np.zeros((32, 32, 3), np.uint8), threshold=threshold)
