"""Export delegation contracts and optional real cross-runtime parity."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rfdetr_jax import export as export_module


def test_export_forwards_options_without_changing_live_model(monkeypatch):
    captured = {}

    class Reference:
        def export(self, output_dir='output', *, format='onnx', dynamic_batch=False, notes=None):
            captured.update(output_dir=output_dir, format=format, dynamic_batch=dynamic_batch, notes=notes)
            return Path(output_dir) / 'result.onnx'

    model = object()
    monkeypatch.setattr(export_module, '_upstream_api', lambda: Reference)
    monkeypatch.setattr(export_module, 'build_export_model', lambda current: Reference() if current is model else None)
    result = export_module.export(model, output_dir='chosen', format='onnx', dynamic_batch=True, notes={'run': 7})
    assert result == Path('chosen/result.onnx')
    assert captured == {'output_dir': 'chosen', 'format': 'onnx', 'dynamic_batch': True, 'notes': {'run': 7}}
    with pytest.raises(TypeError, match='unknown'):
        export_module.export(model, unknown=True)


@pytest.fixture(scope='module', params=['detection', 'segmentation', 'keypoints'])
def reconstructed(request):
    torch = pytest.importorskip('torch')
    pytest.importorskip('rfdetr')
    import jax
    import jax.numpy as jnp

    from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec

    task = request.param
    spec = LWDETRSpec(resolution=32, positional_encoding_size=2, patch_size=16, num_windows=1,
                      dec_layers=1, num_queries=2, num_select=2, group_detr=1, num_classes=1,
                      hidden_dim=32, sa_nheads=2, ca_nheads=2, dim_feedforward=32, out_feature_indexes=(1, 2),
                      segmentation_head=task == 'segmentation', use_grouppose_keypoints=task == 'keypoints',
                      num_keypoints_per_class=(2,) if task == 'keypoints' else ())
    model = LWDETR(spec)
    x = np.random.default_rng(12).normal(size=(1, 3, 32, 32)).astype('float32')
    params = model.init(jax.random.key(4), jnp.array(x), None, train=True)['params']
    # Zero-initialized box/keypoint heads and layer scales can hide wrong export paths.
    generator = np.random.default_rng(42)

    def perturb(value):
        array = np.asarray(value)
        return jnp.asarray(generator.normal(0, .03, array.shape), dtype=value.dtype) if not array.any() else value

    params = jax.tree.map(perturb, params)
    facade = SimpleNamespace(spec=spec, params=params, class_names=['object'],
                             model_config=SimpleNamespace(postprocess_trace_alpha=.2), size='tiny')
    reference = export_module.build_export_model(facade)
    with jax.default_matmul_precision('highest'):
        expected = model.apply({'params': params}, jnp.array(x), None, train=False)
    with torch.no_grad():
        actual = reference.model.model(torch.from_numpy(x))
    return facade, reference, x, expected, actual


def test_reconstructed_torch_model_matches_jax_all_task_outputs(reconstructed):
    _, _, _, expected, actual = reconstructed
    for key in ('pred_logits', 'pred_boxes', 'pred_masks', 'pred_keypoints'):
        if key in expected:
            np.testing.assert_allclose(expected[key], actual[key].numpy(), atol=1e-4, rtol=1e-4, err_msg=key)


def test_actual_onnx_artifact_matches_jax_outputs(reconstructed, tmp_path):
    ort = pytest.importorskip('onnxruntime')
    pytest.importorskip('onnx')
    pytest.importorskip('onnx_graphsurgeon')
    pytest.importorskip('onnxsim')
    pytest.importorskip('polygraphy')
    _, reference, image, expected, _ = reconstructed
    path = reference.export(output_dir=str(tmp_path), verbose=False, output_name='contract', notes={'test': 'jax'})
    assert Path(path).is_file()
    import onnx

    properties = {item.key: item.value for item in onnx.load(path).metadata_props}
    assert properties['rfdetr_notes'] == '{"test": "jax"}'
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    values = session.run(None, {'input': image})
    names = ['pred_boxes', 'pred_logits']
    names += ['pred_masks'] if 'pred_masks' in expected else ['pred_keypoints'] if 'pred_keypoints' in expected else []
    for key, value in zip(names, values, strict=True):
        np.testing.assert_allclose(value, expected[key], atol=2e-4, rtol=2e-4, err_msg=key)


def test_roboflow_export_is_sanitized_local_bundle(reconstructed, tmp_path):
    import torch

    _, reference, _, _, _ = reconstructed
    reference.model.args.dataset_dir = '/private/example/dataset'
    reference.model.args.output_dir = '/private/example/output'
    reference.model.args.resume = '/private/example/checkpoint'
    reference.export_for_roboflow(tmp_path)
    assert (tmp_path / 'class_names.txt').read_text() == 'object'
    with torch.serialization.safe_globals([SimpleNamespace]):
        saved = torch.load(tmp_path / 'weights.pt', weights_only=True)
    assert saved['args'].dataset_dir is None
    assert saved['args'].output_dir == 'output' and saved['args'].resume == ''
    assert saved['args'].class_names == ['object']
    assert reference.model.args.dataset_dir == '/private/example/dataset'
    reference.model.model.load_state_dict(saved['model'], strict=True)
