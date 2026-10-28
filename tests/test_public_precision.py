"""Public precision resolution and its real checkpointed training boundary."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from rfdetr_jax.api_training import training_config
from rfdetr_jax.training.precision import _cuda_capability, resolve_training_precision


@pytest.mark.parametrize("field", ["amp", "freeze_encoder", "fused_optimizer", "ia_bce_loss"])
@pytest.mark.parametrize("value", [False, True, 0, 1, 0., 1., "false", "true", "OFF", "ON", "no", "yes",
                                  "f", "t", "n", "y", "0", "1", b"false", b"true", np.bool_(False)])
def test_model_booleans_match_published_configuration(field, value):
    pytest.importorskip("rfdetr")
    from rfdetr.config import RFDETRNanoConfig

    from rfdetr_jax.variants import variant_config

    expected = getattr(RFDETRNanoConfig(**{field: value}), field)
    actual = getattr(variant_config("RFDETRNano", **{field: value}), field)
    assert actual is expected


@pytest.mark.parametrize("value", [None, 2, "invalid", " false ", [], {}, b"\xff"])
def test_invalid_model_booleans_fail_before_weight_loading(value):
    from rfdetr_jax.variants import variant_config

    with pytest.raises(ValueError, match="amp must be a boolean"):
        variant_config("RFDETRNano", amp=value)


def test_false_model_strings_do_not_enable_precision_or_unsupported_features():
    from rfdetr_jax.variants import variant_config

    config = variant_config("RFDETRNano", amp="false", compile="false", backbone_lora="false",
                            freeze_encoder="false", fused_optimizer="false", ia_bce_loss="false",
                            segmentation_head="false", use_grouppose_keypoints="false")
    for field in ("amp", "compile", "backbone_lora", "freeze_encoder", "fused_optimizer", "ia_bce_loss",
                  "segmentation_head", "use_grouppose_keypoints"):
        assert getattr(config, field) is False
    with pytest.warns(FutureWarning):
        resolved = resolve_training_precision("auto", config.amp, SimpleNamespace(platform="gpu"))
    assert resolved["compute_dtype"] == "float32"


@pytest.mark.parametrize("capability", ["8.9", 89, (8, 9), [8, 9]])
def test_selected_jax_device_capability_formats(capability):
    assert _cuda_capability(SimpleNamespace(compute_capability=capability)) == (8, 9)


def test_capability_fallback_uses_selected_local_device(monkeypatch):
    from jaxlib import gpu_triton

    selected = []
    monkeypatch.setattr(gpu_triton, "get_compute_capability", lambda index: selected.append(index) or 89,
                        raising=False)
    assert _cuda_capability(SimpleNamespace(local_hardware_id=3)) == (8, 9)
    assert selected == [3]


@pytest.mark.parametrize("requested", [None, "auto", "bf16", "fp16"])
@pytest.mark.parametrize("legacy", [True, False])
def test_cpu_keeps_float32_pixels_heads_and_master_state(requested, legacy):
    context = pytest.warns(FutureWarning) if requested == "auto" and not legacy else nullcontext()
    with context:
        result = resolve_training_precision(requested, legacy, SimpleNamespace(platform="cpu"))
    assert result["compute_dtype"] == result["head_dtype"] == result["pixel_dtype"] == "float32"
    assert not result["loss_scaling"]


@pytest.mark.parametrize("requested,legacy,capability,expected", [
    (None, True, "8.9", "32-true"), ("auto", False, "8.9", "32-true"),
    ("auto", True, "8.9", "bf16-mixed"), ("auto", True, "7.5", "16-mixed"),
    ("fp16", False, "8.9", "16-mixed"), ("bf16", False, "8.9", "bf16-mixed"),
    ("bf16", True, "7.5", "bf16-mixed"),
])
def test_resolver_matches_actual_published_trainer_without_cuda(requested, legacy, capability, expected, monkeypatch):
    pytest.importorskip("rfdetr")
    pytest.importorskip("pytorch_lightning")
    pytest.importorskip("torchmetrics", minversion="1.8.2")
    pytest.importorskip("faster_coco_eval", minversion="1.7.2")
    from rfdetr.config import RFDETRNanoConfig, TrainConfig
    from rfdetr.training import trainer

    # Execute the published resolver, but replace the final Trainer constructor:
    # mocked capability probes must never initialize a real CUDA context.
    monkeypatch.setattr(trainer, "Trainer", lambda **kwargs: kwargs)
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer, "_cuda_native_bf16_on_devices", lambda devices: float(capability) >= 8)
    monkeypatch.setattr(trainer, "_cuda_bf16_supported_on_devices", lambda devices: True)
    reference = trainer.build_trainer(
        TrainConfig(dataset_dir=".", tensorboard=False, amp_dtype=requested),
        RFDETRNanoConfig(pretrain_weights=None, amp=legacy), accelerator="gpu", devices=[0],
        include_training_callbacks=False,
    )
    device = SimpleNamespace(platform="gpu", compute_capability=capability)
    actual = resolve_training_precision(requested, legacy, device)
    assert reference["precision"] == expected
    assert actual["compute_dtype"] == {"32-true": "float32", "16-mixed": "float16",
                                       "bf16-mixed": "bfloat16"}[expected]
    assert actual["head_dtype"] == ("float32" if expected == "32-true" else "activation")
    assert actual["pixel_dtype"] == "float32"
    assert actual["loss_scaling"] == (expected == "16-mixed")


@pytest.mark.parametrize("value", ["invalid", True, 12])
def test_unknown_precision_matches_published_warning_fallback(value):
    with pytest.warns(UserWarning, match="Unknown amp_dtype"):
        assert training_config(amp_dtype=value).amp_dtype == "auto"


def test_fp8_is_rejected_before_constructing_training_state():
    with pytest.raises(NotImplementedError, match="FP8"):
        resolve_training_precision("fp8", False, SimpleNamespace(platform="gpu", compute_capability="8.9"))


@pytest.mark.parametrize("constructor_options", [{}, {"amp": False}, {"amp": True}],
                         ids=["inherit-saved-amp", "explicit-compatible-amp", "explicit-incompatible-amp"])
def test_fresh_constructor_resume_inherits_saved_legacy_amp_policy(tmp_path, monkeypatch, constructor_options):
    """Exercise public resume wiring without initializing a model or running a step."""
    from collections import defaultdict
    from dataclasses import asdict

    import jax

    from rfdetr_jax import api_training, detr
    from rfdetr_jax.models.lwdetr import LWDETRSpec
    from rfdetr_jax.training import run
    from rfdetr_jax.variants import variant_config

    spec = LWDETRSpec()
    saved = {"model_metadata": {"model_spec": asdict(spec), "category_to_label": {1: 1},
                                 "model_options": {"amp": False}},
             "run_config": {"seed": 42, "data_seed": 7}, "criterion_config": {"ia_bce_loss": True}}
    config = training_config(dataset_dir=tmp_path, dataset_file="coco", output_dir=tmp_path / "output",
                             tensorboard=False, resume="checkpoint.pth")
    monkeypatch.setattr(api_training, "_resume_configuration", lambda kwargs, task: (config, saved))
    monkeypatch.setattr(api_training, "_dataset_format", lambda root, configured: configured)
    dataset = SimpleNamespace(category_ids=[1], category_names=["person"],
                              categories=[{"id": 1, "name": "person"}])
    monkeypatch.setattr(api_training, "_prepare_split", lambda *args, **kwargs: (None, None, dataset, 1))
    monkeypatch.setattr(detr, "resolve_device", lambda value: SimpleNamespace(platform="gpu", compute_capability="8.9"))
    monkeypatch.setattr(jax, "default_device", lambda device: nullcontext())
    installed = []
    model = SimpleNamespace(spec=spec, params={}, callbacks=defaultdict(list),
                            model_config=variant_config("RFDETRNano", **constructor_options),
                            _install=lambda *args: installed.append(args))

    def train_backend(argv, **kwargs):
        assert kwargs["model_spec"].compute_dtype == "float32"
        assert kwargs["metadata"]["model_options"]["amp"] is False
        assert kwargs["metadata"]["training_precision"]["loss_scaling"] is False
        return SimpleNamespace(params={}, state=SimpleNamespace(params={}), metadata=kwargs["metadata"], log={})

    monkeypatch.setattr(run, "run_training", train_backend)
    if constructor_options.get("amp") is True:
        with pytest.raises(ValueError, match="resume must preserve resolved training precision"):
            api_training.train(model, resume="checkpoint.pth")
        assert not installed
    else:
        with pytest.warns(FutureWarning, match="ModelConfig.amp is deprecated"):
            api_training.train(model, resume="checkpoint.pth")
        assert len(installed) == 1
        assert installed[0][3].amp is False


@pytest.mark.parametrize("options,expected", [({}, "auto"), ({"amp_dtype": "fp16"}, "fp16"),
                                             ({"amp_dtype": None}, None)])
def test_evaluate_precision_is_per_call_not_inherited_training_state(
        tmp_path, monkeypatch, options, expected, write_dataset):
    from rfdetr_jax import api_training
    from rfdetr_jax.models.lwdetr import LWDETRSpec
    from rfdetr_jax.training import precision

    write_dataset(tmp_path, "detection")
    model = SimpleNamespace(spec=LWDETRSpec(num_classes=1), params={},
                            model_config=SimpleNamespace(device="cpu", amp=True),
                            _backend=SimpleNamespace(metadata={"public_training_config": {"amp_dtype": "fp16"}}),
                            _last_train_kwargs={"amp_dtype": "fp16"})

    class Resolved(Exception):
        pass

    def resolve(requested, *args):
        assert requested == expected
        raise Resolved

    monkeypatch.setattr(precision, "resolve_training_precision", resolve)
    with pytest.raises(Resolved):
        api_training.evaluate(model, dataset_dir=tmp_path, **options)


def test_published_evaluate_builds_fresh_precision_configuration():
    pytest.importorskip("pytorch_lightning")
    from rfdetr.config import RFDETRNanoConfig, TrainConfig
    from rfdetr.detr import _prepare_run_config

    detector = SimpleNamespace(model_config=RFDETRNanoConfig(pretrain_weights=None),
                               get_train_config=lambda **kwargs: TrainConfig(**kwargs),
                               _last_train_kwargs={"amp_dtype": "fp16"})
    config, _, _ = _prepare_run_config(detector, for_eval=True, dataset_dir=".", batch_size=1)
    assert config.amp_dtype == "auto"


@pytest.mark.parametrize("task", ["detection", "segmentation", "keypoints"])
def test_mixed_training_checkpoint_keeps_public_prediction_float32(tmp_path, task):
    from dataclasses import replace

    import jax.numpy as jnp

    from rfdetr_jax import RFDETR, RFDETRKeypointPreview, RFDETRNano, RFDETRSegNano

    cls = {"detection": RFDETRNano, "segmentation": RFDETRSegNano, "keypoints": RFDETRKeypointPreview}[task]
    options = dict(pretrain_weights=None, device="cpu", resolution=32, patch_size=16,
                   positional_encoding_size=2, num_windows=1, dec_layers=1, group_detr=1,
                   num_queries=2, num_select=2, hidden_dim=32, sa_nheads=2, ca_nheads=2,
                   dim_feedforward=32, out_feature_indexes=[1], num_classes=1,
                   amp="false", freeze_encoder="true", fused_optimizer="false", ia_bce_loss="false")
    if task == "keypoints":
        options["num_keypoints_per_class"] = [2]
    model = cls(**options)
    pixels = jnp.zeros((1, 3, 32, 32), jnp.float32)
    expected = model._backend._forward(pixels)
    mixed_spec = replace(model.spec, compute_dtype="float16", head_dtype="activation")
    metadata = {**model._backend.metadata, "training_precision": {"requested": "fp16", "compute_dtype": "float16"}}
    model._install(mixed_spec, model.params, metadata, model.model_config)
    candidates = [model]
    for suffix in ("npz", "pth"):
        checkpoint = tmp_path / f"mixed-trained.{suffix}"
        model.save(checkpoint)
        candidates.append(RFDETR.from_checkpoint(checkpoint, device="cpu"))
    for candidate in candidates:
        for field, expected_bool in {"amp": False, "freeze_encoder": True, "fused_optimizer": False,
                                     "ia_bce_loss": False}.items():
            assert getattr(candidate.model_config, field) is expected_bool
            assert candidate._backend.metadata["model_options"][field] is expected_bool
        assert candidate.spec.compute_dtype == "float16"
        assert candidate._backend.model.spec.compute_dtype == "float32"
        actual = candidate._backend._forward(pixels)
        for key in ("pred_logits", "pred_boxes", "pred_masks", "pred_keypoints"):
            if key in expected:
                assert actual[key].dtype == jnp.float32
                np.testing.assert_array_equal(actual[key], expected[key])
        candidate.inference(compile=False, dtype="bfloat16")
        assert candidate._optimized_dtype == jnp.dtype("bfloat16")
        candidate.remove_optimized_model()
        np.testing.assert_array_equal(candidate._backend._forward(pixels)["pred_logits"], expected["pred_logits"])


@pytest.mark.parametrize("task", ["detection", "segmentation", "keypoints"])
def test_real_public_cpu_training_persists_requested_and_resolved_precision(tmp_path, task, write_dataset):
    import jax

    from rfdetr_jax import RFDETRKeypointPreview, RFDETRNano, RFDETRSegNano
    from rfdetr_jax.training.checkpoint import checkpoint_metadata

    root = tmp_path / "data"
    write_dataset(root, task)
    cls = {"detection": RFDETRNano, "segmentation": RFDETRSegNano, "keypoints": RFDETRKeypointPreview}[task]
    options = dict(pretrain_weights=None, device="cpu", resolution=32, patch_size=16,
                   positional_encoding_size=2, num_windows=1, dec_layers=1, group_detr=1,
                   num_queries=2, num_select=2, hidden_dim=32, sa_nheads=2, ca_nheads=2,
                   dim_feedforward=32, out_feature_indexes=[1], num_classes=1)
    if task == "keypoints":
        options["num_keypoints_per_class"] = [2]
    model = cls(**options)
    model.train(dataset_dir=root, output_dir=tmp_path / "run", epochs=1, batch_size=1,
                multi_scale=False, num_workers=0, tensorboard=False, no_precompile=True,
                no_compile_cache=True, amp_dtype="fp16", seed=0,
                keypoint_oks_sigmas=[.1, .1] if task == "keypoints" else None)
    state = model._training_result.state
    assert model.spec.compute_dtype == model.model_config.compute_dtype == "float32"
    assert model.spec.head_dtype == "float32" and state.loss_scale is None
    assert all(np.asarray(value).dtype == np.float32 for value in jax.tree.leaves(state.params))
    assert all(np.asarray(value).dtype == np.float32 for value in jax.tree.leaves(state.ema_params))
    for raw, average in zip(jax.tree.leaves(state.params), jax.tree.leaves(state.ema_params), strict=True):
        np.testing.assert_array_equal(raw, average)  # Published EMA copies the first attempted update.
    saved = checkpoint_metadata(model._training_result.paths["checkpoint"])
    assert saved["model_metadata"]["training_precision"]["requested"] == "fp16"
    assert saved["model_metadata"]["training_precision"]["compute_dtype"] == "float32"
    assert saved["runtime_config"]["pixel_dtype"] == "float32"
    assert saved["run_config"]["requested_amp_dtype"] == "fp16"
    assert saved["run_config"]["resolved_compute_dtype"] == "float32"
    assert saved["run_config"]["ema_first_update_copy"] is True
    assert saved["run_config"]["keypoint_box_normalized_accumulation"] is (task == "keypoints")
    assert saved["run_config"]["automatic_accumulation_normalization"] is (task != "keypoints")
