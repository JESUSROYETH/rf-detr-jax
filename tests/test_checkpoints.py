"""Variant defaults, trusted loading boundaries, and pretrained parameter preservation."""
from __future__ import annotations

import hashlib
import io
import json
import pickle
import sys
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rfdetr_jax import checkpoints
from rfdetr_jax.convert import flatten_params, save_params_npz
from rfdetr_jax.variants import VARIANT_DEFAULTS, spec_from_config, variant_config


@pytest.mark.parametrize(("name", "resolution", "layers", "patch", "queries"), [
    ("RFDETRBase", 560, 3, 14, 300), ("RFDETRNano", 384, 2, 16, 300),
    ("RFDETRSmall", 512, 3, 16, 300), ("RFDETRMedium", 576, 4, 16, 300),
    ("RFDETRLarge", 704, 4, 16, 300), ("RFDETRLargeDeprecated", 560, 3, 14, 300),
    ("RFDETRSegPreview", 432, 4, 12, 200), ("RFDETRSegNano", 312, 4, 12, 100),
    ("RFDETRSegSmall", 384, 4, 12, 100), ("RFDETRSegMedium", 432, 5, 12, 200),
    ("RFDETRSegLarge", 504, 5, 12, 200), ("RFDETRSegXLarge", 624, 6, 12, 300),
    ("RFDETRSeg2XLarge", 768, 6, 12, 300), ("RFDETRKeypointPreview", 576, 4, 12, 100),
])
def test_pinned_variant_defaults(name, resolution, layers, patch, queries):
    config = variant_config(name)
    spec = spec_from_config(config)
    assert (spec.resolution, spec.dec_layers, spec.patch_size, spec.num_queries) == (resolution, layers, patch, queries)
    assert config.pretrain_weights in checkpoints.OFFICIAL_WEIGHTS
    assert spec.num_classes == 90 and spec.detector_classes == 91
    assert spec.segmentation_head == name.startswith("RFDETRSeg")
    assert spec.use_grouppose_keypoints == (name == "RFDETRKeypointPreview")
    if name == "RFDETRKeypointPreview":
        assert spec.num_keypoints_per_class == (17,) and spec.dual_projector and spec.dual_projector_kp_only


def test_explicit_resolution_and_config_copy_contract():
    nano = variant_config("RFDETRNano", resolution=640)
    assert nano.positional_encoding_size == 40
    assert variant_config("RFDETRBase", resolution=672).positional_encoding_size == 37
    copied = nano.model_copy(update={"num_classes": 4})
    assert copied.num_classes == 4 and nano.num_classes == 90
    assert copied.model_fields_set == {"resolution", "num_classes"}
    with pytest.raises(TypeError, match="resoluton"):
        variant_config("RFDETRNano", resoluton=640)


def test_registered_download_is_atomic_checked_and_cached(monkeypatch, tmp_path):
    payload = b"synthetic checkpoint"
    monkeypatch.setenv("RF_HOME", str(tmp_path))
    monkeypatch.setitem(checkpoints.OFFICIAL_WEIGHTS, "fixture.pt", ("https://official.invalid/fixture", hashlib.md5(
        payload).hexdigest()))
    calls = []

    def download(url, timeout):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(checkpoints.urllib.request, "urlopen", download)
    path = checkpoints.resolve_weights("fixture.pt")
    assert path.read_bytes() == payload
    assert checkpoints.resolve_weights("fixture.pt") == path and len(calls) == 1
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        checkpoints.resolve_weights("fixture.pt")
    assert len(calls) == 1


def test_corrupt_download_never_becomes_cached_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("RF_HOME", str(tmp_path))
    monkeypatch.setitem(checkpoints.OFFICIAL_WEIGHTS, "fixture.pt", ("https://official.invalid/fixture", "bad"))
    monkeypatch.setattr(checkpoints.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"broken"))
    with pytest.raises(ValueError, match="checksum"):
        checkpoints.resolve_weights("fixture.pt")
    assert list(tmp_path.iterdir()) == []


def test_missing_explicit_checkpoint_never_downloads_or_initializes(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoints.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("downloaded"))
    with pytest.raises(FileNotFoundError):
        checkpoints.load_variant("RFDETRNano", pretrain_weights=tmp_path / "missing.pth")


@pytest.mark.parametrize("trusted", [False, True])
def test_safe_torch_loading_never_automatically_enables_pickle(monkeypatch, trusted):
    calls = []

    def load(path, **kwargs):
        calls.append(kwargs["weights_only"])
        if kwargs["weights_only"]:
            raise pickle.UnpicklingError("custom global")
        return {"model": {"tensor": 1}}

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.10.0", load=load,
                        serialization=SimpleNamespace(safe_globals=lambda values: nullcontext())))
    if trusted:
        with pytest.warns(UserWarning, match="arbitrary Python"):
            checkpoints.safe_torch_load("checkpoint.pth", trust_checkpoint=True)
        assert calls == [True, False]
    else:
        with pytest.raises(ValueError, match="trust_checkpoint"):
            checkpoints.safe_torch_load("checkpoint.pth")
        assert calls == [True]


@pytest.mark.parametrize("variant", ["RFDETRNano", "RFDETRSegNano", "RFDETRKeypointPreview"])
def test_explicit_random_then_npz_reload_and_upstream_head_resize(tmp_path, variant):
    kwargs = dict(pretrain_weights=None, resolution=32, patch_size=16, positional_encoding_size=2,
                  num_windows=1, dec_layers=1, group_detr=1, num_queries=2, num_select=2, hidden_dim=32,
                  sa_nheads=2, ca_nheads=2, dim_feedforward=32, out_feature_indexes=[1, 2], num_classes=1)
    if variant == "RFDETRKeypointPreview":
        kwargs["num_keypoints_per_class"] = [2]
    spec, params, metadata, config = checkpoints.load_variant(variant, **kwargs)
    assert config.pretrain_weights is None
    path = tmp_path / "model.npz"
    save_params_npz(path, params)
    path.with_suffix(".npz.json").write_text(json.dumps(metadata))
    name, options = checkpoints.inspect_checkpoint(path)
    assert name == variant and options["num_classes"] == 1
    restored_spec, restored, restored_metadata, _ = checkpoints.load_variant(name, **options)
    assert restored_spec == spec and restored_metadata["model_name"] == variant
    for key, value in flatten_params(params).items():
        np.testing.assert_array_equal(value, flatten_params(restored)[key])
    larger = replace(spec, num_classes=4)
    if spec.use_grouppose_keypoints:
        larger = replace(larger, num_keypoints_per_class=(3,))
    adapted = checkpoints.adapt_parameters(params, spec, larger)
    np.testing.assert_array_equal(adapted["class_embed"]["weight"][:2], params["class_embed"]["weight"])
    np.testing.assert_array_equal(adapted["class_embed"]["weight"][2:4], params["class_embed"]["weight"])
    np.testing.assert_array_equal(adapted["backbone_0"]["encoder"]["embeddings"]["cls_token"],
                                  params["backbone_0"]["encoder"]["embeddings"]["cls_token"])
    with pytest.raises(ValueError, match="architecture"):
        checkpoints.adapt_parameters(params, spec, replace(spec, hidden_dim=64))


def test_grouped_query_shrink_preserves_each_groups_rows(monkeypatch):
    spec = spec_from_config(variant_config("RFDETRNano", num_queries=2, group_detr=2))
    params = {"query_feat": np.arange(24).reshape(12, 2), "refpoint_embed": np.arange(48).reshape(12, 4),
              "transformer": {"enc_output_0": {"weight": np.ones((2, 2))},
                              "enc_output_2": {"weight": np.ones((2, 2))},
                              "enc_output_norm_2": {"weight": np.ones(2)}}}
    changed = checkpoints._adapt_parameters(params, spec, source_queries=4, source_groups=3)
    np.testing.assert_array_equal(changed["query_feat"], params["query_feat"][[0, 1, 4, 5]])
    assert set(changed["transformer"]) == {"enc_output_0"}
    with pytest.raises(ValueError, match="expansion"):
        checkpoints._adapt_parameters(params, replace(spec, num_queries=5), source_queries=4, source_groups=3)


def test_registry_and_config_defaults_do_not_share_mutable_lists():
    first = variant_config("RFDETRNano")
    first.out_feature_indexes.append(0)
    assert variant_config("RFDETRNano").out_feature_indexes == [3, 6, 9, 12]
    assert VARIANT_DEFAULTS["RFDETRNano"]["out_feature_indexes"] == [3, 6, 9, 12]


@pytest.mark.parametrize("variant", ["RFDETRNano", "RFDETRSegNano", "RFDETRKeypointPreview"])
def test_safe_upstream_checkpoint_save_and_reload(tmp_path, variant):
    pytest.importorskip("torch")
    kwargs = dict(pretrain_weights=None, resolution=32, patch_size=16, positional_encoding_size=2,
                  num_windows=1, dec_layers=1, group_detr=1, num_queries=2, num_select=2, hidden_dim=32,
                  sa_nheads=2, ca_nheads=2, dim_feedforward=32, out_feature_indexes=[1, 2], num_classes=1,
                  postprocess_trace_alpha=.7, amp=False, freeze_encoder=True,
                  fused_optimizer=False, ia_bce_loss=False)
    if variant == "RFDETRKeypointPreview":
        kwargs["num_keypoints_per_class"] = [0, 2]
    spec, params, metadata, config = checkpoints.load_variant(variant, **kwargs)
    metadata["pretrain_weights"] = "/private/local/source.pth"
    metadata["class_names"] = ["object"]
    for suffix in (".npz", ".pth"):
        path = tmp_path / ("saved" + suffix)
        checkpoints.save_checkpoint(path, params, spec, metadata, config)
        name, options = checkpoints.inspect_checkpoint(path)
        restored_spec, restored, restored_metadata, restored_config = checkpoints.load_variant(name, **options)
        assert restored_spec == spec
        assert restored_config.postprocess_trace_alpha == .7
        assert restored_metadata["postprocess_trace_alpha"] == .7
        for key in ("amp", "freeze_encoder", "fused_optimizer", "ia_bce_loss"):
            assert getattr(restored_config, key) == kwargs[key]
            assert restored_metadata["model_options"][key] == kwargs[key]
        # Named construction from a local checkpoint does not call inspect_checkpoint.
        direct_options = {**kwargs, "pretrain_weights": str(path)}
        overrides = dict(postprocess_trace_alpha=0., amp=True, freeze_encoder=False,
                         fused_optimizer=True, ia_bce_loss=True)
        for key in overrides:
            direct_options.pop(key)
        direct = checkpoints.load_variant(variant, **direct_options)[3]
        overridden = checkpoints.load_variant(variant, **direct_options, **overrides)[3]
        for key, value in overrides.items():
            assert getattr(direct, key) == kwargs[key]
            assert getattr(overridden, key) == value
        assert restored_metadata["category_to_label"] == metadata["category_to_label"] or {
            int(k): v for k, v in restored_metadata["category_to_label"].items()} == metadata["category_to_label"]
        for key, value in flatten_params(params).items():
            np.testing.assert_array_equal(value, flatten_params(restored)[key])
    raw = checkpoints.safe_torch_load(tmp_path / "saved.pth")
    assert raw["model_config"]["pretrain_weights"] is None
    assert "pretrain_weights" not in raw["metadata"]
    assert raw["args"]["dataset_dir"] is None
    assert raw["args"]["class_names"] == ["object"]
    state = raw["model"]
    assert tuple(state["_kp_active_mask"].shape) == (len(spec.num_keypoints_per_class),
                                                   max(spec.num_keypoints_per_class, default=0))
    if spec.use_grouppose_keypoints:
        assert tuple(state["transformer.decoder.keypoint_class_mask"].shape) == (3, 3)
    import torch

    lightning = tmp_path / "lightning.ckpt"
    torch.save({"state_dict": {"model." + key: value for key, value in state.items()},
                "hyper_parameters": {"model_config": raw["model_config"]}}, lightning)
    name, options = checkpoints.inspect_checkpoint(lightning)
    lightning_spec, lightning_params, _, lightning_config = checkpoints.load_variant(name, **options)
    assert lightning_spec == spec
    assert lightning_config.postprocess_trace_alpha == .7
    assert lightning_config.amp is False and lightning_config.freeze_encoder is True
    for key, value in flatten_params(params).items():
        np.testing.assert_array_equal(value, flatten_params(lightning_params)[key])


def test_reverse_torch_paths_expand_every_consecutive_module_index():
    pytest.importorskip("torch")
    spec = spec_from_config(variant_config("RFDETRNano"))
    params = {"backbone_0": {"projector": {"stages_0_0": {"weight": np.ones((2, 2))}},
                              "encoder": {"encoder": {"layer_0": {"weight": np.ones(2)}}}},
              "query_feat": np.ones((2, 2))}
    state = checkpoints.torch_state_dict(params, spec)
    assert "backbone.0.projector.stages.0.0.weight" in state
    assert "backbone.0.encoder.encoder.encoder.layer.0.weight" in state
    assert "query_feat.weight" in state


def test_native_checkpoint_without_named_variant_retains_exact_custom_geometry(tmp_path):
    spec = spec_from_config(variant_config("RFDETRSegNano", resolution=48, patch_size=16,
                                          num_windows=1, dec_layers=1, num_classes=2))
    path = tmp_path / "custom.npz"
    path.touch()
    path.with_suffix(".npz.json").write_text(json.dumps({"model_spec": spec.__dict__}))
    name, options = checkpoints.inspect_checkpoint(path)
    assert name == "RFDETRSegNano"
    assert options["resolution"] == 48 and options["dec_layers"] == 1 and options["num_classes"] == 2


def test_native_resume_payload_is_safe_bytes_and_preserves_model(tmp_path):
    torch = pytest.importorskip("torch")
    path, native = tmp_path / "resume.pth", tmp_path / "native.npz"
    torch.save({"model": {"weight": torch.tensor([3.0])}}, path)
    np.savez(native, optimizer=np.arange(10))
    expected = native.read_bytes()
    checkpoints.embed_native_resume(path, native)
    restored = checkpoints.safe_torch_load(path)
    assert restored["model"]["weight"].tolist() == [3.0]
    payload = restored["rfdetr_jax_training_state"]
    assert payload.dtype == torch.uint8 and payload.ndim == 1
    assert payload.numpy().tobytes() == expected
