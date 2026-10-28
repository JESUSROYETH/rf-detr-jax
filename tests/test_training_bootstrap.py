"""Fresh-checkpoint training regressions using tiny models and backend spies.

The backend spy avoids optimizer execution; existing integration tests retain
responsibility for real optimizer/scaler/RNG/native-resume restoration.
"""
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest


def _copy_tree(params):
    import jax
    return jax.tree.map(lambda value: np.array(value, copy=True), params)


def _assert_tree_equal(actual, expected):
    import jax
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(got, want)


def _gaussian_heads(params):
    # Explicit known tiny architecture: decoder plus two independent groups.
    return [params["keypoint_embed"]["layers_2"],
            params["transformer"]["enc_out_keypoint_embed_0"]["layers_2"],
            params["transformer"]["enc_out_keypoint_embed_1"]["layers_2"]]


def _expected_training_source(case, source=None):
    expected = _copy_tree(case.source if source is None else source)
    if case.task == "keypoints":
        for head in _gaussian_heads(expected):
            head["weight"][4:7] = 0
            head["bias"][4:7] = 0
    return expected


@pytest.fixture(params=["detection", "segmentation", "keypoints"])
def bootstrap_case(request, tmp_path, monkeypatch, write_dataset):
    from rfdetr_jax import RFDETR, RFDETRKeypointPreview, RFDETRNano, RFDETRSegNano
    from rfdetr_jax.checkpoints import save_checkpoint
    from rfdetr_jax.training import run

    task = request.param
    root = tmp_path / "dataset"
    write_dataset(root, task)
    cls = {"detection": RFDETRNano, "segmentation": RFDETRSegNano,
           "keypoints": RFDETRKeypointPreview}[task]
    options = dict(device="cpu", resolution=32, positional_encoding_size=2,
                   patch_size=16, num_windows=1, dec_layers=1, group_detr=2,
                   num_queries=2, num_select=2, hidden_dim=32, sa_nheads=2,
                   ca_nheads=2, dim_feedforward=32, out_feature_indexes=[1],
                   num_classes=1, gradient_checkpointing=False)
    if task == "keypoints":
        options["num_keypoints_per_class"] = [2]
    initial = cls(pretrain_weights=None, **options)
    source = _copy_tree(initial.params)
    source["class_embed"]["bias"][:] = [3., 7.]
    if task == "keypoints":
        for index, head in enumerate(_gaussian_heads(source)):
            # Distinct nonzero Gaussian values prevent a constant-zero fixture
            # from hiding a missing train-only reset.
            head["weight"][:] = np.arange(head["weight"].size, dtype=np.float32).reshape(
                head["weight"].shape) + 1 + index * 1000
            head["bias"][:] = np.arange(8, dtype=np.float32) + 20 + index * 10
    checkpoint = tmp_path / "custom-pretrained.npz"
    save_checkpoint(checkpoint, source, initial.spec, {"task": task}, initial.model_config)
    model = RFDETR.from_checkpoint(checkpoint, device="cpu")
    captured = []
    seen = SimpleNamespace(resume_state=None)

    def backend(argv, *, initial_params, model_spec, category_to_label, metadata, **kwargs):
        captured.append(dict(argv=list(argv), params=None if initial_params is None else _copy_tree(initial_params),
                             spec=model_spec, mapping=category_to_label.copy(), metadata=metadata.copy()))
        if initial_params is None:
            assert seen.resume_state is not None
            state = seen.resume_state
        else:
            state = SimpleNamespace(params=initial_params, ema_params=initial_params, step=2)
        return run.TrainingResult(state.params, state, {**metadata, "weights": "raw", "step": 2},
                                  {"epochs": []}, {})

    monkeypatch.setattr(run, "run_training", backend)
    train_options = dict(dataset_dir=root, output_dir=tmp_path / "run", batch_size=1,
                         epochs=1, num_workers=0, multi_scale=False, tensorboard=False,
                         no_precompile=True, no_compile_cache=True, amp_dtype=None)
    return SimpleNamespace(model=model, cls=cls, options=options, root=root, task=task,
                            source=source, source_spec=initial.spec, source_config=initial.model_config,
                            checkpoint=checkpoint, captured=captured, seen=seen,
                            train_options=train_options)


def test_fresh_train_resets_only_keypoint_gaussian_rows_from_untouched_source(bootstrap_case):
    case = bootstrap_case
    _assert_tree_equal(case.model.params, case.source)
    case.model.train(**case.train_options)
    _assert_tree_equal(case.captured[-1]["params"], _expected_training_source(case))
    # Resetting training input must not change the checkpoint's source arrays.
    if case.task == "keypoints":
        for head in _gaussian_heads(case.source):
            assert np.all(head["bias"][4:7] != 0)


def test_fresh_train_rereads_checkpoint_instead_of_current_weights(bootstrap_case):
    import jax

    from rfdetr_jax.checkpoints import save_checkpoint

    case = bootstrap_case
    original_fields = set(case.model.model_config.model_fields_set)
    for iteration in range(2):
        # Poison all live parameters, independently of the original source.
        live = jax.tree.map(lambda value: np.asarray(value) + np.float32(10000), case.model.params)
        case.model._install(case.model.spec, live, case.model._backend.metadata, case.model.model_config)
        case.model.train(**case.train_options)
        _assert_tree_equal(case.captured[-1]["params"], _expected_training_source(case))
        assert case.captured[-1]["mapping"] == {5: 0}
        assert case.captured[-1]["metadata"]["category_names"] == ["custom"]
        assert case.model.model_config.model_fields_set == original_fields
        assert case.model.model_config.pretrain_weights == str(case.checkpoint)
        if iteration == 0:
            # The second run must reopen this same custom filename and observe
            # replacement data, not cache the originally deserialized tensors.
            case.source = jax.tree.map(lambda value: value + np.float32(2), case.source)
            save_checkpoint(case.checkpoint, case.source, case.source_spec,
                            {"task": case.task}, case.source_config)


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_fresh_train_rejects_missing_or_corrupt_source_without_live_fallback(bootstrap_case, failure):
    case = bootstrap_case
    if failure == "missing":
        case.checkpoint.unlink()
    else:
        case.checkpoint.write_bytes(b"not a numpy archive")
    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        case.model.train(**case.train_options)
    assert not case.captured


def test_from_checkpoint_marks_step_free_source_trained_and_forces_its_path(bootstrap_case, tmp_path):
    from rfdetr_jax import RFDETR

    case = bootstrap_case
    loaded = RFDETR.from_checkpoint(case.checkpoint, device="cpu",
                                    pretrain_weights=str(tmp_path / "must-not-open.npz"))
    assert loaded._has_been_trained is True
    assert loaded.model_config.pretrain_weights == str(case.checkpoint)
    _assert_tree_equal(loaded.params, case.source)  # Ordinary reload must not reset Gaussian rows.


def test_ordinary_constructor_starts_untrained_despite_source_step_metadata(bootstrap_case):
    from rfdetr_jax.checkpoints import save_checkpoint

    case = bootstrap_case
    save_checkpoint(case.checkpoint, case.source, case.source_spec,
                    {"task": case.task, "step": 123}, case.source_config)
    loaded = case.cls(pretrain_weights=str(case.checkpoint), **case.options)
    assert loaded._has_been_trained is False
    _assert_tree_equal(loaded.params, case.source)


@pytest.mark.parametrize("resume", [False, True])
def test_repeated_train_warning_precedes_config_validation_even_for_resume(bootstrap_case, resume):
    case = bootstrap_case
    # This assertion is tested separately; set it deliberately here to isolate
    # the warning boundary from from_checkpoint identity behavior.
    case.model._has_been_trained = True
    kwargs = {"unsupported_option": 1}
    if resume:
        kwargs["resume"] = "unused-native.npz"
    with pytest.warns(UserWarning, match="resume"), pytest.raises((TypeError, FileNotFoundError)):
        case.model.train(**kwargs)
    assert not case.captured


@pytest.mark.parametrize("requested_seed", [None, 41])
def test_explicit_native_resume_bypasses_fresh_load_and_gaussian_reset(bootstrap_case, monkeypatch, requested_seed):
    from rfdetr_jax import api_training, checkpoints

    case = bootstrap_case
    state = SimpleNamespace(params=_copy_tree(case.source),
                            ema_params=_copy_tree(case.source), step=12,
                            loss_scale=SimpleNamespace(scale=1024, growth_tracker=3, skipped_steps=1))
    case.seen.resume_state = state
    config = api_training.training_config(**{**case.train_options, "resume": "native.npz", "seed": requested_seed})
    saved = {"run_config": {"postprocess_trace_alpha": .2, "seed": 41, "data_seed": 7},
             "model_metadata": {"model_spec": asdict(case.source_spec),
                                "category_to_label": {5: 0}, "model_options": {}},
             "criterion_config": {"ia_bce_loss": True}, "epoch_log": []}
    monkeypatch.setattr(api_training, "_resume_configuration", lambda *args: (config, saved))

    def forbidden(*args, **kwargs):
        pytest.fail("explicit native resume must bypass fresh pretrained loading/adaptation")

    monkeypatch.setattr(checkpoints, "load_variant", forbidden)
    monkeypatch.setattr(checkpoints, "adapt_parameters", forbidden)
    case.model.train(resume="native.npz")
    assert case.captured[-1]["params"] is None
    argv = case.captured[-1]["argv"]
    assert argv[argv.index("--seed") + 1] == "41"
    assert argv[argv.index("--data-seed") + 1] == "7"
    assert case.model._training_result.state is state
    _assert_tree_equal(case.model.params, state.params)
    # Unlike fresh pretrained training, resumed Gaussian rows remain nonzero.
    if case.task == "keypoints":
        for head in _gaussian_heads(case.model.params):
            assert np.all(np.asarray(head["bias"])[4:7] != 0)


def test_later_larger_aligned_head_recovers_source_rows_not_previously_truncated_rows(
        bootstrap_case, monkeypatch):
    from rfdetr_jax import RFDETR, api_training
    from rfdetr_jax.checkpoints import adapt_parameters, save_checkpoint

    case = bootstrap_case
    source_spec = replace(case.source_spec, num_classes=5)
    source = adapt_parameters(case.source, case.source_spec, source_spec)
    # Tag decoder and both encoder classifier rows independently.
    classifiers = [source["class_embed"], source["transformer"]["enc_out_class_embed_0"],
                   source["transformer"]["enc_out_class_embed_1"]]
    for index, head in enumerate(classifiers):
        head["bias"] = np.arange(6, dtype=np.float32) + index * 10
        head["weight"] = np.broadcast_to(np.arange(6, dtype=np.float32)[:, None] + index * 10,
                                         head["weight"].shape).copy()
    source_config = case.source_config.model_copy(update=asdict(source_spec))
    save_checkpoint(case.checkpoint, source, source_spec, {"task": case.task}, source_config)
    case.model = RFDETR.from_checkpoint(case.checkpoint, device="cpu")
    original_schema = api_training._schema
    target_counts = iter([2, 4])  # Three output rows followed by five.

    def aligned_schema(model, dataset, config):
        spec, mapping = original_schema(model, dataset, config)
        return replace(spec, num_classes=next(target_counts)), mapping

    monkeypatch.setattr(api_training, "_schema", aligned_schema)
    for rows in (3, 5):
        case.model.train(**case.train_options)
        actual = case.captured[-1]["params"]
        heads = [actual["class_embed"], actual["transformer"]["enc_out_class_embed_0"],
                 actual["transformer"]["enc_out_class_embed_1"]]
        for index, head in enumerate(heads):
            np.testing.assert_array_equal(head["bias"], np.arange(rows) + index * 10)
            np.testing.assert_array_equal(head["weight"][:, 0], np.arange(rows) + index * 10)


def test_fresh_loader_uses_effective_spec_and_safe_trust_without_persisting_source_config(monkeypatch):
    from rfdetr_jax import checkpoints
    from rfdetr_jax.models.lwdetr import LWDETRSpec
    from rfdetr_jax.variants import ModelConfig

    spec = LWDETRSpec(num_classes=7, resolution=64, positional_encoding_size=4,
                      compute_dtype="float16", head_dtype="float16")
    config = ModelConfig(pretrain_weights="custom.npz", amp=False, freeze_encoder=True,
                         num_classes=90, resolution=384, model_fields_set={"freeze_encoder"})
    original = config.model_dump()
    params = {"sentinel": np.array([13.], np.float32)}
    captured = []

    def loader(variant, *, trust_checkpoint, **kwargs):
        captured.append((variant, trust_checkpoint, kwargs))
        # Source-derived metadata/config must not become the public config.
        return spec, params, {"category_to_label": {90: 90}}, ModelConfig(
            model_fields_set={"num_classes", "resolution"}, pretrain_weights="different.pth")

    monkeypatch.setattr(checkpoints, "load_variant", loader)
    actual = checkpoints.load_pretrained_training_parameters("RFDETRNano", config, spec)
    assert actual is params
    variant, trust, options = captured[0]
    assert variant == "RFDETRNano" and trust is False
    assert options["pretrain_weights"] == "custom.npz"
    assert options["freeze_encoder"] is True and options["amp"] is False
    assert {name: options[name] for name in asdict(spec)} == asdict(spec)
    assert "model_fields_set" not in options
    assert config.model_dump() == original and config.model_fields_set == {"freeze_encoder"}


@pytest.mark.parametrize("changed", [
    {"num_classes": 2}, {"compute_dtype": "float16"}, {"head_dtype": "bfloat16"},
])
def test_fresh_loader_rejects_any_drift_from_aligned_spec(monkeypatch, changed):
    from rfdetr_jax import checkpoints
    from rfdetr_jax.models.lwdetr import LWDETRSpec
    from rfdetr_jax.variants import ModelConfig

    spec = LWDETRSpec(num_classes=1, compute_dtype="float32", head_dtype="float32")
    config = ModelConfig(pretrain_weights="custom.npz", model_fields_set=set())
    monkeypatch.setattr(checkpoints, "load_variant", lambda *args, **kwargs: (
        replace(spec, **changed), {}, {}, config))
    with pytest.raises(ValueError, match=next(iter(changed))):
        checkpoints.load_pretrained_training_parameters("RFDETRNano", config, spec)
