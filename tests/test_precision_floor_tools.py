"""Precision diagnostics must reject absent evidence without running a real model."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "precision_floors", Path(__file__).resolve().parents[1] / "scripts/compare_precision_floors.py",
)
floors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(floors)


class Bundle(dict):
    @property
    def files(self):
        return list(self)


def bundles():
    result = []
    for dtype in (np.float32, np.float64):
        result.append(Bundle({
            "meta::dtype": dtype.__name__, "meta::num_queries": "2",
            "input_pixel_values": np.zeros((1, 3, 2, 2), dtype),
            "input_pixel_mask": np.ones((1, 2, 2), bool),
            "sd::bias": np.ones(2, dtype),
            "topk::eval": np.array([[[0, 1]], [[2, 3]]]),
            "eval_pred_logits": np.zeros((1, 4, 1), dtype),
            "eval_pred_boxes": np.zeros((1, 4, 4), dtype),
        }))
    return result


def outputs(bundle):
    return {key[5:]: value.copy() for key, value in bundle.items() if key.startswith("eval_")}


def test_matching_bundles_and_complete_outputs_pass(capsys):
    left, right = bundles()
    common = floors.validate_bundles(left, right, ["eval"])["eval"]
    assert floors.compare_outputs(outputs(left), left, right, "eval", common, 2) == 0
    assert "WITHIN MEDIAN MARGIN" in capsys.readouterr().out


@pytest.mark.parametrize("key", [
    "input_pixel_values", "input_pixel_mask", "sd::bias", "topk::eval",
    "eval_pred_boxes", "eval_pred_logits",
])
@pytest.mark.parametrize("side", [0, 1])
def test_missing_bundle_evidence_is_rejected(key, side):
    data = bundles()
    del data[side][key]
    with pytest.raises(ValueError):
        floors.validate_bundles(*data, ["eval"])


@pytest.mark.parametrize("key", ["input_pixel_values", "input_pixel_mask", "sd::bias", "meta::num_queries"])
def test_mismatched_reference_context_is_rejected(key):
    left, right = bundles()
    if key.startswith("meta::"):
        right[key] = "3"
    else:
        right[key] = right[key] + 1
    with pytest.raises(ValueError):
        floors.validate_bundles(left, right, ["eval"])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("key", ["input_pixel_values", "sd::bias", "eval_pred_logits"])
def test_nonfinite_reference_evidence_is_rejected(key, value):
    left, right = bundles()
    right[key].flat[0] = value
    with pytest.raises(ValueError):
        floors.validate_bundles(left, right, ["eval"])


@pytest.mark.parametrize("selection", [
    np.array([[[0, 0]], [[2, 3]]]), np.array([[[-1, 1]], [[2, 3]]]),
    np.array([[[0.0, 1.0]], [[2.0, 3.0]]]), np.empty((2, 1, 0), int),
    np.array([[0, 1], [2, 3]]), np.array([[[0, 1]]]),
])
def test_malformed_selections_are_rejected(selection):
    left, right = bundles()
    left["topk::eval"] = selection
    with pytest.raises(ValueError):
        floors.validate_bundles(left, right, ["eval"])


def test_zero_coverage_is_rejected():
    left, right = bundles()
    left["topk::eval"] = left["topk::eval"][..., ::-1]
    with pytest.raises(ValueError, match="no comparable"):
        floors.validate_bundles(left, right, ["eval"])
    assert floors.compare_outputs(outputs(left), left, right, "eval", np.zeros((1, 4), bool), 2) == 1


@pytest.mark.parametrize("name", ["pred_logits", "pred_boxes", "aux0_pred_masks", "aux0_pred_keypoints"])
@pytest.mark.parametrize("missing_from", ["model", "float32", "float64"])
def test_missing_output_on_any_side_fails(name, missing_from):
    left, right = bundles()
    left["eval_" + name] = np.zeros((1, 4, 2), np.float32)
    right["eval_" + name] = np.zeros((1, 4, 2), np.float64)
    named = outputs(left)
    target, key = {"model": (named, name), "float32": (left, "eval_" + name),
                   "float64": (right, "eval_" + name)}[missing_from]
    del target[key]
    assert floors.compare_outputs(named, left, right, "eval", np.ones((1, 4), bool), 2) == 1


@pytest.mark.parametrize("shape", [(1, 1, 4), (1, 4, 0), (2, 4, 4)])
def test_shapes_cannot_broadcast_into_agreement(shape):
    left, right = bundles()
    named = outputs(left)
    named["pred_boxes"] = np.zeros(shape)
    assert floors.compare_outputs(named, left, right, "eval", np.ones((1, 4), bool), 2) == 1


def test_partial_coverage_is_reported_and_nonfinite_excluded_outputs_fail(capsys):
    left, right = bundles()
    left["topk::eval"][1] = left["topk::eval"][1, :, ::-1]
    common = floors.validate_bundles(left, right, ["eval"])["eval"]
    np.testing.assert_array_equal(common, [[True, True, False, False]])
    named = outputs(left)
    assert floors.compare_outputs(named, left, right, "eval", common, 2) == 0
    assert "compared 8/16 elements" in capsys.readouterr().out
    named["pred_boxes"][0, 3, 0] = np.nan
    assert floors.compare_outputs(named, left, right, "eval", common, 2) == 1


@pytest.mark.parametrize("fault", [None, "missing_outputs", "zero_coverage", "model_mismatch"])
def test_cli_status_with_real_npz_bundles(monkeypatch, tmp_path, capsys, fault):
    left, right = bundles()
    if fault == "missing_outputs":
        for data in (left, right):
            del data["eval_pred_boxes"]
            del data["eval_pred_logits"]
    if fault == "zero_coverage":
        left["topk::eval"] = left["topk::eval"][..., ::-1]
    paths = [tmp_path / "float32.npz", tmp_path / "float64.npz"]
    for path, data in zip(paths, (left, right), strict=True):
        np.savez(path, **data)

    class Model:
        def __init__(self, **kwargs):
            assert fault not in {"missing_outputs", "zero_coverage"}, "invalid evidence must fail before model init"

        def init(self, *args, **kwargs):
            return {"params": {}}

        def apply(self, *args, **kwargs):
            named = outputs(right)
            if fault == "model_mismatch":
                named["pred_logits"] += 1
            return named

    monkeypatch.setattr(floors, "spec_from_meta", lambda meta: None)
    monkeypatch.setattr(floors, "LWDETR", Model)
    monkeypatch.setattr(floors, "convert_state_dict", lambda *args, **kwargs: ({}, [], []))
    monkeypatch.setattr(floors.jnp, "asarray", np.asarray)
    monkeypatch.setattr(floors.jax.random, "key", lambda seed: seed)
    monkeypatch.setattr(floors.sys, "argv", [
        "floors", "--float32", str(paths[0]), "--float64", str(paths[1]), "--modes", "eval",
    ])
    assert floors.main() == (0 if fault is None else 1)
    if fault is None:
        assert "Median-error diagnostic only" in capsys.readouterr().out


@pytest.mark.parametrize("options", [
    ["--margin", "nan"], ["--margin", "inf"], ["--margin", "0"], ["--margin", "-1"],
    ["--modes", ""], ["--modes", "unknown"],
])
def test_invalid_options_fail_before_loading_bundles(monkeypatch, options):
    monkeypatch.setattr(floors.sys, "argv", ["floors", "--float32", "a", "--float64", "b", *options])
    with pytest.raises(SystemExit) as error:
        floors.main()
    assert error.value.code == 2
