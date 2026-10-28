"""Parity must fail closed; a report must not manufacture agreement from arbitrary logs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parity = _load_script("check_parity")
reports = _load_script("compare_runs")
dumper = _load_script("dump_torch_reference")


def _outputs():
    return {"pred_logits": np.array([[[0.0], [1.0]]]), "pred_boxes": np.zeros((1, 2, 4))}


def _reference(outputs, mode="train"):
    return {f"{mode}_{name}": value.copy() for name, value in outputs.items()}


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_ordered_parity_accepts_measured_tolerance(mode):
    actual = _outputs()
    reference = _reference(actual, mode)
    actual["pred_logits"] = actual["pred_logits"] + 1e-5
    failures, worst = parity.compare_outputs(actual, reference, mode, 2e-4)
    assert failures == 0
    assert 0 < worst < 2e-4


def test_query_permutation_cannot_hide_class_box_mispairing():
    actual = _outputs()
    reference = _reference(actual)
    actual["pred_logits"] = actual["pred_logits"][:, ::-1]
    failures, worst = parity.compare_outputs(actual, reference, "train", 2e-4)
    assert failures == 1
    assert worst == 1.0


@pytest.mark.parametrize("missing_from", ["model", "reference"])
def test_missing_auxiliary_output_fails(missing_from):
    actual = _outputs()
    reference = _reference(actual)
    if missing_from == "model":
        reference["train_aux0_pred_boxes"] = actual["pred_boxes"].copy()
    else:
        actual["aux0_pred_boxes"] = actual["pred_boxes"].copy()
    assert parity.compare_outputs(actual, reference, "train", 2e-4)[0] == 1


def test_empty_comparison_cannot_pass():
    assert parity.compare_outputs({}, {}, "train", 2e-4)[0] == 2


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("side", ["actual", "reference"])
def test_nonfinite_output_cannot_pass(value, side):
    actual = _outputs()
    reference = _reference(actual)
    target = actual["pred_logits"] if side == "actual" else reference["train_pred_logits"]
    target[0, 0, 0] = value
    assert parity.compare_outputs(actual, reference, "train", 2e-4)[0] == 1


def test_shape_mismatch_cannot_broadcast_into_agreement():
    actual = _outputs()
    reference = _reference(actual)
    actual["pred_boxes"] = np.zeros((1, 1, 4))
    assert parity.compare_outputs(actual, reference, "train", 2e-4)[0] == 1


@pytest.mark.parametrize("tol", [np.nan, np.inf, -1.0])
def test_invalid_tolerance_is_rejected(tol):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        parity.compare_outputs(_outputs(), {}, "train", tol)


@pytest.mark.parametrize("fault", [
    None, "detection_aux", "mismatch", "missing_reference", "missing_model", "injected_mismatch",
    "bad_aux_mask", "missing_aux_mask_reference", "missing_aux_mask_model",
    "bad_aux_keypoints", "missing_aux_keypoints_reference", "missing_aux_keypoints_model",
])
def test_cli_exit_status_remains_strict_for_group_13(monkeypatch, fault):
    actual = _outputs()
    reference = _reference(actual)
    if fault == "mismatch":
        actual["pred_logits"] += 100
    elif fault == "missing_reference":
        del reference["train_pred_boxes"]
    elif fault == "missing_model":
        reference["train_aux0_pred_boxes"] = actual["pred_boxes"].copy()

    if fault == "detection_aux":
        actual["aux_outputs"] = [_outputs()]
        reference.update({f"train_aux0_{key}": value.copy() for key, value in actual["aux_outputs"][0].items()})
    if fault and "aux_" in fault:
        name = "pred_masks" if "mask" in fault else "pred_keypoints"
        auxiliary = _outputs()
        auxiliary[name] = np.zeros((1, 2, 3, 3))
        actual["aux_outputs"] = [auxiliary]
        reference.update({f"train_aux0_{key}": value.copy() for key, value in auxiliary.items()})
        if fault.startswith("bad_"):
            auxiliary[name] += 100
        elif fault.endswith("_reference"):
            del reference[f"train_aux0_{name}"]
        elif fault.endswith("_model"):
            del auxiliary[name]

    if fault == "injected_mismatch":
        # Equal proposal scores and a perfect injected forward must not excuse a broken native path.
        reference["topk::train"] = np.array([[[0, 1]]])
        actual["proposal_indices"] = np.array([[[1, 0]]])
        actual["proposal_scores"] = np.ones((1, 1, 2))
        actual["pred_logits"] += 100

    class Bundle(dict):
        @property
        def files(self):
            return list(self)

    bundle = Bundle(reference, **{
        "meta::group_detr": "13", "input_pixel_values": np.zeros((1, 3, 2, 2)),
        "input_pixel_mask": np.zeros((1, 2, 2)),
    })

    class Model:
        def __init__(self, **kwargs):
            pass

        def init(self, *args, **kwargs):
            return {"params": {}}

        def apply(self, *args, **kwargs):
            return _outputs() if "proposal_indices" in kwargs else actual

    monkeypatch.setattr(parity.np, "load", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(parity, "spec_from_meta", lambda meta: None)
    monkeypatch.setattr(parity, "LWDETR", Model)
    monkeypatch.setattr(parity, "convert_state_dict", lambda *args, **kwargs: ({}, [], []))
    monkeypatch.setattr(parity.jnp, "asarray", np.asarray)
    monkeypatch.setattr(parity.jax.random, "key", lambda seed: seed)
    monkeypatch.setattr(parity.sys, "argv", ["check_parity.py", "--modes", "train"])
    assert parity.main() == (0 if fault in {None, "detection_aux"} else 1)


@pytest.mark.parametrize("step_losses", [{}, {"torch_g1": {"total": 1}, "jax_g1": {"total": 1000}}])
def test_report_describes_values_without_asserting_agreement(monkeypatch, step_losses):
    config = dict(
        encoder="test", dec_layers=2, resolution=32, group_detr=13, num_queries=2,
        dataset_dir="example", batch_size=1, lr=1e-4, lr_encoder=1e-4, weight_decay=1e-4,
        lr_vit_layer_decay=0.8, lr_component_decay=0.7, clip_max_norm=0.1,
        ema_decay=0.99, ema_tau=100,
    )
    left = {"config": config, "epochs": [{"seconds": 1, "losses": {"loss": 1}}]}
    right = {"config": config, "epochs": [{"seconds": 2, "losses": {"loss": 1000}}]}
    monkeypatch.setattr(reports, "_load_step_losses", lambda: step_losses)
    rendered = reports._render_markdown(left, right, 1)
    assert "1.0000 (PyTorch) vs 1000.0000 (JAX)" in rendered
    assert "do not establish statistical equivalence" in rendered
    assert "0.0000%" not in rendered
    assert "**exactly**" not in rendered
    assert "track each other closely" not in rendered
    assert "well-understood consequence" not in rendered
    assert "rerun skips" not in rendered


def test_dumper_materializes_auxiliary_mask_and_keypoint_outputs(monkeypatch):
    """A sparse auxiliary mask must reach the bundle alongside all keypoint predictions."""
    from types import SimpleNamespace

    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def detach(self):
            return self

        def numpy(self):
            return self.value

        def __add__(self, other):
            return Tensor(self.value + other.value)

    monkeypatch.setitem(sys.modules, "torch",
                        SimpleNamespace(einsum=lambda pattern, left, right:
                                        Tensor(np.einsum(pattern, left.value, right.value))))
    spatial = np.arange(8).reshape(1, 2, 2, 2)
    queries = np.array([[[1, 2], [3, 4]]])
    bias = np.array(0.5)
    aux = {name: Tensor(value) for name, value in _outputs().items()}
    aux["pred_masks"] = {"spatial_features": Tensor(spatial), "query_features": Tensor(queries), "bias": Tensor(bias)}
    aux["pred_keypoints"] = Tensor(np.arange(12).reshape(1, 2, 2, 3))
    out = {name: Tensor(value) for name, value in _outputs().items()}
    out["aux_outputs"] = [aux]
    bundle = {}
    dumper.record_outputs(bundle, out, "train")
    assert set(bundle) == {
        "train_pred_logits", "train_pred_boxes", "train_aux0_pred_logits",
        "train_aux0_pred_boxes", "train_aux0_pred_masks", "train_aux0_pred_keypoints",
    }
    np.testing.assert_array_equal(bundle["train_aux0_pred_masks"], np.einsum("bchw,bnc->bnhw", spatial, queries) + bias)
    np.testing.assert_array_equal(bundle["train_aux0_pred_keypoints"], aux["pred_keypoints"].value)


@pytest.mark.parametrize("name", ["pred_logits", "pred_boxes", "aux0_pred_keypoints"])
def test_matching_empty_tensors_cannot_pass(name):
    actual = _outputs()
    actual[name] = np.empty((1, 2, 0))
    reference = _reference(actual)
    assert parity.compare_outputs(actual, reference, "train", 2e-4)[0] == 1


def test_selection_equivalence_ignores_tie_order_but_not_different_choices():
    import numpy as np

    scores = np.zeros((1, 1, 6))
    scores[0, 0] = [5.0, 4.0, 4.0, 1.0, 0.5, 0.1]
    reference = np.array([[[0, 1, 2]]])
    tied_other_order = np.array([[[0, 2, 1]]])
    assert parity.compare_selection(reference, tied_other_order, scores, 2e-4)[0] == 0
    worse_choice = np.array([[[0, 1, 3]]])
    failures, gap, differing = parity.compare_selection(reference, worse_choice, scores, 2e-4)
    assert failures == 1 and gap == 3.0 and differing == 2
    assert parity.compare_selection(reference, np.array([[[0, 1]]]), scores, 2e-4)[0] == 1


def test_selection_check_rejects_invalid_indices():
    import numpy as np

    scores = np.zeros((1, 1, 6))
    scores[0, 0] = [5.0, 4.0, 4.0, 1.0, 0.5, 0.1]
    reference = np.array([[[0, 1, 2]]])
    assert parity.compare_selection(reference, np.array([[[-6, 1, 2]]]), scores, 2e-4)[0] == 1
    assert parity.compare_selection(reference, np.array([[[0, 1, 6]]]), scores, 2e-4)[0] == 1
    assert parity.compare_selection(reference, np.array([[[0, 1, 1]]]), scores, 2e-4)[0] == 1
