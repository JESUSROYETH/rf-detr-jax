"""Every reference-check CLI must fail for missing or malformed numerical evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


loss_check = load_script("check_loss_parity")
step_check = load_script("check_step_loss_parity")
compare_named = load_script("parity_utils").compare_named


@pytest.mark.parametrize(("actual", "reference"), [
    ({}, {}),
    ({"loss": 1.0}, {}),
    ({}, {"loss": 1.0}),
    ({"loss": np.ones(2)}, {"loss": np.ones(1)}),
    ({"loss": np.array([])}, {"loss": np.array([])}),
    ({"loss": np.inf}, {"loss": np.inf}),
    ({"loss": np.nan}, {"loss": 1.0}),
])
def test_shared_gate_rejects_invalid_evidence(actual, reference):
    assert compare_named(actual, reference, 1e-4) > 0


def test_shared_gate_preserves_small_finite_tolerance():
    assert compare_named({"loss": 2.00001}, {"loss": 2.0}, 1e-4) == 0


@pytest.mark.parametrize("fault", [None, "all_missing", "one_missing", "extra_reference", "nonfinite"])
def test_loss_cli_fails_when_recorded_losses_are_missing(monkeypatch, tmp_path, fault):
    bundle = {
        "pred_logits": np.zeros((1, 2, 2)), "pred_boxes": np.zeros((1, 2, 4)),
        "enc_pred_logits": np.zeros((1, 2, 2)), "enc_pred_boxes": np.zeros((1, 2, 4)),
        "target_labels": np.zeros((1, 1)), "target_boxes": np.zeros((1, 1, 4)),
        "target_mask": np.zeros((1, 1), dtype=bool),
        "loss::loss_ce": np.array(1.0), "loss::loss_bbox": np.array(2.0),
        "loss::class_error": np.array(99.0),
    }
    if fault == "all_missing":
        bundle = {key: value for key, value in bundle.items() if not key.startswith("loss::")}
    elif fault == "one_missing":
        del bundle["loss::loss_bbox"]
    elif fault == "extra_reference":
        bundle["loss::lost_auxiliary_term"] = np.array(3.0)
    elif fault == "nonfinite":
        bundle["loss::loss_ce"] = np.array(np.inf)
    path = tmp_path / "losses.npz"
    np.savez(path, **bundle)
    monkeypatch.setattr(loss_check, "compute_losses", lambda *args, **kwargs: {"loss_ce": 1., "loss_bbox": 2.})
    monkeypatch.setattr(loss_check.sys, "argv", ["check_loss_parity.py", "--bundle", str(path)])
    assert loss_check.main() == (0 if fault is None else 1)


@pytest.mark.parametrize("reference", [{}, {"loss": 1.0}, {"loss": 1000.0, "total": 1000.0}])
def test_step_comparison_sets_failure_exit_status(monkeypatch, tmp_path, reference):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    monkeypatch.setattr(step_check, "run_jax", lambda *args: {"loss": 1.0, "total": 1.0})
    monkeypatch.setattr(step_check.sys, "argv", ["check_step_loss_parity.py", "--framework", "jax",
                                               "--compare", str(path)])
    assert step_check.main() == 1


def test_step_comparison_accepts_matching_recorded_losses(monkeypatch, tmp_path):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"loss": 1.0, "total": 1.0}))
    monkeypatch.setattr(step_check, "run_jax", lambda *args: {"loss": 1.0, "total": 1.0})
    monkeypatch.setattr(step_check.sys, "argv", ["check_step_loss_parity.py", "--framework", "jax",
                                               "--compare", str(path)])
    assert step_check.main() == 0


@pytest.mark.parametrize("name", ["check_loss_parity", "check_segmentation_parity", "check_keypoint_parity",
                                 "check_keypoint_eval_parity", "check_step_loss_parity"])
def test_every_sibling_rejects_nonfinite_tolerance_before_running(monkeypatch, name):
    module = load_script(name)
    args = [name, "--tol", "inf"] + (["--framework", "jax"] if name == "check_step_loss_parity" else [])
    monkeypatch.setattr(module.sys, "argv", args)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
