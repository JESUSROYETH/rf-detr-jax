# Contributing

The supported task workflows are COCO detection, instance segmentation and
keypoints in JAX/Flax. Public API compatibility with upstream is a separate,
unmet qualification target; follow docs/COMPATIBILITY.md when changing it. For a bug, include a small input
that demonstrates the wrong behavior, its expected result, and a regression test.

## Local checks

Install the checkout with `python -m pip install -e '.[dev]'` in a Python 3.12
virtual environment, then run:

```bash
make gate
python -m build
```

`make gate` uses CPU JAX and bounds common thread pools. It does not download
weights, datasets or upstream code. The test suite contains JIT compilation and
can take several minutes. For a focused edit, start with the relevant test file;
run the full gate before treating a release candidate as verified.

The package includes its model config JSON files. Check a built wheel in a
separate environment and outside the repository: installed commands must not
rely on `scripts/`, an editable install, or an ignored `upstream/` directory.
CI builds the wheel and source distribution and exercises the installed CLI.

The dependency-floor job installs the direct runtime versions in
`scripts/constraints-min.txt` and exercises the installed public API outside the
checkout, including all three tiny training/reload workflows. The constraints
target Linux/Python 3.12; transitive dependencies are resolver-selected. To
reproduce it, install CPU Torch 2.10.0 first, then install
`python -m pip install -c scripts/constraints-min.txt '.[dev]'` in a fresh
environment. This is separate from testing a current unconstrained resolver.
OpenCV uses the same `opencv-python` distribution required by Supervision;
installing both OpenCV distributions can overwrite their shared `cv2` files.

## Upstream parity

The source parity reference is RF-DETR commit
`6674d8581b7694baf567b993a4f32000a6f4f4c2`.
Keep the PyTorch environment separate from JAX when their accelerator stacks
conflict. To create it explicitly:

```bash
git clone https://github.com/roboflow/rf-detr.git upstream
git -C upstream checkout 6674d8581b7694baf567b993a4f32000a6f4f4c2
python3 -m venv .venv-torch
.venv-torch/bin/python -m pip install -e './upstream[train]'
```

Upstream dependency resolution can change; retain an environment snapshot with
new reference results. Use the checkpoint and variant intended for the test:

```bash
.venv-torch/bin/python scripts/dump_torch_reference.py \
  --variant nano --out artifacts/torch_ref_nano.npz \
  --checkpoint weights/rf-detr-nano.pth --perturb-zeros
python scripts/check_parity.py --bundle artifacts/torch_ref_nano.npz
```

The checker requires ordered outputs from the port's unmodified forward pass to
match the reference within the requested tolerance. When the dumper records
upstream's two-stage proposal selection, the checker also compares selection scores
and runs an injected-selection forward to help locate differences. Neither
score-equivalent choices nor matching injected outputs cancel a native-forward
failure.

Add `--dtype float64` to the dumper for a double-precision reference; the checker
then uses float64 parameters and activations. This is a separate precision mode,
not a replacement for checking the production precision. Some upstream operations
still use float32, and floating-point execution remains backend-dependent.

With matching float32/float64 bundles, `scripts/compare_precision_floors.py`
reports a median-error diagnostic on groups with unchanged upstream selection.
It reports coverage and maximum errors, and fails on missing or invalid evidence.
Exit zero means the compared outputs meet the median margin only: it does not
bound worst-case error, cover excluded groups, establish native-selection parity,
or demonstrate equal error distributions. Keep all experimental results and
cross-hardware comparisons in ignored `artifacts/`.

The published public-API baseline is separately pinned to RF-DETR 1.11.2 in
`docs/COMPATIBILITY.md`. Use a separate environment with that release and its
`train` extra for callback/metric comparisons; an older source checkout is not a
substitute for the published callback. Install the tested CPU Torch 2.10 and
Torchvision 0.25 wheels before resolving the reference dependencies. Optional
reference tests that skip are not evidence of parity.

In that separate Python 3.12 environment, the public reference CI recipe is:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,reference,onnx]' 'rfdetr[train]==1.11.2' 'torchmetrics[detection]==1.8.2'
python -m pip check
python scripts/run_reference_tests.py tests/test_public_evaluation.py tests/test_f1_metrics.py tests/test_export.py tests/test_precision_reference.py tests/test_public_precision.py tests/test_keypoint_accumulation.py tests/test_evaluation_lifecycle_reference.py tests/test_pretrained_bootstrap_reference.py -q
```

The strict runner fails if any selected check skips. The export tests build and
execute actual static ONNX artifacts for all three tasks; they do not certify
other formats, dynamic shapes or deployment hardware.

Mixed-precision reference tests use actual published model forwards, head
gradients and Lightning overflow handling. Their default device is CPU. CUDA
comparisons additionally require matching CUDA builds of Torch and JAX; set
`RFDETR_PRECISION_REFERENCE_DEVICE=cuda` and `JAX_PLATFORMS=cuda,cpu` explicitly.
The CPU backend remains needed for SciPy Hungarian callbacks. Run accelerator
jobs serially on shared hardware. The CUDA training tests are separate from
CPU reference comparisons; see [precision qualification](docs/PRECISION.md).

Reference dump scripts read trusted local PyTorch checkpoints; inspect their
loading behavior before using third-party files. Do not weaken tolerances or
regenerate expected outputs merely to pass. Report output ordering and near-tie
diagnostics separately from a strict parity verdict.

## Reviewing a change

Review substantial changes in successive rounds:

1. Look for concrete counterexamples: invalid inputs, empty targets, checkpoint
   mismatches and differences from the upstream user examples.
2. Reproduce each finding, add a regression test where appropriate, and fix the
   cause. Do not relax numerical tolerances to hide a failure.
3. Re-review the repaired code, preferably with an independent reviewer. Exercise
   the affected workflows and check installed artifacts for packaging changes.

Performance and parity claims should name the source revision, hardware, data,
precision, warmup and timing scope. Preserve raw run records locally under
ignored `artifacts/`, including negative results and limitations. Experiment
reports, measurements and cost/budget calculations are internal and must stay out
of tracked documentation and packages unless the owner explicitly authorizes
publication. A synthetic test proves the tested behavior, not learned accuracy on
a real deployment dataset.

Keep changes independently reviewable. Retain upstream attribution in modified
files and update NOTICE when adapting another component. Do not commit model
weights, datasets, credentials, private conversations or machine-specific cloud
configuration. There is no automated publishing workflow; release authorization
is separate from accepting a code change.
