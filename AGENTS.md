# Working on RF-DETR JAX

Read README.md and CONTRIBUTING.md before changing behavior.
Use the review checklist in CONTRIBUTING.md for substantial changes.

## Invariants

- Preserve PyTorch parameter names and layouts. Conversion is a path rename,
  not a transpose. Inputs are NCHW. Changes need checkpoint and parity evidence.
- Use `jax_default_matmul_precision="highest"` for numerical parity work.
  A zero-initialized head can conceal an incorrect path: perturb those heads in
  reference comparisons. Near ties are a diagnostic, not automatic permission
  to pass a failing parity gate.
- SciPy Hungarian assignment is the correctness reference. The capped auction
  implementation has different convergence/approximation behavior.
- CLI detection/segmentation labels retain raw COCO IDs. The public model API
  uses upstream class indexes and Supervision objects; its dataset adapter maps
  raw IDs explicitly. Keypoints use the upstream schema-slot mapping saved as
  category_to_label. Restore raw IDs at COCO evaluation/CLI output boundaries.
  Validate schemas and capacity; never silently remap or drop targets.
  Keep padding out of evaluation metrics.
- Keep reference fixtures in tests/data tracked and independent of PyTorch at
  test time. Do not regenerate them to make a regression disappear.
- Exercise train, export/reload, evaluate and predict for all three tasks. Reject
  missing masks/keypoints or inconsistent heads instead of dropping supervision.
  Keypoint metrics must include eligible categories with no predictions as zero.
- Do not change default precision or accelerator kernels based on an isolated
  microbenchmark. Measure the complete graph and preserve correctness evidence.

## Public API compatibility

Read docs/COMPATIBILITY.md before changing user-facing classes or methods. The
longer-term compatibility target is migration from RF-DETR with only the
installation/import change. The documented three core workflows and their explicit
limitations are the current compatibility scope.
Passing task-specific CLI tests does not establish public API compatibility. Test
upstream user examples, including constructor defaults, method arguments and
returned objects; do not silently accept unsupported options or substitute random
weights for a pretrained constructor.

## Verification

`make gate` runs Ruff, repository checks and CPU tests. Use `PYTHON=/path/to/python`
when needed. It does not certify CUDA, TPU, long-run accuracy or multi-host jobs.
For packaging changes, build both distributions and test the installed wheel
outside the checkout. For numerical changes, run the relevant gradient and parity
checks; record the upstream commit, precision, hardware and tolerances.

Bound thread counts on shared machines. Do not launch concurrent heavy accelerator
jobs. Small CPU regression tests should precede expensive model runs.

Run heavy accelerator workloads on dedicated hardware, serially, with explicit
resource labels. Keep machine, account and execution configuration outside tracked
files.

## Evidence and scope

Report the symptom, reproduction, fix and validation. Distinguish measured,
historical and unverified claims. Experiments are internal: keep reports, raw
measurements, cost/budget calculations, environment snapshots and failed/null
results in ignored artifacts/, not in tracked documentation or packages. Do not
publish experimental results without the owner's explicit authorization.
Benchmark device-step, epoch and process times are different quantities; include
warmup, compilation, evaluation and checkpoint I/O in their appropriate scopes.

Keep patches focused. Correct documentation when supported behavior changes.
Do not import another repository's contribution rules as this project's rules.

## Release boundary

Do not change visibility, publish packages, create releases, rewrite history, or
contact upstream maintainers without explicit authorization. Keep local
credentials, cloud identity, private context, datasets, weights, cache files and
virtual environments outside tracked files. The small tests/data reference
fixtures are intentional exceptions to the dataset exclusion.
