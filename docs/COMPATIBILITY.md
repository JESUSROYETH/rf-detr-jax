# Upstream user-code compatibility

The intended migration is an installation/import change while existing RF-DETR
user code retains its model classes, calls and result handling. JAX/Flax remains
the training and prediction backend. **Full drop-in compatibility is still an
open qualification target.** This guide describes the supported core workflows
and the options that still differ from upstream.

## Reference versions

The source-port and initial API reference is Roboflow RF-DETR commit
[`6674d8581b7694baf567b993a4f32000a6f4f4c2`](https://github.com/roboflow/rf-detr/tree/6674d8581b7694baf567b993a4f32000a6f4f4c2),
whose local package metadata says `1.10.0.dev1`. That metadata is not a published
PyPI version and must not be used as an installable version pin.

The optional export bridge is tested against the published
[`rfdetr==1.11.2`](https://pypi.org/project/rfdetr/1.11.2/) release. Its wheel
`rfdetr-1.11.2-py3-none-any.whl` has SHA-256
`2f23589567123b59302f1cfa9d14d23630f9f583ee6f3468c5891e33fb103b2a`.
Relative to the source pin, the release keeps the named-variant defaults and task
postprocessing unchanged and changes prediction transfer scheduling, training
configuration and exporters. The release adds exporter arguments
`max_batch_size`, `coreai_precision` and `openvino_precision`, and CoreAI,
OpenVINO and LiteRT formats. Delegation exposes these installed-upstream
arguments; it does not qualify those formats.

This is a fixed, reproducible compatibility baseline, not a promise to track the
latest upstream release automatically. In particular, 1.11.2 adds `cuda_graphs`,
`eval_backend`, `pad_targets_to` and WebDataset configuration not implemented by
the current JAX public facade.

## Implemented surface and evidence

| Surface | Current behavior and evidence | Remaining boundary |
| --- | --- | --- |
| Named models | Detection variants, segmentation variants and `RFDETRKeypointPreview`; explicit random initialization or official pretrained checkpoint selection; variant/default and safe-loading tests | Not every official full-size checkpoint is qualified end to end |
| Prediction | Real Supervision `Detections`/`KeyPoints`; single/list inputs, local paths, URLs, PIL, RGB NumPy and optional Torch tensors; thresholds, shapes, masks, keypoint uncertainty and class/source metadata | CPU fixtures establish result contracts, not full-size accelerator throughput or accuracy |
| Class labels | Public predictions retain model-slot labels; standard COCO detection/segmentation preserves sparse source IDs and the pretrained head; Roboflow/YOLO uses explicit mappings; schemas are indexed by model label | Custom multi-class keypoint datasets still require correct per-class schemas, flip pairs and OKS sigmas |
| Checkpoints | Native NPZ plus metadata and Torch `.pth` state conversion; public variant inference, class-head adaptation and safe-loading checks; upstream fine-tuned checkpoints load, predict and evaluate with their label-indexed class names | Native optimizer/RNG resume is distinct from importing arbitrary upstream Lightning optimizer state |
| Training | Public `train`, fresh non-None checkpoint reload, fresh native initialization for None, pretrained keypoint Gaussian reset, in-memory parameter updates, class/schema adaptation, native optimizer/EMA/RNG resume, local checkpoints and TensorBoard | Only explicitly supported training options; upstream initialization/RNG parity and full Lightning lifecycle equivalence remain open |
| Precision | Public CPU/CUDA request resolution, FP16 dynamic scaling, saved scaler state, upstream accumulation and first-update EMA policies; ordinary prediction returns to FP32 | See [precision policy](PRECISION.md) for behavior and reproduction commands; FP8 remains unsupported. Whole-model strict float32 parity remains incomplete. Float64, injected-selection and median-error checks are separate diagnostics; they do not establish native-forward parity at the production precision or on untested hardware |
| Datasets | Roboflow COCO, native COCO and YOLO detection/polygon/pose parsing; original datasets are preserved | Objects365 and WebDataset are not implemented; additional upstream augmentation backends remain open |
| Evaluation | Fresh standalone options, temporary resolution/device overrides and current live weights; shared published-callback target reconstruction and AP/F1 conventions, with headline box AP/AR plus separate mask/OKS keys | Per-class public metric logging, native-mask-grid evaluation and selectable upstream evaluation engines remain open |
| Inference optimization | JAX inference forward with shape/batch and floating-dtype controls; reversible and explicit inplace lifetimes | JAX compilation and device support differ from Torch, CUDA Graphs and deployment engines |
| Export | Optional export-only Torch reconstruction from current JAX tensors; strict state load; delegates upstream options and writes artifacts locally | Static ONNX and local Roboflow bundles are the tested paths; no general deployment qualification |

Unknown constructor, training or prediction options raise
`rfdetr_jax.configuration.UnsupportedOptionError`, which is both a `TypeError` and a
`ValueError`. Handlers for either base class catch it; a handler specifically for
upstream's `pydantic.ValidationError` does not. When migrating such a handler,
catch `UnsupportedOptionError` explicitly (or `ValueError` if broader handling is
intended). Pydantic-specific error details are not provided.

`pretrain_weights=None` and the deprecated `RFDETRBase`,
`RFDETRLargeDeprecated` and `RFDETRSegPreview` classes warn as upstream does;
`model_config` is updated in place by `train()` and rejects unknown attributes;
`Subclass.from_checkpoint()` returns the variant recorded in the checkpoint.
`class_names` of an untrained custom-class model are `class_<i>` placeholders rather
than upstream's COCO names, and `evaluate(split="test")` does not return upstream's
`test/loss`.

The public facade is `rfdetr_jax.RFDETR` and its named subclasses. The
`rfdetr_jax.inference.RFDETR` dictionary wrapper and command-line tools are lower
level APIs with their own metadata/category contracts. They should not be
substituted in an upstream Supervision example.

## Exact checks and their scope

`scripts/check_public_examples.py --backend rfdetr_jax --task detection` exercises
pretrained inference and Supervision consumers. Select `segmentation` or
`keypoints` for the other tasks and `--backend rfdetr` for the upstream reference.
This opt-in script may download official weights and the example image;
ordinary unit tests do not.

`tests/test_prediction.py` covers input validation, strict threshold behavior
(thresholds outside `[0, 1]` are rejected here, whereas upstream accepts them),
Supervision result fields, empty outputs, class names and keypoint covariance.
The task postprocessing fixture records synthetic outputs from the source pin;
its reference version and generation settings are in
`tests/data/task_postprocess_reference.json`.

`tests/test_export.py` compares JAX exports with the upstream implementation.
The three synthetic models use 32-pixel inputs, one
decoder layer and two queries; zero-initialized heads/layer scales are perturbed
so constant outputs cannot hide a broken path. The checks cover:

- Strict upstream model reconstruction for detection, segmentation and keypoints,
  comparing final tensors to JAX at `atol=rtol=1e-4`.
- Actual static ONNX files executed in CPU ONNX Runtime, compared to JAX at
  `atol=rtol=2e-4`, including embedded notes metadata.
- Local `weights.pt`/`class_names.txt` Roboflow bundles with sanitized paths and
  strict tensor reload. No upload is performed.

TensorRT, TFLite, CoreML, ExecuTorch, CoreAI, OpenVINO and LiteRT have not been
executed here. Dynamic batches, backbone-only export, quantization and target
hardware require their own tests before any compatibility claim.

`tests/test_public_metrics.py` verifies that segmentation or OKS AP cannot be
mistaken for headline box AP. `tests/test_f1_metrics.py` compares matching and
confidence sweeping directly with published upstream code, including stable
score ties, empty classes, missed categories, duplicates, crowd fallback,
segmentation masks, NaNs and float32 threshold boundaries. These compare the
lower-level matcher and sweep, not the complete published callback. Standalone
tests also exercise the public COCO wrapper. Keypoint F1 measures boxes, as upstream does;
keypoint OKS is a separate measurement.

The public evaluator reconstructs the published callback's box/mask targets:
ordinary boxes are clipped and degenerate rows removed, ordinary masks pass
through the model-resolution resize, and crowd annotations are restored through
their separate path. It derives areas from those targets and uses the exact
TorchMetrics float32 AP grids. AP and F1 receive the same targets. Keypoint OKS
retains original COCO annotations and the published weighting across schema
groups, including categories with labeled GT but no predictions. Raw CLI metrics
keep their original-annotation and eligible-category conventions.

`tests/test_public_evaluation.py` compares intermediate targets and final results
with actual RF-DETR 1.11.2 callback/metric code, including the `faster_coco_eval`
and `vernier` engines. Cases include clipped/degenerate boxes, restored crowds,
mask resize round trips, supplied-area differences, empty predictions/GT, score
ties, the recall-grid boundary, detection budgets and mixed keypoint schemas.
These bounded comparisons do not establish identical scores for every dataset.
The dedicated reference CI job treats a skipped selected test as a failure.

`tests/test_evaluation_lifecycle_reference.py` executes published 1.11.2's
evaluation preparation and positional-embedding transplant with lightweight
model/trainer stand-ins. Together with the native lifecycle tests, it checks
fresh task defaults, a required per-call dataset, temporary resolution and device
requests, and preservation of current weights on success and failure. Separate
all-task integration tests run real evaluation at a larger resolution and retain
the original model geometry and parameter objects. Device-routing stand-ins do
not establish accelerator placement; the PE numerical comparison uses FP32.

`eval_max_dets` is independent of the model's selected-output count. The published
callback sorts `[1, 10, eval_max_dets]`: for a value below 10, its AP still uses
the largest budget while the reported recall uses the requested budget. The
public adapter reproduces this edge case. `multi_scale="off"` and `"per-batch"`
preserve their meaning; `"per-sample"` is explicitly unsupported.

Fresh training with a non-None `pretrain_weights` reloads that configured source
directly into the effective dataset-aligned spec, rather than adapting previously
trained or truncated live weights. Pretrained keypoint training resets only
Gaussian precision rows 4:7 in decoder/encoder heads. Native resume bypasses this
initialization and restores its saved state. Ordinary constructors start with an
untrained lifecycle flag; `from_checkpoint` marks the model checkpoint-loaded,
including step-free files, and subsequent training warns about fresh versus
resumed initialization. With `pretrain_weights=None`, fresh training initializes a
new native tree rather than reusing live weights. Omitted seeds use native model
seed 42 and data seed 7; explicit seeds set both. Native resume restores the saved
pair independently. Fresh and resumed runs therefore own distinct seed streams;
this does not claim Torch initialization or global RNG parity.

`tests/test_pretrained_bootstrap_reference.py` uses actual published safe loaders,
classifier resizing, Gaussian resets and lifecycle methods with tiny local
checkpoints. Head-only comparisons bypass whole-model tree validation and do not
establish partial-checkpoint compatibility. `tests/test_training_bootstrap.py`
exercises full native checkpoint trees through a training-backend spy; existing
all-task integration tests supply real optimizer/resume coverage. The helper's
effective geometry overrides do not claim parity for mutable Pydantic field
tracking across repeated dataset changes.

## Remaining full-compatibility qualification

- Extend the installed-wheel inference examples and synthetic training/reload
  checks to full upstream training examples and representative datasets, preserving
  their configuration and downstream consumers. Inference examples do not close
  unsupported training configurations or metric differences.
- Resolve unsupported ordinary training/configuration workflows, including LoRA,
  Lightning integrations, per-class metric logging and the additional dataset and
  augmentation backends. Unsupported behavior must fail explicitly rather than
  silently ignore settings.
- Align the remaining model lifecycle behavior: explicit-None fresh training
  distributions/global RNG, constructor/training device placement, and mutable
  configuration field tracking across repeated dataset changes. Non-None source
  reload, pretrained Gaussian reset and standalone evaluation have bounded
  reference and integration coverage above.
  `pretrain_weights=None` currently initializes all JAX weights randomly; upstream
  can still load a pretrained DINOv2 encoder for compatible Base and
  LargeDeprecated geometry. That initialization difference remains open.
- Extend metric coverage and implement per-class logging, native-mask-grid
  evaluation and the remaining accepted upstream configuration options.
- Qualify every advertised export format/option on its required runtime, and
  qualify real accelerator workloads separately from CPU synthetic checks.

The documented core workflows do not establish full drop-in compatibility.
API compatibility, numerical parity, training convergence and deployment
readiness require separate validation.
