# RF-DETR JAX

A JAX/Flax port of [Roboflow RF-DETR](https://github.com/roboflow/rf-detr), built to
make better use of TPUs and improve GPU training throughput. It supports detection,
instance segmentation and keypoints, with upstream checkpoint conversion.
This is an independent port; it is not an official Roboflow package.

## Why this exists

The goal is to bring RF-DETR's models and workflows to efficient TPU training,
while keeping GPU training practical. Getting a model onto an accelerator is only
part of that work: memory layouts, indexed reads, CPU/device synchronization and
data loading can leave much of the hardware idle.

A native JAX implementation gives this project control over how the model and
training step are compiled, how batches are sharded across devices, and which
operations need hardware-specific kernels. [JAX compiles through XLA](https://docs.jax.dev/en/latest/jit-compilation.html);
the aim is to shape the computation for the hardware and optimize the complete
training path. Work includes deformable-attention sampling, query selection,
matching and the input pipeline.

The migration goal is to keep familiar RF-DETR model classes, checkpoints and
user workflows while changing the execution backend. Parameters retain their
PyTorch names and layouts, so converted checkpoints can be compared tensor by
tensor. Numerical parity tests and reference fixtures help check that performance
changes preserve model behavior.

## Status

**Pre-release.** Detection, instance segmentation and keypoints have training,
checkpoint export/reload, evaluation and prediction workflows. The intended
migration is an installation/import change; full drop-in compatibility remains
incomplete. See the [compatibility guide](docs/COMPATIBILITY.md) for supported
options and remaining gaps.

## Install

Use Python 3.12 on Linux for the currently tested environment. From this checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
rfdetr-jax-train --help
rfdetr-jax-convert --help
rfdetr-jax-eval --help
rfdetr-jax-predict --help
```

For an NVIDIA CUDA 12 environment, install `'.[cuda12,dev]'` instead. For a TPU,
install `'.[tpu,dev]'` on the TPU VM. Check the platform requirements in the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
PyTorch is a required dependency for reading and writing upstream-format `.pth`
checkpoints and for TensorBoard logging; model computation, training and prediction
use JAX. On a CPU-only machine, install a CPU PyTorch
wheel from its official index before this package to avoid CUDA dependencies.
Upstream reference comparisons additionally need `'.[reference]'` and the
pinned RF-DETR source described in [CONTRIBUTING.md](CONTRIBUTING.md).

No weights or datasets are downloaded by installing the package or running the
unit tests. The ignored `upstream/` checkout is only needed for reference scripts;
ordinary training and evaluation work from an installed wheel.

## Python API migration

The public API under development uses the same model class names and methods:

```python
from rfdetr_jax import RFDETRNano, RFDETRSegNano, RFDETRKeypointPreview

model = RFDETRNano()  # Loads the upstream pretrained checkpoint, downloading if needed.
detections = model.predict("example.jpg", threshold=0.5)
# A supervision.Detections object; segmentation adds masks.
# RFDETRKeypointPreview.predict returns supervision.KeyPoints.

model.train(dataset_dir="my-dataset", epochs=10, batch_size=4)
metrics = model.evaluate(dataset_dir="my-dataset", split="val")
restored = RFDETRNano.from_checkpoint("output/checkpoint_best_total.pth")
```

Use `pretrain_weights=None` explicitly for random initialization. Detection,
segmentation and keypoint constructors, prediction objects, checkpoint conversion
and export have focused compatibility tests. Full upstream workflow compatibility
remains a separate qualification target: consult [the contract](docs/COMPATIBILITY.md) before
assuming an existing training configuration is supported.

For ONNX export, install `'.[onnx]'` (or `'.[export]'`) and use
`model.export(output_dir="export")`. This optional bridge copies the current JAX
parameters into a separate CPU RF-DETR 1.11.2 model and runs its exporter.
The export extra constrains Torch/Torchvision to the tested 2.10/0.25 pairing.
Three-task tests compare the resulting ONNX Runtime outputs with JAX outputs.
This does not qualify other export formats or production deployment hardware.

## Train a detector

The training CLI uses RF-DETR Nano geometry at 384 pixels. Prepare a COCO dataset:

```text
my-dataset/
  train/
    _annotations.coco.json
    image-001.jpg
  valid/
    _annotations.coco.json
    image-002.jpg
```

```bash
rfdetr-jax-train \
  --dataset-dir my-dataset --train-split train --val-split valid \
  --num-classes 3 --max-targets 100 \
  --epochs 5 --batch-size 2 --out runs/example/metrics.json
```

This starts from random weights. Five epochs is a usage example, not an accuracy
recipe. Category IDs are used directly, without contiguous remapping:
`--num-classes` must cover the largest category ID, and the head has
`num_classes + 1` outputs. Keep the same category schema in both splits. Set
`--max-targets` high enough to hold every training image's annotations; overflow
is an error. Omitted, `--max-targets` defaults to 20 padded targets per image and
`--num-classes` to 6 for detection (90 for the segmentation and keypoint task
defaults); set both from the dataset. Training drops an incomplete last batch;
evaluation includes it.

The trainer exports inference parameters and metadata alongside the metric log.
See [training and checkpoints](docs/TRAINING.md) for initialization, output files,
explicit COCO image/annotation paths, and the limits of checkpoint restoration.

## Convert and evaluate

Acquire a compatible Nano checkpoint from upstream under its applicable terms,
then convert it locally:

```bash
rfdetr-jax-convert rf-detr-nano.pth --out weights/nano.npz
rfdetr-jax-eval \
  --params weights/nano.npz \
  --images my-coco/val2017 \
  --annotations my-coco/annotations/instances_val2017.json \
  --resolution 384 --num-classes 90 --dec-layers 2
```

The converter preserves tensor layouts; it does not resize class heads or infer
the complete architecture. Evaluation settings must match the checkpoint. The
CLI evaluates boxes and, for the corresponding model, masks or keypoints. See [security](SECURITY.md) before loading legacy
PyTorch files, and [upstream parity](CONTRIBUTING.md#upstream-parity) for the reference versions.

## Segmentation and keypoints

Use the same COCO split layout with per-instance `segmentation` polygons/RLE or
`keypoints` triplets and category keypoint names:

```bash
rfdetr-jax-train --task segmentation \
  --dataset-dir my-masks --train-split train --val-split valid \
  --num-classes 3 --epochs 5 --out runs/masks/metrics.json

rfdetr-jax-train --task keypoints --keypoint-schema 17 \
  --keypoint-flip-pairs 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --dataset-dir my-person-keypoints --train-split train --val-split valid \
  --epochs 5 --out runs/keypoints/metrics.json
```

Defaults follow upstream Nano (384 px), SegNano (312 px) and KeypointPreview
(576 px) geometry. `--model-config` preserves a specific checkpoint architecture.
Keypoint schemas are indexed by model label; the saved metadata maps these labels
back to COCO category IDs. Set flip pairs for the dataset's left/right semantics;
custom keypoint schemas need deliberate OKS sigma choices. See the
[workflow guide](docs/TRAINING.md) for these contracts.

## Predict

Keep exported parameters and their JSON sidecar together:

```bash
rfdetr-jax-predict --params runs/masks/metrics.params.npz \
  --image example.jpg --threshold 0.5 --out predictions.json
```

The same interface handles all three tasks:

```python
from rfdetr_jax.inference import RFDETR

model = RFDETR.from_checkpoint("runs/masks/metrics.params.npz")
result = model.predict("example.jpg", threshold=0.5)
# NumPy scores, labels, boxes; plus masks or keypoints for those models.
```

Boxes and keypoints use original-image pixel coordinates; masks have the original
image size. The JSON command encodes masks as COCO RLE. This lower-level wrapper accepts local
image paths, PIL images and RGB uint8 NumPy arrays. It does not download weights.

## What is supported

| Area | Evidence and boundary |
| --- | --- |
| DINOv2 backbone, projector, deformable decoder and detection heads | Unit tests and layer/output parity checks; whole-model strict float32 parity remains incomplete. Proposal-injection and median-error diagnostics do not establish native-forward equivalence ([details](docs/architecture.md), [limits](docs/COMPATIBILITY.md)) |
| Hungarian matching, classification, box and mask costs/losses | Value/gradient tests and replayed-point loss parity against upstream; SciPy is the reference matcher |
| On-device auction matcher | Optional approximate solver; do not assume identical assignments or convergence in every case |
| Detection, instance segmentation and GroupPose keypoints | Train, export/reload, evaluate and predict; small CPU integration tests |
| Optimizer and data pipeline | AdamW, schedules, EMA, threaded/process loaders; polygon/RLE masks and keypoint schemas |
| GPU and TPU | Backend-specific kernels; CPU tests do not exercise every accelerator path |
| Export | Optional upstream export bridge; tiny CPU ONNX output parity for all three tasks. Other formats and deployment targets remain unqualified. |
| Multi-host training | Not qualified; local-device sharding is available |

LoRA, Lightning callbacks and upstream's additional augmentation backends remain
open compatibility requirements. YOLO detection, polygon and pose datasets use a
temporary COCO annotation index; the original dataset is preserved.

## Documentation

- [Training and checkpoints](docs/TRAINING.md): datasets, task configuration and restoration.
- [API compatibility](docs/COMPATIBILITY.md): supported upstream workflows and remaining gaps.
- [Architecture](docs/architecture.md): tensor conventions and implementation choices.
- [Precision](docs/PRECISION.md): compute dtypes, loss scaling and reproducibility.

## Development

```bash
make gate                         # CPU tests, lint and repository checks
python -m build                   # wheel and source distribution
```

Tests use small synthetic inputs and tracked reference fixtures. GPU-specific
tests skip on CPU; a green CPU run is not accelerator qualification. The
[contributor guide](CONTRIBUTING.md) explains the checks and how to reproduce
upstream parity. [AGENTS.md](AGENTS.md) records the implementation invariants.

```text
src/rfdetr_jax/  packaged models, data, training, evaluation and CLI
tests/          unit/regression tests and small reference fixtures
scripts/        parity, benchmark and environment helpers
docs/           usage, compatibility and design
```

## License and attribution

Licensed under [Apache-2.0](LICENSE). Based on [Roboflow RF-DETR](https://github.com/roboflow/rf-detr);
upstream acknowledgments and notices are retained in [NOTICE](NOTICE).
Weights and datasets have their own terms and are not distributed here.
