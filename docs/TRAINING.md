# Training, evaluation and prediction

The runtime is JAX/Flax. Public model classes follow upstream RF-DETR's Python
workflow; complete compatibility is still being tested (see COMPATIBILITY.md).
The CLI is a lower-level interface with explicit geometry and raw COCO category
IDs. Its defaults and output representation differ from the public model classes.

## Public model workflow

`RFDETRNano`, `RFDETRSegNano` and `RFDETRKeypointPreview` load pretrained weights
by default. Pass `pretrain_weights=None` to initialize randomly. Public `.train()`
accepts Roboflow COCO, original COCO and YOLO datasets; it derives custom class
names and adapts the class head unless the constructor explicitly fixes its size.
Prediction returns Supervision objects with model class indexes, as upstream does.

For a configured non-None `pretrain_weights`, each fresh public `train()` call
reloads that source directly into the dataset-aligned model geometry. It does
not continue the live weights from a previous run. A model created with
`from_checkpoint(path)` uses that path as its source; replacing a custom local
file changes the next fresh run, and a missing or corrupt file fails explicitly.
Pretrained keypoint training resets only the Gaussian precision output rows,
matching the published training module. Ordinary reload, evaluation and prediction
preserve those rows. Use `train(resume=path, ...)` to restore native training state,
including optimizer, EMA, scaler and RNG, without fresh initialization.

With `pretrain_weights=None`, each fresh training call initializes a new native
parameter tree in the dataset-aligned geometry. It does not reuse live constructor
or previously trained weights. An omitted public seed uses native model seed 42
and data seed 7; an explicit seed sets both. Native resume preserves both saved
streams, including when they differ. This policy does not reproduce upstream
Torch global RNG or initialization distributions. This port initializes all weights
randomly for that constructor option; upstream can still preload DINOv2 with
compatible Base/LargeDeprecated geometry.
See [the compatibility contract](COMPATIBILITY.md) for the remaining lifecycle limits.

Public training accepts `amp_dtype="auto"`, `"bf16"`, `"fp16"` or `None`.
CPU resolves to FP32; CUDA auto selects native BF16 when available and otherwise
uses FP16 with dynamic loss scaling. Ordinary prediction uses FP32 after mixed
training. See [precision policy and qualification](PRECISION.md) for accumulation,
overflow and checkpoint behavior.

```python
from rfdetr_jax import RFDETRSegNano

model = RFDETRSegNano()
model.train(dataset_dir="my-dataset", epochs=10, batch_size=4, output_dir="output")
model = RFDETRSegNano.from_checkpoint("output/checkpoint_best_total.pth")
predictions = model.predict("example.jpg")
```

YOLO uses `data.yaml` or `data.yml`, image/label split directories, and `kpt_shape`
for pose labels. Conversion creates a temporary annotation index and preserves
the dataset. Named split paths in YAML take priority; `valid/` and `val/` are
accepted. Missing label files represent background images. Polygon masks follow
upstream's PIL rasterization, and 2D/3D pose visibility conventions are preserved.

## Dataset and model configuration

`rfdetr-jax-train --help` is the authoritative flag list. Detection defaults use
Nano at 384 px, two decoder layers and 13 query groups. `--task segmentation`
selects SegNano at 312 px; `--task keypoints` selects KeypointPreview at 576 px.
Both use patch size 12, four decoder layers and 100 queries. These match the
recorded upstream geometries, not a claim of identical Python API behavior. For a Roboflow COCO export,
set `--dataset-dir`, `--train-split train` and `--val-split valid` explicitly.
For the original COCO layout:

```bash
rfdetr-jax-train \
  --train-images /data/coco/train2017 \
  --train-annotations /data/coco/annotations/instances_train2017.json \
  --val-images /data/coco/val2017 \
  --val-annotations /data/coco/annotations/instances_val2017.json \
  --num-classes 90 --max-targets 300 --batch-size 2 --epochs 5 \
  --out runs/coco/metrics.json
```

Detection and segmentation category IDs are preserved. A dataset with IDs 1, 3 and 7 needs `--num-classes 7`,
not 3. The model creates one extra output slot. Training and validation must use
the same ID-to-name schema. An incomplete training batch is dropped, so the split
must contain at least one complete batch. Validation includes every image.

Training rejects overflow instead of silently discarding annotations. Omitted,
`--max-targets` defaults to 20 and `--num-classes` to 6 (90 for the segmentation
and keypoint task defaults). Choose
`--max-targets` from the actual data; large values increase memory and compilation
cost. `--target-buckets` can reduce padding but creates multiple compiled shapes.
The auction matcher requires target capacity no greater than queries per group.
SciPy Hungarian is the default reference solver; auction is optional and its
capped fallback does not guarantee the same assignment or loss trajectory.

## Initialization and outputs

With no initialization flag, training starts randomly. Use `--init-weights` for
an NPZ with original upstream state-dict keys; use `--init-params` for an NPZ
produced by the converter or this trainer. The flags are mutually exclusive.
The file must exist and match the model's tensor names and shapes.
CLI class-head resizing is not automatic; a 91-output COCO checkpoint does not fit a
four-output custom dataset model.

```bash
rfdetr-jax-train \
  --dataset-dir my-dataset --train-split train --val-split valid \
  --num-classes 3 --max-targets 100 --epochs 5 --batch-size 2 \
  --save-params runs/example/model.npz --out runs/example/metrics.json

rfdetr-jax-eval \
  --params runs/example/model.npz --model-config runs/example/model.npz.json \
  --images my-dataset/valid --annotations my-dataset/valid/_annotations.coco.json
```

The lower-level trainer and CLI export the EMA parameters when EMA is enabled, otherwise the raw
parameters. By default the filename is the metrics path with `.params.npz`
in place of `.json`; `--save-params` chooses another destination. Its `.json`
sidecar carries the model specification, task, raw category IDs/names,
`category_to_label` mapping, keypoint flip pairs/OKS sigmas and training step.
Keep it with the NPZ: evaluation and prediction discover `<params>.json`
automatically. `--model-config` chooses a different metadata file or plain
`LWDETRSpec` JSON. The metric log records each epoch.

These parameter exports are separate from resumable training snapshots. The
trainer also writes `checkpoint_last.npz` under `--output-dir`, containing model,
Adam state, EMA, optimizer step, random state, completed epochs and the training
contract, including the FP16 scaler when enabled. Use `--resume` with that snapshot to continue the same planned schedule.
Changing the planned horizon or incompatible optimization settings is rejected.
CPU tests compare uninterrupted and interrupted/resumed training, including a
partial gradient-accumulation group. This is not evidence of bitwise recovery
across different hardware, software versions or changed datasets.

Public `.train()` keeps the final raw model in memory for subsequent prediction,
evaluation and `save()`, matching the upstream default without `run_test`.
EMA and best-checkpoint files are separate explicit choices.

Public `evaluate()` creates a fresh task configuration on every call. Supply
`dataset_dir` each time and repeat any non-default validation settings, such as
`eval_max_dets`, `eval_batch_size` or keypoint OKS sigmas, when comparing with a
training run. Saved training options do not become standalone evaluation
defaults. A checkpoint that carries only label-indexed class names (an upstream
fine-tuned checkpoint, or a model that has not been trained here) is evaluated
with the dataset's own label space, derived exactly as `train()` derives it:
contiguous labels over the kept Roboflow/YOLO categories, while native COCO keeps
raw ids. For Roboflow/YOLO datasets a class-count difference produces a
`UserWarning`; renamed classes warn for every format; a dataset needing more
class slots than the head provides is an error. Evaluation uses the current in-
memory weights; a resolution override adapts positional embeddings temporarily
and preserves the inference model's geometry, parameters and configured device,
including when evaluation fails.

Public `.train()` writes `.pth` files with native resume state embedded alongside
upstream-readable weights, plus `last_ema.pth` (final EMA weights, when EMA is on)
and `training_config.json`, as upstream does. A resumed run without an explicit
`amp_dtype` keeps the saved run's resolved precision even when the resuming device
would resolve `"auto"` differently; an explicit request that changes the precision
is rejected. A copied checkpoint is self-contained; it does not
depend on an adjacent native state file. `from_checkpoint` loads inference weights;
`train(resume=...)` restores the training state. Omitted public training settings
are inherited from the saved checkpoint.
Keep the original output directory to preserve earlier best-epoch files. A
standalone latest checkpoint restores the training trajectory, but does not carry
every historical best model; a new directory may have no best checkpoint until
a later epoch wins again.

## Instance segmentation

Use `--task segmentation`. Each non-crowd object needs `segmentation`: COCO
polygons, compressed RLE and uncompressed RLE are accepted. Masks follow the same
resize, crop and flip as their boxes, with nearest-neighbor target resizing.
Training matches queries to objects with upstream's point-sampled mask cross-entropy
and Dice costs in addition to the class and box costs, and includes the sampled mask
cross-entropy and Dice losses; evaluation reports bbox and segmentation metrics.
Missing or empty mask supervision is an error.

Prediction resizes mask logits to the original image dimensions and thresholds
at zero. Python results contain boolean masks; JSON outputs use compressed RLE.

## Keypoints and category mappings

Use `--task keypoints`. An annotation's `keypoints` is a flat list of x, y and
visibility triplets. COCO visibility is 0 (unlabeled), 1 (occluded) or 2 (visible).
Category `keypoints` names declare its schema even when an image has no objects.
The loader checks consistency and supports empty and partially labeled images.

`--keypoint-schema 17` is the compact upstream person layout: model label 0 has
17 points. `--keypoint-schema 0,17` is the legacy layout with points at label 1.
Without an explicit schema, the trainer derives compact slots from sorted
keypoint-bearing categories. Categories map to active slots in sorted raw-ID
order; other categories occupy free slots. The JSON sidecar records this mapping
so evaluation and prediction can restore raw IDs. A keypoint count is indexed by
**model label**, not raw COCO category ID. Validation splits must use the same
category schema and mapping.

Set `--keypoint-flip-pairs` to comma-separated left/right index pairs. For the
standard 17-point person schema use:

```text
1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
```

Without flip pairs, the keypoint trainer disables horizontal flips to avoid
corrupting left/right semantics. All categories share the supplied pair
definition, so multi-category datasets must use compatible point ordering. `--keypoint-oks-sigmas` supplies positive comma-separated values
covering every point. Defaults follow upstream person sigmas for 17 points and
uniform 0.05 for other counts; the latter is a convention, not a validated metric
calibration for a custom object.

Evaluation reports bbox and OKS metrics. Keypoint aggregates macro-average
eligible categories, including zero scores for categories with no predictions;
an aggregate that omits categories without predictions is not comparable with
these scores. Predictions carry padded
`keypoints` arrays and `keypoint_counts`; use the count for each detection.

## Checkpoint conversion and prediction

For a task-specific upstream checkpoint, create a JSON model specification matching
its architecture and schema. The converter can validate every tensor and attach
COCO category metadata:

```bash
rfdetr-jax-convert checkpoint.pth --out weights/model.npz \
  --model-config model-spec.json \
  --annotations my-dataset/train/_annotations.coco.json
rfdetr-jax-predict --params weights/model.npz \
  --image example.jpg --threshold 0.5 --out predictions.json
```

Conversion without `--model-config` remains available for existing detection
workflows; prediction then requires an explicit specification. Conversion does
not resize heads or guess the geometry. Use the saved training sidecar to reload
trained parameters. Predictions accept local files, PIL images or HWC RGB uint8
NumPy arrays and return scores, raw category labels, xyxy boxes and task outputs
in original-image coordinates. Loading is local and never downloads weights.

## Runtime choices

Use float32 and highest matmul precision for parity comparisons. bf16, fused
attention, sampler choice and on-device matching change numerical behavior and
must be qualified on the target device. The benchmark wrappers in `scripts/` call
the same training loop that the installed command owns.

`--data-parallel` shards across local devices and requires the auction matcher
and divisible train/evaluation batch sizes. Multi-host orchestration is not a
supported workflow. CPU tests do not exercise TPU Pallas or GPU cuDNN kernels.
