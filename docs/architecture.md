# Architecture and numerical parity

Parameters retain RF-DETR's upstream names and memory layouts; inputs are NCHW.
See [upstream parity](../CONTRIBUTING.md#upstream-parity) for reference versions
and reproduction commands.

## Checking numerical parity

`scripts/dump_torch_reference.py` records a PyTorch model's weights, inputs and
outputs. `scripts/check_parity.py` compares that bundle with the JAX implementation.
Use `jax_default_matmul_precision="highest"` for reference comparisons and
`--perturb-zeros` when dumping references: zero-initialized heads can conceal an
incorrect path by returning constant outputs.

Near-tied encoder scores can change which proposal a learned query slot receives.
The strict gate checks the port's unmodified forward outputs in their original
order. With recorded upstream selections, it also checks selected-score agreement
and outputs with those proposals injected. Those extra comparisons help isolate
the cause of a difference; they cannot excuse a native-forward failure.

`--dtype float64` on the dumper selects double-precision parameters and activations
in both frameworks. It helps investigate accumulated float32 error, but does not
prove correctness at another precision or on another backend. Some upstream
operations retain float32 arithmetic even with float64 activations.

The proposal grid, valid ratios and sine-embedding divisions use `precise_divide`
to prevent a broadcast division from being rewritten as multiplication by a
rounded reciprocal. Grid indices use `arange`. Resampling weights follow ATen's
operation order in the input dtype; fused multiply-add and backend differences can
still affect rounding. These implementation choices do not guarantee bitwise
agreement for the complete model.

`scripts/compare_precision_floors.py` compares median errors against a float64
reference with its proposal selection injected. Only groups with identical
upstream selections in both precisions contribute. A successful diagnostic does
not bound the largest error, validate excluded groups or prove identical error
distributions. Whole-model strict float32 parity remains incomplete; investigate
failures rather than replacing their verdict with these diagnostics.

Reference fixtures in `tests/data` cover resampling and task postprocessing
without requiring PyTorch at test time. Experiment results remain in ignored
`artifacts/`.

## Design notes

**The matcher runs on the host, or on the device.** Segmentation models add upstream's
point-sampled mask cross-entropy and Dice costs (one shared uniform draw per layer) to the class
and box costs before solving. Hungarian assignment is gradient-free upstream
(`@torch.no_grad`), so `jax.pure_callback` into SciPy keeps the entire training step inside one
`jit` without affecting correctness. Every output layer's cost matrix is stacked into one call, so a
step pays one round trip rather than one per layer. `CriterionConfig.matcher="auction"` removes even
that: it approximates the assignment on device and allows sharding. Its capped
greedy fallback has no optimality guarantee. Targets are padded to a fixed `max_targets` with a validity
mask, which is what lets the cost matrix keep a static shape; PyTorch's advanced-indexing scatters
become `.at[...].set(..., mode="drop")` with invalid rows steered out of bounds.

**Resampling is reimplemented, not delegated.** `jax.image.resize` is not a drop-in substitute for
`torch.nn.functional.interpolate`: its cubic kernel uses Keys `a = -0.5` where PyTorch's plain
bicubic path uses `a = -0.75`, and its antialias normalisation differs from ATen's. Worse, ATen's
*antialiased* bicubic path itself switches to Pillow's `a = -0.5` kernel. Since every call site has
statically known sizes, `utilities/resample.py` constructs separable weights following ATen's
formulas. That module imports no JAX, so the PyTorch process can share the data pipeline —
which is what makes an identical-batch training comparison possible.

**Parameter groups become pytrees.** PyTorch expresses RF-DETR's ViT layer-wise decay as one AdamW
group per tensor. Optax has no groups, so the same schedule is a per-parameter learning-rate tree
and a per-parameter weight-decay tree; the resulting update is identical to decoupled AdamW.

**Query-group attention.** `group_detr` query groups are folded onto the batch axis for
self-attention and unfolded afterwards, reproducing PyTorch's `split`/`cat` ordering exactly.
