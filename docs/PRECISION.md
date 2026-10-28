# Precision policy and qualification

The public API follows the published RF-DETR 1.11.2 precision request policy.
The training and ordinary prediction policies differ, as they do upstream.
This document describes the implementation and the bounded checks; it does not
establish convergence or production accuracy.

| Request | CPU | NVIDIA CUDA |
| --- | --- | --- |
| `amp_dtype=None` | FP32 | FP32 |
| `amp_dtype="auto"` | FP32 | BF16 on devices with native BF16 support; otherwise FP16 |
| `amp_dtype="bf16"` | FP32 | BF16, with an emulation warning before Ampere |
| `amp_dtype="fp16"` | FP32 | FP16 with dynamic loss scaling |
| `amp_dtype="fp8"` | Explicitly unsupported | Explicitly unsupported |

Legacy model `amp=False` disables automatic mixed precision. An explicit
`bf16` or `fp16` training request takes precedence over that legacy flag.
Unknown precision values warn and fall back to `auto`, following the published
configuration. TPU resolution accepts FP32 or BF16; the CUDA checks do not
qualify TPU execution.

Preprocessing and input storage remain FP32. Eligible model operations use the
resolved compute dtype; parameters, Adam moments and EMA remain FP32. Head
casts occur at operation boundaries so residuals and normalization do not lose
precision prematurely. Ordinary public `predict()` uses FP32 after training
and reload. Explicit inference optimization can select another dtype.
Standalone `evaluate()` resolves a fresh precision request, with `auto` as its
default, rather than inheriting an earlier explicit training request.

FP16 scales the backward loss and unscales gradients before clipping. A
nonfinite gradient skips the parameter and Adam-state update, reduces the
scale, and advances the attempted-step scheduler and EMA, matching the tested
Lightning lifecycle. The first public EMA update copies the observed weights;
later updates average them. Native version-2 training checkpoints preserve the
scaler and counters. Version-1 checkpoints can resume unscaled training; they
cannot supply missing FP16 scaler state.

Public keypoint accumulation follows the published manual-optimization path:
raw loss numerators are divided by the accumulated object-count denominator,
and prior gradients are rescaled as each microbatch arrives. Averaging already
normalized microbatch losses gives a different update when object counts differ.
The raw training CLI retains its historical behavior unless this policy is
selected explicitly.

Detection and segmentation use Lightning's configured accumulation divisor,
including the final partial group. For example, with accumulation set to two,
a last single microbatch still contributes half its loss gradient. These
automatic and keypoint policies are saved and preserved during native resume.

## Reproduction

Install the published reference dependencies using CONTRIBUTING.md. CPU
reference checks are selected explicitly and must not skip:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/run_reference_tests.py tests/test_precision_reference.py tests/test_public_precision.py -q
```

Use a CUDA environment for actual updates and standalone checkpoint resume:

```bash
JAX_PLATFORMS=cuda,cpu JAX_DEFAULT_MATMUL_PRECISION=highest \
  XLA_FLAGS=--xla_gpu_deterministic_ops=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/run_reference_tests.py tests/test_gpu_training_precision.py tests/test_gpu_public_precision.py -q
```

For actual CUDA output and head-gradient comparisons, use the same environment
and set `RFDETR_PRECISION_REFERENCE_DEVICE=cuda`, then run
`python scripts/run_reference_tests.py tests/test_precision_reference.py -q`.
The reference file also includes CPU Lightning lifecycle controls. Device-context
checks are in `tests/test_training_device_placement.py`; run them with CUDA as
the process default to verify that selected CPU work stays on CPU.

The tests use tiny synthetic models and data. Default CPU CI skips the CUDA
tests; those skips cannot be counted as accelerator passes. Exact checkpoint
serialization and numerical repeatability of subsequent GPU operations are
separate checks. Exact subsequent-update comparisons use deterministic XLA GPU
operations; default CUDA kernels do not guarantee bitwise repeatability.

Public methods establish an explicit `jax.default_device` context. Low-level
callers tracing a model for a non-default backend must likewise set a matching
device context; supplying only `jit(backend=...)` or input sharding does not
expose the intended backend to Python-side precision selection. Device objects
and the platform strings `cpu`, `gpu` and `tpu` are recognized.
