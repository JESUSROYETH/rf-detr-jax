#!/usr/bin/env bash
# Run the TPU benchmark suite on the node, one accelerator job at a time.
#
# Copy this repository to the node and run it from the copy. Results land in artifacts/;
# scripts/tpu_fetch_artifacts.sh brings them back into local ignored artifacts/.
set -x
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python=${PYTHON:-${TPU_REMOTE_PYTHON:-python3}}
run="./scripts/tpuenv.sh $python"

# One step, end to end, at the port's own settings: the number every other one is relative to.
$run scripts/bench_tpu.py --label reference --out artifacts/bench_baseline.json

# One variable at a time: precision, the matcher callback, gradient checkpointing, the backward pass.
$run scripts/ablate_tpu.py --out artifacts/ablation.json

# Where the step's time sits, component by component, and as nested prefixes of the real step.
$run scripts/profile_parts_tpu.py --out artifacts/parts.json
$run scripts/bench_stages_tpu.py --out artifacts/stages.json

# Which part of the forward pass costs what, with each piece stubbed out in turn.
$run scripts/bisect_forward_tpu.py --out artifacts/bisect_forward.json
$run scripts/bisect_forward_tpu.py --backward --out artifacts/bisect_backward.json
$run scripts/bisect_two_stage_tpu.py --out artifacts/two_stage.json
$run scripts/bisect_backbone_tpu.py --out artifacts/backbone_bisect.json

# Compare convolution layouts and linear-layer shapes, including cases with no improvement.
$run scripts/bench_conv_layout_tpu.py --out artifacts/conv_layout.json
$run scripts/bench_linear_shape_tpu.py --out artifacts/linear_shape.json

# Which deformable-attention sampler wins, against the size of the level it samples.
$run scripts/bench_sampler_tpu.py --out artifacts/sampler_crossover.json

# The two assignment solvers, on realistic and on deliberately degenerate targets.
$run scripts/bench_matcher_tpu.py --out artifacts/matcher.json

# Every configuration, single-chip and sharded, across batch sizes.
$run scripts/sweep_tpu.py --out artifacts/sweep.json

# The same before/after on the other released variants, at their own resolutions.
$run scripts/bench_variants_tpu.py --out artifacts/variants.json

# Whether a fresh process reuses the persistent XLA executables for this model.
rm -rf ~/.jax_cache_probe
for pass in cold warm; do
  JAX_COMPILATION_CACHE_DIR=~/.jax_cache_probe $run scripts/bench_tpu.py \
      --label "cache_$pass" --matcher auction --sampler matmul --grad-checkpoint off \
      --precision default --host-init --steps 4 --warmup 1 \
      --out "artifacts/bench_cache_$pass.json"
done

# The host pipeline, which binds once the step is fast enough.
$python scripts/bench_loader.py --out artifacts/loader.json

# End to end, at one batch size so both runs take the same number of optimizer steps.
$python scripts/train_jax.py --epochs 5 --batch-size 2 \
    --precision highest --matcher hungarian --sampler gather --grad-checkpoint on \
    --init-weights "" --out artifacts/train_tpu_reference.json
$python scripts/train_jax.py --epochs 5 --batch-size 2 \
    --precision default --matcher auction --sampler matmul --grad-checkpoint off \
    --host-init --workers 16 --init-weights "" --out artifacts/train_tpu_optimized.json

echo TPU_MEASUREMENTS_COMPLETE
