#!/usr/bin/env bash
# Run a PyTorch command with bounded CPU footprint.
# On a shared machine, unbounded intra-op thread pools starve other jobs, so cap thread counts
# and drop scheduling priority.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-4}
exec nice -n 15 "$@"
