#!/usr/bin/env bash
# Run a JAX command with bounded CPU and GPU footprint.
# On a shared machine, unbounded thread pools and GPU preallocation starve other jobs, so cap
# thread counts, limit XLA compilation parallelism and drop scheduling priority.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_force_compilation_parallelism=2}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}
# Persist compiled executables so repeated runs skip XLA compilation.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/.jax_cache}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
exec nice -n 15 "$@"
