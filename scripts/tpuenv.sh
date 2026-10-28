#!/usr/bin/env bash
# Remote-side wrapper: persistent XLA compilation cache + repo on PYTHONPATH.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.jax_cache_tpu}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=xla_gpu_per_fusion_autotune_cache_dir
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
exec "$@"
