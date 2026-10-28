"""Default XLA settings for the CUDA backend.

XLA reads ``XLA_FLAGS`` once, when the backend starts, so :func:`apply_gpu_xla_defaults` must run
before the first JAX computation. It only touches the environment when the CUDA plugin is
installed, and never overrides a setting the user already chose.
"""

from __future__ import annotations

import importlib.util
import os

# Disable Triton GEMMs by default. This controls XLA's matmul implementation choice; users can
# override it through XLA_FLAGS. Evaluate alternatives using the complete training step.
GPU_XLA_DEFAULTS = {"--xla_gpu_enable_triton_gemm": "false"}


def apply_gpu_xla_defaults(environ: dict | None = None) -> list[str]:
    """Append :data:`GPU_XLA_DEFAULTS` to ``XLA_FLAGS`` for a CUDA build; return the flags added."""
    environ = os.environ if environ is None else environ
    if importlib.util.find_spec("jax_cuda12_plugin") is None and importlib.util.find_spec("jax_cuda13_plugin") is None:
        return []
    current = environ.get("XLA_FLAGS", "")
    added = [f"{flag}={value}" for flag, value in GPU_XLA_DEFAULTS.items() if flag not in current]
    if added:
        environ["XLA_FLAGS"] = " ".join([current, *added]).strip()
    return added
