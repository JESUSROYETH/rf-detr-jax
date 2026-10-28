"""Compatibility import for shared benchmark and reference configurations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfdetr_jax.training.run_config import DEFAULT_RUN, RunConfig  # noqa: E402, F401
