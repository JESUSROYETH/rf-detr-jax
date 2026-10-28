"""Source-checkout compatibility wrapper for the packaged training command."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfdetr_jax.training.run import evaluate, main, precompile_steps, spec_for  # noqa: E402, F401

if __name__ == "__main__":
    main()
