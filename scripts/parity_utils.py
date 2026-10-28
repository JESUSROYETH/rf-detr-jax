"""Strict numerical comparison shared by standalone reference checks."""

from __future__ import annotations

import numpy as np


def compare_named(actual: dict, reference: dict, tolerance: float, *, relative: bool = False) -> int:
    """Return failure count; missing, empty, nonfinite or differently shaped values always fail.

    Relative mode scales the absolute tolerance by ``max(1, max(abs(reference)))``.
    It never broadcasts shapes or accepts a comparison with no named outputs.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    names = set(actual) | set(reference)
    if not names:
        print("  FAIL no outputs to compare")
        return 1
    failures = 0
    for name in sorted(names):
        if name not in actual or name not in reference:
            print(f"  FAIL {name}: missing model output or reference")
            failures += 1
            continue
        value, target = np.asarray(actual[name]), np.asarray(reference[name])
        if value.shape != target.shape or not value.size:
            print(f"  FAIL {name}: invalid shapes {value.shape} vs {target.shape}")
            failures += 1
            continue
        if not np.isfinite(value).all() or not np.isfinite(target).all():
            print(f"  FAIL {name}: nonfinite values")
            failures += 1
            continue
        error = float(np.max(np.abs(value.astype(np.float64) - target.astype(np.float64))))
        scale = max(1.0, float(np.max(np.abs(target.astype(np.float64))))) if relative else 1.0
        failed = error > tolerance * scale
        failures += failed
        print(f"  {'FAIL' if failed else 'OK  '} {name:24s} maxabs={error:.3e} tolerance={tolerance * scale:.3e}")
    return failures
