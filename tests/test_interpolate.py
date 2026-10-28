"""Interpolation parity against recorded ``torch.nn.functional.interpolate`` outputs.

The reference values in ``tests/data/interpolate_reference.npz`` were produced by
``torch.nn.functional.interpolate`` (``scripts/dump_torch_interpolate_reference.py``, torch 2.10.0)
so this test needs no PyTorch installation to run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rfdetr_jax.utilities.resample import resize_2d_numpy

_REFERENCE = np.load(Path(__file__).parent / "data" / "interpolate_reference.npz")
_CASES = [key for key in _REFERENCE.files if key != "x"]


@pytest.mark.parametrize("case", _CASES)
def test_matches_torch(case: str) -> None:
    """Each supported mode reproduces PyTorch to float32 precision."""
    mode, align_corners, antialias, size = case.split("|")
    height, width = (int(value) for value in size.split("x"))
    actual = resize_2d_numpy(
        _REFERENCE["x"], (height, width), mode, align_corners == "True", antialias == "True"
    )
    assert np.abs(actual - _REFERENCE[case]).max() < 2e-5


def test_jax_matches_numpy() -> None:
    """The device implementation agrees with the host implementation it shares weights with."""
    from rfdetr_jax.utilities.interpolate import resize_2d

    for case in _CASES:
        mode, align_corners, antialias, size = case.split("|")
        height, width = (int(value) for value in size.split("x"))
        device = np.asarray(
            resize_2d(_REFERENCE["x"], (height, width), mode, align_corners == "True", antialias == "True")
        )
        assert np.abs(device - _REFERENCE[case]).max() < 2e-5


def test_rejects_unknown_mode() -> None:
    """An unsupported mode name is rejected rather than silently mishandled."""
    with pytest.raises(ValueError, match="Unsupported interpolate mode"):
        resize_2d_numpy(np.zeros((1, 1, 4, 4), dtype="float32"), (2, 2), "lanczos")


def test_antialias_with_align_corners_is_explicitly_unsupported() -> None:
    with pytest.raises(NotImplementedError, match="align_corners"):
        resize_2d_numpy(np.zeros((1, 1, 4, 4), dtype="float32"), (8, 8), "bicubic", True, True)


def test_float64_inputs_resample_with_float64_weights() -> None:
    """A float64 resize carries float64 weights, as ATen computes them for double inputs."""
    import jax
    import jax.numpy as jnp

    from rfdetr_jax.utilities.interpolate import resize_2d
    from rfdetr_jax.utilities.resample import border_weights, weights_for_axis

    x = np.asarray(_REFERENCE["x"], dtype=np.float64)
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        out = np.asarray(resize_2d(jnp.asarray(x), (16, 16), "bilinear"))
    finally:
        jax.config.update("jax_enable_x64", previous)
    assert out.dtype == np.float64
    wh = border_weights(x.shape[-2], 16, "bilinear", False, np.float64)
    ww = border_weights(x.shape[-1], 16, "bilinear", False, np.float64)
    expected = np.einsum("oh,nchw,pw->ncop", wh, x, ww)
    assert np.abs(out - expected).max() < 1e-13
    host = resize_2d_numpy(x, (16, 16), "bilinear")
    assert host.dtype == np.float64
    assert np.abs(host - expected).max() < 1e-13
    assert weights_for_axis(x.shape[-2], 16, "bilinear", False, False, "float64").dtype == np.float64
    assert weights_for_axis(x.shape[-2], 16, "bilinear", False, False).dtype == np.float32
