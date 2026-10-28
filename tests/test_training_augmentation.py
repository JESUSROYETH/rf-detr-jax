"""Multi-scale inputs retain the upstream interpolation and normalized-target contract."""
import random

import jax.numpy as jnp
import numpy as np
import pytest

from rfdetr_jax.training.augmentation import compute_multi_scale_scales, resize_training_batch, scale_for_step


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("geometry", [(384, 16, 2), (312, 12, 2), (576, 12, 2), (32, 16, 1)])
def test_scale_choices_match_reference(expanded, geometry):
    pytest.importorskip("rfdetr")
    from rfdetr.datasets.coco import compute_multi_scale_scales as reference

    resolution, patch, windows = geometry
    scales = compute_multi_scale_scales(resolution, expanded, patch, windows)
    assert scales == reference(resolution, expanded, patch, windows)
    for step in (0, 1, 2, 14, 100):
        assert scale_for_step(scales, step) == random.Random(step).choice(scales)


@pytest.mark.parametrize("source,scale", [(32, 48), (48, 32), (72, 48), (32, 112)])
def test_resize_matches_torch_without_changing_normalized_targets(source, scale):
    torch = pytest.importorskip("torch")
    pixels = np.random.default_rng(3).normal(size=(2, 3, source, source)).astype(np.float32)
    batch = {"pixel_values": jnp.array(pixels), "boxes": jnp.array([[[.5, .5, .2, .2]]] * 2),
             "masks": jnp.ones((2, 1, 32, 32)), "keypoints": jnp.array([[[[.4, .8, 2]]]] * 2)}
    resized = resize_training_batch(batch, scale)
    expected = torch.nn.functional.interpolate(torch.from_numpy(pixels), size=scale,
                                               mode="bilinear", align_corners=False)
    np.testing.assert_allclose(resized["pixel_values"], expected.numpy(), atol=5e-7)
    for key in ("boxes", "masks", "keypoints"):
        assert resized[key] is batch[key]
    assert batch["pixel_values"].shape[-2:] == (source, source)
