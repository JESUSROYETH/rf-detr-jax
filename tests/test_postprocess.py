"""Top-k selection and box rescaling."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rfdetr_jax.models.postprocess import post_process


def test_selects_highest_scoring_pairs() -> None:
    """The top query/class pair is returned first, with its class and query recovered."""
    logits = jnp.full((1, 4, 3), -10.0).at[0, 2, 1].set(10.0)
    boxes = jnp.tile(jnp.asarray([0.5, 0.5, 0.2, 0.2]), (1, 4, 1))
    scores, labels, pixel_boxes = post_process(logits, boxes, jnp.asarray([[100, 200]]), num_select=3)
    assert float(scores[0, 0]) > 0.99
    assert int(labels[0, 0]) == 1
    assert np.allclose(np.asarray(pixel_boxes[0, 0]), [80.0, 40.0, 120.0, 60.0])


def test_boxes_are_clamped_to_the_image() -> None:
    """Unbounded regression is clamped into the image rather than returned out of range."""
    logits = jnp.zeros((1, 1, 1))
    boxes = jnp.asarray([[[0.5, 0.5, 4.0, 4.0]]])
    _, _, pixel_boxes = post_process(logits, boxes, jnp.asarray([[100, 200]]), num_select=1)
    assert np.allclose(np.asarray(pixel_boxes[0, 0]), [0.0, 0.0, 200.0, 100.0])


def test_num_select_caps_at_available_pairs() -> None:
    """Requesting more detections than exist returns every available pair."""
    logits = jnp.zeros((1, 2, 3))
    boxes = jnp.tile(jnp.asarray([0.5, 0.5, 0.1, 0.1]), (1, 2, 1))
    scores, _, _ = post_process(logits, boxes, jnp.asarray([[10, 10]]), num_select=100)
    assert scores.shape == (1, 6)
