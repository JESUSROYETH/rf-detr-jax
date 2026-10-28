"""Box conversions and IoU variants."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rfdetr_jax.utilities.box_ops import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    elementwise_generalized_box_iou,
    generalized_box_iou,
)


def test_cxcywh_roundtrip() -> None:
    """Converting to xyxy and back recovers the original boxes."""
    boxes = jnp.asarray([[0.5, 0.5, 0.2, 0.4], [0.1, 0.9, 0.05, 0.05]])
    assert np.allclose(np.asarray(box_xyxy_to_cxcywh(box_cxcywh_to_xyxy(boxes))), np.asarray(boxes), atol=1e-6)


def test_negative_sizes_are_clamped() -> None:
    """A negative width or height collapses to a zero-size box rather than inverting it."""
    converted = np.asarray(box_cxcywh_to_xyxy(jnp.asarray([[0.5, 0.5, -0.2, -0.4]])))
    assert converted[0, 0] == converted[0, 2] and converted[0, 1] == converted[0, 3]


def test_identical_boxes_have_giou_one() -> None:
    """Identical non-degenerate boxes give GIoU exactly 1.0."""
    boxes = jnp.asarray([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 4.0]])
    assert np.allclose(np.diag(np.asarray(generalized_box_iou(boxes, boxes))), 1.0, atol=1e-6)


def test_elementwise_matches_pairwise_diagonal() -> None:
    """The elementwise GIoU equals the diagonal of the pairwise matrix."""
    rng = np.random.RandomState(0)
    a = np.sort(rng.rand(8, 2), axis=1)
    b = np.sort(rng.rand(8, 2), axis=1)
    boxes1 = jnp.asarray(np.concatenate([a, a + 0.3], axis=1).astype("float32"))
    boxes2 = jnp.asarray(np.concatenate([b, b + 0.2], axis=1).astype("float32"))
    pairwise = np.diag(np.asarray(generalized_box_iou(boxes1, boxes2)))
    assert np.allclose(np.asarray(elementwise_generalized_box_iou(boxes1, boxes2)), pairwise, atol=1e-6)


def test_disjoint_boxes_have_negative_giou() -> None:
    """Far-apart boxes score below zero, as GIoU requires."""
    boxes1 = jnp.asarray([[0.0, 0.0, 1.0, 1.0]])
    boxes2 = jnp.asarray([[10.0, 10.0, 11.0, 11.0]])
    assert float(generalized_box_iou(boxes1, boxes2)[0, 0]) < 0.0
