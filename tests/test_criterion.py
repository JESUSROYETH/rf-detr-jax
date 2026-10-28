"""Set-prediction losses and the padded Hungarian matcher."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfdetr_jax.models.criterion import (
    CriterionConfig,
    Targets,
    compute_losses,
    num_boxes_for_targets,
    weighted_total,
)
from rfdetr_jax.models.matcher import build_cost_matrix, hungarian_match


def _outputs(batch: int = 2, queries: int = 20, classes: int = 7, seed: int = 0) -> dict:
    """Build a small random output dict with aux and encoder layers."""
    rng = np.random.RandomState(seed)

    def layer() -> dict:
        """One layer of random logits and well-formed boxes."""
        centers = rng.uniform(0.2, 0.8, size=(batch, queries, 2)).astype("float32")
        sizes = rng.uniform(0.05, 0.3, size=(batch, queries, 2)).astype("float32")
        return {
            "pred_logits": jnp.asarray(rng.randn(batch, queries, classes).astype("float32")),
            "pred_boxes": jnp.asarray(np.concatenate([centers, sizes], axis=-1)),
        }

    outputs = layer()
    outputs["aux_outputs"] = [layer()]
    outputs["enc_outputs"] = layer()
    return outputs


def _targets(batch: int = 2, max_targets: int = 4) -> Targets:
    """Build padded targets where the second image has fewer objects than the first."""
    labels = np.zeros((batch, max_targets), dtype="int32")
    boxes = np.zeros((batch, max_targets, 4), dtype="float32")
    mask = np.zeros((batch, max_targets), dtype=bool)
    counts = [3, 1]
    rng = np.random.RandomState(1)
    for image in range(batch):
        count = counts[image % len(counts)]
        labels[image, :count] = rng.randint(0, 7, size=count)
        boxes[image, :count, :2] = rng.uniform(0.3, 0.7, size=(count, 2))
        boxes[image, :count, 2:] = rng.uniform(0.1, 0.2, size=(count, 2))
        mask[image, :count] = True
    return Targets(labels=jnp.asarray(labels), boxes=jnp.asarray(boxes), mask=jnp.asarray(mask))


def test_num_boxes_multiplies_by_group_count() -> None:
    """The denominator counts real targets once per DETR group and never drops below one."""
    targets = _targets()
    assert float(num_boxes_for_targets(targets, 1, False)) == 4.0
    assert float(num_boxes_for_targets(targets, 13, False)) == 52.0
    assert float(num_boxes_for_targets(targets, 13, True)) == 4.0
    empty = Targets(
        labels=jnp.zeros((1, 3), jnp.int32), boxes=jnp.zeros((1, 3, 4)), mask=jnp.zeros((1, 3), bool)
    )
    assert float(num_boxes_for_targets(empty, 1, False)) == 1.0


def test_matcher_assigns_each_target_once() -> None:
    """Every real target receives exactly one distinct query, and padded slots receive none."""
    outputs = _outputs()
    targets = _targets()
    cost = build_cost_matrix(
        outputs["pred_logits"], outputs["pred_boxes"], targets.labels, targets.boxes, targets.mask,
        2.0, 5.0, 2.0, 0.25,
    )
    rows, cols, valid = hungarian_match(cost, targets.sizes, group_detr=1)
    rows, cols, valid = np.asarray(rows), np.asarray(cols), np.asarray(valid)
    for image, size in enumerate(np.asarray(targets.sizes)):
        assert valid[image].sum() == size
        assert len(set(rows[image][valid[image]].tolist())) == size
        assert sorted(cols[image][valid[image]].tolist()) == list(range(size))


def test_matcher_groups_are_independent() -> None:
    """With ``group_detr`` groups each group contributes its own full assignment."""
    outputs = _outputs(queries=20)
    targets = _targets()
    cost = build_cost_matrix(
        outputs["pred_logits"], outputs["pred_boxes"], targets.labels, targets.boxes, targets.mask,
        2.0, 5.0, 2.0, 0.25,
    )
    _, _, valid = hungarian_match(cost, targets.sizes, group_detr=4)
    assert np.asarray(valid).sum(1).tolist() == [3 * 4, 1 * 4]


def test_matcher_handles_an_empty_image() -> None:
    """An image with no targets yields no matches instead of failing."""
    outputs = _outputs(batch=1, queries=8)
    targets = Targets(
        labels=jnp.zeros((1, 3), jnp.int32), boxes=jnp.zeros((1, 3, 4)), mask=jnp.zeros((1, 3), bool)
    )
    cost = build_cost_matrix(
        outputs["pred_logits"], outputs["pred_boxes"], targets.labels, targets.boxes, targets.mask,
        2.0, 5.0, 2.0, 0.25,
    )
    _, _, valid = hungarian_match(cost, targets.sizes, group_detr=1)
    assert not np.asarray(valid).any()


@pytest.mark.parametrize("ia_bce", [True, False])
def test_losses_are_finite_and_named(ia_bce: bool) -> None:
    """Both classification variants produce finite, correctly keyed losses."""
    config = CriterionConfig(num_classes=7, group_detr=1, ia_bce_loss=ia_bce)
    losses = compute_losses(_outputs(), _targets(), config, train=False)
    expected = {
        "loss_ce", "loss_bbox", "loss_giou",
        "loss_ce_0", "loss_bbox_0", "loss_giou_0",
        "loss_ce_enc", "loss_bbox_enc", "loss_giou_enc",
    }
    assert set(losses) == expected
    assert all(np.isfinite(float(value)) for value in losses.values())


def test_losses_are_differentiable_under_jit() -> None:
    """The whole criterion, matcher callback included, traces and differentiates."""
    config = CriterionConfig(num_classes=7, group_detr=1)
    weights = config.weight_dict(2)

    def total(outputs: dict, targets: Targets) -> jnp.ndarray:
        """Weighted total loss for one batch."""
        return weighted_total(compute_losses(outputs, targets, config, train=False), weights)

    grads = jax.jit(jax.grad(total))(_outputs(), _targets())
    assert np.isfinite(np.asarray(grads["pred_logits"])).all()
    assert np.abs(np.asarray(grads["pred_boxes"])).max() > 0


def test_weight_dict_covers_aux_and_encoder_layers() -> None:
    """Auxiliary and encoder losses each get the base weight under their own suffix."""
    weights = CriterionConfig().weight_dict(dec_layers=3)
    assert weights["loss_bbox"] == weights["loss_bbox_0"] == weights["loss_bbox_enc"] == 5.0
    assert "loss_bbox_1" in weights and "loss_bbox_2" not in weights


def _keypoint_batch() -> tuple[dict, Targets, CriterionConfig]:
    """Build a small keypoint-enabled batch, targets and loss configuration."""
    schema = (3, 2)
    padded = len(schema) * max(schema)
    rng = np.random.RandomState(3)
    outputs = _outputs(batch=2, queries=12, classes=5)
    outputs["pred_keypoints"] = jnp.asarray(rng.randn(2, 12, padded, 8).astype("float32"))
    for aux in outputs["aux_outputs"]:
        aux["pred_keypoints"] = jnp.asarray(rng.randn(2, 12, padded, 8).astype("float32"))
    outputs["enc_outputs"]["pred_keypoints"] = jnp.asarray(rng.randn(2, 12, padded, 8).astype("float32"))

    keypoints = rng.uniform(0.1, 0.9, size=(2, 4, max(schema), 3)).astype("float32")
    keypoints[..., 2] = rng.randint(0, 3, size=(2, 4, max(schema)))
    base = _targets()
    targets = Targets(
        labels=jnp.asarray(np.array([[0, 1, 0, 0], [1, 0, 0, 0]], dtype="int32")),
        boxes=base.boxes,
        mask=base.mask,
        keypoints=jnp.asarray(keypoints),
    )
    config = CriterionConfig(
        num_classes=5,
        group_detr=1,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=schema,
        keypoint_l1_loss_coef=5.0,
        keypoint_findable_loss_coef=1.0,
        keypoint_visible_loss_coef=1.0,
        keypoint_nll_loss_coef=0.5,
    )
    return outputs, targets, config


def test_keypoint_losses_are_named_and_finite() -> None:
    """Enabling keypoints adds four finite loss terms on every output layer."""
    outputs, targets, config = _keypoint_batch()
    losses = compute_losses(outputs, targets, config, train=False)
    for suffix in ("", "_0", "_enc"):
        for name in ("l1", "findable", "visible", "nll"):
            key = f"loss_keypoints_{name}{suffix}"
            assert key in losses
            assert np.isfinite(float(losses[key]))


def test_keypoint_weight_dict_covers_every_layer() -> None:
    """Keypoint coefficients propagate to the auxiliary and encoder suffixes."""
    _, _, config = _keypoint_batch()
    weights = config.weight_dict(dec_layers=2)
    assert weights["loss_keypoints_l1"] == 5.0
    assert weights["loss_keypoints_nll_enc"] == 0.5
    assert weights["loss_keypoints_findable_0"] == 1.0


def test_keypoint_losses_are_differentiable_under_jit() -> None:
    """The keypoint terms, matcher costs included, trace and differentiate."""
    outputs, targets, config = _keypoint_batch()
    weights = config.weight_dict(2)

    def total(model_outputs: dict, batch_targets: Targets) -> jnp.ndarray:
        """Weighted total loss for one keypoint batch."""
        return weighted_total(compute_losses(model_outputs, batch_targets, config, train=False), weights)

    grads = jax.jit(jax.grad(total))(outputs, targets)
    assert np.isfinite(np.asarray(grads["pred_keypoints"])).all()
    assert np.abs(np.asarray(grads["pred_keypoints"])).max() > 0


def _reference_mask_costs(pred_masks: np.ndarray, target_masks: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Upstream's pairwise point-sampled mask costs, written out in NumPy for one image."""

    def sample_bilinear_border(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        height, width = image.shape
        x = np.clip(pts[:, 0] * width - 0.5, 0, width - 1)
        y = np.clip(pts[:, 1] * height - 0.5, 0, height - 1)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
        wx, wy = x - x0, y - y0
        return (image[y0, x0] * (1 - wx) * (1 - wy) + image[y0, x1] * wx * (1 - wy)
                + image[y1, x0] * (1 - wx) * wy + image[y1, x1] * wx * wy)

    def sample_nearest(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        height, width = image.shape
        x = np.clip(np.floor(pts[:, 0] * width).astype(int), 0, width - 1)
        y = np.clip(np.floor(pts[:, 1] * height).astype(int), 0, height - 1)
        return image[y, x]

    logits = np.stack([sample_bilinear_border(mask, points) for mask in pred_masks])  # [Q, P]
    labels = np.stack([sample_nearest(mask, points) for mask in target_masks])  # [T, P]
    pos = np.logaddexp(0, -logits)
    neg = np.logaddexp(0, logits)
    ce = (pos @ labels.T + neg @ (1 - labels).T) / points.shape[0]
    probabilities = 1 / (1 + np.exp(-logits))
    dice = 1 - (2 * probabilities @ labels.T + 1) / (probabilities.sum(-1)[:, None] + labels.sum(-1)[None, :] + 1)
    return 5.0 * ce + 5.0 * dice


def test_cost_matrix_adds_point_sampled_mask_costs_like_upstream() -> None:
    """Segmentation matching costs include the BCE and Dice terms upstream weights at 5.0 each."""
    outputs = _outputs(queries=6)
    targets = _targets()
    rng = np.random.RandomState(4)
    pred_masks = rng.randn(2, 6, 8, 8).astype("float32") * 3
    target_masks = (rng.rand(2, 4, 16, 16) > 0.5).astype("float32")
    points = rng.rand(9, 2).astype("float32")
    base = build_cost_matrix(outputs["pred_logits"], outputs["pred_boxes"], targets.labels, targets.boxes,
                             targets.mask, 2.0, 5.0, 2.0, 0.25)
    with_masks = build_cost_matrix(outputs["pred_logits"], outputs["pred_boxes"], targets.labels, targets.boxes,
                                   targets.mask, 2.0, 5.0, 2.0, 0.25, pred_masks=jnp.asarray(pred_masks),
                                   target_masks=jnp.asarray(target_masks), mask_costs=(5.0, 5.0),
                                   mask_points=jnp.asarray(points))
    for image in range(2):
        expected = _reference_mask_costs(pred_masks[image], target_masks[image], points)
        valid = np.asarray(targets.mask[image])
        observed = np.asarray(with_masks[image] - base[image])[:, valid]
        np.testing.assert_allclose(observed, expected[:, valid], rtol=1e-5, atol=1e-5)


def test_layer_cost_samples_mask_points_from_the_key() -> None:
    """Segmentation layers need a key; the draw is ``Hm*Wm // ratio`` shared points, like ``torch.rand(1, P, 2)``."""
    from rfdetr_jax.models.criterion import CriterionConfig, layer_cost

    outputs = _outputs(queries=6)
    targets = _targets()
    rng = np.random.RandomState(5)
    layer = dict(outputs, pred_masks=jnp.asarray(rng.randn(2, 6, 8, 8).astype("float32")))
    layer.pop("aux_outputs")
    layer.pop("enc_outputs")
    targets = Targets(labels=targets.labels, boxes=targets.boxes, mask=targets.mask,
                      masks=jnp.asarray((rng.rand(2, 4, 16, 16) > 0.5).astype("float32")))
    config = CriterionConfig(num_classes=7, segmentation_head=True, mask_point_sample_ratio=16)
    with pytest.raises(ValueError, match="PRNG key"):
        layer_cost(layer, targets, config)
    key = jax.random.key(11)
    from_key = layer_cost(layer, targets, config, key=key)
    explicit = layer_cost(layer, targets, config, mask_points=jax.random.uniform(key, (4, 2)))
    np.testing.assert_allclose(np.asarray(from_key), np.asarray(explicit), rtol=1e-6)
    without = layer_cost({k: v for k, v in layer.items() if k != "pred_masks"}, targets, config)
    assert not np.allclose(np.asarray(from_key), np.asarray(without))


@pytest.mark.parametrize("option", ["use_varifocal_loss", "use_position_supervised_loss"])
def test_unimplemented_classification_loss_variants_fail_explicitly(option: str) -> None:
    from rfdetr_jax.models.criterion import CriterionConfig

    with pytest.raises(NotImplementedError, match=option):
        CriterionConfig(**{option: True})
