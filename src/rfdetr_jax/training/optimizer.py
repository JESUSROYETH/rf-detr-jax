# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""Optimizer construction, ported from ``rfdetr.training.param_groups`` and ``configure_optimizers``.

PyTorch expresses RF-DETR's layer-wise decay as one AdamW parameter group per tensor. Optax has no
parameter groups, so the same schedule is expressed as two pytrees — a per-parameter learning-rate
multiplier and a per-parameter weight decay — applied by :func:`scale_by_learning_rate_tree` and
:func:`add_decayed_weights_tree`. The resulting update is identical to decoupled AdamW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax

from rfdetr_jax.convert import flatten_params, unflatten_params

Array = jax.Array

#: Substrings that zero a backbone parameter's weight decay, per ``get_dinov2_weight_decay_rate``.
_NO_DECAY_MARKERS = ("gamma", "pos_embed", "rel_pos", "bias", "norm", "embeddings")


@dataclass(frozen=True)
class OptimizerSpec:
    """Learning rates and decay factors, mirroring the matching ``TrainConfig`` fields."""

    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    weight_decay: float = 1e-4
    lr_vit_layer_decay: float = 0.8
    lr_component_decay: float = 0.7
    clip_max_norm: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


def dinov2_lr_decay_rate(path: str, lr_decay_rate: float, num_layers: int) -> float:
    """Return the ViT layer-wise LR multiplier for one backbone parameter.

    Mirrors ``get_dinov2_lr_decay_rate``: embeddings sit at layer 0, encoder block ``i`` at layer
    ``i + 1``, and anything else at ``num_layers + 1`` (multiplier 1.0).
    """
    layer_id = num_layers + 1
    if path.startswith("backbone"):
        if "embeddings" in path:
            layer_id = 0
        elif ".layer_" in path:
            marker = path[path.find(".layer_") + len(".layer_") :]
            layer_id = int(marker.split(".")[0]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def dinov2_weight_decay_rate(path: str) -> float:
    """Return ``0.0`` for norms, biases and embeddings, and ``1.0`` otherwise."""
    return 0.0 if any(marker in path for marker in _NO_DECAY_MARKERS) else 1.0


def build_param_group_trees(
    params: dict,
    spec: OptimizerSpec,
    out_feature_indexes: tuple[int, ...],
    *,
    lite_refpoint_refine: bool = True,
) -> tuple[dict, dict]:
    """Build per-parameter learning-rate and weight-decay pytrees.

    Args:
        params: The Flax parameter tree.
        spec: Learning rates and decay factors.
        out_feature_indexes: Backbone output stages, used to size the layer-decay schedule.
        lite_refpoint_refine: With iterative refinement (``False``) upstream registers the shared
            box head under ``transformer.decoder.bbox_embed`` first, so it takes the decoder rate.

    Returns:
        A ``(learning_rates, weight_decays)`` pair of pytrees matching ``params``.
    """
    num_layers = out_feature_indexes[-1] + 1
    learning_rates: dict[tuple[str, ...], Any] = {}
    weight_decays: dict[tuple[str, ...], Any] = {}

    for path in flatten_params(params):
        dotted = ".".join(path)
        # The upstream grouping keys off `backbone.0.encoder`, which is this port's `backbone_0.encoder`;
        # `backbone_0.projector` deliberately falls through to the default group.
        if dotted.startswith("backbone_0.encoder"):
            rate = dinov2_lr_decay_rate(dotted, spec.lr_vit_layer_decay, num_layers)
            learning_rates[path] = spec.lr_encoder * rate * spec.lr_component_decay**2
            weight_decays[path] = spec.weight_decay * dinov2_weight_decay_rate(dotted)
        elif "transformer.decoder" in dotted or (not lite_refpoint_refine and path[0] == "bbox_embed"):
            learning_rates[path] = spec.lr * spec.lr_component_decay
            weight_decays[path] = spec.weight_decay
        else:
            learning_rates[path] = spec.lr
            weight_decays[path] = spec.weight_decay

    return unflatten_params(learning_rates), unflatten_params(weight_decays)


def scale_by_learning_rate_tree(learning_rates: dict) -> optax.GradientTransformation:
    """Scale updates by a per-parameter learning rate and a shared schedule factor."""

    def init_fn(params: dict) -> optax.EmptyState:
        """No state is needed; the multipliers are static."""
        del params
        return optax.EmptyState()

    def update_fn(updates: dict, state: optax.EmptyState, params: dict | None = None) -> tuple[dict, Any]:
        """Multiply each update by ``-learning_rate`` for its parameter."""
        del params
        scaled = jax.tree.map(lambda update, rate: update * -rate, updates, learning_rates)
        return scaled, state

    return optax.GradientTransformation(init_fn, update_fn)


def add_decayed_weights_tree(weight_decays: dict) -> optax.GradientTransformation:
    """Add ``weight_decay * param`` to each update, the decoupled AdamW decay term."""

    def init_fn(params: dict) -> optax.EmptyState:
        """No state is needed; the decay rates are static."""
        del params
        return optax.EmptyState()

    def update_fn(updates: dict, state: optax.EmptyState, params: dict | None = None) -> tuple[dict, Any]:
        """Fold the decay term into the update."""
        if params is None:
            raise ValueError("add_decayed_weights_tree requires params; wrap it in a transform that passes them.")
        decayed = jax.tree.map(
            lambda update, param, decay: update + decay * param, updates, params, weight_decays
        )
        return decayed, state

    return optax.GradientTransformation(init_fn, update_fn)


@jax.tree_util.register_dataclass
@dataclass
class FusedAdamWState:
    """State for :func:`fused_adamw`: the update counter and Adam's two moment trees.

    Attributes:
        count: Updates applied so far, matching what the optax chain's counters hold.
        mu: First-moment tree.
        nu: Second-moment tree.
    """

    count: Array
    mu: dict
    nu: dict


def fused_adamw(
    learning_rates: dict,
    weight_decays: dict,
    spec: OptimizerSpec,
    schedule: optax.Schedule | None = None,
) -> optax.GradientTransformation:
    """RF-DETR's AdamW as one pass over the parameter tree instead of five.

    ``optax.chain`` composes global-norm clipping, Adam, the decoupled decay term, the
    per-parameter rate and the schedule as separate transformations, and each is its own
    ``tree_map`` over every leaf. On a tree with several hundred small leaves that is several
    hundred kernels reading the same buffers again and again, and this model spends more time there
    than in its own backward pass.

    The arithmetic is unchanged, term for term, including optax's bias correction and the exact
    branch its clipping takes; it is just evaluated in a single ``tree_map``, so each parameter's
    gradient, moments and value are read once. ``tests/test_optimizer.py`` pins the result against
    the chain it replaces.

    Args:
        learning_rates: Per-parameter learning rates, shaped like the parameter tree.
        weight_decays: Per-parameter decay rates, shaped like the parameter tree.
        spec: Adam's betas and epsilon, and the clipping threshold.
        schedule: Optional step-indexed multiplier, applied as ``optax.scale_by_schedule`` would.

    Returns:
        An ``optax`` transformation with the same signature as the chain.
    """
    clip_norm = spec.clip_max_norm if spec.clip_max_norm and spec.clip_max_norm > 0 else None

    def init_fn(params: dict) -> FusedAdamWState:
        """Start both moment trees at zero, as ``optax.scale_by_adam`` does."""
        zeros = jax.tree.map(jnp.zeros_like, params)
        return FusedAdamWState(count=jnp.zeros((), jnp.int32), mu=zeros, nu=jax.tree.map(jnp.zeros_like, params))

    def update_fn(updates: dict, state: FusedAdamWState, params: dict | None = None):
        """Clip, take one Adam step, decay, scale and schedule, in one traversal."""
        if params is None:
            raise ValueError("fused_adamw needs params for the decoupled weight-decay term")

        # `optax.clip_by_global_norm`: leave the tree alone below the threshold, rescale above it.
        keep = None
        if clip_norm is not None:
            keep = jnp.squeeze(optax.global_norm(updates) < clip_norm)
            global_norm = optax.global_norm(updates)

        count = state.count + 1
        correct_mu = 1 - spec.beta1**count
        correct_nu = 1 - spec.beta2**count
        factor = jnp.asarray(schedule(state.count) if schedule is not None else 1.0)

        def one(gradient, mu, nu, param, rate, decay):
            """Everything the chain does to a single parameter, in one fused expression."""
            if clip_norm is not None:
                gradient = jax.lax.select(
                    keep, gradient, (gradient / global_norm.astype(gradient.dtype)) * clip_norm
                )
            mu = spec.beta1 * mu + (1 - spec.beta1) * gradient
            nu = spec.beta2 * nu + (1 - spec.beta2) * jnp.square(gradient)
            update = (mu / correct_mu.astype(mu.dtype)) / (
                jnp.sqrt(nu / correct_nu.astype(nu.dtype)) + spec.eps
            )
            update = (update + decay * param) * -rate * factor.astype(param.dtype)
            return update, mu, nu

        combined = jax.tree.map(
            one, updates, state.mu, state.nu, params, learning_rates, weight_decays
        )
        structure = jax.tree.structure(updates)
        scaled, mu, nu = (
            jax.tree.unflatten(structure, [leaf[index] for leaf in jax.tree.leaves(combined, is_leaf=_is_triple)])
            for index in range(3)
        )
        return scaled, FusedAdamWState(count=count, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


def _is_triple(node) -> bool:
    """Treat the ``(update, mu, nu)`` tuple a fused leaf returns as a leaf itself."""
    return isinstance(node, tuple) and len(node) == 3


def build_optimizer(
    params: dict,
    spec: OptimizerSpec,
    out_feature_indexes: tuple[int, ...],
    schedule: optax.Schedule | None = None,
    fused: bool = True,
    lite_refpoint_refine: bool = True,
) -> optax.GradientTransformation:
    """Assemble the AdamW optimizer with RF-DETR's layer-wise learning rates.

    Args:
        params: The parameter tree, used to shape the per-parameter multipliers.
        spec: Learning rates and decay factors.
        out_feature_indexes: Backbone output stages.
        schedule: Optional step-indexed multiplier applied on top of every per-parameter rate,
            i.e. the ``LambdaLR`` factor.
        fused: Whether to use :func:`fused_adamw`, which computes the same update in one pass over
            the tree. Set it to ``False`` for the stage-by-stage ``optax.chain``.

    Returns:
        The composed optax transformation.
    """
    learning_rates, weight_decays = build_param_group_trees(params, spec, out_feature_indexes,
                                                            lite_refpoint_refine=lite_refpoint_refine)
    if fused:
        return fused_adamw(learning_rates, weight_decays, spec, schedule)
    stages: list[optax.GradientTransformation] = []
    if spec.clip_max_norm and spec.clip_max_norm > 0:
        stages.append(optax.clip_by_global_norm(spec.clip_max_norm))
    stages.append(optax.scale_by_adam(b1=spec.beta1, b2=spec.beta2, eps=spec.eps))
    stages.append(add_decayed_weights_tree(weight_decays))
    stages.append(scale_by_learning_rate_tree(learning_rates))
    if schedule is not None:
        stages.append(optax.scale_by_schedule(schedule))
    return optax.chain(*stages)


def managed_schedule(
    preset: str,
    total_steps: int,
    steps_per_epoch: int,
    warmup_steps: int,
    lr_drop: int,
    min_factor: float = 0.0,
) -> optax.Schedule:
    """Reproduce RF-DETR's built-in ``"step"`` / ``"cosine"`` ``LambdaLR`` factor.

    Args:
        preset: ``"step"`` or ``"cosine"``.
        total_steps: Total optimizer steps in the run.
        steps_per_epoch: Optimizer steps per epoch.
        warmup_steps: Linear warmup length in optimizer steps.
        lr_drop: Epoch after which the step schedule drops the LR 10x.
        min_factor: Floor for the cosine schedule.

    Returns:
        A schedule returning the multiplier for a given step count.
    """

    def schedule(step: Array) -> Array:
        """Return the LR multiplier for ``step``."""
        step = jnp.asarray(step, jnp.float32)
        warmup = step / max(1, warmup_steps)
        if preset == "cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            main = min_factor + (1 - min_factor) * 0.5 * (1 + jnp.cos(math.pi * progress))
        else:
            main = jnp.where(step < lr_drop * steps_per_epoch, 1.0, 0.1)
        return jnp.where(step < warmup_steps, warmup, main)

    return schedule


def ema_decay_at(step: Array, decay: float, tau: float) -> Array:
    """Return the EMA decay for an update index, matching ``ModelEma._get_decay``.

    ``updates`` is 1-indexed upstream, so ``step`` here is the 1-indexed update count.
    """
    if tau == 0:
        return jnp.asarray(decay, jnp.float32)
    return jnp.asarray(decay, jnp.float32) * (1 - jnp.exp(-jnp.asarray(step, jnp.float32) / tau))


def ema_update(ema_params: dict, params: dict, decay: Array) -> dict:
    """Return the EMA tree after one update: ``decay * ema + (1 - decay) * params``."""
    return jax.tree.map(lambda e, p: decay * e + (1.0 - decay) * p, ema_params, params)
