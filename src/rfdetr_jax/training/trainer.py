# ------------------------------------------------------------------------
# RF-DETR JAX
# Modified JAX/Flax port of RF-DETR.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""Training loop for the JAX RF-DETR port.

One ``jit``-compiled step covers the forward pass, the match, the loss, the AdamW update and the
EMA update. With ``CriterionConfig.matcher="auction"`` the step touches the host nowhere at all,
which is what lets :func:`data_parallel` spread it across every chip and lets the loop keep the
accelerator fed instead of blocking on each step's metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rfdetr_jax.convert import flatten_params, unflatten_params
from rfdetr_jax.models.criterion import CriterionConfig, Targets, compute_losses, num_boxes_for_targets, weighted_total
from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec, initial_class_bias
from rfdetr_jax.models.ops import sampling_offsets_bias_init
from rfdetr_jax.training.optimizer import (
    ema_decay_at,
    ema_update,
)

Array = jax.Array


@jax.tree_util.register_dataclass
@dataclass
class LossScaleState:
    """Dynamic FP16 scale and its persisted successful/skipped update counters."""

    scale: Array
    growth_tracker: Array
    skipped_steps: Array


@jax.tree_util.register_dataclass
@dataclass
class TrainState:
    """Mutable training state carried across steps.

    Attributes:
        params: Model parameters.
        opt_state: Optax optimizer state.
        ema_params: EMA-averaged parameters, or ``None`` when EMA is disabled.
        step: Optimizer steps taken so far.
        loss_scale: FP16 dynamic scale, or ``None`` for unscaled training. With scaling,
            ``step`` counts attempted updates, matching Lightning even when Adam is skipped.
    """

    params: dict
    opt_state: Any
    ema_params: dict | None
    step: Array
    loss_scale: LossScaleState | None = None


@dataclass(frozen=True)
class TrainSettings:
    """Loop-level knobs, mirroring the matching ``TrainConfig`` fields."""

    epochs: int = 5
    batch_size: int = 4
    max_targets: int = 20
    use_ema: bool = True
    ema_decay: float = 0.993
    ema_tau: int = 100
    ema_first_update_copy: bool = False
    lr_scheduler: str = "step"
    lr_drop: int = 100
    warmup_epochs: float = 0.0
    lr_min_factor: float = 0.0
    seed: int = 42
    grad_accum_steps: int = 1
    keypoint_box_normalized_accumulation: bool = False
    automatic_accumulation_normalization: bool = False
    freeze_encoder: bool = False
    loss_scaling: bool = False
    initial_loss_scale: float = 65536.
    loss_scale_growth_factor: float = 2.
    loss_scale_backoff_factor: float = .5
    loss_scale_growth_interval: int = 2000


def init_params(model: LWDETR, spec: LWDETRSpec, key: Array) -> dict:
    """Initialise parameters and apply RF-DETR's head initialisations.

    The upstream constructor zeroes the box head's final layer, sets the classification bias to the
    focal-loss prior, zeroes ``refpoint_embed``, and runs Xavier over every transformer parameter of
    rank > 1 plus the deformable-attention resets. The native policy implements
    these resets; it does not reproduce Torch RNG or every initialization distribution.

    Initialisation runs at the smallest valid input size rather than the training resolution. No
    parameter shape depends on the input — the position table is sized from
    ``positional_encoding_size``, not from the image — so this yields an identical tree while
    avoiding a full-resolution trace, which dominates start-up time.

    Args:
        model: The detector module.
        spec: The model spec, used for the head shapes.
        key: PRNG key.

    Returns:
        The initialised parameter tree.
    """
    init_resolution = spec.patch_size * spec.num_windows * 2
    dummy = jnp.zeros((1, 3, init_resolution, init_resolution), jnp.float32)
    mask = jnp.zeros((1, init_resolution, init_resolution), bool)
    params = model.init(key, dummy, mask, train=True)["params"]
    params = jax.tree.map(lambda x: x, params)

    flat = flatten_params(params)
    xavier_key = key

    for path, value in list(flat.items()):
        # Transformer._reset_parameters: Xavier over every rank > 1 parameter.
        if path[0] == "transformer" and value.ndim > 1:
            xavier_key, subkey = jax.random.split(xavier_key)
            flat[path] = jax.nn.initializers.xavier_uniform()(subkey, value.shape, value.dtype)

    for path, value in list(flat.items()):
        dotted = ".".join(path)
        # MSDeformAttn._reset_parameters, applied after the Xavier sweep as upstream does.
        if dotted.endswith("sampling_offsets.weight"):
            flat[path] = jnp.zeros_like(value)
        elif dotted.endswith("sampling_offsets.bias"):
            # Each deformable module sizes its own head count (the keypoint cross-attention divides
            # ``ca_nheads`` by the keypoint downscale), so derive it from the tensor.
            heads = value.shape[0] // (spec.num_feature_levels * spec.dec_n_points * 2)
            flat[path] = sampling_offsets_bias_init(heads, spec.num_feature_levels, spec.dec_n_points)
        elif dotted.endswith(
            ("attention_weights.weight", "attention_weights.bias", "value_proj.bias", "output_proj.bias")
        ):
            flat[path] = jnp.zeros_like(value)

    # LWDETR.__init__ head initialisation.
    flat[("class_embed", "bias")] = jnp.full_like(
        flat[("class_embed", "bias")], initial_class_bias(spec.detector_classes)
    )
    flat[("bbox_embed", "layers_2", "weight")] = jnp.zeros_like(flat[("bbox_embed", "layers_2", "weight")])
    flat[("bbox_embed", "layers_2", "bias")] = jnp.zeros_like(flat[("bbox_embed", "layers_2", "bias")])
    flat[("refpoint_embed",)] = jnp.zeros_like(flat[("refpoint_embed",)])
    for path, value in list(flat.items()):
        dotted = ".".join(path)
        if path[0] == "transformer" and path[-2:] == ("out_proj", "bias") and (*path[:-2], "in_proj_weight") in flat:
            # torch.nn.MultiheadAttention zeroes its output projection bias; plain Linear out_proj
            # modules (query initialiser, inst_out_proj) keep their default initialisation.
            flat[path] = jnp.zeros_like(value)
        elif dotted.endswith("adaLN_modulation_2.bias"):
            flat[path] = jnp.zeros_like(value)
        elif (path[:2] == ("backbone_0", "encoder") and path[-1] == "bias"
              and getattr(flat.get((*path[:-1], "weight")), "ndim", 0) in (2, 4)):
            # Hugging Face initialises Linear and Conv2d biases to zero.
            flat[path] = jnp.zeros_like(value)
    if ("keypoint_embed", "layers_2", "weight") in flat:
        # GroupPose zeroes the keypoint head's last layer before the encoder heads copy it.
        flat[("keypoint_embed", "layers_2", "weight")] = jnp.zeros_like(flat[("keypoint_embed", "layers_2", "weight")])
        flat[("keypoint_embed", "layers_2", "bias")] = jnp.zeros_like(flat[("keypoint_embed", "layers_2", "bias")])

    # The two-stage encoder heads are deep copies of the decoder heads made after those inits.
    # They must be genuine copies, not aliases: the jitted step donates its input buffers, and
    # donating one buffer through several tree slots is rejected by the runtime.
    def _copy(value: Array) -> Array:
        """Return a distinct device buffer holding the same values."""
        return jax.device_put(np.asarray(value))

    for group in range(spec.group_detr):
        for source, destination in (
            (("class_embed", "weight"), ("transformer", f"enc_out_class_embed_{group}", "weight")),
            (("class_embed", "bias"), ("transformer", f"enc_out_class_embed_{group}", "bias")),
        ):
            if destination in flat:
                flat[destination] = _copy(flat[source])
        for layer in range(3):
            for suffix in ("weight", "bias"):
                source = ("bbox_embed", f"layers_{layer}", suffix)
                destination = ("transformer", f"enc_out_bbox_embed_{group}", f"layers_{layer}", suffix)
                if destination in flat:
                    flat[destination] = _copy(flat[source])
                source = ("keypoint_embed", f"layers_{layer}", suffix)
                destination = ("transformer", f"enc_out_keypoint_embed_{group}", f"layers_{layer}", suffix)
                if destination in flat and source in flat:
                    flat[destination] = _copy(flat[source])

    return unflatten_params(flat)


def create_train_state(
    params: dict,
    optimizer: optax.GradientTransformation,
    use_ema: bool,
    *, loss_scaling: bool = False, initial_loss_scale: float = 65536.,
) -> TrainState:
    """Build the initial training state.

    The EMA tree is a real copy rather than an alias of ``params``: the jitted step donates its
    input buffers, and donating the same buffer twice is an error. ``jnp.copy`` can be elided by
    XLA and hand back the same buffer, so the copy round-trips through the host instead.
    """
    ema_params = jax.tree.map(lambda x: jax.device_put(np.asarray(x)), params) if use_ema else None
    if not np.isfinite(initial_loss_scale) or not 0 < initial_loss_scale <= np.finfo(np.float32).max:
        raise ValueError("initial_loss_scale must be finite, positive and representable in float32")
    loss_scale = (LossScaleState(jnp.float32(initial_loss_scale), jnp.int32(0), jnp.int32(0))
                  if loss_scaling else None)
    return TrainState(params=params, opt_state=optimizer.init(params), ema_params=ema_params,
                      step=jnp.int32(0), loss_scale=loss_scale)


def _shard_map():
    """Return ``(shard_map, replication-check keyword)`` across JAX versions."""
    try:
        from jax import shard_map  # JAX >= 0.7

        return shard_map, "check_vma"
    except ImportError:  # pragma: no cover - older JAX
        from jax.experimental.shard_map import shard_map

        return shard_map, "check_rep"


def make_train_step(
    model: LWDETR,
    optimizer: optax.GradientTransformation,
    criterion_config: CriterionConfig,
    weights: dict[str, float],
    settings: TrainSettings,
    batch_sharding=None,
    optimizer_schedule=None,
):
    """Build the jitted training step.

    Args:
        model: The detector.
        optimizer: The optax transformation.
        criterion_config: Loss configuration.
        weights: Loss weights from ``CriterionConfig.weight_dict``.
        settings: Loop-level knobs (EMA).
        batch_sharding: The batch sharding from :func:`data_parallel`. With it, the step runs as an
            explicit per-device program (``shard_map``): each device takes the forward pass, the
            match and the loss of its own slice, the loss denominator and the gradients are summed
            over the batch axis, and the replicated update follows. The loss is a sum over images
            divided by the global target count, so this is the same arithmetic as the unsplit step.
            Explicit per-device execution keeps the auction matcher's ``while_loop`` local rather
            than relying on sharding propagation to partition its work.
        optimizer_schedule: Optional schedule applied using attempted-step count after AdamW.
            When supplied, the optimizer itself must not also contain this schedule. FP16 uses
            this to advance LR on overflow without advancing Adam's bias-correction counter.

    Returns:
        A function ``(state, batch, dropout_key) -> (state, metrics)``.
    """
    if model.spec.segmentation_head != criterion_config.segmentation_head:
        raise ValueError("Model and criterion segmentation_head settings must agree")
    if model.spec.use_grouppose_keypoints != criterion_config.use_grouppose_keypoints:
        raise ValueError("Model and criterion use_grouppose_keypoints settings must agree")
    if model.spec.use_grouppose_keypoints:
        if model.spec.num_keypoints_per_class != criterion_config.num_keypoints_per_class:
            raise ValueError("Model and criterion keypoint schemas must agree")
        if not any(weight > 0 for name, weight in weights.items() if name.startswith("loss_keypoints_")):
            raise ValueError("Keypoint training requires a positive keypoint loss weight")
    if settings.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be positive")
    if settings.keypoint_box_normalized_accumulation and not criterion_config.use_grouppose_keypoints:
        raise ValueError("keypoint_box_normalized_accumulation requires a keypoint model")
    if settings.keypoint_box_normalized_accumulation and settings.automatic_accumulation_normalization:
        raise ValueError("manual keypoint and automatic accumulation normalization are mutually exclusive")
    if (not np.isfinite(settings.loss_scale_growth_factor) or settings.loss_scale_growth_factor <= 1 or
            not np.isfinite(settings.loss_scale_backoff_factor) or not 0 < settings.loss_scale_backoff_factor < 1 or
            type(settings.loss_scale_growth_interval) is not int or settings.loss_scale_growth_interval < 1):
        raise ValueError("loss scaling requires growth_factor > 1, 0 < backoff_factor < 1 and positive growth_interval")

    def make_loss_fn(config: CriterionConfig):
        def loss_fn(params: dict, batch: dict, dropout_key: Array, num_boxes=None) -> tuple[Array, dict]:
            """Forward, match and reduce the batch to a scalar loss."""
            if config.segmentation_head and "masks" not in batch:
                raise ValueError("Segmentation training requires batch['masks']")
            if config.use_grouppose_keypoints and "keypoints" not in batch:
                raise ValueError("Keypoint training requires batch['keypoints']")
            dropout_key, sampling_key = jax.random.split(dropout_key)
            if settings.freeze_encoder:
                params = {**params, "backbone_0": {**params["backbone_0"],
                          "encoder": jax.tree.map(jax.lax.stop_gradient, params["backbone_0"]["encoder"])}}
            outputs = model.apply(
                {"params": params},
                batch["pixel_values"],
                None,
                train=True,
                rngs={"dropout": dropout_key},
            )
            targets = Targets(
                labels=batch["labels"], boxes=batch["boxes"], mask=batch["target_mask"],
                masks=batch.get("masks") if config.segmentation_head else None,
                keypoints=batch.get("keypoints") if config.use_grouppose_keypoints else None,
            )
            losses = compute_losses(outputs, targets, config, train=True, key=sampling_key, num_boxes=num_boxes)
            return weighted_total(losses, weights), losses

        return loss_fn

    def gradient_step(loss_fn, params, batch, key, scale, config):
        """Average microbatch gradients, retaining activations for only one microbatch."""
        count = batch["pixel_values"].shape[0] if settings.grad_accum_steps > 1 else 1
        automatic_denominator = settings.grad_accum_steps if settings.automatic_accumulation_normalization else count
        if settings.keypoint_box_normalized_accumulation:
            def normalizer(microbatch):
                targets = Targets(microbatch["labels"], microbatch["boxes"], microbatch["target_mask"])
                return num_boxes_for_targets(targets, config.group_detr, config.sum_group_losses, config.axis_name)

            def objective(params, microbatch, microkey, cumulative, denominator):
                raw_total, raw_losses = loss_fn(params, microbatch, microkey, num_boxes=jnp.float32(1.))
                logged = {name: value / denominator if name in weights else value
                          for name, value in raw_losses.items()}
                return (raw_total / cumulative) * scale, (raw_total / denominator, logged)

            differentiate = jax.value_and_grad(objective, has_aux=True)
            if settings.grad_accum_steps == 1:
                denominator = normalizer(batch)
                (_, metrics), grads = differentiate(params, batch, key, denominator, denominator)
                return metrics, grads
            keys = jax.random.split(key, count)
            first = jax.tree.map(lambda value: value[0], batch)
            denominator = normalizer(first)
            (_, metrics), grads = differentiate(params, first, keys[0], denominator, denominator)

            def accumulate_manual(carry, values):
                accumulated_metrics, previous_grads, previous = carry
                microbatch, microkey = values
                denominator = normalizer(microbatch)
                cumulative = previous + denominator
                (_, metrics), grads = differentiate(params, microbatch, microkey, cumulative, denominator)
                # Match RFDETRModelModule._rescale_accumulated_gradients before each backward.
                grads = jax.tree.map(lambda old, new: old * (previous / cumulative) + new, previous_grads, grads)
                return (jax.tree.map(lambda old, new: old + new, accumulated_metrics, metrics), grads, cumulative), None

            (metrics, grads, _), _ = jax.lax.scan(
                accumulate_manual, (metrics, grads, denominator),
                (jax.tree.map(lambda value: value[1:], batch), keys[1:]),
            )
            return jax.tree.map(lambda value: value / count, metrics), grads

        def scaled_loss(params, batch, key):
            total, losses = loss_fn(params, batch, key)
            # Divide before backward: averaging scaled gradients afterwards overflows FP16
            # intermediates count times earlier, even when the effective gradient is finite.
            return (total / automatic_denominator) * scale, (total, losses)

        differentiate_scaled = jax.value_and_grad(scaled_loss, has_aux=True)

        def differentiate(params, batch, key):
            if not settings.loss_scaling:
                return jax.value_and_grad(loss_fn, has_aux=True)(params, batch, key)
            (_, unscaled), grads = differentiate_scaled(params, batch, key)
            return unscaled, grads
        if settings.grad_accum_steps == 1:
            return differentiate(params, batch, key)
        keys = jax.random.split(key, count)
        first = jax.tree.map(lambda value: value[0], batch)
        (total, losses), grads = differentiate(params, first, keys[0])

        def accumulate(carry, values):
            microbatch, microkey = values
            result = differentiate(params, microbatch, microkey)
            return jax.tree.map(lambda a, b: a + b, carry, result), None

        accumulated, _ = jax.lax.scan(
            accumulate, ((total, losses), grads),
            (jax.tree.map(lambda value: value[1:], batch), keys[1:]),
        )
        metrics, grads = accumulated
        if not settings.loss_scaling:
            grads = jax.tree.map(lambda value: value / automatic_denominator, grads)
        return jax.tree.map(lambda value: value / count, metrics), grads

    def advance_ema(state, params):
        ema_params = state.ema_params
        if ema_params is not None:
            decay = ema_decay_at(state.step + 1, settings.ema_decay, settings.ema_tau)
            ema_params = ema_update(ema_params, params, decay)
            if settings.ema_first_update_copy:
                # Lightning AveragedModel copies its first observed model, including a skipped Adam attempt.
                ema_params = jax.tree.map(lambda average, value: jnp.where(state.step == 0, value, average),
                                          ema_params, params)
            if settings.freeze_encoder:
                ema_params = {**ema_params, "backbone_0": {**ema_params["backbone_0"],
                              "encoder": state.ema_params["backbone_0"]["encoder"]}}
        return ema_params

    def update(state: TrainState, total: Array, losses: dict, grads: dict) -> tuple[TrainState, dict]:
        """Apply the optimizer and the EMA to already-reduced gradients."""
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        if optimizer_schedule is not None:
            # Lightning advances its scheduler on an attempted step even when GradScaler skips Adam.
            factor = jnp.asarray(optimizer_schedule(state.step))
            updates = jax.tree.map(lambda value: value * factor.astype(value.dtype), updates)
        params = optax.apply_updates(state.params, updates)
        if settings.freeze_encoder:
            # Freezing also excludes decoupled weight decay from custom optimizers.
            params = {**params, "backbone_0": {**params["backbone_0"],
                      "encoder": state.params["backbone_0"]["encoder"]}}

        ema_params = advance_ema(state, params)
        metrics = {"loss": total, **losses, "grad_norm": optax.global_norm(grads)}
        return TrainState(params, opt_state, ema_params, state.step + 1, state.loss_scale), metrics

    def finish_step(state, total, losses, grads):
        if not settings.loss_scaling:
            if state.loss_scale is not None:
                raise ValueError("scaled state requires loss_scaling=True")
            return update(state, total, losses, grads)
        if state.loss_scale is None:
            raise ValueError("loss_scaling=True requires a loss scaler in the training state")
        scaler = state.loss_scale
        grads = jax.tree.map(lambda value: value / scaler.scale, grads)
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(grads)]))
        candidate, metrics = update(state, total, losses, grads)
        selected = jax.tree.map(lambda fresh, old: jnp.where(finite, fresh, old), candidate, state)
        selected.step = state.step + 1
        selected.ema_params = advance_ema(state, selected.params)
        tracker = scaler.growth_tracker + 1
        grow = tracker == settings.loss_scale_growth_interval
        increased = scaler.scale * settings.loss_scale_growth_factor
        increased = jnp.where(jnp.isfinite(increased), increased, scaler.scale)
        selected.loss_scale = LossScaleState(
            jnp.where(finite, jnp.where(grow, increased, scaler.scale),
                      scaler.scale * settings.loss_scale_backoff_factor),
            jnp.where(finite & ~grow, tracker, 0),
            scaler.skipped_steps + (~finite).astype(jnp.int32),
        )
        metrics.update(loss_scale=selected.loss_scale.scale, overflow=(~finite).astype(jnp.float32))
        return selected, metrics

    if batch_sharding is None:
        loss_fn = make_loss_fn(criterion_config)

        def step(state: TrainState, batch: dict, dropout_key: Array) -> tuple[TrainState, dict]:
            """Run one optimizer step and the EMA update."""
            scale = state.loss_scale.scale if state.loss_scale is not None else jnp.float32(1.)
            (total, losses), grads = gradient_step(loss_fn, state.params, batch, dropout_key, scale, criterion_config)
            return finish_step(state, total, losses, grads)

        return jax.jit(step, donate_argnums=(0,))

    import dataclasses

    from jax.sharding import PartitionSpec

    axis = batch_sharding.spec[0]
    local_config = dataclasses.replace(criterion_config, axis_name=axis)
    local_loss_fn = make_loss_fn(local_config)

    def local_step(state: TrainState, batch: dict, dropout_key: Array) -> tuple[TrainState, dict]:
        """One device's share: its slice's loss, then gradients summed across the batch axis."""
        dropout_key = jax.random.fold_in(dropout_key, jax.lax.axis_index(axis))
        scale = state.loss_scale.scale if state.loss_scale is not None else jnp.float32(1.)
        (total, losses), grads = gradient_step(local_loss_fn, state.params, batch, dropout_key, scale, local_config)
        # Each device's loss is its images' share of the global sum, so the global loss and its
        # gradient are sums over devices.
        total, losses, grads = jax.lax.psum((total, losses, grads), axis)
        return finish_step(state, total, losses, grads)

    shard_map, check = _shard_map()
    sharded = shard_map(
        local_step,
        mesh=batch_sharding.mesh,
        in_specs=(PartitionSpec(), PartitionSpec(None, axis) if settings.grad_accum_steps > 1
                  else PartitionSpec(axis), PartitionSpec()),
        out_specs=(PartitionSpec(), PartitionSpec()),
        **{check: False},
    )
    return jax.jit(sharded, donate_argnums=(0,))


def data_parallel(state: TrainState, devices: list | None = None):
    """Replicate ``state`` across ``devices`` and return ``(state, batch_sharding)``.

    Data parallelism here is pure sharding propagation: parameters and optimizer state are
    replicated, the batch is split along its leading axis, and GSPMD derives the gradient
    all-reduce from that. Nothing in the step needs to change, provided the step is host-free —
    a ``pure_callback`` inside it would have to be resolved per shard.

    Args:
        state: The training state to replicate.
        devices: Devices to shard over; defaults to every local device.

    Returns:
        A ``(state, batch_sharding)`` pair. Pass ``batch_sharding`` to ``jax.device_put`` on each
        batch; the leading axis must be divisible by the device count.
    """
    devices = devices or jax.devices()
    # The mesh axis is explicitly an inferred one. Newer JAX defaults new meshes to explicit
    # sharding, under which mixing a replicated operand with a sharded one is an error rather than
    # something the compiler resolves — and this step does exactly that in several places.
    axis_type = getattr(jax.sharding, "AxisType", None)
    keywords = {"devices": devices}
    if axis_type is not None:
        keywords["axis_types"] = (axis_type.Auto,)
    mesh = jax.make_mesh((len(devices),), ("batch",), **keywords)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
    return jax.device_put(state, replicated), sharded


def make_eval_step(model: LWDETR):
    """Build a jitted evaluation forward pass returning logits and boxes."""

    def forward(params: dict, pixel_values: Array) -> tuple[Array, Array]:
        """Run the model in inference mode."""
        outputs = model.apply({"params": params}, pixel_values, None, train=False)
        return outputs["pred_logits"], outputs["pred_boxes"]

    return jax.jit(forward)


def make_predict_step(model: LWDETR, num_select: int, batch_sharding=None):
    """Build a jitted inference step: forward pass and post-processing in one executable.

    Args:
        model: The detector.
        num_select: Detections kept per image.
        batch_sharding: The batch sharding from :func:`data_parallel`, when evaluating across
            devices. Each device then runs the forward pass and the top-k on its own slice. That
            also keeps the sort off a batch-sharded array, which aborts the TPU compiler
            (``bitonic_sort_emitter``: "Stable sort expects more than 1 operand", JAX 0.11.2).

    Returns:
        A function ``(params, pixel_values, orig_sizes) -> (scores, labels, boxes)``.
    """
    from rfdetr_jax.models.postprocess import post_process

    def predict(params: dict, pixel_values: Array, orig_sizes: Array) -> tuple[Array, Array, Array]:
        """Run the model in inference mode and select the top detections."""
        outputs = model.apply({"params": params}, pixel_values, None, train=False)
        return post_process(outputs["pred_logits"], outputs["pred_boxes"], orig_sizes, num_select=num_select)

    if batch_sharding is None:
        return jax.jit(predict)

    from jax.sharding import PartitionSpec

    axis = batch_sharding.spec[0]
    shard_map, check = _shard_map()
    return jax.jit(
        shard_map(
            predict,
            mesh=batch_sharding.mesh,
            in_specs=(PartitionSpec(), PartitionSpec(axis), PartitionSpec(axis)),
            out_specs=(PartitionSpec(axis),) * 3,
            **{check: False},
        )
    )


def device_batches(batches, names: tuple[str, ...], sharding=None, depth: int = 2):
    """Move each batch's ``names`` arrays to the device on a background thread, ``depth`` ahead.

    The copy of the next batch then overlaps the current step instead of sitting between two
    steps on the dispatching thread. Other keys pass through untouched.
    """
    from rfdetr_jax.datasets.coco import prefetch

    # JAX's default-device context is thread-local. Capture the caller's
    # selection before prefetch starts its worker, including platform strings.
    default_device = jax.config.jax_default_device

    def put(batch: dict) -> dict:
        """Transfer one batch."""
        with jax.default_device(default_device):
            moved = {
                name: jnp.asarray(batch[name]) if sharding is None else jax.device_put(batch[name], sharding)
                for name in names
            }
        return {**batch, **moved}

    def transfer():
        try:
            for batch in batches:
                yield put(batch)
        finally:
            close = getattr(batches, "close", None)
            if close is not None:
                close()

    return prefetch(transfer(), depth=depth)


def stack_microbatches(batches: list[dict]) -> dict:
    """Stack accumulation inputs, padding differing target buckets without adding real targets."""
    width = max(batch["labels"].shape[1] for batch in batches)
    prepared = []
    for batch in batches:
        padded = {}
        for name, value in batch.items():
            if name in ("labels", "boxes", "target_mask", "masks", "keypoints"):
                padding = [(0, 0)] * value.ndim
                padding[1] = (0, width - value.shape[1])
                value = jnp.pad(value, padding)
            padded[name] = value
        prepared.append(padded)
    return jax.tree.map(lambda *values: jnp.stack(values), *prepared)


@dataclass
class EpochLog:
    """Per-epoch training record."""

    epoch: int
    losses: dict[str, float]
    seconds: float
    steps: int


def train(
    model: LWDETR,
    spec: LWDETRSpec,
    state: TrainState,
    optimizer: optax.GradientTransformation,
    criterion_config: CriterionConfig,
    settings: TrainSettings,
    batches_per_epoch: int,
    batch_iterator,
    on_epoch_end=None,
) -> tuple[TrainState, list[EpochLog]]:
    """Run the training loop.

    Args:
        model: The detector.
        spec: Model spec, used for the aux-loss weight map.
        state: Initial training state.
        optimizer: The optax transformation.
        criterion_config: Loss configuration.
        settings: Loop-level knobs.
        batches_per_epoch: Number of steps per epoch.
        batch_iterator: An iterable yielding one batch dict per step, for the whole run.
        on_epoch_end: Optional callback ``(epoch, state) -> None``.

    Returns:
        A ``(state, logs)`` pair holding the final training state and one :class:`EpochLog` per epoch.
    """
    weights = criterion_config.weight_dict(spec.dec_layers)
    step_fn = make_train_step(model, optimizer, criterion_config, weights, settings)
    key = jax.random.key(settings.seed)

    logs: list[EpochLog] = []
    iterator = iter(batch_iterator)
    for epoch in range(settings.epochs):
        started = time.time()
        # Metrics accumulate on device. Reading one per step would insert a synchronization into
        # every iteration and serialize the loop against the accelerator; one read per epoch keeps
        # dispatch running ahead.
        totals: dict[str, Array] | None = None
        count = 0
        for offset in range(0, batches_per_epoch, settings.grad_accum_steps):
            group_size = min(settings.grad_accum_steps, batches_per_epoch - offset)
            microbatches = [{name: jnp.asarray(batch[name])
                             for name in ("pixel_values", "labels", "boxes", "target_mask", "masks", "keypoints")
                             if name in batch} for batch in (next(iterator) for _ in range(group_size))]
            device_batch = stack_microbatches(microbatches) if settings.grad_accum_steps > 1 else microbatches[0]
            key, dropout_key = jax.random.split(key)
            state, metrics = step_fn(state, device_batch, dropout_key)
            weighted = {name: value * group_size for name, value in metrics.items()}
            totals = weighted if totals is None else {name: totals[name] + value for name, value in weighted.items()}
            count += group_size
        logs.append(
            EpochLog(
                epoch=epoch,
                losses={name: float(value) / max(1, count) for name, value in (totals or {}).items()},
                seconds=time.time() - started,
                steps=(count + settings.grad_accum_steps - 1) // settings.grad_accum_steps,
            )
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch, state)
    return state, logs
