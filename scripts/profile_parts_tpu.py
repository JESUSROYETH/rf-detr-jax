"""Time the RF-DETR train step's components separately to attribute its TPU cost.

Each component is jitted on its own with realistic shapes and timed forward-only and with a
backward pass, so the sum of the parts can be compared against the whole step. Deformable
attention is timed both as the port implements it and in the layouts a TPU prefers, since a gather
along the minormost axis and its transposed scatter are the classic accelerator pathology here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402


def bench(name: str, fn, args, repeats: int = 10, warmup: int = 3) -> dict:
    """Compile, warm up and time ``fn(*args)``, returning a record with the warm median."""
    started = time.perf_counter()
    jax.block_until_ready(fn(*args))
    compile_seconds = time.perf_counter() - started
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeats):
        began = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - began)
    record = {
        "name": name,
        "compile_seconds": compile_seconds,
        "median_ms": statistics.median(samples) * 1e3,
        "min_ms": min(samples) * 1e3,
    }
    print(json.dumps(record), flush=True)
    return record


def grad_of(fn):
    """Return a jitted gradient of the scalar sum of ``fn``'s output w.r.t. its first argument."""
    return jax.jit(jax.grad(lambda *a: jnp.sum(fn(*a)).astype(jnp.float32)))


def main() -> None:
    """Run every component benchmark and write the records as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/parts.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--precision", default="highest", choices=["highest", "float32", "default"])
    parser.add_argument("--compute-dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--sampler", default="gather", choices=["gather", "matmul", "auto"])
    args = parser.parse_args()

    from rfdetr_jax.models.ops import bilinear_grid_sample, ms_deform_attn_core_rows

    jax.config.update("jax_default_matmul_precision", args.precision)
    records = []
    key = jax.random.key(0)
    batch = args.batch_size

    # Decoder cross-attention shapes for RF-DETR Nano in training mode: 16 heads of width 16 over a
    # single 24x24 level, 300 queries in each of 13 groups, 2 sample points per level.
    heads, head_dim, spatial, queries, points = 16, 16, 24, 300 * 13, 2
    length = spatial * spatial
    value = jax.random.normal(key, (batch, heads, head_dim, length), jnp.float32)
    locations = jax.random.uniform(key, (batch, queries, heads, 1, points, 2), jnp.float32)
    weights = jax.random.uniform(key, (batch, queries, heads, points), jnp.float32)

    rows_value = value.transpose(0, 1, 3, 2).reshape(batch * heads, length, head_dim)
    for sampler in ("gather", "matmul"):

        def deform_core(v, loc, w, sampler=sampler):
            """Run the multi-scale deformable attention kernel on one level."""
            return ms_deform_attn_core_rows(v, ((spatial, spatial),), loc, w, sampler)

        records.append(bench(f"deform_attn_{sampler}_forward", jax.jit(deform_core), (rows_value, locations, weights)))
        records.append(bench(f"deform_attn_{sampler}_grad", grad_of(deform_core), (rows_value, locations, weights)))

    flat_value = value.reshape(batch * heads, head_dim, spatial, spatial)
    grid = jax.random.uniform(key, (batch * heads, queries, points, 2), jnp.float32) * 2 - 1
    records.append(bench("grid_sample_forward", jax.jit(bilinear_grid_sample), (flat_value, grid)))
    records.append(bench("grid_sample_grad", grad_of(bilinear_grid_sample), (flat_value, grid)))

    # The same gather in the two candidate layouts, to price the pathology directly.
    source_minor = value.reshape(batch * heads, head_dim, length)
    source_major = jnp.swapaxes(source_minor, 1, 2)
    index = jax.random.randint(key, (batch * heads, queries * points), 0, length, jnp.int32)

    def gather_minor(src, idx):
        """Gather along the minormost axis, as ``bilinear_grid_sample`` does today."""
        broadcast = jnp.broadcast_to(idx[:, None, :], (src.shape[0], src.shape[1], idx.shape[1]))
        return jnp.take_along_axis(src, broadcast, axis=2)

    def gather_major(src, idx):
        """Gather whole rows along the major axis, the layout a TPU prefers."""
        return jnp.take_along_axis(src, idx[:, :, None], axis=1)

    records.append(bench("gather_minor_axis_forward", jax.jit(gather_minor), (source_minor, index)))
    records.append(bench("gather_major_axis_forward", jax.jit(gather_major), (source_major, index)))
    records.append(bench("gather_minor_axis_grad", grad_of(gather_minor), (source_minor, index)))
    records.append(bench("gather_major_axis_grad", grad_of(gather_major), (source_major, index)))

    # Backbone and full model, forward and with a backward pass.
    from shared_run_config import RunConfig
    from train_jax import spec_for

    from rfdetr_jax.models.lwdetr import LWDETR
    from rfdetr_jax.training.trainer import init_params

    run = RunConfig(resolution=args.resolution, batch_size=batch, gradient_checkpointing=False)
    spec = dataclasses.replace(
        spec_for(run), deform_sampler=args.sampler, compute_dtype=args.compute_dtype
    )
    model = LWDETR(spec=spec)
    with jax.default_device(jax.devices("cpu")[0]):
        params = init_params(model, spec, jax.random.key(0))
    params = jax.device_put(jax.tree.map(np.asarray, params))
    pixels = jax.random.normal(key, (batch, 3, args.resolution, args.resolution), jnp.float32)

    def only_backbone(module, x, train):
        """Module method selecting just the backbone branch of the detector."""
        return module.backbone_0(x, train=train)[0][0]

    def backbone_forward(p, x):
        """Run only the DINOv2 backbone and projector."""
        return model.apply({"params": p}, x, True, method=only_backbone)

    records.append(bench("backbone_forward", jax.jit(backbone_forward), (params, pixels)))
    records.append(bench("backbone_grad", grad_of(backbone_forward), (params, pixels)))

    def model_forward(p, x):
        """Run the whole detector in training mode, returning the main logits."""
        return model.apply({"params": p}, x, None, train=True)["pred_logits"]

    records.append(bench("model_forward_train", jax.jit(model_forward), (params, pixels)))
    records.append(bench("model_grad_train", grad_of(model_forward), (params, pixels)))

    # What is left inside the transformer once deformable attention stops dominating.
    from rfdetr_jax.models.criterion import CriterionConfig, Targets, compute_losses
    from rfdetr_jax.models.layers import MLP, Linear, MultiheadAttention

    hidden, groups, per_group, memory_len = 256, 13, 300, spatial * spatial
    attention = MultiheadAttention(hidden, 8)
    grouped = jax.random.normal(key, (groups * batch, per_group, hidden), jnp.float32)
    attention_params = attention.init(key, grouped, grouped, grouped)

    def grouped_self_attention(p, x):
        """One decoder layer's group-DETR self-attention."""
        return attention.apply(p, x, x, x)

    attention_args = (attention_params, grouped)
    records.append(bench("decoder_self_attention_forward", jax.jit(grouped_self_attention), attention_args))
    records.append(bench("decoder_self_attention_grad", grad_of(grouped_self_attention), attention_args))

    ffn = MLP(hidden, 2048, hidden, 2)
    queries_all = jax.random.normal(key, (batch, groups * per_group, hidden), jnp.float32)
    ffn_params = ffn.init(key, queries_all)

    def feed_forward(p, x):
        """One decoder layer's feed-forward block."""
        return ffn.apply(p, x)

    records.append(bench("decoder_ffn_forward", jax.jit(feed_forward), (ffn_params, queries_all)))
    records.append(bench("decoder_ffn_grad", grad_of(feed_forward), (ffn_params, queries_all)))

    scores = jax.random.normal(key, (groups, batch, memory_len), jnp.float32)
    records.append(bench("two_stage_top_k", jax.jit(lambda s: jax.lax.top_k(s, per_group)[1]), (scores,)))

    # Selecting the top-k rows is another indexed read, this time of 256-wide rows. Price it against
    # the same selection written as a one-hot contraction.
    projected = jax.random.normal(key, (groups, batch, memory_len, hidden), jnp.float32)
    indices = jax.lax.top_k(scores, per_group)[1]

    def select_gather(p, idx):
        """Gather the selected encoder rows."""
        return jnp.take_along_axis(p, idx[..., None], axis=2)

    def select_onehot(p, idx):
        """Select the same rows by contracting a one-hot matrix against the encoder memory."""
        onehot = (idx[..., None] == jnp.arange(memory_len)[None, None, None, :]).astype(p.dtype)
        return jnp.einsum("gbqk,gbkc->gbqc", onehot, p)

    records.append(bench("two_stage_select_gather", jax.jit(select_gather), (projected, indices)))
    records.append(bench("two_stage_select_onehot", jax.jit(select_onehot), (projected, indices)))
    records.append(bench("two_stage_select_gather_grad", grad_of(select_gather), (projected, indices)))
    records.append(bench("two_stage_select_onehot_grad", grad_of(select_onehot), (projected, indices)))

    head = Linear(hidden, hidden)
    memory = jax.random.normal(key, (batch, memory_len, hidden), jnp.float32)
    head_params = head.init(key, memory)
    stacked_weight = jnp.stack([head_params["params"]["weight"]] * groups)

    def grouped_projection(w, x):
        """The fused per-group encoder projection."""
        return jnp.einsum("...i,goi->g...o", x, w)

    projection_args = (stacked_weight, memory)
    records.append(bench("two_stage_projection_forward", jax.jit(grouped_projection), projection_args))
    records.append(bench("two_stage_projection_grad", grad_of(grouped_projection), projection_args))

    classes = 7
    outputs = {
        "pred_logits": jax.random.normal(key, (batch, groups * per_group, classes), jnp.float32),
        "pred_boxes": jax.random.uniform(key, (batch, groups * per_group, 4), jnp.float32) * 0.5 + 0.25,
    }
    outputs["aux_outputs"] = [dict(outputs)]
    outputs["enc_outputs"] = dict(pred_logits=outputs["pred_logits"], pred_boxes=outputs["pred_boxes"])
    # Targets must differ from one another: identical ones put the auction in its worst case, where
    # every column competes for the same rows, and that is not what a batch looks like.
    target_mask = jnp.zeros((batch, 20), bool).at[:, :4].set(True)
    targets = Targets(
        labels=jax.random.randint(key, (batch, 20), 0, classes, jnp.int32),
        boxes=jnp.concatenate(
            [
                jax.random.uniform(key, (batch, 20, 2), jnp.float32) * 0.5 + 0.25,
                jax.random.uniform(key, (batch, 20, 2), jnp.float32) * 0.25 + 0.05,
            ],
            axis=-1,
        ),
        mask=target_mask,
    )
    for matcher in ("hungarian", "auction"):
        config = CriterionConfig(num_classes=classes, group_detr=groups, matcher=matcher)

        def criterion(o, config=config):
            """Every configured loss for one set of outputs."""
            return sum(compute_losses(o, targets, config, train=True).values())

        records.append(bench(f"criterion_{matcher}_forward", jax.jit(criterion), (outputs,)))
        records.append(bench(f"criterion_{matcher}_grad", grad_of(criterion), (outputs,)))

    # The optimizer touches every parameter several times per step; on a tree of this many small
    # leaves that is hundreds of kernels over a few hundred megabytes.
    from rfdetr_jax.training.optimizer import OptimizerSpec, build_optimizer, ema_update

    optimizer = build_optimizer(params, OptimizerSpec(), spec.out_feature_indexes)
    opt_state = optimizer.init(params)
    grads = jax.tree.map(lambda leaf: jnp.full_like(leaf, 1e-3), params)
    ema = jax.device_put(jax.tree.map(np.asarray, params))

    def optimizer_step(grads, opt_state, params, ema):
        """One AdamW update, its parameter application and the EMA update."""
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, ema_update(ema, params, jnp.float32(0.99))

    records.append(
        bench("optimizer_update", jax.jit(optimizer_step), (grads, opt_state, params, ema))
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "device": jax.devices()[0].device_kind,
                "precision": args.precision,
                "compute_dtype": args.compute_dtype,
                "sampler": args.sampler,
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
