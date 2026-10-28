"""Atomic, non-pickle training checkpoints including optimizer, EMA and RNG state."""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from rfdetr_jax.training.trainer import TrainState


def _state_payload(state: TrainState, *, version=2) -> dict:
    payload = {"params": state.params, "opt_state": state.opt_state,
               "ema_params": state.ema_params, "step": state.step}
    if version >= 2:
        payload["loss_scale"] = state.loss_scale
    return payload


def save_training_checkpoint(path, state: TrainState, key, metadata: dict) -> None:
    """Atomically save a complete optimizer boundary; no Python object loading is used."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    leaves, structure = jax.tree.flatten(_state_payload(state))
    payload = serialization.to_bytes(jax.device_get(leaves))
    metadata = {**metadata, "checkpoint_version": 2, "state_structure": str(structure)}
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, state=np.frombuffer(payload, dtype=np.uint8),
                     rng=np.asarray(jax.random.key_data(key)), metadata=json.dumps(metadata))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _checkpoint_source(path):
    """Unwrap native state embedded in a safely loaded, self-contained PyTorch checkpoint."""
    if Path(path).suffix.lower() not in (".pth", ".pt", ".ckpt"):
        return path
    from rfdetr_jax.checkpoints import safe_torch_load
    checkpoint = safe_torch_load(path)
    value = checkpoint.get("rfdetr_jax_training_state")
    if value is None:
        raise ValueError("checkpoint has no full JAX training state; use it as initial weights instead")
    if not hasattr(value, "detach"):
        raise ValueError("embedded training state must be a uint8 tensor")
    array = value.detach().cpu().numpy()
    if array.dtype != np.uint8 or array.ndim != 1:
        raise ValueError("embedded training state must be a one-dimensional uint8 tensor")
    return io.BytesIO(array.tobytes())


def checkpoint_metadata(path) -> dict:
    """Read the schedule/model/data contract without allocating model state."""
    with np.load(_checkpoint_source(path), allow_pickle=False) as archive:
        if set(archive.files) != {"state", "rng", "metadata"}:
            raise ValueError("resume requires a full training checkpoint, not inference parameters")
        metadata = json.loads(str(archive["metadata"]))
    if metadata.get("checkpoint_version") not in (1, 2):
        raise ValueError("unsupported training checkpoint version")
    return metadata


def load_training_checkpoint(path, template: TrainState):
    """Restore parameters, optimizer counters/moments, EMA, step and dropout RNG."""
    metadata = checkpoint_metadata(path)
    version = metadata["checkpoint_version"]
    if version == 1 and template.loss_scale is not None:
        raise ValueError("version-1 checkpoints contain no FP16 loss scaler; resume with the original precision")
    template_leaves, structure = jax.tree.flatten(_state_payload(template, version=version))
    if metadata["state_structure"] != str(structure):
        raise ValueError("checkpoint optimizer/EMA structure differs from the configured training state")
    with np.load(_checkpoint_source(path), allow_pickle=False) as archive:
        leaves = serialization.from_bytes(template_leaves, archive["state"].tobytes())
        key = jax.random.wrap_key_data(np.asarray(archive["rng"], dtype=np.uint32))
    if any(np.shape(value) != np.shape(expected) for value, expected in zip(leaves, template_leaves, strict=True)):
        raise ValueError("checkpoint state shapes differ from the configured model/optimizer")
    if any(np.asarray(value).dtype != np.asarray(expected).dtype
           for value, expected in zip(leaves, template_leaves, strict=True)):
        raise ValueError("checkpoint state dtypes differ from the configured model/optimizer")
    restored = jax.tree.unflatten(structure, leaves)
    restored = jax.tree.map(jax.device_put, restored)
    return TrainState(**restored), key, metadata
