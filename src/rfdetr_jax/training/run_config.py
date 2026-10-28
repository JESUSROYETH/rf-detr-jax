"""Hyperparameters shared by the PyTorch and JAX training runs, so the two are comparable."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    """One training configuration, applied identically to both backends."""

    dataset_dir: str = "data/synthetic"
    train_split: str = "train"
    val_split: str = "val"
    resolution: int = 384
    num_classes: int = 6
    batch_size: int = 2
    max_targets: int = 20
    epochs: int = 5
    seed: int = 42
    data_seed: int = 7
    grad_accum_steps: int = 1
    multi_scale: bool = False
    expanded_scales: bool = True
    scale_jitter: bool = True
    freeze_encoder: bool = False
    fused_optimizer: bool = True
    requested_amp_dtype: str | None = None
    effective_amp_dtype: str | None = None
    resolved_compute_dtype: str = "float32"
    loss_scaling: bool = False
    keypoint_box_normalized_accumulation: bool = False
    automatic_accumulation_normalization: bool = False

    task: str = "detection"
    num_keypoints_per_class: tuple[int, ...] = ()
    keypoint_flip_pairs: tuple[int, ...] = ()
    keypoint_oks_sigmas: tuple[float, ...] = ()
    mask_downsample_ratio: int = 4
    postprocess_trace_alpha: float = 0.2

    # Model (RF-DETR Nano geometry).
    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 16
    num_windows: int = 2
    dec_layers: int = 2
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    group_detr: int = 13
    projector_scale: tuple[str, ...] = ("P4",)
    out_feature_indexes: tuple[int, ...] = (3, 6, 9, 12)
    gradient_checkpointing: bool = True

    # Optimization.
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    weight_decay: float = 1e-4
    lr_vit_layer_decay: float = 0.8
    lr_component_decay: float = 0.7
    clip_max_norm: float = 0.1
    lr_scheduler: str = "step"
    lr_drop: int = 100
    warmup_epochs: float = 0.0
    lr_min_factor: float = 0.0

    # EMA.
    use_ema: bool = True
    ema_decay: float = 0.993
    ema_tau: int = 100
    ema_first_update_copy: bool = False

    @property
    def positional_encoding_size(self) -> int:
        """Positional-encoding grid size implied by the resolution and patch size."""
        return self.resolution // self.patch_size

    def train_paths(self) -> tuple[str, str]:
        """Return the ``(image_dir, annotation_file)`` pair for the training split."""
        return (
            f"{self.dataset_dir}/{self.train_split}",
            f"{self.dataset_dir}/{self.train_split}/_annotations.coco.json",
        )

    def val_paths(self) -> tuple[str, str]:
        """Return the ``(image_dir, annotation_file)`` pair for the validation split."""
        return (
            f"{self.dataset_dir}/{self.val_split}",
            f"{self.dataset_dir}/{self.val_split}/_annotations.coco.json",
        )


DEFAULT_RUN = RunConfig()
