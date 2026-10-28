# ------------------------------------------------------------------------
# RF-DETR JAX — modified from RF-DETR, Copyright (c) 2025 Roboflow.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# ------------------------------------------------------------------------
"""Named architecture defaults pinned to upstream 6674d858 (no Torch dependency)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields
from types import SimpleNamespace

from rfdetr_jax.configuration import UnsupportedOptionError, boolean_option
from rfdetr_jax.models.lwdetr import LWDETRSpec

# Literal variant defaults preserve the upstream inheritance result.
VARIANT_DEFAULTS = {'RFDETRBase': {'encoder': 'dinov2_windowed_small',
                'hidden_dim': 256,
                'patch_size': 14,
                'num_windows': 4,
                'dec_layers': 3,
                'sa_nheads': 8,
                'ca_nheads': 16,
                'dec_n_points': 2,
                'num_queries': 300,
                'num_select': 300,
                'projector_scale': ['P4'],
                'out_feature_indexes': [2, 5, 8, 11],
                'pretrain_weights': 'rf-detr-base.pth',
                'resolution': 560,
                'positional_encoding_size': 37},
 'RFDETRLargeDeprecated': {'encoder': 'dinov2_windowed_base',
                           'hidden_dim': 384,
                           'patch_size': 14,
                           'num_windows': 4,
                           'dec_layers': 3,
                           'sa_nheads': 12,
                           'ca_nheads': 24,
                           'dec_n_points': 4,
                           'num_queries': 300,
                           'num_select': 300,
                           'projector_scale': ['P3', 'P5'],
                           'out_feature_indexes': [2, 5, 8, 11],
                           'pretrain_weights': 'rf-detr-large.pth',
                           'resolution': 560,
                           'positional_encoding_size': 37},
 'RFDETRNano': {'encoder': 'dinov2_windowed_small',
                'hidden_dim': 256,
                'patch_size': 16,
                'num_windows': 2,
                'dec_layers': 2,
                'sa_nheads': 8,
                'ca_nheads': 16,
                'dec_n_points': 2,
                'num_queries': 300,
                'num_select': 300,
                'projector_scale': ['P4'],
                'out_feature_indexes': [3, 6, 9, 12],
                'pretrain_weights': 'rf-detr-nano.pth',
                'resolution': 384,
                'positional_encoding_size': 24},
 'RFDETRSmall': {'encoder': 'dinov2_windowed_small',
                 'hidden_dim': 256,
                 'patch_size': 16,
                 'num_windows': 2,
                 'dec_layers': 3,
                 'sa_nheads': 8,
                 'ca_nheads': 16,
                 'dec_n_points': 2,
                 'num_queries': 300,
                 'num_select': 300,
                 'projector_scale': ['P4'],
                 'out_feature_indexes': [3, 6, 9, 12],
                 'pretrain_weights': 'rf-detr-small.pth',
                 'resolution': 512,
                 'positional_encoding_size': 32},
 'RFDETRMedium': {'encoder': 'dinov2_windowed_small',
                  'hidden_dim': 256,
                  'patch_size': 16,
                  'num_windows': 2,
                  'dec_layers': 4,
                  'sa_nheads': 8,
                  'ca_nheads': 16,
                  'dec_n_points': 2,
                  'num_queries': 300,
                  'num_select': 300,
                  'projector_scale': ['P4'],
                  'out_feature_indexes': [3, 6, 9, 12],
                  'pretrain_weights': 'rf-detr-medium.pth',
                  'resolution': 576,
                  'positional_encoding_size': 36},
 'RFDETRLarge': {'encoder': 'dinov2_windowed_small',
                 'hidden_dim': 256,
                 'dec_layers': 4,
                 'sa_nheads': 8,
                 'ca_nheads': 16,
                 'dec_n_points': 2,
                 'num_windows': 2,
                 'patch_size': 16,
                 'projector_scale': ['P4'],
                 'out_feature_indexes': [3, 6, 9, 12],
                 'num_classes': 90,
                 'positional_encoding_size': 44,
                 'pretrain_weights': 'rf-detr-large-2026.pth',
                 'resolution': 704,
                 'num_queries': 300,
                 'num_select': 300},
 'RFDETRSegPreview': {'encoder': 'dinov2_windowed_small',
                      'hidden_dim': 256,
                      'patch_size': 12,
                      'num_windows': 2,
                      'dec_layers': 4,
                      'sa_nheads': 8,
                      'ca_nheads': 16,
                      'dec_n_points': 2,
                      'num_queries': 200,
                      'num_select': 200,
                      'projector_scale': ['P4'],
                      'out_feature_indexes': [3, 6, 9, 12],
                      'pretrain_weights': 'rf-detr-seg-preview.pt',
                      'resolution': 432,
                      'positional_encoding_size': 36,
                      'segmentation_head': True,
                      'num_classes': 90},
 'RFDETRSegNano': {'encoder': 'dinov2_windowed_small',
                   'hidden_dim': 256,
                   'patch_size': 12,
                   'num_windows': 1,
                   'dec_layers': 4,
                   'sa_nheads': 8,
                   'ca_nheads': 16,
                   'dec_n_points': 2,
                   'num_queries': 100,
                   'num_select': 100,
                   'projector_scale': ['P4'],
                   'out_feature_indexes': [3, 6, 9, 12],
                   'pretrain_weights': 'rf-detr-seg-nano.pt',
                   'resolution': 312,
                   'positional_encoding_size': 26,
                   'segmentation_head': True,
                   'num_classes': 90},
 'RFDETRSegSmall': {'encoder': 'dinov2_windowed_small',
                    'hidden_dim': 256,
                    'patch_size': 12,
                    'num_windows': 2,
                    'dec_layers': 4,
                    'sa_nheads': 8,
                    'ca_nheads': 16,
                    'dec_n_points': 2,
                    'num_queries': 100,
                    'num_select': 100,
                    'projector_scale': ['P4'],
                    'out_feature_indexes': [3, 6, 9, 12],
                    'pretrain_weights': 'rf-detr-seg-small.pt',
                    'resolution': 384,
                    'positional_encoding_size': 32,
                    'segmentation_head': True,
                    'num_classes': 90},
 'RFDETRSegMedium': {'encoder': 'dinov2_windowed_small',
                     'hidden_dim': 256,
                     'patch_size': 12,
                     'num_windows': 2,
                     'dec_layers': 5,
                     'sa_nheads': 8,
                     'ca_nheads': 16,
                     'dec_n_points': 2,
                     'num_queries': 200,
                     'num_select': 200,
                     'projector_scale': ['P4'],
                     'out_feature_indexes': [3, 6, 9, 12],
                     'pretrain_weights': 'rf-detr-seg-medium.pt',
                     'resolution': 432,
                     'positional_encoding_size': 36,
                     'segmentation_head': True,
                     'num_classes': 90},
 'RFDETRSegLarge': {'encoder': 'dinov2_windowed_small',
                    'hidden_dim': 256,
                    'patch_size': 12,
                    'num_windows': 2,
                    'dec_layers': 5,
                    'sa_nheads': 8,
                    'ca_nheads': 16,
                    'dec_n_points': 2,
                    'num_queries': 200,
                    'num_select': 200,
                    'projector_scale': ['P4'],
                    'out_feature_indexes': [3, 6, 9, 12],
                    'pretrain_weights': 'rf-detr-seg-large.pt',
                    'resolution': 504,
                    'positional_encoding_size': 42,
                    'segmentation_head': True,
                    'num_classes': 90},
 'RFDETRSegXLarge': {'encoder': 'dinov2_windowed_small',
                     'hidden_dim': 256,
                     'patch_size': 12,
                     'num_windows': 2,
                     'dec_layers': 6,
                     'sa_nheads': 8,
                     'ca_nheads': 16,
                     'dec_n_points': 2,
                     'num_queries': 300,
                     'num_select': 300,
                     'projector_scale': ['P4'],
                     'out_feature_indexes': [3, 6, 9, 12],
                     'pretrain_weights': 'rf-detr-seg-xlarge.pt',
                     'resolution': 624,
                     'positional_encoding_size': 52,
                     'segmentation_head': True,
                     'num_classes': 90},
 'RFDETRSeg2XLarge': {'encoder': 'dinov2_windowed_small',
                      'hidden_dim': 256,
                      'patch_size': 12,
                      'num_windows': 2,
                      'dec_layers': 6,
                      'sa_nheads': 8,
                      'ca_nheads': 16,
                      'dec_n_points': 2,
                      'num_queries': 300,
                      'num_select': 300,
                      'projector_scale': ['P4'],
                      'out_feature_indexes': [3, 6, 9, 12],
                      'pretrain_weights': 'rf-detr-seg-xxlarge.pt',
                      'resolution': 768,
                      'positional_encoding_size': 64,
                      'segmentation_head': True,
                      'num_classes': 90},
 'RFDETRKeypointPreview': {'encoder': 'dinov2_windowed_small',
                           'hidden_dim': 256,
                           'patch_size': 12,
                           'num_windows': 2,
                           'dec_layers': 4,
                           'sa_nheads': 8,
                           'ca_nheads': 16,
                           'dec_n_points': 2,
                           'num_queries': 100,
                           'num_select': 100,
                           'projector_scale': ['P4'],
                           'out_feature_indexes': [3, 6, 9, 12],
                           'pretrain_weights': 'rf-detr-keypoint-preview-xlarge.pth',
                           'resolution': 576,
                           'positional_encoding_size': 48,
                           'use_grouppose_keypoints': True,
                           'dual_projector': True,
                           'dual_projector_kp_only': True,
                           'num_keypoints_per_class': [17],
                           'keypoint_cross_attn': True,
                           'inter_instance_kp_attn': False,
                           'grouppose_keypoint_dim_downscale': 1,
                           'num_classes': 90}}


class PretrainWeightsCompatibilityWarning(UserWarning):
    """Construction options that leave the published pretrained weights unused (as upstream warns)."""


class ModelConfig(SimpleNamespace):
    """Mutable architecture view with upstream's serialization and explicit-field contract."""

    def __setattr__(self, name, value):
        # Upstream's Pydantic config rejects unknown attributes; keep misspelled edits from passing silently.
        if name != "model_fields_set" and vars(self) and name not in vars(self):
            raise ValueError(f"Unknown attribute: {name!r}")
        super().__setattr__(name, value)

    def model_dump(self, **kwargs):
        return {key: deepcopy(value) for key, value in vars(self).items() if key != "model_fields_set"}

    def model_copy(self, *, update=None, deep=False):
        values = self.model_dump()
        values.update(update or {})
        result = ModelConfig(**values)
        result.model_fields_set = self.model_fields_set | set(update or {})
        return result


def variant_config(variant_name: str, **kwargs) -> ModelConfig:
    """Resolve explicit constructor overrides without silently accepting misspelled options."""
    if variant_name not in VARIANT_DEFAULTS:
        raise ValueError(f"unknown RF-DETR variant {variant_name!r}")
    values = asdict(LWDETRSpec())
    values.update(device="auto", amp=True, compile=False, fused_optimizer=True, num_channels=3,
                  backbone_lora=False, freeze_encoder=False, license="Apache-2.0", ia_bce_loss=True,
                  num_decoder_registers=0, postprocess_trace_alpha=0.2, model_name=variant_name)
    defaults = VARIANT_DEFAULTS[variant_name]
    values.update(deepcopy(defaults))
    unknown = kwargs.keys() - values.keys()
    if unknown:
        raise UnsupportedOptionError(f"unsupported model options: {', '.join(sorted(unknown))}")
    boolean_fields = {name for name, value in values.items() if type(value) is bool}
    values.update({name: boolean_option(name, value) if name in boolean_fields else value
                   for name, value in kwargs.items()})
    if values["compile"]:
        raise NotImplementedError("constructor compile=True requests Torch compilation; JAX steps already use jit")
    if values["num_channels"] != 3 or values["backbone_lora"] or values["num_decoder_registers"]:
        raise ValueError("RGB inputs, no LoRA, and zero decoder registers are currently required")
    if ("resolution" in kwargs and "positional_encoding_size" not in kwargs
            and defaults["positional_encoding_size"] == defaults["resolution"] // defaults["patch_size"]):
        values["positional_encoding_size"] = values["resolution"] // values["patch_size"]
    for key in ("projector_scale", "out_feature_indexes", "num_keypoints_per_class"):
        values[key] = list(values[key])
    result = ModelConfig(**values)
    result.model_fields_set = set(kwargs)
    return result


def spec_from_config(config: ModelConfig) -> LWDETRSpec:
    values = {field.name: getattr(config, field.name) for field in fields(LWDETRSpec)}
    for key in ("projector_scale", "out_feature_indexes", "num_keypoints_per_class"):
        values[key] = tuple(values[key])
    return LWDETRSpec(**values)


__all__ = [*VARIANT_DEFAULTS, "RFDETRSeg"]  # noqa: F822


def __getattr__(name):
    """Create public model classes lazily so loading defaults never imports the facade."""
    if name in globals():
        return globals()[name]
    if name not in __all__:
        raise AttributeError(name)
    from rfdetr_jax.detr import RFDETR

    if name == "RFDETRSeg":
        cls = type(name, (RFDETR,), {"_variant_name": "RFDETRSegPreview", "__module__": __name__})
    else:
        parent = __getattr__("RFDETRSeg") if "Seg" in name else RFDETR
        size = VARIANT_DEFAULTS[name]["pretrain_weights"].rsplit(".", 1)[0].replace("rf-detr", "rfdetr")
        if name == "RFDETRSeg2XLarge":
            size = "rfdetr-seg-2xlarge"
        if name == "RFDETRLarge":
            size = "rfdetr-large"
        if name == "RFDETRKeypointPreview":
            size = "rfdetr-keypoint-preview"
        cls = type(name, (parent,), {"_variant_name": name, "size": size, "__module__": __name__})
    globals()[name] = cls
    return cls
