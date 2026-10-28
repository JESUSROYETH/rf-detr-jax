"""JAX/Flax RF-DETR with upstream-style public model classes."""

__version__ = "0.1.0"

_VARIANTS = (
    "RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge", "RFDETRBase",
    "RFDETRLargeDeprecated", "RFDETRSeg", "RFDETRSegPreview", "RFDETRSegNano",
    "RFDETRSegSmall", "RFDETRSegMedium", "RFDETRSegLarge", "RFDETRSegXLarge",
    "RFDETRSeg2XLarge", "RFDETRKeypointPreview",
)
__all__ = ["RFDETR", "from_checkpoint", "__version__", *_VARIANTS]


def from_checkpoint(path, **kwargs):
    """Load a native or upstream model checkpoint through the public interface."""
    from rfdetr_jax.detr import RFDETR

    return RFDETR.from_checkpoint(path, **kwargs)


def __getattr__(name):
    """Resolve model classes lazily; querying a version must not initialize JAX."""
    if name == "RFDETR":
        from rfdetr_jax.detr import RFDETR

        return RFDETR
    if name in _VARIANTS:
        from rfdetr_jax import variants

        return getattr(variants, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
