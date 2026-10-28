"""Verify release artifact contents without installing or executing them."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

from check_repository import is_private_path


def main() -> None:
    """Require legal notices, runtime configuration, CLI and test fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    wheels = list(args.directory.glob("*.whl"))
    sdists = list(args.directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        parser.error("expected exactly one wheel and one source distribution")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        if any(is_private_path(Path(name)) for name in names):
            raise SystemExit("wheel contains local-only content")
        for suffix in ("/LICENSE", "/NOTICE", "rfdetr_jax/cli.py", "rfdetr_jax/training/run.py",
                       "rfdetr_jax/inference.py", "rfdetr_jax/configuration.py", "rfdetr_jax/evaluation/tasks.py",
                       "rfdetr_jax/detr.py", "rfdetr_jax/prediction.py", "rfdetr_jax/variants.py",
                       "rfdetr_jax/checkpoints.py", "rfdetr_jax/api_training.py", "rfdetr_jax/export.py",
                       "rfdetr_jax/training/checkpoint.py", "rfdetr_jax/datasets/yolo.py",
                       "rfdetr_jax/assets/coco_classes.py", "rfdetr_jax/evaluation/f1_metrics.py",
                       "rfdetr_jax/evaluation/public_evaluation.py"):
            if not any(name.endswith(suffix) for name in names):
                raise SystemExit(f"wheel is missing {suffix}")
        configs = [name for name in names if "/dinov2_configs/" in name and name.endswith(".json")]
        if len(configs) != 6:
            raise SystemExit("wheel must contain all six DINOv2 configurations")
        if any(name.startswith(("tests/", "scripts/", "docs/", "upstream/")) for name in names):
            raise SystemExit("wheel includes development-only content")
    with tarfile.open(sdists[0]) as archive:
        names = ["/".join(Path(name).parts[1:]) for name in archive.getnames()]
        for required in (
            "LICENSE", "NOTICE", "Makefile", "scripts/check_repository.py",
            "scripts/constraints-min.txt", "scripts/run_reference_tests.py",
            "tests/data/interpolate_reference.npz", "tests/data/nano_state_dict_shapes.json",
            "tests/data/task_postprocess_reference.npz", "tests/data/task_postprocess_reference.json",
        ):
            if required not in names:
                raise SystemExit(f"source distribution is missing {required}")
        if any(is_private_path(Path(name)) for name in names):
            raise SystemExit("source distribution contains local-only content")
    print("Wheel and source distribution contents passed.")


if __name__ == "__main__":
    main()
