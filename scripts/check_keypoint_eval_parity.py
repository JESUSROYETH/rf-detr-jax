"""Compare this port's OKS keypoint metric against upstream's ``MetricKeypointOKS``.

Runs under the torch environment: it imports both the upstream metric and this port's
framework-agnostic evaluator, and scores the same synthetic ground truth and detections with each.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parity_utils import compare_named  # noqa: E402

from rfdetr_jax.evaluation.keypoint_oks import evaluate_keypoints  # noqa: E402


def build_dataset(num_images: int, num_people: int, seed: int) -> dict:
    """Build a small COCO keypoint ground truth with a 17-point person category."""
    rng = np.random.RandomState(seed)
    images, annotations = [], []
    annotation_id = 1
    for image_id in range(1, num_images + 1):
        images.append({"id": image_id, "file_name": f"{image_id}.jpg", "width": 200, "height": 200})
        for _ in range(num_people):
            x0, y0 = rng.uniform(5, 100, size=2)
            width, height = rng.uniform(40, 90, size=2)
            keypoints = []
            visible = 0
            for _ in range(17):
                visibility = int(rng.choice([0, 1, 2], p=[0.15, 0.25, 0.60]))
                visible += visibility > 0
                keypoints += [
                    float(x0 + rng.uniform(0, width)),
                    float(y0 + rng.uniform(0, height)),
                    visibility,
                ]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [float(x0), float(y0), float(width), float(height)],
                    "area": float(width * height),
                    "iscrowd": 0,
                    "keypoints": keypoints,
                    "num_keypoints": visible,
                }
            )
            annotation_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "person", "keypoints": [f"k{i}" for i in range(17)], "skeleton": []}
        ],
    }


def build_detections(dataset: dict, jitter: float, seed: int) -> list[dict]:
    """Perturb the ground-truth keypoints into predictions of varying quality."""
    rng = np.random.RandomState(seed)
    detections = []
    for annotation in dataset["annotations"]:
        keypoints = np.asarray(annotation["keypoints"], dtype=np.float64).reshape(-1, 3)
        noisy = keypoints.copy()
        noisy[:, :2] += rng.normal(0.0, jitter, size=(keypoints.shape[0], 2))
        noisy[:, 2] = 1.0
        detections.append(
            {
                "image_id": annotation["image_id"],
                "category_id": 1,
                "keypoints": [float(value) for value in noisy.reshape(-1)],
                "score": float(rng.uniform(0.3, 0.99)),
            }
        )
    return detections


def upstream_metrics(dataset: dict, detections: list[dict], sigmas: list[float] | None) -> dict[str, float]:
    """Score the detections with upstream's ``MetricKeypointOKS``."""
    import torch
    from faster_coco_eval import COCO as FasterCOCO
    from rfdetr.evaluation.keypoint_oks import MetricKeypointOKS

    coco = FasterCOCO()
    coco.dataset = json.loads(json.dumps(dataset))
    coco.createIndex()

    metric = MetricKeypointOKS(coco, keypoint_oks_sigmas=sigmas)
    grouped: dict[int, list[dict]] = {}
    for record in detections:
        grouped.setdefault(record["image_id"], []).append(record)
    predictions = {}
    for image_id, records in grouped.items():
        keypoints = torch.tensor(
            [np.asarray(r["keypoints"], dtype=np.float32).reshape(-1, 3).tolist() for r in records]
        )
        predictions[image_id] = {
            "scores": torch.tensor([r["score"] for r in records], dtype=torch.float32),
            "labels": torch.tensor([r["category_id"] for r in records], dtype=torch.int64),
            "boxes": torch.zeros((len(records), 4), dtype=torch.float32),
            "keypoints": keypoints,
        }
    metric.update(predictions)
    return {key: float(value) for key, value in metric.compute().items()}


def main() -> int:
    """Score the same data with both implementations and report the difference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()
    if not np.isfinite(args.tol) or args.tol < 0:
        parser.error("--tol must be finite and nonnegative")

    failures = 0
    for jitter, sigmas in ((0.0, None), (3.0, None), (8.0, None), (3.0, [0.05] * 17)):
        dataset = build_dataset(num_images=6, num_people=2, seed=0)
        detections = build_detections(dataset, jitter, seed=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt.json"
            path.write_text(json.dumps(dataset))
            ours = evaluate_keypoints(path, detections, sigmas)
        theirs = upstream_metrics(dataset, detections, sigmas)

        label = f"jitter={jitter} sigmas={'default' if sigmas is None else 'uniform'}"
        print(f"\n{label}")
        keys = ("map", "map_50", "map_75", "mar")
        # A missing metric on both sides is still a failed comparison.
        if any(key not in ours or key not in theirs for key in keys):
            print("  FAIL expected OKS metric is missing")
            failures += 1
        failures += compare_named(
            {key: ours[key] for key in keys if key in ours},
            {key: theirs[key] for key in keys if key in theirs}, args.tol,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
