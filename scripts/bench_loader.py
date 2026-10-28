"""Measure the host data pipeline's throughput against the thread count it is given.

The training step only goes as fast as the batches reaching it, so once the accelerator step drops
below the time it takes to decode and augment a batch, this is the number that matters.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfdetr_jax.datasets.coco import CocoDetection, iterate_batches  # noqa: E402


def main() -> None:
    """Time one batch at each requested worker count and check the stream is unchanged."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/synthetic/train")
    parser.add_argument("--annotations", default="data/synthetic/train/_annotations.coco.json")
    parser.add_argument("--out", default="artifacts/loader.json")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--batch-sizes", default="2,8,32")
    parser.add_argument("--workers", default="0,4,8,16,32")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    dataset = CocoDetection(args.images, args.annotations)
    reference = next(iterate_batches(dataset, args.resolution, 4, 20, train=True, seed=11))
    checked = next(iterate_batches(dataset, args.resolution, 4, 20, train=True, seed=11, workers=8))
    identical = all(np.array_equal(reference[name], checked[name]) for name in reference)

    records = []
    for batch_size in (int(v) for v in args.batch_sizes.split(",")):
        for workers in (int(v) for v in args.workers.split(",")):
            stream = iterate_batches(
                dataset, args.resolution, batch_size, 20, train=True, seed=0, epochs=50, workers=workers
            )
            for _ in range(3):
                next(stream)
            samples = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                next(stream)
                samples.append(time.perf_counter() - started)
            median = statistics.median(samples)
            record = {
                "batch_size": batch_size,
                "workers": workers,
                "batch_ms": median * 1e3,
                "images_per_second": batch_size / median,
            }
            records.append(record)
            print(json.dumps(record), flush=True)

    payload = {"images": len(dataset), "stream_identical_with_workers": identical, "records": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps({"stream_identical_with_workers": identical}))


if __name__ == "__main__":
    main()
