"""Opt-in installed-package acceptance: official weights, URL input and Supervision.

This command may download pretrained weights and the reference image. Run one
task per process with bounded CPU threads. Switch --backend to execute the same
consumer code against upstream; no implementation fallbacks or mocks are used.
"""
from __future__ import annotations

import argparse
import importlib
import json

import numpy as np
import supervision as sv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("rfdetr", "rfdetr_jax"), default="rfdetr_jax")
    parser.add_argument("--task", choices=("detection", "segmentation", "keypoints"), required=True)
    parser.add_argument("--image", default="https://media.roboflow.com/dog.jpg")
    args = parser.parse_args()
    api = importlib.import_module(args.backend)
    class_names = importlib.import_module(args.backend + ".assets.coco_classes").COCO_CLASSES
    variants = dict(detection="RFDETRNano", segmentation="RFDETRSegNano", keypoints="RFDETRKeypointPreview")
    model = getattr(api, variants[args.task])()
    result = model.predict(args.image, threshold=.5)
    if args.task == "keypoints":
        assert isinstance(result, sv.KeyPoints)
        # KeyPoints has per-object data; collection-level metadata is Detections-only.
        source = result.data["source_image"][0] if len(result) else None
        annotated = sv.VertexAnnotator().annotate(source, result) if source is not None else None
        assert np.isfinite(result.detection_confidence).all()
        assert np.isfinite(result.keypoint_confidence).all()
    else:
        assert isinstance(result, sv.Detections)
        source = result.metadata["source_image"]
        labels = [class_names[class_id] for class_id in result.class_id]
        annotator = sv.MaskAnnotator() if args.task == "segmentation" else sv.BoxAnnotator()
        annotated = annotator.annotate(source, result)
        annotated = sv.LabelAnnotator().annotate(annotated, result, labels)
        assert np.isfinite(result.confidence).all()
    assert np.asarray(annotated).shape == np.asarray(source).shape
    print(json.dumps(dict(backend=args.backend, task=args.task, model=type(model).__name__,
                          result=type(result).__name__, count=len(result),
                          source_shape=list(np.asarray(source).shape) if source is not None else None)))


if __name__ == "__main__":
    main()
