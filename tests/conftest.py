"""Dataset fixtures shared by source and installed-wheel tests."""
import json

import pytest
from PIL import Image


def _write_dataset(root, task):
    category = {"id": 5, "name": "custom"}
    if task == "keypoints":
        category.update(keypoints=["left", "right"], skeleton=[[1, 2]])
    for split in ("train", "valid", "test"):
        folder = root / split
        folder.mkdir(parents=True)
        Image.new("RGB", (32, 32), (60, 120, 180)).save(folder / "image.png")
        annotation = {"id": 1, "image_id": 1, "category_id": 5, "bbox": [8, 8, 16, 16],
                      "area": 256, "iscrowd": 0}
        if task == "segmentation":
            annotation["segmentation"] = [[8, 8, 24, 8, 24, 24, 8, 24]]
        if task == "keypoints":
            annotation.update(keypoints=[12, 12, 2, 20, 20, 2], num_keypoints=2)
        (folder / "_annotations.coco.json").write_text(json.dumps({
            "images": [{"id": 1, "file_name": "image.png", "width": 32, "height": 32}],
            "annotations": [annotation], "categories": [category],
        }))



@pytest.fixture
def write_dataset():
    return _write_dataset

