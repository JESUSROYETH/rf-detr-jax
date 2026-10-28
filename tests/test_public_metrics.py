"""Public metrics must retain the meaning and names of the upstream callback."""
import pytest

from rfdetr_jax.evaluation.public_metrics import format_public_metrics


def test_detection_headline_names():
    metrics = dict(mAP=.1, mAP_50=.2, mAP_75=.3, mAR_max=.4, mAP_small=.9)
    assert format_public_metrics(metrics, "test", "detection") == {
        "test/mAP_50_95": .1, "test/mAP_50": .2, "test/mAP_75": .3, "test/mAR": .4}


@pytest.mark.parametrize("task", ["segmentation", "keypoints"])
def test_task_ap_never_overwrites_box_headline(task):
    metrics = dict(mAP=.9, mAP_50=.8, mAP_75=.7, mAR_max=.6,
                   bbox_mAP=.1, bbox_mAP_50=.2, bbox_mAP_75=.3, bbox_mAR_max=.4)
    actual = format_public_metrics(metrics, "val", task)
    assert actual["val/mAP_50_95"] == .1
    assert actual["val/mAP_50"] == .2
    assert actual["val/mAP_75"] == .3
    assert actual["val/mAR"] == .4
    task_name = "segm_mAP_50_95" if task == "segmentation" else "keypoint_map_50_95"
    assert actual[f"val/{task_name}"] == .9
    if task == "keypoints":
        assert actual["val/keypoint_map_75"] == .7
        assert actual["val/keypoint_mAR"] == .6
    assert "val/F1" not in actual


def test_missing_box_companion_fails_instead_of_claiming_task_ap_is_box_ap():
    with pytest.raises(KeyError, match="bbox_mAP"):
        format_public_metrics(dict(mAP=.9, mAP_50=.8, mAP_75=.7, mAR_max=.6), "val", "segmentation")


def test_unavailable_keypoints_not_logged_and_separate_f1_preserved():
    metrics = dict(mAP=-1, mAP_50=-1, mAP_75=-1, mAR_max=-1,
                   bbox_mAP=.1, bbox_mAP_50=.2, bbox_mAP_75=.3, bbox_mAR_max=.4,
                   F1=.8, precision=.9, recall=.7)
    actual = format_public_metrics(metrics, "val", "keypoints")
    assert not any("keypoint" in key for key in actual)
    assert actual["val/F1"] == .8 and actual["val/precision"] == .9 and actual["val/recall"] == .7


@pytest.mark.parametrize("split,task", [("validation", "detection"), ("val", "pose")])
def test_unknown_namespace_fails(split, task):
    with pytest.raises(ValueError):
        format_public_metrics({}, split, task)
