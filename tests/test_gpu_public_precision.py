"""CUDA public workflows, including precision policy and standalone resume."""
import json
import shutil

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not any(device.platform == "gpu" and "NVIDIA" in device.device_kind for device in jax.devices()),
    reason="requires NVIDIA CUDA; CPU checks cannot qualify public mixed-precision training",
)


@pytest.mark.parametrize("task", ["detection", "segmentation", "keypoints"])
@pytest.mark.parametrize("precision", ["auto", "fp16"])
def test_cuda_public_train_standalone_resume_evaluate_predict(tmp_path, task, precision, write_dataset):
    import supervision as sv

    from rfdetr_jax import RFDETR, RFDETRKeypointPreview, RFDETRNano, RFDETRSegNano
    from rfdetr_jax.training.checkpoint import checkpoint_metadata
    from rfdetr_jax.training.precision import _cuda_capability

    dataset = tmp_path / "data"
    write_dataset(dataset, task)
    # Several real steps let the default FP16 scaler recover from initial
    # overflow. Every record has supervision, rather than padding a single item.
    annotation_path = dataset / "train" / "_annotations.coco.json"
    annotations = json.loads(annotation_path.read_text())
    native_bf16 = _cuda_capability(jax.devices("gpu")[0]) >= (8, 0)
    expected_dtype = "float16" if precision == "fp16" or not native_bf16 else "bfloat16"
    count = 16 if expected_dtype == "float16" else 2
    annotations["images"] = [dict(annotations["images"][0], id=i) for i in range(1, count + 1)]
    annotations["annotations"] = [dict(annotations["annotations"][0], id=i, image_id=i)
                                  for i in range(1, count + 1)]
    annotation_path.write_text(json.dumps(annotations))
    cls = {"detection": RFDETRNano, "segmentation": RFDETRSegNano,
           "keypoints": RFDETRKeypointPreview}[task]
    options = dict(pretrain_weights=None, device="cuda", resolution=32, patch_size=16,
                   positional_encoding_size=2, num_windows=1, dec_layers=1, group_detr=1,
                   num_queries=2, num_select=2, hidden_dim=32, sa_nheads=2, ca_nheads=2,
                   dim_feedforward=32, out_feature_indexes=[1], num_classes=1,
                   gradient_checkpointing=False)
    if task == "keypoints":
        options["num_keypoints_per_class"] = [2]
    model = cls(**options)

    class Interrupted(Exception):
        pass

    def stop_after_first_epoch(record):
        assert record["epoch"] == 0
        raise Interrupted

    model.callbacks["on_fit_epoch_end"].append(stop_after_first_epoch)
    partial = tmp_path / "partial"
    with pytest.raises(Interrupted):
        model.train(dataset_dir=dataset, output_dir=partial, epochs=2, batch_size=1,
                    grad_accum_steps=2, multi_scale=False, num_workers=0, tensorboard=False,
                    no_precompile=True, no_compile_cache=True, amp_dtype=precision, seed=0,
                    keypoint_oks_sigmas=[.1, .1] if task == "keypoints" else None)
    standalone = tmp_path / "standalone.pth"
    shutil.copyfile(partial / "checkpoint.pth", standalone)
    shutil.rmtree(partial)
    saved = checkpoint_metadata(standalone)
    assert saved["model_metadata"]["training_precision"]["requested"] == precision
    assert saved["runtime_config"]["pixel_dtype"] == "float32"
    assert saved["model_metadata"]["training_precision"]["compute_dtype"] == expected_dtype
    resumed = RFDETR.from_checkpoint(standalone, device="cuda")
    resumed.train(resume=standalone, output_dir=tmp_path / "resumed")
    state = resumed._training_result.state
    assert int(state.step) == count  # Two epochs, accumulation of two.
    if expected_dtype == "float16":
        assert state.loss_scale is not None
        assert int(state.loss_scale.skipped_steps) < int(state.step)
    else:
        assert state.loss_scale is None
    for value in jax.tree.leaves((state.params, state.opt_state, state.ema_params)):
        if np.issubdtype(np.asarray(value).dtype, np.floating):
            assert np.asarray(value).dtype == np.float32
    output = tmp_path / "trained.npz"
    resumed.save(output)
    restored = RFDETR.from_checkpoint(output, device="cuda")
    assert restored._backend.model.spec.compute_dtype == "float32"
    assert restored._backend.model.spec.head_dtype == "float32"
    result = restored.predict(dataset / "test" / "image.png", threshold=0)
    assert isinstance(result, sv.KeyPoints if task == "keypoints" else sv.Detections)
    metrics = restored.evaluate(dataset_dir=dataset, batch_size=1, num_workers=0,
                                tensorboard=False,
                                keypoint_oks_sigmas=[.1, .1] if task == "keypoints" else None)
    assert {"test/F1", "test/mAP_50_95"}.issubset(metrics)
    assert all(np.isfinite(value) for value in metrics.values())
