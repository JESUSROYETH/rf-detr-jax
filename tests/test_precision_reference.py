"""Published-stack precision evidence. CPU by default; explicit GPU runs use
RFDETR_PRECISION_REFERENCE_DEVICE=cuda and require both Torch and JAX CUDA.
"""
import importlib.metadata
import os

import pytest


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("overflow_index", [0, 1])
def test_lightning_overflow_advances_schedule_and_upstream_ema_but_not_adam(manual, overflow_index):
    torch = pytest.importorskip("torch")
    pl = pytest.importorskip("pytorch_lightning")
    pytest.importorskip("rfdetr")
    assert importlib.metadata.version("rfdetr") == "1.11.2"
    from pytorch_lightning.plugins.precision import MixedPrecision
    from rfdetr.training.callbacks.ema import RFDETREMACallback
    from torch.utils.data import DataLoader

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
            self.automatic_optimization = not manual

        def training_step(self, batch, batch_idx):
            loss = self.weight * (float("inf") if batch_idx == overflow_index else 1.)
            if self.automatic_optimization:
                return loss
            optimizer = self.optimizers()
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            self.lr_schedulers().step()
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=.1, weight_decay=0)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .5 ** step)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    ema = RFDETREMACallback(decay=.5, tau=0)
    observed = []

    class Observe(pl.Callback):
        def on_train_batch_end(self, trainer, model, *args):
            optimizer = trainer.optimizers[0]
            observed.append(dict(step=trainer.global_step, weight=model.weight.item(),
                                 adam_step=float(optimizer.state[model.weight].get("step", 0)),
                                 scheduler=trainer.lr_scheduler_configs[0].scheduler.last_epoch,
                                 ema_updates=int(ema._average_model.n_averaged),
                                 ema_weight=ema._average_model.module.weight.item(),
                                 scale=trainer.precision_plugin.scaler.get_scale()))

    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    trainer = pl.Trainer(accelerator="cpu", devices=1,
                         plugins=[MixedPrecision("16-mixed", "cpu", scaler=scaler)],
                         max_epochs=1, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False, callbacks=[ema, Observe()])
    trainer.fit(Model(), DataLoader([0, 0, 0], batch_size=1))
    assert [row["step"] for row in observed] == [1, 2, 3]
    assert [row["scheduler"] for row in observed] == [1, 2, 3]
    assert [row["ema_updates"] for row in observed] == [1, 2, 3]
    assert [row["adam_step"] for row in observed] == ([1, 1, 2] if overflow_index == 1 else [0, 1, 2])
    assert [row["scale"] for row in observed] == ([8, 4, 4] if overflow_index == 1 else [4, 4, 4])
    weights = [.9, .9, .875] if overflow_index == 1 else [1., .95, .925]
    ema_weights = [.9, .9, .8875] if overflow_index == 1 else [1., .975, .95]
    assert [row["weight"] for row in observed] == pytest.approx(weights)
    assert [row["ema_weight"] for row in observed] == pytest.approx(ema_weights)


@pytest.fixture(scope="module", params=["detection", "segmentation", "keypoints"])
def mixed_model(request):
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp
    import numpy as np

    pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    assert importlib.metadata.version("rfdetr") == "1.11.2"
    from rfdetr_jax.export import build_export_model
    from rfdetr_jax.models.lwdetr import LWDETR, LWDETRSpec

    task = request.param
    spec = LWDETRSpec(resolution=32, positional_encoding_size=2, patch_size=16, num_windows=1,
                      dec_layers=1, num_queries=2, num_select=2, group_detr=1, num_classes=1,
                      hidden_dim=32, sa_nheads=2, ca_nheads=2, dim_feedforward=32, out_feature_indexes=(1, 2),
                      segmentation_head=task == "segmentation", use_grouppose_keypoints=task == "keypoints",
                      num_keypoints_per_class=(2,) if task == "keypoints" else ())
    image = np.random.default_rng(12).normal(size=(1, 3, 32, 32)).astype("float32")
    with jax.default_device(jax.devices("cpu")[0]):
        params = LWDETR(spec).init(jax.random.key(4), jnp.array(image), None, train=True)["params"]
        generator = np.random.default_rng(42)
        # Nonzero box/keypoint outputs and residual scales prevent trivial zero-head agreement.
        params = jax.tree.map(lambda value: jnp.asarray(generator.normal(0, .03, value.shape), value.dtype)
                             if not np.asarray(value).any() else value, params)
    facade = SimpleNamespace(spec=spec, params=params, class_names=["object"],
                             model_config=SimpleNamespace(postprocess_trace_alpha=.2), size="tiny")
    reference = build_export_model(facade).model.model
    device = os.environ.get("RFDETR_PRECISION_REFERENCE_DEVICE", "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("RFDETR_PRECISION_REFERENCE_DEVICE must be cpu or cuda")
    target = jax.devices("gpu" if device == "cuda" else "cpu")[0]
    params = jax.tree.map(lambda value: jax.device_put(value, target), params)
    reference.to(device)
    import torch
    previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield spec, params, image, reference
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32
    jax.clear_caches()


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_mixed_precision_all_task_heads_against_actual_torch_autocast(mixed_model, dtype):
    from dataclasses import replace

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch

    from rfdetr_jax.models.lwdetr import LWDETR

    spec, params, image, reference = mixed_model
    device = os.environ.get("RFDETR_PRECISION_REFERENCE_DEVICE", "cpu")
    target = jax.devices("gpu" if device == "cuda" else "cpu")[0]
    with torch.no_grad(), torch.autocast(device, dtype=getattr(torch, dtype)):
        expected = reference(torch.from_numpy(image).to(device))
    mixed_spec = replace(spec, compute_dtype=dtype, head_dtype="activation")
    head_names = {"class_embed", "bbox_embed", "keypoint_embed", "query_features_proj"}
    with jax.default_device(target), jax.default_matmul_precision("highest"):
        actual, captured = LWDETR(mixed_spec).apply(
            {"params": params}, jnp.array(image), None, train=False,
            capture_intermediates=lambda module, method: module.name in head_names,
            mutable=["intermediates"])
    for name in {"class_embed", "bbox_embed"} | ({"keypoint_embed"} if spec.use_grouppose_keypoints else set()):
        assert captured["intermediates"][name]["__call__"][0].dtype == jnp.dtype(dtype)
    if spec.segmentation_head:
        projection = captured["intermediates"]["segmentation_head"]["query_features_proj"]["__call__"][0]
        assert projection.dtype == jnp.dtype(dtype)
    tolerances = ({"pred_logits": .01, "pred_boxes": .002, "pred_keypoints": .003, "pred_masks": .1}
                  if dtype == "bfloat16" else
                  {"pred_logits": .002, "pred_boxes": .0003, "pred_keypoints": .0005, "pred_masks": .02})
    fp32_torch = fp32_jax = None
    if device == "cuda":
        with torch.no_grad():
            fp32_torch = reference(torch.from_numpy(image).to(device))
        with jax.default_device(target), jax.default_matmul_precision("highest"):
            fp32_jax = LWDETR(spec).apply({"params": params}, jnp.array(image), None, train=False)
    for key in ("pred_logits", "pred_boxes", "pred_masks", "pred_keypoints"):
        if key in expected:
            ours = np.asarray(actual[key], np.float32)
            theirs = expected[key].float().cpu().numpy()
            print(dtype, key, "maximum_absolute_error", float(np.max(np.abs(ours - theirs))))
            assert np.isfinite(ours).all()
            if fp32_torch is not None:
                torch_base = fp32_torch[key].float().cpu().numpy()
                jax_base = np.asarray(fp32_jax[key], np.float32)
                np.testing.assert_allclose(jax_base, torch_base, atol=1e-4, rtol=1e-4, err_msg=key + " FP32")
                print(dtype, key, "torch_mixed_vs_fp32", float(np.max(np.abs(theirs - torch_base))),
                      "jax_mixed_vs_fp32", float(np.max(np.abs(ours - jax_base))),
                      "fp32_cross_runtime", float(np.max(np.abs(jax_base - torch_base))))
            np.testing.assert_allclose(ours, theirs, atol=tolerances[key], rtol=0, err_msg=key)
            if key == "pred_masks":
                # Mask logits accumulate many low-precision products. Bound the
                # resulting probability error separately, rather than using
                # relative error around logits crossing zero.
                probability_error = np.max(np.abs(1 / (1 + np.exp(-ours)) - 1 / (1 + np.exp(-theirs))))
                print(dtype, "mask_probability_error", float(probability_error))
                assert probability_error < (.02 if dtype == "bfloat16" else .004)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_mixed_head_parameter_gradients_remain_fp32_and_match_torch(mixed_model, dtype):
    import re

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from flax.traverse_util import flatten_dict

    from rfdetr_jax.models.heads.segmentation import MLPBlock
    from rfdetr_jax.models.layers import MLP

    spec, params, _, reference = mixed_model
    if spec.segmentation_head:
        module = MLPBlock(spec.hidden_dim, compute_dtype=dtype)
        head_params = params["segmentation_head"]["query_features_block"]
        torch_head = reference.segmentation_head.query_features_block
    else:
        name = "keypoint_embed" if spec.use_grouppose_keypoints else "bbox_embed"
        module = MLP(spec.hidden_dim, spec.hidden_dim, 8 if spec.use_grouppose_keypoints else 4, 3)
        head_params = params[name]
        torch_head = getattr(reference, name)
    inputs = np.random.default_rng(22).normal(size=(1, 2, spec.hidden_dim)).astype("float32")
    torch_head.zero_grad(set_to_none=True)
    device = os.environ.get("RFDETR_PRECISION_REFERENCE_DEVICE", "cpu")
    target = jax.devices("gpu" if device == "cuda" else "cpu")[0]
    with torch.autocast(device, dtype=getattr(torch, dtype)):
        expected = torch_head(torch.from_numpy(inputs).to(device)).float().square().mean()
    expected.backward()
    with jax.default_device(target), jax.default_matmul_precision("highest"):
        values = jnp.asarray(inputs) if spec.segmentation_head else jnp.asarray(inputs, dtype)
        loss, gradients = jax.value_and_grad(
            lambda current: jnp.square(module.apply({"params": current}, values).astype(jnp.float32)).mean()
        )(head_params)
    actual = {re.sub(r"_(\d+)", r".\1", ".".join(key)): np.asarray(value)
              for key, value in flatten_dict(gradients).items()}
    expected_gradients = {key: value.grad.cpu().numpy() for key, value in torch_head.named_parameters()}
    assert actual.keys() == expected_gradients.keys()
    assert all(value.dtype == np.float32 and np.isfinite(value).all() for value in actual.values())
    keys = sorted(actual)
    ours = np.concatenate([actual[key].ravel() for key in keys])
    theirs = np.concatenate([expected_gradients[key].ravel() for key in keys])
    relative_error = np.linalg.norm(ours - theirs) / np.linalg.norm(theirs)
    print(dtype, "head_gradient_relative_l2_error", float(relative_error))
    assert relative_error < (.03 if dtype == "bfloat16" else .005)
    assert float(loss) == pytest.approx(float(expected.detach()), rel=.02 if dtype == "bfloat16" else .003)


def test_published_keypoint_accumulation_uses_box_weighted_numerators():
    from types import MethodType, SimpleNamespace

    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    pytest.importorskip("rfdetr")
    assert importlib.metadata.version("rfdetr") == "1.11.2"
    from rfdetr.models.criterion import SetCriterion
    from rfdetr.training.module_model import RFDETRModelModule

    parameter = torch.nn.Parameter(torch.tensor(1.))
    model = SimpleNamespace(_accumulated_box_normalizer=None, parameters=lambda: iter([parameter]))
    model._rescale_accumulated_gradients = MethodType(RFDETRModelModule._rescale_accumulated_gradients, model)
    for numerator, denominator in [(2., 1.), (12., 3.)]:
        RFDETRModelModule._scale_loss_for_accumulation(
            model, parameter * numerator, torch.tensor(denominator)).backward()
    assert parameter.grad.item() == 3.5  # Mean of normalized micro-loss gradients would be three.
    model._accumulated_box_normalizer = None
    parameter.grad = None
    RFDETRModelModule._scale_loss_for_accumulation(model, parameter * 10., torch.tensor(2.)).backward()
    assert parameter.grad.item() == 5.

    criterion = SimpleNamespace(training=True, group_detr=3, sum_group_losses=False,
                                _output_device=lambda outputs: torch.device("cpu"))
    targets = [{"labels": torch.tensor([0, 0, 0]), "valid": torch.tensor([True, False, True])},
               {"labels": torch.tensor([0, 0, 0]), "valid": torch.tensor([True, False, False])}]
    normalizer = SetCriterion.num_boxes_for_targets(criterion, {}, targets)
    assert normalizer.item() == 9.
    criterion.sum_group_losses = True
    assert SetCriterion.num_boxes_for_targets(criterion, {}, targets).item() == 3.
    assert SetCriterion.num_boxes_for_targets(criterion, {}, []).item() == 1.


def test_lightning_automatic_partial_window_keeps_configured_divisor():
    torch = pytest.importorskip("torch")
    pl = pytest.importorskip("pytorch_lightning")
    from torch.utils.data import DataLoader

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))

        def training_step(self, batch, batch_idx):
            return self.weight * batch.float().mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=.1)

    model = Model()
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, accumulate_grad_batches=2,
                         logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False)
    trainer.fit(model, DataLoader([2., 4., 6.], batch_size=1))
    assert trainer.global_step == 2
    assert model.weight.item() == pytest.approx(.4)  # 1 - .1*(2+4)/2 - .1*6/2.


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_depthwise_segmentation_gradients_against_actual_upstream(dtype):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax.traverse_util import flatten_dict, unflatten_dict

    torch = pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    assert importlib.metadata.version("rfdetr") == "1.11.2"
    from rfdetr.models.heads.segmentation import DepthwiseConvBlock as TorchBlock

    from rfdetr_jax.models.heads.segmentation import DepthwiseConvBlock

    device = os.environ.get("RFDETR_PRECISION_REFERENCE_DEVICE", "cpu")
    target = jax.devices("gpu" if device == "cuda" else "cpu")[0]
    torch.manual_seed(7)
    reference = TorchBlock(32).to(device)
    inputs = np.random.default_rng(5).normal(size=(1, 32, 8, 8)).astype("float32")
    torch_inputs = torch.tensor(inputs, device=device, requires_grad=True)
    with torch.autocast(device, dtype=getattr(torch, dtype)):
        loss = reference(torch_inputs).float().square().mean()
    loss.backward()
    with jax.default_device(target), jax.default_matmul_precision("highest"):
        params = unflatten_dict({tuple(name.split(".")): jnp.asarray(value.detach().cpu().numpy())
                                 for name, value in reference.named_parameters()})
        block = DepthwiseConvBlock(32, compute_dtype=dtype)
        objective = lambda parameters, values: jnp.square(  # noqa: E731
            block.apply({"params": parameters}, values).astype(jnp.float32)).mean()
        gradients, input_gradient = jax.grad(objective, argnums=(0, 1))(params, jnp.asarray(inputs))
    actual = {".".join(key): np.asarray(value) for key, value in flatten_dict(gradients).items()}
    expected = {key: value.grad.cpu().numpy() for key, value in reference.named_parameters()}
    actual["input"] = np.asarray(input_gradient)
    expected["input"] = torch_inputs.grad.cpu().numpy()
    assert actual.keys() == expected.keys()
    for name, ours in actual.items():
        assert ours.dtype == np.float32 and np.isfinite(ours).all()
        relative = np.linalg.norm(ours - expected[name]) / np.linalg.norm(expected[name])
        print(dtype, "depthwise_gradient", name, "relative_l2", float(relative))
        assert relative < (.03 if dtype == "bfloat16" else .005), name
