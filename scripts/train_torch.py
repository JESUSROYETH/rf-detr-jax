"""Train upstream RF-DETR (PyTorch) on the shared batch stream and log per-epoch metrics.

Run with the torch reference environment. This is a deliberately minimal loop that mirrors
``RFDETRModelModule.configure_optimizers`` / ``training_step`` rather than going through PyTorch
Lightning, so the JAX loop can be compared against it line for line.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# `rfdetr.training.__init__` imports pytorch_lightning, which this minimal loop does not need,
# so load the param-group helper straight from its file instead of through the package.
import importlib.util  # noqa: E402

import rfdetr  # noqa: E402
from rfdetr._namespace import _namespace_from_configs  # noqa: E402
from rfdetr.config import RFDETRNanoConfig, TrainConfig  # noqa: E402
from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config  # noqa: E402

_PARAM_GROUPS_SPEC = importlib.util.spec_from_file_location(
    "_rfdetr_param_groups", Path(rfdetr.__file__).parent / "training" / "param_groups.py"
)
_PARAM_GROUPS = importlib.util.module_from_spec(_PARAM_GROUPS_SPEC)
_PARAM_GROUPS_SPEC.loader.exec_module(_PARAM_GROUPS)
get_param_dict = _PARAM_GROUPS.get_param_dict
from rfdetr.utilities.tensors import NestedTensor  # noqa: E402
from shared_run_config import DEFAULT_RUN, RunConfig  # noqa: E402

from rfdetr_jax.datasets.coco import CocoDetection, iterate_batches, prefetch  # noqa: E402
from rfdetr_jax.evaluation.coco_eval import accumulate_detections, evaluate_detections  # noqa: E402


def build_configs(run: RunConfig) -> tuple[RFDETRNanoConfig, TrainConfig]:
    """Build the upstream model and train configs for a run."""
    model_config = RFDETRNanoConfig(
        pretrain_weights=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        resolution=run.resolution,
        positional_encoding_size=run.positional_encoding_size,
        num_classes=run.num_classes,
        group_detr=run.group_detr,
        num_queries=run.num_queries,
        num_select=run.num_select,
        dec_layers=run.dec_layers,
        gradient_checkpointing=run.gradient_checkpointing,
    )
    train_config = TrainConfig(
        dataset_dir=run.dataset_dir,
        output_dir="output",
        lr=run.lr,
        lr_encoder=run.lr_encoder,
        weight_decay=run.weight_decay,
        lr_vit_layer_decay=run.lr_vit_layer_decay,
        lr_component_decay=run.lr_component_decay,
        batch_size=run.batch_size,
        epochs=run.epochs,
        use_ema=run.use_ema,
        ema_decay=run.ema_decay,
        ema_tau=run.ema_tau,
        lr_drop=run.lr_drop,
    )
    return model_config, train_config


def to_torch_batch(batch: dict, device: torch.device) -> tuple[NestedTensor, list[dict]]:
    """Convert a shared numpy batch into the model's ``(NestedTensor, targets)`` input."""
    pixel_values = torch.from_numpy(batch["pixel_values"]).to(device)
    mask = torch.zeros(
        (pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3]), dtype=torch.bool, device=device
    )
    targets = []
    for index in range(pixel_values.shape[0]):
        keep = batch["target_mask"][index]
        targets.append(
            {
                "labels": torch.from_numpy(batch["labels"][index][keep].astype("int64")).to(device),
                "boxes": torch.from_numpy(batch["boxes"][index][keep]).to(device),
            }
        )
    return NestedTensor(pixel_values, mask), targets


@torch.no_grad()
def evaluate(model, run: RunConfig, dataset: CocoDetection, device: torch.device) -> dict[str, float]:
    """Score the model on the validation split with COCO detection metrics."""
    model.eval()
    detections: list[dict] = []
    for batch in iterate_batches(
        dataset, run.resolution, run.batch_size, run.max_targets, train=False, drop_last=False
    ):
        samples, _ = to_torch_batch(batch, device)
        outputs = model(samples)
        probabilities = outputs["pred_logits"].sigmoid().flatten(1)
        num_classes = outputs["pred_logits"].shape[-1]
        k = min(run.num_select, probabilities.shape[1])
        scores, indexes = torch.topk(probabilities, k, dim=1)
        query_index = indexes // num_classes
        labels = indexes % num_classes

        from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy

        boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"])
        boxes = torch.gather(boxes, 1, query_index.unsqueeze(-1).expand(-1, -1, 4))
        sizes = torch.from_numpy(batch["orig_sizes"]).to(device).float()
        scale = torch.stack([sizes[:, 1], sizes[:, 0], sizes[:, 1], sizes[:, 0]], dim=1)[:, None, :]
        boxes = (boxes * scale).clamp(min=0)
        boxes = torch.minimum(boxes, scale)
        detections.extend(
            accumulate_detections(
                batch["image_ids"],
                scores.cpu().numpy(),
                labels.cpu().numpy(),
                boxes.cpu().numpy(),
                dataset.category_ids,
            )
        )
    model.train()
    return evaluate_detections(run.val_paths()[1], detections, max_dets=run.num_select)


def main() -> None:
    """Train for the configured number of epochs and write a JSON log."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_RUN.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_RUN.batch_size)
    parser.add_argument("--out", default="artifacts/train_torch.json")
    parser.add_argument("--init-weights", default="artifacts/init_state_dict.npz")
    parser.add_argument("--save-init", action="store_true", help="Write the initial weights for the JAX run.")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-steps", type=int, default=10, help="Record this many per-step losses.")
    args = parser.parse_args()

    # On a shared machine an unbounded intra-op pool lets one training process saturate every
    # core; cap it like scripts/torchenv.sh does.
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))

    run = RunConfig(epochs=args.epochs, batch_size=args.batch_size)
    torch.manual_seed(run.seed)
    np.random.seed(run.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config, train_config = build_configs(run)
    model = build_model_from_config(model_config, train_config).to(device)
    criterion, _ = build_criterion_from_config(model_config, train_config)
    criterion.to(device)

    if args.save_init:
        Path(args.init_weights).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.init_weights,
            **{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()},
        )
        print(f"saved initial weights to {args.init_weights}")
    elif Path(args.init_weights).exists():
        with np.load(args.init_weights) as data:
            model.load_state_dict({key: torch.from_numpy(data[key]) for key in data.files}, strict=True)
        print(f"loaded initial weights from {args.init_weights}")

    namespace = _namespace_from_configs(model_config, train_config)
    param_dicts = [group for group in get_param_dict(namespace, model) if group["params"].requires_grad]
    optimizer = torch.optim.AdamW(param_dicts, lr=run.lr, weight_decay=run.weight_decay)

    train_dataset = CocoDetection(*run.train_paths())
    val_dataset = CocoDetection(*run.val_paths())
    steps_per_epoch = len(train_dataset) // run.batch_size
    total_steps = steps_per_epoch * run.epochs
    warmup_steps = int(run.warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int) -> float:
        """Reproduce RF-DETR's managed step schedule."""
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if run.lr_scheduler == "cosine":
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0 if current_step < run.lr_drop * steps_per_epoch else 0.1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    ema_model = None
    ema_updates = 1
    if run.use_ema:
        import copy

        ema_model = copy.deepcopy(model).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)

    batches = prefetch(
        iterate_batches(
            train_dataset,
            run.resolution,
            run.batch_size,
            run.max_targets,
            train=True,
            seed=run.data_seed,
            epochs=run.epochs,
        )
    )

    model.train()
    criterion.train()
    log: dict = {"framework": "torch", "config": run.__dict__, "epochs": [], "first_steps": []}
    for epoch in range(run.epochs):
        started = time.time()
        totals: dict[str, float] = {}
        count = 0
        for step_index in range(steps_per_epoch):
            batch = next(batches)
            samples, targets = to_torch_batch(batch, device)
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            weights = criterion.weight_dict
            loss = torch.stack([loss_dict[k] * weights[k] for k in loss_dict if k in weights]).sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), run.clip_max_norm)
            optimizer.step()
            scheduler.step()

            if ema_model is not None:
                decay = run.ema_decay
                if run.ema_tau != 0:
                    decay = run.ema_decay * (1 - math.exp(-ema_updates / run.ema_tau))
                with torch.no_grad():
                    for ema_value, model_value in zip(
                        ema_model.state_dict().values(), model.state_dict().values(), strict=True
                    ):
                        ema_value.copy_(decay * ema_value + (1.0 - decay) * model_value)
                ema_updates += 1

            if epoch == 0 and len(log["first_steps"]) < args.log_steps:
                log["first_steps"].append(
                    {
                        "loss": float(loss),
                        "grad_norm": float(grad_norm),
                        **{name: float(value) for name, value in loss_dict.items() if name in weights},
                    }
                )
            totals["loss"] = totals.get("loss", 0.0) + float(loss)
            totals["grad_norm"] = totals.get("grad_norm", 0.0) + float(grad_norm)
            for name, value in loss_dict.items():
                if name in weights:
                    totals[name] = totals.get(name, 0.0) + float(value)
            count += 1
            if step_index % 20 == 0:
                print(f"  epoch {epoch} step {step_index}/{steps_per_epoch}", flush=True)

        record = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "steps": count,
            "losses": {name: value / max(1, count) for name, value in totals.items()},
        }
        if (epoch + 1) % args.eval_every == 0 or epoch == run.epochs - 1:
            record["val"] = evaluate(model, run, val_dataset, device)
            if ema_model is not None:
                record["val_ema"] = evaluate(ema_model, run, val_dataset, device)
        log["epochs"].append(record)
        print(
            f"epoch {epoch}: loss={record['losses']['loss']:.4f} "
            f"mAP={record.get('val', {}).get('mAP', float('nan')):.4f} "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(log, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
