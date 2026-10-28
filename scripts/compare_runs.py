"""Summarise a PyTorch run and a JAX run side by side."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    """Read a training log written by ``train_torch.py`` or ``train_jax.py``."""
    return json.loads(Path(path).read_text())


def _format_row(label: str, left: float | None, right: float | None, width: int = 13) -> str:
    """Render one comparison row with an absolute and relative delta."""
    if left is None or right is None:
        return f"  {label:<20s} {'-':>{width}} {'-':>{width}} {'-':>{width}}"
    delta = right - left
    relative = abs(delta) / max(1e-9, abs(left))
    return (
        f"  {label:<20s} {left:>{width}.5f} {right:>{width}.5f} "
        f"{delta:>+{width}.5f}  ({relative * 100:5.2f}%)"
    )


def _load_step_losses() -> dict[str, dict]:
    """Load whichever single-step loss measurements ``check_step_loss_parity.py`` has written."""
    results = {}
    for key in ("torch_g1", "jax_g1", "torch_g13", "jax_g13", "torch_g13_cuda"):
        path = Path(f"artifacts/step_loss_{key}.json")
        if path.exists():
            results[key] = json.loads(path.read_text())
    return results


def _render_markdown(torch_log: dict, jax_log: dict, epochs: int) -> str:
    """Render the side-by-side comparison as a Markdown report."""
    config = torch_log["config"]
    torch_last = torch_log["epochs"][epochs - 1]
    jax_last = jax_log["epochs"][epochs - 1]
    lines = [
        "# RF-DETR: PyTorch vs JAX training comparison",
        "",
        "**Summary.** Over "
        f"{epochs} epochs: final training loss "
        f"{torch_last['losses']['loss']:.4f} (PyTorch) vs {jax_last['losses']['loss']:.4f} (JAX), and "
        f"EMA mAP {torch_last.get('val_ema', {}).get('mAP', float('nan')):.4f} vs "
        f"{jax_last.get('val_ema', {}).get('mAP', float('nan')):.4f}. "
        "These observations do not establish statistical equivalence or explain the differences.",
        "",
        "The comparison recipe shares initialization and a NumPy data pipeline. Verify that the",
        "input runs used the same weights, data, settings and batch stream before interpreting",
        "their differences. This report does not verify those identities from the logs.",
        "",
        "## Setup (PyTorch log; compare the JAX configuration separately)",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Model | RF-DETR geometry (`{config['encoder']}`, {config['dec_layers']} decoder layers) |",
        f"| Resolution | {config['resolution']} |",
        f"| Query groups (`group_detr`) | {config['group_detr']} |",
        f"| Queries | {config['num_queries']} |",
        f"| Dataset | `{config['dataset_dir']}` |",
        f"| Batch size | {config['batch_size']} |",
        f"| Epochs | {epochs} |",
        f"| Optimizer | AdamW, lr={config['lr']}, encoder lr={config['lr_encoder']}, wd={config['weight_decay']} |",
        f"| Layer-wise decay | ViT {config['lr_vit_layer_decay']}, component {config['lr_component_decay']} |",
        f"| Grad clip | {config['clip_max_norm']} |",
        f"| EMA | decay={config['ema_decay']}, tau={config['ema_tau']} |",
        "",
    ]

    step_losses = _load_step_losses()
    if step_losses:
        lines += [
            "## Available single-step measurements",
            "",
            "Local single-step loss files, labeled by the reference script's backend/device convention",
            "(`scripts/check_step_loss_parity.py`):",
            "",
            "| Backend | Device | `group_detr` | Total loss |",
            "| --- | --- | ---: | ---: |",
        ]
        for label, device, groups, key in (
            ("PyTorch", "CPU", 1, "torch_g1"),
            ("JAX", "GPU", 1, "jax_g1"),
            ("PyTorch", "CPU", 13, "torch_g13"),
            ("JAX", "GPU", 13, "jax_g13"),
            ("PyTorch", "CUDA", 13, "torch_g13_cuda"),
        ):
            if key in step_losses:
                lines.append(f"| {label} | {device} | {groups} | {step_losses[key]['total']:.6f} |")
        lines += [
            "",
            "These are separate local measurements; their connection to the training runs must be",
            "verified from their recorded configurations. Rounded agreement is not bitwise equality. Two-stage",
            "query selection can amplify near ties by changing proposal/query pairings, but these",
            "tables alone do not establish it as the cause of all run-level differences.",
            "",
        ]

    first_torch = torch_log.get("first_steps") or []
    first_jax = jax_log.get("first_steps") or []
    if first_torch and first_jax:
        lines += [
            "## Step-by-step agreement",
            "",
            "The recorded first optimizer steps; initialization and batch identity are not checked here.",
            "",
            "| Step | PyTorch loss | JAX loss | Relative difference |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for index, (left, right) in enumerate(zip(first_torch, first_jax, strict=False)):
            relative = abs(right["loss"] - left["loss"]) / max(1e-9, abs(left["loss"]))
            lines.append(f"| {index} | {left['loss']:.6f} | {right['loss']:.6f} | {relative * 100:.4f}% |")
        lines.append("")

    lines += [
        "## Per-epoch results",
        "",
        "| Epoch | PyTorch loss | JAX loss | PyTorch mAP | JAX mAP | PyTorch mAP@50 | JAX mAP@50 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index in range(epochs):
        left = torch_log["epochs"][index]
        right = jax_log["epochs"][index]
        left_val = left.get("val", {})
        right_val = right.get("val", {})
        lines.append(
            f"| {index} | {left['losses']['loss']:.4f} | {right['losses']['loss']:.4f} | "
            f"{left_val.get('mAP', float('nan')):.4f} | {right_val.get('mAP', float('nan')):.4f} | "
            f"{left_val.get('mAP_50', float('nan')):.4f} | {right_val.get('mAP_50', float('nan')):.4f} |"
        )

    lines += [
        "",
        "## Loss components, final epoch",
        "",
        "| Loss | PyTorch | JAX | Relative difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    left = torch_log["epochs"][epochs - 1]["losses"]
    right = jax_log["epochs"][epochs - 1]["losses"]
    for key in sorted(set(left) & set(right)):
        relative = abs(right[key] - left[key]) / max(1e-9, abs(left[key]))
        lines.append(f"| `{key}` | {left[key]:.5f} | {right[key]:.5f} | {relative * 100:.2f}% |")

    torch_epochs = [entry["seconds"] for entry in torch_log["epochs"][:epochs]]
    jax_epochs = [entry["seconds"] for entry in jax_log["epochs"][:epochs]]
    torch_steady = sum(torch_epochs[1:]) / max(1, len(torch_epochs) - 1)
    jax_steady = sum(jax_epochs[1:]) / max(1, len(jax_epochs) - 1)
    lines += [
        "",
        "## Wall clock",
        "",
        "| Backend | Epoch 0 | Steady-state per epoch | Total |",
        "| --- | ---: | ---: | ---: |",
        f"| PyTorch | {torch_epochs[0]:.1f}s | {torch_steady:.1f}s | {sum(torch_epochs):.1f}s |",
        f"| JAX | {jax_epochs[0]:.1f}s | {jax_steady:.1f}s | {sum(jax_epochs):.1f}s |",
        "",
        "These are the logged training-epoch durations, not total process wall time. "
        "Compilation, evaluation, precompilation and checkpoint I/O may be accounted for differently "
        "by the input runs. Confirm timing boundaries and hardware before comparing throughput. "
        "Warm-cache behavior depends on cache compatibility and is not measured by this table.",
        "",
        "## Reproducing",
        "",
        "Illustrative synthetic-data recipe: pin source/environment versions and supply the full",
        "recorded configuration to reproduce a particular result.",
        "",
        "```bash",
        "# 1. Generate the synthetic COCO dataset (uses upstream's generator)",
        "./.venv-torch/bin/python -c \\",
        "  \"from rfdetr.datasets.synthetic import generate_coco_dataset as g; \\",
        "   g('data/synthetic', 480, img_size=512, min_objects=1, max_objects=6, split_ratios=(0.75, 0.25))\"",
        "",
        "# 2. PyTorch baseline (also writes the shared initial weights)",
        "./scripts/torchenv.sh ./.venv-torch/bin/python scripts/train_torch.py \\",
        f"    --epochs {epochs} --batch-size {config['batch_size']} --out artifacts/train_torch.json --save-init",
        "",
        "# 3. JAX run, from the same weights and the same batches",
        "./scripts/jaxenv.sh ./.venv-jax/bin/python scripts/train_jax.py \\",
        f"    --epochs {epochs} --batch-size {config['batch_size']} --out artifacts/train_jax.json",
        "",
        "# 4. Report",
        "python3 scripts/compare_runs.py --markdown artifacts/comparison.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Print a per-epoch loss and mAP comparison between the two runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch", default="artifacts/train_torch.json")
    parser.add_argument("--jax", default="artifacts/train_jax.json")
    parser.add_argument("--out", default=None, help="Optional path for a machine-readable summary.")
    parser.add_argument("--markdown", default=None, help="Optional path for a Markdown report.")
    args = parser.parse_args()

    torch_log = load(args.torch)
    jax_log = load(args.jax)
    epochs = min(len(torch_log["epochs"]), len(jax_log["epochs"]))

    summary: dict = {"epochs": []}

    first_torch_steps = torch_log.get("first_steps") or []
    first_jax_steps = jax_log.get("first_steps") or []
    if first_torch_steps and first_jax_steps:
        print("Recorded first optimizer steps (weight and batch identity not verified)")
        print(f"  {'step':>4s} {'torch loss':>13s} {'jax loss':>13s} {'rel. diff':>11s}")
        step_rows = []
        for index, (left, right) in enumerate(zip(first_torch_steps, first_jax_steps, strict=False)):
            relative = abs(right["loss"] - left["loss"]) / max(1e-9, abs(left["loss"]))
            step_rows.append({"step": index, "torch": left["loss"], "jax": right["loss"], "relative": relative})
            print(f"  {index:>4d} {left['loss']:>13.6f} {right['loss']:>13.6f} {relative * 100:>10.4f}%")
        summary["first_steps"] = step_rows

    print(f"\n{'':22s}{'PyTorch':>13s}{'JAX':>13s}{'delta':>13s}")
    for index in range(epochs):
        left = torch_log["epochs"][index]
        right = jax_log["epochs"][index]
        print(f"\nepoch {index}  ({left['seconds']:.1f}s torch vs {right['seconds']:.1f}s jax)")
        keys = [key for key in left["losses"] if key in right["losses"]]
        for key in sorted(keys):
            print(_format_row(key, left["losses"][key], right["losses"][key]))
        for section in ("val", "val_ema"):
            if section in left and section in right:
                for metric in ("mAP", "mAP_50", "mAR_max"):
                    print(_format_row(f"{section}/{metric}", left[section].get(metric), right[section].get(metric)))
        summary["epochs"].append(
            {
                "epoch": index,
                "torch": {"losses": left["losses"], "val": left.get("val"), "seconds": left["seconds"]},
                "jax": {"losses": right["losses"], "val": right.get("val"), "seconds": right["seconds"]},
            }
        )

    first_torch = torch_log["epochs"][0]["losses"]["loss"]
    first_jax = jax_log["epochs"][0]["losses"]["loss"]
    print(
        f"\nepoch-0 mean loss: torch={first_torch:.5f} jax={first_jax:.5f} "
        f"(relative difference {abs(first_jax - first_torch) / max(1e-9, abs(first_torch)) * 100:.3f}%)"
    )
    torch_seconds = sum(entry["seconds"] for entry in torch_log["epochs"][:epochs])
    jax_seconds = sum(entry["seconds"] for entry in jax_log["epochs"][:epochs])
    print(f"total train time: torch={torch_seconds:.1f}s jax={jax_seconds:.1f}s")
    summary["totals"] = {"torch_seconds": torch_seconds, "jax_seconds": jax_seconds}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")

    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(_render_markdown(torch_log, jax_log, epochs))
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
