#!/usr/bin/env python3
"""Find the checkpoint with the smallest val_loss."""

import argparse
import re
from pathlib import Path


def find_best_checkpoint(logs_dir: str, top_n: int = 5) -> None:
    """Find checkpoints with smallest val_loss.

    Args:
        logs_dir: Directory containing experiment folders
        top_n: Number of top checkpoints to display
    """
    logs_path = Path(logs_dir)

    if not logs_path.exists():
        print(f"Error: Directory '{logs_dir}' not found")
        return

    print(f"Searching for best checkpoints in: {logs_path.resolve()}")
    print("-" * 60)

    # Pattern to extract val_loss from filename
    pattern = re.compile(r"epoch=(\d+)-val_loss=([0-9.]+)\.ckpt$")

    all_ckpts = []

    # Check if this is a ckpts directory directly (contains .ckpt files)
    direct_ckpts = list(logs_path.glob("*.ckpt"))
    if direct_ckpts:
        # Direct ckpts directory
        exp_name = logs_path.parent.name if logs_path.name == "ckpts" else logs_path.name
        for ckpt_file in direct_ckpts:
            match = pattern.search(ckpt_file.name)
            if match:
                epoch = int(match.group(1))
                val_loss = float(match.group(2))
                all_ckpts.append({
                    "exp_name": exp_name,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "path": str(ckpt_file),
                })
    else:
        # Search in subdirectories (logs/exp_name/ckpts structure)
        for exp_dir in logs_path.iterdir():
            if not exp_dir.is_dir():
                continue

            ckpts_dir = exp_dir / "ckpts"
            if not ckpts_dir.exists():
                continue

            for ckpt_file in ckpts_dir.glob("*.ckpt"):
                match = pattern.search(ckpt_file.name)
                if match:
                    epoch = int(match.group(1))
                    val_loss = float(match.group(2))
                    all_ckpts.append({
                        "exp_name": exp_dir.name,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "path": str(ckpt_file),
                    })

    if not all_ckpts:
        print("No checkpoint files found")
        return

    # Sort by val_loss
    all_ckpts.sort(key=lambda x: x["val_loss"])

    print(f"\nTop {top_n} checkpoints by val_loss:\n")
    for i, ckpt in enumerate(all_ckpts[:top_n], 1):
        print(f"{i}. val_loss={ckpt['val_loss']:.5f} | epoch={ckpt['epoch']:03d} | {ckpt['exp_name']}")

    print("\n" + "=" * 60)
    print("BEST CHECKPOINT:")
    print("=" * 60)
    best = all_ckpts[0]
    print(f"  Experiment: {best['exp_name']}")
    print(f"  Epoch:      {best['epoch']}")
    print(f"  val_loss:   {best['val_loss']:.5f}")
    print(f"  Path:       {best['path']}")
    print("\nTest command:")
    print(f"pixi run python src/hepattn/experiments/lhcb/main.py test \\")
    print(f"  --config src/hepattn/experiments/lhcb/config/base.yaml \\")
    print(f"  --ckpt_path {best['path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find checkpoint with smallest val_loss")
    parser.add_argument(
        "logs_dir",
        nargs="?",
        default="logs",
        help="Directory containing experiment folders (default: logs)",
    )
    parser.add_argument(
        "-n",
        "--top",
        type=int,
        default=5,
        help="Number of top checkpoints to display (default: 5)",
    )

    args = parser.parse_args()
    find_best_checkpoint(args.logs_dir, args.top)
