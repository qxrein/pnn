#!/usr/bin/env python3
"""Generate publication figures from evaluation NPZ results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluate import load_checkpoint
from src.utils import resolve_device
from src.visualization import generate_all_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PINN evaluation results")
    parser.add_argument("--results", type=str, required=True, help="Path to results.npz")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint for PDE residual plot")
    parser.add_argument("--history", type=str, default=None, help="Optional training history CSV")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = ROOT / results_path

    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    output_dir = Path(args.output_dir) if args.output_dir else Path(config.paths.figure_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    history = Path(args.history) if args.history else Path(config.paths.history_file)
    if not history.is_absolute():
        history = ROOT / history

    model = config_obj = device = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        device = resolve_device(config.training.device)
        model, ckpt_meta = load_checkpoint(checkpoint, device)
        config_obj = ckpt_meta["config"]

    paths = generate_all_figures(
        results_npz=results_path,
        history_file=history,
        output_dir=output_dir,
        model=model,
        config=config_obj or config,
        device=device,
    )
    print(f"Generated {len(paths)} figures in {output_dir}")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
