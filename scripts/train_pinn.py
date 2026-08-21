#!/usr/bin/env python3
"""Train the grating Helmholtz PINN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.train import train_pinn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 2D Helmholtz PINN for grating scattering")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--smoke", action="store_true", help="Apply smoke-test overrides")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config = load_config(config_path, smoke=args.smoke)
    summary = train_pinn(config)
    print("Training complete.")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
