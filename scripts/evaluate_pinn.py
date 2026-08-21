#!/usr/bin/env python3
"""Evaluate a trained PINN checkpoint.

Optional independent reference validation
-----------------------------------------
Provide a reference NPZ file (exported from an external EM solver) via the
``--reference`` flag to compare PINN predictions against ground-truth fields::

    python scripts/evaluate_pinn.py \\
        --config configs/default.yaml \\
        --checkpoint outputs/checkpoints/best_model.pt \\
        --reference path/to/reference.npz

If ``--reference`` is omitted, or the specified file does not exist, the
reference comparison step is skipped and a clear message is printed.
Exit code is always 0 when the PINN evaluation itself succeeds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluate import evaluate_pinn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained PINN checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Optional path to a reference-field NPZ file for independent "
            "validation.  Expected keys: x (Nx,), z (Nz,), "
            "E_real (Nz, Nx), E_imag (Nz, Nx)."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint

    reference_path: Path | None = None
    if args.reference is not None:
        reference_path = Path(args.reference)
        if not reference_path.is_absolute():
            reference_path = ROOT / reference_path

    config = load_config(config_path)
    result = evaluate_pinn(config, checkpoint, reference_path=reference_path)

    print("Evaluation complete.")
    for k, v in result.items():
        if k == "metrics":
            print("  metrics:")
            for mk, mv in v.items():
                print(f"    {mk}: {mv}")
        elif k == "reference_validation":
            if v is None:
                if reference_path is None:
                    print("  reference_validation: skipped (no --reference provided)")
                else:
                    print("  reference_validation: skipped (file not found)")
            else:
                print("  reference_validation:")
                for rk, rv in v.items():
                    print(f"    {rk}: {rv}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
