#!/usr/bin/env python3
"""Export a solver-side RCWA companion NPZ.  Never writes the canonical file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reference_companion import (
    COMPANION_FILENAME,
    NFINE_DEFAULT,
    write_companion,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RCWA vector-field companion")
    parser.add_argument(
        "--canonical",
        default="outputs/reference_lambda_0p8_geometry_consistent_20260824.npz",
        help="Immutable canonical RCWA NPZ (read-only)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reference_companion",
        help="Directory for companion NPZ and manifest.json",
    )
    parser.add_argument("--filename", default=COMPANION_FILENAME)
    parser.add_argument("--nfine", type=int, default=NFINE_DEFAULT)
    args = parser.parse_args()

    canonical = Path(args.canonical)
    if not canonical.is_absolute():
        canonical = ROOT / canonical
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    if not canonical.exists():
        raise SystemExit(f"Canonical reference not found: {canonical}")

    manifest = write_companion(
        canonical, output_dir, nfine=args.nfine, filename=args.filename,
    )
    print(f"companion path          : {manifest['companion_path']}")
    print(f"canonical SHA-256       : {manifest['canonical_sha256']}")
    print(f"companion SHA-256       : {manifest['companion_sha256']}")
    print(f"generator commit        : {manifest['generator_commit']}")
    print(f"configuration hash      : {manifest['generator_config_hash']}")
    print(f"maximum reconstruction error : {manifest['max_reconstruction_error']:.3e}")
    print(f"boundary coefficient error   : {manifest['boundary_coefficient_error']:.3e}")
    print("residual norms by region:")
    for region, norms in manifest["residual_norms_by_region"].items():
        print(f"  {region}: {norms}")
    print("homogeneous-slab residuals:")
    for name, value in manifest["residual_homogeneous_slabs"].items():
        print(f"  {name}: {value:.3e}")
    print("interface jump norms:")
    for name, norms in manifest["interface_jump_norms"].items():
        print(f"  {name}: {norms}")


if __name__ == "__main__":
    main()
