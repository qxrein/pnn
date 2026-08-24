"""Strict metadata validation for RCWA references used in new PINN runs."""
from __future__ import annotations
from pathlib import Path
import numpy as np

STALE_REFERENCE_NAME = "reference_lambda_0p8.npz"

def validate_reference(path: str | Path, physics) -> dict:
    path = Path(path)
    if path.name == STALE_REFERENCE_NAME:
        raise ValueError(f"{path} is stale/invalid for new experiments; use the geometry-consistent reference")
    with np.load(path, allow_pickle=False) as data:
        required = ("field_representation", "wavelength", "period", "ridge_width", "ridge_height",
                    "n_ridge", "n_substrate", "geometry_convention", "c_refl", "c_trans",
                    "R_m", "T_m", "z_top_monitor", "z_bot_monitor")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Reference missing required metadata: {', '.join(missing)}")
        if str(data["field_representation"].item()) != "total":
            raise ValueError("Reference field representation must be total")
        expected = {"wavelength": physics.wavelength, "period": physics.period,
                    "ridge_width": physics.ridge_width, "ridge_height": physics.ridge_height,
                    "n_ridge": physics.n_ridge, "n_substrate": physics.n_substrate}
        for key, value in expected.items():
            if not np.isclose(float(data[key]), value, rtol=0., atol=1e-12):
                raise ValueError(f"Reference {key}={float(data[key])} does not match PINN configuration {value}")
        geometry = str(data["geometry_convention"].item())
        if geometry != "dielectric ridge on substrate; substrate outside ridge footprint":
            raise ValueError(f"Unsupported reference geometry convention: {geometry}")
        energy = float(data["R_total"]) + float(data["T_total"])
        if not np.isclose(energy, 1.0, atol=1e-7):
            raise ValueError(f"Reference energy is not physical: R+T={energy}")
        return {"path": str(path), "field_representation": "total", "geometry_convention": geometry,
                "R_total": float(data["R_total"]), "T_total": float(data["T_total"]),
                "energy_balance": energy, "z_top_monitor": float(data["z_top_monitor"]),
                "z_bot_monitor": float(data["z_bot_monitor"]), "n_orders": int((len(data["c_refl"])-1)//2)}
