#!/usr/bin/env python3
"""Supervised gate for the three mode-representation variants.

This script intentionally uses only analytical manufactured fields.  It does
not train the grating and does not consume any RCWA pointwise data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config
from src.mode_aware_fourier import (ExplicitFourierModalNetwork, ModeAwareFeatureMLP,
                                    mode_specification)
from src.maxwell_feature_variants import SubdomainMLP_Variant


def exact(x, z, p, n, m):
    spec = mode_specification(m, p.period, p.k0, n)
    e = torch.exp(1j * (spec["kx"] * x - spec["kz"] * z))
    return torch.stack((e.real, e.imag, (spec["kz"] / p.k0 * e).real,
                        (spec["kz"] / p.k0 * e).imag, (spec["kx"] / p.k0 * e).real,
                        (spec["kx"] / p.k0 * e).imag), dim=1), spec


def metrics(model, p, n, m):
    x = torch.linspace(0., p.period, 96, dtype=torch.float64)
    z = torch.linspace(0., p.domain_height, 72, dtype=torch.float64)
    X, Z = torch.meshgrid(x, z, indexing="xy")
    with torch.no_grad(): out = model(X.ravel(), Z.ravel())
    truth, spec = exact(X.ravel(), Z.ravel(), p, n, m)
    e, et = out[:, 0]+1j*out[:, 1], truth[:, 0]+1j*truth[:, 1]
    h, ht = out[:, 2]+1j*out[:, 3], truth[:, 2]+1j*truth[:, 3]
    # Sample at z=0 with an endpoint-exclusive DFT-compatible grid.
    xm = torch.linspace(0., p.period, 128+1, dtype=torch.float64)[:-1]
    with torch.no_grad(): em = model(xm, torch.zeros_like(xm))[:, 0].numpy()+1j*model(xm, torch.zeros_like(xm))[:, 1].numpy()
    amp = np.mean(em * np.exp(-1j * spec["kx"] * xm.numpy()))
    return {"complex_E_error": float(torch.linalg.vector_norm(e-et)/torch.linalg.vector_norm(et)),
            "complex_H_error": float(torch.linalg.vector_norm(h-ht)/(torch.linalg.vector_norm(ht)+1e-12)),
            "phase_error_deg": float(torch.sqrt(torch.mean(torch.angle(e/(et+1e-12))**2))*180/np.pi),
            "modal_amplitude_error": float(abs(abs(amp)-1.)),
            "kx": spec["kx"], "kz_real": spec["kz"].real, "kz_imag": spec["kz"].imag,
            "status": spec["status"]}


def train(model, p, n, m, epochs):
    rng = np.random.default_rng(42); torch.manual_seed(42)
    x = torch.tensor(rng.uniform(0., p.period, 1024), dtype=torch.float64)
    z = torch.tensor(rng.uniform(0., p.domain_height, 1024), dtype=torch.float64)
    target, _ = exact(x, z, p, n, m)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad(); loss = torch.mean((model(x,z)-target)**2); loss.backward(); opt.step()
    return model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--output-dir", default="outputs/mode_aware_representation")
    args = ap.parse_args(); out = ROOT / args.output_dir
    if out.exists(): raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    p = load_config(ROOT / "configs/default.yaml").physics; p.period = .8; p.ridge_width = .32
    cases = [("air", 0), ("substrate", 0), ("air", 1), ("air", -1), ("substrate", 1), ("substrate", -1)]
    report = {"training_data": "analytical manufactured modes only; no RCWA pointwise anchor",
              "modal_data_loss_weight": 0, "cases": [], "valid": False}
    for medium, m in cases:
        n = p.n_air if medium == "air" else p.n_substrate
        a = train(SubdomainMLP_Variant(0., p.domain_height, n*p.k0, p.period, p.k0,
                                       "global_k0", 4, 64, 4).double(), p, n, m, args.epochs)
        b = train(ModeAwareFeatureMLP(orders=(-1,0,1), period=p.period, k0=p.k0, n=n,
                                      z_lo=0., z_hi=p.domain_height, hidden_layers=4, hidden_width=64).double(), p, n, m, args.epochs)
        c = ExplicitFourierModalNetwork(orders=(-1,0,1), period=p.period, k0=p.k0, n=n,
                                        z_lo=0., z_hi=p.domain_height, use_coefficient_mlp=False).double()
        c.set_exact_mode(m)
        item = {"medium": medium, "n": n, "order": m,
                "global_k0_mlp": metrics(a,p,n,m), "mode_aware_feature_mlp": metrics(b,p,n,m),
                "explicit_fourier_modal": metrics(c,p,n,m)}
        report["cases"].append(item)
    report["valid"] = all(c["explicit_fourier_modal"]["complex_E_error"] < .01 and c["explicit_fourier_modal"]["phase_error_deg"] < 1.
                          for c in report["cases"])
    report["invalid_reason"] = None if report["valid"] else "One or more manufactured representation gates failed."
    (out / "mode_aware_representation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
