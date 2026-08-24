#!/usr/bin/env python3
"""Render total-field comparisons from already-completed Phase 6 checkpoints."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.maxwell_layered_bg import compute_background_coefficients
from src.mode_aware_fourier import ExplicitFourierModalDD
from src.reference_data import interpolate_reference_to_grid,load_reference_npz,normalize_reference_orientation
from scripts.train_lbg import evaluate,make_lambda_0p8

p=make_lambda_0p8(load_config(ROOT/'configs/default.yaml').physics);ref=normalize_reference_orientation(load_reference_npz(ROOT/'outputs/reference_lambda_0p8_geometry_consistent_20260824.npz'))
for seed in (42,123):
    ck=torch.load(ROOT/f'outputs/phase6/seed{seed}/best_checkpoint.pt',map_location='cpu',weights_only=True)
    model=ExplicitFourierModalDD(p,modal_order_max=ck['modal_order_max'],use_coefficient_mlp=True).double();model.load_state_dict(ck['state_dict'])
    fields=evaluate(model,p,torch.device('cpu'),torch.float64,'layered_bg',compute_background_coefficients(p));rr,ri,_=interpolate_reference_to_grid(ref,fields['x'][0],fields['z'][:,0])
    pin=fields['E_real']+1j*fields['E_imag'];rcwa=rr+1j*ri;extent=[fields['x'].min(),fields['x'].max(),fields['z'].max(),fields['z'].min()]
    fig,ax=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    for a,value,title,cmap in ((ax[0],abs(pin),'PINN |E total|','viridis'),(ax[1],abs(rcwa),'RCWA |E total|','viridis'),(ax[2],abs(pin-rcwa),'|PINN − RCWA|','magma')):
        im=a.imshow(value,extent=extent,aspect='auto',origin='upper',cmap=cmap);a.set(title=title,xlabel='x',ylabel='z');fig.colorbar(im,ax=a)
    fig.savefig(ROOT/f'outputs/phase6/total_field_comparison_seed{seed}.png',dpi=160);plt.close(fig)
