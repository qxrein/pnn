#!/usr/bin/env python3
"""Phase 5 source, sampling, lateral-mode, and compact contrast diagnostics.

No RCWA field samples enter any training loss.  RCWA is generated only after
training for evaluation of the contrast sweep.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PhysicsConfig, load_config
from src.diagnostic_sampling import deterministic_region_samples
from src.field_comparison import compare_fields, extract_modal_amplitudes
from src.maxwell_feature_variants import SubdomainMLP_Variant, Maxwell2DDD_Variant
from src.maxwell_layered_bg import compute_background_coefficients, maxwell_2d_lbg_pde_residual
from src.maxwell_2d_nondim import sample_nd_points
from src.utils import resolve_device, resolve_training_dtype, set_seed
from scripts.generate_reference import solve_rcwa
from scripts.train_lbg import evaluate, layered_bg_loss


def source_audit(physics, dtype):
    coeff = compute_background_coefficients(physics)
    class Zero(torch.nn.Module):
        def field_components(self, x, z):
            q = x * 0 + z * 0
            return q, q, q, q, q, q
    net = Zero()
    groups = {"air": (0.1, .5, physics.eps_air),
              "ridge": ((physics.ridge_x_min + physics.ridge_x_max)/2,
                        (physics.ridge_z_min + physics.ridge_z_max)/2, physics.eps_ridge),
              "substrate": (.1, 1.8, physics.eps_substrate)}
    result = {}
    for name, (xv, zv, eps) in groups.items():
        x, z = torch.tensor([xv], dtype=dtype), torch.tensor([zv], dtype=dtype)
        r = maxwell_2d_lbg_pde_residual(net, x, z, physics, eps, coeff)
        result[name] = {"residual": [float(q.item()) for q in r],
                        "residual_l2": float(torch.sqrt(sum(q.square() for q in r)).item())}
    return result


def save_sampling(physics, out, seed):
    samples = deterministic_region_samples(physics, 256, 128, 64, 64, seed)
    np.savez(out / "region_sampling_points.npz", **{f"{k}_x": v[:,0] for k,v in samples.items()},
             **{f"{k}_z": v[:,1] for k,v in samples.items()})
    x = np.linspace(0, physics.period, 256); z = np.linspace(0, physics.domain_height, 256); X,Z=np.meshgrid(x,z)
    eps = np.where((Z >= physics.ridge_base_z), physics.eps_substrate, physics.eps_air)
    ridge=(X>=physics.ridge_x_min)&(X<=physics.ridge_x_max)&(Z>=physics.ridge_z_min)&(Z<=physics.ridge_z_max)
    eps[ridge]=physics.eps_ridge; de=np.zeros_like(eps); de[ridge]=physics.eps_ridge-physics.eps_substrate
    fig, axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,data,title in zip(axes,[eps,de],["epsilon_r", "delta_epsilon"]):
        ax.imshow(data, extent=[0,physics.period,physics.domain_height,0], aspect="auto", cmap="viridis")
        for n,p in samples.items(): ax.scatter(p[:,0],p[:,1],s=3,label=n)
        ax.set(title=title,xlabel="x",ylabel="z")
    axes[1].legend(loc="upper left",bbox_to_anchor=(1,1)); fig.savefig(out/"region_sampling.png",dpi=150); plt.close(fig)
    return {k: int(len(v)) for k,v in samples.items()}


def lateral_mode(physics, out, m, epochs, dtype, feature_variant):
    set_seed(42 + m)
    net=SubdomainMLP_Variant(0, physics.domain_height, physics.k0*physics.n_substrate,
                             physics.period, physics.k0, feature_variant, 4,64,4).to(dtype=dtype)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3); kx=m*2*np.pi/physics.period
    kz=np.sqrt((physics.n_substrate*physics.k0)**2-kx**2)
    rng=np.random.default_rng(42+m); x=torch.tensor(rng.uniform(0,physics.period,2048),dtype=dtype); z=torch.tensor(rng.uniform(0,physics.domain_height,2048),dtype=dtype)
    target=torch.exp(1j*(kx*x-kz*z))
    for _ in range(epochs):
        opt.zero_grad(); pred=net.forward(x,z)[:,:2]; loss=((pred[:,0]-target.real)**2+(pred[:,1]-target.imag)**2).mean(); loss.backward(); opt.step()
    xx=torch.linspace(0,physics.period,128,dtype=dtype); zz=torch.linspace(0,physics.domain_height,128,dtype=dtype); X,Z=torch.meshgrid(xx,zz,indexing="xy")
    with torch.no_grad(): p=net.forward(X.ravel(),Z.ravel())[:,:2]; pred=(p[:,0]+1j*p[:,1]).reshape(X.shape).cpu().numpy()
    truth=np.exp(1j*(kx*X.cpu().numpy()-kz*Z.cpu().numpy())); err=np.linalg.norm(pred-truth)/np.linalg.norm(truth)
    phase=float(np.sqrt(np.mean(np.angle(pred/(truth+1e-12))**2))*180/np.pi)
    # Periodic DFT at z=0 uses the m order coefficient.
    modal=np.mean(pred[0]*np.exp(-1j*kx*xx.cpu().numpy()))
    return {"m":m,"feature_variant":feature_variant,"kx":float(kx),"kz":float(kz),"complex_l2":float(err),"phase_rmse_deg":phase,
            "modal_amplitude":float(abs(modal)),"modal_phase_deg":float(np.angle(modal)*180/np.pi),"final_loss":float(loss.detach())}


def compact_contrast_sweep(base, out, args, device, dtype):
    rows=[]
    for contrast in (0.,.25,.5,1.):
        eps=base.eps_substrate+contrast*(base.eps_ridge-base.eps_substrate)
        p=replace(base,n_ridge=float(np.sqrt(eps)))
        coeff=compute_background_coefficients(p); set_seed(42)
        pts=sample_nd_points(p,args.n_per_region,64,64,64,device,dtype,seed=42)
        model=Maxwell2DDD_Variant(p,args.feature_variant,4,64,4).to(device=device,dtype=dtype); opt=torch.optim.Adam(model.parameters(),lr=5e-4)
        for _ in range(args.contrast_epochs):
            opt.zero_grad(); L=layered_bg_loss(model,pts,p,coeff,1,500,500,200,100,rcwa_amps=None,w_modal=0); L['total'].backward(); opt.step()
        fields=evaluate(model,p,device,dtype,"layered_bg",coeff); x,z=fields['x'][0],fields['z'][:,0]
        _,_,rr,ri,amps=solve_rcwa(p,N_harmonics=25,Nfine=512)
        dual=compare_fields(fields['E_scat_real'],fields['E_scat_imag'],rr,ri,fields['z'],p,formulation="layered_bg",region_mask="external_only")
        modal=extract_modal_amplitudes(fields['E_real']+1j*fields['E_imag'],x,z,p,n_orders=1,formulation="layered_bg")
        i={v:j for j,v in enumerate(modal['orders'])}
        rows.append({"contrast":contrast,"n_ridge":p.n_ridge,"r0":abs(modal['r_m_complex'][i[0]]),"t0":abs(modal['t_m_complex'][i[0]]),"tm1":abs(modal['t_m_complex'][i[-1]]),"tp1":abs(modal['t_m_complex'][i[1]]),"R_total":modal['R_total'],"T_total":modal['T_total'],"pde_air":float(L['pde_air'].detach()),"pde_ridge":float(L['pde_grat'].detach()),"pde_substrate":float(L['pde_sub'].detach()),"phase_error_deg":dual['total/phase_rmse_deg'],"complex_field_error":dual['total/complex_l2']})
    with (out/'contrast_sweep.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(figsize=(7,4));
    for key in ('r0','t0','tm1','tp1'): ax.plot([r['contrast'] for r in rows],[r[key] for r in rows],marker='o',label=key)
    ax.set(xlabel='contrast',ylabel='modal amplitude',title='LBG compact contrast sweep');ax.legend();fig.savefig(out/'modal_vs_contrast.png',dpi=150);plt.close(fig)
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default='outputs/phase5_diagnostics');ap.add_argument('--contrast-epochs',type=int,default=250);ap.add_argument('--n-per-region',type=int,default=128);ap.add_argument('--lateral-epochs',type=int,default=1000);ap.add_argument('--feature-variant',default='global_k0');ap.add_argument('--skip-contrast',action='store_true');args=ap.parse_args()
    out=ROOT/args.output_dir
    if out.exists(): raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True); base=load_config(ROOT/'configs/default.yaml').physics;base.period=.8;base.ridge_width=.32
    device=resolve_device('cpu');dtype=resolve_training_dtype('float64',device)
    report={'phase_a_source':source_audit(base,dtype),'phase_b_counts':save_sampling(base,out,42),'phase_c_lateral_modes':[lateral_mode(base,out,m,args.lateral_epochs,dtype,args.feature_variant) for m in (-1,1)],'phase_d_contrast_sweep':[] if args.skip_contrast else compact_contrast_sweep(base,out,args,device,dtype),'valid':False,'invalid_reason':'Diagnostic run only; Phase 5 is not physically validated.'}
    (out/'phase5_diagnostics_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
