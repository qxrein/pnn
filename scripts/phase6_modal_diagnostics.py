#!/usr/bin/env python3
"""Phase A/B diagnostics: contrast-source spectrum and modal gradient audit."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.maxwell_2d_nondim import sample_nd_points
from src.maxwell_layered_bg import compute_background_coefficients
from src.modal_diagnostics import component_gradient_audit,projected_lbg_modal_loss,source_fourier_spectrum
from src.mode_aware_fourier import ExplicitFourierModalDD
from scripts.train_lbg import layered_bg_loss,make_lambda_0p8

ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default='outputs/phase6_modal_diagnostics')
args=ap.parse_args();out=ROOT/args.output_dir
if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
out.mkdir(parents=True)
p=make_lambda_0p8(load_config(ROOT/'configs/default.yaml').physics);coeff=compute_background_coefficients(p);dtype=torch.float64;device=torch.device('cpu')
s=source_fourier_spectrum(p,coeff);fig,ax=plt.subplots(figsize=(8,4))
for i,m in enumerate(s['orders']):ax.plot(s['z'],s['magnitude'][:,i],label=f'{m:+d}')
ax.set(yscale='log',xlabel='z',ylabel='|S_m(z)|',title='Ridge contrast-source Fourier spectrum');ax.legend(ncol=4,fontsize=7);fig.tight_layout();fig.savefig(out/'source_fourier_spectrum.png',dpi=160);plt.close(fig)
model=ExplicitFourierModalDD(p,modal_order_max=3,use_coefficient_mlp=True).to(dtype=dtype);pts=sample_nd_points(p,64,32,32,32,device,dtype,seed=42)
L=layered_bg_loss(model,pts,p,coeff,1,500,500,200,100,use_dtn=True,n_dtn_orders=8,w_modal=0)
components={'air_PDE':L['pde_air'],'grating_PDE':L['pde_grat'],'substrate_PDE':L['pde_sub'],'interface_E':L['E_int1']+L['E_int2'],'interface_H':L['H_int1']+L['H_int2'],'DtN':L['top']+L['bottom']}
audit=component_gradient_audit(model,components)
projected=projected_lbg_modal_loss(model.net_grat,torch.linspace(p.ridge_z_min+.01,p.ridge_z_max-.01,16,dtype=dtype),p,coeff,3)
report={'source_orders':s['orders'].tolist(),'source_max_by_order':{str(m):float(s['magnitude'][:,i].max()) for i,m in enumerate(s['orders'])},'gradient_audit':audit,'loss_components':{k:float(v.detach()) for k,v in components.items()},'residual_formulation_comparison':{'pointwise_grating_pde':float(L['pde_grat'].detach()),'projected_galerkin_pde':float(projected.detach()),'note':'Projected residual explicitly uses spatial epsilon(x,z)E and is not mixed into pointwise training.'}}
(out/'phase_ab_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
