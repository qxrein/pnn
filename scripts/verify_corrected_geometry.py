#!/usr/bin/env python3
"""Short corrected-geometry gates; no optimisation or RCWA training anchor."""
from __future__ import annotations
import json,sys
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.geometry import epsilon_r
from src.maxwell_layered_bg import compute_background_coefficients,maxwell_2d_lbg_pde_residual
from src.modal_diagnostics import binary_ridge_mask_fourier,source_fourier_spectrum
from src.mode_aware_fourier import ExplicitFourierModalDD
from scripts.train_lbg import make_lambda_0p8
out=ROOT/'outputs/corrected_geometry_gates'
if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
out.mkdir(parents=True);p=make_lambda_0p8(load_config(ROOT/'configs/default.yaml').physics);coeff=compute_background_coefficients(p)
x=np.linspace(0,p.period,256,endpoint=False);z=np.linspace(0,p.domain_height,256);X,Z=np.meshgrid(x,z,indexing='xy');xt=torch.tensor(X.ravel(),dtype=torch.float64);zt=torch.tensor(Z.ravel(),dtype=torch.float64);eps=epsilon_r(xt,zt,p).reshape(X.shape).numpy();bg=np.where(Z>=p.ridge_base_z,p.eps_substrate,p.eps_air);delta=eps-bg
fig,ax=plt.subplots(1,3,figsize=(12,3.3),constrained_layout=True)
for a,v,t in zip(ax,(eps,bg,delta),(r'$\epsilon_r$',r'$\epsilon_{bg}$',r'$\Delta\epsilon$')):
 im=a.imshow(v,extent=[0,p.period,p.domain_height,0],aspect='auto',cmap='RdBu_r');a.set(title=t,xlabel='x',ylabel='z');fig.colorbar(im,ax=a)
fig.savefig(out/'geometry_eps_bg_delta.png',dpi=160);plt.close(fig)
model=ExplicitFourierModalDD(p,1).double();zz=torch.tensor([.5,1.3,1.6],dtype=torch.float64);xx=torch.tensor([.1,.5,.1],dtype=torch.float64);res=maxwell_2d_lbg_pde_residual(model.net_grat,xx,zz,p,epsilon_r(xx,zz,p),coeff);resmag=np.array([float(sum(r[i].square() for r in res).sqrt().detach()) for i in range(3)])
s=source_fourier_spectrum(p,coeff);orders=s['orders'];fill=p.ridge_width/p.period;analytic=binary_ridge_mask_fourier(orders,fill);xq=np.linspace(0,p.period,4096,endpoint=False);mask=((xq>=p.ridge_x_min)&(xq<=p.ridge_x_max)).astype(float);numeric=np.array([np.mean(mask*np.exp(-2j*np.pi*m*xq/p.period)) for m in orders])
report={'ridge_area':p.ridge_width*p.ridge_height,'grid_pde_points_inside_ridge':int(((X>=p.ridge_x_min)&(X<=p.ridge_x_max)&(Z>=p.ridge_z_min)&(Z<=p.ridge_z_max)).sum()),'grid_pde_points_total':int(X.size),'zero_field_residual_magnitude':{'air':resmag[0],'ridge':resmag[1],'substrate':resmag[2]},'source_pm1_max':float(s['magnitude'][:,np.where(orders==1)[0][0]].max()),'binary_mask_fft_vs_analytic_max_error':float(np.max(abs(numeric-analytic))),'zero_contrast_exact_modal_field':'covered by test_zero_contrast_explicit_modal_field_stays_zero','weak_ridge_continuity':'covered by test_shallow_ridge_rcwa_orders_grow_continuously_from_zero_contrast'}
(out/'corrected_geometry_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
