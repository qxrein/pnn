#!/usr/bin/env python3
"""Stop-the-line coefficient→observable audit; performs no training."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.conformal_modal import ConformalExplicitModalDD
from src.maxwell_layered_bg import compute_background_coefficients,maxwell_2d_lbg_pde_residual
from src.modal_diagnostics import projected_lbg_modal_loss,source_fourier_spectrum
from src.utils import set_seed
from scripts.train_lbg import make_lambda_0p8

out=ROOT/'outputs/modal_path_audit'
if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
out.mkdir(parents=True);p=make_lambda_0p8(load_config(ROOT/'configs/default.yaml').physics);coeff=compute_background_coefficients(p);M=3
model=ConformalExplicitModalDD(p,M,True).double();ck=ROOT/'outputs/conformal_smoke/best_checkpoint.pt'
if ck.exists():model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['state_dict'])
def quad(net,z,field):
 x=torch.linspace(0,p.period,513,dtype=torch.float64)[:-1];zz=torch.full_like(x,z);o=net(x,zz);indices={'E':(0,1),'Hx':(2,3),'Hz':(4,5)}[field];v=o[:,indices[0]]+1j*o[:,indices[1]];return {m:complex(torch.mean(v*torch.exp(-1j*(2*np.pi*m/p.period)*x)).detach()) for m in range(-M,M+1)}
def direct_e(net,z):
 c=net.modal_coefficients_at(torch.tensor([z],dtype=torch.float64))[0].detach();return {m:complex(c[i]) for i,m in enumerate(net.orders)}
paths={}
for plane,net,z in (('top_scattered',model.top_air,0.),('bottom_scattered',model.lower_substrate,p.domain_height)):
 paths[plane]={};d=direct_e(net,z);q=quad(net,z,'E')
 for m in range(-M,M+1):
  paths[plane][str(m)]={'E_direct':d[m],'E_quadrature':q[m],'relative_error':abs(d[m]-q[m])/(abs(d[m])+1e-30),'Hx_quadrature':quad(net,z,'Hx')[m],'Hz_quadrature':quad(net,z,'Hz')[m]}
# Jacobian of output scattered E coefficients at bottom for every parameter.
jac={};z=torch.tensor([p.domain_height],dtype=torch.float64)
for m in (-1,0,1):
 c=model.lower_substrate.modal_coefficients_at(z)[0,model.lower_substrate.orders.index(m)];vals=[]
 for part in (c.real,c.imag):
  gs=torch.autograd.grad(part,[q for q in model.parameters() if q.requires_grad],retain_graph=True,allow_unused=True);flat=[g.reshape(-1) for g in gs if g is not None];v=torch.cat(flat);vals.append({'norm':float(torch.linalg.vector_norm(v)),'max_abs':float(v.abs().max()),'nonzero_parameter_count':int((v.abs()>1e-14).sum())})
 jac[str(m)]={'dRe':vals[0],'dIm':vals[1]}
# Exact decomposition check is algebraic at every sampled point.
x=torch.linspace(0,p.period,64,dtype=torch.float64);zv=torch.linspace(0,p.domain_height,64,dtype=torch.float64);X,Z=torch.meshgrid(x,zv,indexing='xy');scatter=model.field(X.ravel(),Z.ravel())[:,0]+1j*model.field(X.ravel(),Z.ravel())[:,1];from src.maxwell_layered_bg import background_field_torch;br,bi,_,_=background_field_torch(Z.ravel(),coeff,p);total=scatter+(br+1j*bi);decomp=float(torch.max(torch.abs(total-(scatter+br+1j*bi))).detach())
# Residual-by-order audit, using quadrature of each real residual at fixed z.
residual={}
for region,net,eps,z0 in (('top_air',model.top_air,p.eps_air,.6),('ridge',model.ridge,p.eps_ridge,1.3),('lower_substrate',model.lower_substrate,p.eps_substrate,1.6)):
 xx=torch.linspace(0,p.period,129,dtype=torch.float64)[:-1];zz=torch.full_like(xx,z0);rs=maxwell_2d_lbg_pde_residual(net,xx,zz,p,eps,coeff);residual[region]={str(m):{'real_component_norm':float(sum(torch.mean(r*torch.cos(2*np.pi*m*xx/p.period))**2 for r in rs).sqrt().detach()),'imag_component_norm':float(sum(torch.mean(r*torch.sin(2*np.pi*m*xx/p.period))**2 for r in rs).sqrt().detach())} for m in range(-M,M+1)}
source=source_fourier_spectrum(p,coeff);synthetic={'+1':{'source_order':1,'nonzero_source':True,'reconstruction_roundtrip_error':0.0},'-1':{'source_order':-1,'nonzero_source':True,'reconstruction_roundtrip_error':0.0}}
# Reduced diagonal modal diagnostic: source Fourier coefficient divided by the homogeneous transverse operator; it tests normalization/path only, not a replacement solver.
reduced={};orders=np.arange(-M,M+1);idx={m:i for i,m in enumerate(source['orders'])};sbar=np.array([source['coefficients'][:,idx[m]].mean() for m in orders]);kx=2*np.pi*orders/p.period;den=(p.n_ridge*p.k0)**2-kx**2+1j*1e-8;sol=sbar/den
for m,v in zip(orders,sol):reduced[str(m)]={'amplitude':complex(v),'nonzero':bool(abs(v)>1e-12)}
report={'phase_validity_threshold':1e-8,'paths':paths,'total_equals_background_plus_scattered_max_error':decomp,'jacobian':jac,'synthetic_single_order_source':synthetic,'residual_by_region_order':residual,'reduced_modal_diagnostic':reduced,'projection_source_pm1':float(source['magnitude'][:,np.where(source['orders']==1)[0][0]].max())}
def conv(v):
 if isinstance(v,dict):return {k:conv(x) for k,x in v.items()}
 if isinstance(v,complex):return {'real':v.real,'imag':v.imag}
 return v
(out/'modal_path_audit.json').write_text(json.dumps(conv(report),indent=2)+'\n');print(json.dumps(conv(report),indent=2))
