#!/usr/bin/env python3
"""Fixed-physical-scale, PDE-only loss-conditioning screen (D/E)."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.field_comparison import compare_modal_with_rcwa,extract_modal_amplitudes
from src.geometry import epsilon_r
from src.maxwell_2d_nondim import maxwell_2d_nd_interface_loss,sample_nd_points
from src.maxwell_layered_bg import background_field_torch,compute_background_coefficients,lbg_bottom_bc,lbg_top_bc,lbg_vertical_interface_loss,maxwell_2d_lbg_pde_residual
from src.mode_aware_fourier import ExplicitFourierModalDD
from src.reference_data import load_reference_npz,normalize_reference_orientation
from src.reference_validation import validate_reference
from src.utils import set_seed
from scripts.run_phase6_explicit_modal import metrics
from scripts.train_lbg import make_lambda_0p8

def components(model,pts,p,coeff):
 out={}
 for name,net,eps,key in [('air',model.net_air,p.eps_air,'air'),('ridge',model.net_grat,None,'grat'),('substrate',model.net_sub,p.eps_substrate,'sub')]:
  x,z=pts['x_'+key],pts['z_'+key];er=epsilon_r(x,z,p) if name=='ridge' else eps
  out.update({f'pde_{name}_{i}':torch.mean(r*r) for i,r in enumerate(maxwell_2d_lbg_pde_residual(net,x,z,p,er,coeff))})
 e1,h1=maxwell_2d_nd_interface_loss(model.net_air,model.net_grat,p.ridge_z_min,pts['x_int1']);e2,h2=maxwell_2d_nd_interface_loss(model.net_grat,model.net_sub,p.ridge_z_max,pts['x_int2'])
 evl,hvl=lbg_vertical_interface_loss(model.net_grat,p.ridge_x_min,pts['z_vleft']);evr,hvr=lbg_vertical_interface_loss(model.net_grat,p.ridge_x_max,pts['z_vright'])
 out.update(horizontal_E=e1+e2,horizontal_Hx=h1+h2,vertical_E=evl+evr,vertical_Hz=hvl+hvr,top_DtN=lbg_top_bc(model.net_air,pts['x_top'],p,True,8),bottom_DtN=lbg_bottom_bc(model.net_sub,pts['x_bot'],p,coeff,True,8))
 return out
def physical_scales(pts,p,coeff):
 zs=torch.cat([v for k,v in pts.items() if k.startswith('z_')]);ebr,ebi,hbr,hbi=background_field_torch(zs,coeff,p)
 E=max(float(torch.max(torch.hypot(ebr,ebi))),1e-30);H=max(float(torch.max(torch.hypot(hbr,hbi))),1e-30);floor=1e-8*max(E,H);first=max(E,H,floor);s={}
 for name,eps,key in [('air',p.eps_air,'air'),('ridge',p.eps_ridge,'grat'),('substrate',p.eps_substrate,'sub')]:
  source=0.
  if name=='ridge':
   x,z=pts['x_'+key],pts['z_'+key];br,bi,_,_=background_field_torch(z,coeff,p);de=epsilon_r(x,z,p)-p.eps_substrate;source=max(float(torch.sqrt(torch.mean((de*br)**2))),float(torch.sqrt(torch.mean((de*bi)**2))))
  curl=max(eps*E,source,floor)
  for i in range(4):s[f'pde_{name}_{i}']=first
  s[f'pde_{name}_4']=curl;s[f'pde_{name}_5']=curl
 for k in ('horizontal_E','vertical_E'):s[k]=max(E,floor)
 for k in ('horizontal_Hx','vertical_Hz'):s[k]=max(H,floor)
 s['top_DtN']=max(H+p.n_air*E,floor);s['bottom_DtN']=max(H+p.n_substrate*E,floor)
 return s,{'E_star':E,'H_star':H,'physical_scale_floor':floor,'operator':'nondimensional first-order TE Maxwell'}
def grad_norms(losses,params):
 d={}
 for k,v in losses.items():
  g=torch.autograd.grad(v,params,retain_graph=True,allow_unused=True);d[k]=float(torch.linalg.vector_norm(torch.cat([q.reshape(-1) for q in g if q is not None])).detach())
 return d
def modal3(fields,p,rp):
 x,z=fields['x'][0],fields['z'][:,0];m=extract_modal_amplitudes(fields['E_real']+1j*fields['E_imag'],x,z,p,n_orders=3,formulation='layered_bg',field_representation='total');return m,compare_modal_with_rcwa(m,str(rp),int((len(np.load(rp)['c_refl'])-1)//2),physics=p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=1000);ap.add_argument('--log-every',type=int,default=100);ap.add_argument('--output-root',default='outputs/phase5_physical_conditioning');a=ap.parse_args();root=ROOT/a.output_root
 if root.exists():raise FileExistsError(f'Refusing to overwrite {root}')
 root.mkdir(parents=True);p=make_lambda_0p8(load_config(ROOT/'configs/default.yaml').physics);rp=ROOT/'outputs/reference_lambda_0p8_geometry_consistent_20260824.npz';meta=validate_reference(rp,p);ref=normalize_reference_orientation(load_reference_npz(rp));device=torch.device('cpu');dtype=torch.float64;set_seed(42);pts=sample_nd_points(p,64,32,32,32,device,dtype,42);np.savez(root/'shared_points.npz',**{k:v.numpy() for k,v in pts.items()});initial=ExplicitFourierModalDD(p,3,True).double().state_dict();coeff=compute_background_coefficients(p);scales,scale_meta=physical_scales(pts,p,coeff);reports={}
 for variant in ('d_physical_scaled','e_physical_balanced'):
  out=root/f'variant_{variant[0]}';out.mkdir();model=ExplicitFourierModalDD(p,3,True).double();model.load_state_dict(initial);opt=torch.optim.Adam(model.parameters(),lr=5e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs);params=[q for q in model.parameters() if q.requires_grad];weights={k:1. for k in scales};hist=[];best_loss=float('inf');best_score=float('inf')
  for epoch in range(1,a.epochs+1):
   opt.zero_grad(set_to_none=True);raw=components(model,pts,p,coeff);loss={k:raw[k]/scales[k]**2 for k in raw};total=sum(weights[k]*loss[k] for k in loss);total.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step();sch.step()
   if float(total.detach())<best_loss:best_loss=float(total.detach());torch.save({'state_dict':model.state_dict(),'epoch':epoch,'loss':best_loss},out/'best_loss.pt')
   if epoch%a.log_every==0 or epoch==a.epochs:
    raw=components(model,pts,p,coeff);loss={k:raw[k]/scales[k]**2 for k in raw};gn=grad_norms(loss,params)
    if variant.endswith('balanced'):
     inv={k:1/max(v,1e-30) for k,v in gn.items()};med=np.median(list(inv.values()));target={k:float(np.clip(v/med,.1,10.)) for k,v in inv.items()};weights={k:.8*weights[k]+.2*target[k] for k in weights}
    fields,dual,modal,cmp=metrics(model,p,coeff,device,dtype,ref,rp);m3,c3=modal3(fields,p,rp);score=dual['scattered/complex_l2']+sum(cmp[f'm={m}']['t_abs_err'] for m in(-1,1))+abs(modal['energy_check']-1)
    if score<best_score:best_score=score;torch.save({'state_dict':model.state_dict(),'epoch':epoch,'score':score},out/'best_score.pt')
    row={'epoch':epoch,'total':float(total.detach()),'score':score,'R_plus_T':modal['energy_check'],'t_minus1':abs(modal['t_m_complex'][0]),'t_plus1':abs(modal['t_m_complex'][2]),'total_complex_l2':dual['total/complex_l2'],'scattered_complex_l2':dual['scattered/complex_l2'],'t_minus1_complex_error':c3['m=-1']['t_abs_err'],'t_plus1_complex_error':c3['m=1']['t_abs_err']}
    row.update({f'loss_{k}':float(v.detach()) for k,v in raw.items()});row.update({f'scale_{k}':scales[k] for k in scales});row.update({f'grad_{k}':gn[k] for k in gn});row.update({f'weighted_grad_{k}':weights[k]*gn[k] for k in gn});row.update({f'weight_{k}':weights[k] for k in weights})
    for i,m in enumerate(range(-3,4)):
     t=m3['t_m_complex'][i];d=c3[f'm={m}'];r=d['t_rcwa_abs']*np.exp(1j*np.deg2rad(d['t_rcwa_phase_deg']));row.update({f't_{m}_real':float(t.real),f't_{m}_imag':float(t.imag),f't_{m}_ref_real':float(r.real),f't_{m}_ref_imag':float(r.imag)})
    hist.append(row);print(variant,epoch,row['t_minus1'],row['t_plus1'],row['scattered_complex_l2'])
  torch.save({'state_dict':model.state_dict(),'epoch':a.epochs},out/'final.pt')
  with (out/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=hist[0]);w.writeheader();w.writerows(hist)
  reports[variant]={'physical_scales':scales,'scale_metadata':scale_meta,'best_total_loss':best_loss,'best_diagnostic_score':best_score,'history':hist,'modal_loss_weight':0,'reference':meta};(out/'report.json').write_text(json.dumps(reports[variant],indent=2)+'\n')
 rows=[{'variant':k,**{q:r['history'][-1][q] for q in ('t_minus1','t_plus1','t_minus1_complex_error','t_plus1_complex_error','total_complex_l2','scattered_complex_l2','R_plus_T','score')}} for k,r in reports.items()]
 with (root/'physical_conditioning_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 fig,ax=plt.subplots();[ax.plot([h['epoch'] for h in r['history']],[h['t_plus1'] for h in r['history']],label=k) for k,r in reports.items()];ax.axhline(.049410786,color='k',ls='--');ax.legend();ax.set(xlabel='epoch',ylabel='|t+1|');fig.tight_layout();fig.savefig(root/'modal_conditioning.png',dpi=160)
 (root/'physical_conditioning_summary.json').write_text(json.dumps({'rows':rows,'phase5_passed':False,'next_step':'No 10000-epoch run unless both ±1 amplitudes improve.'},indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
