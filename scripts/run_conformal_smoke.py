#!/usr/bin/env python3
"""500-epoch geometry-conforming LBG/DtN smoke; no modal data loss."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.conformal_modal import REGIONS,ConformalExplicitModalDD,sample_conformal_points,trace_loss
from src.field_comparison import compare_fields,compare_modal_with_rcwa,extract_modal_amplitudes
from src.geometry import epsilon_r_grid
from src.maxwell_layered_bg import background_field_np,compute_background_coefficients,lbg_bottom_bc,lbg_top_bc,maxwell_2d_lbg_pde_residual
from src.reference_data import interpolate_reference_to_grid,load_reference_npz,normalize_reference_orientation
from src.reference_validation import validate_reference
from src.utils import set_seed
from scripts.train_lbg import make_lambda_0p8
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=500);args=ap.parse_args();out=ROOT/'outputs/conformal_smoke'
 if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
 out.mkdir(parents=True);p=make_lambda_0p8(load_config(ROOT/'configs/default.yaml').physics);rp=ROOT/'outputs/reference_lambda_0p8_geometry_consistent_20260824.npz';meta=validate_reference(rp,p);ref=normalize_reference_orientation(load_reference_npz(rp));device=torch.device('cpu');dtype=torch.float64;set_seed(42);model=ConformalExplicitModalDD(p,3).double();pts=sample_conformal_points(p,64,32,32,device,dtype,42);np.savez(out/'region_samples.npz',**{k:v.detach().numpy() if isinstance(v,torch.Tensor) else np.array(v) for k,v in pts.items()});coeff=compute_background_coefficients(p);opt=torch.optim.Adam(model.parameters(),lr=5e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs);hist=[];best=1e99
 eps={'top_air':p.eps_air,'left_substrate':p.eps_substrate,'ridge':p.eps_ridge,'right_substrate':p.eps_substrate,'lower_substrate':p.eps_substrate}
 def loss():
  lp={};
  for name in REGIONS:
   x,z=pts[name];r=maxwell_2d_lbg_pde_residual(getattr(model,name),x,z,p,eps[name],coeff);lp[name]=sum(torch.mean(q*q) for q in r)/6
  zt=torch.full_like(pts['x_ridge'],p.ridge_z_min);zb=torch.full_like(pts['x_ridge'],p.ridge_z_max);lz=pts['z_left_edge'];rz=pts['z_right_edge']
  interfaces={'ridge_top':trace_loss(model.top_air,model.ridge,pts['x_ridge'],zt,'hx'),'ridge_bottom':trace_loss(model.ridge,model.lower_substrate,pts['x_ridge'],zb,'hx'),'ridge_left':trace_loss(model.left_substrate,model.ridge,torch.full_like(lz,p.ridge_x_min),lz,'hz'),'ridge_right':trace_loss(model.ridge,model.right_substrate,torch.full_like(rz,p.ridge_x_max),rz,'hz'),'air_sub_left':trace_loss(model.top_air,model.left_substrate,pts['x_left'],torch.full_like(pts['x_left'],p.ridge_z_min),'hx'),'air_sub_right':trace_loss(model.top_air,model.right_substrate,pts['x_right'],torch.full_like(pts['x_right'],p.ridge_z_min),'hx'),'sub_seam_left':trace_loss(model.left_substrate,model.lower_substrate,pts['x_left'],torch.full_like(pts['x_left'],p.ridge_z_max),'hx'),'sub_seam_right':trace_loss(model.right_substrate,model.lower_substrate,pts['x_right'],torch.full_like(pts['x_right'],p.ridge_z_max),'hx')}
  top=lbg_top_bc(model.top_air,pts['x_top'],p,True,8);bot=lbg_bottom_bc(model.lower_substrate,pts['x_bottom'],p,coeff,True,8);total=sum(lp.values())+500*sum(interfaces.values())+200*top+100*bot;return lp,interfaces,top,bot,total
 for ep in range(1,args.epochs+1):
  opt.zero_grad();lp,inter,top,bot,total=loss();total.backward();opt.step();sch.step()
  if float(total.detach())<best:best=float(total.detach());bestep=ep;state={k:v.detach().clone() for k,v in model.state_dict().items()}
  if ep%250==0:
   hist.append({'epoch':ep,'total':float(total.detach()),**{f'pde_{k}':float(v.detach()) for k,v in lp.items()},**{f'interface_{k}':float(v.detach()) for k,v in inter.items()},'top_DtN':float(top.detach()),'bottom_DtN':float(bot.detach())});print(ep,hist[-1]['total'])
 torch.save({'state_dict':state,'best_epoch':bestep,'best_loss':best,'metadata':meta},out/'best_checkpoint.pt');
 xg,zg,_=epsilon_r_grid(p,device,dtype);xf=xg.ravel();zf=zg.ravel();pred=torch.zeros((len(xf),2),dtype=dtype)
 with torch.enable_grad():
  masks=model.assert_partition(xf,zf)
  for name in REGIONS:
   if masks[name].any():pred[masks[name]]=getattr(model,name)(xf[masks[name]],zf[masks[name]])[:,:2]
 er=pred[:,0].detach().numpy().reshape(xg.shape);ei=pred[:,1].detach().numpy().reshape(xg.shape);br,bi,_,_=background_field_np(zg.detach().numpy().ravel(),coeff);rr,ri,_=interpolate_reference_to_grid(ref,xg[0].numpy(),zg[:,0].numpy());dual=compare_fields(er,ei,rr,ri,zg.detach().numpy(),p,'layered_bg','external_only');totalfield=dual['pinn_E_total_r']+1j*dual['pinn_E_total_i'];modal=extract_modal_amplitudes(totalfield,xg[0].numpy(),zg[:,0].numpy(),p,n_orders=1,formulation='layered_bg',field_representation='total');cmp=compare_modal_with_rcwa(modal,str(rp),75,p)
 with (out/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=hist[0].keys());w.writeheader();w.writerows(hist)
 report={'best_epoch':bestep,'best_loss':best,'metadata':meta,'history':hist,'dual':{k:v for k,v in dual.items() if not isinstance(v,np.ndarray)},'modal':cmp,'phase5_passed':False};(out/'report.json').write_text(json.dumps(report,indent=2,default=str));print(json.dumps({'dual':report['dual'],'modal':cmp['summary']},indent=2,default=str))
if __name__=='__main__':main()
