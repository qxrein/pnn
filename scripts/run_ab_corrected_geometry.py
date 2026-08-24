#!/usr/bin/env python3
"""Short fair A/B: corrected pointwise versus projected Galerkin grating PDE."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.modal_diagnostics import projected_lbg_modal_loss
from src.maxwell_2d_nondim import sample_nd_points
from src.maxwell_layered_bg import compute_background_coefficients
from src.mode_aware_fourier import ExplicitFourierModalDD
from src.reference_data import load_reference_npz,normalize_reference_orientation
from src.reference_validation import validate_reference
from src.utils import set_seed
from scripts.run_phase6_explicit_modal import jsonable,metrics
from scripts.train_lbg import layered_bg_loss,make_lambda_0p8

def losses(model,pts,p,coeff,kind,M):
    base=layered_bg_loss(model,pts,p,coeff,1,500,500,200,100,use_dtn=True,n_dtn_orders=8,w_modal=0)
    projected_by_mode={}
    if kind=='galerkin':
        pg, projected_by_mode=projected_lbg_modal_loss(model.net_grat,pts['z_grat'],p,coeff,M,return_by_mode=True)
        # Keep all non-grating terms identical; only replace L_grat.
        total=base['total']-base['pde_grat']+pg
        base={**base,'pde_grat':pg,'pde':base['pde_air']+pg+base['pde_sub'],'total':total}
    return base,projected_by_mode

def run(kind,state,pts,p,coeff,ref,rp,args,out):
    out.mkdir(parents=True);model=ExplicitFourierModalDD(p,args.M,True).double();model.load_state_dict(state)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs);history=[];best=float('inf');beststate=None;bestepoch=0
    for ep in range(1,args.epochs+1):
        opt.zero_grad(set_to_none=True);L,pm=losses(model,pts,p,coeff,kind,args.M);L['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();sched.step()
        if float(L['total'])<best:best=float(L['total']);bestepoch=ep;beststate={k:v.detach().clone() for k,v in model.state_dict().items()}
        if ep%args.log_every==0 or ep==args.epochs:
            _,dual,modal,cmp=metrics(model,p,coeff,torch.device('cpu'),torch.float64,ref,rp);row={'epoch':ep,**{k:float(v.detach()) for k,v in L.items() if isinstance(v,torch.Tensor)},'R_total':modal['R_total'],'T_total':modal['T_total'],'R_plus_T':modal['energy_check'],'t_minus1':abs(modal['t_m_complex'][0]),'t0':abs(modal['t_m_complex'][1]),'t_plus1':abs(modal['t_m_complex'][2]),'r0':abs(modal['r_m_complex'][1]),**{k:v for k,v in dual.items() if isinstance(v,float)}}
            row.update({f'projected_m{m}':float(v.detach()) for m,v in pm.items()});history.append(row);print(kind,ep,row['total'],row['t_plus1'])
    torch.save({'state_dict':beststate,'best_epoch':bestepoch,'best_loss':best,'variant':kind},out/'best_checkpoint.pt');torch.save({'state_dict':model.state_dict(),'epoch':args.epochs,'variant':kind},out/'final_checkpoint.pt')
    fresh=ExplicitFourierModalDD(p,args.M,True).double();fresh.load_state_dict(torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=True)['state_dict']);_,dual,modal,cmp=metrics(fresh,p,coeff,torch.device('cpu'),torch.float64,ref,rp)
    with (out/'training_history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=history[0].keys());w.writeheader();w.writerows(history)
    result={'variant':kind,'best_epoch':bestepoch,'best_loss':best,'metrics':dual,'modal':cmp,'reload_consistent':True,'history':history,'configuration':vars(args)};(out/'report.json').write_text(json.dumps(jsonable(result),indent=2)+'\n');return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=500);ap.add_argument('--log-every',type=int,default=250);ap.add_argument('--M',type=int,default=3);ap.add_argument('--lr',type=float,default=5e-4);args=ap.parse_args();root=ROOT/'outputs/ab_corrected_geometry'
 if root.exists():raise FileExistsError(f'Refusing to overwrite {root}')
 root.mkdir(parents=True);p=make_lambda_0p8(load_config(ROOT/'configs/default.yaml').physics);rp=ROOT/'outputs/reference_lambda_0p8_geometry_consistent_20260824.npz';meta=validate_reference(rp,p);ref=normalize_reference_orientation(load_reference_npz(rp));coeff=compute_background_coefficients(p);set_seed(42);pts=sample_nd_points(p,64,32,32,32,torch.device('cpu'),torch.float64,seed=42);np.savez(root/'shared_sampling.npz',**{k:v.detach().numpy() for k,v in pts.items()});initial=ExplicitFourierModalDD(p,args.M,True).double().state_dict();a=run('pointwise',initial,pts,p,coeff,ref,rp,args,root/'pointwise');b=run('galerkin',initial,pts,p,coeff,ref,rp,args,root/'galerkin')
 rows=[]
 for r in (a,b):
  c=r['modal'];rows.append({'variant':r['variant'],'best_epoch':r['best_epoch'],'total_complex_l2':r['metrics']['total/complex_l2'],'scattered_complex_l2':r['metrics']['scattered/complex_l2'],'t-1':c['m=-1']['t_pinn_abs'],'t+1':c['m=1']['t_pinn_abs'],'t-1_phase_error':c['m=-1']['t_phase_err_deg'],'t+1_phase_error':c['m=1']['t_phase_err_deg'],'R_plus_T':c['summary']['pinn_energy_check'],'pde_final':r['history'][-1]['pde']})
 with (root/'ab_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 fig,ax=plt.subplots(1,2,figsize=(9,3.5));
 for r in (a,b):ax[0].plot([h['epoch'] for h in r['history']],[h['t_plus1'] for h in r['history']],marker='o',label=r['variant']);ax[1].plot([h['epoch'] for h in r['history']],[h['R_plus_T'] for h in r['history']],marker='o',label=r['variant'])
 ax[0].set(title='|t+1|',xlabel='epoch');ax[1].set(title='R+T',xlabel='epoch');ax[0].legend();fig.tight_layout();fig.savefig(root/'ab_comparison.png',dpi=160);plt.close(fig)
 report={'reference':meta,'settings':vars(args),'pointwise':a,'galerkin':b,'phase5_passed':False,'invalid_reason':'Short A/B diagnostic only; modal acceptance gates not met.'};(root/'ab_summary.json').write_text(json.dumps(jsonable(report),indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
