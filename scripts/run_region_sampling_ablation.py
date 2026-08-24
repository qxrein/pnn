#!/usr/bin/env python3
"""Phase D: deterministic ridge/interface/corner sampling ablation."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.diagnostic_sampling import deterministic_region_samples
from src.maxwell_2d_nondim import sample_nd_points
from src.reference_data import load_reference_npz,normalize_reference_orientation
from src.reference_validation import validate_reference
from scripts.run_phase6_explicit_modal import run_seed
from scripts.train_lbg import make_lambda_0p8

ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=500);args0=ap.parse_args();root=ROOT/'outputs/region_sampling_ablation'
if root.exists():raise FileExistsError(f'Refusing to overwrite {root}')
root.mkdir(parents=True);p=make_lambda_0p8(load_config(ROOT/'configs/default.yaml').physics);rp=ROOT/'outputs/reference_lambda_0p8_geometry_consistent_20260824.npz';rm=validate_reference(rp,p);ref=normalize_reference_orientation(load_reference_npz(rp));args=SimpleNamespace(epochs=args0.epochs,log_every=250,n_per_region=64,n_interface=32,n_bc=32,modal_order_max=1,n_dtn_orders=8,lr=5e-4)
def builder(kind):
 def make(physics,args,device,dtype,seed):
  pts=sample_nd_points(physics,args.n_per_region,args.n_interface,args.n_bc,args.n_bc,device,dtype,seed=seed)
  if kind!='current':
   d=deterministic_region_samples(physics,256 if kind in ('ridge','ridge_interface_corner') else 64,128,128,64,seed)
   ridge=d['ridge'];pts['x_grat']=torch.cat((pts['x_grat'],torch.as_tensor(ridge[:,0],device=device,dtype=dtype)));pts['z_grat']=torch.cat((pts['z_grat'],torch.as_tensor(ridge[:,1],device=device,dtype=dtype)))
   if kind=='ridge_interface_corner':
    # Interfaces are enforced only through interface losses; corner patches
    # are saved, but never inserted into the discontinuous strong-form PDE.
    pts['x_int1']=torch.cat((pts['x_int1'],torch.as_tensor(d['horizontal_top'][:,0],device=device,dtype=dtype)))
    pts['x_int2']=torch.cat((pts['x_int2'],torch.as_tensor(d['horizontal_bottom'][:,0],device=device,dtype=dtype)))
    pts['corner_patches']=torch.as_tensor(np.concatenate([d[k] for k in d if k.startswith('corner_')]),device=device,dtype=dtype)
  return pts
 return make
rows=[]
for kind in ('current','ridge','ridge_interface_corner'):
 report,_,_=run_seed(42,args,p,ref,rp,{'ablation':'sampling','kind':kind,'modal_loss_weight':0.,'reference':rm},root/kind,point_builder=builder(kind));c=report['modal_comparison'];last=report['last_log'];rows.append({'sampling':kind,'t-1':c['m=-1']['t_pinn_abs'],'t+1':c['m=1']['t_pinn_abs'],'pde_grating':last['pde_grating'],'pde_substrate':last['pde_substrate'],'total_complex_l2':report['final_metrics']['total/complex_l2'],'R_plus_T':c['summary']['pinn_energy_check']})
(root/'region_sampling_ablation.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
