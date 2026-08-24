#!/usr/bin/env python3
"""Phase E: carry explicit-modal checkpoint across contrast-source stages."""
from __future__ import annotations
import argparse,json,sys,torch
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.reference_data import load_reference_npz,normalize_reference_orientation
from src.reference_validation import validate_reference
from scripts.run_phase6_explicit_modal import run_seed
from scripts.train_lbg import make_lambda_0p8
ap=argparse.ArgumentParser();ap.add_argument('--epochs-per-stage',type=int,default=500);args0=ap.parse_args();root=ROOT/'outputs/source_continuation'
if root.exists():raise FileExistsError(f'Refusing to overwrite {root}')
root.mkdir(parents=True);p=make_lambda_0p8(load_config(ROOT/'configs/default.yaml').physics);rp=ROOT/'outputs/reference_lambda_0p8_geometry_consistent_20260824.npz';rm=validate_reference(rp,p);ref=normalize_reference_orientation(load_reference_npz(rp));args=SimpleNamespace(epochs=args0.epochs_per_stage,log_every=250,n_per_region=64,n_interface=32,n_bc=32,modal_order_max=1,n_dtn_orders=8,lr=5e-4)
state=None;rows=[]
for alpha in (.1,.25,.5,.75,1.):
 report,_,_=run_seed(42,args,p,ref,rp,{'ablation':'source_continuation','alpha':alpha,'modal_loss_weight':0.,'reference':rm},root/f'alpha_{str(alpha).replace(".","p")}',initial_state=state,source_alpha=alpha)
 state=torch.load(root/f'alpha_{str(alpha).replace(".","p")}/seed42/best_checkpoint.pt',map_location='cpu',weights_only=True)['state_dict'];c=report['modal_comparison'];rows.append({'alpha':alpha,'pde_grating':report['last_log']['pde_grating'],'t-1':c['m=-1']['t_pinn_abs'],'t+1':c['m=1']['t_pinn_abs'],'R_plus_T':c['summary']['pinn_energy_check'],'total_complex_l2':report['final_metrics']['total/complex_l2']})
(root/'source_continuation.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
