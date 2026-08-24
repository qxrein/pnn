#!/usr/bin/env python3
"""Phase C: fixed-seed internal modal-order ablation (PDE/DtN only)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.reference_data import load_reference_npz,normalize_reference_orientation
from src.reference_validation import validate_reference
from scripts.run_phase6_explicit_modal import run_seed
from scripts.train_lbg import make_lambda_0p8

ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=500);ap.add_argument('--seed',type=int,default=42);args0=ap.parse_args()
root=ROOT/'outputs/modal_order_ablation'
if root.exists():raise FileExistsError(f'Refusing to overwrite {root}')
root.mkdir(parents=True);p=make_lambda_0p8(load_config(ROOT/'configs/default.yaml').physics);rp=ROOT/'outputs/reference_lambda_0p8_geometry_consistent_20260824.npz';rm=validate_reference(rp,p);ref=normalize_reference_orientation(load_reference_npz(rp)); rows=[]
for M in (1,3,5,7,11):
    args=SimpleNamespace(epochs=args0.epochs,log_every=250,n_per_region=64,n_interface=32,n_bc=32,modal_order_max=M,n_dtn_orders=8,lr=5e-4)
    meta={'ablation':'modal_order','modal_loss_weight':0.,'reference':rm,'formulation':'layered_bg','boundary':'DtN','seed':args0.seed,'M':M}
    report,_,_=run_seed(args0.seed,args,p,ref,rp,meta,root/f'M{M}')
    c=report['modal_comparison']; score=sum(c[f'm={m}']['r_abs_err']+c[f'm={m}']['t_abs_err'] for m in (-1,0,1))
    rows.append({'M':M,'best_epoch':report['best_epoch'],'modal_response_score':score,'total_complex_l2':report['final_metrics']['total/complex_l2'],'scattered_complex_l2':report['final_metrics']['scattered/complex_l2'],'R_total':c['summary']['R_pinn_total'],'T_total':c['summary']['T_pinn_total'],'R_plus_T':c['summary']['pinn_energy_check'],'tm1':c['m=-1']['t_pinn_abs'],'tp1':c['m=1']['t_pinn_abs'],'report':str(root/f'M{M}/seed{args0.seed}/phase6_seed{args0.seed}_report.json')})
(root/'modal_order_ablation.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
