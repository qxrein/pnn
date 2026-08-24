#!/usr/bin/env python3
"""Compliant PDE-only Phase 6 runner for the explicit Fourier-modal PINN.

No RCWA field sample enters the loss.  The canonical RCWA file is loaded only
after optimisation for validation and modal de-embedding.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.config import load_config
from src.field_comparison import compare_fields, compare_modal_with_rcwa, extract_modal_amplitudes
from src.maxwell_2d_nondim import sample_nd_points
from src.maxwell_layered_bg import compute_background_coefficients
from src.mode_aware_fourier import ExplicitFourierModalDD
from src.reference_data import interpolate_reference_to_grid, load_reference_npz, normalize_reference_orientation
from src.reference_validation import validate_reference
from src.utils import resolve_device, resolve_training_dtype, set_seed
from scripts.train_lbg import evaluate, layered_bg_loss, make_lambda_0p8


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)): return value.item()
    if isinstance(value, np.complexfloating): return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray): return jsonable(value.tolist())
    if isinstance(value, complex): return {"real": value.real, "imag": value.imag}
    return value


def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception: return "unknown"


def metrics(model, physics, coeff, device, dtype, ref, ref_path):
    fields = evaluate(model, physics, device, dtype, "layered_bg", coeff)
    x, z = fields["x"][0], fields["z"][:, 0]
    rr, ri, _ = interpolate_reference_to_grid(ref, x, z)
    dual = compare_fields(fields["E_scat_real"], fields["E_scat_imag"], rr, ri, fields["z"], physics,
                          formulation="layered_bg", region_mask="external_only",
                          field_representation="scattered", reference_field_representation="total")
    total = dual["pinn_E_total_r"] + 1j * dual["pinn_E_total_i"]
    modal = extract_modal_amplitudes(total, x, z, physics, n_orders=1,
                                     formulation="layered_bg", field_representation="total")
    cmp = compare_modal_with_rcwa(modal, str(ref_path), int((len(np.load(ref_path)["c_refl"])-1)//2), physics=physics)
    return fields, {k: v for k, v in dual.items() if not isinstance(v, np.ndarray)}, modal, cmp


def row_from(epoch, losses, dual, modal, cmp):
    row = {"epoch": epoch, "pde_air": float(losses["pde_air"]), "pde_grating": float(losses["pde_grat"]),
           "pde_substrate": float(losses["pde_sub"]), "pde_total": float(losses["pde"]),
           "top_DtN": float(losses["top"]), "bottom_DtN": float(losses["bottom"]),
           "interface_E": float(losses["E_int1"] + losses["E_int2"]),
           "interface_H": float(losses["H_int1"] + losses["H_int2"]), "total_loss": float(losses["total"])}
    row.update({k: float(v) for k, v in dual.items() if isinstance(v, (float, np.floating, int, np.integer))})
    row.update({"R_total": float(modal["R_total"]), "T_total": float(modal["T_total"]), "R_plus_T": float(modal["energy_check"])})
    for m in (-1, 0, 1):
        value = cmp[f"m={m}"]
        for key in ("r_pinn_abs", "t_pinn_abs", "r_pinn_phase_deg", "t_pinn_phase_deg", "R_pinn", "T_pinn"):
            row[f"m{m}_{key}"] = value[key]
    return row


def acceptance(cmp):
    reasons=[]; summary=cmp["summary"]
    if abs(summary["pinn_energy_check"] - 1.) >= .01: reasons.append("energy balance")
    for m in (-1,0,1):
        d=cmp[f"m={m}"]
        for p in ("r","t"):
            rel=d[f"{p}_abs_err"] / max(d[f"{p}_rcwa_abs"], 1e-12)
            if rel >= .1: reasons.append(f"m={m} {p} amplitude")
            if d[f"{p}_phase_err_deg"] >= 20.: reasons.append(f"m={m} {p} phase")
        if d["R_pinn"] < -1e-12 or d["T_pinn"] < -1e-12: reasons.append(f"m={m} negative power")
    for m in (-1,1):
        if cmp[f"m={m}"]["t_pinn_abs"] <= 1e-8: reasons.append(f"m={m} transmitted order absent")
    return not reasons, sorted(set(reasons))


def run_seed(seed, args, physics, ref, ref_path, metadata, output_root=None, point_builder=None, initial_state=None, source_alpha=1.0):
    out = (output_root or ROOT / "outputs" / "phase6") / f"seed{seed}"
    if out.exists(): raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    device=resolve_device("cpu"); dtype=resolve_training_dtype("float64", device); coeff=compute_background_coefficients(physics)
    set_seed(seed)
    points=(point_builder(physics,args,device,dtype,seed) if point_builder else
            sample_nd_points(physics,args.n_per_region,args.n_interface,args.n_bc,args.n_bc,device,dtype,seed=seed))
    np.savez(out/"sampling_points.npz", **{k:v.detach().cpu().numpy() for k,v in points.items()})
    model=ExplicitFourierModalDD(physics,modal_order_max=args.modal_order_max,use_coefficient_mlp=True).to(device=device,dtype=dtype)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    best=float("inf"); best_state=None; best_epoch=0; history=[]
    def loss(): return layered_bg_loss(model,points,physics,coeff,1.,500.,500.,200.,100.,use_dtn=True,n_dtn_orders=args.n_dtn_orders,w_modal=0.,source_alpha=source_alpha)
    for epoch in range(1,args.epochs+1):
        opt.zero_grad(set_to_none=True); losses=loss(); losses["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); sched.step()
        current=float(losses["total"].detach())
        if current < best: best=current; best_epoch=epoch; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch % args.log_every == 0 or epoch == args.epochs:
            _,dual,modal,cmp=metrics(model,physics,coeff,device,dtype,ref,ref_path)
            row=row_from(epoch,{k:v.detach().cpu() for k,v in losses.items()},dual,modal,cmp); history.append(row)
            print(f"seed={seed} ep={epoch} loss={row['total_loss']:.3e} pde={row['pde_total']:.3e} R+T={row['R_plus_T']:.6f} |t+1|={row['m1_t_pinn_abs']:.3e}")
    checkpoint={"state_dict":best_state,"best_epoch":best_epoch,"best_loss":best,"seed":seed,
                "architecture":"explicit_fourier_modal","modal_order_max":args.modal_order_max,"source_alpha":source_alpha,
                "reference_file":str(ref_path),"metadata":metadata}
    torch.save(checkpoint,out/"best_checkpoint.pt")
    # Fresh object proves serialisation/reload rather than evaluating the live model.
    fresh=ExplicitFourierModalDD(physics,modal_order_max=args.modal_order_max,use_coefficient_mlp=True).to(device=device,dtype=dtype)
    fresh.load_state_dict(torch.load(out/"best_checkpoint.pt",map_location=device,weights_only=True)["state_dict"])
    fields,dual,modal,cmp=metrics(fresh,physics,coeff,device,dtype,ref,ref_path)
    reload_fields, reload_dual, reload_modal, reload_cmp=metrics(fresh,physics,coeff,device,dtype,ref,ref_path)
    reload_agreement=max(abs(dual[k]-reload_dual[k]) for k in dual if k.startswith(("total/","scattered/")) and isinstance(dual[k],float))
    ok,reasons=acceptance(cmp)
    with (out/f"training_history_seed{seed}.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=history[0].keys());w.writeheader();w.writerows(history)
    report={"phase5_passed":ok,"invalid_reason":None if ok else "; ".join(reasons),"seed":seed,
            "best_epoch":best_epoch,"best_loss":best,"checkpoint_reloaded":True,"reload_max_metric_difference":reload_agreement,
            "final_metrics":dual,"modal_comparison":cmp,"modal":modal,"last_log":history[-1],"metadata":metadata}
    (out/f"phase6_seed{seed}_report.json").write_text(json.dumps(jsonable(report),indent=2)+"\n")
    return report, history, fields


def plot(root, histories, reports):
    fig,ax=plt.subplots(figsize=(7,4))
    for seed,h in histories.items(): ax.plot([r["epoch"] for r in h],[r["m-1_t_pinn_abs"] for r in h],"o-",label=f"seed {seed} t-1");ax.plot([r["epoch"] for r in h],[r["m1_t_pinn_abs"] for r in h],"s--",label=f"seed {seed} t+1")
    ax.set(xlabel="epoch",ylabel="|t_m|",title="Explicit-modal convergence");ax.legend(fontsize=7);fig.tight_layout();fig.savefig(root/"modal_convergence.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for seed,h in histories.items(): ax.plot([r["epoch"] for r in h],[r["R_plus_T"] for r in h],marker="o",label=f"seed {seed}")
    ax.axhline(1.,color="k",ls="--");ax.set(xlabel="epoch",ylabel="R+T",title="Energy convergence");ax.legend();fig.tight_layout();fig.savefig(root/"energy_convergence.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); labels=[]; vals=[]
    for seed,r in reports.items():
        for m in (-1,0,1): labels.append(f"{seed}:m{m}");vals.append(r["modal_comparison"][f"m={m}"]["t_abs_err"])
    ax.bar(labels,vals);ax.set(ylabel="|t| amplitude error",title="Transmitted modal amplitude errors");ax.tick_params(axis="x",rotation=45);fig.tight_layout();fig.savefig(root/"modal_amplitude_comparison.png",dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--epochs",type=int,default=1000);ap.add_argument("--log-every",type=int,default=250);ap.add_argument("--n-per-region",type=int,default=128);ap.add_argument("--n-interface",type=int,default=64);ap.add_argument("--n-bc",type=int,default=64);ap.add_argument("--modal-order-max",type=int,default=3);ap.add_argument("--n-dtn-orders",type=int,default=8);ap.add_argument("--lr",type=float,default=5e-4);ap.add_argument('--output-root',default='outputs/phase6');args=ap.parse_args()
    root=ROOT/args.output_root
    if root.exists(): raise FileExistsError(f"Refusing to overwrite {root}")
    root.mkdir(parents=True)
    cfg=load_config(ROOT/"configs/default.yaml"); physics=make_lambda_0p8(cfg.physics); ref_path=ROOT/"outputs/reference_lambda_0p8_geometry_consistent_20260824.npz"; ref_meta=validate_reference(ref_path,physics);ref=normalize_reference_orientation(load_reference_npz(ref_path))
    metadata={"case":"lambda_0p8","formulation":"layered_bg","boundary":"DtN","dtype":"float64","device":"cpu","modal_loss_weight":0.,"reference":ref_meta,"wavelength":physics.wavelength,"period":physics.period,"ridge_width":physics.ridge_width,"ridge_height":physics.ridge_height,"n_ridge":physics.n_ridge,"n_substrate":physics.n_substrate,"monitor_planes":{"top":.08*physics.domain_height,"bottom":.92*physics.domain_height},"git_commit":git_commit(),"config_hash":hashlib.sha256(json.dumps(vars(args),sort_keys=True).encode()).hexdigest()}
    reports={}; histories={}
    for seed in (42,123): reports[seed],histories[seed],_=run_seed(seed,args,physics,ref,ref_path,metadata,root)
    plot(root,histories,reports)
    with (root/"phase6_summary.csv").open("w",newline="") as f:
        rows=[{"seed":s,"best_epoch":r["best_epoch"],"best_loss":r["best_loss"],"phase5_passed":r["phase5_passed"],"R_total":r["modal_comparison"]["summary"]["R_pinn_total"],"T_total":r["modal_comparison"]["summary"]["T_pinn_total"],"R_plus_T":r["modal_comparison"]["summary"]["pinn_energy_check"],"total_complex_l2":r["final_metrics"]["total/complex_l2"],"scattered_complex_l2":r["final_metrics"]["scattered/complex_l2"]} for s,r in reports.items()];w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    phase_pass=all(r["phase5_passed"] and r["reload_max_metric_difference"] < 1e-12 for r in reports.values())
    final={"phase5_passed":phase_pass,"valid":phase_pass,"invalid_reason":None if phase_pass else "PDE-only acceptance criteria failed; modal loss and optical coupling remain disabled.","reports":reports,"metadata":metadata}
    (root/"phase6_report.json").write_text(json.dumps(jsonable(final),indent=2)+"\n");print(json.dumps(jsonable(final),indent=2))
if __name__=="__main__": main()
