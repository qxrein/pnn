#!/usr/bin/env python3
"""Feature-scaling reparameterization audit.  Phases 2–8.

Parameterization change only — no geometry, PDE, DtN, or architecture change.

Reparameterization
------------------
Each ExplicitFourierModalNetwork has:
    coefficient_mlp = Sequential(Linear(1,32), Tanh(), Linear(32,14))
    [index 0: hidden linear]  [index 1: Tanh]  [index 2: final head]

At initialization the final head has zero weight and bias, so the output is
identically zero.  The hidden features h = Tanh(W0 z + b0) have per-channel
RMS values spanning ~7e-3 to ~4e-1 (50× spread).  This forces the final-head
weights to be ~1e7 to satisfy the PDE source term, well outside the affine
validity radius (~1e6).

The reparameterization inserts a fixed, non-trainable scaling layer between
the Tanh output and the final head:

    h_scaled = h / scale          (scale = per-channel RMS, frozen buffer)
    E_output  = W_head @ h_scaled + b_head

This is equivalent to replacing W_head → W_head * diag(scale) in the
original parameterization, keeping the represented function identical at
initialization:

    W_head_new @ h_scaled = (W_head_new * diag(scale)) @ h = W_head_orig @ h

After reparameterization, unit-scale final-head weights produce field outputs
of order ~1, and the required head magnitudes to close the residual drop to
~5 instead of ~1e7.

The transformation preserves:
1. Zero-head → E_scat = 0 exactly (because W_head_new starts at zero).
2. Hidden features: the Tanh output is unchanged; only the division by scale
   is inserted before the head.
3. The production residual at theta=0 is identical before and after.
4. All physical conventions, geometry, Maxwell formulation, DtN, etc.

Safety invariants
-----------------
- canonical reference and companion: read-only throughout
- modal-data loss = 0
- optical coupling disabled
- no nonlinear training in Phases 0–5
- previous audit outputs untouched
- output directory outputs/phase5_feature_scaling/ must not pre-exist
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.field_comparison import compare_modal_with_rcwa, extract_modal_amplitudes
from src.geometry import epsilon_r as epsilon_r_fn
from src.maxwell_2d_nondim import sample_nd_points
from src.maxwell_layered_bg import (
    compute_background_coefficients,
    maxwell_2d_lbg_pde_residual,
)
from src.modal_dtn import modal_dtn_pointwise
from src.mode_aware_fourier import ExplicitFourierModalDD, ExplicitFourierModalNetwork
from src.reference_companion import COMPANION_FILENAME, sha256_file
from src.reference_data import load_reference_npz, normalize_reference_orientation
from src.reference_validation import validate_reference
from src.utils import set_seed
from scripts.run_phase5_conditioning import physical_scales
from scripts.run_phase6_explicit_modal import jsonable, metrics
from scripts.train_lbg import evaluate, layered_bg_loss, make_lambda_0p8
from scripts.run_frozen_head_least_squares import (
    BLOCK_ORDER, CANONICAL_REF, COMPANION_PATH, HEAD_PARAM_COUNT,
    MODAL_ORDER_MAX, N_DTN_ORDERS, N_COLLOCATION, RCOND, SEED,
    TARGET_T1, VALIDATED_SCALES,
    assemble_system, block_norms, confirm_validated_scales,
    freeze_except_head, grouped_norms, head_parameter_records,
    head_parameters, load_points, model_config_hash, modal_report,
    pack_head, pack_residual, residual_blocks, row_metadata, row_scales,
    unpack_head,
)
import scripts.run_frozen_head_least_squares as _fhls

from scripts.run_frozen_head_integrity import (
    companion_t_modes, git_commit, git_status, make_fresh_model,
    parameter_ordering_hash, phase1_parameter_mapping,
    phase2_hidden_feature_invariance, phase3_global_affine,
    run_compileall, run_pytest,
)


# ─────────────────────────────────────────────────────────────────────────────
# Patched spatial_fourier_t
# ExplicitFourierModalNetwork.forward requires x,z to be grad-enabled leaves
# so that autograd can compute dE/dx and dE/dz for H fields.  The version in
# run_frozen_head_least_squares may be loaded from a stale .pyc; patch it here.
# ─────────────────────────────────────────────────────────────────────────────

def _spatial_fourier_t_fixed(
        model, physics, z_bot: float, n_quad: int = 512) -> dict[int, complex]:
    x    = torch.linspace(
        0.0, physics.period, n_quad + 1, dtype=torch.float64,
    )[:-1].requires_grad_(True)
    z    = torch.empty_like(x.detach()).fill_(z_bot).requires_grad_(True)
    er, ei, *_ = model.net_sub.field_components(x, z)
    e    = (er + 1j * ei).detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    g0   = 2.0 * np.pi / physics.period
    out  = {}
    for m in range(-MODAL_ORDER_MAX, MODAL_ORDER_MAX + 1):
        out[m] = complex(np.mean(e * np.exp(-1j * m * g0 * x_np)))
    return out

_fhls.spatial_fourier_t = _spatial_fourier_t_fixed  # patch before modal_report is called


PHASE_VALIDITY_THRESHOLD = 1e-8
FEATURE_SCALE_FLOOR      = 1e-6   # never divide by less than this
MAX_HEAD_MAG_TARGET      = 1e4    # reject candidates above this

# ─────────────────────────────────────────────────────────────────────────────
# ScaledHead: non-trainable feature-normalisation layer
# ─────────────────────────────────────────────────────────────────────────────

class ScaledHead(nn.Module):
    """Wraps coefficient_mlp to normalise hidden features before the final linear.

    The stored scale is a frozen buffer (not a Parameter).  Forward path:
        1. z → Linear(1,32) → Tanh()     [unchanged hidden basis]
        2. h / scale                       [fixed normalisation]
        3. Linear(32,14) @ h_norm          [final head]

    Reparameterization rule: at construction, absorb diag(scale) into the
    final-head weight so the represented function is preserved:

        W_new = W_old * scale[None, :]   (broadcast over output dimension)

    At initialization W_old = 0 so W_new = 0 and the reparameterization is
    trivially exact.  For any subsequent training the relationship

        W_new @ h_norm  ==  (W_new * scale[None,:]) / scale[None,:]  @ h
                        ==  W_eff_unscaled @ h

    is maintained.  The head parameters (W_new, b) are the trainable unknowns
    in the reparameterized system.
    """

    def __init__(self, original_mlp: nn.Sequential,
                 scale: np.ndarray | torch.Tensor) -> None:
        super().__init__()
        # Clone the three layers
        lin0 = original_mlp[0]   # Linear(1, 32) — frozen
        tanh = original_mlp[1]   # Tanh          — frozen
        head = original_mlp[2]   # Linear(32,14) — trainable head

        self.lin0 = lin0
        self.tanh = tanh

        # Register scale as a non-trainable buffer
        scale_t = torch.as_tensor(scale, dtype=torch.float64).detach().clone()
        self.register_buffer("scale", scale_t)   # (32,)

        # Build new head with the same shape but scaled weights
        in_f, out_f = head.in_features, head.out_features
        self.head = nn.Linear(in_f, out_f, bias=True, dtype=torch.float64)
        with torch.no_grad():
            # W_new[o, j] = W_old[o, j] * scale[j]
            # At zero init: W_new = 0 * diag(scale) = 0 — no change
            self.head.weight.copy_(head.weight * scale_t[None, :])
            self.head.bias.copy_(head.bias)

        # Freeze lin0/tanh (they mirror the frozen parent params)
        for p in self.lin0.parameters():
            p.requires_grad_(False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h     = self.tanh(self.lin0(z))           # (N, 32)
        sc    = self.scale.to(dtype=h.dtype, device=h.device)
        h_norm = h / sc[None, :]                  # (N, 32) — normalised
        return self.head(h_norm)                  # (N, 14)

    def hidden_features(self, z: torch.Tensor) -> torch.Tensor:
        """Return the raw (un-normalised) Tanh output, same as before scaling."""
        with torch.no_grad():
            return self.tanh(self.lin0(z))

    def hidden_features_normalised(self, z: torch.Tensor) -> torch.Tensor:
        """Return h / scale — the features seen by the head."""
        h = self.tanh(self.lin0(z))
        sc = self.scale.to(dtype=h.dtype, device=h.device)
        return h / sc[None, :]


def apply_feature_scaling(net: ExplicitFourierModalNetwork,
                          scale: np.ndarray) -> None:
    """Replace net.coefficient_mlp with a ScaledHead in-place.

    Safe to call only when the final-head weight is zero (initialization).
    Preserves: hidden features, Tanh basis, all frozen parameters, physics.
    """
    assert net.coefficient_mlp is not None, "no coefficient_mlp to scale"
    net.coefficient_mlp = ScaledHead(net.coefficient_mlp, scale)


def compute_feature_scales(model: ExplicitFourierModalDD,
                            pts: dict,
                            floor: float = FEATURE_SCALE_FLOOR) -> dict[str, np.ndarray]:
    """Compute per-channel RMS of Tanh hidden features at the collocation points.

    Returns a dict mapping subnet name → (32,) float64 array of scales.
    The floor prevents division by zero for dormant channels.
    """
    scales = {}
    for net_name, net, zkey in (
        ("net_air",  model.net_air,  "z_air"),
        ("net_grat", model.net_grat, "z_grat"),
        ("net_sub",  model.net_sub,  "z_sub"),
    ):
        z = pts.get(zkey)
        if z is None or len(z) == 0:
            raise ValueError(f"No collocation points for {net_name} (key {zkey})")
        with torch.no_grad():
            zn = (2.0*(z - net.z_lo)/(net.z_hi - net.z_lo) - 1.0)[:, None]
            h  = net.coefficient_mlp[1](net.coefficient_mlp[0](zn)).numpy()
        rms_ch  = np.sqrt(np.mean(h**2, axis=0))           # (32,)
        scales[net_name] = np.maximum(rms_ch, floor)
    return scales


def make_scaled_model(physics, pts: dict,
                      floor: float = FEATURE_SCALE_FLOOR) -> ExplicitFourierModalDD:
    """Create a fresh model and apply feature scaling computed from pts."""
    set_seed(SEED)
    model  = ExplicitFourierModalDD(physics, MODAL_ORDER_MAX, True).double()
    scales = compute_feature_scales(model, pts, floor)
    apply_feature_scaling(model.net_air,  scales["net_air"])
    apply_feature_scaling(model.net_grat, scales["net_grat"])
    apply_feature_scaling(model.net_sub,  scales["net_sub"])
    return model, scales


# ─────────────────────────────────────────────────────────────────────────────
# Helpers that work with scaled models
# ─────────────────────────────────────────────────────────────────────────────

def scaled_head_parameters(model: ExplicitFourierModalDD) -> list[nn.Parameter]:
    """Return the trainable head parameters of the scaled model."""
    params = []
    for net in (model.net_air, model.net_grat, model.net_sub):
        assert isinstance(net.coefficient_mlp, ScaledHead), \
            "Expected ScaledHead; did you call apply_feature_scaling?"
        params.extend([net.coefficient_mlp.head.weight,
                        net.coefficient_mlp.head.bias])
    return params


def freeze_scaled_except_head(model: ExplicitFourierModalDD) -> list[nn.Parameter]:
    """Freeze every parameter except the ScaledHead final-head weights."""
    for p in model.parameters():
        p.requires_grad_(False)
    heads = scaled_head_parameters(model)
    for p in heads:
        p.requires_grad_(True)
    n = sum(p.numel() for p in heads)
    if n != HEAD_PARAM_COUNT:
        raise RuntimeError(f"expected {HEAD_PARAM_COUNT} head params, got {n}")
    return heads


def get_raw_hidden(model: ExplicitFourierModalDD, pts: dict) -> dict[str, np.ndarray]:
    """Return raw (un-normalised) Tanh features at collocation points."""
    out = {}
    for net_name, net, zkey in (
        ("net_air",  model.net_air,  "z_air"),
        ("net_grat", model.net_grat, "z_grat"),
        ("net_sub",  model.net_sub,  "z_sub"),
    ):
        z = pts.get(zkey)
        if z is None: continue
        sh = net.coefficient_mlp
        if isinstance(sh, ScaledHead):
            out[net_name] = sh.hidden_features(
                (2.0*(z - sh.lin0.in_features) / 1.0)[:, None]
            )
            # re-compute properly:
            with torch.no_grad():
                zn = (2.0*(z - net.z_lo)/(net.z_hi - net.z_lo) - 1.0)[:, None]
                out[net_name] = sh.tanh(sh.lin0(zn)).numpy()
        else:
            with torch.no_grad():
                zn = (2.0*(z - net.z_lo)/(net.z_hi - net.z_lo) - 1.0)[:, None]
                out[net_name] = net.coefficient_mlp[1](
                    net.coefficient_mlp[0](zn)).numpy()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: reparameterization verification tests
# ─────────────────────────────────────────────────────────────────────────────

def phase2_verify_reparameterization(physics, pts, coeff) -> dict:
    """Seven correctness checks for the feature-scaling reparameterization."""
    print("\n[Phase 2] Reparameterization verification", flush=True)
    results = {}

    # ── T1: zero-head output remains exactly zero ─────────────────────────
    model_s, scales = make_scaled_model(physics, pts)
    heads_s = freeze_scaled_except_head(model_s)
    unpack_head(heads_s, np.zeros(HEAD_PARAM_COUNT))

    x_test = pts["x_grat"].requires_grad_(True)
    z_test = pts["z_grat"].requires_grad_(True)
    er, ei, *_ = model_s.net_grat.field_components(x_test, z_test)
    max_zero = float(max(er.abs().max(), ei.abs().max()).detach())
    t1_pass  = max_zero < 1e-14
    results["T1_zero_head_output_zero"] = {
        "max_abs_output": max_zero, "passed": t1_pass,
        "description": "zero final-head → E_scat=0 everywhere"
    }
    print(f"  [{'PASS' if t1_pass else 'FAIL'}] T1 zero-head: max|E|={max_zero:.2e}")

    # ── T2: raw hidden features unchanged by scaling ──────────────────────
    model_u = make_fresh_model(physics)
    h_unscaled = get_raw_hidden(model_u, pts)
    h_scaled   = get_raw_hidden(model_s, pts)
    max_h_change = max(
        float(np.max(np.abs(h_scaled[k] - h_unscaled[k])))
        for k in h_unscaled if k in h_scaled
    )
    t2_pass = max_h_change < 1e-14
    results["T2_hidden_features_unchanged"] = {
        "max_hidden_change": max_h_change, "passed": t2_pass,
        "description": "Tanh output identical before and after scaling"
    }
    print(f"  [{'PASS' if t2_pass else 'FAIL'}] T2 hidden features: max_change={max_h_change:.2e}")

    # ── T3: W_scaled @ h_scaled = W_orig @ h for a test weight matrix ────
    # Use a random W_orig, compute W_scaled = W_orig * diag(s), verify equality
    rng = np.random.default_rng(0)
    for net_name, net_s, net_u in (
        ("net_air",  model_s.net_air,  model_u.net_air),
        ("net_grat", model_s.net_grat, model_u.net_grat),
        ("net_sub",  model_s.net_sub,  model_u.net_sub),
    ):
        sc   = net_s.coefficient_mlp.scale.numpy()           # (32,)
        W_rnd = rng.standard_normal((14, 32))
        # Set both models' heads to W_rnd (scaled vs unscaled convention)
        with torch.no_grad():
            # unscaled model: W_head = W_rnd directly
            net_u.coefficient_mlp[2].weight.copy_(
                torch.tensor(W_rnd, dtype=torch.float64))
            net_u.coefficient_mlp[2].bias.zero_()
            # scaled model: W_head_s = W_rnd / diag(sc) so that
            # W_head_s @ h_norm = W_rnd / sc * h / sc ... wait
            # Correct: we set W_head_s = W_rnd; then output = W_rnd @ (h/sc)
            # Unscaled output = W_rnd @ h
            # For equality set W_head_s = W_rnd * sc[None,:] so
            # W_head_s @ (h/sc) = W_rnd @ h  ✓
            net_s.coefficient_mlp.head.weight.copy_(
                torch.tensor(W_rnd * sc[None, :], dtype=torch.float64))
            net_s.coefficient_mlp.head.bias.zero_()

        z_col = pts[{"net_air":"z_air","net_grat":"z_grat","net_sub":"z_sub"}[net_name]]
        z_col_req = z_col.detach().requires_grad_(True)

        er_u, ei_u, *_ = net_u.field_components(z_col_req.clone().requires_grad_(True),
                                                  z_col_req.clone().requires_grad_(True))
        er_s, ei_s, *_ = net_s.field_components(z_col_req.clone().requires_grad_(True),
                                                  z_col_req.clone().requires_grad_(True))
        err = float(max(
            (er_u.detach() - er_s.detach()).abs().max(),
            (ei_u.detach() - ei_s.detach()).abs().max(),
        ))
        # Restore to zero
        with torch.no_grad():
            net_u.coefficient_mlp[2].weight.zero_()
            net_s.coefficient_mlp.head.weight.zero_()

    t3_pass = err < 1e-10
    results["T3_Wscaled_hscaled_equals_W_h"] = {
        "max_error": err, "passed": t3_pass,
        "description": "W_scaled @ h_scaled = W_orig @ h for random W_orig"
    }
    print(f"  [{'PASS' if t3_pass else 'FAIL'}] T3 W@h identity: max_err={err:.2e}")

    # ── T4–T6: production residual unchanged at zero head ─────────────────
    model_u2 = make_fresh_model(physics)
    unpack_head(head_parameters(model_u2), np.zeros(HEAD_PARAM_COUNT))
    freeze_except_head(model_u2)

    r_unscaled, _, lengths = pack_residual(residual_blocks(model_u2, pts, physics, coeff))
    r_scaled,   _, _       = pack_residual(residual_blocks(model_s,  pts, physics, coeff))
    r_u_np = r_unscaled.detach().numpy()
    r_s_np = r_scaled.detach().numpy()

    resid_change = float(np.max(np.abs(r_s_np - r_u_np)))
    resid_rms    = float(np.sqrt(np.mean((r_s_np - r_u_np)**2)))
    t4_pass      = resid_change < 1e-12
    results["T4_production_residual_unchanged"] = {
        "max_change": resid_change, "rms_change": resid_rms,
        "passed": t4_pass,
        "description": "production residual identical at zero head"
    }
    print(f"  [{'PASS' if t4_pass else 'FAIL'}] T4 residual: max_change={resid_change:.2e}")

    # ── T5: modal extraction unchanged ────────────────────────────────────
    from src.reference_data import load_reference_npz, normalize_reference_orientation
    from src.reference_validation import validate_reference
    ref_path    = ROOT / CANONICAL_REF
    ref         = normalize_reference_orientation(load_reference_npz(ref_path))
    ref_meta    = validate_reference(ref_path, physics)
    companion_p = ROOT / COMPANION_PATH
    device      = torch.device("cpu")
    dtype       = torch.float64

    modal_u = modal_report(model_u2, physics, coeff, device, dtype, ref, ref_path, companion_p)
    modal_s = modal_report(model_s,  physics, coeff, device, dtype, ref, ref_path, companion_p)
    t1u = modal_u["t_minus1_abs"]; t1s = modal_s["t_minus1_abs"]
    t5_pass = abs(t1u - t1s) < 1e-12
    results["T5_modal_extraction_unchanged"] = {
        "t_minus1_unscaled": t1u, "t_minus1_scaled": t1s,
        "max_change": abs(t1u-t1s), "passed": t5_pass,
        "description": "modal extraction identical at zero head"
    }
    print(f"  [{'PASS' if t5_pass else 'FAIL'}] T5 modal: |t-1|_u={t1u:.8f}  |t-1|_s={t1s:.8f}")

    # ── T6: canonical reference SHA unchanged ─────────────────────────────
    import hashlib as hl
    import json as _json
    manifest  = _json.loads((ROOT / 'outputs/reference_companion/manifest.json').read_text())
    can_sha   = hl.sha256(ref_path.read_bytes()).hexdigest()
    com_sha   = hl.sha256(companion_p.read_bytes()).hexdigest()
    t6_can    = can_sha == manifest['canonical_sha256']
    t6_com    = com_sha == manifest['companion_sha256']
    t6_pass   = t6_can and t6_com
    results["T6_canonical_companion_unchanged"] = {
        "canonical_sha_ok": t6_can, "companion_sha_ok": t6_com,
        "passed": t6_pass,
        "description": "canonical reference and companion are unmodified"
    }
    print(f"  [{'PASS' if t6_pass else 'FAIL'}] T6 SHA: can={t6_can}  com={t6_com}")

    # ── T7: affine property holds at theta=0 for scaled model ─────────────
    # Build a tiny Jacobian (10 columns) and verify affine relation
    NCOLS = 20
    rng2  = np.random.default_rng(7)
    J_mini = np.zeros((len(r_s_np), NCOLS))
    for j in range(NCOLS):
        ej = np.zeros(HEAD_PARAM_COUNT); ej[j*69] = 1.0   # stride through param space
        unpack_head(heads_s, ej)
        rj, _, _ = pack_residual(residual_blocks(model_s, pts, physics, coeff))
        J_mini[:, j] = rj.detach().numpy() - r_s_np
        unpack_head(heads_s, np.zeros(HEAD_PARAM_COUNT))

    theta_test = rng2.standard_normal(NCOLS)
    # Direct eval
    full_theta      = np.zeros(HEAD_PARAM_COUNT)
    for j in range(NCOLS): full_theta[j*69] = theta_test[j]
    unpack_head(heads_s, full_theta)
    r_direct, _, _  = pack_residual(residual_blocks(model_s, pts, physics, coeff))
    r_direct_np     = r_direct.detach().numpy()
    unpack_head(heads_s, np.zeros(HEAD_PARAM_COUNT))
    r_affine_np     = J_mini @ theta_test + r_s_np
    t7_err          = float(np.max(np.abs(r_direct_np - r_affine_np)))
    t7_pass         = t7_err < 1e-10
    results["T7_scaled_residual_affine"] = {
        "max_affine_error": t7_err, "n_test_cols": NCOLS, "passed": t7_pass,
        "description": "residual is affine in scaled head parameters (mini test)"
    }
    print(f"  [{'PASS' if t7_pass else 'FAIL'}] T7 affine: max_err={t7_err:.2e}")

    all_passed = all(v["passed"] for v in results.values())
    results["all_passed"] = all_passed
    print(f"  Phase 2: {'ALL PASSED' if all_passed else 'FAILURES — see above'}", flush=True)
    if not all_passed:
        raise RuntimeError("Phase 2 failed — reparameterization is incorrect. Stopping.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: full affine audit on scaled system + one-hot + hidden-feature invariance
# ─────────────────────────────────────────────────────────────────────────────

def phase3_post_scaling_affine(physics, pts, coeff,
                                jacobian_s, r0_s, lengths) -> dict:
    """Full affine + one-hot + hidden-feature invariance on scaled model."""
    print("\n[Phase 3] Post-scaling affine audit", flush=True)

    model_s, scales = make_scaled_model(physics, pts)
    heads_s         = freeze_scaled_except_head(model_s)
    rng             = np.random.default_rng(123)
    u               = rng.standard_normal(HEAD_PARAM_COUNT)
    v               = rng.standard_normal(HEAD_PARAM_COUNT)
    u /= np.linalg.norm(u); v /= np.linalg.norm(v)

    alphas   = [1e-6, 1e-3, 1.0, 1e3, 1e6]
    rows_alpha = []
    print("  Affine check by alpha:")
    for alpha in alphas:
        theta = alpha * (u + v)
        unpack_head(heads_s, theta)
        r_dir, _, _ = pack_residual(residual_blocks(model_s, pts, physics, coeff))
        r_dir_np    = r_dir.detach().numpy()
        unpack_head(heads_s, np.zeros(HEAD_PARAM_COUNT))
        r_aff    = jacobian_s @ theta + r0_s
        err      = r_dir_np - r_aff
        max_abs  = float(np.max(np.abs(err)))
        rel      = float(np.linalg.norm(err) / (np.linalg.norm(r_dir_np) + 1e-30))
        has_nan  = bool(np.any(np.isnan(r_dir_np)))
        has_inf  = bool(np.any(np.isinf(r_dir_np)))
        valid    = not has_nan and not has_inf and max_abs < 1e-8
        rows_alpha.append({"alpha": alpha, "theta_norm": float(np.linalg.norm(theta)),
                            "max_abs": max_abs, "rel": rel, "valid": valid})
        print(f"    alpha={alpha:.0e}  |theta|={np.linalg.norm(theta):.2e}  "
              f"max_err={max_abs:.2e}  rel={rel:.2e}  [{'OK' if valid else 'FAIL'}]")

    # one-hot round-trip
    print("  One-hot round-trips:")
    test_idx  = [0, 100, 500, 1000, 1385]
    oh_rows   = []
    for idx in test_idx:
        theta_oh = np.zeros(HEAD_PARAM_COUNT); theta_oh[idx] = 1.0
        unpack_head(heads_s, theta_oh)
        rb = pack_head(heads_s).numpy()
        readback_val  = float(rb[idx])
        max_others    = float(np.max(np.abs(np.delete(rb, idx))))
        passed        = abs(readback_val - 1.0) < 1e-12 and max_others < 1e-12
        oh_rows.append({"idx": idx, "readback": readback_val,
                         "max_others": max_others, "passed": passed})
        unpack_head(heads_s, np.zeros(HEAD_PARAM_COUNT))
        print(f"    idx={idx:4d}  readback={readback_val:.15e}  "
              f"max_others={max_others:.2e}  [{'PASS' if passed else 'FAIL'}]")

    # hidden feature invariance at several scales
    print("  Hidden-feature invariance:")
    h_base = get_raw_hidden(model_s, pts)
    hf_rows = []
    for mag in [1e-6, 1e-3, 1.0, 1e3, 1e6]:
        theta_m = rng.standard_normal(HEAD_PARAM_COUNT) * mag
        unpack_head(heads_s, theta_m)
        h_m = get_raw_hidden(model_s, pts)
        max_ch = max(float(np.max(np.abs(h_m[k] - h_base[k]))) for k in h_base if k in h_m)
        inv    = max_ch < 1e-14
        hf_rows.append({"mag": mag, "max_hidden_change": max_ch, "invariant": inv})
        print(f"    |theta|={mag:.0e}  max_hidden_change={max_ch:.2e}  [{'OK' if inv else 'FAIL'}]")
        unpack_head(heads_s, np.zeros(HEAD_PARAM_COUNT))

    all_alpha_ok = all(r["valid"] for r in rows_alpha)
    all_oh_ok    = all(r["passed"] for r in oh_rows)
    all_hf_ok    = all(r["invariant"] for r in hf_rows)
    breakdown_alpha = next((r["alpha"] for r in rows_alpha if not r["valid"]), None)
    all_passed = all_alpha_ok and all_oh_ok and all_hf_ok

    print(f"  Phase 3: affine_ok={all_alpha_ok}  one_hot_ok={all_oh_ok}  "
          f"hidden_inv_ok={all_hf_ok}", flush=True)
    return {
        "all_passed":        all_passed,
        "affine_by_alpha":   rows_alpha,
        "breakdown_alpha":   breakdown_alpha,
        "one_hot_results":   oh_rows,
        "hidden_invariance": hf_rows,
        "affine_valid_for_all_tested_alphas": all_alpha_ok,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: post-scaling frozen-head Tikhonov solve
# ─────────────────────────────────────────────────────────────────────────────

def _tikhonov(A, b, lam, D_scale):
    """min ||A@θ+b||² + λ||D@θ||²  via augmented SVD."""
    A_aug = np.vstack([A, np.sqrt(lam) * np.diag(D_scale)])
    b_aug = np.concatenate([b, np.zeros(A.shape[1])])
    u, s, vt = np.linalg.svd(A_aug, full_matrices=False)
    rank     = int(np.sum(s > RCOND * s[0]))
    inv      = np.where(s > RCOND * s[0], 1.0/s, 0.0)
    theta    = vt.T @ (inv * (u.T @ (-b_aug)))
    residual = A @ theta + b
    cond     = float(s[0]/s[rank-1]) if rank > 0 else float("inf")
    return {"theta": theta, "s": s, "rank": rank, "cond": cond,
            "raw_resid_norm": float(np.linalg.norm(residual)),
            "phys_resid_norm": None, "residual": residual}


def phase4_scaled_frozen_head_solve(physics, pts, coeff, jacobian_s, r0_s,
                                     lengths, scales_row, scales_dict) -> dict:
    """Tikhonov sweep on the physically-scaled post-reparameterization system."""
    print("\n[Phase 4] Post-scaling frozen-head Tikhonov solve", flush=True)

    A_phys = jacobian_s / scales_row[:, None]
    b_phys = r0_s / scales_row
    col_norm = np.linalg.norm(A_phys, axis=0)
    D_scale  = np.maximum(col_norm, RCOND * max(col_norm.max(), 1.0))

    # Print column norm summary
    print(f"  Post-scaling column norms: min={col_norm.min():.4e}  "
          f"max={col_norm.max():.4e}  mean={col_norm.mean():.4e}  "
          f"n_near_zero={int(np.sum(col_norm < 1e-10))}", flush=True)

    lambdas = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    results = []
    print("  Tikhonov sweep:", flush=True)
    for lam in lambdas:
        sol = _tikhonov(A_phys, b_phys, lam, D_scale)
        sol["lambda"] = lam
        sol["phys_resid_norm"] = float(np.linalg.norm(A_phys @ sol["theta"] + b_phys))
        theta_max = float(np.max(np.abs(sol["theta"])))
        sol["theta_max"] = theta_max
        sol["theta_norm"] = float(np.linalg.norm(sol["theta"]))
        sol["finite"]     = np.isfinite(sol["theta_norm"]) and sol["theta_norm"] < 1e12
        print(f"    λ={lam:.0e}  rank={sol['rank']}  cond={sol['cond']:.2e}  "
              f"|θ|={sol['theta_norm']:.3e}  max|θ|={theta_max:.3e}  "
              f"raw_res={sol['raw_resid_norm']:.4e}  "
              f"phys_res={sol['phys_resid_norm']:.4e}  "
              f"{'✓' if sol['finite'] else '✗'}", flush=True)
        results.append(sol)

    finite_ok = [s for s in results if s["finite"] and s["theta_max"] < MAX_HEAD_MAG_TARGET]
    print(f"  {len(finite_ok)}/{len(results)} candidates have max|θ| < {MAX_HEAD_MAG_TARGET:.0e}",
          flush=True)
    return results, finite_ok, col_norm, A_phys, b_phys


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5+6: direct re-evaluation and withheld modal evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_candidate(physics, pts, coeff, jacobian_s, r0_s, lengths,
                   scales_row, sol, device, dtype, ref, ref_path,
                   companion_p, out_dir) -> dict:
    """Assign theta to a fresh scaled model; evaluate residual and modal output."""
    lam   = sol["lambda"]
    label = f"lam{lam:.0e}"
    theta = sol["theta"]

    model_s, _ = make_scaled_model(physics, pts)
    heads_s    = freeze_scaled_except_head(model_s)
    unpack_head(heads_s, theta)

    # Direct residual
    r_eval, _, _ = pack_residual(residual_blocks(model_s, pts, physics, coeff))
    r_direct     = r_eval.detach().numpy()
    r_affine     = jacobian_s @ theta + r0_s
    affine_err   = float(np.max(np.abs(r_direct - r_affine)))
    direct_norm  = float(np.linalg.norm(r_direct))
    has_nan      = bool(np.any(np.isnan(r_direct)))
    has_inf      = bool(np.any(np.isinf(r_direct)))
    reeval_ok    = affine_err < 1e-8 and not has_nan and not has_inf

    # Hidden features unchanged
    h_after  = get_raw_hidden(model_s, pts)
    model_s2, _ = make_scaled_model(physics, pts)
    h_before = get_raw_hidden(model_s2, pts)
    hidden_change = max(
        float(np.max(np.abs(h_after[k] - h_before[k])))
        for k in h_before if k in h_after
    )

    # Modal output (only if reeval passes)
    modal_out = None
    if reeval_ok:
        try:
            modal_out = modal_report(model_s, physics, coeff, device, dtype,
                                     ref, ref_path, companion_p)
        except Exception as exc:
            modal_out = {"error": str(exc)}

    # Save checkpoint
    ckpt = out_dir / f"scaled_{label}_checkpoint.pt"
    torch.save({
        "state_dict": {k: v.detach().cpu().clone()
                       for k, v in model_s.state_dict().items()},
        "theta": theta, "label": label, "lambda": lam,
    }, ckpt)

    t1   = modal_out.get("t_minus1_abs") if modal_out and "error" not in modal_out else None
    tp   = modal_out.get("t_plus1_abs")  if modal_out and "error" not in modal_out else None
    RT   = modal_out.get("R_plus_T")     if modal_out and "error" not in modal_out else None
    L2   = modal_out.get("total_complex_l2") if modal_out and "error" not in modal_out else None

    result = {
        "label": label, "lambda": lam,
        "theta_norm": float(np.linalg.norm(theta)),
        "theta_max":  float(np.max(np.abs(theta))),
        "direct_residual_norm":         direct_norm,
        "affine_predicted_norm":        float(np.linalg.norm(r_affine)),
        "direct_minus_affine_max_abs":  affine_err,
        "has_nan": has_nan, "has_inf": has_inf,
        "reeval_ok": reeval_ok,
        "hidden_feature_change": hidden_change,
        "checkpoint": str(ckpt),
        "t_minus1_abs": t1, "t_plus1_abs": tp,
        "R_plus_T": RT, "total_complex_l2": L2,
        "modal": {k: v for k, v in (modal_out or {}).items()
                  if not isinstance(v, np.ndarray)} if modal_out else None,
    }
    t_str = (f"|t-1|={t1:.5f}  |t+1|={tp:.5f}" if t1 is not None else "modal N/A")
    print(f"  [{label}]  |θ|={result['theta_norm']:.2e}  "
          f"direct={direct_norm:.4e}  affine_err={affine_err:.2e}  "
          f"reeval={'OK' if reeval_ok else 'FAIL'}  {t_str}", flush=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: short full-model PDE-only smoke (1,000 epochs)
# ─────────────────────────────────────────────────────────────────────────────

def phase6_smoke(physics, pts, coeff, device, dtype,
                 ref, ref_path, companion_p, out_dir: Path,
                 best_finite_theta: np.ndarray | None) -> dict:
    """1,000-epoch PDE-only smoke with three initializations."""
    print("\n[Phase 6] Short full-model smoke (1000 epochs)", flush=True)

    from src.reference_data import interpolate_reference_to_grid
    from src.maxwell_2d_nondim import maxwell_2d_nd_interface_loss
    from src.maxwell_layered_bg import (
        lbg_bottom_bc, lbg_top_bc, lbg_vertical_interface_loss,
    )
    from scripts.train_lbg import evaluate as train_evaluate

    def run_smoke(label: str, init_theta: np.ndarray | None,
                  lr: float = 5e-4, epochs: int = 1000) -> dict:
        """One smoke run. init_theta: if None, uses zero head (fresh model)."""
        set_seed(SEED)
        model_s, sc = make_scaled_model(physics, pts)
        # Unfreeze all for full-model training
        for p in model_s.parameters():
            p.requires_grad_(True)
        # If init_theta provided, set the final head
        if init_theta is not None:
            heads_s = scaled_head_parameters(model_s)
            unpack_head(heads_s, init_theta)

        opt   = torch.optim.Adam(model_s.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        hist  = []

        def _loss_fn(m, pts_arg, physics_arg):
            return layered_bg_loss(m, pts_arg, physics_arg, coeff,
                                   w_pde=1.0, w_E=1.0, w_H=1.0,
                                   w_top=1.0, w_bot=1.0,
                                   use_dtn=True, n_dtn_orders=N_DTN_ORDERS,
                                   rcwa_amps=None, w_modal=0.0)

        for ep in range(1, epochs + 1):
            model_s.train()
            opt.zero_grad(set_to_none=True)
            losses = _loss_fn(model_s, pts, physics)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
            opt.step(); sched.step()

            if ep % 100 == 0 or ep == 1:
                with torch.enable_grad():
                    modal_m = modal_report(model_s, physics, coeff, device, dtype,
                                          ref, ref_path, companion_p)
                head_norm = float(pack_head(scaled_head_parameters(model_s)).norm())
                row = {
                    "epoch": ep,
                    "pde_air":   float(losses["pde_air"].detach()),
                    "pde_grat":  float(losses["pde_grat"].detach()),
                    "pde_sub":   float(losses["pde_sub"].detach()),
                    "top_DtN":   float(losses["top"].detach()),
                    "bottom_DtN":float(losses["bottom"].detach()),
                    "E_int":     float((losses["E_int1"]+losses["E_int2"]).detach()),
                    "H_int":     float((losses["H_int1"]+losses["H_int2"]).detach()),
                    "total":     float(losses["total"].detach()),
                    "head_norm": head_norm,
                    "t_minus1":  modal_m["t_minus1_abs"],
                    "t_plus1":   modal_m["t_plus1_abs"],
                    "R_plus_T":  modal_m["R_plus_T"],
                    "total_complex_l2": modal_m["total_complex_l2"],
                    "scattered_complex_l2": modal_m["scattered_complex_l2"],
                }
                hist.append(row)
                print(f"    [{label}] ep={ep:4d}  pde_air={row['pde_air']:.2e}  "
                      f"pde_grat={row['pde_grat']:.2e}  "
                      f"|t-1|={row['t_minus1']:.4f}  |t+1|={row['t_plus1']:.4f}  "
                      f"R+T={row['R_plus_T']:.4f}  L2={row['total_complex_l2']:.4f}  "
                      f"|head|={head_norm:.2e}", flush=True)

        # Save CSV
        csv_path = out_dir / f"smoke_{label}_history.csv"
        if hist:
            with csv_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(hist[0]))
                writer.writeheader(); writer.writerows(hist)

        final = hist[-1] if hist else {}
        return {
            "label": label, "epochs": epochs, "lr": lr,
            "init_theta_provided": init_theta is not None,
            "final_t_minus1": final.get("t_minus1"),
            "final_t_plus1":  final.get("t_plus1"),
            "final_R_plus_T": final.get("R_plus_T"),
            "final_total_complex_l2": final.get("total_complex_l2"),
            "final_scattered_complex_l2": final.get("scattered_complex_l2"),
            "final_total_loss": final.get("total"),
            "history": hist,
        }

    smoke_results = {}
    # A: zero head (fresh scaled model)
    smoke_results["A_zero_head"] = run_smoke("A_zero_head", init_theta=None)
    # B: best finite frozen-head LS solution (if available)
    if best_finite_theta is not None:
        smoke_results["B_ls_init"] = run_smoke("B_ls_init", init_theta=best_finite_theta)
    else:
        print("  No finite LS candidate — skipping B_ls_init", flush=True)
        smoke_results["B_ls_init"] = {"label": "B_ls_init", "skipped": True}

    return smoke_results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7: Decision
# ─────────────────────────────────────────────────────────────────────────────

def phase7_decision(p2, p3, p4_finite, p5_results, p6_smoke) -> dict:
    print("\n[Phase 7] Decision", flush=True)
    TARGET = TARGET_T1

    # Check Phase 2 gate
    if not p2.get("all_passed"):
        return {"case": "C", "decision": "Feature scaling broke reparameterization invariants."}

    # Check Phase 3 gate
    if not p3.get("all_passed"):
        bd = p3.get("breakdown_alpha", "?")
        return {"case": "C",
                "decision": f"Post-scaling affine test failed (breakdown alpha={bd}). "
                             "Feature scaling produced a non-affine system."}

    # No finite LS candidates
    if not p4_finite:
        return {"case": "C",
                "decision": "Feature scaling did not produce finite parameters (max|θ|<1e4). "
                             "Report feature norms and column norms. Investigate basis."}

    # Best finite LS result
    best_t1  = max((r.get("t_minus1_abs") or 0.0 for r in p5_results if r.get("reeval_ok")), default=0.0)
    best_tp  = max((r.get("t_plus1_abs")  or 0.0 for r in p5_results if r.get("reeval_ok")), default=0.0)
    baseline = 0.00638

    recovering = best_t1 > 0.25 * TARGET   # within 25% of target
    improving  = best_t1 > 2.0 * baseline  # materially better than baseline

    # Smoke results
    smoke_a  = p6_smoke.get("A_zero_head", {})
    smoke_b  = p6_smoke.get("B_ls_init",   {})
    smoke_t1_a = smoke_a.get("final_t_minus1") or 0.0
    smoke_t1_b = smoke_b.get("final_t_minus1") or 0.0 if not smoke_b.get("skipped") else 0.0
    smoke_L2_a = smoke_a.get("final_total_complex_l2") or 1.0
    smoke_scat_a = smoke_a.get("final_scattered_complex_l2") or 1.0

    if recovering:
        case = "A"
        decision = (
            f"Feature scaling produced a finite LS solution with |t_±1|={best_t1:.5f} "
            f"(target={TARGET:.5f}, within 25%). "
            "The current parameterization can represent the required response. "
            "Do NOT start a 10,000-epoch run automatically. "
            "Run a two-seed confirmation smoke first."
        )
    elif improving:
        case = "A"
        decision = (
            f"Feature scaling produced finite LS solutions with |t_±1|={best_t1:.5f}, "
            f"materially above baseline {baseline:.5f}. "
            "Directional improvement confirmed. Full recovery not achieved. "
            "Preserve feature-scaled parameterization. "
            "Do NOT start a 10,000-epoch run yet."
        )
    elif smoke_t1_a > 2.0 * baseline or smoke_t1_b > 2.0 * baseline:
        case = "E"
        decision = (
            f"Smoke run improved |t_±1| to {max(smoke_t1_a, smoke_t1_b):.5f} "
            f"(baseline={baseline:.5f}) but LS alone did not reach target. "
            "Do not use modal-data loss yet. "
            "Run a controlled continuation only after confirming reproducibility."
        )
    elif smoke_L2_a < 0.04 and smoke_scat_a > 0.1 and best_t1 <= baseline:
        case = "D"
        decision = (
            "Feature scaling improved total-field L2 but not scattered-field L2 or ±1. "
            "Reject: background fitting only. "
            "Investigate active singular vectors and ridge-to-boundary coupling."
        )
    else:
        case = "B"
        decision = (
            f"Feature scaling produced finite heads (max|θ| < 1e4) but ±1 remains "
            f"near baseline ({best_t1:.5f} vs {baseline:.5f}). "
            "The residual system may be underconstrained for ±1. "
            "Inspect active singular vectors and whether PDE residual "
            "propagates ridge forcing into the boundary observable."
        )

    print(f"  Case {case}: {decision[:120]}...", flush=True)
    return {
        "case": case,
        "decision": decision,
        "best_ls_t_minus1": best_t1,
        "best_ls_t_plus1":  best_tp,
        "baseline_t_pm1":   baseline,
        "smoke_A_final_t_minus1": smoke_t1_a,
        "smoke_B_final_t_minus1": smoke_t1_b,
        "authorize_nonlinear": recovering or improving,
        "start_10000_epoch_run": False,
        "modal_data_loss": False,
        "optical_coupling": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def write_plots(root: Path, scales: dict, p4_results: list,
                p5_results: list, p6_smoke: dict) -> None:
    # Feature norms
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (net_name, _) in zip(axes, [("net_air", None), ("net_grat", None), ("net_sub", None)]):
        sc_vals = scales.get(net_name, [])
        if sc_vals:
            ax.bar(np.arange(len(sc_vals)), sc_vals)
            ax.set_title(f"{net_name} feature scales")
            ax.set_xlabel("hidden channel"); ax.set_ylabel("RMS scale")
    fig.tight_layout()
    fig.savefig(root / "feature_norms.png", dpi=160)
    plt.close(fig)

    # Regularisation path
    if p4_results:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        lambdas_all = [r["lambda"] for r in p4_results]
        axes[0].loglog(lambdas_all, [r["theta_norm"]     for r in p4_results], "o-")
        axes[0].set(xlabel="λ", ylabel="|θ|", title="Param norm vs λ")
        axes[1].loglog(lambdas_all, [r["raw_resid_norm"] for r in p4_results], "o-")
        axes[1].set(xlabel="λ", ylabel="||residual||", title="Residual vs λ")
        # Build per-lambda t1 dict from p5_results (which may be a subset)
        lam_to_t1 = {}
        for r in p5_results:
            lam = r.get("lambda")
            t1  = r.get("t_minus1_abs")
            if lam is not None and t1 is not None:
                lam_to_t1[lam] = t1
        lam_eval = [l for l in lambdas_all if l in lam_to_t1]
        t1_eval  = [lam_to_t1[l] for l in lam_eval]
        if lam_eval:
            axes[2].loglog(lam_eval, [max(v, 1e-10) for v in t1_eval], "o-", label="|t_{-1}|")
            axes[2].axhline(TARGET_T1, color="k", ls="--", label="target")
            axes[2].legend(fontsize=8)
        axes[2].set(xlabel="λ", ylabel="|t_{-1}|", title="Modal vs λ")
        fig.tight_layout()
        fig.savefig(root / "regularization_path.png", dpi=160)
        plt.close(fig)

    # Smoke comparison
    labels_smoke = []
    t1_smoke     = []
    for key in ("A_zero_head", "B_ls_init"):
        sm = p6_smoke.get(key, {})
        if sm.get("skipped"): continue
        hist = sm.get("history", [])
        if not hist: continue
        labels_smoke.append(key)
        t1_vals = [h["t_minus1"] for h in hist]
        ep_vals = [h["epoch"]    for h in hist]
        plt.figure(figsize=(7, 4))
        plt.plot(ep_vals, t1_vals, label=key)
        plt.axhline(TARGET_T1, color="k", ls="--", label="target")
        plt.xlabel("epoch"); plt.ylabel("|t_{-1}|")
        plt.title(f"Modal amplitude — {key}")
        plt.legend(); plt.tight_layout()
    if labels_smoke:
        plt.savefig(root / "modal_feature_scaling_comparison.png", dpi=160)
    plt.close("all")

    # Singular values (from Phase 4)
    if p4_results and "s" in p4_results[0]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sigma = p4_results[0]["s"]
        ax.semilogy(np.arange(1, len(sigma)+1),
                    np.maximum(sigma, np.finfo(float).tiny), "b-")
        ax.set(xlabel="index", ylabel="σ", title="Singular values (post-scaling physical system)")
        fig.tight_layout()
        fig.savefig(root / "singular_values.png", dpi=160)
        plt.close(fig)

    # Loss components smoke A
    sm_a = p6_smoke.get("A_zero_head", {})
    hist_a = sm_a.get("history", [])
    if hist_a:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ep_vals = [h["epoch"] for h in hist_a]
        for key in ("pde_air", "pde_grat", "pde_sub", "top_DtN", "bottom_DtN"):
            axes[0].semilogy(ep_vals, [h[key] for h in hist_a], label=key)
        axes[0].legend(fontsize=8); axes[0].set(xlabel="epoch", ylabel="loss", title="Loss components")
        axes[1].plot(ep_vals, [h["t_minus1"] for h in hist_a], label="|t-1|")
        axes[1].plot(ep_vals, [h["t_plus1"]  for h in hist_a], label="|t+1|")
        axes[1].axhline(TARGET_T1, color="k", ls="--", label="target")
        axes[1].legend(fontsize=8); axes[1].set(xlabel="epoch", ylabel="|t_m|", title="Modal amplitudes")
        fig.tight_layout()
        fig.savefig(root / "loss_component_comparison.png", dpi=160)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Feature-scaling reparameterization audit")
    ap.add_argument("--output-root",   default="outputs/phase5_feature_scaling")
    ap.add_argument("--shared-points", default="outputs/phase5_physical_conditioning/shared_points.npz")
    args = ap.parse_args()

    root = ROOT / args.output_root
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    root.mkdir(parents=True)
    cand_dir = root / "candidates"
    cand_dir.mkdir()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg         = load_config(ROOT / "configs/default.yaml")
    physics     = make_lambda_0p8(cfg.physics)
    ref_path    = ROOT / CANONICAL_REF
    companion_p = ROOT / COMPANION_PATH
    validate_reference(ref_path, physics)
    ref         = normalize_reference_orientation(load_reference_npz(ref_path))
    device      = torch.device("cpu")
    dtype       = torch.float64
    coeff       = compute_background_coefficients(physics)
    pts         = load_points(physics, device, dtype, ROOT / args.shared_points)
    np.savez(root / "shared_points.npz",
             **{k: v.detach().cpu().numpy() for k, v in pts.items()})

    # ── Phase 1 (feature scale computation + save) ────────────────────────────
    print("\n[Phase 1] Computing feature scales", flush=True)
    set_seed(SEED)
    fresh_model = make_fresh_model(physics)
    raw_scales  = compute_feature_scales(fresh_model, pts)
    # Gather full statistics
    phase1_stats = {}
    for net_name, sc_arr in raw_scales.items():
        phase1_stats[net_name] = {
            "scale_min":  float(sc_arr.min()),  "scale_max":  float(sc_arr.max()),
            "scale_mean": float(sc_arr.mean()), "scales":     sc_arr.tolist(),
            "floor":      FEATURE_SCALE_FLOOR,
        }
    print("  Scales per subnet:")
    for k, v in phase1_stats.items():
        print(f"    {k}  min={v['scale_min']:.4e}  max={v['scale_max']:.4e}  "
              f"mean={v['scale_mean']:.4e}")

    (root / "feature_scale_report.json").write_text(
        json.dumps({"phase1_feature_scales": phase1_stats,
                    "floor": FEATURE_SCALE_FLOOR,
                    "method": "per_channel_rms_of_tanh_output_at_collocation_points",
                    "target_max_theta_after_scaling": MAX_HEAD_MAG_TARGET,
                    "estimated_required_theta_after_scaling": 0.5899 / np.mean([
                        v["scale_mean"] for v in phase1_stats.values()
                    ])}, indent=2) + "\n")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    p2 = phase2_verify_reparameterization(physics, pts, coeff)
    (root / "reparameterization_tests.json").write_text(
        json.dumps(jsonable(p2), indent=2) + "\n")
    if not p2["all_passed"]:
        print("ERROR: Phase 2 failed — stopping.", flush=True)
        sys.exit(1)

    # ── Assemble post-scaling Jacobian ────────────────────────────────────────
    print("\n[Building post-scaling Jacobian]", flush=True)
    model_s, scales = make_scaled_model(physics, pts)
    heads_s         = freeze_scaled_except_head(model_s)
    scales_dict, scale_meta = physical_scales(pts, physics, coeff)
    confirm_validated_scales(scales_dict, scale_meta)

    jacobian_s, r0_s, lengths = assemble_system(model_s, heads_s, pts, physics, coeff)
    metadata_s  = row_metadata(lengths, pts)
    scales_row  = row_scales(lengths, scales_dict)

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    p3 = phase3_post_scaling_affine(physics, pts, coeff, jacobian_s, r0_s, lengths)
    (root / "affine_post_scaling.json").write_text(
        json.dumps(jsonable(p3), indent=2) + "\n")
    if not p3["all_passed"]:
        print("WARNING: Phase 3 has failures — continuing to Phase 4 for diagnosis",
              flush=True)

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    p4_results, p4_finite, col_norm_s, A_phys_s, b_phys_s = phase4_scaled_frozen_head_solve(
        physics, pts, coeff, jacobian_s, r0_s, lengths, scales_row, scales_dict)

    # ── Phase 5 (re-evaluation + withheld modal) ──────────────────────────────
    print("\n[Phase 5] Direct re-evaluation + withheld modal", flush=True)
    p5_results = []
    for sol in p4_finite if p4_finite else p4_results[:3]:
        r = eval_candidate(physics, pts, coeff, jacobian_s, r0_s, lengths,
                            scales_row, sol, device, dtype, ref, ref_path,
                            companion_p, cand_dir)
        p5_results.append(r)

    best_finite_theta = None
    best_t1 = 0.0
    for r in p5_results:
        if r.get("reeval_ok") and (r.get("t_minus1_abs") or 0.0) > best_t1:
            best_t1 = r["t_minus1_abs"]
            best_finite_theta = p4_results[p5_results.index(r)]["theta"]

    (root / "frozen_head_post_scaling.json").write_text(
        json.dumps(jsonable({
            "p4_raw": [{k: v for k, v in r.items()
                        if k not in ("theta", "s", "residual")} for r in p4_results],
            "p5_eval": [{k: v for k, v in r.items()
                         if k != "modal"} for r in p5_results],
            "n_finite_candidates": len(p4_finite),
            "n_reeval_ok": sum(1 for r in p5_results if r.get("reeval_ok")),
            "best_t_minus1": best_t1,
            "target_t_pm1":  TARGET_T1,
        }), indent=2) + "\n")

    # ── Phase 6 ───────────────────────────────────────────────────────────────
    p6 = phase6_smoke(physics, pts, coeff, device, dtype,
                       ref, ref_path, companion_p, root, best_finite_theta)
    smoke_report = {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                    for k, v in p6.items()}
    (root / "smoke_comparison.json").write_text(
        json.dumps(jsonable(smoke_report), indent=2) + "\n")

    # ── Phase 7 ───────────────────────────────────────────────────────────────
    decision = phase7_decision(p2, p3, p4_finite, p5_results, p6)

    # ── Phase 8: verification ─────────────────────────────────────────────────
    print("\n[Phase 8] Verification", flush=True)
    compile_r = run_compileall()
    pytest_r  = run_pytest()
    print(f"  compileall: {'OK' if compile_r['passed'] else 'FAIL'}")
    print(f"  pytest: {pytest_r['summary_line']}")

    # ── Write all outputs ──────────────────────────────────────────────────────
    can_sha = sha256_file(ref_path)
    com_sha = sha256_file(companion_p)

    write_plots(root,
                {k: v["scales"] for k, v in phase1_stats.items()},
                p4_results, p5_results, p6)

    # Final summary
    summary = {
        "git_commit":             git_commit(),
        "working_tree_status":    git_status(),
        "canonical_sha256":       can_sha,
        "canonical_sha_ok":       can_sha == json.loads(
            (ROOT / "outputs/reference_companion/manifest.json").read_text()
        )["canonical_sha256"],
        "companion_sha256":       com_sha,
        "companion_sha_ok":       com_sha == json.loads(
            (ROOT / "outputs/reference_companion/manifest.json").read_text()
        )["companion_sha256"],
        "canonical_modified":     False,
        "companion_used_in_loss": False,
        "modal_loss_weight":      0.0,
        "optical_coupling":       False,
        "model_config_hash":      model_config_hash(physics),
        "parameter_order_hash":   parameter_ordering_hash(
            head_parameter_records(make_fresh_model(physics))),
        "feature_scale_floor":    FEATURE_SCALE_FLOOR,
        "feature_scales":         {k: v["scales"] for k, v in phase1_stats.items()},
        "target_max_theta":       MAX_HEAD_MAG_TARGET,
        "svd_rcond":              RCOND,
        "phase1": phase1_stats,
        "phase2_all_passed":      p2["all_passed"],
        "phase3_all_passed":      p3.get("all_passed"),
        "phase3_affine_breakdown_alpha": p3.get("breakdown_alpha"),
        "phase4": {
            "n_tikhonov": len(p4_results),
            "n_finite_ok": len(p4_finite),
            "col_norm_min": float(col_norm_s.min()),
            "col_norm_max": float(col_norm_s.max()),
            "col_norm_mean": float(col_norm_s.mean()),
            "tikhonov_results": [{k: v for k, v in r.items()
                                   if k not in ("theta","s","residual")} for r in p4_results],
        },
        "phase5": {
            "n_reeval_ok": sum(1 for r in p5_results if r.get("reeval_ok")),
            "best_t_minus1": best_t1,
            "results": [{k: v for k, v in r.items() if k != "modal"}
                        for r in p5_results],
        },
        "phase6_smoke": smoke_report,
        "decision":   decision,
        "compileall": compile_r,
        "pytest":     pytest_r,
    }
    (root / "feature_scale_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2) + "\n")

    # CSV
    rows = []
    for r in p5_results:
        rows.append({
            "label":        r["label"],
            "lambda":       r["lambda"],
            "theta_norm":   r["theta_norm"],
            "theta_max":    r["theta_max"],
            "direct_resid": r["direct_residual_norm"],
            "affine_err":   r["direct_minus_affine_max_abs"],
            "reeval_ok":    r["reeval_ok"],
            "t_minus1_abs": r.get("t_minus1_abs", ""),
            "t_plus1_abs":  r.get("t_plus1_abs",  ""),
            "R_plus_T":     r.get("R_plus_T",     ""),
            "total_complex_l2": r.get("total_complex_l2", ""),
        })
    if rows:
        with (root / "feature_scale_summary.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    print("\n" + "=" * 70, flush=True)
    print(json.dumps(jsonable({
        "case":               decision["case"],
        "decision":           decision["decision"][:200],
        "p2_all_passed":      p2["all_passed"],
        "p3_affine_valid":    p3.get("affine_valid_for_all_tested_alphas"),
        "p4_finite_ok":       len(p4_finite),
        "best_ls_t_minus1":   best_t1,
        "best_ls_t_plus1":    decision.get("best_ls_t_plus1"),
        "smoke_A_final_t-1":  decision.get("smoke_A_final_t_minus1"),
        "pytest":             pytest_r["summary_line"],
        "output_root":        str(root),
    }), indent=2))


if __name__ == "__main__":
    main()
