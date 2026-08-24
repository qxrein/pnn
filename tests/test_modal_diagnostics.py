from __future__ import annotations
import numpy as np
import torch
from src.config import PhysicsConfig
from src.maxwell_layered_bg import compute_background_coefficients, maxwell_2d_lbg_pde_residual
from src.mode_aware_fourier import ExplicitFourierModalDD
from src.modal_diagnostics import binary_ridge_mask_fourier, component_gradient_audit, projected_lbg_modal_loss, source_fourier_spectrum
from scripts.train_lbg import layered_bg_loss
from src.maxwell_2d_nondim import sample_nd_points

def test_corrected_ridge_source_has_nonzero_plus_minus_one_projection():
    p=PhysicsConfig(period=.8,ridge_width=.32); s=source_fourier_spectrum(p,compute_background_coefficients(p))
    index={m:i for i,m in enumerate(s['orders'])}
    assert np.max(s['magnitude'][:,index[-1]]) > 1e-5
    assert np.max(s['magnitude'][:,index[1]]) > 1e-5

def test_modal_coefficient_changes_reconstruction_and_grating_pde_loss():
    p=PhysicsConfig(period=.8,ridge_width=.32);model=ExplicitFourierModalDD(p,modal_order_max=1).double();coeff=compute_background_coefficients(p)
    x=torch.linspace(0,p.period,13,dtype=torch.float64);z=torch.full_like(x,1.3)
    initial=model.net_grat(x,z).detach().clone(); initial_loss=sum(v.square().mean() for v in maxwell_2d_lbg_pde_residual(model.net_grat,x,z,p,p.eps_ridge,coeff))
    # A z-dependent coefficient perturbation is required: a constant pure
    # homogeneous mode is in the null space of the uniform-medium PDE.
    with torch.no_grad(): model.net_grat.coefficient_mlp[-1].weight[4,0]=.1
    perturbed=model.net_grat(x,z).detach(); perturbed_loss=sum(v.square().mean() for v in maxwell_2d_lbg_pde_residual(model.net_grat,x,z,p,p.eps_ridge,coeff))
    assert not torch.allclose(initial,perturbed)
    assert not torch.allclose(initial_loss,perturbed_loss)

def test_forced_grating_pde_reaches_plus_minus_one_modal_heads():
    p=PhysicsConfig(period=.8,ridge_width=.32);model=ExplicitFourierModalDD(p,modal_order_max=1).double();coeff=compute_background_coefficients(p)
    pts=sample_nd_points(p,64,16,16,16,torch.device('cpu'),torch.float64,seed=42)
    losses=layered_bg_loss(model,pts,p,coeff,1,500,500,200,100,use_dtn=True,w_modal=0)
    audit=component_gradient_audit(model,{'grating':losses['pde_grat']})['grating']['grat']
    assert audit['-1'] > 1e-8
    assert audit['1'] > 1e-8

def test_projected_residual_is_differentiable_for_modal_network():
    p=PhysicsConfig(period=.8,ridge_width=.32);model=ExplicitFourierModalDD(p,modal_order_max=1).double(); loss=projected_lbg_modal_loss(model.net_grat,torch.linspace(1.22,1.38,5,dtype=torch.float64),p,compute_background_coefficients(p),1)
    loss.backward()
    assert model.net_grat.coefficient_mlp[-1].weight.grad is not None

def test_binary_ridge_fft_projection_matches_analytic_mask_coefficients():
    fill=.4; orders=np.arange(-11,12); x=np.linspace(0.,1.,4096,endpoint=False); mask=(np.abs(x-.5)<=fill/2).astype(float)
    numeric=np.array([np.mean(mask*np.exp(-2j*np.pi*m*x)) for m in orders])
    np.testing.assert_allclose(numeric,binary_ridge_mask_fourier(orders,fill),atol=5e-4)
