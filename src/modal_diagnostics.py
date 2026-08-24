"""Source spectra and gradient audits for explicit Fourier-modal PINNs."""
from __future__ import annotations
import numpy as np
import torch
from src.maxwell_layered_bg import background_field_np, delta_eps_np


def source_fourier_spectrum(physics, coeff, orders=range(-11, 12), nx: int = 512, nz: int = 64):
    """Project S=(eps-eps_bg)E_bg onto periodic x Fourier orders in the ridge."""
    x=np.linspace(0.,physics.period,nx,endpoint=False)
    z=np.linspace(physics.ridge_z_min+1e-8,physics.ridge_z_max-1e-8,nz)
    X,Z=np.meshgrid(x,z,indexing="xy")
    delta=delta_eps_np(X.copy(),Z.copy(),physics)
    br,bi,_,_=background_field_np(Z.ravel(),coeff); source=delta*(br.reshape(Z.shape)+1j*bi.reshape(Z.shape))
    orders=np.asarray(list(orders),int); g0=2*np.pi/physics.period
    coefficients=np.asarray([np.mean(source*np.exp(-1j*m*g0*X),axis=1) for m in orders]).T
    return {"x":x,"z":z,"source":source,"orders":orders,"coefficients":coefficients,
            "magnitude":np.abs(coefficients)}


def modal_parameter_gradient_norms(model):
    """Per-order gradient norms for each subnet's constant/modal-head parameters."""
    result={}
    for region in ("air","grat","sub"):
        net=getattr(model,f"net_{region}"); vals=[]
        for i,m in enumerate(net.orders):
            terms=[]
            if net.coefficients.grad is not None: terms.append(net.coefficients.grad[i].square().sum())
            if net.coefficient_mlp is not None:
                last=net.coefficient_mlp[-1]
                if last.weight.grad is not None:
                    terms.append(last.weight.grad[2*i:2*i+2].square().sum())
                if last.bias.grad is not None: terms.append(last.bias.grad[2*i:2*i+2].square().sum())
            vals.append(float(torch.sqrt(sum(terms)).detach()) if terms else 0.0)
        result[region]={str(m):v for m,v in zip(net.orders,vals)}
    return result


def component_gradient_audit(model, loss_components: dict):
    """Backprop each scalar loss independently and report every retained mode."""
    result={}
    for name,loss in loss_components.items():
        model.zero_grad(set_to_none=True); loss.backward(retain_graph=True)
        result[name]=modal_parameter_gradient_norms(model)
    model.zero_grad(set_to_none=True)
    return result


def projected_lbg_modal_loss(subnet, z_points, physics, coeff, n_orders: int,
                              return_by_mode: bool = False):
    """Optional Galerkin residual, explicitly distinct from pointwise loss.

    This numerically projects every first-order residual onto retained Fourier
    orders and, crucially, projects the *spatial* material product eps(x,z)E.
    It is not mixed into production training unless a caller elects to do so.
    """
    from src.geometry import epsilon_r
    from src.maxwell_layered_bg import background_field_torch, delta_eps_tensor
    nx=max(4*n_orders+4,32); x0=torch.linspace(0.,physics.period,nx+1,dtype=z_points.dtype,device=z_points.device)[:-1]
    x=x0.repeat(len(z_points)).detach().clone().requires_grad_(True); z=torch.repeat_interleave(z_points,nx).detach().clone().requires_grad_(True)
    Er,Ei,Hr,Hi,Hrz,Hiz=subnet.field_components(x,z); one=torch.ones_like(Er)
    def g(f,v):
        q=torch.autograd.grad(f,v,one,create_graph=True,retain_graph=True,allow_unused=True)[0]
        return q if q is not None else torch.zeros_like(v)
    k0=physics.k0; br,bi,_,_=background_field_torch(z.detach(),coeff,physics);eps=epsilon_r(x.detach(),z.detach(),physics);delta=delta_eps_tensor(x.detach(),z.detach(),physics)
    residuals=(g(Er,z)/k0-Hi,g(Ei,z)/k0+Hr,g(Er,x)/k0+Hiz,g(Ei,x)/k0-Hrz,g(Hr,z)/k0-g(Hrz,x)/k0-eps*Ei-delta*bi,g(Hi,z)/k0-g(Hiz,x)/k0+eps*Er+delta*br)
    loss=torch.zeros((),dtype=x.dtype,device=x.device); by_mode={}
    for r in residuals:
        grid=r.reshape(len(z_points),nx)
        for m in range(-n_orders,n_orders+1):
            # Real residual projections: sin/cos are the real modal basis.
            phase=2*np.pi*m*x0/physics.period
            term=torch.mean(torch.mean(grid*torch.cos(phase),dim=1)**2+torch.mean(grid*torch.sin(phase),dim=1)**2)
            loss=loss+term; by_mode[m]=by_mode.get(m,torch.zeros_like(term))+term/len(residuals)
    loss=loss/((2*n_orders+1)*len(residuals))
    if return_by_mode:
        return loss,{m:v/len(residuals) for m,v in by_mode.items()}
    return loss


def binary_ridge_mask_fourier(orders, fill_fraction: float) -> np.ndarray:
    """Analytic coefficients of a centred rectangular ridge mask on [0,L]."""
    m=np.asarray(orders,dtype=int); result=np.empty(len(m),complex)
    zero=m==0; result[zero]=fill_fraction
    result[~zero]=((-1.)**m[~zero])*np.sin(np.pi*m[~zero]*fill_fraction)/(np.pi*m[~zero])
    return result
