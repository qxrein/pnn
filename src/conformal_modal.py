"""Geometry-conforming five-region explicit-modal decomposition."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from src.mode_aware_fourier import ExplicitFourierModalNetwork

REGIONS=("top_air","left_substrate","ridge","right_substrate","lower_substrate")

class ConformalExplicitModalDD(nn.Module):
    def __init__(self, physics, modal_order_max=3, use_coefficient_mlp=True):
        super().__init__();self.physics=physics;self.modal_order_max=modal_order_max
        orders=tuple(range(-modal_order_max,modal_order_max+1));kw=dict(orders=orders,period=physics.period,k0=physics.k0,use_coefficient_mlp=use_coefficient_mlp)
        self.top_air=ExplicitFourierModalNetwork(n=physics.n_air,z_lo=0.,z_hi=physics.ridge_z_min,**kw)
        self.left_substrate=ExplicitFourierModalNetwork(n=physics.n_substrate,z_lo=physics.ridge_z_min,z_hi=physics.ridge_z_max,**kw)
        self.ridge=ExplicitFourierModalNetwork(n=physics.n_ridge,z_lo=physics.ridge_z_min,z_hi=physics.ridge_z_max,**kw)
        self.right_substrate=ExplicitFourierModalNetwork(n=physics.n_substrate,z_lo=physics.ridge_z_min,z_hi=physics.ridge_z_max,**kw)
        self.lower_substrate=ExplicitFourierModalNetwork(n=physics.n_substrate,z_lo=physics.ridge_z_max,z_hi=physics.domain_height,**kw)
    def masks(self,x,z):
        p=self.physics;top=z<p.ridge_z_min;mid=(z>=p.ridge_z_min)&(z<=p.ridge_z_max)
        left=mid&(x<p.ridge_x_min);ridge=mid&(x>=p.ridge_x_min)&(x<=p.ridge_x_max);right=mid&(x>p.ridge_x_max);lower=z>p.ridge_z_max
        return dict(zip(REGIONS,(top,left,ridge,right,lower)))
    def assert_partition(self,x,z):
        masks=self.masks(x,z);count=sum(v.to(torch.int8) for v in masks.values());assert torch.all(count==1),'domain overlap/gap';return masks
    def field(self,x,z):
        out=torch.zeros((len(x),6),dtype=x.dtype,device=x.device);m=self.assert_partition(x,z)
        for name in REGIONS:
            if m[name].any():out[m[name]]=getattr(self,name)(x[m[name]],z[m[name]])
        return out

def sample_conformal_points(p,n,ni,nb,device,dtype,seed=42,margin=5e-3):
    rng=np.random.default_rng(seed);t=lambda v:torch.as_tensor(v,dtype=dtype,device=device);r=lambda a,b,k:rng.uniform(a,b,k)
    zl,zh=p.ridge_z_min,p.ridge_z_max;xl,xr=p.ridge_x_min,p.ridge_x_max
    return {'top_air':(t(r(0,p.period,n)),t(r(margin,zl-margin,n))),'left_substrate':(t(r(0,xl-margin,n)),t(r(zl+margin,zh-margin,n))),'ridge':(t(r(xl+margin,xr-margin,n)),t(r(zl+margin,zh-margin,n))),'right_substrate':(t(r(xr+margin,p.period,n)),t(r(zl+margin,zh-margin,n))),'lower_substrate':(t(r(0,p.period,n)),t(r(zh+margin,p.domain_height-margin,n))),'x_top':t(r(0,p.period,nb)),'x_bottom':t(r(0,p.period,nb)),'x_ridge':t(r(xl+margin,xr-margin,ni)),'x_left':t(r(0,xl-margin,ni)),'x_right':t(r(xr+margin,p.period,ni)),'z_left_edge':t(r(zl+margin,zh-margin,ni)),'z_right_edge':t(r(zl+margin,zh-margin,ni)),'corners':t(np.array([[xl,zl],[xr,zl],[xl,zh],[xr,zh]]))}

def trace_loss(a,b,x,z,component):
    oa=a(x,z);ob=b(x,z);idx=(0,1,2,3) if component=='hx' else (0,1,4,5)
    return sum(torch.mean((oa[:,i]-ob[:,i])**2) for i in idx)
