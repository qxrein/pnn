import torch
from src.config import PhysicsConfig
from src.conformal_modal import ConformalExplicitModalDD, sample_conformal_points
from src.geometry import epsilon_r

def test_conformal_regions_partition_and_match_canonical_materials():
    p=PhysicsConfig(period=.8,ridge_width=.32);m=ConformalExplicitModalDD(p,1).double()
    x=torch.tensor([.1,.1,.4,.7,.4],dtype=torch.float64);z=torch.tensor([.5,1.3,1.3,1.3,1.6],dtype=torch.float64);masks=m.assert_partition(x,z)
    assert [next(k for k,v in masks.items() if v[i]) for i in range(5)]==['top_air','left_substrate','ridge','right_substrate','lower_substrate']
    eps=epsilon_r(x,z,p);assert torch.allclose(eps,torch.tensor([p.eps_air,p.eps_substrate,p.eps_ridge,p.eps_substrate,p.eps_substrate],dtype=torch.float64))

def test_conformal_samples_are_exclusive_and_exclude_interfaces():
    p=PhysicsConfig(period=.8,ridge_width=.32);pts=sample_conformal_points(p,8,4,4,torch.device('cpu'),torch.float64)
    m=ConformalExplicitModalDD(p,1).double()
    for region in ('top_air','left_substrate','ridge','right_substrate','lower_substrate'):
        x,z=pts[region];masks=m.assert_partition(x,z);assert masks[region].all()
    assert pts['corners'].shape==(4,2)
