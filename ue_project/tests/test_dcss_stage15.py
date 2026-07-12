import os, tempfile
import torch
from scripts.dcss_stage15_audit import audit_sets
from dcss.resume import build_resume_run_dir, diagnostic_gate
from dcss.stage1 import projected_coefficient_metrics
from dcss.stage15 import constrained_direction, fixed_batch_order, gradient_component_stats, gradient_cosine, object_aligned_warp

def test_dataset_overlap_audit():
    assert audit_sets([],[],["a"],["b"])["conclusion"]=="clean checkpoint independent"
    assert audit_sets(None,None,["a"],["b"])["conclusion"]=="metadata insufficient"

def test_equal_budget_batch_order():
    assert fixed_batch_order(100,0)==fixed_batch_order(100,0) and fixed_batch_order(100,0)!=fixed_batch_order(100,1)

def test_projected_coefficient_metrics_stage15():
    result=projected_coefficient_metrics(torch.randn(10,4),torch.eye(4)[:,:2]); assert result["projected_coefficient_norm_cv"]>=0

def test_gradient_component_norms(): assert gradient_component_stats(torch.tensor([3.,4.]))["l2"]==5
def test_gradient_conflict_cosine(): assert gradient_cosine(torch.tensor([1.]),torch.tensor([-1.])) < 0

def test_constrained_direction_synthetic():
    direction,_=constrained_direction(torch.tensor([1.,1.]),[torch.tensor([1.,0.])]); assert direction[0] <= 1e-6

def test_constrained_direction_feasibility():
    d,s=constrained_direction(torch.tensor([1.,-1.]),[torch.tensor([1.,0.])]); assert s["status"]=="feasible" and d[0]<=1e-6

def test_constrained_no_silent_fallback_stage15():
    try: constrained_direction(torch.tensor([1.]),[torch.tensor([1.])],max_iterations=0)
    except RuntimeError: return
    raise AssertionError("silent fallback")

def _warp():
    return object_aligned_warp(torch.ones(3,8,8),[{"cls":14,"bbox":[.5,.5,.5,.5]},{"cls":1,"bbox":[.5,.5,.1,.1]}],32,14,1)
def test_object_aligned_warp(): assert _warp()[0].shape==(3,32,32)
def test_non_target_overlap_exclusion():
    canvas,_,nt,_=_warp(); assert float((canvas*nt).abs().max())==0
def test_valid_support_area(): assert 0 < _warp()[3]["valid_support_area"] < 1

def test_stage15_artifact_isolation():
    with tempfile.TemporaryDirectory() as root: assert build_resume_run_dir(root,"M0","x").startswith(os.path.abspath(root))

def test_victim_initialization_identity():
    torch.manual_seed(0); first=torch.nn.Linear(3,2).state_dict(); torch.manual_seed(0); second=torch.nn.Linear(3,2).state_dict(); assert all(torch.equal(first[k],second[k]) for k in first)

def test_gate_calculation_stage15():
    result=diagnostic_gate({"target_unit_coverage":.5,"target_projected_energy":.5,"non_target_leakage":.1,"R_shift":2.,"budget_consistent":True}); assert result["pass"]
