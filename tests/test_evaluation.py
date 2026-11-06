import numpy as np

from src.metrics import binary_metrics
from src.probe import aggregate


def test_density_uses_same_topologies_as_ap():
    rows=[]
    for variant,label in (("pair_only",[0,0,0,0]),("mixed_a",[1,1,0,0])):
        rows.append(dict(run_id="unit",stage="untrained",method="fused_logit",target="projected-triad",
            family_id=0,topology_id=variant,variant=variant,**binary_metrics(label,[.8,.7,.2,.1])))
    topology,summary=aggregate(rows,samples=20)
    overall=next(r for r in summary if r["variant"]=="all")
    assert overall["metrics"]["density"]["mean"]==.5
    assert overall["metrics"]["density"]["n_topologies"]==1
    empty=next(r for r in summary if r["variant"]=="pair_only")
    assert np.isnan(empty["metrics"]["ap"]["mean"])
    assert next(r for r in topology if r["variant"]=="pair_only")["density"]==0
