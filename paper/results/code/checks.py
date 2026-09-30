"""Focused checks of replay, unseen-evidence isolation, stopping and budgets."""
from pathlib import Path
import torch
from bare import DEVICE, Fusion, Utility, policy_features, predict_cached, parse_sex, save_json

def main():
    torch.set_num_threads(4);torch.manual_seed(2026)
    fusion=Fusion().to(DEVICE).eval();utility=Utility().to(DEVICE).eval()
    z=torch.randn(4,10,128,device=DEVICE);sex=torch.tensor([0,1,0,1],device=DEVICE)
    desc=torch.randn(4,10,3,device=DEVICE);checks={}
    with torch.no_grad():
        for ids in [[0],[0,2],[0,2,7]]:
            selected=torch.tensor(ids,device=DEVICE).expand(4,-1)
            _,_,_,features=policy_features(fusion,z,sex,desc,selected)
            unseen=[j for j in range(10) if j not in ids]
            altered=z.clone();altered[:,unseen]=torch.randn_like(altered[:,unseen])*100
            _,_,_,features2=policy_features(fusion,altered,sex,desc,selected)
            error=float((features-features2).abs().max())
            assert error==0, 'Unseen encoded evidence leaked into policy'
            assert features.shape==(4,9,395)
            checks['unseen_isolation_'+str(ids)]=error
        cache=dict(z=z.cpu(),sex=sex.cpu(),desc=desc.cpu(),y=torch.tensor([12.,60.,120.,180.]),id=torch.arange(4))
        policy=(utility,torch.zeros(395,device=DEVICE),torch.ones(395,device=DEVICE))
        for budget in [0,1,2,3]:
            p=predict_cached(fusion,cache,policy=policy,budget=budget)
            assert all(p.crops==budget)
            for selected in p.selected:
                ids=[] if selected=='' else selected.split(',');assert len(ids)==len(set(ids))
            checks['budget_'+str(budget)]=True
        stopped=predict_cached(fusion,cache,policy=policy,tau=float('inf'))
        assert all(stopped.crops==0)
        globalp=predict_cached(fusion,cache,mode='global')
        assert (stopped.pred-globalp.pred).abs().max()==0
        checks['immediate_stop_equals_global']=True
    assert parse_sex(' FALSE ')==0 and parse_sex('TrUe')==1
    try:parse_sex('unknown')
    except ValueError:checks['invalid_sex_rejected']=True
    else:raise AssertionError('Unexpected sex flag accepted')
    save_json(Path(__file__).parent/'runs'/'overnight'/'implementation_checks.json',checks)
    print(checks,flush=True)

if __name__=='__main__':main()
