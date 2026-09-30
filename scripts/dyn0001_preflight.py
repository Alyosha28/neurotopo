"""Engineering-only 259025 smoke test for DYN-0001 branch and Gram tangent."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import numpy as np
import torch
from neurotopo.learned_protected_pilot import LearnedMemory,generate,state_hash
from neurotopo.state_refresh_core import fit_prototypes
from neurotopo.refresh_factor_core import CONDITIONS,refresh_rollout
from neurotopo.equal_logit_future import _one,_extend
from neurotopo.run import source_fingerprints

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows=[]
    with TemporaryDirectory(prefix='dyn0001-engineering-') as scratch:
        parent=Path(scratch);original=generate(512,256,[259025,8,1],'HOLD',np.pi,0.)
        data=_extend(original,259025,'LONG_H')
        calibration=generate(512,8,[259025,8,2],'MIX',.4,0.)
        for arm in ('REL','INV'):
            model=LearnedMemory(arm,259025);before=state_hash(model)
            proto,_=fit_prototypes(model,calibration)
            for condition in CONDITIONS:
                raw=refresh_rollout(model,original,condition,proto['prototypes'])
                path=parent/'259025'/arm/'test'/condition/'LONG_H.npz';path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,**raw)
            result,arrays=_one(model,data,proto['prototypes'],parent,259025,arm,'LONG_H')
            assert arrays['logits'].shape==(5,512,1017)
            assert result['suffix_transitions']==5*512*1016
            assert state_hash(model)==before
            rows.append({'arm':arm,'equal_valid':result['conditions']['EQUAL8']['valid_count'],
                         'tangent_available':result['tangent']['available_count'],
                         'local_current_max':result['tangent']['current_tangent_abs_max'],
                         'baseline_max_abs_logit_error':result['baseline_max_abs_logit_error']})
    receipt={'pass':True,'engineering_root':259025,'model_arms':['REL','INV'],'research_models':0,
             'optimizer_updates':0,'source_sha256':source_fingerprints(),'rows':rows}
    out=Path('artifacts/validation/dyn0001_preflight.json')
    out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
