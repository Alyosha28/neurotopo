from pathlib import Path
import numpy as np
import torch
from neurotopo.learned_protected_pilot import LearnedMemory,generate,predict_with_gram,state_hash
from neurotopo.state_refresh_core import fit_prototypes,refresh_rollout
from neurotopo.state_refresh_analysis import refresh_diagnostics
from neurotopo.run import source_fingerprints,write_json
from neurotopo.order_observation import sha

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);rows=[]
    cal=generate(32,8,[249024,4],'MIX');data=generate(32,16,[249024,5],'UPDATE',np.pi,0.)
    for arm in ('REL','INV'):
        m=LearnedMemory(arm,249024);before=state_hash(m);p,s=fit_prototypes(m,cal);z,g=predict_with_gram(m,data)
        for condition in ('NONE','GAUGE8','RESET8','RESET1'):
            raw=refresh_rollout(m,data,condition,p['prototypes']);diag=refresh_diagnostics(raw,condition)
            if condition=='NONE':np.testing.assert_array_equal(raw['logits'],z);np.testing.assert_array_equal(raw['gram'],g)
            if condition=='GAUGE8':assert diag['local_logit_abs_max']<=1e-5 and diag['local_gram_abs_max']<=1e-5
            rows.append({'arm':arm,'condition':condition,'diagnostics':diag})
        assert state_hash(m)==before
    write_json(Path('artifacts/validation/exp0024_preflight.json'),{'pass':True,'source_sha256':source_fingerprints(),'config_sha256':sha(Path('configs/exp0024_v1.json')),'rows':rows,'optimizer_updates':0,'scope':'Untrained engineering root249024 only; prototype fitting on engineering inputs, not research models.'})
if __name__=='__main__':main()
