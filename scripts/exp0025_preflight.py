from pathlib import Path
import numpy as np
import torch
from neurotopo.learned_protected_pilot import LearnedMemory,generate,predict_with_gram,state_hash
from neurotopo.state_refresh_core import fit_prototypes
from neurotopo.refresh_factor_core import refresh_rollout
from neurotopo.refresh_replication_analysis import refresh_diagnostics
from neurotopo.run import source_fingerprints,write_json
from neurotopo.order_observation import sha

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);rows=[]
    cal=generate(32,8,[259025,4],'MIX');data=generate(32,16,[259025,5],'UPDATE',np.pi,0.)
    for arm in ('REL','INV'):
        m=LearnedMemory(arm,259025);before=state_hash(m);p,s=fit_prototypes(m,cal);z,g=predict_with_gram(m,data)
        for condition in ('NONE','GAUGE8','FULL8','EQUAL8','SCALE8'):
            raw=refresh_rollout(m,data,condition,p['prototypes']);diag=refresh_diagnostics(raw,condition)
            if condition=='NONE':np.testing.assert_array_equal(raw['logits'],z);np.testing.assert_array_equal(raw['gram'],g)
            if condition=='GAUGE8':assert diag['local_logit_abs_max']<=1e-5 and diag['local_gram_abs_max']<=1e-5
            rows.append({'arm':arm,'condition':condition,'diagnostics':diag})
        assert state_hash(m)==before
    write_json(Path('artifacts/validation/exp0025_preflight.json'),{'pass':True,'source_sha256':source_fingerprints(),'config_sha256':sha(Path('configs/exp0025_v2.json')),'rows':rows,'optimizer_updates':0,'scope':'Untrained engineering root259025 only; prototype fitting on engineering inputs, not research models.'})
if __name__=='__main__':main()
