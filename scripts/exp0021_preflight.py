"""Engineering-only preflight: no research seeds or saved trained weights."""
from pathlib import Path
import json
import numpy as np
import torch
from neurotopo.learned_protected_pilot import LearnedMemory, generate, predict_with_gram, state_hash
from neurotopo.protected_interventions import intervention_rollout, perturbations
from neurotopo.run import source_fingerprints, write_json
from neurotopo.order_observation import sha

def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    data = generate(16,64,[209021,0,0,0],'UPDATE',np.pi,0.)
    rows=[]
    for arm in ('RAW','INV','REL'):
        model=LearnedMemory(arm,209021)
        z,g=predict_with_gram(model,data)
        intervention=intervention_rollout(model,data,209021)
        np.testing.assert_array_equal(z[:,32:],intervention['NONE']['logits'])
        np.testing.assert_array_equal(g[:,32:],intervention['NONE']['gram'])
        h=intervention['NONE']['before'];effects=perturbations(h,209021)
        for name in ('NORM_SHAM','ANGLE_SHAM'):
            v=effects[name]['after']
            np.testing.assert_allclose(v.swapaxes(1,2)@v,h.swapaxes(1,2)@h,atol=2e-6,rtol=0)
        rows.append({'arm':arm,'parameters':sum(p.numel() for p in model.parameters()),'encoder':state_hash(model,'encoder.'),'readout':state_hash(model,'readout.'),'none_max_error':0})
    assert len({r['encoder'] for r in rows})==len({r['readout'] for r in rows})==1
    write_json(Path('artifacts/validation/exp0021_preflight.json'),{'pass':True,'source_sha256':source_fingerprints(),'config_sha256':sha(Path('configs/exp0021_v1.json')),'rows':rows,'engineering_updates':0,'scope':'Reuse unchanged training/model, engineered interventions on untrained root209021 only; no research data loaded.'})
if __name__=='__main__':main()
