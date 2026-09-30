"""Engineering-only geometry/continuation receipt, zero training updates."""
from pathlib import Path
import time
import numpy as np
import torch
from neurotopo.equal_output_geometry import candidate_bank,prefix_states,suffix_data,rollout,select_candidates
from neurotopo.learned_protected_pilot import LearnedMemory,generate,state_hash
from neurotopo.run import source_fingerprints,write_json
from neurotopo.order_observation import sha

def main():
    start=time.monotonic();torch.set_num_threads(1);rows=[]
    p=generate(16,32,[229022,1],'UPDATE',np.pi,0.)
    search=suffix_data(229022,p['targets'][:,-1],4,229023)
    for arm in ('RAW','INV','REL'):
        m=LearnedMemory(arm,229022);before=state_hash(m);h=prefix_states(m,p)
        for r in (.01,.05,.15):
            bank=candidate_bank(m,h,229022,r)
            assert bank['valid'].all() and bank['sham_valid'].all()
            z,g=rollout(m,bank['states'],search);idx,scores=select_candidates(z,search['targets'],bank['valid'])
            assert np.isfinite(z).all() and np.isfinite(scores).all()
            rows.append({'arm':arm,'radius':r,'max_logit_error':float(bank['logit_error'].max()),'max_radius_error':float(bank['radius_error'].max()),'valid_fraction':float(bank['valid'].mean())})
        assert state_hash(m)==before
    write_json(Path('artifacts/validation/exp0022_preflight.json'),{'pass':True,'source_sha256':source_fingerprints(),'config_sha256':sha(Path('configs/exp0022_v2.json')),'rows':rows,'seconds':time.monotonic()-start,'training_updates':0,'scope':'Untrained engineering seed229022 only, no research weights/data.'})
if __name__=='__main__':main()
