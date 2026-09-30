from pathlib import Path
import json
import numpy as np
import torch
from neurotopo.schedule_core import MAIN,EXTRA,rollout,threshold
from neurotopo.schedule_analysis import summarize
from neurotopo.learned_protected_pilot import LearnedMemory,generate,state_hash
from neurotopo.state_refresh_core import fit_prototypes
from neurotopo.refresh_factor_core import refresh_rollout
from neurotopo.run import source_fingerprints,write_json
from neurotopo.order_observation import sha

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cal=generate(128,8,[269026,11],'MIX');data=generate(128,64,[269026,12],'UPDATE',np.pi,0.);rows=[]
    for arm in ('REL','INV'):
        model=LearnedMemory(arm,269026);before=state_hash(model);p,_=fit_prototypes(model,cal);theta=threshold(p)
        for rotation in ('ORIGINAL','IDENTITY'):
            counts=None
            for c in MAIN+EXTRA if rotation=='ORIGINAL' else MAIN:
                raw=rollout(model,data,p['prototypes'],c,rotation,theta,counts if c=='COUNT90' else None,c=='NONE')
                metric=summarize(raw,data,c,theta)
                if c=='ADAPT90':counts=raw['attempt_counts']
                if rotation=='ORIGINAL' and c in ('NONE','K8','FULL8','SCALE8','GAUGE8'):
                    old=refresh_rollout(model,data,'EQUAL8' if c=='K8' else c,p['prototypes'])
                    np.testing.assert_array_equal(raw['logits'],old['logits'])
                rows.append({'arm':arm,'rotation':rotation,'condition':c,'attempts':metric['metrics']['attempts'],'valid':metric['metrics']['valid']})
        assert before==state_hash(model)
    write_json(Path('artifacts/validation/exp0026_preflight.json'),{'pass':True,'source_sha256':source_fingerprints(),'config_sha256':sha(Path('configs/exp0026_v2.json')),'engineering_seed':269026,'optimizer_updates':0,'scope':'Engineering fixtures only, all schedules and same-shape exact every8 replay.','rows':rows})
    print('EXP0026 engineering preflight PASS')
if __name__=='__main__':main()
