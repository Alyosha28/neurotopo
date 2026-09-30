"""150-update T1 engineering fit; fixed root209002, not pilot data."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from neurotopo.learned_protected_pilot import LearnedMemory,radial,sha,tensors
from neurotopo.frozen_binary_probe import save_json


def main():
    path=Path('artifacts/validation/exp0020_v2_preflight.json')
    if path.exists():raise FileExistsError(path)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    b0=np.tile([0,0,1,1],4).astype(np.int64);u=np.tile([0,1,0,1],4).astype(np.int64)[:,None]
    d=dict(b0=b0,commands=u,common=np.tile(np.eye(3,dtype=np.float32),(16,2,1,1)),leak=np.tile(np.eye(3,dtype=np.float32),(16,1,1,1)),targets=np.column_stack((b0,(b0+u[:,0])%2)).astype(np.float32))
    rows=[];start=time.monotonic()
    for arm in ['RAW','INV','REL']:
        model=LearnedMemory(arm,209002);opt=torch.optim.Adam(model.parameters(),lr=.01)
        target=torch.from_numpy(d['targets']);initial=float(torch.nn.functional.binary_cross_entropy_with_logits(model(*tensors(d)),target).detach())
        for _ in range(50):
            opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(model(*tensors(d)),target);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        final=float(torch.nn.functional.binary_cross_entropy_with_logits(model(*tensors(d)),target).detach())
        rows.append(dict(arm=arm,initial=initial,final=final,updates=50,pass_=final<=.9*initial))
    result={'pass':all(r['pass_'] for r in rows),'scope':'ENGINEERING_T1_ONLY','research_roots_called':False,'engineering_updates':150,'records':rows,'seconds':time.monotonic()-start,'model_source_sha256':sha(Path('src/neurotopo/learned_protected_pilot.py')),'config_sha256':sha(Path('configs/exp0020_v2.json'))}
    save_json(path,result);print(json.dumps(result,indent=2))
    if not result['pass']:raise SystemExit(1)


if __name__=='__main__':main()
