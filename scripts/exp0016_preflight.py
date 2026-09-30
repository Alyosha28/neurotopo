"""One bounded engineering overfit probe on a nonresearch stream; not control qualification."""
from pathlib import Path
import json,time
import numpy as np
import torch
from torch.nn import functional as F
from neurotopo.orbit_control_pilot import build_model,sha
from neurotopo.orbit_information import generate,posterior

def main():
    path=Path('artifacts/validation/exp0016_preflight.json')
    if path.exists():raise FileExistsError('Keep the previous engineering attempt')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    y=generate(64,32,19016)['y'];q=posterior(y)
    x=F.one_hot(torch.from_numpy(y.astype(np.int64)),16).float();target=torch.from_numpy(q).float()
    started=time.monotonic();records=[]
    # Faster lr only for tiny-overfit engineering; research lr is frozen at .001.
    for arm in ['H','U','D']:
        model=build_model(arm,19016);opt=torch.optim.Adam(model.parameters(),lr=.05)
        initial=float((-(target*model(x)).sum(-1).mean()).detach())
        for step in range(100):
            opt.zero_grad(set_to_none=True);loss=-(target*model(x)).sum(-1).mean();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        final=float((-(target*model(x)).sum(-1).mean()).detach())
        records.append(dict(arm=arm,updates=100,engineering_lr=.05,initial=initial,final=final,passed=final<=.8*initial))
    report=dict(pass_=all(r['passed'] for r in records),scope='ENGINEERING_ONLY',root=19016,updates=300,records=records,seconds=time.monotonic()-started,
                model_code_sha256=sha(Path('src/neurotopo/orbit_control_pilot.py')),config_sha256=sha(Path('configs/exp0016_v1.json')),research_roots_called=False)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['pass_']:raise SystemExit(1)

if __name__=='__main__':main()
