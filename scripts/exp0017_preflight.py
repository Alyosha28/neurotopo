"""Bounded engineering overfit and hardware timing; no scientific roots."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from neurotopo.order_observation import ControlGRU, generate, input_tensor, sha, CELLS, numeric_code_hash
from neurotopo.run import write_json, source_fingerprints


def main():
    path=Path('artifacts/validation/exp0017_preflight.json')
    if path.exists():raise FileExistsError(path)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    batches=[generate(4,8,c,[179001,0,i,0]) for i,c in enumerate(CELLS)]
    x=input_tensor(np.concatenate([d['tokens'] for d in batches]))
    q=torch.from_numpy(np.concatenate([d['target'] for d in batches])).float()
    rows=[]
    for arm in ['D16','D64']:
        model=ControlGRU(arm,179001);opt=torch.optim.Adam(model.parameters(),lr=.03)
        initial=float((-(q*F.log_softmax(model(x),-1)).sum(-1).mean()).detach())
        for _ in range(100):
            opt.zero_grad(set_to_none=True);loss=-(q*F.log_softmax(model(x),-1)).sum(-1).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        final=float((-(q*F.log_softmax(model(x),-1)).sum(-1).mean()).detach())
        # Fixed dummy timing data; no experiment task or labels involved.
        dummy=F.one_hot(torch.arange(64*40).reshape(64,40)%5,5).float()
        start=time.monotonic()
        for _ in range(10):
            opt.zero_grad(set_to_none=True);loss=model(dummy).square().mean();loss.backward();opt.step()
        step_seconds=(time.monotonic()-start)/10
        start=time.monotonic()
        with torch.no_grad():model(F.one_hot(torch.arange(128*160).reshape(128,160)%5,5).float())
        eval_batch_seconds=time.monotonic()-start
        rows.append(dict(arm=arm,initial=initial,final=final,overfit_pass=final<=.8*initial,step_seconds=step_seconds,eval_batch_seconds=eval_batch_seconds,engineering_updates=110))
    # Factor two allowance for generator/oracle, IO and variance. Still an estimate.
    estimated=2*sum(12*(750*r['step_seconds']+64*r['eval_batch_seconds']) for r in rows)
    report=dict(pass_=all(r['overfit_pass'] for r in rows) and estimated<900,scope='ENGINEERING_ONLY',research_roots_called=False,engineering_updates=220,records=rows,estimated_worker_seconds=estimated,source_sha256=source_fingerprints(),config_sha256=sha(Path('configs/exp0017_v2.json')))
    report['pass']=report.pop('pass_')
    report['numeric_code_sha256']=numeric_code_hash()
    write_json(path,report);print(json.dumps(report,indent=2))
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()
