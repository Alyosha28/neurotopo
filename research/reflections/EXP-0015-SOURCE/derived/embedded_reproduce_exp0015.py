"""Independent explicit-matrix recomputation of existing EXP-0015 raw arrays.

No task generator imports, new samples, optimizers, or primary filter imports.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def file_hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def filter_matrix(y):
    transition=np.zeros((16,16),dtype=np.float64)
    for old in range(16):
        transition[old,old]=.7
        transition[(old+1)%16,old]=.2
        transition[(old-1)%16,old]=.1
    q=np.full((len(y),16),1/16,dtype=np.float64)
    for t in range(y.shape[1]):
        q=q@transition.T
        likelihood=np.full_like(q,.4/15)
        likelihood[np.arange(len(y)),y[:,t]]=.6
        q*=likelihood
        q/=q.sum(axis=1)[:,None]
    return q


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();start=time.monotonic()
    if args.output.exists():raise FileExistsError('No overwrite')
    manifest=json.loads((args.run/'manifest.json').read_text());analysis=json.loads((args.run/'analysis.json').read_text())
    assert manifest['status']=='COMPLETED' and manifest['research_updates']==0
    assert file_hash(args.run/'raw.npz')==manifest['raw_sha256']
    assert file_hash(args.run/'oracle.npz')==manifest['oracle_sha256']
    with np.load(args.run/'raw.npz',allow_pickle=False) as archive:
        raw={k:archive[k] for k in archive.files}
    for key,value in raw.items():
        assert list(value.shape)==manifest['arrays'][key]['shape']
        assert str(value.dtype)==manifest['arrays'][key]['dtype']
        assert hashlib.sha256(value.tobytes()).hexdigest()==manifest['arrays'][key]['sha256']
    assert raw['y'].shape==(16384,64)
    full=filter_matrix(raw['y']);one=filter_matrix(raw['y'][:,-1:]);eight=filter_matrix(raw['y'][:,-8:])
    def divergence(p,q):return np.sum(p*np.log(p/q),axis=1)
    a=divergence(full,np.full_like(full,1/16));b=divergence(full,one);c=divergence(full,eight)
    with np.load(args.run/'oracle.npz',allow_pickle=False) as saved:
        errors={name:float(np.max(np.abs(value-saved[name]))) for name,value in [('full',full),('suffix1',one),('suffix8',eight),('uniform_kl',a),('suffix1_kl',b),('suffix8_kl',c)]}
        # Different block size and separate column reductions from primary code.
        rng=np.random.Generator(np.random.PCG64(77100007));boot=np.empty((10000,2))
        for begin in range(0,10000,25):
            end=min(begin+25,10000);indices=rng.integers(0,16384,size=(end-begin,16384))
            boot[begin:end,0]=b[indices].mean(axis=1)
            boot[begin:end,1]=c[indices].mean(axis=1)
        errors['bootstrap_means']=float(np.max(np.abs(boot-saved['bootstrap_means'])))
    interval=np.quantile(boot,[.025,.975],axis=0,method='linear')
    values=[float(a.mean()),float(interval[0,0]),float(interval[0,1])]
    failed=[key for key,x,threshold in zip(['uniform_mean','suffix1_lower','suffix8_lower'],values,[.2,.05,.02]) if x<threshold]
    verdict='TASK_INFORMATION_FAILURE' if failed else 'TASK_INFORMATION_ONLY'
    for key,x in zip(['uniform_mean','suffix1_lower','suffix8_lower'],values):
        errors[key]=abs(x-analysis['metrics'][key])
    elapsed=time.monotonic()-start
    passed=max(errors.values())<=1e-10 and failed==analysis['failed'] and verdict==analysis['substatus'] and elapsed<=300
    report=dict(pass_=passed,scope='RAW_ARRAY_SECOND_IMPLEMENTATION_NOT_INDEPENDENT_SAMPLE_REPLICATION',
                errors=errors,maximum_error=max(errors.values()),recomputed_failed=failed,
                recomputed_substatus=verdict,seconds=elapsed,task_samples_generated=0,optimization_updates=0,
                script_sha256=file_hash(Path(__file__)),raw_sha256=file_hash(args.run/'raw.npz'),oracle_sha256=file_hash(args.run/'oracle.npz'),
                manifest_sha256=file_hash(args.run/'manifest.json'),analysis_sha256=file_hash(args.run/'analysis.json'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(report,indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
