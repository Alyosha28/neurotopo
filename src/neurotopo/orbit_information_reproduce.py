"""Independent matrix-filter audit of existing EXP-0015 arrays and provenance."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
import numpy as np


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:raise ValueError(message)


def verify_provenance(manifest: dict[str, Any], source: Path) -> dict[str, str]:
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    require(commit==manifest['git_commit'],'Source commit differs')
    require(not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip(),'Source checkout dirty')
    actual={str(p.relative_to(source)):file_hash(p) for p in sorted((source/'src').rglob('*.py')) if not p.name.startswith('._')}
    require(actual==manifest['source_sha256'],'Source fingerprints differ')
    for relative,expected in {**actual,**manifest['frozen_hashes']}.items():
        relative_path=Path(relative)
        require(not relative_path.is_absolute() and '..' not in relative_path.parts,'Unsafe frozen path')
        path=source/relative_path
        require(file_hash(path)==expected,f'Frozen file hash differs: {relative}')
        blob=subprocess.check_output(['git','-C',str(source),'show',f'{commit}:{relative}'])
        require(hashlib.sha256(blob).hexdigest()==expected,f'File differs from committed bytes: {relative}')
    config_path=source/'configs/exp0015_v1.json'
    require(file_hash(config_path)==manifest['config_sha256'],'Config digest differs')
    require(json.loads(config_path.read_text())==manifest['config'],'Config content differs')
    validation_path=source/'artifacts/validation/suite.json'
    require(file_hash(validation_path)==manifest['validation_sha256'],'Validation receipt differs')
    validation=json.loads(validation_path.read_text())
    require(validation.get('pass') is True and validation.get('source_sha256')==actual,'Validation not passing for these sources')
    require(sys.version==manifest['python'] and np.__version__==manifest['numpy']=='2.3.3','Runtime versions differ')
    require(file_hash(Path(sys.executable).resolve())==manifest['executable_sha256'],'Interpreter digest differs')
    own_relative='src/neurotopo/orbit_information_reproduce.py'
    require(actual.get(own_relative)==file_hash(Path(__file__)),'Executing auditor differs from frozen code')
    return dict(source_commit=commit,validation_sha256=file_hash(validation_path),source_directory=str(source))


def filter_matrix(y: np.ndarray) -> np.ndarray:
    transition=np.zeros((16,16),dtype=np.float64)
    for old in range(16):
        transition[old,old]=.7;transition[(old+1)%16,old]=.2;transition[(old-1)%16,old]=.1
    q=np.full((len(y),16),1/16,dtype=np.float64)
    for t in range(y.shape[1]):
        q=q@transition.T
        likelihood=np.full_like(q,.4/15);likelihood[np.arange(len(y)),y[:,t]]=.6
        q*=likelihood;q/=q.sum(axis=1)[:,None]
    return q


def verify_result(metrics: dict[str,float], analysis: dict[str,Any]) -> dict[str,float]:
    require(set(metrics)==set(analysis['metrics']),'Analysis metric fields differ')
    errors={key:abs(x-analysis['metrics'][key]) for key,x in metrics.items()}
    require(all(np.isfinite(x) and x<=1e-10 for x in errors.values()),'Analysis numerical fields differ')
    failed=[key for key,threshold in [('uniform_mean',.2),('suffix1_lower',.05),('suffix8_lower',.02)] if metrics[key]<threshold]
    verdict='TASK_INFORMATION_FAILURE' if failed else 'TASK_INFORMATION_ONLY'
    decision='INCONCLUSIVE' if failed else 'SUPPORTED'
    require(failed==analysis['failed'] and verdict==analysis['substatus'] and decision==analysis['decision'],'Analysis verdict differs')
    require(analysis['gate1_claim'] is False and analysis['research_updates']==0,'Invalid claim/update status')
    return errors


def evaluate(run: Path, source_override: Path | None = None) -> dict[str,Any]:
    manifest=json.loads((run/'manifest.json').read_text());analysis=json.loads((run/'analysis.json').read_text())
    require(manifest['status']=='COMPLETED' and manifest['research_updates']==0,'Incomplete screen')
    provenance=verify_provenance(manifest,source_override or Path(manifest['source_repo']))
    require(file_hash(run/'raw.npz')==manifest['raw_sha256'],'Raw archive digest differs')
    require(file_hash(run/'oracle.npz')==manifest['oracle_sha256'],'Oracle archive digest differs')
    with np.load(run/'raw.npz',allow_pickle=False) as archive:raw={k:archive[k] for k in archive.files}
    for key,value in raw.items():
        require(list(value.shape)==manifest['arrays'][key]['shape'] and str(value.dtype)==manifest['arrays'][key]['dtype'],'Raw array descriptor differs')
        require(hashlib.sha256(value.tobytes()).hexdigest()==manifest['arrays'][key]['sha256'],'Raw array digest differs')
    require(raw['y'].shape==(16384,64),'Wrong scientific sample dimensions')
    full=filter_matrix(raw['y']);one=filter_matrix(raw['y'][:,-1:]);eight=filter_matrix(raw['y'][:,-8:])
    def divergence(p,q):return np.sum(p*np.log(p/q),axis=1)
    a=divergence(full,np.full_like(full,1/16));b=divergence(full,one);c=divergence(full,eight)
    with np.load(run/'oracle.npz',allow_pickle=False) as saved:
        errors={name:float(np.max(np.abs(value-saved[name]))) for name,value in [('full',full),('suffix1',one),('suffix8',eight),('uniform_kl',a),('suffix1_kl',b),('suffix8_kl',c)]}
        rng=np.random.Generator(np.random.PCG64(77100007));boot=np.empty((10000,2))
        for begin in range(0,10000,25):
            end=min(begin+25,10000);indices=rng.integers(0,16384,size=(end-begin,16384))
            boot[begin:end,0]=b[indices].mean(axis=1);boot[begin:end,1]=c[indices].mean(axis=1)
        errors['bootstrap_means']=float(np.max(np.abs(boot-saved['bootstrap_means'])))
    require(all(np.isfinite(x) and x<=1e-10 for x in errors.values()),'Independent oracle/bootstrap differ')
    interval=np.quantile(boot,[.025,.975],axis=0,method='linear')
    metrics=dict(uniform_mean=float(a.mean()),suffix1_mean=float(b.mean()),suffix8_mean=float(c.mean()),
                 suffix1_lower=float(interval[0,0]),suffix1_upper=float(interval[1,0]),
                 suffix8_lower=float(interval[0,1]),suffix8_upper=float(interval[1,1]))
    errors.update(verify_result(metrics,analysis))
    return dict(pass_=True,scope='RAW_ARRAY_SECOND_IMPLEMENTATION_NOT_INDEPENDENT_SAMPLE_REPLICATION',
                errors=errors,maximum_error=max(errors.values()),recomputed_metrics=metrics,
                recomputed_substatus=analysis['substatus'],recomputed_decision=analysis['decision'],
                provenance=provenance,raw_sha256=file_hash(run/'raw.npz'),oracle_sha256=file_hash(run/'oracle.npz'),
                manifest_sha256=file_hash(run/'manifest.json'),analysis_sha256=file_hash(run/'analysis.json'))


def execute(run: Path, output: Path, source: Path | None = None) -> dict[str,Any]:
    if output.exists():raise FileExistsError('No overwrite')
    started=time.monotonic()
    def deadline(signum,frame):raise TimeoutError('Independent reproduction exceeded 300 seconds')
    previous=signal.signal(signal.SIGALRM,deadline)
    signal.setitimer(signal.ITIMER_REAL,300)
    try:
        report=evaluate(run,source)
        if time.monotonic()-started>300:raise TimeoutError('Reproduction deadline exceeded')
    except Exception as error:
        report=dict(pass_=False,substatus='REPRODUCTION_UNRESOLVED',decision='INCONCLUSIVE',error=repr(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    report.update(seconds=time.monotonic()-started,task_samples_generated=0,optimization_updates=0,
                  auditor_sha256=file_hash(Path(__file__)))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    return report


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--source',type=Path)
    args=parser.parse_args();report=execute(args.run,args.output,args.source);print(json.dumps(report,indent=2),flush=True)
    if not report['pass_']:raise SystemExit(1)


if __name__=='__main__':main()
