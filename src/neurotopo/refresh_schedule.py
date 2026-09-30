"""EXP0026v2 immutable frozen-model schedule runner."""
from __future__ import annotations
import argparse,json,platform,signal,sys,time
from pathlib import Path
import numpy as np
import torch
from .learned_protected_pilot import LearnedMemory,generate,state_hash
from .schedule_core import MAIN,EXTRA,rollout,threshold
from .schedule_analysis import summarize,save,read
from .order_observation import sha
from .run import git_output,source_fingerprints

SEEDS=tuple(range(25001,25013))+(23005,)
ARMS=('REL','INV');CELLS=('LONG_H','LONG_U');CAP=3600


def npz(path:Path,raw:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**raw)


def validate_config(c:dict)->None:
    expected={'experiment_id':'EXP-0026','version':'v2','workflow':'refresh_schedule','protocol':'experiments/EXP-0026/v2.md','output':'artifacts/EXP-0026/v2','parent':'artifacts/EXP-0025/v2','data_parent':'artifacts/DYN-0001/v3','seeds':list(SEEDS),'arms':list(ARMS),'examples':512,'length':1024,'matching_criterion_version':'G1-MATCH-v3','gate1_claim':False,'resource':{'worker_wall_seconds_cap':3600,'reproduction_wall_seconds_cap':600,'research_update_cap':0,'general_validation_overfit_cap':900}}
    if c!=expected:raise ValueError('Registered configuration mismatch')
    if torch.__version__.split('+')[0]!='2.8.0' or np.__version__!='2.3.3':raise RuntimeError('Frozen package versions required')


def run(config:Path,output:Path|None=None)->Path:
    started=time.monotonic();cpu=time.process_time();c=read(config);validate_config(c)
    if git_output('status','--porcelain'):raise RuntimeError('Clean source checkout required')
    source=source_fingerprints();suite=read('artifacts/validation/suite.json');pre=read('artifacts/validation/exp0026_preflight.json')
    if not suite['pass'] or not pre['pass'] or source!=suite['source_sha256'] or source!=pre['source_sha256'] or pre['config_sha256']!=sha(config):raise RuntimeError('Source-matched validation required')
    out=output or Path(c['output'])
    if output is not None and not output.is_absolute():raise ValueError('Absolute override required')
    out.mkdir(parents=True,exist_ok=False)
    parent=Path(c['parent']);dp=Path(c['data_parent']);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    frozen={str(p):sha(p) for p in (config,Path(c['protocol']),Path('research/entries/EXP-0026.json'),Path('uv.lock'),Path('artifacts/validation/suite.json'),Path('artifacts/validation/exp0026_preflight.json'),parent/'manifest.json',parent/'completion.json',dp/'manifest.json',dp/'completion.json')}
    thresholds={};parents={}
    # Freeze all theta values from stored calibration before tau generation/evaluation.
    for seed in SEEDS:
        for arm in ARMS:
            folder=parent/str(seed)/arm;row=read(folder/'result.json');cp=folder/'checkpoint-1000.pt';pp=folder/'prototypes.npz'
            if sha(cp)!=row['artifact_sha256']['checkpoint-1000.pt'] or sha(pp)!=row['artifact_sha256']['prototypes.npz']:raise RuntimeError('Parent checkpoint/prototype changed')
            for p in (cp,pp,folder/'result.json'):frozen[str(p)]=sha(p)
            for cell in CELLS:
                reference=folder/'test'/'EQUAL8'/f'{cell}.npz'
                if sha(reference)!=row['evaluation']['EQUAL8'][cell]['artifact_sha256']:raise RuntimeError('Parent EQUAL reference changed')
                frozen[str(reference)]=sha(reference)
                reference=dp/str(seed)/arm/f'{cell}.npz';frozen[str(reference)]=sha(reference)
            with np.load(pp) as f:cal={k:f[k] for k in f.files}
            if cal['calibration_gram'].shape!=(512,9,3) or cal['calibration_logits'].shape!=(512,9):raise RuntimeError('Calibration shape')
            thresholds[f'{seed}/{arm}']=threshold(cal);parents[seed,arm]=row
        for cell in CELLS:
            path=dp/'data'/str(seed)/f'{cell}.npz';frozen[str(path)]=sha(path)
    save(out/'manifest.json',{'experiment_id':'EXP-0026','version':'v2','config':c,'git_commit':git_output('rev-parse','HEAD'),
        'source_sha256':source,'frozen_hashes':frozen,'thresholds':thresholds,'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'platform':platform.platform(),'threads':1,'training_updates':0,'prototype_fits':0,'gate1_claim':False})
    rows=[];main_count=tau_count=base_tokens=command_tokens=0;attempts=valid=0;tau_inputs={};model_resources=[]
    def budget():
        if time.monotonic()-started>CAP:raise TimeoutError('EXP0026 run budget exhausted')
    try:
        for seed in SEEDS:
            for mode_id,cell in enumerate(CELLS,1):
                budget();data=generate(512,1024,[267001,seed,mode_id,1024],'HOLD' if cell=='LONG_H' else 'UPDATE',np.pi,0.)
                p=out/'tau_data'/str(seed)/f'{cell}.npz';npz(p,data);tau_inputs[str(p.relative_to(out))]=sha(p)
        save(out/'inputs_frozen.json',{'theta':thresholds,'tau_inputs':tau_inputs,'source_frozen_hashes':frozen})
        for seed in SEEDS:
            for arm in ARMS:
                folder=parent/str(seed)/arm;state=torch.load(folder/'checkpoint-1000.pt',map_location='cpu',weights_only=True)
                model=LearnedMemory(arm,seed);model.load_state_dict(state['model']);model.eval();before=state_hash(model)
                nparams=sum(p.numel() for p in model.parameters())
                if nparams!=({'REL':340,'INV':406}[arm]):raise RuntimeError('Parameter count changed')
                model_resources.append({'seed':seed,'arm':arm,'parameters':nparams,'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'state_values':6,'state_bytes':24,'prototype_values':12,'prototype_bytes':48,'optimizer_updates':0,'executed_flops':'NOT_MEASURED','peak_memory':'NOT_MEASURED'})
                if before!=parents[seed,arm]['frozen_model_state_sha256']:raise RuntimeError('Frozen model mismatch')
                with np.load(folder/'prototypes.npz') as f:proto=f['prototypes'].copy()
                theta=thresholds[f'{seed}/{arm}']
                for phase in ('main','tau'):
                    for cell in CELLS:
                        path=dp/'data'/str(seed)/f'{cell}.npz' if phase=='main' else out/'tau_data'/str(seed)/f'{cell}.npz'
                        with np.load(path) as f:data={k:f[k] for k in f.files}
                        if data['commands'].shape!=(512,1024) or np.max(np.abs(data['leak']-np.eye(3)))!=0:raise RuntimeError('Input contract failed')
                        oracle=np.bitwise_xor.accumulate(np.column_stack((data['b0'],data['commands'])),axis=1)
                        if not np.array_equal(oracle,data['targets']):raise RuntimeError('Wrong target')
                        input_digest=sha(path)
                        for rotation in ('ORIGINAL','IDENTITY'):
                            conditions=('NONE',) if phase=='tau' else MAIN+EXTRA if rotation=='ORIGINAL' else MAIN
                            adapt_counts=None
                            for condition in conditions:
                                budget();t0=time.monotonic();c0=time.process_time()
                                diagnostics=phase=='main' and condition=='NONE'
                                raw=rollout(model,data,proto,condition,rotation,theta,adapt_counts if condition=='COUNT90' else None,diagnostics)
                                if condition=='ADAPT90':adapt_counts=raw['attempt_counts'].copy()
                                if phase=='main' and rotation=='ORIGINAL' and condition in ('NONE','ONCE8'):
                                    with np.load(dp/str(seed)/arm/f'{cell}.npz') as old:
                                        reference=old['logits'][0 if condition=='NONE' else 3]
                                        if not np.array_equal(raw['logits'][:,8:],reference):raise RuntimeError('DYN0001 replay mismatch '+condition)
                                if phase=='main' and rotation=='ORIGINAL' and condition=='K8':
                                    with np.load(folder/'test'/'EQUAL8'/f'{cell}.npz') as old:
                                        if not np.array_equal(raw['logits'][:,:257],old['logits']):raise RuntimeError('EXP0025 K8 replay mismatch')
                                summary=summarize(raw,data,condition,theta)
                                rel=Path(str(seed))/arm/phase/cell/rotation/(condition+'.npz');npz(out/rel,raw)
                                row={'seed':seed,'arm':arm,'phase':phase,'cell':cell,'rotation':rotation,'condition':condition,'theta':theta,'summary':summary,
                                     'input_path':str(path.resolve()),'input_sha256':input_digest,'raw_path':str(rel),'raw_sha256':sha(out/rel),
                                     'wall_seconds':time.monotonic()-t0,'cpu_seconds':time.process_time()-c0}
                                save((out/rel).with_suffix('.json'),row);rows.append(row)
                                base_tokens+=512*1024;command_tokens+=512*8*2 if diagnostics else 0
                                attempts+=summary['metrics']['attempts'];valid+=summary['metrics']['valid']
                                if phase=='main':main_count+=1
                                else:tau_count+=1
                            print(json.dumps({'seed':seed,'arm':arm,'phase':phase,'cell':cell,'rotation':rotation,'status':'COMPLETED'}),flush=True)
                if state_hash(model)!=before:raise RuntimeError('Frozen weights changed')
        budget()
        if (main_count,tau_count,base_tokens,command_tokens)!=(1196,104,681574400,851968):raise RuntimeError('Registered matrix counts differ')
        save(out/'completion.json',{'status':'COMPLETED','rows':rows,'tau_inputs':tau_inputs,'main_rollouts':main_count,'tau_rollouts':tau_count,
                                  'base_transitions':base_tokens,'command_transitions':command_tokens,'diagnostic_refresh_proposals':command_tokens//2,'refresh_attempts':attempts,'refresh_valid':valid,
                                  'training_updates':0,'prototype_fits':0,'model_resources':model_resources,'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu})
    except BaseException as error:
        save(out/'failure.json',{'error':repr(error),'completed_rollouts':len(rows),'main_rollouts':main_count,'tau_rollouts':tau_count,
                                'base_transitions':base_tokens,'command_transitions':command_tokens,'refresh_attempts':attempts,'wall_seconds':time.monotonic()-started})
        raise
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    def abort(*_):raise TimeoutError('EXP0026 worker deadline')
    signal.signal(signal.SIGTERM,abort);run(a.config,a.output)
