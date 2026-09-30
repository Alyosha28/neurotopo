"""Post-hoc pairwise inverse bounds from immutable snapshots; no model execution."""
from __future__ import annotations
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback
import numpy as np

ETAS=np.array([0.,1e-16,1e-14,1e-12,1e-10,1e-8,1e-6])
TIMES=[0,1,2,4,8,16,32,64,128]


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(distance:np.ndarray,eta:float)->tuple[np.ndarray,np.ndarray]:
    if eta==0:return distance==0,np.zeros(distance.shape,dtype=bool)
    band=1e-12*np.maximum(distance,2*eta)
    yes=distance<2*eta-band
    uncertain=np.abs(distance-2*eta)<=band
    return yes,uncertain


def norms_scalar(a:np.ndarray,b:np.ndarray)->np.ndarray:
    values=[math.sqrt(math.fsum((float(x)-float(y))**2 for x,y in zip(u,v)))
            for u,v in zip(a.reshape(-1,a.shape[-1]),b.reshape(-1,b.shape[-1]))]
    return np.asarray(values).reshape(a.shape[:-1])


def exact_square(a:np.ndarray,b:np.ndarray)->Fraction:
    return sum(((Fraction.from_float(float(x))-Fraction.from_float(float(y)))**2 for x,y in zip(a,b)),Fraction(0))


def extended_median(logs:np.ndarray,infinite:np.ndarray,valid:np.ndarray)->dict:
    a=logs[valid].copy();a[infinite[valid]]=np.inf
    if not len(a):return {'kind':'UNDEFINED','log10':None}
    a.sort();center=(a[(len(a)-1)//2]+a[len(a)//2])/2
    if np.isinf(center):return {'kind':'INFINITE','log10':None}
    return {'kind':'FINITE','log10':float(center)}


def main()->None:
    out=Path('research/evidence/INV-0001');out.mkdir(exist_ok=False)
    start=time.monotonic()
    def abort(*_):raise TimeoutError('60-second single analysis and verification budget')
    signal.signal(signal.SIGALRM,abort);signal.setitimer(signal.ITIMER_REAL,60)
    try:
        # Reviewed analytic cases before inspecting derived empirical metrics.
        assert abs(2./2e-8-1e8)<1e-8
        assert overlap(np.array([.2]),.1)[1][0]
        assert overlap(np.array([.2]),.11)[0][0]
        assert overlap(np.array([0.]),0.)[0][0]
        assert extended_median(np.array([0.]),np.array([True]),np.array([True]))['kind']=='INFINITE'
        assert exact_square(np.array([1.]),np.array([-1.]))==4
        rows=[];inputs={};max_norm_error=max_saved_error=max_gain_error=0.;exact_checks=0;rawfiles=[]
        for seed in [17001,17002,17003]:
            for arm in ['D64','D16']:
                for precision in ['float32','float64']:
                    base=Path(f'artifacts/EXP-0019/v2/seed-{seed}/{arm}/{precision}')
                    file=base/'trajectories.npz';record=json.loads((base/'result.json').read_text())
                    assert sha(file)==record['trajectories_sha256'];inputs[str(file)]=sha(file);inputs[str(base/'result.json')]=sha(base/'result.json')
                    raw={}
                    with np.load(file) as data:
                        states=data['hidden'].astype(np.float64)
                        assert states.shape==(3,384,9,int(arm[1:])) and np.isfinite(states).all()
                        indices=data['sample_token_indices']-14
                        for pair,branch in [('opposite',1),('same',2)]:
                            a,b=states[0],states[branch]
                            distance=np.sqrt(np.sum((a-b)**2,axis=-1));scalar=norms_scalar(a,b)
                            error=float(np.max(np.abs(distance-scalar)));max_norm_error=max(max_norm_error,error);assert error<=1e-12
                            saved=data[pair+'_distance'][:,indices]
                            err=float(np.max(np.abs(distance-saved)));max_saved_error=max(max_saved_error,err);assert err<=1e-12
                            d0=distance[:,0];valid0=d0>0
                            inf=(distance==0)&valid0[:,None];finite=(distance>0)&valid0[:,None]
                            logs=np.zeros_like(distance)
                            for t in range(9):
                                good=finite[:,t]
                                logs[good,t]=np.log10(d0[good])-np.log10(distance[good,t])
                                independent=np.array([math.log10(float(scalar[i,0]))-math.log10(float(scalar[i,t])) for i in np.flatnonzero(good)])
                                if len(independent):
                                    err=float(np.max(np.abs(logs[good,t]-independent)));max_gain_error=max(max_gain_error,err);assert err<=1e-10
                            raw.update({pair+'_distance':distance,pair+'_log10_gain':logs,pair+'_gain_finite':finite,pair+'_gain_infinite':inf,pair+'_initial_valid':valid0})
                            for t,k in enumerate(TIMES):
                                d=distance[:,t];f=finite[:,t];infinite=inf[:,t]
                                ratios=logs[f,t]
                                row=dict(seed=seed,arm=arm,precision=precision,pair=pair,suffix_operations=k,n=384,
                                         initial_distinct_count=int(valid0.sum()),infinite_gain_fraction=float(infinite.mean()),undefined_count=int((~valid0).sum()),
                                         median_log10_gain=extended_median(logs[:,t],infinite,valid0),
                                         finite_log10_gain_quantiles=np.quantile(ratios,[.1,.5,.9]).tolist() if len(ratios) else None,
                                         terminal_distance_median=float(np.median(d)),critical_noise_radius_median=float(np.median(d/2)),
                                         overlap_fraction=[],boundary_fraction=[],robust_b_accuracy_upper=[],overlap_h0_error_lower_median=[])
                                exact_cache={}
                                for eta in ETAS:
                                    yes,boundary=overlap(d,float(eta));yes2,boundary2=overlap(scalar[:,t],float(eta))
                                    assert np.array_equal(yes,yes2) and np.array_equal(boundary,boundary2)
                                    selected={0,192,383}|set(np.flatnonzero(boundary).tolist())
                                    radius=Fraction.from_float(float(eta))*2;threshold=radius*radius
                                    for i in selected:
                                        if i not in exact_cache:exact_cache[i]=exact_square(a[i,t],b[i,t])
                                        sq=exact_cache[i];exact_checks+=1
                                        if yes[i]:assert sq<=threshold
                                        elif not boundary[i]:assert sq>threshold
                                    fraction=float(yes.mean());row['overlap_fraction'].append(fraction)
                                    row['boundary_fraction'].append(float(boundary.mean()))
                                    row['robust_b_accuracy_upper'].append(1.-fraction/2 if pair=='opposite' else None)
                                    row['overlap_h0_error_lower_median'].append(float(np.median(d0[yes]/2)) if yes.any() else None)
                                rows.append(row)
                    rawpath=out/f'seed-{seed}-{arm}-{precision}.npz'
                    with rawpath.open('xb') as f:np.savez_compressed(f,**raw)
                    rawfiles.append({'path':str(rawpath),'sha256':sha(rawpath)})
        assert len(rows)==216 and all(sha(Path(p))==h for p,h in inputs.items())
        summaries=[]
        for arm in ['D64','D16']:
            for precision in ['float32','float64']:
                for pair in ['opposite','same']:
                    for k in TIMES:
                        rs=[r for r in rows if (r['arm'],r['precision'],r['pair'],r['suffix_operations'])==(arm,precision,pair,k)]
                        medians=[r['median_log10_gain'] for r in rs]
                        finite_all=all(x['kind']=='FINITE' for x in medians)
                        summaries.append(dict(arm=arm,precision=precision,pair=pair,suffix_operations=k,
                                              per_seed_median_log10_gain=medians,
                                              median_log10_gain_range=[min(x['log10'] for x in medians),max(x['log10'] for x in medians)] if finite_all else None,
                                              infinite_gain_fraction_mean=float(np.mean([r['infinite_gain_fraction'] for r in rs])),
                                              critical_noise_radius_seed_medians=[r['critical_noise_radius_median'] for r in rs],
                                              overlap_fraction_mean=np.mean([r['overlap_fraction'] for r in rs],axis=0).tolist(),
                                              boundary_fraction_mean=np.mean([r['boundary_fraction'] for r in rs],axis=0).tolist(),
                                              robust_b_accuracy_upper_mean=np.mean([r['robust_b_accuracy_upper'] for r in rs],axis=0).tolist() if pair=='opposite' else None))
        result=dict(status='ANALYZED',verification_status='NUMERICALLY_VERIFIED',scope='Post-hoc finite-pair inverse lower bounds and hypothetical bounded-observation scenarios; no actual noise measurement or inverse solver',
                    eta_l2=ETAS.tolist(),times=TIMES,rows=rows,summaries=summaries,
                    verification=dict(hand_cases_pass=True,max_scalar_norm_error=max_norm_error,max_saved_distance_error=max_saved_error,max_log10_gain_error=max_gain_error,exact_fraction_comparisons=exact_checks,input_hashes_unchanged=True),
                    input_sha256=inputs,rawfiles=rawfiles,script_sha256=sha(Path(__file__)),design_sha256=sha(Path('research/design/inv0001_pairwise_inverse_protocol.md')),
                    seconds=time.monotonic()-start,new_samples=0,model_executions=0,training_updates=0,numpy=np.__version__,python=sys.version)
        encoded=json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
        with (out/'analysis.json').open('x') as f:f.write(encoded)
        print(json.dumps({'status':result['verification_status'],'seconds':result['seconds'],'verification':result['verification'],'analysis_bytes':len(encoded.encode())}))
    except BaseException as error:
        record={'error':repr(error),'traceback':traceback.format_exc(),'seconds':time.monotonic()-start}
        encoded=json.dumps(record,indent=2)
        with (out/f'failure-{time.time_ns()}.json').open('x') as f:f.write(encoded)
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
