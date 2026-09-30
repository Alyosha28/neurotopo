"""Expose numerator and all predeclared eligibility layers without changing endpoints."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=json.loads((a.run/'results.json').read_text());rows=[]
for group in ('micro','natural'):
 for context in r[group]:
  c=context['context'];seed,arm,cell,t=c['seed'],c['arm'],c['cell'],c['anchor']
  name=f'micro_{cell}_{t}.npz' if group=='micro' else f'natural_{t}.npz'
  with np.load(a.run/str(seed)/arm/name) as f:
   target=f['targets'];base=f['NONE__pre_logits'];baseok=((base>=0)==target).all(1);base_margin=((2*target-1)*base).min(1)
   for summary in context['branches']:
    name=summary['branch']
    if not name.startswith(('FULL','EQUAL')):continue
    event=0 if name.endswith('before') else int(name[-1]);post=f[name+'__event_post_z'];gauge='GAUGE_'+name.split('_')[1]
    initial=(base[:,0]>=0)==target[:,0];valid=f[name+'__event_valid'];postok=(post>=0)==target[:,event]
    eligible=initial&baseok&valid&postok
    if name.startswith('EQUAL'):eligible&=np.abs(post.astype(float)-f[name+'__event_pre_z'].astype(float))<=1e-5
    gaugeok=f[gauge+'__event_valid']&(((f[gauge+'__pre_logits']>=0)==target).all(1))
    strict=eligible&gaugeok&(base_margin>=1e-4)&((2*target[:,event]-1)*post>=1e-4)
    bad=((2*target[:,event+1:]-1)*f[name+'__pre_logits'][:,event+1:]).min(1)<=-1e-4
    assert int((strict&bad).sum())==summary['damage_tests']
    rows.append({'group':group,**c,'branch':name,'tests':len(target),'initial_correct':int(initial.sum()),'baseline_all_correct':int(baseok.sum()),'eligible_before_gauge_margin':int(eligible.sum()),'strict_eligible':int(strict.sum()),'damage':int((strict&bad).sum())})
with a.output.open('x') as f:json.dump({'scope':'Derived explicit denominator layers, same fixed raw tests and1e-4 criteria; pooled tests are not independent training seeds','rows':rows},f,indent=2);f.write('\n')
print('strict denominator audit PASS',len(rows))
